use crate::model::gemma_mapper::matmul;
use crate::model::ternary::{quantize_ternary, pack_ternary, ternary_matmul};

/// Routing data collected during MoE forward pass.
/// Used for load balance loss computation and router gradient.
#[derive(Clone, Debug)]
pub struct MoERoutingData {
    pub layer_idx: usize,
    pub gate_logits: Vec<f32>,
    pub gate_probs: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub expert_weights: Vec<f32>,
    pub pre_ffn_input: Vec<f32>,
    pub expert_outputs: Vec<f32>,
}

/// Ternary-quantized expert weight matrix.
/// Stores {-1, 0, +1} values with absmean scale — 8× less memory than FP32.
#[derive(Clone, Debug)]
pub struct TernaryExpert {
    /// Unpacked ternary values [rows × cols].
    pub values: Vec<i8>,
    /// Packed 2-bit weights (4 values/byte).
    pub packed: Vec<u8>,
    /// Absmean scale factor.
    pub scale: f32,
    pub rows: usize,
    pub cols: usize,
}

impl TernaryExpert {
    /// Quantize FP32 weights to ternary.
    pub fn from_fp32(weights: &[f32], rows: usize, cols: usize) -> Self {
        let (values, scale) = quantize_ternary(weights);
        let packed = pack_ternary(&values);
        Self { values, packed, scale, rows, cols }
    }

    /// Ternary matmul: output = scale * ternary @ input
    /// input: [seq × cols], output: [seq × rows]
    pub fn forward(&self, input: &[f32], seq: usize) -> Vec<f32> {
        ternary_matmul(&self.values, self.scale, input, self.rows, self.cols, seq)
    }

    /// Parallel forward using rayon — ~4-8x faster on multi-core for large matrices.
    pub fn forward_parallel(&self, input: &[f32], seq: usize) -> Vec<f32> {
        crate::model::ternary::ternary_matmul_parallel(&self.values, self.scale, input, self.rows, self.cols, seq)
    }

    /// Single-token forward.
    pub fn forward_single(&self, input: &[f32]) -> Vec<f32> {
        ternary_matmul(&self.values, self.scale, input, self.rows, self.cols, 1)
    }

    /// Dequantize to FP32 (for checkpoint saving).
    pub fn to_fp32(&self) -> Vec<f32> {
        self.values.iter().map(|&v| v as f32 * self.scale).collect()
    }

    pub fn memory_bytes(&self) -> usize {
        self.values.len() + self.packed.len() + 4
    }
}

/// CPU-only Mixture of Experts layer with ternary expert weights.
///
/// Expert weights stored as {-1, 0, +1} (1 byte/value) with absmean scale.
/// Router stays FP32 (tiny, trainable). LoRA adapters (FP32) modify the
/// dequantized base output during training.
///
/// Memory: 4 experts × 3 projections × [1536×12288] = 216 MB FP32 → ~27 MB ternary.
pub struct CpuMoELayer {
    /// Router weights: [hidden_dim, num_experts]. FP32 — trainable.
    pub gate_weights: Vec<f32>,
    /// Expert gate projections (ternary): [num_experts].
    pub expert_gate: Vec<TernaryExpert>,
    /// Expert up projections (ternary): [num_experts].
    pub expert_up: Vec<TernaryExpert>,
    /// Expert down projections (ternary): [num_experts].
    pub expert_down: Vec<TernaryExpert>,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub use_gelu: bool,
}

impl CpuMoELayer {
    pub fn new(hidden_dim: usize, intermediate_dim: usize, num_experts: usize, top_k: usize) -> Self {
        // Initialize with zero ternary weights
        let make_zero_expert = |rows: usize, cols: usize| {
            TernaryExpert {
                values: vec![0i8; rows * cols],
                packed: pack_ternary(&vec![0i8; rows * cols]),
                scale: 1.0,
                rows,
                cols,
            }
        };
        Self {
            gate_weights: vec![0.0; num_experts * hidden_dim],
            expert_gate: (0..num_experts).map(|_| make_zero_expert(intermediate_dim, hidden_dim)).collect(),
            expert_up: (0..num_experts).map(|_| make_zero_expert(intermediate_dim, hidden_dim)).collect(),
            expert_down: (0..num_experts).map(|_| make_zero_expert(hidden_dim, intermediate_dim)).collect(),
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k,
            use_gelu: false,
        }
    }

    /// Forward pass: route tokens to top-k experts, compute weighted output.
    pub fn forward(&self, input: &[f32], seq: usize) -> Vec<f32> {
        self.forward_with_routing(input, seq, None, 0, None).0
    }

    /// Parallel forward using rayon — use for decode (seq=1) with large expert matrices.
    pub fn forward_parallel(&self, input: &[f32], seq: usize) -> Vec<f32> {
        self.forward_parallel_impl(input, seq, None, 0, None)
    }

    /// Forward pass that also collects routing data for loss/gradient computation.
    pub fn forward_with_routing(
        &self,
        input: &[f32],
        seq: usize,
        routing_collector: Option<&mut Vec<MoERoutingData>>,
        layer_idx: usize,
        lora_manager: Option<&crate::training::lora::LoraManager>,
    ) -> (Vec<f32>, Option<&mut Vec<MoERoutingData>>) {
        let h = self.hidden_dim;
        let ne = self.num_experts;
        let tk = self.top_k;

        let gate_logits = matmul(input, &self.gate_weights, seq, h, ne);
        let (selected_experts, expert_weights, gate_probs) = self.top_k_select_with_probs(&gate_logits, seq);

        let mut output = vec![0.0f32; seq * h];
        let mut expert_outputs = if routing_collector.is_some() {
            vec![0.0f32; seq * tk * h]
        } else {
            Vec::new()
        };

        for t in 0..seq {
            for k_idx in 0..tk {
                let expert_idx = selected_experts[t * tk + k_idx];
                let weight = expert_weights[t * tk + k_idx];
                if weight.abs() < 1e-8 { continue; }

                let token = &input[t * h..(t + 1) * h];
                let expert_out = self.expert_forward_with_lora(expert_idx, token, lora_manager, layer_idx);

                if !expert_outputs.is_empty() {
                    let dst = t * tk * h + k_idx * h;
                    expert_outputs[dst..dst + h].copy_from_slice(&expert_out);
                }

                for (i, &v) in expert_out.iter().enumerate() {
                    output[t * h + i] += weight * v;
                }
            }
        }

        if let Some(collector) = routing_collector {
            collector.push(MoERoutingData {
                layer_idx,
                gate_logits,
                gate_probs,
                selected_experts: selected_experts.clone(),
                expert_weights: expert_weights.clone(),
                pre_ffn_input: input.to_vec(),
                expert_outputs,
            });
        }

        (output, None)
    }

    /// Parallel MoE forward using rayon for expert matmuls.
    fn forward_parallel_impl(
        &self,
        input: &[f32],
        seq: usize,
        _routing_collector: Option<&mut Vec<MoERoutingData>>,
        layer_idx: usize,
        lora_manager: Option<&crate::training::lora::LoraManager>,
    ) -> Vec<f32> {
        let h = self.hidden_dim;
        let ne = self.num_experts;
        let tk = self.top_k;

        let gate_logits = matmul(input, &self.gate_weights, seq, h, ne);
        let (selected_experts, expert_weights, _gate_probs) = self.top_k_select_with_probs(&gate_logits, seq);

        let mut output = vec![0.0f32; seq * h];

        for t in 0..seq {
            for k_idx in 0..tk {
                let expert_idx = selected_experts[t * tk + k_idx];
                let weight = expert_weights[t * tk + k_idx];
                if weight.abs() < 1e-8 { continue; }

                let token = &input[t * h..(t + 1) * h];
                let expert_out = self.expert_forward_parallel_with_lora(expert_idx, token, lora_manager, layer_idx);

                for (i, &v) in expert_out.iter().enumerate() {
                    output[t * h + i] += weight * v;
                }
            }
        }

        output
    }

    /// Expert forward with parallel matmul for each projection.
    fn expert_forward_parallel_with_lora(
        &self,
        expert_idx: usize,
        token: &[f32],
        lora_manager: Option<&crate::training::lora::LoraManager>,
        layer_idx: usize,
    ) -> Vec<f32> {
        let id = self.intermediate_dim;

        // gate: parallel ternary matmul + LoRA
        let mut gated = self.expert_gate[expert_idx].forward_parallel(token, 1);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.gate", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, token, 1) {
                for (i, v) in lo.iter().enumerate() { gated[i] += v; }
            }
        }
        let gated: Vec<f32> = if self.use_gelu {
            gated.iter().map(|&x| crate::model::gemma_mapper::gelu_tanh(x)).collect()
        } else {
            gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
        };

        // up: parallel ternary matmul + LoRA
        let mut upped = self.expert_up[expert_idx].forward_parallel(token, 1);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.up", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, token, 1) {
                for (i, v) in lo.iter().enumerate() { upped[i] += v; }
            }
        }

        let mut combined = vec![0.0; id];
        for i in 0..id { combined[i] = gated[i] * upped[i]; }

        // down: parallel ternary matmul + LoRA
        let mut down_out = self.expert_down[expert_idx].forward_parallel(&combined, 1);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.down", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, &combined, 1) {
                for (i, v) in lo.iter().enumerate() { down_out[i] += v; }
            }
        }

        down_out
    }

    pub fn top_k_select_with_probs(&self, gate_logits: &[f32], seq: usize) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
        let ne = self.num_experts;
        let tk = self.top_k;
        let mut selected = vec![0usize; seq * tk];
        let mut weights = vec![0.0f32; seq * tk];
        let mut all_probs = vec![0.0f32; seq * ne];

        for t in 0..seq {
            let logits = &gate_logits[t * ne..(t + 1) * ne];
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for (e, &l) in logits.iter().enumerate() {
                all_probs[t * ne + e] = (l - max_logit).exp();
                sum_exp += all_probs[t * ne + e];
            }
            if sum_exp > 0.0 {
                for e in 0..ne { all_probs[t * ne + e] /= sum_exp; }
            }

            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let max_l = indexed[0].1;
            let mut top_sum = 0.0f32;
            for k in 0..tk.min(ne) {
                let exp_val = (indexed[k].1 - max_l).exp();
                top_sum += exp_val;
                selected[t * tk + k] = indexed[k].0;
                weights[t * tk + k] = exp_val;
            }
            if top_sum > 0.0 {
                for k in 0..tk.min(ne) { weights[t * tk + k] /= top_sum; }
            }
        }

        (selected, weights, all_probs)
    }

    /// Single expert forward with optional LoRA adapters on gate/up/down.
    fn expert_forward_with_lora(
        &self,
        expert_idx: usize,
        token: &[f32],
        lora_manager: Option<&crate::training::lora::LoraManager>,
        layer_idx: usize,
    ) -> Vec<f32> {
        let id = self.intermediate_dim;

        // gate: ternary matmul + LoRA
        let mut gated = self.expert_gate[expert_idx].forward_single(token);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.gate", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, token, 1) {
                for (i, v) in lo.iter().enumerate() { gated[i] += v; }
            }
        }
        let gated: Vec<f32> = if self.use_gelu {
            gated.iter().map(|&x| crate::model::gemma_mapper::gelu_tanh(x)).collect()
        } else {
            gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
        };

        // up: ternary matmul + LoRA
        let mut upped = self.expert_up[expert_idx].forward_single(token);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.up", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, token, 1) {
                for (i, v) in lo.iter().enumerate() { upped[i] += v; }
            }
        }

        // combined = gated * upped
        let mut combined = vec![0.0; id];
        for i in 0..id { combined[i] = gated[i] * upped[i]; }

        // down: ternary matmul + LoRA
        let mut down_out = self.expert_down[expert_idx].forward_single(&combined);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.down", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, &combined, 1) {
                for (i, v) in lo.iter().enumerate() { down_out[i] += v; }
            }
        }

        down_out
    }

    /// Expert forward that returns intermediate activations for backward pass.
    pub fn expert_forward_store_act(
        &self,
        expert_idx: usize,
        token: &[f32],
        lora_manager: Option<&crate::training::lora::LoraManager>,
        layer_idx: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let id = self.intermediate_dim;

        let mut gated = self.expert_gate[expert_idx].forward_single(token);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.gate", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, token, 1) {
                for (i, v) in lo.iter().enumerate() { gated[i] += v; }
            }
        }
        let gated: Vec<f32> = if self.use_gelu {
            gated.iter().map(|&x| crate::model::gemma_mapper::gelu_tanh(x)).collect()
        } else {
            gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
        };

        let mut upped = self.expert_up[expert_idx].forward_single(token);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.up", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, token, 1) {
                for (i, v) in lo.iter().enumerate() { upped[i] += v; }
            }
        }

        let mut combined = vec![0.0; id];
        for i in 0..id { combined[i] = gated[i] * upped[i]; }

        let mut down_out = self.expert_down[expert_idx].forward_single(&combined);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.down", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, &combined, 1) {
                for (i, v) in lo.iter().enumerate() { down_out[i] += v; }
            }
        }

        (down_out, gated, upped, combined)
    }

    /// Total memory for expert weights (ternary).
    pub fn expert_memory_bytes(&self) -> usize {
        self.expert_gate.iter().map(|e| e.memory_bytes()).sum::<usize>()
        + self.expert_up.iter().map(|e| e.memory_bytes()).sum::<usize>()
        + self.expert_down.iter().map(|e| e.memory_bytes()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_moe_basic() {
        let hd = 4;
        let id = 8;
        let ne = 2;
        let tk = 1;

        let mut moe = CpuMoELayer::new(hd, id, ne, tk);
        moe.gate_weights[0] = 10.0;
        // Expert 0: make gate/up semi-identity via FP32 → ternary
        // Use large values so ternary quantization maps them to ±1 (not 0)
        let mut gate_fp32 = vec![0.3f32; id * hd];
        let mut up_fp32 = vec![0.3f32; id * hd];
        let mut down_fp32 = vec![0.3f32; hd * id];
        for i in 0..hd.min(id) {
            gate_fp32[i * id + i] = 5.0;
            up_fp32[i * id + i] = 5.0;
            down_fp32[i * hd + i] = 5.0;
        }
        moe.expert_gate[0] = TernaryExpert::from_fp32(&gate_fp32, id, hd);
        moe.expert_up[0] = TernaryExpert::from_fp32(&up_fp32, id, hd);
        moe.expert_down[0] = TernaryExpert::from_fp32(&down_fp32, hd, id);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = moe.forward(&input, 1);
        assert_eq!(output.len(), 4);
        assert!(output.iter().any(|&x| x.abs() > 0.01),
            "output should be non-zero, got: {:?}", output);
    }

    #[test]
    fn test_ternary_expert_memory() {
        let w = vec![0.5f32; 1536 * 6144]; // ~9.4M values
        let te = TernaryExpert::from_fp32(&w, 6144, 1536);
        let fp32_bytes = w.len() * 4;
        let ternary_bytes = te.memory_bytes();
        assert!(ternary_bytes < fp32_bytes / 3, "ternary {} should be << fp32 {}", ternary_bytes, fp32_bytes);
    }
}
