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

    /// Packed parallel forward — uses 2-bit packed weights for 4× less memory bandwidth.
    /// This is the fastest path for decode (seq=1).
    pub fn forward_packed_parallel(&self, input: &[f32], seq: usize) -> Vec<f32> {
        crate::model::ternary::ternary_matmul_packed_parallel(
            &self.packed, self.scale, input, self.rows, self.cols, seq,
        )
    }

    /// Batch projection: parallel over rows for `seq > 1`, serial for one row.
    ///
    /// The per-row accumulation is unchanged, so the result is bit-identical to
    /// [`Self::forward`].
    pub fn forward_batch(&self, input: &[f32], seq: usize) -> Vec<f32> {
        if seq > 1 {
            self.forward_parallel(input, seq)
        } else {
            self.forward(input, seq)
        }
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

    /// GPU-ready packed weights: u32 format and per-row scales.
    pub fn gpu_packed(&self) -> (Vec<u32>, Vec<f32>) {
        let total = self.rows * self.cols;
        let packed_u32 = crate::model::ternary::pack_ternary_u32(&self.packed, total);
        let scales = vec![self.scale; self.rows];
        (packed_u32, scales)
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

    /// Packed parallel forward — 4× less memory bandwidth than unpacked.
    /// Fastest path for decode (seq=1).
    pub fn forward_packed_parallel(&self, input: &[f32], seq: usize) -> Vec<f32> {
        self.forward_packed_parallel_impl(input, seq, None, 0, None)
    }

    /// Prefill forward: batch the tokens that chose each expert and run the
    /// parallel expert kernel on the batch.
    ///
    /// The serial path runs one expert matmul per (token, top-k slot); at
    /// prefill lengths those are hundreds of independent one-row matmuls, so no
    /// thread ever has more than one row to work on. This groups the rows of a
    /// top-k slot by expert, which lets the existing rayon kernel split a real
    /// token batch across threads.
    ///
    /// The arithmetic is unchanged: each token still accumulates its top-k
    /// contributions in top-k order, and each row of a batched ternary matmul
    /// uses the same accumulation loop as the single-row kernel. `seq <= 1` is
    /// routed to the serial path, so decode is untouched.
    ///
    /// LoRA and routing collection need the per-token path; the caller must not
    /// use this method when either is active.
    pub fn forward_prefill(&self, input: &[f32], seq: usize) -> Vec<f32> {
        if seq <= 1 {
            return self.forward(input, seq);
        }

        let h = self.hidden_dim;
        let ne = self.num_experts;
        let tk = self.top_k;

        let gate_logits = matmul(input, &self.gate_weights, seq, h, ne);
        let (selected_experts, expert_weights, _gate_probs) =
            self.top_k_select_with_probs(&gate_logits, seq);

        let mut output = vec![0.0f32; seq * h];
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); ne];

        // One top-k slot at a time, so every token receives its contributions
        // in the same order as the serial loop.
        for k_idx in 0..tk {
            for toks in groups.iter_mut() {
                toks.clear();
            }
            for t in 0..seq {
                if expert_weights[t * tk + k_idx].abs() < 1e-8 {
                    continue;
                }
                groups[selected_experts[t * tk + k_idx]].push(t);
            }

            for (expert_idx, toks) in groups.iter().enumerate() {
                if toks.is_empty() {
                    continue;
                }
                let batch: Vec<f32> = toks
                    .iter()
                    .flat_map(|&t| input[t * h..(t + 1) * h].iter().copied())
                    .collect();
                let expert_out = self.expert_forward_batch(expert_idx, &batch, toks.len());

                for (row, &t) in toks.iter().enumerate() {
                    let weight = expert_weights[t * tk + k_idx];
                    let src = row * h;
                    let dst = t * h;
                    for i in 0..h {
                        output[dst + i] += weight * expert_out[src + i];
                    }
                }
            }
        }

        output
    }

    /// Expert forward for a token batch: the three projections on the parallel
    /// kernel, with the same activation as the serial expert path.
    fn expert_forward_batch(&self, expert_idx: usize, input: &[f32], rows: usize) -> Vec<f32> {
        let id = self.intermediate_dim;

        let gated = self.expert_gate[expert_idx].forward_batch(input, rows);
        let gated: Vec<f32> = if self.use_gelu {
            gated.iter().map(|&x| crate::model::gemma_mapper::gelu_tanh(x)).collect()
        } else {
            gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
        };

        let upped = self.expert_up[expert_idx].forward_batch(input, rows);

        let mut combined = vec![0.0f32; rows * id];
        for i in 0..combined.len() {
            combined[i] = gated[i] * upped[i];
        }

        self.expert_down[expert_idx].forward_batch(&combined, rows)
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

    /// Packed parallel MoE forward — 4× less memory bandwidth.
    fn forward_packed_parallel_impl(
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
                let expert_out = self.expert_forward_packed_parallel_with_lora(expert_idx, token, lora_manager, layer_idx);

                for (i, &v) in expert_out.iter().enumerate() {
                    output[t * h + i] += weight * v;
                }
            }
        }

        output
    }

    /// Expert forward with unpacked parallel matmul for each projection.
    fn expert_forward_parallel_with_lora(
        &self,
        expert_idx: usize,
        token: &[f32],
        lora_manager: Option<&crate::training::lora::LoraManager>,
        layer_idx: usize,
    ) -> Vec<f32> {
        let id = self.intermediate_dim;

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

        let mut upped = self.expert_up[expert_idx].forward_parallel(token, 1);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.up", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, token, 1) {
                for (i, v) in lo.iter().enumerate() { upped[i] += v; }
            }
        }

        let mut combined = vec![0.0; id];
        for i in 0..id { combined[i] = gated[i] * upped[i]; }

        let mut down_out = self.expert_down[expert_idx].forward_parallel(&combined, 1);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.down", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, &combined, 1) {
                for (i, v) in lo.iter().enumerate() { down_out[i] += v; }
            }
        }

        down_out
    }

    /// Expert forward with packed parallel matmul — 4× less memory bandwidth.
    fn expert_forward_packed_parallel_with_lora(
        &self,
        expert_idx: usize,
        token: &[f32],
        lora_manager: Option<&crate::training::lora::LoraManager>,
        layer_idx: usize,
    ) -> Vec<f32> {
        let id = self.intermediate_dim;

        let mut gated = self.expert_gate[expert_idx].forward_packed_parallel(token, 1);
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

        let mut upped = self.expert_up[expert_idx].forward_packed_parallel(token, 1);
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.up", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, token, 1) {
                for (i, v) in lo.iter().enumerate() { upped[i] += v; }
            }
        }

        let mut combined = vec![0.0; id];
        for i in 0..id { combined[i] = gated[i] * upped[i]; }

        let mut down_out = self.expert_down[expert_idx].forward_packed_parallel(&combined, 1);
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

    /// Deterministic SplitMix64 so a failing case is reproducible without a
    /// fixture.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }

        fn ternary(&mut self) -> i8 {
            ((self.next_u64() >> 33) % 3) as i8 - 1
        }
    }

    fn random_expert(rng: &mut Rng, rows: usize, cols: usize) -> TernaryExpert {
        let values: Vec<i8> = (0..rows * cols).map(|_| rng.ternary()).collect();
        TernaryExpert { packed: pack_ternary(&values), values, scale: 0.02, rows, cols }
    }

    /// Random router and distinct random experts, so both the top-k selection
    /// and the per-expert weight matter.
    fn random_moe(
        rng: &mut Rng,
        hidden_dim: usize,
        intermediate_dim: usize,
        num_experts: usize,
        top_k: usize,
    ) -> CpuMoELayer {
        let mut moe = CpuMoELayer::new(hidden_dim, intermediate_dim, num_experts, top_k);
        moe.use_gelu = true;
        for w in moe.gate_weights.iter_mut() {
            *w = rng.unit() * 0.5;
        }
        for e in 0..num_experts {
            moe.expert_gate[e] = random_expert(rng, intermediate_dim, hidden_dim);
            moe.expert_up[e] = random_expert(rng, intermediate_dim, hidden_dim);
            moe.expert_down[e] = random_expert(rng, hidden_dim, intermediate_dim);
        }
        moe
    }

    /// The batched prefill path must reproduce the serial path exactly: same
    /// top-k choice, same expert order, same per-row accumulation.
    #[test]
    fn moe_prefill_matches_serial() {
        let (hd, id, ne, tk, seq) = (16usize, 24usize, 4usize, 2usize, 12usize);
        let mut rng = Rng(0xE0E0_0003);
        let moe = random_moe(&mut rng, hd, id, ne, tk);
        let input: Vec<f32> = (0..seq * hd).map(|_| rng.unit()).collect();

        // The batch must exercise more than one expert at more than one slot,
        // otherwise the grouping would be a no-op.
        let logits = matmul(&input, &moe.gate_weights, seq, hd, ne);
        let (selected, weights, _) = moe.top_k_select_with_probs(&logits, seq);
        let mut pairs: Vec<Vec<usize>> =
            (0..seq).map(|t| selected[t * tk..(t + 1) * tk].to_vec()).collect();
        pairs.sort();
        pairs.dedup();
        assert!(pairs.len() > 1, "router picked one expert pair for every token: {pairs:?}");
        assert!(
            weights.iter().skip(1).step_by(tk).any(|&w| w > 0.05),
            "top-k weights are degenerate, the second slot would not matter"
        );

        let serial = moe.forward(&input, seq);
        let batched = moe.forward_prefill(&input, seq);
        assert_eq!(serial.len(), batched.len());
        let mut worst = 0.0f32;
        for (&a, &b) in serial.iter().zip(&batched) {
            worst = worst.max((a - b).abs());
        }
        assert!(serial.iter().any(|x| x.abs() > 1e-6), "test produced zero output");
        assert!(worst < 1e-5, "batched MoE differs from serial by {worst}");
    }

    /// One top-k slot: the result must be exactly the weight times the expert
    /// output, and a single-token prefill must go through the serial path.
    #[test]
    fn moe_prefill_single_token_is_serial() {
        let mut rng = Rng(0xE0E0_0004);
        let moe = random_moe(&mut rng, 8, 16, 2, 1);
        let input: Vec<f32> = (0..8).map(|_| rng.unit()).collect();
        assert_eq!(moe.forward(&input, 1), moe.forward_prefill(&input, 1));
    }

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
