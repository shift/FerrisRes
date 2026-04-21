use crate::model::gemma_mapper::matmul;

/// Routing data collected during MoE forward pass.
/// Used for load balance loss computation and router gradient.
#[derive(Clone, Debug)]
pub struct MoERoutingData {
    /// Layer index this routing data came from.
    pub layer_idx: usize,
    /// Raw router logits: [seq, num_experts] (pre-softmax).
    pub gate_logits: Vec<f32>,
    /// Softmax probabilities over ALL experts: [seq, num_experts].
    pub gate_probs: Vec<f32>,
    /// Selected expert indices: [seq, top_k].
    pub selected_experts: Vec<usize>,
    /// Expert weights after top-k softmax: [seq, top_k].
    pub expert_weights: Vec<f32>,
    /// Pre-FFN normalized input: [seq, hidden_dim].
    pub pre_ffn_input: Vec<f32>,
    /// Expert outputs per token: [seq, top_k, hidden_dim].
    pub expert_outputs: Vec<f32>,
}

/// CPU-only Mixture of Experts layer. Stores all weights as `Vec<f32>`.
/// Used by CpuBlockAttnResLayer for MoE FFN.
///
/// Each expert has a gate_proj, up_proj, and down_proj.
/// The router computes softmax logits, selects top-k experts per token,
/// and produces a weighted sum of expert outputs.
///
/// Activation: supports both SwiGLU (silu-based) and GeLU (tanh approximation).
pub struct CpuMoELayer {
    /// Router weights: [hidden_dim, num_experts].
    pub gate_weights: Vec<f32>,
    /// Expert gate projections (for activation): [num_experts][intermediate * hidden].
    pub expert_gate: Vec<Vec<f32>>,
    /// Expert up projections: [num_experts][intermediate * hidden].
    pub expert_up: Vec<Vec<f32>>,
    /// Expert down projections: [num_experts][hidden * intermediate].
    pub expert_down: Vec<Vec<f32>>,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    /// Activation type: true = GeLU (tanh approx), false = SwiGLU (silu).
    pub use_gelu: bool,
}

impl CpuMoELayer {
    pub fn new(hidden_dim: usize, intermediate_dim: usize, num_experts: usize, top_k: usize) -> Self {
        let gate_weights = vec![0.0; num_experts * hidden_dim];
        let expert_gate = (0..num_experts)
            .map(|_| vec![0.0; intermediate_dim * hidden_dim])
            .collect();
        let expert_up = (0..num_experts)
            .map(|_| vec![0.0; intermediate_dim * hidden_dim])
            .collect();
        let expert_down = (0..num_experts)
            .map(|_| vec![0.0; hidden_dim * intermediate_dim])
            .collect();

        Self {
            gate_weights,
            expert_gate,
            expert_up,
            expert_down,
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k,
            use_gelu: false, // Default SwiGLU
        }
    }

    /// Forward pass: route tokens to top-k experts, compute weighted output.
    /// Input: [seq, hidden_dim], Output: [seq, hidden_dim].
    /// Forward pass: route tokens to top-k experts, compute weighted output.
    /// Input: [seq, hidden_dim], Output: [seq, hidden_dim].
    pub fn forward(&self, input: &[f32], seq: usize) -> Vec<f32> {
        self.forward_with_routing(input, seq, None, 0, None).0
    }

    /// Forward pass that also collects routing data for loss/gradient computation.
    /// Returns (output [seq, hidden_dim], Option<MoERoutingData>).
    ///
    /// When `routing_collector` is Some, routing data is appended to it.
    /// This avoids a separate forward pass — routing is collected inline.
    ///
    /// `layer_idx`: layer number for routing data metadata.
    /// `lora_manager` / `lora_layer_idx`: if Some, apply LoRA to expert projections.
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

        // 1. Compute gate logits: [seq, num_experts]
        let gate_logits = matmul(input, &self.gate_weights, seq, h, ne);

        // 2. Top-k selection with softmax weighting + collect full softmax probs
        let (selected_experts, expert_weights, gate_probs) = self.top_k_select_with_probs(&gate_logits, seq);

        // 3. Compute expert outputs and accumulate
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

                // Store expert output for routing collector
                if !expert_outputs.is_empty() {
                    let dst = t * tk * h + k_idx * h;
                    expert_outputs[dst..dst + h].copy_from_slice(&expert_out);
                }

                for (i, &v) in expert_out.iter().enumerate() {
                    output[t * h + i] += weight * v;
                }
            }
        }

        // 4. Collect routing data
        if let Some(collector) = routing_collector {
            collector.push(MoERoutingData {
                layer_idx,
                gate_logits: gate_logits.clone(),
                gate_probs,
                selected_experts: selected_experts.clone(),
                expert_weights: expert_weights.clone(),
                pre_ffn_input: input.to_vec(),
                expert_outputs,
            });
        }

        (output, None)
    }

    /// Top-k expert selection with softmax normalization.
    /// Returns (selected_expert_indices[seq * top_k], weights[seq * top_k]).
    /// Top-k selection with softmax, also returns full softmax probabilities.
    /// Returns (selected[seq*top_k], weights[seq*top_k], probs[seq*num_experts]).
    fn top_k_select_with_probs(&self, gate_logits: &[f32], seq: usize) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
        let ne = self.num_experts;
        let tk = self.top_k;
        let mut selected = vec![0usize; seq * tk];
        let mut weights = vec![0.0f32; seq * tk];
        let mut all_probs = vec![0.0f32; seq * ne];

        for t in 0..seq {
            let logits = &gate_logits[t * ne..(t + 1) * ne];

            // Full softmax for load balance loss
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for (e, &l) in logits.iter().enumerate() {
                all_probs[t * ne + e] = (l - max_logit).exp();
                sum_exp += all_probs[t * ne + e];
            }
            if sum_exp > 0.0 {
                for e in 0..ne { all_probs[t * ne + e] /= sum_exp; }
            }

            // Find top-k
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Softmax over top-k logits for weight normalization
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
    /// `lora_manager`: if Some, apply LoRA deltas to expert projections.
    /// `layer_idx`: layer number for LoRA lookup.
    ///
    /// LoRA target names: "moe.expert.{e}.gate", "moe.expert.{e}.up", "moe.expert.{e}.down"
    fn expert_forward_with_lora(
        &self,
        expert_idx: usize,
        token: &[f32],
        lora_manager: Option<&crate::training::lora::LoraManager>,
        layer_idx: usize,
    ) -> Vec<f32> {
        let h = self.hidden_dim;
        let id = self.intermediate_dim;
        let gate = &self.expert_gate[expert_idx];
        let up = &self.expert_up[expert_idx];
        let down = &self.expert_down[expert_idx];

        // gate projection: [1, h] @ [h, id] = [1, id]
        let mut gated = matmul(token, gate, 1, h, id);
        // Apply LoRA to gate
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.gate", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, token, 1) {
                for (i, v) in lo.iter().enumerate() { gated[i] += v; }
            }
        }
        // Apply activation
        let gated: Vec<f32> = if self.use_gelu {
            gated.iter().map(|&x| crate::model::gemma_mapper::gelu_tanh(x)).collect()
        } else {
            gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
        };

        // up projection: [1, h] @ [h, id] = [1, id]
        let mut upped = matmul(token, up, 1, h, id);
        // Apply LoRA to up
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.up", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, token, 1) {
                for (i, v) in lo.iter().enumerate() { upped[i] += v; }
            }
        }

        // Element-wise multiply
        let mut combined = vec![0.0; id];
        for i in 0..id {
            combined[i] = gated[i] * upped[i];
        }

        // down projection: [1, id] @ [id, h] = [1, h]
        let mut down_out = matmul(&combined, down, 1, id, h);
        // Apply LoRA to down
        if let Some(lm) = lora_manager {
            let name = format!("moe.expert.{}.down", expert_idx);
            if let Some(lo) = lm.forward(layer_idx, &name, &combined, 1) {
                for (i, v) in lo.iter().enumerate() { down_out[i] += v; }
            }
        }

        down_out
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
        // Set router to always pick expert 0
        moe.gate_weights[0] = 10.0; // Expert 0 gets high score for token[0]
        // Make expert 0 identity-like: gate=eye, up=eye, down=eye
        for i in 0..hd.min(id) {
            moe.expert_gate[0][i * id + i] = 1.0;
            moe.expert_up[0][i * id + i] = 1.0;
            moe.expert_down[0][i * hd + i] = 1.0;
        }

        let input = vec![1.0, 2.0, 3.0, 4.0]; // seq=1, hd=4
        let output = moe.forward(&input, 1);
        assert_eq!(output.len(), 4);
        // With silu(1)=0.73, *up(1), then down: should produce nonzero output
        assert!(output.iter().any(|&x| x.abs() > 0.01));
    }
}
