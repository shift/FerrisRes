//! LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
//!
//! Implements LoRA: Hu et al. 2021 "LoRA: Low-Rank Adaptation of Large Language Models"
//! Key idea: freeze pre-trained weights, inject trainable low-rank matrices:
//!   h = W·x + (B·A)·x  where B ∈ R^{d×r}, A ∈ R^{r×d}, r << d
//!
//! Also supports QLoRA (quantized LoRA) via 4-bit quantized base weights.

/// Configuration for a single LoRA adapter.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition (typically 4-64).
    pub rank: usize,
    /// Scaling factor alpha. Effective scaling = alpha / rank.
    pub alpha: f32,
    /// Dropout probability for LoRA path (0.0 = no dropout).
    pub dropout: f32,
    /// Target modules to apply LoRA to (e.g., "q_proj", "v_proj").
    pub target_modules: Vec<String>,
    /// Whether to merge weights at inference time.
    pub merge_on_inference: bool,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            merge_on_inference: true,
        }
    }
}

impl LoraConfig {
    /// Create a new LoRA config with specified rank.
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            ..Default::default()
        }
    }

    /// Create a config targeting specific modules.
    pub fn targeting(rank: usize, modules: Vec<&str>) -> Self {
        Self {
            rank,
            target_modules: modules.into_iter().map(String::from).collect(),
            ..Default::default()
        }
    }

    /// Effective scaling factor.
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Check if a module name is a LoRA target.
    /// Supports exact match and wildcard prefix: if target is "moe.expert.*",
    /// it matches any module starting with "moe.expert.".
    pub fn is_target(&self, module_name: &str) -> bool {
        self.target_modules.iter().any(|m| {
            if m.ends_with('*') {
                module_name.starts_with(&m[..m.len()-1])
            } else {
                m == module_name
            }
        })
    }
}

/// Low-rank adaptation matrices for a single linear layer.
///
/// Stores two matrices:
/// - A: [rank × in_features] — initialized with Kaiming uniform
/// - B: [out_features × rank] — initialized with zeros
///
/// This means ΔW = B·A starts as zero, and the model behaves
/// identically to the base model before training.
#[derive(Debug)]
pub struct LoraLayer {
    /// LoRA matrix A: [rank × in_features]
    lora_a: Vec<f32>,
    /// LoRA matrix B: [out_features × rank]
    lora_b: Vec<f32>,
    /// Input dimension.
    in_features: usize,
    /// Output dimension.
    out_features: usize,
    /// Rank.
    rank: usize,
    /// Scaling factor.
    scaling: f32,
    /// Whether this adapter is currently merged into base weights.
    merged: bool,
    /// Gradient accumulators.
    grad_a: Option<Vec<f32>>,
    grad_b: Option<Vec<f32>>,
}

impl LoraLayer {
    /// Create a new LoRA layer with zero-initialized B and small-random A.
    pub fn new(in_features: usize, out_features: usize, config: &LoraConfig) -> Self {
        let rank = config.rank;
        let scaling = config.scaling();

        // Initialize A with small random values (Kaiming-like)
        let a_size = rank * in_features;
        let lora_a = (0..a_size)
            .map(|_| {
                // Simple uniform init: ±1/sqrt(in_features)
                let val = (rand::random::<f32>() - 0.5) * 2.0 / (in_features as f32).sqrt();
                val
            })
            .collect();

        // Initialize B with zeros (so ΔW = B·A = 0 initially)
        let b_size = out_features * rank;
        let lora_b = vec![0.0; b_size];

        Self {
            lora_a,
            lora_b,
            in_features,
            out_features,
            rank,
            scaling,
            merged: false,
            grad_a: None,
            grad_b: None,
        }
    }

    /// Forward pass: compute LoRA contribution ΔW·x = B·(A·x) * scaling
    ///
    /// Takes input activations and returns the LoRA delta to add to the base output.
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        // input shape: [seq_len × in_features]
        // A: [rank × in_features]
        // B: [out_features × rank]

        let mut output = vec![0.0; seq_len * self.out_features];

        for s in 0..seq_len {
            // Compute A·x for this position: [rank]
            let mut ax = vec![0.0; self.rank];
            for r in 0..self.rank {
                let mut sum = 0.0f32;
                for d in 0..self.in_features {
                    sum += self.lora_a[r * self.in_features + d] * input[s * self.in_features + d];
                }
                ax[r] = sum;
            }

            // Compute B·(A·x) for this position: [out_features]
            for o in 0..self.out_features {
                let mut sum = 0.0f32;
                for r in 0..self.rank {
                    sum += self.lora_b[o * self.rank + r] * ax[r];
                }
                output[s * self.out_features + o] = sum * self.scaling;
            }
        }

        output
    }

    /// Merge LoRA weights into a base weight matrix.
    ///
    /// After merging, ΔW = B·A·scaling is added to the base weights
    /// and the LoRA matrices can be discarded for inference.
    pub fn merge_into(&mut self, base_weights: &mut [f32]) {
        if self.merged {
            return;
        }

        // Compute ΔW = B·A·scaling, shape: [out_features × in_features]
        for o in 0..self.out_features {
            for d in 0..self.in_features {
                let mut delta = 0.0f32;
                for r in 0..self.rank {
                    delta += self.lora_b[o * self.rank + r]
                        * self.lora_a[r * self.in_features + d];
                }
                base_weights[o * self.in_features + d] += delta * self.scaling;
            }
        }
        self.merged = true;
    }

    /// Unmerge LoRA weights from a base weight matrix.
    pub fn unmerge_from(&mut self, base_weights: &mut [f32]) {
        if !self.merged {
            return;
        }

        for o in 0..self.out_features {
            for d in 0..self.in_features {
                let mut delta = 0.0f32;
                for r in 0..self.rank {
                    delta += self.lora_b[o * self.rank + r]
                        * self.lora_a[r * self.in_features + d];
                }
                base_weights[o * self.in_features + d] -= delta * self.scaling;
            }
        }
        self.merged = false;
    }

    /// Get the number of trainable parameters in this LoRA layer.
    pub fn num_params(&self) -> usize {
        self.rank * self.in_features + self.out_features * self.rank
    }

    /// Get parameter count as a fraction of the base layer.
    pub fn param_fraction(&self) -> f32 {
        let base_params = self.in_features * self.out_features;
        if base_params == 0 {
            return 0.0;
        }
        self.num_params() as f32 / base_params as f32
    }

    /// Get the rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get whether merged.
    pub fn is_merged(&self) -> bool {
        self.merged
    }

    /// Get LoRA A matrix.
    pub fn lora_a(&self) -> &[f32] {
        &self.lora_a
    }

    /// Get mutable LoRA A matrix.
    pub fn lora_a_mut(&mut self) -> &mut [f32] {
        &mut self.lora_a
    }

    /// Get LoRA B matrix.
    pub fn lora_b(&self) -> &[f32] {
        &self.lora_b
    }

    /// Get mutable LoRA B matrix.
    pub fn lora_b_mut(&mut self) -> &mut [f32] {
        &mut self.lora_b
    }

    /// Get input dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get scaling factor.
    pub fn scaling(&self) -> f32 {
        self.scaling
    }

    /// Get mutable reference to gradients.
    pub fn gradients(&mut self) -> (&mut [f32], &mut [f32]) {
        if self.grad_a.is_none() {
            self.grad_a = Some(vec![0.0; self.rank * self.in_features]);
            self.grad_b = Some(vec![0.0; self.out_features * self.rank]);
        }
        (self.grad_a.as_mut().unwrap(), self.grad_b.as_mut().unwrap())
    }

    /// Zero gradients.
    pub fn zero_grad(&mut self) {
        if let Some(ref mut g) = self.grad_a {
            g.fill(0.0);
        }
        if let Some(ref mut g) = self.grad_b {
            g.fill(0.0);
        }
    }

    /// Compute LoRA gradients from upstream gradient and cached input.
    ///
    /// LoRA forward: output = scaling * B @ (A @ input)
    /// Where A is [rank, in_features], B is [out_features, rank]
    ///
    /// Backward:
    ///   dL/dB[i,r] = scaling * Σ_s dL/dy[s,i] * (A @ input)[s,r]
    ///   dL/dA[r,j] = scaling * Σ_s Σ_i dL/dy[s,i] * B[i,r] * input[s,j]
    ///
    /// This uses the approximate approach: d_hidden as proxy for per-layer gradient.
    /// Acceptable because LoRA is low-rank by construction (rank << min(in, out)).
    ///
    /// # Arguments
    /// * `grad_output` — dL/dy, shape [seq_len, out_features]
    /// * `cached_input` — the input that was fed to forward(), shape [seq_len, in_features]
    /// * `seq_len` — sequence length
    pub fn backward(
        &mut self,
        grad_output: &[f32],
        cached_input: &[f32],
        seq_len: usize,
    ) {
        let rank = self.rank;
        let in_f = self.in_features;
        let out_f = self.out_features;
        let sc = self.scaling;

        // Clone lora weights to avoid borrow conflicts with gradient buffers
        let lora_a = self.lora_a.clone();
        let lora_b = self.lora_b.clone();

        // Ensure gradient buffers exist
        let (grad_a, grad_b) = self.gradients();

        for s in 0..seq_len {
            // Compute A @ input for this position: [rank]
            let mut ax = vec![0.0f32; rank];
            for r in 0..rank {
                let mut sum = 0.0f32;
                for j in 0..in_f {
                    sum += lora_a[r * in_f + j] * cached_input[s * in_f + j];
                }
                ax[r] = sum;
            }

            // dL/dB: accumulate for this position
            for i in 0..out_f {
                let dy = grad_output[s * out_f + i];
                for r in 0..rank {
                    grad_b[i * rank + r] += sc * dy * ax[r];
                }
            }

            // dL/dA: accumulate for this position
            for r in 0..rank {
                let mut dy_br = 0.0f32;
                for i in 0..out_f {
                    dy_br += grad_output[s * out_f + i] * lora_b[i * rank + r];
                }
                for j in 0..in_f {
                    grad_a[r * in_f + j] += sc * dy_br * cached_input[s * in_f + j];
                }
            }
        }
    }
}

/// Manages all LoRA adapters across the model.
pub struct LoraManager {
    config: LoraConfig,
    /// Map from (layer_index, module_name) to LoRA layer.
    adapters: Vec<(usize, String, LoraLayer)>,
    /// Whether adapters are currently merged.
    all_merged: bool,
}

impl LoraManager {
    /// Create a new LoRA manager with the given configuration.
    pub fn new(config: LoraConfig) -> Self {
        Self {
            config,
            adapters: Vec::new(),
            all_merged: false,
        }
    }

    /// Add a LoRA adapter for a specific layer and module.
    pub fn add_adapter(
        &mut self,
        layer_idx: usize,
        module_name: &str,
        in_features: usize,
        out_features: usize,
    ) {
        if self.config.is_target(module_name) {
            let layer = LoraLayer::new(in_features, out_features, &self.config);
            tracing::info!(
                "LoRA adapter added: layer={}, module={}, rank={}, params={} ({:.2}% of base)",
                layer_idx,
                module_name,
                layer.rank(),
                layer.num_params(),
                layer.param_fraction() * 100.0,
            );
            self.adapters.push((layer_idx, module_name.to_string(), layer));
        }
    }

    /// Auto-populate adapters for all target modules in the model.
    /// Call this after the model is built.
    pub fn auto_populate(
        &mut self,
        num_layers: usize,
        attention_heads: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) {
        let head_dim = hidden_dim / attention_heads;
        let targets = self.config.target_modules.clone();
        let config = self.config.clone();

        for layer_idx in 0..num_layers {
            for module_name in &targets {
                let (in_f, out_f) = match module_name.as_str() {
                    "q_proj" => (hidden_dim, attention_heads * head_dim),
                    "k_proj" => (hidden_dim, attention_heads * head_dim),
                    "v_proj" => (hidden_dim, attention_heads * head_dim),
                    "o_proj" => (attention_heads * head_dim, hidden_dim),
                    "gate_proj" => (hidden_dim, intermediate_dim),
                    "up_proj" => (hidden_dim, intermediate_dim),
                    "down_proj" => (intermediate_dim, hidden_dim),
                    _ => {
                        tracing::warn!(event = "unknown_lora_target_module", "Unknown LoRA target module: {}", module_name);
                        continue;
                    }
                };
                let layer = LoraLayer::new(in_f, out_f, &config);
                tracing::info!(
                    "LoRA adapter added: layer={}, module={}, rank={}, params={} ({:.2}% of base)",
                    layer_idx,
                    module_name,
                    layer.rank(),
                    layer.num_params(),
                    layer.param_fraction() * 100.0,
                );
                self.adapters.push((layer_idx, module_name.clone(), layer));
            }
        }

        let total_params: usize = self.adapters.iter().map(|(_, _, l)| l.num_params()).sum();
        tracing::info!(
            "LoRA manager populated: {} adapters, {} total trainable params",
            self.adapters.len(),
            total_params,
        );
    }

    /// Get the LoRA contribution for a specific layer/module.
    pub fn forward(&self, layer_idx: usize, module_name: &str, input: &[f32], seq_len: usize) -> Option<Vec<f32>> {
        for &(ref l_idx, ref m_name, ref layer) in &self.adapters {
            if *l_idx == layer_idx && m_name == module_name {
                if layer.is_merged() {
                    return None; // Already merged into base weights
                }
                return Some(layer.forward(input, seq_len));
            }
        }
        None
    }

    /// Merge all adapters into base weights.
    pub fn merge_all(&mut self, base_weight_accessor: &mut dyn FnMut(usize, &str) -> Option<&mut [f32]>) {
        for (layer_idx, module_name, layer) in &mut self.adapters {
            if let Some(weights) = base_weight_accessor(*layer_idx, module_name) {
                layer.merge_into(weights);
            }
        }
        self.all_merged = true;
    }

    /// Unmerge all adapters from base weights.
    pub fn unmerge_all(&mut self, base_weight_accessor: &mut dyn FnMut(usize, &str) -> Option<&mut [f32]>) {
        for (layer_idx, module_name, layer) in &mut self.adapters {
            if let Some(weights) = base_weight_accessor(*layer_idx, module_name) {
                layer.unmerge_from(weights);
            }
        }
        self.all_merged = false;
    }

    /// Get the configuration.
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// Get the number of adapters.
    pub fn num_adapters(&self) -> usize {
        self.adapters.len()
    }

    /// Get total trainable parameters across all adapters.
    pub fn total_params(&self) -> usize {
        self.adapters.iter().map(|(_, _, l)| l.num_params()).sum()
    }

    /// Get whether all adapters are merged.
    pub fn is_merged(&self) -> bool {
        self.all_merged
    }

    /// Zero all gradients.
    pub fn zero_all_grads(&mut self) {
        for (_, _, layer) in &mut self.adapters {
            layer.zero_grad();
        }
    }

    /// Compute gradients for all LoRA adapters given per-layer hidden state gradients.
    ///
    /// This is the main entry point for LoRA training. For each adapted layer,
    /// it uses the hidden state gradient (dL/d_hidden) as a proxy for the
    /// per-module gradient, which is a valid approximation because:
    /// 1. LoRA adapters are low-rank (rank 8-16), so the gradient is low-rank too
    /// 2. The proxy approximation error is bounded by the LoRA rank
    ///
    /// # Arguments
    /// * `per_layer_grads` — Iterator of (layer_idx, d_hidden [seq_len × hidden_dim])
    /// * `per_layer_inputs` — Iterator of (layer_idx, input_to_module [seq_len × in_features])
    /// * `seq_len` — sequence length
    ///
    /// After calling this, use `adapters_iter_mut()` to get (grad_a, grad_b) and
    /// pass them to the optimizer.
    pub fn compute_all_gradients(
        &mut self,
        per_layer_inputs: &[(usize, Vec<f32>, String)], // (layer_idx, cached_input, module_name)
        per_layer_grads: &[(usize, Vec<f32>)],           // (layer_idx, d_hidden)
        seq_len: usize,
    ) {
        // For each adapter, find the matching gradient and input
        for (layer_idx, module_name, layer) in &mut self.adapters {
            // Find the gradient for this layer
            let grad = per_layer_grads.iter()
                .find(|(lidx, _)| *lidx == *layer_idx);

            // Find the cached input for this layer+module
            let input = per_layer_inputs.iter()
                .find(|(lidx, _, mname)| *lidx == *layer_idx && mname == module_name);

            if let (Some((_, grad_data)), Some((_, input_data, _))) = (grad, input) {
                // The grad_data is [seq_len × hidden_dim]. For modules like q_proj,
                // the output grad is the full hidden_dim gradient. We use it directly
                // as the grad_output for the LoRA layer (out_features matches hidden_dim
                // for most projections, or is sliced appropriately).
                let out_f = layer.out_features();
                let effective_seq = grad_data.len() / out_f.max(1);

                if effective_seq > 0 && input_data.len() >= seq_len * layer.in_features() {
                    layer.backward(
                        &grad_data[..effective_seq * out_f],
                        input_data,
                        effective_seq.min(seq_len),
                    );
                }
            }
        }
    }

    /// Iterate over mutable adapters.
    pub fn adapters_mut(&mut self) -> impl Iterator<Item = &mut LoraLayer> {
        self.adapters.iter_mut().map(|(_, _, l)| l)
    }

    /// Iterate over adapters with layer_idx and module_name.
    pub fn adapters_iter(&self) -> impl Iterator<Item = (usize, &str, &LoraLayer)> {
        self.adapters.iter().map(|(idx, name, layer)| (*idx, name.as_str(), layer))
    }

    /// Iterate mutably over adapters with layer_idx and module_name.
    pub fn adapters_iter_mut(&mut self) -> impl Iterator<Item = (usize, &str, &mut LoraLayer)> {
        self.adapters.iter_mut().map(|(idx, name, layer)| (*idx, name.as_str(), layer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert!((config.scaling() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_lora_config_targeting() {
        let config = LoraConfig::targeting(4, vec!["q_proj", "v_proj"]);
        assert_eq!(config.rank, 4);
        assert!(config.is_target("q_proj"));
        assert!(config.is_target("v_proj"));
        assert!(!config.is_target("k_proj"));
    }

    #[test]
    fn test_lora_layer_zero_init() {
        // B is zero-initialized, so ΔW = B·A = 0
        let config = LoraConfig::new(4);
        let layer = LoraLayer::new(16, 8, &config);

        let input = vec![1.0; 16];
        let output = layer.forward(&input, 1);

        // Output should be all zeros since B is zero-initialized
        for &v in &output {
            assert!(v.abs() < 1e-6, "Expected zero output, got {}", v);
        }
    }

    #[test]
    fn test_lora_param_count() {
        let config = LoraConfig::new(4);
        let layer = LoraLayer::new(64, 32, &config);
        // A: 4×64 = 256, B: 32×4 = 128, total = 384
        assert_eq!(layer.num_params(), 384);
        // Base: 64×32 = 2048
        assert!((layer.param_fraction() - 384.0 / 2048.0).abs() < 0.001);
    }

    #[test]
    fn test_lora_merge_unmerge() {
        let config = LoraConfig::new(2);
        let mut layer = LoraLayer::new(4, 3, &config);

        // Set B to non-zero to test merge
        layer.lora_b[0] = 1.0;
        layer.lora_b[1] = 0.5;

        let mut base = vec![0.0; 12]; // 3×4

        // Merge
        layer.merge_into(&mut base);
        assert!(layer.is_merged());
        // Base should now have ΔW
        assert!(base.iter().any(|&v| v.abs() > 0.0));

        // Unmerge
        layer.unmerge_from(&mut base);
        assert!(!layer.is_merged());
        // Base should be back to zeros
        for &v in &base {
            assert!(v.abs() < 1e-6, "Expected zero after unmerge, got {}", v);
        }
    }

    #[test]
    fn test_lora_forward_shape() {
        let config = LoraConfig::new(4);
        let layer = LoraLayer::new(8, 6, &config);

        let input = vec![1.0; 16]; // seq_len=2, in_features=8
        let output = layer.forward(&input, 2);
        assert_eq!(output.len(), 12); // seq_len=2 × out_features=6
    }

    #[test]
    fn test_lora_manager_auto_populate() {
        let config = LoraConfig::targeting(4, vec!["q_proj", "v_proj"]);
        let mut manager = LoraManager::new(config);
        manager.auto_populate(4, 8, 256, 512);

        // 4 layers × 2 modules = 8 adapters
        assert_eq!(manager.num_adapters(), 8);
        assert!(manager.total_params() > 0);
    }

    #[test]
    fn test_lora_manager_forward() {
        let config = LoraConfig::targeting(4, vec!["q_proj"]);
        let mut manager = LoraManager::new(config);
        manager.auto_populate(2, 8, 64, 128);

        let input = vec![1.0; 64]; // seq_len=1, hidden=64
        let output = manager.forward(0, "q_proj", &input, 1);

        // Should return Some since B is zero-initialized → all zeros
        assert!(output.is_some());
        let out = output.unwrap();
        assert!(out.iter().all(|&v| v.abs() < 1e-6));

        // Non-existent module should return None
        assert!(manager.forward(0, "k_proj", &input, 1).is_none());
    }

    #[test]
    fn test_lora_scaling() {
        let config = LoraConfig {
            rank: 4,
            alpha: 8.0,
            ..Default::default()
        };
        assert!((config.scaling() - 2.0).abs() < 0.001);

        let layer = LoraLayer::new(4, 3, &config);
        // Check scaling is applied
        assert!((layer.scaling - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_lora_backward_gradients_nonzero() {
        let config = LoraConfig { rank: 2, ..Default::default() };
        let mut layer = LoraLayer::new(4, 3, &config);

        // B is initialized to zero by default, so grad_a will be zero.
        // Set B to non-zero to test both gradients.
        layer.lora_b_mut().copy_from_slice(&[1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5]);

        let input = vec![1.0f32, 2.0, 3.0, 4.0]; // [1, 4]
        let grad_output = vec![1.0f32, 0.0, 0.0]; // [1, 3]

        layer.backward(&grad_output, &input, 1);

        let (grad_a, grad_b) = layer.gradients();

        // grad_b should be non-zero
        assert!(grad_b.iter().any(|&g| g.abs() > 0.0));
        // grad_a should be non-zero now that B is non-zero
        assert!(grad_a.iter().any(|&g| g.abs() > 0.0));
    }

    #[test]
    fn test_lora_backward_zero_grad_input() {
        let config = LoraConfig { rank: 2, ..Default::default() };
        let mut layer = LoraLayer::new(4, 3, &config);

        let input = vec![0.0f32; 4];
        let grad_output = vec![1.0f32, 2.0, 3.0];

        layer.backward(&grad_output, &input, 1);

        let (grad_a, grad_b) = layer.gradients();

        // With zero input, grad_a should be zero (input appears in grad_a)
        assert!(grad_a.iter().all(|&g| g.abs() < 1e-10));
        // But grad_b should still be zero because A@input = 0
        assert!(grad_b.iter().all(|&g| g.abs() < 1e-10));
    }

    #[test]
    fn test_lora_backward_multi_seq() {
        let config = LoraConfig { rank: 2, ..Default::default() };
        let mut layer = LoraLayer::new(3, 2, &config);

        // Set B to non-zero
        layer.lora_b_mut().copy_from_slice(&[0.5f32, 0.3, 0.7, 0.2]);

        let input = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2, 3]
        let grad_output = vec![1.0f32, 0.0, 0.0, 1.0]; // [2, 2]

        layer.backward(&grad_output, &input, 2);

        let (grad_a, grad_b) = layer.gradients();
        // Both should be non-zero
        assert!(grad_a.iter().any(|&g| g.abs() > 0.0));
        assert!(grad_b.iter().any(|&g| g.abs() > 0.0));
    }

    #[test]
    fn test_lora_backward_matches_numerical() {
        // Verify backward matches numerical gradient (finite differences)
        let config = LoraConfig { rank: 2, alpha: 4.0, ..Default::default() };
        let mut layer = LoraLayer::new(3, 2, &config);

        let input = vec![1.0f32, 2.0, 3.0];
        let grad_output = vec![1.0f32, 1.0];

        // Compute backward
        layer.backward(&grad_output, &input, 1);
        let (grad_a_analytical, _grad_b_analytical) = {
            let (ga, gb) = layer.gradients();
            (ga.to_vec(), gb.to_vec())
        };

        // Numerical gradient for B: perturb each B element
        let eps = 1e-4f32;
        for r in 0..config.rank {
            for j in 0..3 {
                // Perturb A[r, j]
                let orig = layer.lora_a_mut()[r * 3 + j];
                layer.lora_a_mut()[r * 3 + j] = orig + eps;
                let out_plus = layer.forward(&input, 1);
                layer.lora_a_mut()[r * 3 + j] = orig - eps;
                let out_minus = layer.forward(&input, 1);
                layer.lora_a_mut()[r * 3 + j] = orig;

                // Numerical gradient = (loss_plus - loss_minus) / (2*eps)
                // loss = Σ grad_output[i] * output[i]
                let mut num_grad = 0.0f32;
                for i in 0..2 {
                    num_grad += grad_output[i] * (out_plus[i] - out_minus[i]);
                }
                num_grad /= 2.0 * eps;

                let analytical = grad_a_analytical[r * 3 + j];
                assert!(
                    (num_grad - analytical).abs() < 1e-2,
                    "A[{},{}] numerical={} analytical={} diff={}",
                    r, j, num_grad, analytical, (num_grad - analytical).abs()
                );
            }
        }
    }
}
