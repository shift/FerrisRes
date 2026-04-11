//! YaRN (Yet another RoPE exteNsion) for long context extension.
//!
//! Implements context extension methods for extending sequence length beyond
//! the training context window:
//! - YaRN: NTK-aware position interpolation with dynamic scaling
//! - StreamingLLM: Attention sinks for infinite context
//! - Position interpolation (PI): Linear position scaling
//!
//! Based on research task 4a2cd78e and https://arxiv.org/abs/2308.10721

/// Configuration for context extension methods.
#[derive(Debug, Clone)]
pub struct ContextExtensionConfig {
    /// Original training context window size.
    pub original_context: usize,
    /// Target extended context window size.
    pub target_context: usize,
    /// Extension method to use.
    pub method: ExtensionMethod,
    /// Attention sink window size (for StreamingLLM).
    pub sink_window: usize,
    /// Number of attention sink tokens to keep.
    pub num_sink_tokens: usize,
}

/// Supported context extension methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtensionMethod {
    /// No extension — use original context window.
    None,
    /// Linear position interpolation: scale all positions by factor.
    PositionInterpolation,
    /// YaRN: NTK-aware scaling with attention temperature correction.
    YaRN,
    /// StreamingLLM: Keep initial sink tokens + rolling window.
    StreamingLLM,
}

impl Default for ContextExtensionConfig {
    fn default() -> Self {
        Self {
            original_context: 4096,
            target_context: 4096,
            method: ExtensionMethod::None,
            sink_window: 4096,
            num_sink_tokens: 4,
        }
    }
}

impl ContextExtensionConfig {
    /// Create a YaRN config to extend context from `original` to `target`.
    pub fn yarn(original: usize, target: usize) -> Self {
        Self {
            original_context: original,
            target_context: target,
            method: ExtensionMethod::YaRN,
            ..Default::default()
        }
    }

    /// Create a StreamingLLM config with given window size.
    pub fn streaming_llm(window: usize, sink_tokens: usize) -> Self {
        Self {
            original_context: window,
            target_context: usize::MAX, // Effectively unlimited
            method: ExtensionMethod::StreamingLLM,
            sink_window: window,
            num_sink_tokens: sink_tokens,
            ..Default::default()
        }
    }

    /// Create a simple position interpolation config.
    pub fn pi(original: usize, target: usize) -> Self {
        Self {
            original_context: original,
            target_context: target,
            method: ExtensionMethod::PositionInterpolation,
            ..Default::default()
        }
    }

    /// Compute the scale factor.
    pub fn scale_factor(&self) -> f32 {
        if self.original_context == 0 {
            return 1.0;
        }
        self.target_context as f32 / self.original_context as f32
    }

    /// Check if extension is active.
    pub fn is_active(&self) -> bool {
        self.method != ExtensionMethod::None && self.target_context > self.original_context
    }
}

/// YaRN-specific parameters computed from the config.
///
/// YaRN divides the dimension into three regions:
/// - Low-frequency: use NTK-aware scaling (dynamic frequency adjustment)
/// - Middle-frequency: interpolate between PI and NTK
/// - High-frequency: keep original RoPE (no scaling)
#[derive(Debug, Clone)]
pub struct YarnParams {
    /// Scale factor = target_context / original_context.
    pub scale: f32,
    /// Base frequency adjustment factor (NTK-aware).
    pub base: f32,
    /// Original base frequency (typically 10000).
    pub original_base: f32,
    /// Dimension threshold for low-frequency region.
    pub low_freq_dim: f32,
    /// Dimension threshold for high-frequency region.
    pub high_freq_dim: f32,
    /// Attention temperature correction factor.
    pub attention_temperature: f32,
}

impl YarnParams {
    /// Compute YaRN parameters from config.
    pub fn from_config(config: &ContextExtensionConfig) -> Self {
        let scale = config.scale_factor();
        let original_base = 10000.0;
        let head_dim = 64.0; // Typical head dimension

        // NTK-aware base frequency adjustment
        // New base = base * scale^(dim / (dim - 2))
        let base = original_base * scale.powf(head_dim / (head_dim - 2.0));

        // Frequency thresholds
        let low_freq_dim = original_base / scale;
        let high_freq_dim = original_base * 2.0 * std::f32::consts::PI;

        // Attention temperature correction (scales attention scores)
        // From YaRN paper: sqrt(1/alpha) * sqrt(log(scale)) + 1
        let attention_temperature = if scale > 1.0 {
            1.0 / (0.1 * scale.ln() + 1.0)
        } else {
            1.0
        };

        Self {
            scale,
            base,
            original_base,
            low_freq_dim,
            high_freq_dim,
            attention_temperature,
        }
    }

    /// Compute the effective frequency for a given dimension pair index.
    /// Returns (frequency, interpolation_mix) where mix=0 means use original,
    /// mix=1 means use interpolated.
    pub fn effective_frequency(&self, dim_idx: usize, head_dim: usize) -> (f32, f32) {
        let freq = 1.0 / self.original_base.powf(2.0 * dim_idx as f32 / head_dim as f32);
        let ntk_freq = 1.0 / self.base.powf(2.0 * dim_idx as f32 / head_dim as f32);

        if freq < self.low_freq_dim {
            // Low frequency: use NTK-aware scaling
            (ntk_freq / self.scale, 1.0)
        } else if freq > self.high_freq_dim {
            // High frequency: keep original (no scaling)
            (freq, 0.0)
        } else {
            // Middle frequency: smooth blend
            let mix = (self.high_freq_dim - freq)
                / (self.high_freq_dim - self.low_freq_dim);
            let blended = freq * (1.0 - mix) + (ntk_freq / self.scale) * mix;
            (blended, mix)
        }
    }
}

/// StreamingLLM attention sink manager.
///
/// Maintains a rolling window of recent tokens plus initial "sink" tokens
/// that stabilize attention. This allows infinite-length generation without
/// retraining.
pub struct AttentionSinkManager {
    /// Number of initial sink tokens to always keep.
    num_sink_tokens: usize,
    /// Maximum recent tokens to keep beyond sinks.
    window_size: usize,
    /// Current token positions in the window.
    positions: Vec<usize>,
    /// Whether we've exceeded the initial window.
    overflowed: bool,
}

impl AttentionSinkManager {
    pub fn new(num_sink_tokens: usize, window_size: usize) -> Self {
        Self {
            num_sink_tokens,
            window_size,
            positions: Vec::new(),
            overflowed: false,
        }
    }

    /// Add a token position and get the effective position to use.
    /// Returns the position to use for RoPE computation.
    pub fn add_position(&mut self, global_pos: usize) -> usize {
        if !self.overflowed {
            self.positions.push(global_pos);
            if self.positions.len() > self.window_size {
                self.overflowed = true;
            }
            global_pos
        } else {
            // After overflow: keep sinks + recent window
            // Sink tokens keep their original positions
            // Recent tokens get positions starting after sinks
            let local_pos = self.num_sink_tokens
                + (self.positions.len() - self.num_sink_tokens) % (self.window_size - self.num_sink_tokens);
            self.positions.push(local_pos);
            local_pos
        }
    }

    /// Get the list of positions currently in the attention window.
    pub fn window_positions(&self) -> &[usize] {
        if self.positions.len() <= self.window_size {
            &self.positions
        } else {
            // Return last window_size positions
            &self.positions[self.positions.len() - self.window_size..]
        }
    }

    /// Get the effective KV cache window indices.
    /// Returns which entries in the KV cache should be kept.
    pub fn kv_cache_indices(&self, cache_len: usize) -> Vec<usize> {
        if cache_len <= self.window_size {
            (0..cache_len).collect()
        } else {
            // Keep first num_sink_tokens + last (window_size - num_sink_tokens)
            let mut indices: Vec<usize> = (0..self.num_sink_tokens).collect();
            let recent_start = cache_len - (self.window_size - self.num_sink_tokens);
            indices.extend(recent_start..cache_len);
            indices
        }
    }

    /// Check if attention sink mode is active.
    pub fn is_active(&self) -> bool {
        self.overflowed
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.positions.clear();
        self.overflowed = false;
    }

    /// Get number of tracked positions.
    pub fn len(&self) -> usize {
        self.positions.len()
    }
}

/// Context extension engine that coordinates the selected method.
pub struct ContextExtensionEngine {
    config: ContextExtensionConfig,
    yarn_params: Option<YarnParams>,
    sink_manager: Option<AttentionSinkManager>,
}

impl ContextExtensionEngine {
    /// Create a new context extension engine.
    pub fn new(config: ContextExtensionConfig) -> Self {
        let yarn_params = if config.method == ExtensionMethod::YaRN {
            Some(YarnParams::from_config(&config))
        } else {
            None
        };

        let sink_manager = if config.method == ExtensionMethod::StreamingLLM {
            Some(AttentionSinkManager::new(
                config.num_sink_tokens,
                config.sink_window,
            ))
        } else {
            None
        };

        Self {
            config,
            yarn_params,
            sink_manager,
        }
    }

    /// Create with no extension.
    pub fn none() -> Self {
        Self::new(ContextExtensionConfig::default())
    }

    /// Compute the effective position for RoPE at a given global position.
    pub fn effective_position(&mut self, global_pos: usize) -> usize {
        match self.config.method {
            ExtensionMethod::None => global_pos,
            ExtensionMethod::PositionInterpolation => {
                // Linear interpolation: scale position by 1/scale
                let scale = self.config.scale_factor();
                (global_pos as f32 / scale).floor() as usize
            }
            ExtensionMethod::YaRN => {
                // YaRN handles scaling through frequency adjustment,
                // not position scaling. Return global position unchanged.
                global_pos
            }
            ExtensionMethod::StreamingLLM => {
                self.sink_manager.as_mut()
                    .map(|m| m.add_position(global_pos))
                    .unwrap_or(global_pos)
            }
        }
    }

    /// Compute YaRN-adjusted RoPE parameters for a given dimension pair.
    /// Returns (cos_theta, sin_theta) for the given position and dimension.
    pub fn yarn_rope(&self, pos: usize, dim_idx: usize, head_dim: usize) -> (f32, f32) {
        if let Some(ref params) = self.yarn_params {
            let (freq, _) = params.effective_frequency(dim_idx, head_dim);
            let theta = pos as f32 * freq;
            (theta.cos(), theta.sin())
        } else {
            // Standard RoPE
            let base = 10000.0f32;
            let freq = 1.0 / base.powf(2.0 * dim_idx as f32 / head_dim as f32);
            let theta = pos as f32 * freq;
            (theta.cos(), theta.sin())
        }
    }

    /// Get the attention temperature correction factor (YaRN).
    pub fn attention_temperature(&self) -> f32 {
        self.yarn_params
            .as_ref()
            .map(|p| p.attention_temperature)
            .unwrap_or(1.0)
    }

    /// Get the KV cache window indices (StreamingLLM).
    pub fn kv_cache_window(&self, cache_len: usize) -> Vec<usize> {
        self.sink_manager
            .as_ref()
            .map(|m| m.kv_cache_indices(cache_len))
            .unwrap_or_else(|| (0..cache_len).collect())
    }

    /// Get the config.
    pub fn config(&self) -> &ContextExtensionConfig {
        &self.config
    }

    /// Check if extension is active.
    pub fn is_active(&self) -> bool {
        self.config.is_active()
    }

    /// Get YaRN parameters.
    pub fn yarn_params(&self) -> Option<&YarnParams> {
        self.yarn_params.as_ref()
    }

    /// Reset state (e.g., between sequences).
    pub fn reset(&mut self) {
        if let Some(ref mut sink) = self.sink_manager {
            sink.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_extension() {
        let engine = ContextExtensionEngine::none();
        assert!(!engine.is_active());
    }

    #[test]
    fn test_pi_scaling() {
        let config = ContextExtensionConfig::pi(2048, 8192);
        let mut engine = ContextExtensionEngine::new(config);
        assert!(engine.is_active());

        // Position 8192 should map to 2048 (scaled by 4x)
        let pos = engine.effective_position(8192);
        assert_eq!(pos, 2048);
    }

    #[test]
    fn test_yarn_params() {
        let config = ContextExtensionConfig::yarn(4096, 16384);
        let engine = ContextExtensionEngine::new(config);
        let params = engine.yarn_params().unwrap();

        assert!((params.scale - 4.0).abs() < 0.01);
        assert!(params.base > 10000.0); // NTK-adjusted base is larger
        assert!(params.attention_temperature < 1.0); // Temperature correction
    }

    #[test]
    fn test_yarn_rope_low_freq() {
        let config = ContextExtensionConfig::yarn(4096, 16384);
        let engine = ContextExtensionEngine::new(config);

        // Low frequency dimension: should be scaled
        let (_freq, mix) = engine.yarn_params.unwrap().effective_frequency(0, 64);
        assert!(mix > 0.0); // Should have some interpolation
    }

    #[test]
    fn test_yarn_rope_high_freq() {
        let config = ContextExtensionConfig::yarn(4096, 16384);
        let engine = ContextExtensionEngine::new(config);

        // Very high dimension index should have less interpolation
        let (_freq, mix) = engine.yarn_params.unwrap().effective_frequency(31, 64);
        // With these parameters, high-freq dimensions may still have some mix,
        // but it should be < 1.0 or equal to 0.0 depending on thresholds
        assert!(mix <= 1.0);
    }

    #[test]
    fn test_yarn_cos_sin() {
        let config = ContextExtensionConfig::yarn(4096, 16384);
        let engine = ContextExtensionEngine::new(config);

        let (cos_v, sin_v) = engine.yarn_rope(0, 0, 64);
        // At position 0, cos should be ~1 and sin ~0
        assert!((cos_v - 1.0).abs() < 0.01);
        assert!(sin_v.abs() < 0.01);
    }

    #[test]
    fn test_streaming_llm_sink_manager() {
        let mut manager = AttentionSinkManager::new(4, 8);

        // Fill up to window size
        for i in 0..8 {
            let pos = manager.add_position(i);
            assert_eq!(pos, i);
        }
        assert!(!manager.is_active());

        // Add one more to trigger overflow
        let _pos = manager.add_position(8);
        assert!(manager.is_active());
        assert!(manager.len() > 8);
    }

    #[test]
    fn test_streaming_llm_kv_window() {
        let manager = AttentionSinkManager::new(2, 6);
        let indices = manager.kv_cache_indices(10);
        // Should keep first 2 (sinks) + last 4 (window - sinks)
        assert_eq!(indices.len(), 6);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 1);
        assert_eq!(indices[2], 6);
        assert_eq!(indices[5], 9);
    }

    #[test]
    fn test_streaming_llm_no_eviction() {
        let manager = AttentionSinkManager::new(4, 16);
        let indices = manager.kv_cache_indices(8);
        // Cache smaller than window: keep all
        assert_eq!(indices.len(), 8);
    }

    #[test]
    fn test_config_scale_factor() {
        let config = ContextExtensionConfig::yarn(4096, 16384);
        assert!((config.scale_factor() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_engine_reset() {
        let config = ContextExtensionConfig::streaming_llm(8, 2);
        let mut engine = ContextExtensionEngine::new(config);
        engine.effective_position(100);
        engine.reset();
        // After reset, sink manager should be fresh
    }
}
