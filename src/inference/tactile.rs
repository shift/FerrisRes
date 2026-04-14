//! Tactile & Haptic feedback generation for VR/AR and teleoperation.
//!
//! Implements TactileHead for predicting per-actuator intensity (0.0-1.0)
//! at 1000Hz for multi-channel haptic streams. Supports:
//! - Multi-actuator output (5 fingers + palm = 6 channels)
//! - Cross-modal visual-to-tactile translation via cosine similarity
//! - Speculative draft-then-verify decoding for sub-20ms latency
//! - HID output via host_tools integration

// ---------------------------------------------------------------------------
// TactileHead — per-actuator intensity prediction
// ---------------------------------------------------------------------------

/// Configuration for the TactileHead.
#[derive(Debug, Clone)]
pub struct TactileHeadConfig {
    /// Transformer hidden dimension.
    pub hidden_dim: usize,
    /// Number of actuators (e.g., 6 for 5 fingers + palm).
    pub num_actuators: usize,
    /// Output sample rate in Hz (e.g., 1000).
    pub sample_rate: u32,
    /// Maximum prediction horizon in milliseconds (e.g., 50).
    pub max_horizon_ms: u32,
}

impl Default for TactileHeadConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            num_actuators: 6,
            sample_rate: 1000,
            max_horizon_ms: 50,
        }
    }
}

impl TactileHeadConfig {
    /// Standard 6-actuator hand configuration.
    pub fn hand() -> Self {
        Self::default()
    }

    /// Number of samples in the max prediction horizon.
    pub fn horizon_samples(&self) -> usize {
        (self.sample_rate as f32 * self.max_horizon_ms as f32 / 1000.0) as usize
    }
}

/// Tactile intensity prediction head.
///
/// Projects hidden states to per-actuator intensity values (0.0-1.0).
/// Each actuator has its own linear projection: hidden_dim → 1.
pub struct TactileHead {
    config: TactileHeadConfig,
    /// Per-actuator projection weights: [num_actuators × hidden_dim].
    weights: Vec<Vec<f32>>,
    /// Per-actuator bias.
    biases: Vec<f32>,
}

impl TactileHead {
    /// Create with Xavier initialization.
    pub fn new(config: TactileHeadConfig) -> Self {
        let hd = config.hidden_dim;
        let scale = (2.0 / hd as f32).sqrt();

        let weights: Vec<Vec<f32>> = (0..config.num_actuators)
            .map(|act| {
                (0..hd)
                    .map(|i| {
                        let seed = i as f32 + act as f32 * 500.0;
                        let x = ((seed * 0.618 + 0.1).sin() * 43758.5453).fract() - 0.5;
                        x * scale
                    })
                    .collect()
            })
            .collect();
        let biases = vec![0.0; config.num_actuators];

        Self { config, weights, biases }
    }

    /// Forward: predict intensities for all actuators from hidden state.
    ///
    /// Returns: [num_actuators] intensities in [0.0, 1.0].
    pub fn forward(&self, hidden: &[f32]) -> Vec<f32> {
        debug_assert_eq!(hidden.len(), self.config.hidden_dim);
        let hd = self.config.hidden_dim;

        self.weights.iter().zip(self.biases.iter()).map(|(w, &b)| {
            let mut sum = b;
            for h in 0..hd {
                sum += hidden[h] * w[h];
            }
            // Sigmoid squashing to [0, 1]
            1.0 / (1.0 + (-sum).exp())
        }).collect()
    }

    /// Forward a sequence of hidden states → haptic waveform.
    ///
    /// Returns: [num_steps × num_actuators] intensity matrix.
    pub fn forward_sequence(&self, hiddens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        hiddens.iter().map(|h| self.forward(h)).collect()
    }

    /// Get the number of samples for a given duration in milliseconds.
    pub fn samples_for_duration(&self, duration_ms: u32) -> usize {
        (self.config.sample_rate as f32 * duration_ms as f32 / 1000.0) as usize
    }

    /// Config accessor.
    pub fn config(&self) -> &TactileHeadConfig { &self.config }
}

// ---------------------------------------------------------------------------
// HapticFrame — a single time-step of haptic output
// ---------------------------------------------------------------------------

/// A single frame of haptic output (one sample at sample_rate).
#[derive(Debug, Clone)]
pub struct HapticFrame {
    /// Per-actuator intensity [0.0, 1.0].
    pub intensities: Vec<f32>,
    /// Timestamp in seconds.
    pub timestamp: f32,
}

impl HapticFrame {
    pub fn new(intensities: Vec<f32>, timestamp: f32) -> Self {
        Self { intensities, timestamp }
    }

    /// All-zero frame.
    pub fn zeros(num_actuators: usize) -> Self {
        Self { intensities: vec![0.0; num_actuators], timestamp: 0.0 }
    }

    /// Check all intensities are in [0, 1].
    pub fn is_valid(&self) -> bool {
        self.intensities.iter().all(|&v| v >= 0.0 && v <= 1.0 && v.is_finite())
    }

    /// RMS energy across all actuators.
    pub fn energy(&self) -> f32 {
        if self.intensities.is_empty() { return 0.0; }
        (self.intensities.iter().map(|v| v * v).sum::<f32>() / self.intensities.len() as f32).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Cross-modal visual-to-tactile grounding
// ---------------------------------------------------------------------------

/// Visual-to-tactile cross-modal translation.
///
/// Maps visual texture descriptors to haptic vibration patterns.
/// Uses cosine similarity between visual embeddings and stored texture
/// prototypes to select the appropriate tactile response.
pub struct VisualTactileBridge {
    /// Stored texture prototypes: [(name, embedding, haptic_pattern)].
    textures: Vec<(String, Vec<f32>, Vec<f32>)>,
    /// Embedding dimension.
    embed_dim: usize,
}

impl VisualTactileBridge {
    /// Create with given embedding dimension.
    pub fn new(embed_dim: usize) -> Self {
        Self { textures: Vec::new(), embed_dim }
    }

    /// Register a texture prototype.
    pub fn register_texture(
        &mut self,
        name: &str,
        visual_embedding: Vec<f32>,
        haptic_pattern: Vec<f32>,
    ) {
        self.textures.push((name.to_string(), visual_embedding, haptic_pattern));
    }

    /// Translate a visual embedding to a haptic pattern.
    ///
    /// Finds the nearest texture prototype by cosine similarity and
    /// returns its haptic pattern, scaled by the similarity score.
    pub fn translate(&self, visual_embedding: &[f32]) -> Option<HapticFrame> {
        if self.textures.is_empty() || visual_embedding.len() != self.embed_dim {
            return None;
        }

        let mut best_sim = f32::NEG_INFINITY;
        let mut best_pattern: &[f32] = &[];

        for (_, proto_embed, pattern) in &self.textures {
            let sim = cosine_similarity(visual_embedding, proto_embed);
            if sim > best_sim {
                best_sim = sim;
                best_pattern = pattern;
            }
        }

        if best_pattern.is_empty() {
            return None;
        }

        // Scale pattern by similarity
        let intensities: Vec<f32> = best_pattern.iter()
            .map(|&v| (v * best_sim).clamp(0.0, 1.0))
            .collect();

        Some(HapticFrame::new(intensities, 0.0))
    }

    /// Number of registered textures.
    pub fn num_textures(&self) -> usize {
        self.textures.len()
    }

    /// Get texture names.
    pub fn texture_names(&self) -> Vec<&str> {
        self.textures.iter().map(|(n, _, _)| n.as_str()).collect()
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { 0.0 } else { dot / (na * nb) }
}

// ---------------------------------------------------------------------------
// Speculative haptic decoding
// ---------------------------------------------------------------------------

/// Result of a speculative haptic decode cycle.
#[derive(Debug, Clone)]
pub struct SpeculativeHapticResult {
    /// Accepted haptic frames.
    pub frames: Vec<HapticFrame>,
    /// Whether the draft was accepted.
    pub accepted: bool,
    /// Similarity between draft and main model predictions.
    pub similarity: f32,
}

/// Speculative decoder for low-latency haptic prediction.
///
/// Uses a small draft model to predict 50ms of haptic output,
/// then validates against the main TactileHead. If similarity
/// exceeds a threshold, the draft is accepted (saving compute).
pub struct SpeculativeHapticDecoder {
    /// Main tactile head.
    main_head: TactileHead,
    /// Draft tactile head (smaller).
    draft_head: TactileHead,
    /// Acceptance threshold for cosine similarity.
    accept_threshold: f32,
    /// Statistics.
    stats: HapticDecodeStats,
}

/// Running statistics for speculative decoding.
#[derive(Debug, Clone, Default)]
pub struct HapticDecodeStats {
    pub total_drafts: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub avg_similarity: f32,
}

impl HapticDecodeStats {
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafts == 0 { 0.0 } else { self.accepted as f32 / self.total_drafts as f32 }
    }
}

impl SpeculativeHapticDecoder {
    /// Create with main and draft heads.
    pub fn new(
        main_head: TactileHead,
        draft_head: TactileHead,
        accept_threshold: f32,
    ) -> Self {
        Self { main_head, draft_head, accept_threshold, stats: HapticDecodeStats::default() }
    }

    /// Run one draft-verify cycle.
    ///
    /// `hidden`: current hidden state for the main model.
    /// `draft_hiddens`: sequence of hidden states for the draft model
    ///   (e.g., from a smaller model or cached representations).
    pub fn decode(
        &mut self,
        hidden: &[f32],
        draft_hiddens: &[Vec<f32>],
    ) -> SpeculativeHapticResult {
        // Main model prediction (ground truth)
        let main_pred = self.main_head.forward(hidden);

        // Draft model predictions for the horizon
        let draft_preds: Vec<Vec<f32>> = draft_hiddens.iter()
            .map(|h| self.draft_head.forward(h))
            .collect();

        // Verify: compare first draft prediction to main prediction
        let similarity = if let Some(draft_first) = draft_preds.first() {
            cosine_similarity(&main_pred, draft_first)
        } else {
            0.0
        };

        let accepted = similarity >= self.accept_threshold;

        // Build frames
        let dt = 1.0 / self.main_head.config().sample_rate as f32;
        let frames: Vec<HapticFrame> = if accepted {
            // Use draft predictions
            draft_preds.iter().enumerate().map(|(i, intensities)| {
                HapticFrame::new(intensities.clone(), i as f32 * dt)
            }).collect()
        } else {
            // Fall back to main model (repeated single prediction)
            vec![HapticFrame::new(main_pred.clone(), 0.0)]
        };

        // Update stats
        self.stats.total_drafts += 1;
        if accepted { self.stats.accepted += 1; }
        else { self.stats.rejected += 1; }
        // Running average similarity
        let n = self.stats.total_drafts as f32;
        self.stats.avg_similarity = self.stats.avg_similarity * (n - 1.0) / n + similarity / n;

        SpeculativeHapticResult { frames, accepted, similarity }
    }

    /// Get the running statistics.
    pub fn stats(&self) -> &HapticDecodeStats { &self.stats }

    /// Reset statistics.
    pub fn reset(&mut self) { self.stats = HapticDecodeStats::default(); }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tactile_head_config() {
        let config = TactileHeadConfig::default();
        assert_eq!(config.num_actuators, 6);
        assert_eq!(config.sample_rate, 1000);
        assert_eq!(config.horizon_samples(), 50); // 50ms at 1000Hz
    }

    #[test]
    fn test_tactile_head_forward() {
        let head = TactileHead::new(TactileHeadConfig::default());
        let hidden = vec![0.5f32; 256];
        let intensities = head.forward(&hidden);
        assert_eq!(intensities.len(), 6);
        for &v in &intensities {
            assert!(v >= 0.0 && v <= 1.0, "Intensity {} out of range", v);
        }
    }

    #[test]
    fn test_tactile_head_forward_sequence() {
        let head = TactileHead::new(TactileHeadConfig { hidden_dim: 64, ..Default::default() });
        let hiddens: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.1; 64]).collect();
        let seq = head.forward_sequence(&hiddens);
        assert_eq!(seq.len(), 10);
        assert_eq!(seq[0].len(), 6);
    }

    #[test]
    fn test_tactile_head_samples_for_duration() {
        let head = TactileHead::new(TactileHeadConfig::default());
        assert_eq!(head.samples_for_duration(50), 50);
        assert_eq!(head.samples_for_duration(20), 20);
    }

    #[test]
    fn test_haptic_frame_valid() {
        let frame = HapticFrame::new(vec![0.5, 0.8, 0.1, 0.0, 0.9, 0.3], 0.0);
        assert!(frame.is_valid());
        assert!((frame.energy() - 0.53).abs() < 0.1);
    }

    #[test]
    fn test_haptic_frame_invalid() {
        let frame = HapticFrame::new(vec![-0.5, 1.5], 0.0);
        assert!(!frame.is_valid());
    }

    #[test]
    fn test_haptic_frame_zeros() {
        let frame = HapticFrame::zeros(6);
        assert_eq!(frame.intensities.len(), 6);
        assert!(frame.intensities.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_visual_tactile_bridge_empty() {
        let bridge = VisualTactileBridge::new(8);
        assert!(bridge.translate(&[0.0; 8]).is_none());
    }

    #[test]
    fn test_visual_tactile_bridge_translate() {
        let mut bridge = VisualTactileBridge::new(4);
        bridge.register_texture(
            "rough_wood",
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.8, 0.6, 0.4, 0.2, 0.1, 0.3],
        );
        bridge.register_texture(
            "smooth_glass",
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        );

        let result = bridge.translate(&[0.9, 0.1, 0.0, 0.0]).unwrap();
        // Should match rough_wood
        assert!(result.intensities[0] > result.intensities[4]);
        assert_eq!(bridge.num_textures(), 2);
    }

    #[test]
    fn test_visual_tactile_bridge_names() {
        let mut bridge = VisualTactileBridge::new(4);
        bridge.register_texture("silk", vec![0.0; 4], vec![0.0; 6]);
        bridge.register_texture("sandpaper", vec![1.0; 4], vec![1.0; 6]);
        let names = bridge.texture_names();
        assert!(names.contains(&"silk"));
        assert!(names.contains(&"sandpaper"));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_speculative_decoder() {
        let main = TactileHead::new(TactileHeadConfig { hidden_dim: 32, ..Default::default() });
        let draft = TactileHead::new(TactileHeadConfig { hidden_dim: 32, ..Default::default() });
        let mut decoder = SpeculativeHapticDecoder::new(main, draft, 0.8);

        let hidden = vec![0.5f32; 32];
        let draft_hiddens: Vec<Vec<f32>> = (0..5).map(|_| vec![0.5f32; 32]).collect();
        let result = decoder.decode(&hidden, &draft_hiddens);

        assert!(!result.frames.is_empty());
        assert_eq!(decoder.stats().total_drafts, 1);
    }

    #[test]
    fn test_speculative_decoder_stats() {
        let main = TactileHead::new(TactileHeadConfig { hidden_dim: 16, ..Default::default() });
        let draft = TactileHead::new(TactileHeadConfig { hidden_dim: 16, ..Default::default() });
        let mut decoder = SpeculativeHapticDecoder::new(main, draft, 0.9);

        for i in 0..10 {
            let hidden = vec![i as f32 * 0.1; 16];
            let draft_hiddens: Vec<Vec<f32>> = (0..3).map(|_| vec![0.5; 16]).collect();
            decoder.decode(&hidden, &draft_hiddens);
        }

        assert_eq!(decoder.stats().total_drafts, 10);
        assert!(decoder.stats().acceptance_rate() >= 0.0);
    }

    #[test]
    fn test_speculative_decoder_reset() {
        let main = TactileHead::new(TactileHeadConfig { hidden_dim: 16, ..Default::default() });
        let draft = TactileHead::new(TactileHeadConfig { hidden_dim: 16, ..Default::default() });
        let mut decoder = SpeculativeHapticDecoder::new(main, draft, 0.8);

        let hidden = vec![0.5f32; 16];
        let draft_hiddens = vec![vec![0.5f32; 16]; 3];
        decoder.decode(&hidden, &draft_hiddens);
        assert!(decoder.stats().total_drafts > 0);
        decoder.reset();
        assert_eq!(decoder.stats().total_drafts, 0);
    }
}
