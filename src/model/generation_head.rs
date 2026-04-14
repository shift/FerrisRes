//! Multimodal generation heads for image, speech, and video output.
//!
//! Implements output heads that predict discrete tokens for different modalities:
//! - **VisionHead**: Predicts VQ-VAE codebook indices for image generation
//!   (32×32 grid = 1024 tokens per image, supports progressive row-by-row decode)
//! - **SpeechHead**: Predicts EnCodec multi-codebook indices for TTS
//!   (N codebooks × T frames, integrates with existing EnCodecConv decoder)
//! - **VideoHead**: Predicts I-frame + motion-compensated P-frame residuals
//!   using Block AttnRes temporal structure
//!
//! All heads use CPU-only `Vec<f32>` weights — output heads run once per
//! autoregressive step, where GPU upload overhead exceeds CPU matmul cost.
//! See reasoning `7ee32b13` for rationale.

// ---------------------------------------------------------------------------
// VisionHead — image token prediction
// ---------------------------------------------------------------------------

/// Configuration for the VisionHead.
#[derive(Debug, Clone)]
pub struct VisionHeadConfig {
    /// Transformer hidden dimension.
    pub hidden_dim: usize,
    /// VQ-VAE codebook size (e.g., 8192).
    pub codebook_size: usize,
    /// Grid resolution (e.g., 32 → 32×32 = 1024 tokens).
    pub grid_size: usize,
}

impl VisionHeadConfig {
    pub fn new(hidden_dim: usize, codebook_size: usize, grid_size: usize) -> Self {
        Self { hidden_dim, codebook_size, grid_size }
    }

    /// Total tokens per image.
    pub fn num_tokens(&self) -> usize {
        self.grid_size * self.grid_size
    }

    /// Default: 768 hidden, 8192 codebook, 32×32 grid.
    pub fn default_768() -> Self {
        Self::new(768, 8192, 32)
    }
}

/// Image generation head: projects hidden states → VQ-VAE codebook logits.
///
/// Generates images as a grid of discrete tokens. Supports row-by-row
/// progressive decoding for streaming output to SSE/WebSocket clients.
pub struct VisionHead {
    config: VisionHeadConfig,
    /// Projection weights: [hidden_dim × codebook_size], row-major.
    weight: Vec<f32>,
    /// Bias: [codebook_size].
    bias: Vec<f32>,
}

impl VisionHead {
    /// Create with Xavier initialization.
    pub fn new(config: VisionHeadConfig) -> Self {
        let hd = config.hidden_dim;
        let cs = config.codebook_size;
        let scale = (2.0 / (hd + cs) as f32).sqrt();
        let weight: Vec<f32> = (0..hd * cs)
            .map(|i| {
                let x = ((i as f32 * 0.618 + 0.1).sin() * 43758.5453).fract() - 0.5;
                x * scale
            })
            .collect();
        let bias = vec![0.0; cs];
        Self { config, weight, bias }
    }

    /// Create with pre-trained weights.
    pub fn with_weights(config: VisionHeadConfig, weight: Vec<f32>, bias: Vec<f32>) -> Self {
        assert_eq!(weight.len(), config.hidden_dim * config.codebook_size);
        assert_eq!(bias.len(), config.codebook_size);
        Self { config, weight, bias }
    }

    /// Forward pass for a single position → [codebook_size] logits.
    pub fn forward(&self, hidden: &[f32]) -> Vec<f32> {
        debug_assert_eq!(hidden.len(), self.config.hidden_dim);
        let hd = self.config.hidden_dim;
        let cs = self.config.codebook_size;
        let mut logits = self.bias.clone();
        for c in 0..cs {
            let mut sum = 0.0f32;
            for h in 0..hd {
                sum += hidden[h] * self.weight[h * cs + c];
            }
            logits[c] += sum;
        }
        logits
    }

    /// Forward for an entire grid → [num_tokens × codebook_size] logits.
    pub fn forward_grid(&self, hiddens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        hiddens.iter().map(|h| self.forward(h)).collect()
    }

    /// Greedy decode: argmax per position → token indices.
    pub fn decode_greedy(&self, hiddens: &[Vec<f32>]) -> Vec<usize> {
        hiddens.iter().map(|h| argmax(&self.forward(h))).collect()
    }

    /// Progressive decode: yield rows of `grid_size` token indices.
    pub fn decode_progressive(&self, hiddens: &[Vec<f32>]) -> Vec<Vec<usize>> {
        let gs = self.config.grid_size;
        hiddens.chunks(gs).map(|row| {
            row.iter().map(|h| argmax(&self.forward(h))).collect()
        }).collect()
    }

    /// Convert flat tokens → 2D grid.
    pub fn tokens_to_grid(&self, tokens: &[usize]) -> Vec<Vec<usize>> {
        tokens.chunks(self.config.grid_size).map(|r| r.to_vec()).collect()
    }

    /// Config accessor.
    pub fn config(&self) -> &VisionHeadConfig { &self.config }

    /// Weight accessor (for serialization).
    pub fn weight(&self) -> &[f32] { &self.weight }

    /// Bias accessor (for serialization).
    pub fn bias(&self) -> &[f32] { &self.bias }
}

// ---------------------------------------------------------------------------
// SpeechHead — multi-codebook TTS prediction
// ---------------------------------------------------------------------------

/// Configuration for the SpeechHead.
#[derive(Debug, Clone)]
pub struct SpeechHeadConfig {
    /// Transformer hidden dimension.
    pub hidden_dim: usize,
    /// Number of EnCodec codebooks (e.g., 8).
    pub num_codebooks: usize,
    /// Codebook vocabulary size (e.g., 1024).
    pub codebook_size: usize,
}

impl SpeechHeadConfig {
    pub fn new(hidden_dim: usize, num_codebooks: usize, codebook_size: usize) -> Self {
        Self { hidden_dim, num_codebooks, codebook_size }
    }

    /// Default EnCodec-compatible config.
    pub fn default_encodec() -> Self {
        Self::new(768, 8, 1024)
    }
}

/// Speech generation head: predicts N codebook indices per frame.
///
/// Each codebook has its own linear projection from hidden_dim → codebook_size.
/// Voice conditioning via LoRA is supported through the LoraManager.
pub struct SpeechHead {
    config: SpeechHeadConfig,
    /// Per-codebook projection weights: [num_codebooks × (hidden_dim × codebook_size)].
    weights: Vec<Vec<f32>>,
    /// Per-codebook biases: [num_codebooks × codebook_size].
    biases: Vec<Vec<f32>>,
}

impl SpeechHead {
    /// Create with Xavier initialization.
    pub fn new(config: SpeechHeadConfig) -> Self {
        let hd = config.hidden_dim;
        let cs = config.codebook_size;
        let nc = config.num_codebooks;
        let scale = (2.0 / (hd + cs) as f32).sqrt();
        let weights: Vec<Vec<f32>> = (0..nc)
            .map(|cb| {
                (0..hd * cs)
                    .map(|i| {
                        let seed = i as f32 + cb as f32 * 1000.0;
                        let x = ((seed * 0.618 + 0.1).sin() * 43758.5453).fract() - 0.5;
                        x * scale
                    })
                    .collect()
            })
            .collect();
        let biases = (0..nc)
            .map(|_| vec![0.0f32; cs])
            .collect();
        Self { config, weights, biases }
    }

    /// Forward: predict all codebook logits for one frame.
    ///
    /// Returns: [num_codebooks] vectors of [codebook_size] logits.
    pub fn forward(&self, hidden: &[f32]) -> Vec<Vec<f32>> {
        debug_assert_eq!(hidden.len(), self.config.hidden_dim);
        let hd = self.config.hidden_dim;
        let cs = self.config.codebook_size;
        self.weights.iter().zip(self.biases.iter()).map(|(w, b)| {
            let mut logits = b.clone();
            for c in 0..cs {
                for h in 0..hd {
                    logits[c] += hidden[h] * w[h * cs + c];
                }
            }
            logits
        }).collect()
    }

    /// Forward for a sequence of frames → [num_frames × num_codebooks × codebook_size].
    pub fn forward_sequence(&self, hiddens: &[Vec<f32>]) -> Vec<Vec<Vec<f32>>> {
        hiddens.iter().map(|h| self.forward(h)).collect()
    }

    /// Greedy decode → [num_frames × num_codebooks] token indices.
    pub fn decode_greedy(&self, hiddens: &[Vec<f32>]) -> Vec<Vec<usize>> {
        hiddens.iter().map(|h| {
            self.forward(h).iter().map(|cb| argmax(cb)).collect()
        }).collect()
    }

    /// Config accessor.
    pub fn config(&self) -> &SpeechHeadConfig { &self.config }

    /// Number of codebooks.
    pub fn num_codebooks(&self) -> usize { self.config.num_codebooks }

    /// Codebook size.
    pub fn codebook_size(&self) -> usize { self.config.codebook_size }
}

// ---------------------------------------------------------------------------
// VideoHead — I-frame + P-frame residual prediction
// ---------------------------------------------------------------------------

/// Frame type for video generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoFrameType {
    /// Keyframe: full image via VisionHead.
    IFrame,
    /// Predicted frame: motion-compensated residual from previous frame.
    PFrame,
}

/// A generated video frame with metadata.
#[derive(Debug, Clone)]
pub struct GeneratedVideoFrame {
    /// Token indices (I-frame: grid_size², P-frame: compressed residual tokens).
    pub tokens: Vec<usize>,
    /// Frame type.
    pub frame_type: VideoFrameType,
    /// Frame index in the sequence.
    pub frame_idx: usize,
    /// Timestamp in seconds.
    pub timestamp: f32,
}

/// Configuration for the VideoHead.
#[derive(Debug, Clone)]
pub struct VideoHeadConfig {
    /// Transformer hidden dimension.
    pub hidden_dim: usize,
    /// VQ-VAE codebook size for I-frames.
    pub codebook_size: usize,
    /// Residual codebook size for P-frames (usually smaller).
    pub residual_codebook_size: usize,
    /// Spatial grid size (same as VisionHead).
    pub grid_size: usize,
    /// Target frames per second.
    pub fps: f32,
    /// Interval between I-frames (every N frames).
    pub iframe_interval: usize,
}

impl VideoHeadConfig {
    pub fn new(
        hidden_dim: usize,
        codebook_size: usize,
        residual_codebook_size: usize,
        grid_size: usize,
        fps: f32,
        iframe_interval: usize,
    ) -> Self {
        Self { hidden_dim, codebook_size, residual_codebook_size, grid_size, fps, iframe_interval }
    }

    /// Default: 30 fps, I-frame every 30 frames (1 second).
    pub fn default_30fps() -> Self {
        Self::new(768, 8192, 1024, 32, 30.0, 30)
    }
}

/// Video generation head: predicts I-frames and P-frame residuals.
///
/// Uses separate projections for keyframes (full codebook) and residuals
/// (smaller codebook). Temporal structure leverages Block AttnRes for
/// linear-time long-sequence processing.
pub struct VideoHead {
    config: VideoHeadConfig,
    /// I-frame projection: [hidden_dim × codebook_size].
    iframe_proj: Vec<f32>,
    /// P-frame residual projection: [hidden_dim × residual_codebook_size].
    pframe_proj: Vec<f32>,
    /// P-frame bias: [residual_codebook_size].
    pframe_bias: Vec<f32>,
    /// Last generated frame tokens (for residual reference).
    last_frame_tokens: Option<Vec<usize>>,
}

impl VideoHead {
    /// Create with Xavier initialization.
    pub fn new(config: VideoHeadConfig) -> Self {
        let hd = config.hidden_dim;
        let cs = config.codebook_size;
        let rcs = config.residual_codebook_size;

        let scale_if = (2.0 / (hd + cs) as f32).sqrt();
        let iframe_proj: Vec<f32> = (0..hd * cs)
            .map(|i| {
                let x = ((i as f32 * 0.618 + 0.1).sin() * 43758.5453).fract() - 0.5;
                x * scale_if
            })
            .collect();

        let scale_pf = (2.0 / (hd + rcs) as f32).sqrt();
        let pframe_proj: Vec<f32> = (0..hd * rcs)
            .map(|i| {
                let x = ((i as f32 * 0.314 + 0.2).sin() * 43758.5453).fract() - 0.5;
                x * scale_pf
            })
            .collect();
        let pframe_bias = vec![0.0; rcs];

        Self {
            config,
            iframe_proj,
            pframe_proj,
            pframe_bias,
            last_frame_tokens: None,
        }
    }

    /// Generate a single frame's token sequence.
    pub fn generate_frame(&mut self, hiddens: &[Vec<f32>], frame_idx: usize) -> GeneratedVideoFrame {
        let is_iframe = frame_idx % self.config.iframe_interval == 0;
        let frame_type = if is_iframe { VideoFrameType::IFrame } else { VideoFrameType::PFrame };
        let timestamp = frame_idx as f32 / self.config.fps;
        let hd = self.config.hidden_dim;

        let tokens: Vec<usize> = if is_iframe {
            let cs = self.config.codebook_size;
            hiddens.iter().map(|h| {
                let best = (0..cs).map(|c| {
                    let mut s = 0.0f32;
                    for d in 0..hd { s += h[d] * self.iframe_proj[d * cs + c]; }
                    (c, s)
                }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                best.map(|(i, _)| i).unwrap_or(0)
            }).collect()
        } else {
            let rcs = self.config.residual_codebook_size;
            hiddens.iter().map(|h| {
                let best = (0..rcs).map(|c| {
                    let mut s = self.pframe_bias[c];
                    for d in 0..hd { s += h[d] * self.pframe_proj[d * rcs + c]; }
                    (c, s)
                }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                best.map(|(i, _)| i).unwrap_or(0)
            }).collect()
        };

        self.last_frame_tokens = Some(tokens.clone());
        GeneratedVideoFrame { tokens, frame_type, frame_idx, timestamp }
    }

    /// Config accessor.
    pub fn config(&self) -> &VideoHeadConfig { &self.config }

    /// Last generated frame tokens.
    pub fn last_frame_tokens(&self) -> Option<&[usize]> {
        self.last_frame_tokens.as_deref()
    }
}

// ---------------------------------------------------------------------------
// VideoStreamReconstructor — decode token streams back to pixel frames
// ---------------------------------------------------------------------------

/// Reconstructs video frames from I-frame + P-frame token sequences.
///
/// Uses VQ-VAE codebook lookup for I-frames and additive residual
/// reconstruction for P-frames.
pub struct VideoStreamReconstructor {
    /// I-frame codebook: [codebook_size × embed_dim].
    iframe_codebook: Vec<f32>,
    /// P-frame residual codebook: [residual_codebook_size × embed_dim].
    pframe_codebook: Vec<f32>,
    /// Grid size.
    grid_size: usize,
    /// Embedding dimension per spatial position.
    embed_dim: usize,
    /// Last reconstructed frame (for P-frame prediction).
    last_frame: Option<Vec<f32>>,
}

impl VideoStreamReconstructor {
    /// Create with codebook weights.
    pub fn new(
        iframe_codebook: Vec<f32>,
        pframe_codebook: Vec<f32>,
        grid_size: usize,
        embed_dim: usize,
    ) -> Self {
        Self { iframe_codebook, pframe_codebook, grid_size, last_frame: None, embed_dim }
    }

    /// Create with synthetic codebooks for testing.
    pub fn new_synthetic(
        grid_size: usize, codebook_size: usize, residual_size: usize, embed_dim: usize,
    ) -> Self {
        let iframe_codebook: Vec<f32> = (0..codebook_size * embed_dim)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();
        let pframe_codebook: Vec<f32> = (0..residual_size * embed_dim)
            .map(|i| (i as f32 * 0.1).cos() * 0.2)
            .collect();
        Self::new(iframe_codebook, pframe_codebook, grid_size, embed_dim)
    }

    /// Reconstruct a frame from generated tokens.
    ///
    /// Returns [grid_size² × embed_dim] f32 values.
    pub fn reconstruct(&mut self, frame: &GeneratedVideoFrame) -> Vec<f32> {
        let gs2 = self.grid_size * self.grid_size;
        match frame.frame_type {
            VideoFrameType::IFrame => {
                let mut recon = Vec::with_capacity(gs2 * self.embed_dim);
                for &token in &frame.tokens {
                    let off = token * self.embed_dim;
                    for d in 0..self.embed_dim {
                        recon.push(self.iframe_codebook.get(off + d).copied().unwrap_or(0.0));
                    }
                }
                self.last_frame = Some(recon.clone());
                recon
            }
            VideoFrameType::PFrame => {
                let mut recon = self.last_frame.clone()
                    .unwrap_or_else(|| vec![0.0; gs2 * self.embed_dim]);
                for (i, &token) in frame.tokens.iter().enumerate() {
                    if i * self.embed_dim + self.embed_dim > recon.len() { break; }
                    let off = token * self.embed_dim;
                    for d in 0..self.embed_dim {
                        recon[i * self.embed_dim + d] +=
                            self.pframe_codebook.get(off + d).copied().unwrap_or(0.0);
                    }
                }
                self.last_frame = Some(recon.clone());
                recon
            }
        }
    }

    /// Reset state (new video).
    pub fn reset(&mut self) { self.last_frame = None; }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Argmax: index of the maximum value.
fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hidden(dim: usize, value: f32) -> Vec<f32> { vec![value; dim] }

    // -- VisionHead --

    #[test]
    fn test_vision_head_config() {
        let c = VisionHeadConfig::default_768();
        assert_eq!(c.hidden_dim, 768);
        assert_eq!(c.codebook_size, 8192);
        assert_eq!(c.grid_size, 32);
        assert_eq!(c.num_tokens(), 1024);
    }

    #[test]
    fn test_vision_head_forward_shape() {
        let head = VisionHead::new(VisionHeadConfig::new(64, 256, 4));
        let logits = head.forward(&make_hidden(64, 0.5));
        assert_eq!(logits.len(), 256);
        assert!(logits.iter().any(|&l| l != 0.0));
    }

    #[test]
    fn test_vision_head_grid() {
        let head = VisionHead::new(VisionHeadConfig::new(32, 64, 4));
        let hiddens: Vec<Vec<f32>> = (0..16).map(|i| make_hidden(32, i as f32 * 0.1)).collect();
        let logits = head.forward_grid(&hiddens);
        assert_eq!(logits.len(), 16);
        assert_eq!(logits[0].len(), 64);
    }

    #[test]
    fn test_vision_head_greedy() {
        let head = VisionHead::new(VisionHeadConfig::new(32, 64, 4));
        let hiddens: Vec<Vec<f32>> = (0..16).map(|_| make_hidden(32, 0.5)).collect();
        let tokens = head.decode_greedy(&hiddens);
        assert_eq!(tokens.len(), 16);
        assert!(tokens.iter().all(|&t| t < 64));
    }

    #[test]
    fn test_vision_head_progressive() {
        let head = VisionHead::new(VisionHeadConfig::new(32, 64, 4));
        let hiddens: Vec<Vec<f32>> = (0..16).map(|_| make_hidden(32, 0.5)).collect();
        let rows = head.decode_progressive(&hiddens);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].len(), 4);
    }

    #[test]
    fn test_vision_head_tokens_to_grid() {
        let head = VisionHead::new(VisionHeadConfig::new(32, 64, 3));
        let grid = head.tokens_to_grid(&(0..9).collect::<Vec<_>>());
        assert_eq!(grid.len(), 3);
        assert_eq!(grid[2], vec![6, 7, 8]);
    }

    #[test]
    fn test_vision_head_with_weights() {
        let cfg = VisionHeadConfig::new(4, 8, 2);
        let w = vec![1.0; 32];
        let b = vec![0.0; 8];
        let head = VisionHead::with_weights(cfg, w.clone(), b.clone());
        assert_eq!(head.weight().len(), 32);
        assert_eq!(head.bias().len(), 8);
    }

    // -- SpeechHead --

    #[test]
    fn test_speech_head_config() {
        let c = SpeechHeadConfig::default_encodec();
        assert_eq!(c.hidden_dim, 768);
        assert_eq!(c.num_codebooks, 8);
        assert_eq!(c.codebook_size, 1024);
    }

    #[test]
    fn test_speech_head_forward() {
        let head = SpeechHead::new(SpeechHeadConfig::new(64, 4, 128));
        let logits = head.forward(&make_hidden(64, 0.3));
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), 128);
    }

    #[test]
    fn test_speech_head_sequence() {
        let head = SpeechHead::new(SpeechHeadConfig::new(32, 3, 64));
        let hiddens: Vec<Vec<f32>> = (0..10).map(|i| make_hidden(32, i as f32 * 0.1)).collect();
        let seq = head.forward_sequence(&hiddens);
        assert_eq!(seq.len(), 10);
        assert_eq!(seq[0].len(), 3);
        assert_eq!(seq[0][0].len(), 64);
    }

    #[test]
    fn test_speech_head_greedy() {
        let head = SpeechHead::new(SpeechHeadConfig::new(32, 3, 64));
        let hiddens: Vec<Vec<f32>> = (0..5).map(|_| make_hidden(32, 0.5)).collect();
        let tokens = head.decode_greedy(&hiddens);
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].len(), 3);
        assert!(tokens.iter().all(|f| f.iter().all(|&t| t < 64)));
    }

    #[test]
    fn test_speech_head_accessors() {
        let head = SpeechHead::new(SpeechHeadConfig::new(32, 6, 128));
        assert_eq!(head.num_codebooks(), 6);
        assert_eq!(head.codebook_size(), 128);
    }

    // -- VideoHead --

    #[test]
    fn test_video_head_config() {
        let c = VideoHeadConfig::default_30fps();
        assert_eq!(c.fps, 30.0);
        assert_eq!(c.iframe_interval, 30);
    }

    #[test]
    fn test_video_head_iframe() {
        let mut head = VideoHead::new(VideoHeadConfig::new(32, 64, 32, 4, 30.0, 10));
        let hiddens: Vec<Vec<f32>> = (0..16).map(|_| make_hidden(32, 0.5)).collect();
        let frame = head.generate_frame(&hiddens, 0);
        assert_eq!(frame.frame_type, VideoFrameType::IFrame);
        assert_eq!(frame.frame_idx, 0);
        assert_eq!(frame.tokens.len(), 16);
        assert!((frame.timestamp).abs() < 1e-5);
    }

    #[test]
    fn test_video_head_pframe() {
        let mut head = VideoHead::new(VideoHeadConfig::new(32, 64, 32, 4, 30.0, 10));
        let h: Vec<Vec<f32>> = (0..16).map(|_| make_hidden(32, 0.5)).collect();
        head.generate_frame(&h, 0);
        let pf_hiddens: Vec<Vec<f32>> = (0..8).map(|_| make_hidden(32, 0.2)).collect();
        let frame = head.generate_frame(&pf_hiddens, 1);
        assert_eq!(frame.frame_type, VideoFrameType::PFrame);
        assert_eq!(frame.tokens.len(), 8);
    }

    #[test]
    fn test_video_head_iframe_interval() {
        let mut head = VideoHead::new(VideoHeadConfig::new(32, 64, 32, 4, 30.0, 5));
        let h: Vec<Vec<f32>> = (0..16).map(|_| make_hidden(32, 0.5)).collect();
        assert_eq!(head.generate_frame(&h, 0).frame_type, VideoFrameType::IFrame);
        assert_eq!(head.generate_frame(&h, 4).frame_type, VideoFrameType::PFrame);
        assert_eq!(head.generate_frame(&h, 5).frame_type, VideoFrameType::IFrame);
        assert_eq!(head.generate_frame(&h, 10).frame_type, VideoFrameType::IFrame);
    }

    // -- VideoStreamReconstructor --

    #[test]
    fn test_reconstructor_iframe() {
        let mut recon = VideoStreamReconstructor::new_synthetic(4, 64, 32, 8);
        let frame = GeneratedVideoFrame {
            tokens: (0..16).collect(),
            frame_type: VideoFrameType::IFrame,
            frame_idx: 0, timestamp: 0.0,
        };
        let pixels = recon.reconstruct(&frame);
        assert_eq!(pixels.len(), 16 * 8);
    }

    #[test]
    fn test_reconstructor_pframe_adds_residual() {
        let mut recon = VideoStreamReconstructor::new_synthetic(4, 64, 32, 8);
        let iframe = GeneratedVideoFrame {
            tokens: (0..16).collect(),
            frame_type: VideoFrameType::IFrame, frame_idx: 0, timestamp: 0.0,
        };
        let i_pixels = recon.reconstruct(&iframe);
        let pframe = GeneratedVideoFrame {
            tokens: vec![0; 16],
            frame_type: VideoFrameType::PFrame, frame_idx: 1, timestamp: 1.0 / 30.0,
        };
        let p_pixels = recon.reconstruct(&pframe);
        // P-frame should differ (residual added)
        assert_ne!(i_pixels, p_pixels);
    }

    #[test]
    fn test_reconstructor_pframe_no_prior() {
        let mut recon = VideoStreamReconstructor::new_synthetic(4, 64, 32, 8);
        // P-frame without prior I-frame — should not panic
        let pframe = GeneratedVideoFrame {
            tokens: vec![0; 16],
            frame_type: VideoFrameType::PFrame, frame_idx: 0, timestamp: 0.0,
        };
        let pixels = recon.reconstruct(&pframe);
        assert_eq!(pixels.len(), 16 * 8);
    }

    #[test]
    fn test_reconstructor_reset() {
        let mut recon = VideoStreamReconstructor::new_synthetic(4, 64, 32, 8);
        let f = GeneratedVideoFrame {
            tokens: vec![0; 16], frame_type: VideoFrameType::IFrame, frame_idx: 0, timestamp: 0.0,
        };
        recon.reconstruct(&f);
        recon.reset();
        assert!(recon.last_frame.is_none());
    }
}
