//! Cross-modal attention — unified text/vision/audio fusion.
//!
//! Implements cross-attention layers where Q comes from one modality (e.g., text)
//! and K/V come from another (e.g., vision or audio). Supports three fusion
//! strategies:
//!
//! - **Early fusion**: concatenate modality embeddings before the transformer
//! - **Mid fusion**: cross-attention at specific layers (this module's core)
//! - **Late fusion**: merge after separate encoder outputs
//!
//! Also includes modality token type embeddings so the model can distinguish
//! text, vision, and audio tokens.

use std::sync::Arc;
use wgpu::{Device, Queue};

use crate::compute::buffer::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::compute::kernels::flash_decode::FlashDecodeOp;
use crate::compute::kernels::rmsnorm::RmsNormOp;
use crate::compute::kernels::prefill_attn::PrefillAttnOp;
use crate::error::Result;
use crate::model::linear::Linear;

// ---------------------------------------------------------------------------
// Modality types
// ---------------------------------------------------------------------------

/// Token modality types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Modality {
    Text = 0,
    Vision = 1,
    Audio = 2,
}

impl Modality {
    /// Number of modality types.
    pub fn count() -> usize {
        3
    }

    /// Embedding ID for this modality.
    pub fn id(&self) -> u8 {
        *self as u8
    }

    /// From integer ID.
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Text),
            1 => Some(Self::Vision),
            2 => Some(Self::Audio),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ModalityTypeEmbedding — learnable type embeddings per modality
// ---------------------------------------------------------------------------

/// Learnable type embeddings that are added to token embeddings based on
/// the token's modality (text, vision, audio). Similar to BERT segment
/// embeddings but for modalities.
pub struct ModalityTypeEmbedding {
    /// One embedding vector per modality: [Modality::count() × hidden_dim].
    embeddings: GpuBuffer,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Elementwise op for addition.
    #[allow(dead_code)]
    elementwise: ElementWiseOp,
    #[allow(dead_code)]
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
}

impl ModalityTypeEmbedding {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, hidden_dim: usize) -> Result<Self> {
        let n_modalities = Modality::count();
        let byte_size = n_modalities * hidden_dim * std::mem::size_of::<f32>();
        let embeddings = GpuBuffer::zeros(&device, &queue, byte_size, Some("modality_type_embed"))?;
        let elementwise = ElementWiseOp::new(&device, &queue);

        Ok(Self {
            embeddings,
            hidden_dim,
            elementwise,
            device,
            queue,
        })
    }

    /// Get the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Access the embeddings buffer.
    pub fn embeddings_buffer(&self) -> &GpuBuffer {
        &self.embeddings
    }
}

// ---------------------------------------------------------------------------
// FusionStrategy — how to combine modalities
// ---------------------------------------------------------------------------

/// Strategy for fusing multiple modalities.
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Early fusion: concatenate embeddings before the transformer.
    /// All modalities are projected to the same hidden_dim and concatenated.
    Early {
        /// Maximum number of vision tokens.
        max_vision_tokens: usize,
        /// Maximum number of audio tokens.
        max_audio_tokens: usize,
    },
    /// Mid fusion: cross-attention at specific layers.
    /// Text queries attend to vision/audio K/V at designated layers.
    Mid {
        /// Layer indices where cross-attention is applied.
        cross_attn_layers: Vec<usize>,
    },
    /// Late fusion: average/weighted merge of separate encoder outputs.
    Late {
        /// Weight for text modality.
        text_weight: f32,
        /// Weight for vision modality.
        vision_weight: f32,
        /// Weight for audio modality.
        audio_weight: f32,
    },
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::Mid {
            cross_attn_layers: vec![8, 16, 24],
        }
    }
}

impl FusionStrategy {
    /// Early fusion with default sizes.
    pub fn early() -> Self {
        Self::Early {
            max_vision_tokens: 256,
            max_audio_tokens: 128,
        }
    }

    /// Mid fusion at specific layers.
    pub fn mid(layers: Vec<usize>) -> Self {
        Self::Mid { cross_attn_layers: layers }
    }

    /// Late fusion with equal weights.
    pub fn late_equal() -> Self {
        Self::Late {
            text_weight: 1.0,
            vision_weight: 1.0,
            audio_weight: 1.0,
        }
    }

    /// Check if this is early fusion.
    pub fn is_early(&self) -> bool {
        matches!(self, Self::Early { .. })
    }

    /// Check if this is mid fusion.
    pub fn is_mid(&self) -> bool {
        matches!(self, Self::Mid { .. })
    }

    /// Check if this is late fusion.
    pub fn is_late(&self) -> bool {
        matches!(self, Self::Late { .. })
    }
}

// ---------------------------------------------------------------------------
// CrossModalAttentionLayer — Q from one modality, K/V from another
// ---------------------------------------------------------------------------

/// Cross-modal attention layer: Q from text, K/V from another modality.
///
/// This is the core of mid-fusion: at specific transformer layers, text
/// queries attend to vision (or audio) keys and values, allowing the model
/// to ground its text representations in visual/auditory information.
///
/// Architecture:
///   text_hidden → RMSNorm → Q_proj → text_queries
///   cross_hidden → RMSNorm → K_proj, V_proj → cross_keys, cross_values
///   output = softmax(Q × K^T / sqrt(d)) × V
///   output → out_proj → residual add
pub struct CrossModalAttentionLayer {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,

    // Projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,

    // Norms
    q_norm: RmsNormOp,
    kv_norm: RmsNormOp,

    // Attention ops
    prefill_attn: PrefillAttnOp,
    flash_decode: FlashDecodeOp,
    elementwise: ElementWiseOp,

    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl CrossModalAttentionLayer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        hidden_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let head_dim = hidden_dim / num_heads;

        let q_proj = Linear::new(Arc::clone(&device), Arc::clone(&queue), hidden_dim, hidden_dim, false)?;
        let k_proj = Linear::new(Arc::clone(&device), Arc::clone(&queue), hidden_dim, hidden_dim, false)?;
        let v_proj = Linear::new(Arc::clone(&device), Arc::clone(&queue), hidden_dim, hidden_dim, false)?;
        let out_proj = Linear::new(Arc::clone(&device), Arc::clone(&queue), hidden_dim, hidden_dim, false)?;

        let q_norm = RmsNormOp::new(&device)?;
        let kv_norm = RmsNormOp::new(&device)?;
        let prefill_attn = PrefillAttnOp::new(&device)?;
        let flash_decode = FlashDecodeOp::new(&device, &queue)?;
        let elementwise = ElementWiseOp::new(&device, &queue);

        Ok(Self {
            hidden_dim,
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_norm,
            kv_norm,
            prefill_attn,
            flash_decode,
            elementwise,
            device,
            queue,
        })
    }

    /// Cross-attention prefill: text Q attends to cross-modal K/V.
    ///
    /// - `text_hidden`: [seq_len × hidden_dim] text hidden states
    /// - `cross_hidden`: [cross_len × hidden_dim] vision/audio hidden states
    /// - Returns: [seq_len × hidden_dim] output
    pub fn forward_prefill(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        text_hidden: &GpuBuffer,
        cross_hidden: &GpuBuffer,
        seq_len: u32,
        cross_len: u32,
    ) -> Result<GpuBuffer> {
        let hidden_dim = self.hidden_dim;
        let num_heads = self.num_heads as u32;
        let head_dim = self.head_dim as u32;
        let f32_size = std::mem::size_of::<f32>();

        // Norm text and cross
        let normed_q = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("xattn_normed_q"))?;
        self.q_norm.dispatch(&self.device, &self.queue, encoder, text_hidden, &normed_q, seq_len, hidden_dim as u32)?;

        let normed_kv = GpuBuffer::new(&self.device, cross_len as usize * hidden_dim * f32_size, Some("xattn_normed_kv"))?;
        self.kv_norm.dispatch(&self.device, &self.queue, encoder, cross_hidden, &normed_kv, cross_len, hidden_dim as u32)?;

        // Project Q, K, V
        let q = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("xattn_q"))?;
        let k = GpuBuffer::new(&self.device, cross_len as usize * hidden_dim * f32_size, Some("xattn_k"))?;
        let v = GpuBuffer::new(&self.device, cross_len as usize * hidden_dim * f32_size, Some("xattn_v"))?;

        self.q_proj.forward(encoder, &normed_q, &q, seq_len)?;
        self.k_proj.forward(encoder, &normed_kv, &k, cross_len)?;
        self.v_proj.forward(encoder, &normed_kv, &v, cross_len)?;

        // Cross-attention: Q(text) × K(cross)^T × V(cross)
        // No causal mask needed for cross-attention (all cross tokens visible)
        let attn_out = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("xattn_out"))?;
        self.prefill_attn.dispatch(encoder, &q, &k, &v, &attn_out, seq_len, num_heads, head_dim)?;

        // Output projection + residual
        let proj_out = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("xattn_proj"))?;
        self.out_proj.forward(encoder, &attn_out, &proj_out, seq_len)?;

        let residual = GpuBuffer::new(&self.device, seq_len as usize * hidden_dim * f32_size, Some("xattn_residual"))?;
        self.elementwise.dispatch_add(encoder, text_hidden, &proj_out, &residual, seq_len * hidden_dim as u32)?;

        Ok(residual)
    }

    /// Cross-attention decode: single text token attends to cross-modal K/V.
    pub fn forward_decode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        text_hidden: &GpuBuffer,
        cross_k: &GpuBuffer,
        cross_v: &GpuBuffer,
    ) -> Result<GpuBuffer> {
        let hidden_dim = self.hidden_dim;
        let num_heads = self.num_heads as u32;
        let head_dim = self.head_dim as u32;
        let f32_size = std::mem::size_of::<f32>();

        // Norm
        let normed_q = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("xattn_decode_normed"))?;
        self.q_norm.dispatch(&self.device, &self.queue, encoder, text_hidden, &normed_q, 1u32, hidden_dim as u32)?;

        // Project Q only (K/V are precomputed from cross-modal encoder)
        let q = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("xattn_decode_q"))?;
        self.q_proj.forward(encoder, &normed_q, &q, 1u32)?;

        // Flash decode: single query against cross K/V
        let cross_len = cross_k.size() as u32 / (hidden_dim as u32 * f32_size as u32);
        let attn_out = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("xattn_decode_out"))?;
        self.flash_decode.dispatch(
            encoder, &q, cross_k, cross_v, &attn_out,
            cross_len, num_heads, head_dim,
        )?;

        // Output projection + residual
        let proj_out = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("xattn_decode_proj"))?;
        self.out_proj.forward(encoder, &attn_out, &proj_out, 1u32)?;

        let residual = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("xattn_decode_residual"))?;
        self.elementwise.dispatch_add(encoder, text_hidden, &proj_out, &residual, hidden_dim as u32)?;

        Ok(residual)
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Number of heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

// ---------------------------------------------------------------------------
// MultimodalFusion — top-level fusion orchestrator
// ---------------------------------------------------------------------------

/// Orchestrates multimodal fusion using the configured strategy.
pub struct MultimodalFusion {
    strategy: FusionStrategy,
    /// Cross-attention layers for mid-fusion (one per specified layer).
    cross_attn_layers: Vec<CrossModalAttentionLayer>,
    /// Modality type embeddings.
    type_embeddings: ModalityTypeEmbedding,
    /// Hidden dimension.
    hidden_dim: usize,
    #[allow(dead_code)]
    num_heads: usize,
}

impl MultimodalFusion {
    /// Create a new multimodal fusion module.
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        hidden_dim: usize,
        num_heads: usize,
        strategy: FusionStrategy,
    ) -> Result<Self> {
        let type_embeddings = ModalityTypeEmbedding::new(
            Arc::clone(&device), Arc::clone(&queue), hidden_dim,
        )?;

        let cross_attn_layers = match &strategy {
            FusionStrategy::Mid { cross_attn_layers } => {
                cross_attn_layers.iter().map(|_| {
                    CrossModalAttentionLayer::new(
                        Arc::clone(&device), Arc::clone(&queue),
                        hidden_dim, num_heads,
                    )
                }).collect::<Result<Vec<_>>>()?
            }
            _ => Vec::new(),
        };

        Ok(Self {
            strategy,
            cross_attn_layers,
            type_embeddings,
            hidden_dim,
            num_heads,
        })
    }

    /// Get the fusion strategy.
    pub fn strategy(&self) -> &FusionStrategy {
        &self.strategy
    }

    /// Get the modality type embeddings.
    pub fn type_embeddings(&self) -> &ModalityTypeEmbedding {
        &self.type_embeddings
    }

    /// Get cross-attention layer count (for mid-fusion).
    pub fn num_cross_attn_layers(&self) -> usize {
        self.cross_attn_layers.len()
    }

    /// Access a cross-attention layer by index.
    pub fn cross_attn_layer(&self, idx: usize) -> Option<&CrossModalAttentionLayer> {
        self.cross_attn_layers.get(idx)
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Check if a transformer layer should apply cross-attention.
    pub fn should_cross_attend(&self, layer_idx: usize) -> bool {
        match &self.strategy {
            FusionStrategy::Mid { cross_attn_layers } => {
                cross_attn_layers.contains(&layer_idx)
            }
            _ => false,
        }
    }

    /// Get the cross-attention layer index for a given transformer layer.
    /// Returns None if this layer doesn't have cross-attention.
    pub fn cross_attn_index_for_layer(&self, layer_idx: usize) -> Option<usize> {
        match &self.strategy {
            FusionStrategy::Mid { cross_attn_layers } => {
                cross_attn_layers.iter().position(|&l| l == layer_idx)
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_ids() {
        assert_eq!(Modality::Text.id(), 0);
        assert_eq!(Modality::Vision.id(), 1);
        assert_eq!(Modality::Audio.id(), 2);
        assert_eq!(Modality::count(), 3);
        assert_eq!(Modality::from_id(0), Some(Modality::Text));
        assert_eq!(Modality::from_id(1), Some(Modality::Vision));
        assert_eq!(Modality::from_id(2), Some(Modality::Audio));
        assert_eq!(Modality::from_id(3), None);
    }

    #[test]
    fn test_fusion_strategy_early() {
        let s = FusionStrategy::early();
        assert!(s.is_early());
        assert!(!s.is_mid());
        assert!(!s.is_late());
    }

    #[test]
    fn test_fusion_strategy_mid() {
        let s = FusionStrategy::mid(vec![4, 8, 12]);
        assert!(!s.is_early());
        assert!(s.is_mid());
        assert!(!s.is_late());
    }

    #[test]
    fn test_fusion_strategy_late() {
        let s = FusionStrategy::late_equal();
        assert!(!s.is_early());
        assert!(!s.is_mid());
        assert!(s.is_late());
    }

    #[test]
    fn test_fusion_strategy_default() {
        let s = FusionStrategy::default();
        assert!(s.is_mid());
    }

    #[test]
    fn test_multimodal_fusion_cross_attend() {
        let strategy = FusionStrategy::mid(vec![2, 5, 8]);

        // Verify the strategy logic without GPU
        assert!(matches!(&strategy, FusionStrategy::Mid { cross_attn_layers } if cross_attn_layers == &vec![2, 5, 8]));
    }

    #[test]
    fn test_multimodal_fusion_early_strategy() {
        let s = FusionStrategy::early();
        match &s {
            FusionStrategy::Early { max_vision_tokens, max_audio_tokens } => {
                assert_eq!(*max_vision_tokens, 256);
                assert_eq!(*max_audio_tokens, 128);
            }
            _ => panic!("Expected Early"),
        }
    }

    #[test]
    fn test_late_fusion_weights() {
        let s = FusionStrategy::late_equal();
        match &s {
            FusionStrategy::Late { text_weight, vision_weight, audio_weight } => {
                assert_eq!(*text_weight, 1.0);
                assert_eq!(*vision_weight, 1.0);
                assert_eq!(*audio_weight, 1.0);
            }
            _ => panic!("Expected Late"),
        }
    }

    #[test]
    fn test_should_cross_attend_logic() {
        let s = FusionStrategy::mid(vec![2, 5, 8]);
        // Verify logic directly
        if let FusionStrategy::Mid { cross_attn_layers } = &s {
            assert!(cross_attn_layers.contains(&2));
            assert!(cross_attn_layers.contains(&5));
            assert!(cross_attn_layers.contains(&8));
            assert!(!cross_attn_layers.contains(&0));
            assert!(!cross_attn_layers.contains(&3));

            assert_eq!(cross_attn_layers.iter().position(|&l| l == 2), Some(0));
            assert_eq!(cross_attn_layers.iter().position(|&l| l == 5), Some(1));
            assert_eq!(cross_attn_layers.iter().position(|&l| l == 8), Some(2));
            assert_eq!(cross_attn_layers.iter().position(|&l| l == 3), None);
        }
    }

    #[test]
    fn test_early_strategy_no_cross_attn() {
        let s = FusionStrategy::early();
        // Early fusion has no cross-attention layers
        assert!(!matches!(&s, FusionStrategy::Mid { .. }));
    }
}
