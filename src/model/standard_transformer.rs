//! Standard transformer layer — O(n²) full self-attention compatibility mode.
//!
//! This module implements a standard pre-norm transformer layer with explicit
//! gated FFN activation (SiLU or GELU-tanh) within FerrisRes's GPU runtime.
//! This does not imply complete pretrained architecture compatibility. It reuses all existing WGSL kernels and gains access to FerrisRes
//! optimizations (TurboQuant, YaRN, ToMe, iGPU support).
//!
//! Architecture: Pre-norm → Q/K/V → RoPE → Full self-attention → residual → FFN
//!
//! Compare with BlockAttnResLayer which partitions attention into blocks for O(n).

use std::sync::Arc;
use wgpu::{Device, Queue};

use crate::compute::buffer::GpuBuffer;
use crate::compute::kernels::rope::RopeOp;
use crate::compute::kernels::flash_decode::FlashDecodeOp;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::compute::kernels::prefill_attn::PrefillAttnOp;
use crate::error::Result;
use crate::inference::kv_cache::LayerKVCache;
use crate::model::linear::Linear;
use crate::compute::kernels::gpu_transformer::GpuTransformerPipeline;

/// Activation applied to the gate before multiplication with the up projection.
/// Select from checkpoint architecture metadata, never from tensor shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GatedActivation {
    /// SwiGLU used by LLaMA and Mistral.
    #[default]
    Silu,
    /// GELU's tanh approximation, not exact-erf GELU.
    GeluTanh,
}

/// Configuration for a standard transformer model.
#[derive(Debug, Clone)]
pub struct StandardTransformerConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_dim: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub use_bias: bool,
    pub ffn_activation: GatedActivation,
    /// Positive normal finite RMSNorm epsilon; import must set its metadata value.
    pub norm_eps: f32,
}

impl StandardTransformerConfig {
    pub fn new(hidden_dim: usize, num_heads: usize, num_layers: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        let intermediate_dim = hidden_dim * 4; // standard 4× expansion
        Self {
            hidden_dim,
            num_heads,
            num_layers,
            intermediate_dim,
            head_dim,
            norm_eps: 1e-5,
            vocab_size: 32000,
            use_bias: false,
            ffn_activation: GatedActivation::Silu,
        }
    }

    /// Create config matching LLaMA-7B architecture.
    pub fn llama_7b() -> Self {
        Self {
            hidden_dim: 4096,
            num_heads: 32,
            num_layers: 32,
            intermediate_dim: 11008,
            norm_eps: 1e-5,
            head_dim: 128,
            vocab_size: 32000,
            use_bias: false,
            ffn_activation: GatedActivation::Silu,
        }
    }

    /// Create config matching Mistral-7B architecture.
    pub fn mistral_7b() -> Self {
        Self {
            hidden_dim: 4096,
            num_heads: 32,
            num_layers: 32,
            intermediate_dim: 14336,
            norm_eps: 1e-5,
            head_dim: 128,
            vocab_size: 32000,
            use_bias: false,
            ffn_activation: GatedActivation::Silu,
        }
    }

    /// Create config from loaded weight metadata.
    pub fn from_inferred(hidden_dim: usize, num_heads: usize, num_layers: usize, vocab_size: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        Self {
            hidden_dim,
            num_heads,
            num_layers,
            intermediate_dim: hidden_dim * 4,
            head_dim,
            norm_eps: 1e-5,
            vocab_size,
            use_bias: false,
            ffn_activation: GatedActivation::Silu,
        }
    }
}

/// A single standard transformer layer with full O(n²) self-attention.
///
/// This is the compatibility-mode counterpart to [`BlockAttnResLayer`].
/// It implements the same forward interface (forward_prefill,
/// forward_decode_token, forward_decode_token_direct) so it can be used
/// as a drop-in replacement in the TokenGenerator pipeline.
///
/// Structure:
///   input → RMSNorm → Q/K/V projection → RoPE → full attention →
///   out_proj → residual_add → RMSNorm → down(activation(gate) * up) → residual_add → output
pub struct StandardTransformerLayer {
    // Dimensions
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,

    // Attention
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    attn_norm_weight: GpuBuffer,
    rope: RopeOp,

    // Decode
    flash_decode: FlashDecodeOp,

    // Prefill
    prefill_attn: PrefillAttnOp,

    // FFN
    ff_gate: Linear,
    ff_up: Linear,
    ff_down: Linear,
    ff_norm_weight: GpuBuffer,
    norm_eps: f32,
    ffn_activation: GatedActivation,
    ffn_ops: GpuTransformerPipeline,

    // Elementwise
    elementwise: ElementWiseOp,

    // Device refs
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl StandardTransformerLayer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: &StandardTransformerConfig,
    ) -> Result<Self> {
        if !config.norm_eps.is_finite() || config.norm_eps < f32::MIN_POSITIVE {
            return Err(crate::error::FerrisResError::Shape("norm_eps must be positive, normal and finite".into()));
        }
        let hidden_dim = config.hidden_dim;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let intermediate_dim = config.intermediate_dim;

        let q_proj = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            hidden_dim, hidden_dim, config.use_bias,
        )?;
        let k_proj = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            hidden_dim, hidden_dim, config.use_bias,
        )?;
        let v_proj = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            hidden_dim, hidden_dim, config.use_bias,
        )?;
        let out_proj = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            hidden_dim, hidden_dim, config.use_bias,
        )?;

        let ff_gate = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            hidden_dim, intermediate_dim, config.use_bias,
        )?;
        let ff_up = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            hidden_dim, intermediate_dim, config.use_bias,
        )?;
        let ff_down = Linear::new(
            Arc::clone(&device), Arc::clone(&queue),
            intermediate_dim, hidden_dim, config.use_bias,
        )?;

        let attn_norm_weight = GpuBuffer::new(&device, hidden_dim * 4, Some("std_attn_norm_weight"))?;
        let ff_norm_weight = GpuBuffer::new(&device, hidden_dim * 4, Some("std_ff_norm_weight"))?;
        let unit_scale = vec![1.0f32; hidden_dim];
        queue.write_buffer(attn_norm_weight.buffer(), 0, bytemuck::cast_slice(&unit_scale));
        queue.write_buffer(ff_norm_weight.buffer(), 0, bytemuck::cast_slice(&unit_scale));
        let elementwise = ElementWiseOp::new(&device, &queue);
        let rope = RopeOp::new(&device)?;
        let flash_decode = FlashDecodeOp::new(&device, &queue)?;
        let prefill_attn = PrefillAttnOp::new(&device)?;

        Ok(Self {
            hidden_dim,
            num_heads,
            head_dim,
            intermediate_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            attn_norm_weight,
            rope,
            flash_decode,
            prefill_attn,
            ff_gate,
            ff_up,
            ff_down,
            ff_norm_weight,
            norm_eps: config.norm_eps,
            ffn_activation: config.ffn_activation,
            ffn_ops: GpuTransformerPipeline::new(&device)?,
            elementwise,
            device,
            queue,
        })
    }

    /// Prefill: process a batch of tokens with full O(n²) self-attention.
    ///
    /// Uses the existing prefill_attn kernel for batched multi-head attention
    /// with causal masking.
    pub fn forward_prefill(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuBuffer,
        kv_cache: &LayerKVCache,
        seq_len: u32,
    ) -> Result<GpuBuffer> {
        let hidden_dim = self.hidden_dim;
        let num_heads = self.num_heads as u32;
        let head_dim = self.head_dim as u32;
        let f32_size = std::mem::size_of::<f32>();
        let numel = seq_len * hidden_dim as u32;

        // RMSNorm on input
        let normed = GpuBuffer::new(
            &self.device,
            seq_len as usize * hidden_dim * f32_size,
            Some("std_prefill_normed"),
        )?;
        self.ffn_ops.dispatch_rmsnorm_with_epsilon(
            &self.device, &self.queue, encoder, hidden_states, &normed,
            &self.attn_norm_weight, seq_len, hidden_dim as u32, self.norm_eps,
        )?;

        // Q/K/V projections
        let q_buf = GpuBuffer::new(&self.device, numel as usize * f32_size, Some("std_prefill_q"))?;
        let k_buf = GpuBuffer::new(&self.device, numel as usize * f32_size, Some("std_prefill_k"))?;
        let v_buf = GpuBuffer::new(&self.device, numel as usize * f32_size, Some("std_prefill_v"))?;
        self.q_proj.forward(encoder, &normed, &q_buf, seq_len)?;
        self.k_proj.forward(encoder, &normed, &k_buf, seq_len)?;
        self.v_proj.forward(encoder, &normed, &v_buf, seq_len)?;

        // RoPE on Q and K (start_pos=0 for prefill)
        let rope_q = GpuBuffer::new(&self.device, numel as usize * f32_size, Some("std_prefill_rope_q"))?;
        let rope_k = GpuBuffer::new(&self.device, numel as usize * f32_size, Some("std_prefill_rope_k"))?;
        self.rope.dispatch_with_offset(encoder, &q_buf, &rope_q, seq_len, num_heads, head_dim, 0)?;
        self.rope.dispatch_with_offset(encoder, &k_buf, &rope_k, seq_len, num_heads, head_dim, 0)?;

        // Update KV cache with all prefill K/V
        let _ = kv_cache.update_batch(encoder, &rope_k, &v_buf, seq_len)?;

        // Full O(n²) self-attention via prefill kernel
        // prefill_attn computes: output = softmax(Q × K^T / sqrt(d)) × V
        // with causal masking (k_pos <= q_pos) built into the kernel.
        let attn_out = GpuBuffer::new(
            &self.device,
            seq_len as usize * hidden_dim * f32_size,
            Some("std_prefill_attn"),
        )?;
        // Read K/V from the KV cache (which now contains all prefill positions)
        self.prefill_attn.dispatch(
            encoder,
            &rope_q,
            kv_cache.key_buffer(),
            kv_cache.value_buffer(),
            &attn_out,
            seq_len,
            num_heads,
            head_dim,
        )?;

        // Output projection
        let proj_out = GpuBuffer::new(
            &self.device,
            seq_len as usize * hidden_dim * f32_size,
            Some("std_prefill_proj"),
        )?;
        self.out_proj.forward(encoder, &attn_out, &proj_out, seq_len)?;

        // Residual add
        let residual1 = GpuBuffer::new(
            &self.device,
            seq_len as usize * hidden_dim * f32_size,
            Some("std_prefill_res1"),
        )?;
        self.elementwise.dispatch_add(encoder, hidden_states, &proj_out, &residual1, numel)?;

        // FFN
        let ff_normed = GpuBuffer::new(
            &self.device,
            seq_len as usize * hidden_dim * f32_size,
            Some("std_prefill_ff_norm"),
        )?;
        self.ffn_ops.dispatch_rmsnorm_with_epsilon(
            &self.device, &self.queue, encoder, &residual1, &ff_normed,
            &self.ff_norm_weight, seq_len, hidden_dim as u32, self.norm_eps,
        )?;

        let ff_gate = GpuBuffer::new(
            &self.device,
            seq_len as usize * self.intermediate_dim * f32_size,
            Some("std_prefill_gate"),
        )?;
        let ff_up = GpuBuffer::new(
            &self.device,
            seq_len as usize * self.intermediate_dim * f32_size,
            Some("std_prefill_up"),
        )?;
        self.ff_gate.forward(encoder, &ff_normed, &ff_gate, seq_len)?;
        self.ff_up.forward(encoder, &ff_normed, &ff_up, seq_len)?;

        let ff_gated = self.activate_gate(encoder, &ff_gate, &ff_up, seq_len * self.intermediate_dim as u32)?;

        let ff_down = GpuBuffer::new(
            &self.device,
            seq_len as usize * hidden_dim * f32_size,
            Some("std_prefill_down"),
        )?;
        self.ff_down.forward(encoder, &ff_gated, &ff_down, seq_len)?;

        // Final residual
        let output = GpuBuffer::new(
            &self.device,
            seq_len as usize * hidden_dim * f32_size,
            Some("std_prefill_output"),
        )?;
        self.elementwise.dispatch_add(encoder, &residual1, &ff_down, &output, numel)?;

        Ok(output)
    }

    /// Decode a single token (legacy, no position override).
    pub fn forward_decode_token(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuBuffer,
        kv_cache: &LayerKVCache,
    ) -> Result<GpuBuffer> {
        self.forward_decode_token_with_pos(encoder, hidden_states, kv_cache, None)
    }

    /// Decode a single token with optional position override for YaRN/StreamingLLM.
    pub fn forward_decode_token_with_pos(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuBuffer,
        kv_cache: &LayerKVCache,
        effective_pos: Option<u32>,
    ) -> Result<GpuBuffer> {
        self.forward_decode_token_direct(encoder, hidden_states, kv_cache, effective_pos)
    }

    /// Optimized decode: direct K write + in-place RoPE.
    pub fn forward_decode_token_direct(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuBuffer,
        kv_cache: &LayerKVCache,
        effective_pos: Option<u32>,
    ) -> Result<GpuBuffer> {
        let hidden_dim = self.hidden_dim;
        let num_heads = self.num_heads as u32;
        let head_dim = self.head_dim as u32;
        let intermediate_dim = self.intermediate_dim as u32;
        let f32_size = std::mem::size_of::<f32>();

        // RMSNorm
        let normed = GpuBuffer::new(
            &self.device,
            hidden_dim * f32_size,
            Some("std_decode_normed"),
        )?;
        self.ffn_ops.dispatch_rmsnorm_with_epsilon(
            &self.device, &self.queue, encoder, hidden_states, &normed,
            &self.attn_norm_weight, 1, hidden_dim as u32, self.norm_eps,
        )?;

        // Q projection → temp buffer
        let q_buf = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_q"))?;
        self.q_proj.forward(encoder, &normed, &q_buf, 1u32)?;

        // K projection → temp buffer
        let k_buf = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_k"))?;
        self.k_proj.forward(encoder, &normed, &k_buf, 1u32)?;

        // V projection → temp buffer
        let v_buf = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_v"))?;
        self.v_proj.forward(encoder, &normed, &v_buf, 1u32)?;

        // RoPE on Q and K
        let pos = effective_pos.unwrap_or_else(|| kv_cache.current_len());
        let rope_q = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_rope_q"))?;
        let rope_k = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_rope_k"))?;
        self.rope.dispatch_with_offset(encoder, &q_buf, &rope_q, 1u32, num_heads, head_dim, pos)?;
        self.rope.dispatch_with_offset(encoder, &k_buf, &rope_k, 1u32, num_heads, head_dim, pos)?;

        // Update KV cache (K+V)
        let _new_len = kv_cache.update(encoder, &rope_k, &v_buf)?;

        // Flash decode attention (single-query)
        let attn_out = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_attn"))?;
        self.flash_decode.dispatch(
            encoder,
            &rope_q,
            kv_cache.key_buffer(),
            kv_cache.value_buffer(),
            &attn_out,
            kv_cache.current_len(),
            num_heads,
            head_dim,
        )?;

        // Output projection + residual
        let proj_out = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_proj"))?;
        self.out_proj.forward(encoder, &attn_out, &proj_out, 1u32)?;

        let residual1 = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_res1"))?;
        self.elementwise.dispatch_add(encoder, hidden_states, &proj_out, &residual1, hidden_dim as u32)?;

        // FFN
        let ff_normed = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_ff_norm"))?;
        self.ffn_ops.dispatch_rmsnorm_with_epsilon(
            &self.device, &self.queue, encoder, &residual1, &ff_normed,
            &self.ff_norm_weight, 1, hidden_dim as u32, self.norm_eps,
        )?;

        let ff_gate = GpuBuffer::new(
            &self.device,
            intermediate_dim as usize * f32_size,
            Some("std_decode_gate"),
        )?;
        let ff_up = GpuBuffer::new(
            &self.device,
            intermediate_dim as usize * f32_size,
            Some("std_decode_up"),
        )?;
        self.ff_gate.forward(encoder, &ff_normed, &ff_gate, 1u32)?;
        self.ff_up.forward(encoder, &ff_normed, &ff_up, 1u32)?;

        let ff_gated = self.activate_gate(encoder, &ff_gate, &ff_up, intermediate_dim)?;

        let ff_down = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_down"))?;
        self.ff_down.forward(encoder, &ff_gated, &ff_down, 1u32)?;

        let output = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("std_decode_output"))?;
        self.elementwise.dispatch_add(encoder, &residual1, &ff_down, &output, hidden_dim as u32)?;

        Ok(output)
    }

    /// Both forward paths share the same non-aliasing gated activation.
    fn activate_gate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate: &GpuBuffer,
        up: &GpuBuffer,
        numel: u32,
    ) -> Result<GpuBuffer> {
        let output = GpuBuffer::new(&self.device, numel as usize * 4, Some("std_ff_gated"))?;
        match self.ffn_activation {
            GatedActivation::Silu => self.ffn_ops.dispatch_silu_multiply(
                &self.device, &self.queue, encoder, gate, up, &output, numel,
            )?,
            GatedActivation::GeluTanh => {
                let activated = GpuBuffer::new(&self.device, numel as usize * 4, Some("std_ff_gelu"))?;
                self.elementwise.dispatch_gelu(encoder, gate, &activated, numel)?;
                self.elementwise.dispatch_mul(encoder, &activated, up, &output, numel)?;
            }
        }
        Ok(output)
    }

    /// Get the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get the number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Validate both scales before writing either. Signed and zero scales are valid.
    pub fn set_norm_weights(&self, attention: &[f32], ffn: &[f32]) -> Result<()> {
        if [attention, ffn].iter().any(|w| w.len() != self.hidden_dim || w.iter().any(|v| !v.is_finite())) {
            return Err(crate::error::FerrisResError::Shape("norm scales must match hidden_dim and contain only finite values".into()));
        }
        self.queue.write_buffer(self.attn_norm_weight.buffer(), 0, bytemuck::cast_slice(attention));
        self.queue.write_buffer(self.ff_norm_weight.buffer(), 0, bytemuck::cast_slice(ffn));
        Ok(())
    }

    /// Attention and FFN scale buffers, respectively (for checkpoint export).
    pub fn norm_weights(&self) -> (&GpuBuffer, &GpuBuffer) {
        (&self.attn_norm_weight, &self.ff_norm_weight)
    }

    /// Access the Q projection layer (for weight loading).
    pub fn q_proj(&self) -> &Linear {
        &self.q_proj
    }
    /// Access the K projection layer.
    pub fn k_proj(&self) -> &Linear {
        &self.k_proj
    }
    /// Access the V projection layer.
    pub fn v_proj(&self) -> &Linear {
        &self.v_proj
    }
    /// Access the output projection layer.
    pub fn out_proj(&self) -> &Linear {
        &self.out_proj
    }
    /// Access the FFN gate layer.
    pub fn ff_gate(&self) -> &Linear {
        &self.ff_gate
    }
    /// Access the FFN up layer.
    pub fn ff_up(&self) -> &Linear {
        &self.ff_up
    }
    /// Access the FFN down layer.
    pub fn ff_down(&self) -> &Linear {
        &self.ff_down
    }
}

// ---------------------------------------------------------------------------
// StandardTransformerModel
// ---------------------------------------------------------------------------

/// A standard transformer model (O(n²) attention) for compatibility mode.
///
/// This is the counterpart to [`BlockAttnResModel`] but uses
/// [`StandardTransformerLayer`] for full self-attention. It shares the
/// same external interface so TokenGenerator can work with either model type.
pub struct StandardTransformerModel {
    layers: Vec<StandardTransformerLayer>,
    config: StandardTransformerConfig,
    final_norm_weight: GpuBuffer,
    final_norm_ops: GpuTransformerPipeline,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl StandardTransformerModel {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: StandardTransformerConfig,
    ) -> Result<Self> {
        if !config.norm_eps.is_finite() || config.norm_eps < f32::MIN_POSITIVE {
            return Err(crate::error::FerrisResError::Shape("norm_eps must be positive, normal and finite".into()));
        }
        let norm_bytes = config.hidden_dim.checked_mul(4).filter(|&n| n > 0
            && n as u64 <= device.limits().max_storage_buffer_binding_size
            && n as u64 <= device.limits().max_buffer_size)
            .ok_or_else(|| crate::error::FerrisResError::Shape("invalid final norm dimension".into()))?;
        let final_norm_weight = GpuBuffer::new(&device, norm_bytes, Some("std_final_norm_weight"))?;
        queue.write_buffer(final_norm_weight.buffer(), 0, bytemuck::cast_slice(&vec![1.0f32; config.hidden_dim]));
        let final_norm_ops = GpuTransformerPipeline::new(&device)?;
        let num_layers = config.num_layers;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(StandardTransformerLayer::new(
                Arc::clone(&device),
                Arc::clone(&queue),
                &config,
            )?);
        }

        Ok(Self {
            layers,
            config,
            final_norm_weight,
            final_norm_ops,
            device,
            queue,
        })
    }

    /// Upload effective final RMSNorm multipliers after validating all values.
    pub fn set_final_norm_weight(&self, scale: &[f32]) -> Result<()> {
        if scale.len() != self.config.hidden_dim || scale.iter().any(|v| !v.is_finite()) {
            return Err(crate::error::FerrisResError::Shape("final norm scale must match hidden_dim and be finite".into()));
        }
        self.queue.write_buffer(self.final_norm_weight.buffer(), 0, bytemuck::cast_slice(scale));
        Ok(())
    }

    pub fn final_norm_weight(&self) -> &GpuBuffer {
        &self.final_norm_weight
    }

    /// Normalize output hidden states once, after all layers and before LM head.
    pub fn normalize_output(&self, encoder: &mut wgpu::CommandEncoder, hidden: &GpuBuffer, rows: u32) -> Result<GpuBuffer> {
        let bytes = self.config.hidden_dim.checked_mul(rows as usize).and_then(|n| n.checked_mul(4))
            .filter(|&n| rows > 0 && n <= hidden.size()
                && n as u64 <= self.device.limits().max_storage_buffer_binding_size
                && n as u64 <= self.device.limits().max_buffer_size)
            .ok_or_else(|| crate::error::FerrisResError::Shape("invalid final norm rows or input size".into()))?;
        let output = GpuBuffer::new(&self.device, bytes, Some("std_final_norm_output"))?;
        self.final_norm_ops.dispatch_rmsnorm_with_epsilon(
            &self.device, &self.queue, encoder, hidden, &output, &self.final_norm_weight,
            rows, self.config.hidden_dim as u32, self.config.norm_eps,
        )?;
        Ok(output)
    }

    pub fn layers(&self) -> &[StandardTransformerLayer] {
        &self.layers
    }

    pub fn config(&self) -> &StandardTransformerConfig {
        &self.config
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_config_new() {
        let config = StandardTransformerConfig::new(512, 8, 6);
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.intermediate_dim, 2048);
    }

    #[test]
    fn test_llama_7b_config() {
        let config = StandardTransformerConfig::llama_7b();
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.head_dim, 128);
    }

    #[test]
    fn test_mistral_7b_config() {
        let config = StandardTransformerConfig::mistral_7b();
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.intermediate_dim, 14336);
    }

    #[test]
    fn test_from_inferred() {
        let config = StandardTransformerConfig::from_inferred(1024, 16, 12, 50000);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.intermediate_dim, 4096);
        assert_eq!(config.vocab_size, 50000);
    }
}
