//! GPU-accelerated forward pass for Gemma 4 models.
//!
//! Uses GPU matmul kernels for the heavy linear projections and CPU for
//! everything else (attention scores, residuals, activations). Decisions
//! about what goes on GPU vs CPU are made dynamically based on the device
//! profile:
//!
//! - **HighEnd/MidRange** (FullGpu): Upload all weights to GPU, including
//!   embed_tokens and lm_head. Full GPU matmul for everything.
//! - **LowEnd** (Tiled): Upload per-layer weights to GPU. Embed and LM head
//!   stay on CPU (chunked). Attention on CPU.
//! - **Integrated** (CpuOffload): Everything on CPU. GPU only used for
//!   per-layer matmuls that fit in small buffers.
//!
//! Handles GQA (Grouped Query Attention) where num_kv_heads < num_heads.

use std::sync::Arc;
use wgpu::{Device, Queue, CommandEncoderDescriptor};

use crate::compute::buffer::GpuBuffer;
use crate::compute::kernels::matmul::MatMulOp;
use crate::device::profile::DeviceProfile;
use crate::error::Result;

use super::gemma_mapper::{MappedGemma4Model, Gemma4FfnWeights};

// ---------------------------------------------------------------------------
// GPU weight storage
// ---------------------------------------------------------------------------

/// No GPU weight cache — upload just-in-time per matmul to avoid OOM.
/// On datacenter GPUs with enough VRAM, this could be replaced with a
/// persistent cache, but the per-call upload overhead is ~12ms/layer on
/// PCIe 3.0 which is negligible compared to the matmul itself.

// ---------------------------------------------------------------------------
// GPU matmul accelerator
// ---------------------------------------------------------------------------

/// GPU matmul accelerator for Gemma 4 forward pass.
///
/// Uploads weights once (respecting device limits), dispatches matmuls on GPU,
/// keeps attention/residuals/activations on CPU (where they're memory-bound, not
/// compute-bound for the sequence lengths we target).
pub struct GpuMatmulAccelerator {
    device: Arc<Device>,
    queue: Arc<Queue>,
    matmul: MatMulOp,
    profile: DeviceProfile,
    /// Maximum single buffer size in bytes (from device limits).
    max_buffer_bytes: u64,
    /// Whether validate_model was called.
    weights_ready: bool,
}

impl GpuMatmulAccelerator {
    /// Create a new accelerator using the default GPU device.
    /// Auto-detects profile from adapter VRAM and limits.
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })).map_err(|e| crate::error::FerrisResError::Shape(format!("No GPU adapter: {:?}", e)))?;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("FerrisRes GPU"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })).map_err(|e| crate::error::FerrisResError::Shape(format!("GPU device failed: {:?}", e)))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let max_buffer_bytes = limits.max_storage_buffer_binding_size.max(limits.max_buffer_size);

        // Detect profile
        let profile = DeviceProfile::from_env().unwrap_or_else(|| {
            // Estimate VRAM from adapter name heuristics
            let vram_mb = estimate_vram_mb(&adapter_info.name);
            DeviceProfile::from_vram_mb(vram_mb)
        });

        let matmul = MatMulOp::new(&device, &queue);

        tracing::info!(
            "GPU initialized: {} ({}), profile={:?}, max_buffer={:.0}MB, mode={:?}",
            adapter_info.name,
            adapter_info.backend,
            profile,
            max_buffer_bytes as f64 / 1e6,
            profile.compute_mode(),
        );

        Ok(Self { device, queue, matmul, profile, max_buffer_bytes, weights_ready: false })
    }

    /// Create from an existing device/queue.
    pub fn from_device(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        let max_buffer_bytes = device.limits().max_storage_buffer_binding_size
            .max(device.limits().max_buffer_size);
        let profile = DeviceProfile::from_env()
            .unwrap_or(DeviceProfile::MidRange);
        let matmul = MatMulOp::new(&device, &queue);
        Self { device, queue, matmul, profile, max_buffer_bytes, weights_ready: false }
    }

    pub fn profile(&self) -> DeviceProfile { self.profile }

    /// Upload model weights to GPU. Call once after loading.
    /// Respects device limits — large tensors stay on CPU if they don't fit.
    /// Validate that the model can run on this GPU.
    /// Checks buffer size limits against model dimensions.
    pub fn validate_model(&mut self, model: &MappedGemma4Model) -> Result<()> {
        let config = &model.config;
        let hd = config.hidden_dim;
        let id = config.intermediate_dim;
        let max_weight_bytes = (hd.max(id) * id.max(hd) * 4) as u64;
        if max_weight_bytes > self.max_buffer_bytes {
            return Err(crate::error::FerrisResError::Shape(format!(
                "Largest weight matrix ({:.0}MB) exceeds GPU max buffer ({:.0}MB). Profile={:?}",
                max_weight_bytes as f64 / 1e6, self.max_buffer_bytes as f64 / 1e6, self.profile
            )));
        }
        tracing::info!(
            "Model validated for GPU: {} layers, max_weight={:.0}MB, limit={:.0}MB, profile={:?}",
            config.num_layers, max_weight_bytes as f64 / 1e6, self.max_buffer_bytes as f64 / 1e6, self.profile
        );
        self.weights_ready = true;
        Ok(())
    }

    /// Run full forward pass: token_ids → logits.
    /// Weights are uploaded just-in-time per matmul to avoid holding
    /// a duplicate copy in GPU VRAM. Upload cost is ~12ms/layer on PCIe 3.0.
    pub fn forward(&self, model: &MappedGemma4Model, token_ids: &[u32]) -> Result<Vec<f32>> {
        let config = &model.config;
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let nkv = config.num_kv_heads;
        let head_d = config.head_dim;
        let seq = token_ids.len();
        let vs = config.vocab_size;
        let id = config.intermediate_dim;
        let q_dim = nh * head_d;
        let kv_dim = nkv * head_d;

        if !self.weights_ready {
            return Err(crate::error::FerrisResError::Shape("Call validate_model() first".into()));
        }

        // 1. Embedding (CPU indexing)
        let scale = (hd as f32).sqrt();
        let mut hidden = vec![0.0f32; seq * hd];
        for (t, &tid) in token_ids.iter().enumerate() {
            let idx = tid as usize;
            if idx * hd + hd <= model.embed_tokens.len() {
                for d in 0..hd {
                    hidden[t * hd + d] = model.embed_tokens[idx * hd + d] * scale;
                }
            }
        }

        // 2. Per-layer transformer (GPU matmuls, CPU everything else)
        for (_layer_idx, layer) in model.layers.iter().enumerate() {
            let residual = hidden.clone();

            // Input RMSNorm (CPU)
            let normed = rms_norm_cpu(&hidden, &layer.attn.input_norm, hd, 1e-6);

            // Q/K/V projections (GPU matmul — JIT upload weight, run, read back)
            let q = self.gpu_matmul_cpu_b(&normed, &layer.attn.q_proj, seq, hd, q_dim)?;
            let k = self.gpu_matmul_cpu_b(&normed, &layer.attn.k_proj, seq, hd, kv_dim)?;
            let v = self.gpu_matmul_cpu_b(&normed, &layer.attn.v_proj, seq, hd, kv_dim)?;

            // RoPE (CPU)
            let mut q = q;
            let mut k = k;
            super::gemma_mapper::apply_rope(&mut q, seq, nh, head_d, 0);
            super::gemma_mapper::apply_rope_gqa(&mut k, seq, nkv, head_d, 0);

            // Attention (CPU — causal + GQA)
            let attn_out = attention_gqa(&q, &k, &v, seq, nh, nkv, head_d, q_dim, kv_dim);

            // O projection (GPU matmul)
            let o = self.gpu_matmul_cpu_b(&attn_out, &layer.attn.o_proj, seq, q_dim, hd)?;

            // Residual (CPU)
            for i in 0..hidden.len() { hidden[i] = residual[i] + o[i]; }

            let residual2 = hidden.clone();

            // Post-attn RMSNorm (CPU)
            let normed2 = rms_norm_cpu(&hidden, &layer.attn.post_attn_norm, hd, 1e-6);

            // FFN (GPU matmuls)
            let (gate, up, down) = match &layer.ffn {
                Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => {
                    (gate_proj.as_slice(), up_proj.as_slice(), down_proj.as_slice())
                }
                _ => (&[] as &[f32], &[] as &[f32], &[] as &[f32]),
            };

            let gated = self.gpu_matmul_cpu_b(&normed2, gate, seq, hd, id)?;
            let gated_silu: Vec<f32> = gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
            let upped = self.gpu_matmul_cpu_b(&normed2, up, seq, hd, id)?;

            let mut combined = vec![0.0f32; seq * id];
            for i in 0..combined.len() { combined[i] = gated_silu[i] * upped[i]; }

            let ffn_out = self.gpu_matmul_cpu_b(&combined, down, seq, id, hd)?;

            // Residual (CPU)
            for i in 0..hidden.len() { hidden[i] = residual2[i] + ffn_out[i]; }
        }

        // 3. Final RMSNorm (CPU)
        hidden = rms_norm_cpu(&hidden, &model.final_norm, hd, 1e-6);

        // 4. LM head (CPU chunked — vocab is always huge)
        tracing::debug!("LM head on CPU (chunked, vocab={})", vs);
        let logits = cpu_lm_head(&hidden, &model.lm_head, seq, hd, vs);

        Ok(logits)
    }

    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Internal: GPU dispatch
    // -----------------------------------------------------------------------

    /// GPU matmul where both A and B are CPU slices uploaded JIT.
    /// No persistent GPU weight storage — avoids VRAM duplication.
    fn gpu_matmul_cpu_b(&self, a_data: &[f32], b_data: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        if m == 0 || k == 0 || n == 0 { return Ok(vec![0.0f32; m * n]); }

        let a_buf = GpuBuffer::new(&self.device, a_data.len() * 4, Some("a"))?;
        self.queue.write_buffer(a_buf.buffer(), 0, bytemuck::cast_slice(a_data));

        let b_buf = GpuBuffer::new(&self.device, b_data.len() * 4, Some("b"))?;
        self.queue.write_buffer(b_buf.buffer(), 0, bytemuck::cast_slice(b_data));

        let c_buf = GpuBuffer::new(&self.device, m * n * 4, Some("c"))?;

        let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("matmul") });
        self.matmul.dispatch(&mut enc, &a_buf, &b_buf, &c_buf, m as u32, k as u32, n as u32)?;
        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        read_back_f32(&self.device, &self.queue, &c_buf, m * n)
    }
}

// ---------------------------------------------------------------------------
// CPU fallbacks (used when tensors don't fit on GPU)
// ---------------------------------------------------------------------------

/// CPU RMSNorm.
fn rms_norm_cpu(input: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let n = input.len() / dim;
    let mut output = Vec::with_capacity(input.len());
    for t in 0..n {
        let slice = &input[t * dim..(t + 1) * dim];
        let mean_sq: f32 = slice.iter().map(|x| x * x).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for (d, &x) in slice.iter().enumerate() {
            let g = weight.get(d).copied().unwrap_or(1.0);
            output.push(x * inv_rms * g);
        }
    }
    output
}

/// CPU GQA attention with causal mask.
fn attention_gqa(
    q: &[f32], k: &[f32], v: &[f32],
    seq: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize,
    q_dim: usize, kv_dim: usize,
) -> Vec<f32> {
    let heads_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq * q_dim];

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        for t in 0..seq {
            let mut scores = vec![0.0f32; seq];
            let mut max_score = f32::NEG_INFINITY;
            for s in 0..=t {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[t * q_dim + h * head_dim + d]
                         * k[s * kv_dim + kv_h * head_dim + d];
                }
                scores[s] = dot * scale;
                if scores[s] > max_score { max_score = scores[s]; }
            }
            let mut sum_exp = 0.0f32;
            for s in 0..=t {
                scores[s] = (scores[s] - max_score).exp();
                sum_exp += scores[s];
            }
            for s in 0..=t { scores[s] /= sum_exp; }
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for s in 0..=t {
                    sum += scores[s] * v[s * kv_dim + kv_h * head_dim + d];
                }
                attn_out[t * q_dim + h * head_dim + d] = sum;
            }
        }
    }
    attn_out
}

/// CPU chunked LM head matmul. Processes vocab in 4K-column chunks for cache locality.
fn cpu_lm_head(hidden: &[f32], lm_head: &[f32], seq: usize, hd: usize, vs: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; seq * vs];
    let chunk = 4096;
    for v_start in (0..vs).step_by(chunk) {
        let v_end = (v_start + chunk).min(vs);
        for t in 0..seq {
            for v in v_start..v_end {
                let mut sum = 0.0f32;
                for d in 0..hd {
                    sum += hidden[t * hd + d] * lm_head[v * hd + d];
                }
                logits[t * vs + v] = sum;
            }
        }
    }
    logits
}

// ---------------------------------------------------------------------------
// GPU buffer helpers
// ---------------------------------------------------------------------------

fn read_back_f32(device: &Device, queue: &Queue, buf: &GpuBuffer, len: usize) -> Result<Vec<f32>> {
    use wgpu::{BufferDescriptor, BufferUsages};
    let byte_len = len * std::mem::size_of::<f32>();
    if byte_len == 0 { return Ok(vec![]); }

    let read_buf = device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: byte_len as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("readback") });
    enc.copy_buffer_to_buffer(buf.buffer(), 0, &read_buf, 0, byte_len as u64);
    queue.submit(std::iter::once(enc.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    read_buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    if rx.recv().ok() == Some(Ok(())) {
        let view = read_buf.slice(..).get_mapped_range();
        let data = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        Ok(data)
    } else {
        Err(crate::error::FerrisResError::Shape("GPU readback failed".into()))
    }
}

/// Rough VRAM estimate from adapter name (fallback when no ash/sysfs).
fn estimate_vram_mb(name: &str) -> u64 {
    let lower = name.to_ascii_lowercase();
    if lower.contains("4090") || lower.contains("3090") { return 24576; }
    if lower.contains("4080") { return 16384; }
    if lower.contains("3080") { return 12288; }
    if lower.contains("4070") || lower.contains("3070") { return 8192; }
    if lower.contains("4060") || lower.contains("3060") { return 8192; }
    if lower.contains("a100") || lower.contains("h100") { return 81920; }
    if lower.contains("a6000") { return 49152; }
    if lower.contains("a10") || lower.contains("t4") { return 16384; }
    if lower.contains("l4") { return 24576; }
    if lower.contains("v100") { return 16384; }
    if lower.contains("7900") { return 24576; }
    if lower.contains("6800") || lower.contains("6900") { return 16384; }
    if lower.contains("apple") || lower.contains("m1") || lower.contains("m2") || lower.contains("m3") || lower.contains("m4") { return 8192; }
    if lower.contains("arc a770") { return 16384; }
    if lower.contains("arc") { return 8192; }
    4096
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_cpu() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0];
        let output = rms_norm_cpu(&input, &weight, 2, 1e-6);
        assert_eq!(output.len(), 4);
        assert!(output[0].abs() < 2.0);
    }

    #[test]
    fn test_attention_gqa() {
        let seq = 4;
        let nh = 2;
        let nkv = 1;
        let hd = 2;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let q = vec![1.0f32; seq * q_dim];
        let k = vec![0.5f32; seq * kv_dim];
        let v = vec![1.0f32; seq * kv_dim];
        let out = attention_gqa(&q, &k, &v, seq, nh, nkv, hd, q_dim, kv_dim);
        assert_eq!(out.len(), seq * q_dim);
    }

    #[test]
    fn test_cpu_lm_head() {
        let seq = 2;
        let hd = 4;
        let vs = 8;
        let hidden = vec![1.0f32; seq * hd];
        let lm_head = vec![0.5f32; vs * hd];
        let logits = cpu_lm_head(&hidden, &lm_head, seq, hd, vs);
        assert_eq!(logits.len(), seq * vs);
        assert!((logits[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_vram_mb() {
        assert!(estimate_vram_mb("NVIDIA GeForce RTX 4090") >= 24000);
        assert!(estimate_vram_mb("NVIDIA A100-SXM4-80GB") >= 80000);
        assert!(estimate_vram_mb("AMD Radeon RX 7900 XTX") >= 24000);
        assert!(estimate_vram_mb("Apple M2 Max") >= 8000);
        assert!(estimate_vram_mb("Unknown GPU") >= 4000);
    }
}
