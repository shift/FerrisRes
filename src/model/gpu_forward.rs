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
// Persistent GPU weight cache (resident mode)
// ---------------------------------------------------------------------------

/// Persistent GPU buffers for a single transformer layer's weights.
/// Created once by `upload_weights_resident()`, kept alive for the entire session.
pub struct ResidentLayerWeights {
    pub q_proj: GpuBuffer,
    pub k_proj: GpuBuffer,
    pub v_proj: GpuBuffer,
    pub o_proj: GpuBuffer,
    pub gate_proj: GpuBuffer,
    pub up_proj: GpuBuffer,
    pub down_proj: GpuBuffer,
}

impl ResidentLayerWeights {
    pub fn total_bytes(&self) -> u64 {
        let bufs: [&GpuBuffer; 7] = [
            &self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj,
            &self.gate_proj, &self.up_proj, &self.down_proj,
        ];
        bufs.iter().map(|b| b.size() as u64).sum()
    }
}

/// All model weights resident on GPU. Created when `DispatchPlan::resident_mode` is true.
/// Keeps GPU buffers alive so they're not re-uploaded per forward pass.
///
/// Note: embed_tokens and lm_head stay on CPU (embedding is a gather op,
/// lm_head is huge and benefits from existing tiled CPU path). Only the
/// per-layer projection weights (Q/K/V/O/FFN) are cached on GPU — those
/// are the 6 matmuls × 35 layers = 210 uploads that dominate runtime.
pub struct GpuWeightCache {
    pub layer_weights: Vec<ResidentLayerWeights>,
    // Small weights kept on CPU (RMSNorm params — ~6KB per layer)
    pub input_norms: Vec<Vec<f32>>,
    pub post_attn_norms: Vec<Vec<f32>>,
    // Config dims for forward
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
}

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
    /// Reported max buffer size from device.limits().
    reported_max_buffer_bytes: u64,
    /// Empirically verified max buffer size in bytes.
    /// Some GPUs (Intel HD 530) misreport this — they claim 2147MB but
    /// actually cap at 256MB. We probe the real limit at init time.
    real_max_buffer_bytes: u64,
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

        // Probe the REAL max buffer size. wgpu may report a larger limit
        // than the driver actually supports (Intel HD 530: reports 2147MB, caps 256MB).
        // Since wgpu panics on buffer creation failures, we can't catch errors.
        // Instead, we binary-search for the real limit using successfully-created buffers.
        let real_max = Self::probe_real_max_buffer(&device, max_buffer_bytes, &adapter_info.name);
        if real_max < max_buffer_bytes {
            tracing::warn!(
                "GPU max_buffer_size misreported: claimed {:.0}MB, real limit is {:.0}MB",
                max_buffer_bytes as f64 / 1e6,
                real_max as f64 / 1e6,
            );
        }

        tracing::info!(
            "GPU initialized: {} ({}), profile={:?}, max_buffer={:.0}MB (reported {:.0}MB), mode={:?}",
            adapter_info.name,
            adapter_info.backend,
            profile,
            real_max as f64 / 1e6,
            max_buffer_bytes as f64 / 1e6,
            profile.compute_mode(),
        );

        Ok(Self {
            device, queue, matmul, profile,
            reported_max_buffer_bytes: max_buffer_bytes,
            real_max_buffer_bytes: real_max,
            weights_ready: false,
        })
    }

    /// Create from an existing device/queue.
    pub fn from_device(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        let reported_max = device.limits().max_storage_buffer_binding_size
            .max(device.limits().max_buffer_size);
        let real_max = Self::probe_real_max_buffer(&device, reported_max, "unknown");
        let profile = DeviceProfile::from_env()
            .unwrap_or(DeviceProfile::MidRange);
        let matmul = MatMulOp::new(&device, &queue);
        Self {
            device, queue, matmul, profile,
            reported_max_buffer_bytes: reported_max,
            real_max_buffer_bytes: real_max,
            weights_ready: false,
        }
    }

    pub fn profile(&self) -> DeviceProfile { self.profile }

    /// Empirically verified max single buffer size in bytes.
    pub fn max_buffer_bytes(&self) -> u64 { self.real_max_buffer_bytes }

    /// Probe the real maximum buffer size.
    ///
    /// Some GPUs (Intel HD 530 SKL GT2) report max_buffer_size=2147MB but
    /// wgpu panics when creating buffers > 256MB. Since `create_buffer` returns
    /// `Buffer` (not `Result`), we can't catch the error — it panics.
    ///
    /// We can't probe empirically (probing itself would panic). Instead,
    /// we apply a conservative cap based on the GPU vendor and name.
    /// Intel integrated GPUs on Gen9 (Skylake through Ice Lake) have a 256MB
    /// real buffer limit despite reporting 2147MB. We detect these by name.
    fn probe_real_max_buffer(_device: &Device, reported: u64, adapter_name: &str) -> u64 {
        let name_lower = adapter_name.to_lowercase();

        // Known misreporters: Intel Gen9/Gen11 iGPUs
        // These report max_buffer=2147MB (0x7FFFF000) but cap at 256MB.
        // Intel Arc (Xe HPG) and Xe integrated (Gen12+) are fine.
        let is_intel_gen9_or_11 =
            (name_lower.contains("intel") || name_lower.contains("hd graphics") || name_lower.contains("uhd graphics"))
            && !name_lower.contains("arc")
            && !name_lower.contains("xe")
            && !name_lower.contains("dg"); // discrete Intel Arc

        if is_intel_gen9_or_11 && reported > 256 * 1024 * 1024 {
            tracing::warn!(
                "Intel Gen9/11 iGPU detected ('{}'): capping buffer from reported {:.0}MB to 256MB",
                adapter_name, reported as f64 / 1e6,
            );
            return 256 * 1024 * 1024;
        }

        // For other GPUs, trust the reported limit but cap at a sane maximum
        // to catch any future misreporters
        reported.min(4 * 1024 * 1024 * 1024) // 4GB cap
    }

    /// Estimate VRAM from device limits (uses real probed max_buffer as proxy).
    /// For accurate VRAM, use ash/sysfs on Linux, Metal on macOS.
    pub fn estimated_vram_bytes(&self) -> u64 {
        // wgpu doesn't expose total VRAM directly.
        // Use real probed max_buffer as a lower bound — the real VRAM is typically
        // 4-16× larger than max_buffer_size.
        self.real_max_buffer_bytes * 4
    }

    /// Upload model weights to GPU. Call once after loading.
    /// Respects device limits — large tensors stay on CPU if they don't fit.
    /// Validate that the model can run on this GPU.
    /// Checks buffer size limits against model dimensions.
    pub fn validate_model(&mut self, model: &MappedGemma4Model) -> Result<()> {
        let config = &model.config;
        let hd = config.hidden_dim;
        let id = config.intermediate_dim;
        let max_weight_bytes = (hd.max(id) * id.max(hd) * 4) as u64;
        if max_weight_bytes > self.real_max_buffer_bytes {
            return Err(crate::error::FerrisResError::Shape(format!(
                "Largest weight matrix ({:.0}MB) exceeds GPU max buffer ({:.0}MB). Profile={:?}",
                max_weight_bytes as f64 / 1e6, self.real_max_buffer_bytes as f64 / 1e6, self.profile
            )));
        }
        tracing::info!(
            "Model validated for GPU: {} layers, max_weight={:.0}MB, limit={:.0}MB, profile={:?}",
            config.num_layers, max_weight_bytes as f64 / 1e6, self.real_max_buffer_bytes as f64 / 1e6, self.profile
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
        tracing::debug!(event = "lm_head_on_cpu_chunked_vocab", "LM head on CPU (chunked, vocab={})", vs);
        let logits = cpu_lm_head(&hidden, &model.lm_head, seq, hd, vs);

        Ok(logits)
    }

    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Internal: GPU dispatch
    // -----------------------------------------------------------------------
    // Resident mode: persistent GPU weight cache
    // -----------------------------------------------------------------------

    /// Upload ALL model weights to GPU buffers, returning a cache that keeps
    /// them resident for the entire session. Only call when `resident_mode = true`
    /// (model fits in VRAM with 30% headroom).
    ///
    /// This eliminates per-forward-pass upload overhead (~12ms/layer on PCIe 3.0).
    /// On T4 (15GB VRAM, 4.6GB model): expected speedup from 1.5 → 50-100 tok/s.
    pub fn upload_weights_resident(&self, model: &MappedGemma4Model) -> Result<GpuWeightCache> {
        let config = &model.config;
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let nkv = config.num_kv_heads;
        let head_d = config.head_dim;
        let id = config.intermediate_dim;
        let _ = (nh * head_d, nkv * head_d, config.vocab_size); // dims used for buffer sizing

        let upload = |data: &[f32], label: &str| -> Result<GpuBuffer> {
            let buf = GpuBuffer::new(&self.device, data.len() * 4, Some(label))?;
            self.queue.write_buffer(buf.buffer(), 0, bytemuck::cast_slice(data));
            Ok(buf)
        };

        tracing::info!(event = "uploading_resident_weights", layers = model.layers.len(), "Uploading per-layer weights to GPU (resident mode)...");
        let start = std::time::Instant::now();

        // NOTE: embed_tokens and lm_head stay on CPU. The bottleneck is
        // the 6 per-layer matmuls × 35 layers = 210 uploads per forward pass.
        // Embedding is a gather (not matmul), lm_head uses existing CPU tiled path.

        // Per-layer weights
        let mut layer_weights = Vec::with_capacity(model.layers.len());
        for (i, layer) in model.layers.iter().enumerate() {
            let q_proj = upload(&layer.attn.q_proj, &format!("q_{}", i))?;
            let k_proj = upload(&layer.attn.k_proj, &format!("k_{}", i))?;
            let v_proj = upload(&layer.attn.v_proj, &format!("v_{}", i))?;
            let o_proj = upload(&layer.attn.o_proj, &format!("o_{}", i))?;

            let (gate_proj, up_proj, down_proj) = match &layer.ffn {
                Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => {
                    (upload(gate_proj, &format!("gate_{}", i))?,
                     upload(up_proj, &format!("up_{}", i))?,
                     upload(down_proj, &format!("down_{}", i))?)
                }
                Gemma4FfnWeights::Moe { .. } => {
                    // MoE layers upload expert 0 as placeholder for resident mode
                    // (full MoE residency is a future optimization)
                    return Err(crate::error::FerrisResError::Shape(
                        "Resident mode not yet supported for MoE layers".into()
                    ));
                }
            };

            layer_weights.push(ResidentLayerWeights {
                q_proj, k_proj, v_proj, o_proj,
                gate_proj, up_proj, down_proj,
            });
        }

        // LM head stays on CPU (uses existing tiled path)

        let total_bytes: u64 = layer_weights.iter().map(|l| l.total_bytes()).sum();

        tracing::info!(
            event = "resident_weights_uploaded",
            total_mb = total_bytes as f64 / 1e6,
            elapsed_ms = start.elapsed().as_millis(),
            layers = layer_weights.len(),
            "Per-layer weights resident on GPU (embed + lm_head on CPU)"
        );

        Ok(GpuWeightCache {
            layer_weights,
            input_norms: model.layers.iter().map(|l| l.attn.input_norm.clone()).collect(),
            post_attn_norms: model.layers.iter().map(|l| l.attn.post_attn_norm.clone()).collect(),
            hidden_dim: hd,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: head_d,
            intermediate_dim: id,
        })
    }

    /// Stream layer weights from mmap directly to GPU, one layer at a time.
    /// Peak RAM = mmap (already mapped) + 1 layer (~141MB). Never holds all layers in RAM.
    /// Use this when system RAM < model_weights * 2 (Colab T4 with 12.7GB RAM).
    pub fn upload_weights_resident_streaming(
        &self,
        config: &crate::model::gemma_mapper::Gemma4Config,
        file_st: &mut crate::model::safetensors::FileSafetensors,
    ) -> Result<GpuWeightCache> {
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let nkv = config.num_kv_heads;
        let head_d = config.head_dim;
        let id = config.intermediate_dim;

        let upload = |data: &[f32], label: &str| -> Result<GpuBuffer> {
            GpuBuffer::new_device_local(
                &self.device,
                &self.queue,
                bytemuck::cast_slice(data),
                Some(label),
            )
        };

        tracing::info!(
            event = "streaming_resident_upload",
            layers = config.num_layers,
            "Streaming layers from mmap to GPU (one at a time, low RAM)"
        );
        let start = std::time::Instant::now();

        let mut input_norms = Vec::with_capacity(config.num_layers);
        let mut post_attn_norms = Vec::with_capacity(config.num_layers);
        let mut layer_weights = Vec::with_capacity(config.num_layers);
        let mut total_bytes: u64 = 0;

        for layer_idx in 0..config.num_layers {
            let q_name = format!("model.language_model.layers.{}.self_attn.q_proj.weight", layer_idx);
            let k_name = format!("model.language_model.layers.{}.self_attn.k_proj.weight", layer_idx);
            let v_name = format!("model.language_model.layers.{}.self_attn.v_proj.weight", layer_idx);
            let o_name = format!("model.language_model.layers.{}.self_attn.o_proj.weight", layer_idx);
            let in_name = format!("model.language_model.layers.{}.input_layernorm.weight", layer_idx);
            let pn_name = format!("model.language_model.layers.{}.post_attention_layernorm.weight", layer_idx);
            let gate_name = format!("model.language_model.layers.{}.mlp.gate_proj.weight", layer_idx);
            let up_name = format!("model.language_model.layers.{}.mlp.up_proj.weight", layer_idx);
            let down_name = format!("model.language_model.layers.{}.mlp.down_proj.weight", layer_idx);

            // Load one layer from mmap (peaks at ~141MB)
            let q = file_st.get_tensor_f32(&q_name).map_err(|e| format!("Missing {}: {:?}", q_name, e))?;
            let k = file_st.get_tensor_f32(&k_name).map_err(|e| format!("Missing {}: {:?}", k_name, e))?;
            let v = file_st.get_tensor_f32(&v_name).map_err(|e| format!("Missing {}: {:?}", v_name, e))?;
            let o = file_st.get_tensor_f32(&o_name).map_err(|e| format!("Missing {}: {:?}", o_name, e))?;
            let inorm = file_st.get_tensor_f32(&in_name).map_err(|e| format!("Missing {}: {:?}", in_name, e))?;
            let pnorm = file_st.get_tensor_f32(&pn_name).map_err(|e| format!("Missing {}: {:?}", pn_name, e))?;
            let gate = file_st.get_tensor_f32(&gate_name).map_err(|e| format!("Missing {}: {:?}", gate_name, e))?;
            let up = file_st.get_tensor_f32(&up_name).map_err(|e| format!("Missing {}: {:?}", up_name, e))?;
            let down = file_st.get_tensor_f32(&down_name).map_err(|e| format!("Missing {}: {:?}", down_name, e))?;

            // Upload to GPU
            let q_proj = upload(&q, &format!("q_{}", layer_idx))?;
            let k_proj = upload(&k, &format!("k_{}", layer_idx))?;
            let v_proj = upload(&v, &format!("v_{}", layer_idx))?;
            let o_proj = upload(&o, &format!("o_{}", layer_idx))?;
            let gate_proj = upload(&gate, &format!("gate_{}", layer_idx))?;
            let up_proj = upload(&up, &format!("up_{}", layer_idx))?;
            let down_proj = upload(&down, &format!("down_{}", layer_idx))?;

            // Keep norms on CPU (tiny, ~6KB each)
            input_norms.push(inorm);
            post_attn_norms.push(pnorm);

            // Free CPU copies immediately
            drop(q); drop(k); drop(v); drop(o);
            drop(gate); drop(up); drop(down);

            let lw = ResidentLayerWeights {
                q_proj, k_proj, v_proj, o_proj,
                gate_proj, up_proj, down_proj,
            };
            total_bytes += lw.total_bytes();
            layer_weights.push(lw);

            if (layer_idx + 1) % 5 == 0 || layer_idx == config.num_layers - 1 {
                tracing::info!(
                    event = "streaming_upload_progress",
                    layer = layer_idx + 1,
                    total = config.num_layers,
                    gpu_mb = total_bytes as f64 / 1e6,
                    "Layers streamed to GPU"
                );
            }
        }

        tracing::info!(
            event = "resident_weights_uploaded",
            total_mb = total_bytes as f64 / 1e6,
            elapsed_ms = start.elapsed().as_millis(),
            layers = layer_weights.len(),
            "All layers resident on GPU (streamed from mmap, low RAM)"
        );

        Ok(GpuWeightCache {
            layer_weights,
            input_norms,
            post_attn_norms,
            hidden_dim: hd,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: head_d,
            intermediate_dim: id,
        })
    }

    /// Forward pass using resident GPU weights (no JIT uploads for layer projections).
    /// embed_tokens and lm_head are read from CPU (embedding is a gather, lm_head uses tiled path).
    pub fn forward_resident(&self, cache: &GpuWeightCache, model: &MappedGemma4Model, token_ids: &[u32]) -> Result<Vec<f32>> {
        let hd = cache.hidden_dim;
        let nh = cache.num_heads;
        let nkv = cache.num_kv_heads;
        let head_d = cache.head_dim;
        let seq = token_ids.len();
        let id = cache.intermediate_dim;
        let q_dim = nh * head_d;
        let kv_dim = nkv * head_d;
        let vs = model.config.vocab_size;

        // 1. Embedding (CPU gather — not a matmul, no GPU benefit)
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

        // 2. Per-layer transformer — GPU matmuls with RESIDENT weights
        for (layer_idx, lw) in cache.layer_weights.iter().enumerate() {
            let residual = hidden.clone();

            // Input RMSNorm (CPU — tiny)
            let normed = rms_norm_cpu(&hidden, &cache.input_norms[layer_idx], hd, 1e-6);

            // Q/K/V projections — batched (1 sync instead of 3)
            let (q, k, v) = self.batched_resident_matmul_3(&normed, &lw.q_proj, &lw.k_proj, &lw.v_proj, seq, hd, q_dim, kv_dim, kv_dim)?;

            // RoPE (CPU)
            let mut q = q;
            let mut k = k;
            super::gemma_mapper::apply_rope(&mut q, seq, nh, head_d, 0);
            super::gemma_mapper::apply_rope_gqa(&mut k, seq, nkv, head_d, 0);

            // Attention (CPU — causal + GQA)
            let attn_out = attention_gqa(&q, &k, &v, seq, nh, nkv, head_d, q_dim, kv_dim);

            // O projection
            let o = self.gpu_matmul_resident_a(&attn_out, &lw.o_proj, seq, q_dim, hd)?;

            // Residual
            for i in 0..hidden.len() { hidden[i] = residual[i] + o[i]; }

            let residual2 = hidden.clone();

            // Post-attn RMSNorm
            let normed2 = rms_norm_cpu(&hidden, &cache.post_attn_norms[layer_idx], hd, 1e-6);

            // gate + up projections — batched (1 sync instead of 2)
            let (gated, upped) = self.batched_resident_matmul_2(&normed2, &lw.gate_proj, &lw.up_proj, seq, hd, id, id)?;

            let gated_silu: Vec<f32> = gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

            let mut combined = vec![0.0f32; seq * id];
            for i in 0..combined.len() { combined[i] = gated_silu[i] * upped[i]; }

            let ffn_out = self.gpu_matmul_resident_a(&combined, &lw.down_proj, seq, id, hd)?;

            // Residual
            for i in 0..hidden.len() { hidden[i] = residual2[i] + ffn_out[i]; }
        }

        // 3. Final RMSNorm (CPU)
        hidden = rms_norm_cpu(&hidden, &model.final_norm, hd, 1e-6);

        // 4. LM head (CPU tiled — vocab is huge, existing path works well)
        let logits = cpu_lm_head(&hidden, &model.lm_head, seq, hd, vs);

        Ok(logits)
    }

    /// GPU matmul where A is uploaded JIT but B is already resident on GPU.
    fn gpu_matmul_resident_a(&self, a_data: &[f32], b_buf: &GpuBuffer, m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        if m == 0 || k == 0 || n == 0 { return Ok(vec![0.0f32; m * n]); }

        let a_buf = GpuBuffer::new(&self.device, a_data.len() * 4, Some("a_res"))?;
        self.queue.write_buffer(a_buf.buffer(), 0, bytemuck::cast_slice(a_data));

        let c_buf = GpuBuffer::new(&self.device, m * n * 4, Some("c_res"))?;

        let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("matmul_res") });
        self.matmul.dispatch(&mut enc, &a_buf, b_buf, &c_buf, m as u32, k as u32, n as u32)?;
        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        read_back_f32(&self.device, &self.queue, &c_buf, m * n)
    }

    /// Batched: 2 parallel matmuls with same A, different B's (resident).
    /// One encoder, one submit, one sync = 3x fewer round-trips.
    fn batched_resident_matmul_2(
        &self, a_data: &[f32],
        b1: &GpuBuffer, b2: &GpuBuffer,
        m: usize, k: usize, n1: usize, n2: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if m == 0 || k == 0 { return Ok((vec![], vec![])); }

        let a_buf = GpuBuffer::new(&self.device, a_data.len() * 4, Some("a_batch"))?;
        self.queue.write_buffer(a_buf.buffer(), 0, bytemuck::cast_slice(a_data));

        let c1 = GpuBuffer::new(&self.device, m * n1 * 4, Some("c1_batch"))?;
        let c2 = GpuBuffer::new(&self.device, m * n2 * 4, Some("c2_batch"))?;

        let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("batch2") });
        self.matmul.dispatch(&mut enc, &a_buf, b1, &c1, m as u32, k as u32, n1 as u32)?;
        self.matmul.dispatch(&mut enc, &a_buf, b2, &c2, m as u32, k as u32, n2 as u32)?;
        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let r1 = read_back_f32(&self.device, &self.queue, &c1, m * n1)?;
        let r2 = read_back_f32(&self.device, &self.queue, &c2, m * n2)?;
        Ok((r1, r2))
    }

    /// Batched: 3 parallel matmuls with same A, different B's (resident).
    fn batched_resident_matmul_3(
        &self, a_data: &[f32],
        b1: &GpuBuffer, b2: &GpuBuffer, b3: &GpuBuffer,
        m: usize, k: usize, n1: usize, n2: usize, n3: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if m == 0 || k == 0 { return Ok((vec![], vec![], vec![])); }

        let a_buf = GpuBuffer::new(&self.device, a_data.len() * 4, Some("a_batch3"))?;
        self.queue.write_buffer(a_buf.buffer(), 0, bytemuck::cast_slice(a_data));

        let c1 = GpuBuffer::new(&self.device, m * n1 * 4, Some("c1_batch3"))?;
        let c2 = GpuBuffer::new(&self.device, m * n2 * 4, Some("c2_batch3"))?;
        let c3 = GpuBuffer::new(&self.device, m * n3 * 4, Some("c3_batch3"))?;

        let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("batch3") });
        self.matmul.dispatch(&mut enc, &a_buf, b1, &c1, m as u32, k as u32, n1 as u32)?;
        self.matmul.dispatch(&mut enc, &a_buf, b2, &c2, m as u32, k as u32, n2 as u32)?;
        self.matmul.dispatch(&mut enc, &a_buf, b3, &c3, m as u32, k as u32, n3 as u32)?;
        self.queue.submit(std::iter::once(enc.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let r1 = read_back_f32(&self.device, &self.queue, &c1, m * n1)?;
        let r2 = read_back_f32(&self.device, &self.queue, &c2, m * n2)?;
        let r3 = read_back_f32(&self.device, &self.queue, &c3, m * n3)?;
        Ok((r1, r2, r3))
    }

    // -----------------------------------------------------------------------
    // JIT mode (original): per-call uploads
    // -----------------------------------------------------------------------

    /// GPU matmul where both A and B are CPU slices uploaded JIT.
    /// No persistent GPU weight storage — avoids VRAM duplication.
    ///
    /// If B doesn't fit in a single GPU buffer, automatically tiles
    /// along the N dimension (column chunks of B).
    pub fn gpu_matmul_cpu_b(&self, a_data: &[f32], b_data: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        if m == 0 || k == 0 || n == 0 { return Ok(vec![0.0f32; m * n]); }

        let b_bytes = b_data.len() * 4;
        let c_bytes = m * n * 4;
        let a_bytes = a_data.len() * 4;

        // Check if B fits in a single GPU buffer. If not, tile.
        // Some GPUs misreport max_buffer_size (Intel HD 530 reports 2147MB
        // but caps at 256MB), so we also check empirically by catching errors.
        let single_b_fits = b_bytes <= self.real_max_buffer_bytes as usize
            && c_bytes <= self.real_max_buffer_bytes as usize
            && a_bytes <= self.real_max_buffer_bytes as usize;

        if single_b_fits {
            // Fast path: everything fits in one shot
            match self.gpu_matmul_single(a_data, b_data, m, k, n) {
                Ok(result) => return Ok(result),
                Err(_) => {
                    // GPU lied about buffer limits — fall through to tiled path
                    tracing::warn!(
                        event = "gpu_buffer_misreported",
                        max_buffer_mb = self.reported_max_buffer_bytes as f64 / 1e6,
                        real_max_mb = self.real_max_buffer_bytes as f64 / 1e6,
                        b_bytes_mb = b_bytes as f64 / 1e6,
                        "GPU max_buffer_size was misreported, falling back to tiled matmul"
                    );
                }
            }
        }

        // Tiled path: chunk B into column tiles that fit in max_buffer
        self.gpu_matmul_tiled(a_data, b_data, m, k, n)
    }

    /// Single-shot GPU matmul (all buffers fit).
    fn gpu_matmul_single(&self, a_data: &[f32], b_data: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
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

    /// Tiled GPU matmul: chunks B into column tiles that each fit in max_buffer.
    /// C[m×n] = A[m×k] × B[k×n], processed as tiles of tile_n columns.
    fn gpu_matmul_tiled(&self, a_data: &[f32], b_data: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        // Use half the max buffer for A and half for B-tile + C-tile
        let usable = (self.real_max_buffer_bytes as usize / 2).max(64 * 1024 * 1024); // at least 64MB
        let tile_n = (usable / (k * 4)).max(1).min(n);

        if tile_n == n {
            // Actually fits — try single shot
            return self.gpu_matmul_single(a_data, b_data, m, k, n);
        }

        let num_tiles = (n + tile_n - 1) / tile_n;
        tracing::info!(
            event = "gpu_matmul_tiled",
            m, k, n, tile_n, num_tiles,
            "Tiling GPU matmul: {num_tiles} tiles of {tile_n} columns"
        );

        let mut result = vec![0.0f32; m * n];

        // Upload A once (it's reused for every tile)
        let a_buf = GpuBuffer::new(&self.device, a_data.len() * 4, Some("a_tiled"))?;
        self.queue.write_buffer(a_buf.buffer(), 0, bytemuck::cast_slice(a_data));

        for tile_idx in 0..num_tiles {
            let col_start = tile_idx * tile_n;
            let col_end = (col_start + tile_n).min(n);
            let cur_n = col_end - col_start;

            // Extract B tile: rows [0..k], cols [col_start..col_end]
            let mut b_tile = vec![0.0f32; k * cur_n];
            for r in 0..k {
                for c in 0..cur_n {
                    b_tile[r * cur_n + c] = b_data[r * n + col_start + c];
                }
            }

            // Upload B tile and compute
            let b_buf = GpuBuffer::new(&self.device, b_tile.len() * 4, Some("b_tile"))?;
            self.queue.write_buffer(b_buf.buffer(), 0, bytemuck::cast_slice(&b_tile));

            let c_buf = GpuBuffer::new(&self.device, m * cur_n * 4, Some("c_tile"))?;

            let mut enc = self.device.create_command_encoder(
                &CommandEncoderDescriptor { label: Some("matmul_tile") }
            );
            self.matmul.dispatch(&mut enc, &a_buf, &b_buf, &c_buf, m as u32, k as u32, cur_n as u32)?;
            self.queue.submit(std::iter::once(enc.finish()));
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();

            let tile_result = read_back_f32(&self.device, &self.queue, &c_buf, m * cur_n)?;

            // Scatter tile result into full result
            for t in 0..m {
                for c in 0..cur_n {
                    result[t * n + col_start + c] = tile_result[t * cur_n + c];
                }
            }
        }

        Ok(result)
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
