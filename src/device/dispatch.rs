//! Profile-driven dispatch: model-size-aware CPU/GPU delegation.
//!
//! Instead of a manual `--gpu` flag, the `DispatchPlan` computes per-op
//! CPU/GPU decisions from real numbers:
//! - Model size (bytes, from file stat or weight sum)
//! - GPU VRAM (bytes, from device.limits())
//! - Max buffer size (bytes, from device.limits().max_buffer_size)
//! - Per-op weight sizes (bytes, from model config dimensions)
//!
//! No adapter name heuristics. Pure arithmetic.

use crate::device::profile::{ComputeMode, DeviceProfile};

// ---------------------------------------------------------------------------
// OpDispatch — per-operation dispatch decision
// ---------------------------------------------------------------------------

/// Where to run a particular operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpTarget {
    /// Run on CPU (memory-bound ops, or weights don't fit on GPU).
    Cpu,
    /// Run on GPU (compute-bound ops where weights fit).
    Gpu,
    /// Run on GPU in tiled/chunked mode (weights fit chunk-by-chunk).
    GpuTiled,
}

// ---------------------------------------------------------------------------
// DispatchPlan — the full dispatch strategy
// ---------------------------------------------------------------------------

/// Per-operation dispatch decisions, derived from model size and device capabilities.
///
/// Created once at startup from real measurements, then consulted by every
/// matmul/forward call.
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    /// The detected device profile (may be overridden by env var).
    pub profile: DeviceProfile,
    /// Total model weight bytes (all parameters).
    pub model_bytes: u64,
    /// Available GPU VRAM in bytes (from device limits or adapter info).
    pub vram_bytes: u64,
    /// Max single buffer size on GPU (from device.limits().max_buffer_size).
    pub max_buffer_bytes: u64,
    // Per-op decisions
    pub embed_dispatch: OpTarget,
    pub attn_qkv_dispatch: OpTarget,
    pub attn_score_dispatch: OpTarget,
    pub attn_o_dispatch: OpTarget,
    pub ffn_dispatch: OpTarget,
    pub lm_head_dispatch: OpTarget,
    pub gradient_dispatch: OpTarget,
    /// Recommended batch size.
    pub batch_size: u32,
    /// Per-sample activation memory estimate (bytes).
    pub per_sample_bytes: u64,
    /// Whether GPU is available at all.
    pub gpu_available: bool,
    /// Whether ALL model weights can be uploaded to GPU once and kept resident.
    /// True when model_bytes < vram_bytes * 0.7 (30% headroom for activations/gradients).
    /// When true, skip JIT uploads — weights stay in VRAM for the entire session.
    pub resident_mode: bool,
}

impl DispatchPlan {
    /// Create a dispatch plan from model size and device capabilities.
    ///
    /// This is a pure function — no GPU initialization, no side effects.
    /// Call this after loading the model and detecting the GPU.
    pub fn new(
        profile: DeviceProfile,
        model_bytes: u64,
        vram_bytes: u64,
        max_buffer_bytes: u64,
        hidden_dim: u64,
        num_heads: u64,
        head_dim: u64,
        intermediate_dim: u64,
        vocab_size: u64,
        seq_len: u64,
        num_kv_heads: u64,
    ) -> Self {
        let gpu_available = vram_bytes > 0 && max_buffer_bytes > 0;

        // Compute per-op weight sizes
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_weight_bytes = (hidden_dim * (q_dim + 2 * kv_dim)) * 4; // 3 projections × 4 bytes
        let o_weight_bytes = (q_dim * hidden_dim) * 4;
        let ffn_weight_bytes = (hidden_dim * intermediate_dim * 3) * 4; // gate + up + down
        let lm_head_bytes = (hidden_dim * vocab_size) * 4;
        let per_layer_bytes = qkv_weight_bytes + o_weight_bytes + ffn_weight_bytes;

        // Per-sample activation memory:
        // hidden states + attention scores (seq × seq × num_heads) + FFN intermediates
        let attn_score_bytes = seq_len * seq_len * num_heads * 4;
        let ffn_intermediate_bytes = seq_len * intermediate_dim * 4;
        let per_sample_bytes = seq_len * hidden_dim * 4 + attn_score_bytes + ffn_intermediate_bytes;

        // Dispatch decisions: send to GPU if the op's weights fit in max_buffer
        let single_weight_fits = |weight_bytes: u64| -> OpTarget {
            if !gpu_available { return OpTarget::Cpu; }
            if weight_bytes <= max_buffer_bytes { OpTarget::Gpu } else { OpTarget::Cpu }
        };

        // For large weights that don't fit in a single buffer, try tiled
        let tiled_or_cpu = |weight_bytes: u64| -> OpTarget {
            if !gpu_available { return OpTarget::Cpu; }
            if weight_bytes <= max_buffer_bytes { return OpTarget::Gpu; }
            // Tiled: can we fit a column chunk? (4096 columns × hidden_dim)
            let chunk_bytes = 4096 * hidden_dim * 4;
            if chunk_bytes <= max_buffer_bytes { OpTarget::GpuTiled } else { OpTarget::Cpu }
        };

        let embed_dispatch = OpTarget::Cpu; // Embedding is a gather, not a matmul

        let attn_qkv_dispatch = single_weight_fits(qkv_weight_bytes);
        let attn_score_dispatch = OpTarget::Cpu; // Memory-bound, not compute-bound
        let attn_o_dispatch = single_weight_fits(o_weight_bytes);
        let ffn_dispatch = single_weight_fits(ffn_weight_bytes);
        let lm_head_dispatch = tiled_or_cpu(lm_head_bytes);

        // Gradient dispatch: need 2× model for forward + backward activations
        let gradient_bytes = model_bytes * 2 + per_sample_bytes;
        let gradient_dispatch = if !gpu_available {
            OpTarget::Cpu
        } else if gradient_bytes <= vram_bytes {
            OpTarget::Gpu
        } else if per_layer_bytes * 2 <= max_buffer_bytes {
            // Can do per-layer gradient offload
            OpTarget::GpuTiled
        } else {
            OpTarget::Cpu
        };

        // Batch size: how many samples fit in VRAM after model weights?
        let available_for_activations = if gpu_available && vram_bytes > model_bytes {
            vram_bytes - model_bytes
        } else {
            0
        };
        let batch_size = if per_sample_bytes > 0 && available_for_activations > 0 {
            ((available_for_activations / per_sample_bytes).max(1) as u32)
                .min(profile.recommended_batch_size())
        } else {
            profile.recommended_batch_size()
        };

        Self {
            profile,
            model_bytes,
            vram_bytes,
            max_buffer_bytes,
            embed_dispatch,
            attn_qkv_dispatch,
            attn_score_dispatch,
            attn_o_dispatch,
            ffn_dispatch,
            lm_head_dispatch,
            gradient_dispatch,
            batch_size,
            per_sample_bytes,
            gpu_available,
            resident_mode: gpu_available && model_bytes < (vram_bytes as f64 * 0.7) as u64
                && Self::resident_ram_safe(model_bytes),
        }
    }

    /// Check if there's enough RAM for resident mode.
    /// Resident mode needs: mmap'd model + loaded weights + GPU staging buffers.
    /// That's roughly model_bytes × 2.5. If available RAM < model_bytes × 3, skip resident.
    fn resident_ram_safe(model_bytes: u64) -> bool {
        let available_bytes = Self::available_ram_bytes();
        let required = model_bytes * 3; // 3x safety margin
        if available_bytes > 0 {
            available_bytes > required
        } else {
            // Can't determine — be conservative, skip resident
            false
        }
    }

    /// Get available system RAM in bytes.
    pub fn available_ram_bytes() -> u64 {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.available_memory() * 1024 // sysinfo returns KB
    }

    /// Create a CPU-only plan (no GPU available).
    pub fn cpu_only(model_bytes: u64, hidden_dim: u64, vocab_size: u64) -> Self {
        let profile = DeviceProfile::Integrated;
        let seq_len = 32; // default
        Self::new(
            profile, model_bytes, 0, 0,
            hidden_dim, 8, hidden_dim / 8, hidden_dim * 4,
            vocab_size, seq_len, 1,
        )
    }

    /// Get the compute mode for this plan.
    pub fn compute_mode(&self) -> ComputeMode {
        self.profile.compute_mode()
    }

    /// Should we use GPU for a matmul of given weight dimensions?
    pub fn should_gpu_matmul(&self, _m: u64, k: u64, n: u64) -> OpTarget {
        let weight_bytes = k * n * 4;
        if !self.gpu_available { return OpTarget::Cpu; }
        if weight_bytes <= self.max_buffer_bytes { OpTarget::Gpu } else {
            // Try tiled
            let chunk_bytes = 4096 * k * 4;
            if chunk_bytes <= self.max_buffer_bytes { OpTarget::GpuTiled } else { OpTarget::Cpu }
        }
    }

    /// Summary for logging.
    pub fn summary(&self) -> String {
        let target_str = |t: OpTarget| match t {
            OpTarget::Cpu => "CPU",
            OpTarget::Gpu => "GPU",
            OpTarget::GpuTiled => "GPU*T",
        };
        format!(
            "DispatchPlan: profile={:?} model={:.1}GB vram={:.1}GB max_buf={:.0}MB\n  \
             embed={} qkv={} attn={} o={} ffn={} lm_head={} grad={}\n  \
             batch={} per_sample={:.1}MB gpu_available={}",
            self.profile,
            self.model_bytes as f64 / 1e9,
            self.vram_bytes as f64 / 1e9,
            self.max_buffer_bytes as f64 / 1e6,
            target_str(self.embed_dispatch),
            target_str(self.attn_qkv_dispatch),
            target_str(self.attn_score_dispatch),
            target_str(self.attn_o_dispatch),
            target_str(self.ffn_dispatch),
            target_str(self.lm_head_dispatch),
            target_str(self.gradient_dispatch),
            self.batch_size,
            self.per_sample_bytes as f64 / 1e6,
            self.gpu_available,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_plan_cpu_only() {
        let plan = DispatchPlan::cpu_only(1_000_000_000, 768, 32000);
        assert_eq!(plan.attn_qkv_dispatch, OpTarget::Cpu);
        assert_eq!(plan.lm_head_dispatch, OpTarget::Cpu);
        assert!(!plan.gpu_available);
        assert_eq!(plan.compute_mode(), ComputeMode::CpuOffload);
    }

    #[test]
    fn test_dispatch_plan_big_gpu() {
        // 24GB GPU, 10GB model (like Gemma 27B on RTX 4090)
        let plan = DispatchPlan::new(
            DeviceProfile::HighEnd,
            10_000_000_000, // 10GB model
            24_000_000_000, // 24GB VRAM
            4_000_000_000,  // 4GB max buffer
            1536, 8, 192, 6144, 262144, 32, 1,
        );
        assert!(plan.gpu_available);
        // QKV weights: 1536 × (1536 + 2*192) × 4 ≈ 11.6MB — fits easily
        assert_eq!(plan.attn_qkv_dispatch, OpTarget::Gpu);
        // Attention scores: always CPU
        assert_eq!(plan.attn_score_dispatch, OpTarget::Cpu);
        // LM head: 1536 × 262144 × 4 ≈ 1.6GB — fits in 4GB buffer
        assert_eq!(plan.lm_head_dispatch, OpTarget::Gpu);
        // FFN: 1536 × 6144 × 3 × 4 ≈ 113MB — fits
        assert_eq!(plan.ffn_dispatch, OpTarget::Gpu);
    }

    #[test]
    fn test_dispatch_plan_small_gpu() {
        // 256MB iGPU (Intel UHD 620), 10GB model — nothing fits on GPU
        let plan = DispatchPlan::new(
            DeviceProfile::Integrated,
            10_000_000_000,
            256 * 1024 * 1024, // 256MB VRAM
            256 * 1024 * 1024,  // 256MB max buffer
            1536, 8, 192, 6144, 262144, 32, 1,
        );
        // QKV weights: 11.6MB — actually fits in 256MB!
        assert_eq!(plan.attn_qkv_dispatch, OpTarget::Gpu);
        // FFN: 113MB — fits in 256MB
        assert_eq!(plan.ffn_dispatch, OpTarget::Gpu);
        // LM head: 1.6GB — doesn't fit, try tiled (4096 × 1536 × 4 = 25MB — fits)
        assert_eq!(plan.lm_head_dispatch, OpTarget::GpuTiled);
    }

    #[test]
    fn test_dispatch_plan_tiny_gpu() {
        // 64MB buffer limit — only tiny ops fit
        let plan = DispatchPlan::new(
            DeviceProfile::Integrated,
            10_000_000_000,
            64 * 1024 * 1024,
            64 * 1024 * 1024,
            1536, 8, 192, 6144, 262144, 32, 1,
        );
        // QKV: 11.6MB — fits in 64MB
        assert_eq!(plan.attn_qkv_dispatch, OpTarget::Gpu);
        // FFN: 113MB — doesn't fit
        assert_eq!(plan.ffn_dispatch, OpTarget::Cpu);
        // LM head: tiled chunk = 25MB — fits
        assert_eq!(plan.lm_head_dispatch, OpTarget::GpuTiled);
    }

    #[test]
    fn test_should_gpu_matmul() {
        let plan = DispatchPlan::new(
            DeviceProfile::MidRange,
            5_000_000_000,
            12_000_000_000,
            2_000_000_000, // 2GB max buffer
            768, 12, 64, 3072, 32000, 32, 12,
        );
        // Small matmul (768×768 = 2.4MB) — GPU
        assert_eq!(plan.should_gpu_matmul(32, 768, 768), OpTarget::Gpu);
        // Huge matmul (768×32000 = 98MB) — GPU (fits in 2GB)
        assert_eq!(plan.should_gpu_matmul(32, 768, 32000), OpTarget::Gpu);
    }

    #[test]
    fn test_batch_size_calculation() {
        // 24GB GPU, 10GB model, leaves 14GB for activations
        let plan = DispatchPlan::new(
            DeviceProfile::HighEnd,
            10_000_000_000,
            24_000_000_000,
            4_000_000_000,
            1536, 8, 192, 6144, 262144, 32, 1,
        );
        assert!(plan.batch_size >= 1);
    }

    #[test]
    fn test_batch_size_no_vram() {
        // No GPU — batch size from profile default
        let plan = DispatchPlan::cpu_only(1_000_000_000, 768, 32000);
        assert_eq!(plan.batch_size, DeviceProfile::Integrated.recommended_batch_size());
    }

    #[test]
    fn test_summary() {
        let plan = DispatchPlan::cpu_only(1_000_000_000, 768, 32000);
        let s = plan.summary();
        assert!(s.contains("DispatchPlan"));
        assert!(s.contains("CPU"));
    }

    #[test]
    fn test_op_target_equality() {
        assert_eq!(OpTarget::Cpu, OpTarget::Cpu);
        assert_ne!(OpTarget::Cpu, OpTarget::Gpu);
        assert_ne!(OpTarget::Gpu, OpTarget::GpuTiled);
    }

    #[test]
    fn test_resident_mode_t4() {
        // T4: 15GB VRAM, 4.6GB model — resident mode ON
        let plan = DispatchPlan::new(
            DeviceProfile::HighEnd,
            4_600_000_000,  // 4.6GB E2B model
            15_000_000_000, // 15GB T4 VRAM
            4_293_000_000,  // 4.3GB max buffer
            1536, 8, 256, 6144, 262144, 256, 1,
        );
        assert!(plan.resident_mode, "T4 with E2B should enable resident mode");
    }

    #[test]
    fn test_resident_mode_lowend() {
        // Intel HD 530: 256MB VRAM, 10GB model — resident mode OFF
        let plan = DispatchPlan::new(
            DeviceProfile::LowEnd,
            10_000_000_000,
            256 * 1024 * 1024,
            256 * 1024 * 1024,
            1536, 8, 256, 6144, 262144, 32, 1,
        );
        assert!(!plan.resident_mode, "LowEnd should NOT enable resident mode");
    }

    #[test]
    fn test_resident_mode_threshold() {
        // Exactly at 70% VRAM — resident mode OFF (no headroom)
        let model_bytes = 7_000_000_000u64; // 7GB
        let vram_bytes = 10_000_000_000u64; // 10GB (70% exactly)
        let plan = DispatchPlan::new(
            DeviceProfile::HighEnd,
            model_bytes, vram_bytes,
            2_000_000_000,
            1536, 8, 256, 6144, 262144, 32, 1,
        );
        // 7GB == 10GB * 0.7, so NOT less than, should be false
        assert!(!plan.resident_mode);

        // Just under — should be true
        let plan2 = DispatchPlan::new(
            DeviceProfile::HighEnd,
            6_999_999_999, vram_bytes,
            2_000_000_000,
            1536, 8, 256, 6144, 262144, 32, 1,
        );
        assert!(plan2.resident_mode);
    }
}
