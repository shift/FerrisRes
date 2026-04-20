use serde::{Deserialize, Serialize};
use super::capability::GpuKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceProfile {
    Integrated,
    LowEnd,
    MidRange,
    HighEnd,
}

impl DeviceProfile {
    /// Read `FERRIS_DEVICE_PROFILE` env var and parse it into a `DeviceProfile`.
    /// Returns `None` if the env var is not set or contains an unrecognised value.
    /// Emits a warning when the override is active.
    pub fn from_env() -> Option<DeviceProfile> {
        let val = std::env::var("FERRIS_DEVICE_PROFILE").ok()?;
        let profile = match val.to_ascii_lowercase().as_str() {
            "integrated" => DeviceProfile::Integrated,
            "lowend" | "low_end" | "low-end" => DeviceProfile::LowEnd,
            "midrange" | "mid_range" | "mid-range" => DeviceProfile::MidRange,
            "highend" | "high_end" | "high-end" => DeviceProfile::HighEnd,
            other => {
                tracing::warn!(
                    value = other,
                    "FERRIS_DEVICE_PROFILE has unrecognised value; ignoring override"
                );
                return None;
            }
        };
        tracing::warn!(
            profile = ?profile,
            "FERRIS_DEVICE_PROFILE override is active — hardware auto-detection bypassed"
        );
        Some(profile)
    }

    pub fn from_vram_and_kind(vram_mb: u64, gpu_kind: GpuKind) -> Self {
        // When the integrated_gpu_profile feature is enabled, always return Integrated
        // regardless of detected hardware capabilities.
        #[cfg(feature = "integrated_gpu_profile")]
        {
            return DeviceProfile::Integrated;
        }
        #[cfg(not(feature = "integrated_gpu_profile"))]
        match gpu_kind {
            GpuKind::Integrated => DeviceProfile::Integrated,
            GpuKind::Other => DeviceProfile::Integrated,
            GpuKind::Discrete => {
                if vram_mb < 4000 {
                    DeviceProfile::LowEnd
                } else if vram_mb < 8000 {
                    DeviceProfile::MidRange
                } else {
                    DeviceProfile::HighEnd
                }
            }
        }
    }

    pub fn from_vram_mb(vram_mb: u64) -> Self {
        // When the integrated_gpu_profile feature is enabled, always return Integrated.
        #[cfg(feature = "integrated_gpu_profile")]
        {
            return DeviceProfile::Integrated;
        }
        #[cfg(not(feature = "integrated_gpu_profile"))]
        if vram_mb < 4000 {
            DeviceProfile::Integrated
        } else if vram_mb < 8000 {
            DeviceProfile::LowEnd
        } else if vram_mb < 16000 {
            DeviceProfile::MidRange
        } else {
            DeviceProfile::HighEnd
        }
    }

    pub fn compute_mode(&self) -> ComputeMode {
        match self {
            DeviceProfile::Integrated => ComputeMode::CpuOffload,
            DeviceProfile::LowEnd => ComputeMode::Tiled,
            DeviceProfile::MidRange | DeviceProfile::HighEnd => ComputeMode::FullGpu,
        }
    }

    pub fn recommended_batch_size(&self) -> u32 {
        match self {
            DeviceProfile::Integrated => 1,
            DeviceProfile::LowEnd => 2,
            DeviceProfile::MidRange => 4,
            DeviceProfile::HighEnd => 8,
        }
    }

    /// Returns the KV-cache size in bytes.
    pub fn cache_size(&self) -> usize {
        match self {
            DeviceProfile::Integrated => 2 * 1024 * 1024 * 1024,
            DeviceProfile::LowEnd => 4 * 1024 * 1024 * 1024,
            DeviceProfile::MidRange => 8 * 1024 * 1024 * 1024,
            DeviceProfile::HighEnd => 16 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeMode {
    FullGpu,
    Tiled,
    CpuOffload,
}

impl DeviceProfile {
    /// Get recommended workgroup size for compute shaders
    /// Based on GPU capability and profile
    pub fn recommended_workgroup_size(&self, max_invocations: u32) -> u32 {
        // Use a power of 2 that fits within hardware limits
        // Smaller workgroups = more flexibility for different GPU architectures
        match self {
            DeviceProfile::Integrated => {
                // Integrated GPUs benefit from smaller workgroups
                64.min(max_invocations)
            }
            DeviceProfile::LowEnd => {
                64.min(max_invocations)
            }
            DeviceProfile::MidRange => {
                // Mid-range GPUs can handle larger workgroups efficiently
                128.min(max_invocations)
            }
            DeviceProfile::HighEnd => {
                // High-end GPUs benefit from maximum utilization
                256.min(max_invocations)
            }
        }
    }
    
    /// Get recommended tile size for tiled matrix multiplication
    pub fn recommended_tile_size(&self) -> (u32, u32) {
        match self {
            DeviceProfile::Integrated => (8, 8),
            DeviceProfile::LowEnd => (16, 16),
            DeviceProfile::MidRange => (32, 32),
            DeviceProfile::HighEnd => (64, 64),
        }
    }
    
    /// Get recommended optimizer strategy for this device profile.
    ///
    /// - Integrated / LowEnd → SCALE (12.1 MB state, no SVD, runs on RPi)
    /// - MidRange / HighEnd   → AdaMeM (181.6 MB state, SVD every 200 steps)
    pub fn optimizer_hint(&self) -> crate::training::optimizer::OptimizerHint {
        match self {
            DeviceProfile::Integrated | DeviceProfile::LowEnd => {
                crate::training::optimizer::OptimizerHint::Scale
            }
            DeviceProfile::MidRange | DeviceProfile::HighEnd => {
                crate::training::optimizer::OptimizerHint::AdaMeM { rank: 8 }
            }
        }
    }

    /// Get memory transfer optimization hints
    pub fn memory_transfer_hint(&self) -> MemoryTransferHint {
        match self {
            DeviceProfile::Integrated => MemoryTransferHint::PreferCoalesced,
            DeviceProfile::LowEnd => MemoryTransferHint::PreferCoalesced,
            DeviceProfile::MidRange => MemoryTransferHint::Balanced,
            DeviceProfile::HighEnd => MemoryTransferHint::PreferAligned,
        }
    }
}

/// Memory transfer optimization hints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTransferHint {
    /// Prioritize memory coalescing over alignment
    PreferCoalesced,
    /// Balanced between coalescing and alignment
    Balanced,
    /// Prioritize memory alignment for bandwidth
    PreferAligned,
}
