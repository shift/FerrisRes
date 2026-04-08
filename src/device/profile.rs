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
