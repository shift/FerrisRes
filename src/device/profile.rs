use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceProfile {
    Integrated,
    LowEnd,
    MidRange,
    HighEnd,
}

impl DeviceProfile {
    pub fn from_vram_mb(vram_mb: u64) -> Self {
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

    pub fn cache_size(&self) -> usize {
        match self {
            DeviceProfile::Integrated => 2,
            DeviceProfile::LowEnd => 4,
            DeviceProfile::MidRange => 8,
            DeviceProfile::HighEnd => 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeMode {
    FullGpu,
    Tiled,
    CpuOffload,
}
