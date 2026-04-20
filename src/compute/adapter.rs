//! GPU adapter selection with software fallback.
//!
//! Attempts to find the best hardware adapter (Vulkan > Metal > DX12 > GL).
//! If no hardware GPU is available, falls back to the wgpu software adapter
//! which runs all WGSL shaders on CPU. This ensures FerrisRes works everywhere,
//! even on devices without a GPU (headless servers, CI, RPi without drivers).
//!
//! The software adapter is slow but correct — all WGSL kernels execute faithfully.

use std::sync::Arc;
use wgpu::{Device, Queue, Features, Limits};
use crate::error::{FerrisResError, Result};

/// GPU adapter info for diagnostics.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: wgpu::Backend,
    pub is_software: bool,
    pub features: Features,
}

/// Result of adapter selection.
pub struct SelectedDevice {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub info: AdapterInfo,
}

/// Try to create a device+queue with the best available adapter.
///
/// Strategy:
/// 1. Try Vulkan (Linux, Windows, Android)
/// 2. Try Metal (macOS, iOS)
/// 3. Try DX12 (Windows)
/// 4. Try GL (fallback, limited features)
/// 5. Try software adapter (runs on CPU, always available)
///
/// Feature requests are minimal — we only ask for what we need
/// and gracefully degrade when features aren't available.
pub async fn select_device() -> Result<SelectedDevice> {
    let backends_to_try = [
        wgpu::Backends::VULKAN,
        wgpu::Backends::METAL,
        wgpu::Backends::DX12,
        wgpu::Backends::GL,
    ];

    // Phase 1: Try hardware adapters
    for &backends in &backends_to_try {
        if let Some(device) = try_adapter(backends, false).await? {
            tracing::info!(
                "Selected hardware adapter: {} ({:?})",
                device.info.name, device.info.backend
            );
            return Ok(device);
        }
    }

    // Phase 2: Try software fallback
    tracing::warn!("No hardware GPU found, falling back to software adapter");
    if let Some(device) = try_software().await? {
        return Ok(device);
    }

    Err(FerrisResError::Device(
        "No GPU adapter available (hardware or software). wgpu initialization failed.".into(),
    ))
}

/// Try to get a device from a specific backend.
async fn try_adapter(backends: wgpu::Backends, force_software: bool) -> Result<Option<SelectedDevice>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });

    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: force_software,
            compatible_surface: None,
        })
        .await
    {
        Ok(a) => a,
        Err(_) => return Ok(None),
    };

    let info = adapter.get_info();
    let is_software = info.device_type == wgpu::DeviceType::Cpu;

    // Don't select software adapters in the hardware phase
    if is_software && !force_software {
        return Ok(None);
    }

    // Request minimal features — gracefully degrade
    let mut required_features = Features::empty();
    let available = adapter.features();

    // Optional features we can use if available
    let optional_features = [
        Features::IMMEDIATES,
        Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        Features::SHADER_F16,
        Features::BUFFER_BINDING_ARRAY,
    ];

    for feature in optional_features {
        if available.contains(feature) {
            required_features |= feature;
        }
    }

    let (device, queue) = match adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("FerrisRes Device"),
            required_features,
            required_limits: Limits::downlevel_webgl2_defaults()
                .using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })
        .await
    {
        Ok(dq) => dq,
        Err(e) => {
            tracing::warn!(
                "Failed to create device on {:?} adapter '{}': {}",
                info.backend,
                info.name,
                e
            );
            return Ok(None);
        }
    };

    Ok(Some(SelectedDevice {
        device: Arc::new(device),
        queue: Arc::new(queue),
        info: AdapterInfo {
            name: info.name,
            backend: info.backend,
            is_software,
            features: required_features,
        },
    }))
}

/// Try the software fallback adapter.
async fn try_software() -> Result<Option<SelectedDevice>> {
    // The software adapter is exposed via all backends but typically GL or Vulkan
    try_adapter(wgpu::Backends::all(), true).await
}

/// Get a summary of device capabilities for logging.
pub fn device_summary(info: &AdapterInfo) -> String {
    let mut caps = Vec::new();
    if info.features.contains(Features::IMMEDIATES) {
        caps.push("immediates");
    }
    if info.features.contains(Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        caps.push("coop_matrix");
    }
    if info.features.contains(Features::SHADER_F16) {
        caps.push("f16");
    }
    if info.is_software {
        caps.push("SOFTWARE_FALLBACK");
    }
    format!(
        "Adapter: {} (backend={:?}, features=[{}])",
        info.name,
        info.backend,
        caps.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_info_creation() {
        let info = AdapterInfo {
            name: "Test GPU".into(),
            backend: wgpu::Backend::Vulkan,
            is_software: false,
            features: Features::IMMEDIATES | Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        };
        assert_eq!(info.name, "Test GPU");
        assert!(!info.is_software);
        assert!(info.features.contains(Features::IMMEDIATES));
    }

    #[test]
    fn test_device_summary() {
        let info = AdapterInfo {
            name: "Mesa Intel".into(),
            backend: wgpu::Backend::Vulkan,
            is_software: false,
            features: Features::IMMEDIATES | Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        };
        let summary = device_summary(&info);
        assert!(summary.contains("immediates"));
        assert!(summary.contains("coop_matrix"));
        assert!(!summary.contains("SOFTWARE_FALLBACK"));
    }

    #[test]
    fn test_device_summary_software() {
        let info = AdapterInfo {
            name: "Software".into(),
            backend: wgpu::Backend::Gl,
            is_software: true,
            features: Features::empty(),
        };
        let summary = device_summary(&info);
        assert!(summary.contains("SOFTWARE_FALLBACK"));
    }
}
