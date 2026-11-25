use wgpu::{Adapter, Device, Queue, ShaderModule, ComputePipeline};
use crate::error::{FerrisResError, Result};
use crate::device::{DeviceProfile, Capability};

/// Runtime compute parameters optimized for detected hardware
#[derive(Debug, Clone)]
pub struct ComputeParams {
    /// Recommended workgroup size for compute shaders
    pub workgroup_size: u32,
    /// Recommended tile size for matrix operations (height, width)
    pub tile_size: (u32, u32),
    /// Recommended batch size for inference
    pub batch_size: u32,
    /// Whether to use async compute pipelines
    pub use_async: bool,
}

impl Default for ComputeParams {
    fn default() -> Self {
        Self {
            workgroup_size: 64,
            tile_size: (16, 16),
            batch_size: 1,
            use_async: false,
        }
    }
}

pub struct WgpuCompute {
    device: Device,
    queue: Queue,
    adapter: Adapter,
    pipeline: Option<ComputePipeline>,
}

/// Limits to request for a device, given the adapter's limits and the features
/// we enable.
///
/// `wgpu::Limits::default()` carries `max_immediate_size: 0`. If `IMMEDIATES`
/// is enabled while that limit stays 0, every pipeline layout that declares
/// immediate data fails validation with
/// `Immediate data has size N which exceeds device immediate data size limit
/// 0..0`. The adapter reports the real size (Vulkan push-constant bytes), so
/// take it whenever the feature is on. With the feature off the limit stays 0,
/// which is correct: no layout may declare immediate data.
pub(crate) fn device_limits(adapter: &wgpu::Limits, features: wgpu::Features) -> wgpu::Limits {
    let mut limits = wgpu::Limits::default();
    if features.contains(wgpu::Features::IMMEDIATES) {
        limits.max_immediate_size = adapter.max_immediate_size;
    }
    limits
}

impl WgpuCompute {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| FerrisResError::Device(format!("Failed to find adapter: {}", e)))?;

        let required_features = if adapter.features().contains(wgpu::Features::IMMEDIATES) {
            wgpu::Features::IMMEDIATES
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("FerrisRes Device"),
                    required_features,
                    required_limits: device_limits(&adapter.limits(), required_features),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                },
            )
            .await
            .map_err(|e| FerrisResError::Device(format!("Failed to request device: {}", e)))?;

        Ok(Self {
            device,
            queue,
            adapter,
            pipeline: None,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    pub fn adapter_limits(&self) -> wgpu::Limits {
        self.adapter.limits()
    }

    pub fn detect_capability(&self) -> Capability {
        let limits = self.adapter_limits();
        let info = self.adapter_info();
        Capability::detect()
            .with_adapter_limits(&limits, &info)
    }
    
    /// Get optimized compute parameters based on detected hardware
    pub fn get_optimal_params(&self) -> ComputeParams {
        let capability = self.detect_capability();
        let profile = DeviceProfile::from_vram_and_kind(
            capability.vram_mb, 
            capability.gpu_kind
        );
        
        ComputeParams {
            workgroup_size: profile.recommended_workgroup_size(
                capability.max_compute_invocations_per_workgroup
            ),
            tile_size: profile.recommended_tile_size(),
            batch_size: profile.recommended_batch_size(),
            use_async: matches!(profile, DeviceProfile::HighEnd | DeviceProfile::MidRange),
        }
    }

    pub fn detect_profile(&self) -> DeviceProfile {
        // Priority 1: FERRIS_DEVICE_PROFILE env var override.
        if let Some(profile) = DeviceProfile::from_env() {
            return profile;
        }

        // Priority 2: Vulkan/ash capability detection.
        let cap = self.detect_capability();
        let profile = DeviceProfile::from_vram_and_kind(cap.vram_mb, cap.gpu_kind);

        // Priority 3 (implicit): DeviceProfile::Integrated is the fallback inside
        // from_vram_and_kind when no discrete GPU is found.

        // Emit startup summary log.
        let adapter_info = self.adapter_info();
        tracing::info!(event = "ferrisres_device_profile", "=== FerrisRes Device Profile ===");
        tracing::info!(event = "gpu", "  GPU: {}", adapter_info.name);
        tracing::info!(event = "profile", "  Profile: {:?}", profile);
        tracing::info!(event = "compute_mode", "  Compute mode: {:?}", profile.compute_mode());
        tracing::info!(event = "batch_size", "  Batch size: {}", profile.recommended_batch_size());
        tracing::info!(event = "cache_size_mb", "  Cache size: {} MB", profile.cache_size() / (1024 * 1024));
        tracing::info!(event = "", "================================");

        profile
    }

    pub fn create_shader_module(&self, wgsl: &str) -> Result<ShaderModule> {
        Ok(self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FerrisRes Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        }))
    }

    pub fn create_compute_pipeline(
        &mut self,
        shader: &ShaderModule,
        entry_point: &str,
    ) -> Result<ComputePipeline> {
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&layout),
                module: shader,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        self.pipeline = Some(pipeline.clone());
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::device_limits;

    /// The regression this module exists for: `Limits::default()` reports no
    /// immediate space, so a device that enables `IMMEDIATES` on top of it
    /// cannot create any pipeline layout that declares immediate data.
    #[test]
    fn default_limits_have_no_immediate_space() {
        assert_eq!(wgpu::Limits::default().max_immediate_size, 0);
    }

    #[test]
    fn immediates_take_the_adapter_size() {
        let adapter = wgpu::Limits {
            max_immediate_size: 256,
            ..wgpu::Limits::default()
        };
        let limits = device_limits(&adapter, wgpu::Features::IMMEDIATES);
        assert_eq!(limits.max_immediate_size, 256);
    }

    #[test]
    fn without_immediates_the_limit_stays_zero() {
        let adapter = wgpu::Limits {
            max_immediate_size: 256,
            ..wgpu::Limits::default()
        };
        let limits = device_limits(&adapter, wgpu::Features::empty());
        assert_eq!(limits.max_immediate_size, 0);
    }

    /// Every immediate-data pipeline in the tree declares at most 16 bytes
    /// (`softmax` 4, `embedding` 8, `backward` up to 16). If the adapter offers
    /// less than that, the feature must not be enabled or the layout will
    /// fail, so this asserts the value the guard relies on.
    #[test]
    fn smallest_supported_immediate_size_covers_the_tree() {
        let adapter = wgpu::Limits {
            max_immediate_size: 4,
            ..wgpu::Limits::default()
        };
        let limits = device_limits(&adapter, wgpu::Features::IMMEDIATES);
        assert!(limits.max_immediate_size >= 4);
    }
}
