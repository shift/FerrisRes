use wgpu::{Adapter, Device, Queue, ShaderModule, ComputePipeline};
use crate::error::{FerrisResError, Result};
use crate::device::{DeviceProfile, Capability};

pub struct WgpuCompute {
    device: Device,
    queue: Queue,
    adapter: Adapter,
    pipeline: Option<ComputePipeline>,
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
                    required_limits: wgpu::Limits::downlevel_defaults(),
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
        tracing::info!("=== FerrisRes Device Profile ===");
        tracing::info!("  GPU: {}", adapter_info.name);
        tracing::info!("  Profile: {:?}", profile);
        tracing::info!("  Compute mode: {:?}", profile.compute_mode());
        tracing::info!("  Batch size: {}", profile.recommended_batch_size());
        tracing::info!("  Cache size: {} MB", profile.cache_size() / (1024 * 1024));
        tracing::info!("================================");

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
