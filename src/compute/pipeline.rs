use wgpu::{Device, Queue, ShaderModule, ComputePipeline};
use crate::error::{FerrisResError, Result};

pub struct WgpuCompute {
    device: Device,
    queue: Queue,
    pipeline: Option<ComputePipeline>,
}

impl WgpuCompute {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| FerrisResError::Device("No GPU adapter found".into()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("FerrisRes Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: None,
                    experimental_features: None,
                },
                None,
            )
            .await
            .map_err(|e| FerrisResError::Device(format!("Failed to request device: {}", e)))?;

        Ok(Self {
            device,
            queue,
            pipeline: None,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
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
