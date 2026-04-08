use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::GpuBuffer;
use crate::error::Result;

const EMBED_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<private> vocab_size: u32;
var<private> hidden_dim: u32;

@compute @workgroup_size(256)
fn embed_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let token_id = input_ids[tid];
    if (token_id >= vocab_size) {
        return;
    }
    let row_offset = token_id * hidden_dim;
    let out_offset = tid * hidden_dim;
    for (var j: u32 = 0u; j < hidden_dim; j = j + 1u) {
        output[out_offset + j] = weights[row_offset + j];
    }
}
"#;

pub struct TokenEmbedding {
    weight: GpuBuffer,
    vocab_size: usize,
    hidden_dim: usize,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
}

impl TokenEmbedding {
    pub fn new(
        device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        tracing::info!(
            "Creating TokenEmbedding: vocab_size={} hidden_dim={}",
            vocab_size, hidden_dim
        );

        let weight_bytes = vocab_size * hidden_dim * std::mem::size_of::<f32>();
        let weight = GpuBuffer::zeros(&device, weight_bytes, Some("TokenEmbedding Weight"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TokenEmbedding Shader"),
            source: wgpu::ShaderSource::Wgsl(EMBED_WGSL.into()),
        });

        let weights_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let input_ids_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let output_entry = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TokenEmbedding Bind Group Layout"),
            entries: &[weights_entry, input_ids_entry, output_entry],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TokenEmbedding Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 8,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TokenEmbedding Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("embed_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::debug!("TokenEmbedding pipeline created successfully");

        Ok(Self {
            weight,
            vocab_size,
            hidden_dim,
            pipeline,
            bind_group_layout,
            device,
            queue,
        })
    }

    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_ids: &GpuBuffer,
        output: &GpuBuffer,
        batch_size: u32,
    ) -> Result<()> {
        let workgroup_count = (batch_size + 255) / 256;

        tracing::debug!(
            "TokenEmbedding::forward batch_size={} workgroups={}",
            batch_size, workgroup_count
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TokenEmbedding Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_ids.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer().as_entire_binding(),
                },
            ],
        });

        let mut immediates = [0u8; 8];
        immediates[0..4].copy_from_slice(&(self.vocab_size as u32).to_le_bytes());
        immediates[4..8].copy_from_slice(&(self.hidden_dim as u32).to_le_bytes());

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("TokenEmbedding Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &immediates);
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }

    pub fn weight(&self) -> &GpuBuffer {
        &self.weight
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}
