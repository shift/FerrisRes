use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::GpuBuffer;
use crate::error::Result;

const EMBED_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    vocab_size: u32,
    hidden_dim: u32,
    batch_size: u32,
    _padding: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn embed_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.batch_size) { return; }
    let token_id = input_ids[tid];
    let out_offset = tid * params.hidden_dim;
    if (token_id >= params.vocab_size) {
        for (var j = 0u; j < params.hidden_dim; j = j + 1u) {
            output[out_offset + j] = 0.0;
        }
        return;
    }
    let row_offset = token_id * params.hidden_dim;
    for (var j = 0u; j < params.hidden_dim; j = j + 1u) {
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
    queue: Arc<Queue>,
}

impl TokenEmbedding {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        tracing::info!(
            "Creating TokenEmbedding: vocab_size={} hidden_dim={}",
            vocab_size, hidden_dim
        );

        let elements = vocab_size.checked_mul(hidden_dim).filter(|&n| n > 0 && n <= u32::MAX as usize)
            .ok_or_else(|| crate::error::FerrisResError::Shape("invalid embedding dimensions".into()))?;
        let weight_bytes = elements.checked_mul(4).filter(|&n|
            n as u64 <= device.limits().max_storage_buffer_binding_size
                && n as u64 <= device.limits().max_buffer_size)
            .ok_or_else(|| crate::error::FerrisResError::Shape("embedding table exceeds device limits".into()))?;
        let weight = GpuBuffer::zeros(&device, &queue, weight_bytes, Some("TokenEmbedding Weight"))?;

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

        let params_entry = wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TokenEmbedding Bind Group Layout"),
            entries: &[weights_entry, input_ids_entry, output_entry, params_entry],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TokenEmbedding Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TokenEmbedding Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("embed_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::debug!(event = "tokenembedding_pipeline_created_successfully", "TokenEmbedding pipeline created successfully");

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

    /// Gather rows; out-of-vocabulary IDs produce zero rows. Zero batch is a no-op.
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_ids: &GpuBuffer,
        output: &GpuBuffer,
        batch_size: u32,
    ) -> Result<()> {
        if batch_size == 0 { return Ok(()); }
        let elements = (batch_size as usize).checked_mul(self.hidden_dim)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| crate::error::FerrisResError::Shape("embedding output index overflow".into()))?;
        let bytes = elements.checked_mul(4)
            .ok_or_else(|| crate::error::FerrisResError::Shape("embedding output size overflow".into()))?;
        if (input_ids.size() as u64) < u64::from(batch_size) * 4 || output.size() < bytes {
            return Err(crate::error::FerrisResError::Shape("embedding input/output buffer too small".into()));
        }
        let workgroup_count = batch_size.div_ceil(256);
        if workgroup_count > self.device.limits().max_compute_workgroups_per_dimension {
            return Err(crate::error::FerrisResError::Shape("embedding dispatch exceeds device limits".into()));
        }
        let params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TokenEmbedding Params"), size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params, 0, bytemuck::cast_slice(&[
            self.vocab_size as u32, self.hidden_dim as u32, batch_size, 0,
        ]));

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
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("TokenEmbedding Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
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
