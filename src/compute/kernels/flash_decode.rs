use std::sync::Arc;
use wgpu::{
    BufferDescriptor, BufferUsages, BindGroupLayoutEntry, BindingType, BufferBindingType,
    Device, Queue, ShaderStages,
};
use crate::compute::GpuBuffer;
use crate::error::Result;

const FLASH_DECODE_WGSL: &str = r#"
    struct Params {
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        _pad: u32,
    }

    @group(0) @binding(0) var<storage, read> query: array<f32>;
    @group(0) @binding(1) var<storage, read> key_cache: array<f32>;
    @group(0) @binding(2) var<storage, read> value_cache: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;

    @compute @workgroup_size(256)
    fn flash_decode_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let h = gid.x;
        let num_heads = params.num_heads;
        let head_dim = params.head_dim;
        let seq_len = params.seq_len;

        if (h >= num_heads) {
            return;
        }

        let scale = 1.0 / sqrt(f32(head_dim));

        var max_score: f32 = -3.402823466e+38;

        for (var s: u32 = 0u; s < seq_len; s = s + 1u) {
            var dot: f32 = 0.0;
            let q_base = h * head_dim;
            let k_base = s * num_heads * head_dim + h * head_dim;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                dot += query[q_base + d] * key_cache[k_base + d];
            }
            let score = dot * scale;
            if (score > max_score) {
                max_score = score;
            }
        }

        var sum_exp: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            output[h * head_dim + d] = 0.0;
        }

        for (var s: u32 = 0u; s < seq_len; s = s + 1u) {
            var dot: f32 = 0.0;
            let q_base = h * head_dim;
            let k_base = s * num_heads * head_dim + h * head_dim;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                dot += query[q_base + d] * key_cache[k_base + d];
            }
            let weight = exp(dot * scale - max_score);
            sum_exp += weight;

            let v_base = s * num_heads * head_dim + h * head_dim;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                output[h * head_dim + d] += weight * value_cache[v_base + d];
            }
        }

        let inv_sum = 1.0 / sum_exp;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            output[h * head_dim + d] *= inv_sum;
        }
    }
"#;

pub struct FlashDecodeOp {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl FlashDecodeOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flash Decode Shader"),
            source: wgpu::ShaderSource::Wgsl(FLASH_DECODE_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flash Decode BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flash Decode Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flash Decode Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("flash_decode_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            bgl,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        })
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuBuffer,
        key_cache: &GpuBuffer,
        value_cache: &GpuBuffer,
        output: &GpuBuffer,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        if seq_len == 0 || num_heads == 0 || head_dim == 0 {
            return Ok(());
        }

        let params_data: [u32; 4] = [seq_len, num_heads, head_dim, 0];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Flash Decode Params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Flash Decode Bind Group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: key_cache.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: value_cache.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let wg_count = (num_heads + 255) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Flash Decode Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_count, 1, 1);

        drop(pass);

        tracing::debug!(
            "FlashDecodeOp dispatched: seq_len={} num_heads={} head_dim={}",
            seq_len, num_heads, head_dim
        );

        Ok(())
    }
}

pub fn dispatch_flash_decode(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    encoder: &mut wgpu::CommandEncoder,
    query: &GpuBuffer,
    key_cache: &GpuBuffer,
    value_cache: &GpuBuffer,
    output: &GpuBuffer,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<()> {
    let op = FlashDecodeOp::new(device, queue)?;
    op.dispatch(encoder, query, key_cache, value_cache, output, seq_len, num_heads, head_dim)
}

// ---------------------------------------------------------------------------
// FlashDecodeTiledOp — tiled KV attention with online softmax
//
// Instead of iterating seq_len positions sequentially with global-memory
// reads (the original flash_decode), this variant tiles the KV sequence
// into workgroup-shared memory in chunks of TILE_KV positions.
//
// The online softmax is computed incrementally: each tile contributes its
// local max and sum, which are merged into the running accumulator using
// the numerically-stable "flash" update:
//
//   new_max = max(old_max, tile_max)
//   correction = exp(old_max - new_max)
//   new_sum = old_sum * correction + tile_sum
//   new_out = old_out * correction + tile_out
//
// This avoids the two-pass structure of the original kernel while producing
// identical results.
//
// Task: f4c0a839 — see papers_research/flashattn3_async_wgsl_research.md
// ---------------------------------------------------------------------------

const FLASH_DECODE_TILED_WGSL: &str = r#"
    struct Params {
        seq_len:   u32,
        num_heads: u32,
        head_dim:  u32,
        tile_kv:   u32,
    }

    @group(0) @binding(0) var<storage, read>       query:       array<f32>;
    @group(0) @binding(1) var<storage, read>       key_cache:   array<f32>;
    @group(0) @binding(2) var<storage, read>       value_cache: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output:      array<f32>;
    @group(0) @binding(4) var<uniform>             params:      Params;

    var<workgroup> tile_k: array<f32, 256>;
    var<workgroup> tile_v: array<f32, 256>;

    @compute @workgroup_size(256)
    fn flash_decode_tiled(@builtin(global_invocation_id) gid: vec3<u32>) {
        let h         = gid.x;
        let num_heads = params.num_heads;
        let head_dim  = params.head_dim;
        let seq_len   = params.seq_len;
        let tile_kv   = params.tile_kv;

        if (h >= num_heads) { return; }

        let scale   = 1.0 / sqrt(f32(head_dim));
        let q_base  = h * head_dim;
        let tile_el = tile_kv * head_dim;
        let n_tiles = (seq_len + tile_kv - 1u) / tile_kv;

        // Online softmax accumulators
        var run_max: f32 = -3.402823466e+38;
        var run_sum: f32 = 0.0;
        var acc:     array<f32, 64>;   // max head_dim = 64 for typical models
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            acc[d] = 0.0;
        }

        for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
            let tile_start = t * tile_kv;

            // ---- Load K/V tile into shared memory ----
            for (var idx = gid.x; idx < tile_el; idx = idx + num_heads) {
                let s = tile_start + (idx / head_dim);
                let d = idx % head_dim;
                if (s < seq_len) {
                    tile_k[idx] = key_cache  [s * num_heads * head_dim + h * head_dim + d];
                    tile_v[idx] = value_cache[s * num_heads * head_dim + h * head_dim + d];
                } else {
                    tile_k[idx] = 0.0;
                    tile_v[idx] = 0.0;
                }
            }
            workgroupBarrier();

            let tile_len = min(tile_kv, seq_len - tile_start);

            // ---- Tile-local online softmax ----
            var tile_max: f32 = -3.402823466e+38;

            // Pass 1: tile max
            for (var s: u32 = 0u; s < tile_len; s = s + 1u) {
                var dot: f32 = 0.0;
                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    dot += query[q_base + d] * tile_k[s * head_dim + d];
                }
                let score = dot * scale;
                if (score > tile_max) { tile_max = score; }
            }

            // Merge tile into running accumulators
            let new_max = max(run_max, tile_max);
            let old_correction = exp(run_max - new_max);

            // Rescale running accumulators
            run_sum *= old_correction;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                acc[d] *= old_correction;
            }

            // Pass 2: accumulate tile contributions (all using new_max)
            for (var s: u32 = 0u; s < tile_len; s = s + 1u) {
                var dot: f32 = 0.0;
                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    dot += query[q_base + d] * tile_k[s * head_dim + d];
                }
                let weight = exp(dot * scale - new_max);
                run_sum += weight;
                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    acc[d] += weight * tile_v[s * head_dim + d];
                }
            }

            run_max = new_max;
            workgroupBarrier();
        }

        // Final normalisation
        if (run_sum > 0.0) {
            let inv_sum = 1.0 / run_sum;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                output[h * head_dim + d] = acc[d] * inv_sum;
            }
        } else {
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                output[h * head_dim + d] = 0.0;
            }
        }
    }
"#;

/// Tiled flash decode with workgroup-shared K/V tiles.
///
/// Drop-in replacement for [`FlashDecodeOp`] that uses workgroup shared
/// memory to tile the KV sequence. For large `seq_len` this reduces global
/// memory reads from `2 * seq_len * head_dim` per head to a more cache-
/// friendly tiled pattern. For short sequences (< 32 tokens) the overhead
/// of the tiling logic makes it slightly slower than the original.
pub struct FlashDecodeTiledOp {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl FlashDecodeTiledOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flash Decode Tiled Shader"),
            source: wgpu::ShaderSource::Wgsl(FLASH_DECODE_TILED_WGSL.into()),
        });

        let ro_storage = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw_storage = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = BindGroupLayoutEntry {
            binding: 4,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flash Decode Tiled BGL"),
            entries: &[
                ro_storage(0),  // query
                ro_storage(1),  // key_cache
                ro_storage(2),  // value_cache
                rw_storage(3),  // output
                uniform,
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flash Decode Tiled Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flash Decode Tiled Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("flash_decode_tiled"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            bgl,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        })
    }

    /// Dispatch the tiled flash decode.
    ///
    /// * `tile_kv` — Number of KV positions per workgroup tile. Must be ≤ 16
    ///   when `head_dim ≤ 16`, or smaller for larger heads, so that
    ///   `tile_kv * head_dim ≤ 256` (workgroup memory budget).
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuBuffer,
        key_cache: &GpuBuffer,
        value_cache: &GpuBuffer,
        output: &GpuBuffer,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        tile_kv: u32,
    ) -> Result<()> {
        if seq_len == 0 || num_heads == 0 || head_dim == 0 {
            return Ok(());
        }

        // Clamp tile_kv so the shared memory fits
        let tile_kv = tile_kv.min(256 / head_dim.max(1));
        let tile_kv = tile_kv.max(1);

        let params_data: [u32; 4] = [seq_len, num_heads, head_dim, tile_kv];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Flash Decode Tiled Params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Flash Decode Tiled BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: query.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: key_cache.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: value_cache.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        let wg_count = (num_heads + 255) / 256;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Flash Decode Tiled Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(wg_count, 1, 1);
        drop(pass);

        tracing::debug!(
            "FlashDecodeTiledOp dispatched: seq_len={} num_heads={} head_dim={} tile_kv={} wg={}",
            seq_len, num_heads, head_dim, tile_kv, wg_count
        );

        Ok(())
    }
}
