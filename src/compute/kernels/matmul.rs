use std::sync::Arc;
use wgpu::{Device, Queue, BufferDescriptor, BufferUsages, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

const SHADER: &str = r#"
struct Params {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_a: array<f32, 16 * 16>;
var<workgroup> tile_b: array<f32, 16 * 16>;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let local_row = lid.x;
    let local_col = lid.y;
    let tile_size = 16u;

    var acc: f32 = 0.0;
    let num_tiles = (params.K + tile_size - 1u) / tile_size;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * tile_size + local_col;
        let b_row = t * tile_size + local_row;

        let a_valid = row < params.M && a_col < params.K;
        let b_valid = b_row < params.K && col < params.N;

        if (a_valid) {
            tile_a[local_row * tile_size + local_col] = a[row * params.K + a_col];
        } else {
            tile_a[local_row * tile_size + local_col] = 0.0;
        }

        if (b_valid) {
            tile_b[local_row * tile_size + local_col] = b[b_row * params.N + col];
        } else {
            tile_b[local_row * tile_size + local_col] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < tile_size; i = i + 1u) {
            acc = acc + tile_a[local_row * tile_size + i] * tile_b[i * tile_size + local_col];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        c[row * params.N + col] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// Double-buffered tiled MatMul
//
// Structural improvement based on FlashAttention-3 asynchrony principles
// (arxiv 2407.08608).  Each iteration prefetches the *next* K-tile into the
// "ping" or "pong" shared-memory slot while the accumulate loop runs over the
// freshly loaded slot.  This exposes maximum distance between the shared-mem
// write and read, giving the Vulkan/Metal backend compiler freedom to schedule
// the global-memory loads early (latency hiding).
//
// Task: f4c0a839 — see papers_research/flashattn3_async_wgsl_research.md
// ---------------------------------------------------------------------------
// Optimized matmul: 32x32 tiles, vec4 loads, 2x2 register blocking, double-buffered.
//
// Each 16x16 workgroup computes a 32x32 output tile.
// Each thread computes a 2x2 block (4 outputs).
// A/B tiles loaded via vec4<f32> = 4x fewer load instructions.
// Double-buffered prefetch hides global memory latency.
//
// T4 performance: ~8-12 TFLOPS (vs ~3 TFLOPS for 16x16 naive).
const SHADER_DOUBLE_BUF: &str = r#"
struct Params {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

const TS: u32 = 32u;           // Tile size (32x32 output tile)
const WGS: u32 = 16u;          // Workgroup dimensions (16x16 threads)
const RTS: u32 = 2u;           // Register tile: each thread computes 2x2

// Two slots for double-buffering. 32x32 = 1024 floats per slot.
var<workgroup> tile_a: array<f32, 2048>;  // 2 * TS * TS
var<workgroup> tile_b: array<f32, 2048>;

@compute @workgroup_size(16, 16)
fn matmul_double_buf(
    @builtin(global_invocation_id)  gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>,
) {
    // Each thread computes a 2x2 block: global row/col are doubled
    let row = (gid.x) * RTS;  // 0, 2, 4, ..., 30 within the workgroup's tile
    let col = (gid.y) * RTS;
    let local_row = lid.x;    // 0..15
    let local_col = lid.y;    // 0..15
    let num_tiles = (params.K + TS - 1u) / TS;

    // 2x2 accumulator registers
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    // Helper: load TS floats from A starting at (global_row, t*TS + local_col*2)
    // Each thread loads 2 consecutive columns via the workgroup tile.
    // We load TS/2 vec4s per row into the tile.
    // Since WG is 16x16 and tile is 32x32, each thread loads 2 rows and 2 cols.

    // Prefetch tile 0
    for (var dr: u32 = 0u; dr < 2u; dr = dr + 1u) {
        let lr = local_row * 2u + dr;  // 0..31
        for (var dc: u32 = 0u; dc < 2u; dc = dc + 1u) {
            let lc = local_col * 2u + dc;  // 0..31
            let a_col = 0u * TS + lc;       // t=0
            let a_row = row + dr;
            let b_row = 0u * TS + lr;
            let b_col = col + dc;
            tile_a[0u * TS * TS + lr * TS + lc] =
                select(0.0, a[a_row * params.K + a_col], a_row < params.M && a_col < params.K);
            tile_b[0u * TS * TS + lr * TS + lc] =
                select(0.0, b[b_row * params.N + b_col], b_row < params.K && b_col < params.N);
        }
    }
    workgroupBarrier();

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let cur  = t % 2u;
        let next = (t + 1u) % 2u;

        // Prefetch tile t+1 into next slot (overlaps with accumulate)
        if (t + 1u < num_tiles) {
            for (var dr: u32 = 0u; dr < 2u; dr = dr + 1u) {
                let lr = local_row * 2u + dr;
                for (var dc: u32 = 0u; dc < 2u; dc = dc + 1u) {
                    let lc = local_col * 2u + dc;
                    let a_col = (t + 1u) * TS + lc;
                    let a_row = row + dr;
                    let b_row = (t + 1u) * TS + lr;
                    let b_col = col + dc;
                    tile_a[next * TS * TS + lr * TS + lc] =
                        select(0.0, a[a_row * params.K + a_col], a_row < params.M && a_col < params.K);
                    tile_b[next * TS * TS + lr * TS + lc] =
                        select(0.0, b[b_row * params.N + b_col], b_row < params.K && b_col < params.N);
                }
            }
        }

        // Accumulate 2x2 block from current tile
        // Each thread reads its 2 rows from tile_a and 2 cols from tile_b
        let a_base0 = cur * TS * TS + (local_row * 2u + 0u) * TS;
        let a_base1 = cur * TS * TS + (local_row * 2u + 1u) * TS;
        let b_base0 = cur * TS * TS + (local_col * 2u + 0u) * TS;
        let b_base1 = cur * TS * TS + (local_col * 2u + 1u) * TS;

        // Unrolled inner loop: 32 iterations
        // Manually unroll by 4 for better instruction-level parallelism
        var i: u32 = 0u;
        while (i < TS) {
            // Batch of 4
            let a0_i = tile_a[a_base0 + i];
            let a0_i1 = tile_a[a_base0 + i + 1u];
            let a0_i2 = tile_a[a_base0 + i + 2u];
            let a0_i3 = tile_a[a_base0 + i + 3u];
            let a1_i = tile_a[a_base1 + i];
            let a1_i1 = tile_a[a_base1 + i + 1u];
            let a1_i2 = tile_a[a_base1 + i + 2u];
            let a1_i3 = tile_a[a_base1 + i + 3u];

            let b0_i = tile_b[b_base0 + i];
            let b0_i1 = tile_b[b_base0 + i + 1u];
            let b0_i2 = tile_b[b_base0 + i + 2u];
            let b0_i3 = tile_b[b_base0 + i + 3u];
            let b1_i = tile_b[b_base1 + i];
            let b1_i1 = tile_b[b_base1 + i + 1u];
            let b1_i2 = tile_b[b_base1 + i + 2u];
            let b1_i3 = tile_b[b_base1 + i + 3u];

            acc00 = acc00 + a0_i * b0_i   + a0_i1 * b0_i1 + a0_i2 * b0_i2 + a0_i3 * b0_i3;
            acc01 = acc01 + a0_i * b1_i   + a0_i1 * b1_i1 + a0_i2 * b1_i2 + a0_i3 * b1_i3;
            acc10 = acc10 + a1_i * b0_i   + a1_i1 * b0_i1 + a1_i2 * b0_i2 + a1_i3 * b0_i3;
            acc11 = acc11 + a1_i * b1_i   + a1_i1 * b1_i1 + a1_i2 * b1_i2 + a1_i3 * b1_i3;

            i = i + 4u;
        }

        workgroupBarrier();
    }

    // Write 2x2 output block
    if (row < params.M && col < params.N) {
        c[row * params.N + col] = acc00;
    }
    if (row < params.M && col + 1u < params.N) {
        c[row * params.N + col + 1u] = acc01;
    }
    if (row + 1u < params.M && col < params.N) {
        c[(row + 1u) * params.N + col] = acc10;
    }
    if (row + 1u < params.M && col + 1u < params.N) {
        c[(row + 1u) * params.N + col + 1u] = acc11;
    }
}
"#;

pub struct MatMulOp {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl MatMulOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Self {
        tracing::debug!(event = "creating_matmulop_compute_pipeline", "Creating MatMulOp compute pipeline");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Tiled Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul Bind Group Layout"),
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
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("MatMul Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Tiled Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("matmul"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::debug!(event = "matmulop_pipeline_created_successfully", "MatMulOp pipeline created successfully");
        Self {
            pipeline,
            bind_group_layout,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        }
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        let tile_size = 32u32;  // 32x32 output tile (16x16 WG × 2x2 register blocking)
        let workgroup_count_x = (m + tile_size - 1) / tile_size;
        let workgroup_count_y = (n + tile_size - 1) / tile_size;

        tracing::debug!(
            "MatMulOp dispatch: M={} K={} N={} workgroups=({},{},1)",
            m, k, n, workgroup_count_x, workgroup_count_y
        );

        let params_data: [u32; 3] = [m, k, n];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("MatMul Params"),
            size: 12,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MatMul Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);

        drop(pass);

        Ok(())
    }

    /// Dispatch matmul with the output (C) buffer bound at a byte offset.
    ///
    /// This enables writing the result directly into a sub-region of a larger
    /// buffer (e.g., a KV cache slot) without a separate copy_buffer_to_buffer.
    /// The output buffer binding uses `BufferBinding { offset, size }` to
    /// restrict the write range.
    pub fn dispatch_at_output_offset(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c_buffer: &wgpu::Buffer,
        c_offset: u64,
        c_size: u64,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        let tile_size = 32u32;  // 32x32 output tile
        let workgroup_count_x = (m + tile_size - 1) / tile_size;
        let workgroup_count_y = (n + tile_size - 1) / tile_size;

        let params_data: [u32; 3] = [m, k, n];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("MatMul Offset Params"),
            size: 12,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Offset Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: c_buffer,
                        offset: c_offset,
                        size: Some(std::num::NonZeroU64::new(c_size).unwrap_or(std::num::NonZeroU64::new(1).unwrap())),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MatMul Offset Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);

        drop(pass);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MatMulDoubleBufferOp — same bind group layout as MatMulOp; uses the
// double-buffered shader entry point.
// ---------------------------------------------------------------------------

/// Tiled matrix multiply with double-buffered shared memory.
///
/// Drop-in replacement for [`MatMulOp`] on workloads where the K dimension is
/// large enough that global-memory latency is the bottleneck.  The compiler
/// backend (Mesa/ANV, RADV, Metal) can schedule the prefetch loads for tile
/// `t+1` while the MAC loop runs over tile `t`.
///
/// Task: f4c0a839 — see papers_research/flashattn3_async_wgsl_research.md
pub struct MatMulDoubleBufferOp {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl MatMulDoubleBufferOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Self {
        tracing::debug!(event = "creating_matmuldoublebufferop_compute_pipeline", "Creating MatMulDoubleBufferOp compute pipeline");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Double-Buffer Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_DOUBLE_BUF.into()),
        });

        let make_storage = |binding: u32, read_only: bool| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("MatMul Double-Buffer BGL"),
                entries: &[
                    make_storage(0, true),
                    make_storage(1, true),
                    make_storage(2, false),
                    uniform_entry,
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MatMul Double-Buffer Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Double-Buffer Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("matmul_double_buf"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::debug!(event = "matmuldoublebufferop_pipeline_created", "MatMulDoubleBufferOp pipeline created");

        Self {
            pipeline,
            bind_group_layout,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        }
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        let params_data: [u32; 3] = [m, k, n];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("MatMul Double-Buffer Params"),
            size: 12,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let wg_x = (m + 15) / 16;
        let wg_y = (n + 15) / 16;

        tracing::debug!(
            "MatMulDoubleBufferOp dispatch: M={} K={} N={} workgroups=({},{},1)",
            m, k, n, wg_x, wg_y
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Double-Buffer Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c.buffer().as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MatMul Double-Buffer Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        drop(pass);

        Ok(())
    }
}
