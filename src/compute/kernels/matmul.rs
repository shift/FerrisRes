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
// Optimized matmul: 32x32 tiles, 2x2 register blocking, double-buffered.
//
// Each 16x16 workgroup computes a 32x32 output tile.
// Each thread computes a 2x2 block (4 outputs).
// Double-buffered prefetch hides global memory latency.
//
// Task f4c0a839 reported ~8-12 TFLOPS on a T4 (vs ~3 TFLOPS for the 16x16
// naive kernel). That was not re-measured when this kernel was wired into
// `dispatch`; only numerical equivalence was.
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
        // tile_b holds B[k][n] at [k*TS + n]. The two output columns this
        // thread owns are local_col*2 and +1, so the n offset is constant and
        // the k stride is TS. Reading [n*TS + k] instead transposes B, which is
        // what made A·B = [[1,2],[3,4]]·[[5,6],[7,8]] return 17 rather than 19.
        let b_col0 = cur * TS * TS + local_col * 2u;
        let b_col1 = b_col0 + 1u;

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

            let b0_i = tile_b[b_col0 + i * TS];
            let b0_i1 = tile_b[b_col0 + (i + 1u) * TS];
            let b0_i2 = tile_b[b_col0 + (i + 2u) * TS];
            let b0_i3 = tile_b[b_col0 + (i + 3u) * TS];
            let b1_i = tile_b[b_col1 + i * TS];
            let b1_i1 = tile_b[b_col1 + (i + 1u) * TS];
            let b1_i2 = tile_b[b_col1 + (i + 2u) * TS];
            let b1_i3 = tile_b[b_col1 + (i + 3u) * TS];

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

/// MatMul with transposed B: C = A × B^T where A is [M, K] and B is [K, N] (stored row-major).
/// Used for backward pass: dL/dA = dC × B^T where dC=[M,N], B=[K,N] → result [M,K]
const MATMUL_TRANSPOSE_B_WGSL: &str = r#"
struct Params {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;  // [M, K]
@group(0) @binding(1) var<storage, read> b: array<f32>;  // [K, N] — will be read transposed
@group(0) @binding(2) var<storage, read_write> c: array<f32>;  // [M, K]
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_a: array<f32, 16 * 16>;
var<workgroup> tile_b: array<f32, 16 * 16>;

@compute @workgroup_size(16, 16)
fn matmul_tb(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let local_row = lid.x;
    let local_col = lid.y;
    let tile_size = 16u;

    var acc: f32 = 0.0;
    // We tile over the N dimension (the shared dimension for A×B^T)
    let num_tiles = (params.N + tile_size - 1u) / tile_size;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * tile_size + local_col;  // index into N for A... wait.
        // A is [M, K], we want C[m,k] = Σ_n A[m,n] * B[k,n]
        // Actually, we need to rethink: dL/dA[m,k] = Σ_n dC[m,n] * B[k,n]
        // So A = dC [M,N], B = weights [K,N], result C = dL/dA [M,K]
        // The shared dimension is N.
        // A_tile loads A[m, t*tile: t*tile+tile] = dC[m, n_segment]
        // B_tile loads B[k, t*tile: t*tile+tile] = weights[k, n_segment]

        // Tile A: reads from a[row * N + (t*tile + local_col)]  — A is [M, N]
        let a_col_idx = t * tile_size + local_col;
        let a_valid = row < params.M && a_col_idx < params.N;

        // Tile B: reads from b[col * N + (t*tile + local_row)]  — B is [K, N], row=col, col=t_segment
        let b_col_idx = t * tile_size + local_row;
        let b_valid = col < params.K && b_col_idx < params.N;

        // Note: A is [M,N] so tile_a should be loaded with A's row stride = N
        if (a_valid) {
            tile_a[local_row * tile_size + local_col] = a[row * params.N + a_col_idx];
        } else {
            tile_a[local_row * tile_size + local_col] = 0.0;
        }

        // B is [K,N] — we read B[col, b_col_idx] = B[col*N + b_col_idx]
        if (b_valid) {
            tile_b[local_row * tile_size + local_col] = b[col * params.N + b_col_idx];
        } else {
            tile_b[local_row * tile_size + local_col] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < tile_size; i = i + 1u) {
            acc = acc + tile_a[local_row * tile_size + i] * tile_b[local_col * tile_size + i];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.K) {
        c[row * params.K + col] = acc;
    }
}
"#;

/// MatMul with transposed A: C = A^T × B where A is [M, K] and B is [M, N] (both row-major).
/// Used for backward pass: dL/dB = A^T × dC where A=[M,K], dC=[M,N] → result [K,N]
const MATMUL_TRANSPOSE_A_WGSL: &str = r#"
struct Params {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;  // [M, K] — will be read transposed
@group(0) @binding(1) var<storage, read> b: array<f32>;  // [M, N]
@group(0) @binding(2) var<storage, read_write> c: array<f32>;  // [K, N]
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_a: array<f32, 16 * 16>;
var<workgroup> tile_b: array<f32, 16 * 16>;

@compute @workgroup_size(16, 16)
fn matmul_ta(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;  // output row = K dim
    let col = gid.y;  // output col = N dim
    let local_row = lid.x;
    let local_col = lid.y;
    let tile_size = 16u;

    var acc: f32 = 0.0;
    // C[k,n] = Σ_m A[m,k] * B[m,n] = Σ_m A^T[k,m] * B[m,n]
    // Shared dimension is M
    let num_tiles = (params.M + tile_size - 1u) / tile_size;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // A_tile: A[m_segment, k] = A[(t*tile+local_col)*K + row]  — transposed read
        let a_row_idx = t * tile_size + local_col;
        let a_valid = row < params.K && a_row_idx < params.M;

        // B_tile: B[m_segment, n] = B[(t*tile+local_row)*N + col]
        let b_row_idx = t * tile_size + local_row;
        let b_valid = b_row_idx < params.M && col < params.N;

        if (a_valid) {
            tile_a[local_row * tile_size + local_col] = a[a_row_idx * params.K + row];
        } else {
            tile_a[local_row * tile_size + local_col] = 0.0;
        }

        if (b_valid) {
            tile_b[local_row * tile_size + local_col] = b[b_row_idx * params.N + col];
        } else {
            tile_b[local_row * tile_size + local_col] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < tile_size; i = i + 1u) {
            acc = acc + tile_a[local_row * tile_size + i] * tile_b[i * tile_size + local_col];
        }

        workgroupBarrier();
    }

    if (row < params.K && col < params.N) {
        c[row * params.N + col] = acc;
    }
}
"#;

/// Output-tile edge used by [`MatMulOp::dispatch`] for an `m x n` result.
///
/// The double-buffered kernel computes one 32x32 tile per 16x16 workgroup.
/// When either output dimension is smaller than a full tile most lanes work on
/// padded rows or columns, so the 16x16 kernel does the same job with fewer
/// idle lanes and is used instead.
fn forward_tile_edge(m: u32, n: u32) -> u32 {
    if m >= 32 && n >= 32 {
        32
    } else {
        16
    }
}

pub struct MatMulOp {
    pipeline: wgpu::ComputePipeline,
    /// 32x32-tile, double-buffered forward pipeline (`SHADER_DOUBLE_BUF`).
    double_buf_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Pipeline for dL/dA = dC × B^T (transpose-B matmul)
    backward_b_pipeline: wgpu::ComputePipeline,
    /// Pipeline for dL/dB = A^T × dC (transpose-A matmul)
    backward_a_pipeline: wgpu::ComputePipeline,
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

        // The double-buffered kernel uses the same bind group layout (three
        // storage buffers and one uniform), so it shares `pipeline_layout`.
        let double_buf_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Double-Buffer Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_DOUBLE_BUF.into()),
        });
        let double_buf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Double-Buffer Pipeline"),
            layout: Some(&pipeline_layout),
            module: &double_buf_shader_module,
            entry_point: Some("matmul_double_buf"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // === Backward pipelines ===
        let tb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul TransposeB Shader"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_TRANSPOSE_B_WGSL.into()),
        });
        let backward_b_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul TransposeB Pipeline"),
            layout: Some(&pipeline_layout),
            module: &tb_shader,
            entry_point: Some("matmul_tb"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let ta_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul TransposeA Shader"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_TRANSPOSE_A_WGSL.into()),
        });
        let backward_a_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul TransposeA Pipeline"),
            layout: Some(&pipeline_layout),
            module: &ta_shader,
            entry_point: Some("matmul_ta"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            pipeline,
            double_buf_pipeline,
            bind_group_layout,
            backward_b_pipeline,
            backward_a_pipeline,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        }
    }

    /// Create and write the uniform buffer shared by every forward dispatch.
    fn write_params(&self, label: &str, m: u32, k: u32, n: u32) -> wgpu::Buffer {
        let params_data: [u32; 3] = [m, k, n];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: 12,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));
        params_buffer
    }

    /// Encode one forward matmul pass with an explicit pipeline and tile edge.
    fn encode_forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c_binding: wgpu::BindingResource<'_>,
        label: &str,
        m: u32,
        k: u32,
        n: u32,
        pipeline: &wgpu::ComputePipeline,
        tile: u32,
    ) -> Result<()> {
        let params_buffer = self.write_params(&format!("{label} Params"), m, k, n);

        tracing::debug!(
            "MatMulOp dispatch: M={} K={} N={} tile={} workgroups=({},{},1)",
            m,
            k,
            n,
            tile,
            (m + tile - 1) / tile,
            (n + tile - 1) / tile
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
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
                    resource: c_binding,
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroup_count_x = (m + tile - 1) / tile;
        let workgroup_count_y = (n + tile - 1) / tile;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        drop(pass);

        Ok(())
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
        // Use the 32x32 double-buffered kernel whenever both output dimensions
        // span a full tile; otherwise fall back to the 16x16 kernel. Both
        // kernels compute the same product; see `forward_tile_edge`.
        let tile = forward_tile_edge(m, n);
        let pipeline = if tile == 32 {
            &self.double_buf_pipeline
        } else {
            &self.pipeline
        };
        self.encode_forward(
            encoder,
            a,
            b,
            c.buffer().as_entire_binding(),
            "MatMul Compute Pass",
            m,
            k,
            n,
            pipeline,
            tile,
        )
    }

    /// Dispatch the naive 16x16 kernel explicitly.
    ///
    /// This is the reference path for numerical-equivalence tests; production
    /// callers use [`dispatch`](Self::dispatch), which selects the faster
    /// kernel. It is also the fallback for output tiles smaller than 32.
    pub fn dispatch_naive(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        self.encode_forward(
            encoder,
            a,
            b,
            c.buffer().as_entire_binding(),
            "MatMul Naive Compute Pass",
            m,
            k,
            n,
            &self.pipeline,
            16,
        )
    }

    /// The output-tile edge [`dispatch`](Self::dispatch) uses for an `m x n`
    /// result: 32 for the double-buffered kernel, 16 for the naive fallback.
    pub fn selected_tile_edge(&self, m: u32, n: u32) -> u32 {
        forward_tile_edge(m, n)
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
        let tile = forward_tile_edge(m, n);
        let pipeline = if tile == 32 {
            &self.double_buf_pipeline
        } else {
            &self.pipeline
        };
        let c_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: c_buffer,
            offset: c_offset,
            size: Some(
                std::num::NonZeroU64::new(c_size)
                    .unwrap_or(std::num::NonZeroU64::new(1).unwrap()),
            ),
        });

        self.encode_forward(
            encoder,
            a,
            b,
            c_binding,
            "MatMul Offset Compute Pass",
            m,
            k,
            n,
            pipeline,
            tile,
        )
    }

    /// Backward pass: compute dL/dA = dC × B^T
    ///
    /// Given forward C = A × B where A=[M,K], B=[K,N], C=[M,N]:
    ///   dL/dA = dC × B^T  →  dL/dA[m,k] = Σ_n dC[m,n] * B[k,n]
    ///
    /// # Arguments
    /// * `dC` — gradient w.r.t. output [M, N]
    /// * `b` — weight matrix B from forward pass [K, N]
    /// * `dA` — output gradient w.r.t. A [M, K]
    pub fn dispatch_backward_a(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dc: &GpuBuffer,
        b: &GpuBuffer,
        da: &GpuBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let tile_size = 16u32;
        let wg_x = (m + tile_size - 1) / tile_size;
        let wg_y = (k + tile_size - 1) / tile_size;

        // Params for transpose-B: A=[M,N], B=[K,N], C=[M,K]
        // The kernel treats N as the shared (reduction) dimension
        let params_data: [u32; 3] = [m, k, n];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("MatMul Backward-A Params"),
            size: 12,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Backward-A Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dc.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: da.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MatMul Backward-A Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.backward_b_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        drop(pass);

        Ok(())
    }

    /// Backward pass: compute dL/dB = A^T × dC
    ///
    /// Given forward C = A × B where A=[M,K], B=[K,N], C=[M,N]:
    ///   dL/dB = A^T × dC  →  dL/dB[k,n] = Σ_m A[m,k] * dC[m,n]
    ///
    /// # Arguments
    /// * `a` — input matrix A from forward pass [M, K]
    /// * `dC` — gradient w.r.t. output [M, N]
    /// * `dB` — output gradient w.r.t. B [K, N]
    pub fn dispatch_backward_b(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        dc: &GpuBuffer,
        db: &GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        let tile_size = 16u32;
        let wg_x = (k + tile_size - 1) / tile_size;
        let wg_y = (n + tile_size - 1) / tile_size;

        // Params for transpose-A: A=[M,K], B=[M,N], C=[K,N]
        // The kernel treats M as the shared (reduction) dimension
        let params_data: [u32; 3] = [m, k, n];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("MatMul Backward-B Params"),
            size: 12,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Backward-B Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dc.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: db.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MatMul Backward-B Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.backward_a_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
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

        // Each workgroup computes a 32x32 output tile from 16x16 threads, and
        // `gid.x * 2` is the global row. The count must therefore be
        // ceil(m/32), not ceil(m/16): the latter launched twice the needed
        // workgroups and relied on the shader's bounds guards to discard them.
        let wg_x = (m + 31) / 32;
        let wg_y = (n + 31) / 32;

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

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for i in 0..k {
                    sum += a[r * k + i] * b[i * n + col];
                }
                c[r * n + col] = sum;
            }
        }
        c
    }

    fn matmul_backward_a_ref(dc: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut da = vec![0.0f32; m * k];
        for r in 0..m {
            for col in 0..k {
                let mut sum = 0.0f32;
                for i in 0..n {
                    sum += dc[r * n + i] * b[col * n + i];
                }
                da[r * k + col] = sum;
            }
        }
        da
    }

    fn matmul_backward_b_ref(a: &[f32], dc: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut db = vec![0.0f32; k * n];
        for r in 0..k {
            for col in 0..n {
                let mut sum = 0.0f32;
                for i in 0..m {
                    sum += a[i * k + r] * dc[i * n + col];
                }
                db[r * n + col] = sum;
            }
        }
        db
    }

    #[test]
    fn test_forward_tile_edge_selects_double_buffer_only_for_full_tiles() {
        // Production shapes (seq x hidden / intermediate) take the 32x32 path.
        assert_eq!(forward_tile_edge(512, 3072), 32);
        assert_eq!(forward_tile_edge(512, 768), 32);
        // Exactly one tile is enough.
        assert_eq!(forward_tile_edge(32, 32), 32);
        // A dimension below one 32-wide tile keeps the 16x16 kernel.
        assert_eq!(forward_tile_edge(16, 768), 16);
        assert_eq!(forward_tile_edge(768, 31), 16);
        assert_eq!(forward_tile_edge(2, 2), 16);
        assert_eq!(forward_tile_edge(1, 16384), 16);
    }

    #[test]
    fn test_matmul_backward_a_identity() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0];
        let dc = a.clone();
        let da = matmul_backward_a_ref(&dc, &b, 2, 2, 2);
        for (i, (&got, &expected)) in da.iter().zip(a.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-5, "da[{}] = {} expected {}", i, got, expected);
        }
    }

    #[test]
    fn test_matmul_backward_b_identity() {
        let a = vec![1.0f32, 0.0, 0.0, 1.0];
        let dc = vec![1.0f32, 2.0, 3.0, 4.0];
        let db = matmul_backward_b_ref(&a, &dc, 2, 2, 2);
        for (i, (&got, &expected)) in db.iter().zip(dc.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-5, "db[{}] = {} expected {}", i, got, expected);
        }
    }

    #[test]
    fn test_matmul_backward_chain_rule() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = matmul_ref(&a, &b, 2, 3, 2);
        assert_eq!(c.len(), 4);
        let dc = vec![1.0f32, 0.0, 0.0, 1.0];
        let da = matmul_backward_a_ref(&dc, &b, 2, 2, 3);
        assert_eq!(da.len(), 6);
        let db = matmul_backward_b_ref(&a, &dc, 2, 3, 2);
        assert_eq!(db.len(), 6);
        for &g in &da { assert!(g.is_finite()); }
        for &g in &db { assert!(g.is_finite()); }
        assert!(da.iter().sum::<f32>().abs() > 0.1);
        assert!(db.iter().sum::<f32>().abs() > 0.1);
    }

    #[test]
    fn test_matmul_backward_a_specific() {
        let dc = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let da = matmul_backward_a_ref(&dc, &b, 2, 2, 2);
        assert!((da[0] - 5.0).abs() < 1e-5);
        assert!((da[1] - 7.0).abs() < 1e-5);
        assert!((da[2] - 0.0).abs() < 1e-5);
        assert!((da[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_backward_b_specific() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let dc = vec![1.0f32, 0.0, 0.0, 0.0];
        let db = matmul_backward_b_ref(&a, &dc, 2, 2, 2);
        assert!((db[0] - 1.0).abs() < 1e-5);
        assert!((db[1] - 0.0).abs() < 1e-5);
        assert!((db[2] - 2.0).abs() < 1e-5);
        assert!((db[3] - 0.0).abs() < 1e-5);
    }
}
