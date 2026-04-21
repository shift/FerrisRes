//! Fused patch embedding kernel — Implicit GEMM.
//!
//! Combines im2col patch extraction and the linear projection into a single
//! WGSL compute shader. No intermediate `[N_patches, P*P*C]` buffer is ever
//! written; the patch pixels are read directly from the image and accumulated
//! into the output embedding in one pass.
//!
//! Memory saving vs explicit im2col path at 224×224 × RGB × P=16:
//!   im2col buffer  = (14×14) × (16×16×3) × 4 B  ≈ 59 MB
//!   fused buffer   = 0 B
//!
//! Task: b3f74a12 — see papers_research/implicit_gemm_fused_patching_research.md

use std::sync::Arc;
use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    Device, Queue, ShaderStages,
};
use crate::compute::GpuBuffer;
use crate::error::Result;

// ---------------------------------------------------------------------------
// WGSL — tiled Implicit GEMM patch embedding
//
// Workgroup layout: (16, 16)
//   lid.x = local patch index within tile   (n_local)
//   lid.y = local embed-dim index within tile (d_local)
//   gid.x = global patch index
//   gid.y = global embed-dim index
//
// Shared memory:
//   tile_w [TILE_K × TILE_D] — weight slice
//   tile_p [TILE_N × TILE_K] — patch pixel slice (read implicitly from img)
//
// Each output element out[patch_idx, embed_idx] is the dot product of:
//   row  = patch_idx row of the virtual im2col matrix (P*P*C pixels)
//   col  = embed_idx column of the weight matrix [P*P*C, D]
// ---------------------------------------------------------------------------
pub const FUSED_PATCH_EMBED_WGSL: &str = r#"
struct Params {
    height:    u32,   // image H (pixels)
    width:     u32,   // image W (pixels)
    channels:  u32,   // image C
    patch_size: u32,  // P
    embed_dim: u32,   // D
    n_patches: u32,   // (H/P) * (W/P)
    has_bias:  u32,   // 1 = use bias binding, 0 = no bias
    _pad:      u32,
}

@group(0) @binding(0) var<storage, read>       img:    array<f32>;  // [H, W, C] interleaved
@group(0) @binding(1) var<storage, read>       weight: array<f32>;  // [P*P*C, D]
@group(0) @binding(2) var<storage, read>       bias:   array<f32>;  // [D] (may be unused)
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;  // [N_patches, D]
@group(0) @binding(4) var<uniform>             params: Params;

const TILE: u32 = 16u;

var<workgroup> tile_w: array<f32, 256>;   // TILE_K × TILE_D  (16 × 16)
var<workgroup> tile_p: array<f32, 256>;   // TILE_N × TILE_K  (16 × 16)

@compute @workgroup_size(16, 16)
fn fused_patch_embed(
    @builtin(global_invocation_id)  gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>,
    @builtin(workgroup_id)          wid: vec3<u32>,
) {
    let local_n   = lid.x;
    let local_d   = lid.y;
    let global_n  = gid.x;   // wid.x * 16 + local_n
    let global_d  = gid.y;   // wid.y * 16 + local_d

    let kk              = params.patch_size * params.patch_size * params.channels;
    let num_k_tiles     = (kk + TILE - 1u) / TILE;
    let patches_per_row = params.width / params.patch_size;

    var acc: f32 = 0.0;

    for (var t: u32 = 0u; t < num_k_tiles; t = t + 1u) {

        // ------------------------------------------------------------------
        // Load weight tile: tile_w[local_n, local_d] = weight[t*T+local_n, global_d]
        // Thread (local_n, local_d) is responsible for this element.
        // ------------------------------------------------------------------
        let k_w = t * TILE + local_n;
        if (k_w < kk && global_d < params.embed_dim) {
            tile_w[local_n * TILE + local_d] = weight[k_w * params.embed_dim + global_d];
        } else {
            tile_w[local_n * TILE + local_d] = 0.0;
        }

        // ------------------------------------------------------------------
        // Load patch tile (implicit — no im2col buffer):
        // tile_p[local_n, local_d] = pixel value for patch global_n at K-pos t*T+local_d
        // Thread (local_n, local_d) computes its own pixel coordinate.
        // ------------------------------------------------------------------
        let k_p = t * TILE + local_d;
        if (global_n < params.n_patches && k_p < kk) {
            let p_row  = global_n / patches_per_row;
            let p_col  = global_n % patches_per_row;
            let c      = k_p % params.channels;
            let loc_xy = k_p / params.channels;
            let ly     = loc_xy / params.patch_size;
            let lx     = loc_xy % params.patch_size;
            let img_y  = p_row * params.patch_size + ly;
            let img_x  = p_col * params.patch_size + lx;
            tile_p[local_n * TILE + local_d] =
                img[(img_y * params.width + img_x) * params.channels + c];
        } else {
            tile_p[local_n * TILE + local_d] = 0.0;
        }

        workgroupBarrier();

        // ------------------------------------------------------------------
        // Accumulate: acc += dot(tile_p[local_n, :], tile_w[:, local_d])
        // ------------------------------------------------------------------
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            acc = acc + tile_p[local_n * TILE + k] * tile_w[k * TILE + local_d];
        }

        workgroupBarrier();
    }

    // Write output + optional bias
    if (global_n < params.n_patches && global_d < params.embed_dim) {
        var result = acc;
        if (params.has_bias != 0u) {
            result = result + bias[global_d];
        }
        out[global_n * params.embed_dim + global_d] = result;
    }
}
"#;

// ---------------------------------------------------------------------------
// Rust wrapper
// ---------------------------------------------------------------------------

/// Fused patch embedding operation (Implicit GEMM).
///
/// Replaces the two-step `im2col` + `matmul` pipeline with a single shader
/// dispatch that reads image pixels on-the-fly, eliminating the large
/// intermediate patch buffer.
pub struct FusedPatchEmbedOp {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    /// One-element zero buffer used as a dummy when `has_bias = false` so
    /// the binding slot is always populated.
    dummy_bias: wgpu::Buffer,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl FusedPatchEmbedOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Self {
        tracing::debug!(event = "creating_fusedpatchembedop_implicit_gemm_pipeline", "Creating FusedPatchEmbedOp (Implicit GEMM) pipeline");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fused Patch Embed Shader"),
            source: wgpu::ShaderSource::Wgsl(FUSED_PATCH_EMBED_WGSL.into()),
        });

        let ro_entry = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw_entry = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = BindGroupLayoutEntry {
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
            label: Some("Fused Patch Embed BGL"),
            entries: &[
                ro_entry(0), // img
                ro_entry(1), // weight
                ro_entry(2), // bias (or dummy)
                rw_entry(3), // out
                uniform_entry,
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fused Patch Embed Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fused Patch Embed Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fused_patch_embed"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // 4-byte dummy bias buffer (one f32 = 0.0)
        let dummy_bias = device.create_buffer(&BufferDescriptor {
            label: Some("Fused Patch Embed Dummy Bias"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        tracing::debug!(event = "fusedpatchembedop_pipeline_created", "FusedPatchEmbedOp pipeline created");

        Self {
            pipeline,
            bgl,
            dummy_bias,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        }
    }

    /// Dispatch the fused patch embedding.
    ///
    /// # Arguments
    /// * `img`        – GPU buffer containing the image in HWC interleaved f32 layout.
    /// * `weight`     – Projection weight `[P*P*C, embed_dim]` as f32.
    /// * `bias`       – Optional projection bias `[embed_dim]`.
    /// * `output`     – Output buffer `[n_patches, embed_dim]`. Must be pre-allocated.
    /// * `height`/`width`/`channels` – Image dimensions.
    /// * `patch_size` – Patch side length P (image must be divisible by P).
    /// * `embed_dim`  – Projection output dimensionality D.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        img: &GpuBuffer,
        weight: &GpuBuffer,
        bias: Option<&GpuBuffer>,
        output: &GpuBuffer,
        height: u32,
        width: u32,
        channels: u32,
        patch_size: u32,
        embed_dim: u32,
    ) -> Result<()> {
        debug_assert!(
            height % patch_size == 0 && width % patch_size == 0,
            "Image dimensions must be divisible by patch_size"
        );

        let n_patches = (height / patch_size) * (width / patch_size);
        let has_bias: u32 = if bias.is_some() { 1 } else { 0 };

        // Pack params: [height, width, channels, patch_size, embed_dim, n_patches, has_bias, _pad]
        let params_data: [u32; 8] = [
            height, width, channels, patch_size, embed_dim, n_patches, has_bias, 0,
        ];
        let params_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("Fused Patch Embed Params"),
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bias_resource = match bias {
            Some(b) => b.buffer().as_entire_binding(),
            None => self.dummy_bias.as_entire_binding(),
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Patch Embed Bind Group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: img.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bias_resource },
                wgpu::BindGroupEntry { binding: 3, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        // Grid: ceil(n_patches / 16) × ceil(embed_dim / 16)
        let wg_n = (n_patches + 15) / 16;
        let wg_d = (embed_dim + 15) / 16;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fused Patch Embed Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_n, wg_d, 1);
        drop(pass);

        tracing::debug!(
            "FusedPatchEmbedOp dispatched: {}×{}×{} patch_size={} embed_dim={} \
             n_patches={} wg={}×{}",
            height, width, channels, patch_size, embed_dim, n_patches, wg_n, wg_d
        );

        Ok(())
    }

    /// Compute the output buffer size in bytes for the given parameters.
    pub fn output_byte_size(height: u32, width: u32, patch_size: u32, embed_dim: u32) -> usize {
        let n_patches = ((height / patch_size) * (width / patch_size)) as usize;
        n_patches * embed_dim as usize * std::mem::size_of::<f32>()
    }
}
