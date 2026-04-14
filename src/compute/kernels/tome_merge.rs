//! Token Merging (ToMe) merge kernel.
//!
//! After the bipartite matching is computed on the CPU side, this kernel
//! performs the actual token merging on the GPU: matched token pairs are
//! averaged (weighted by their token sizes) and written to a compacted
//! output buffer.
//!
//! The CPU side produces two arrays:
//!   - `merge_map[i]`: for each surviving token index `i` in the output,
//!     either a single source token index (unmerged) or the index of the
//!     *first* token in a merged pair. If `merge_pair_a[i] !=
//!     merge_pair_b[i]`, tokens A and B are merged.
//!   - `merge_pair_a[i]`, `merge_pair_b[i]`: the two source token indices
//!     that produce output token `i`. If `a == b`, the token is unmerged.
//!
//! Task: c9e2d541 — see papers_research/token_merging_tome_research.md

use std::sync::Arc;
use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    Device, Queue, ShaderStages,
};
use crate::compute::GpuBuffer;
use crate::error::Result;

const TOME_MERGE_WGSL: &str = r#"
struct Params {
    n_out:   u32,   // number of output tokens (after merging)
    dim:     u32,   // token embedding dimension (hidden_dim)
    _pad0:   u32,
    _pad1:   u32,
}

@group(0) @binding(0) var<storage, read>       tokens:     array<f32>;  // [N_in, dim]
@group(0) @binding(1) var<storage, read>       token_sizes: array<f32>; // [N_in] weights
@group(0) @binding(2) var<storage, read>       pair_a:     array<u32>;  // [N_out] first token idx
@group(0) @binding(3) var<storage, read>       pair_b:     array<u32>;  // [N_out] second token idx
@group(0) @binding(4) var<storage, read_write> out_tokens: array<f32>;  // [N_out, dim]
@group(0) @binding(5) var<storage, read_write> out_sizes:  array<f32>;  // [N_out]
@group(0) @binding(6) var<uniform>             params:     Params;

@compute @workgroup_size(256)
fn tome_merge_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    if (out_idx >= params.n_out) {
        return;
    }

    let a = pair_a[out_idx];
    let b = pair_b[out_idx];
    let size_a = token_sizes[a];
    let size_b = token_sizes[b];
    let total_size = size_a + size_b;

    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        let val_a = tokens[a * params.dim + d];
        let val_b = tokens[b * params.dim + d];
        out_tokens[out_idx * params.dim + d] = (val_a * size_a + val_b * size_b) / total_size;
    }

    out_sizes[out_idx] = total_size;
}
"#;

/// ToMe merge operation — compacts token pairs on the GPU.
///
/// After CPU-side bipartite matching determines which token pairs to merge,
/// this kernel performs the size-weighted averaging and writes the compacted
/// output.
pub struct TomeMergeOp {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl TomeMergeOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Self {
        tracing::debug!(event = "creating_tomemergeop_pipeline", "Creating TomeMergeOp pipeline");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ToMe Merge Shader"),
            source: wgpu::ShaderSource::Wgsl(TOME_MERGE_WGSL.into()),
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
            binding: 6,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ToMe Merge BGL"),
            entries: &[
                ro_storage(0), // tokens
                ro_storage(1), // token_sizes
                ro_storage(2), // pair_a
                ro_storage(3), // pair_b
                rw_storage(4), // out_tokens
                rw_storage(5), // out_sizes
                uniform,
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ToMe Merge Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ToMe Merge Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("tome_merge_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self { pipeline, bgl, device: Arc::clone(device), queue: Arc::clone(queue) }
    }

    /// Dispatch the ToMe merge kernel.
    ///
    /// # Arguments
    /// * `tokens`      – Input token buffer `[N_in, dim]`.
    /// * `token_sizes` – Per-token size weights `[N_in]` (all 1.0 for unmerged).
    /// * `pair_a`      – For each output token, the first source index `[N_out]` u32.
    /// * `pair_b`      – For each output token, the second source index `[N_out]` u32.
    ///   When `pair_a[i] == pair_b[i]`, the token is copied (not merged).
    /// * `out_tokens`  – Output token buffer `[N_out, dim]`.
    /// * `out_sizes`   – Output token sizes `[N_out]`.
    /// * `n_out`       – Number of output tokens.
    /// * `dim`         – Token embedding dimension.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tokens: &GpuBuffer,
        token_sizes: &GpuBuffer,
        pair_a: &GpuBuffer,
        pair_b: &GpuBuffer,
        out_tokens: &GpuBuffer,
        out_sizes: &GpuBuffer,
        n_out: u32,
        dim: u32,
    ) -> Result<()> {
        let params_data: [u32; 4] = [n_out, dim, 0, 0];
        let params_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("ToMe Merge Params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ToMe Merge BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tokens.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: token_sizes.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pair_a.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pair_b.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: out_tokens.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_sizes.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
            ],
        });

        let wg = (n_out + 255) / 256;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ToMe Merge Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
        drop(pass);

        tracing::debug!(event = "tomemergeop_dispatched_n_out_dim_wg", "TomeMergeOp dispatched: n_out={} dim={} wg={}", n_out, dim, wg);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CPU-side bipartite matching for ToMe
// ---------------------------------------------------------------------------

/// Result of a bipartite soft matching pass.
pub struct TomeMatchResult {
    /// For each output token, the first source token index.
    pub pair_a: Vec<u32>,
    /// For each output token, the second source token index.
    /// Equal to `pair_a[i]` when the token is unmerged (copy).
    pub pair_b: Vec<u32>,
    /// Number of output tokens after merging.
    pub n_out: usize,
}

/// Perform bipartite soft matching on key vectors.
///
/// Splits tokens into odd/even sets A and B. For each token in A, finds the
/// most similar token in B (by cosine similarity on keys). Greedily assigns
/// the top-r pairs and merges them. Remaining tokens are copied unmerged.
///
/// # Arguments
/// * `keys`  – Key vectors `[n_tokens, dim]` in row-major layout.
/// * `n_tokens` – Number of tokens.
/// * `dim`   – Key dimension.
/// * `r`     – Number of pairs to merge.
///
/// # Returns
/// A `TomeMatchResult` with pair assignments.
pub fn bipartite_match(keys: &[f32], n_tokens: usize, dim: usize, r: usize) -> TomeMatchResult {
    let r = r.min(n_tokens / 2);

    // Split into A (odd) and B (even) by position
    let set_a: Vec<usize> = (0..n_tokens).filter(|i| i % 2 == 0).collect();
    let set_b: Vec<usize> = (0..n_tokens).filter(|i| i % 2 == 1).collect();
    let n_a = set_a.len();
    let n_b = set_b.len();

    // Compute cosine similarity matrix [n_a, n_b]
    let mut sim = vec![0.0f32; n_a * n_b];
    for ia in 0..n_a {
        let a_idx = set_a[ia];
        let a_key = &keys[a_idx * dim..(a_idx + 1) * dim];
        let mut a_norm: f32 = 0.0;
        for d in 0..dim { a_norm += a_key[d] * a_key[d]; }
        a_norm = a_norm.sqrt().max(1e-8);

        for ib in 0..n_b {
            let b_idx = set_b[ib];
            let b_key = &keys[b_idx * dim..(b_idx + 1) * dim];
            let mut b_norm: f32 = 0.0;
            let mut dot: f32 = 0.0;
            for d in 0..dim {
                dot += a_key[d] * b_key[d];
                b_norm += b_key[d] * b_key[d];
            }
            b_norm = b_norm.sqrt().max(1e-8);
            sim[ia * n_b + ib] = dot / (a_norm * b_norm);
        }
    }

    // Greedy matching: pick the highest similarity pair, remove both from
    // consideration, repeat r times.
    let mut a_used = vec![false; n_a];
    let mut b_used = vec![false; n_b];
    let mut matched_pairs: Vec<(usize, usize)> = Vec::with_capacity(r);

    for _ in 0..r {
        let mut best_sim: f32 = f32::NEG_INFINITY;
        let mut best_ia: usize = 0;
        let mut best_ib: usize = 0;
        let mut found = false;

        for ia in 0..n_a {
            if a_used[ia] { continue; }
            for ib in 0..n_b {
                if b_used[ib] { continue; }
                let s = sim[ia * n_b + ib];
                if s > best_sim {
                    best_sim = s;
                    best_ia = ia;
                    best_ib = ib;
                    found = true;
                }
            }
        }

        if !found { break; }
        a_used[best_ia] = true;
        b_used[best_ib] = true;
        matched_pairs.push((best_ia, best_ib));
    }

    // Build output: merged pairs first, then unmatched tokens (as copies)
    let n_merged = matched_pairs.len();
    let n_unmatched_a = a_used.iter().filter(|&&u| !u).count();
    let n_unmatched_b = b_used.iter().filter(|&&u| !u).count();
    let n_out = n_merged + n_unmatched_a + n_unmatched_b;

    let mut pair_a = Vec::with_capacity(n_out);
    let mut pair_b = Vec::with_capacity(n_out);

    // Merged pairs
    for (ia, ib) in &matched_pairs {
        pair_a.push(set_a[*ia] as u32);
        pair_b.push(set_b[*ib] as u32);
    }

    // Unmatched A tokens (copy)
    for ia in 0..n_a {
        if !a_used[ia] {
            let idx = set_a[ia] as u32;
            pair_a.push(idx);
            pair_b.push(idx);
        }
    }

    // Unmatched B tokens (copy)
    for ib in 0..n_b {
        if !b_used[ib] {
            let idx = set_b[ib] as u32;
            pair_a.push(idx);
            pair_b.push(idx);
        }
    }

    TomeMatchResult { pair_a, pair_b, n_out }
}
