//! WGSL GPU kernel for 2:4 sparse ternary matrix multiplication.
//!
//! 2:4 sparsity guarantees exactly 2 non-zero values per group of 4,
//! halving compute vs dense ternary. Each group of 4 positions is encoded
//! as: pattern (which 2 of 4 are non-zero, 6 valid patterns) + 2 sign bits.
//!
//! Encoding per group of 4 weights (~1.25 bits/weight):
//!   - 3 bits: pattern (6 valid 2-of-4 patterns: 0011, 0101, 0110, 1001, 1010, 1100)
//!   - 2 bits: signs of the 2 non-zero values
//!   - Total: 5 bits per group of 4

use std::sync::Arc;
use wgpu::{Device, Queue, CommandEncoder, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

/// 6 valid 2-of-4 patterns: which positions (0-3) are non-zero.
/// Pattern index maps to two position indices.
#[allow(dead_code)]
const PATTERN_LUT: [(usize, usize); 6] = [
    (0, 1), // pattern 0: positions 0,1
    (0, 2), // pattern 1: positions 0,2
    (0, 3), // pattern 2: positions 0,3
    (1, 2), // pattern 3: positions 1,2
    (1, 3), // pattern 4: positions 1,3
    (2, 3), // pattern 5: positions 2,3
];

const SPARSE_TERNARY_MATMUL_WGSL: &str = r#"
    struct Params {
        seq_len: u32,
        in_cols: u32,
        out_rows: u32,
        groups_per_row: u32,   // ceil(in_cols / 4)
    }

    @group(0) @binding(0) var<storage, read> patterns: array<u32>;  // packed pattern + sign data
    @group(0) @binding(1) var<storage, read> scales: array<f32>;
    @group(0) @binding(2) var<storage, read> input: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;

    // Pattern lookup: pattern index → (pos_a, pos_b)
    // 6 patterns encoded as a flat array: pattern * 2 + 0 = pos_a, pattern * 2 + 1 = pos_b
    const PATTERNS = array<u32, 12>(
        0u, 1u,   // pattern 0: positions 0,1
        0u, 2u,   // pattern 1: positions 0,2
        0u, 3u,   // pattern 2: positions 0,3
        1u, 2u,   // pattern 3: positions 1,2
        1u, 3u,   // pattern 4: positions 1,3
        2u, 3u,   // pattern 5: positions 2,3
    );

    @compute @workgroup_size(64)
    fn sparse_ternary_matmul_main(
        @builtin(global_invocation_id) gid: vec3<u32>,
    ) {
        let seq_idx = gid.x / params.out_rows;
        let row_idx = gid.x % params.out_rows;

        if (seq_idx >= params.seq_len) || (row_idx >= params.out_rows) {
            return;
        }

        let in_cols = params.in_cols;
        let groups_per_row = params.groups_per_row;
        let input_base = seq_idx * in_cols;

        var sum: f32 = 0.0;

        // Each u32 holds multiple groups (each group is 5 bits: 3 pattern + 2 signs)
        // We pack 6 groups per u32 (30 bits used, 2 bits padding)
        let words_per_row = (groups_per_row + 5u) / 6u;  // ceil(groups/6)

        for (var w = 0u; w < words_per_row; w = w + 1u) {
            let word = patterns[row_idx * words_per_row + w];

            for (var g = 0u; g < 6u; g = g + 1u) {
                let group_idx = w * 6u + g;
                if (group_idx >= groups_per_row) {
                    break;
                }

                let base_bit = g * 5u;
                let packed = (word >> base_bit) & 0x1Fu; // 5 bits

                let pattern = packed & 7u;  // lower 3 bits
                let signs = (packed >> 3u) & 3u; // upper 2 bits

                // Clamp pattern to valid range (0-5)
                let pat = min(pattern, 5u);
                let pos_a = PATTERNS[pat * 2u];
                let pos_b = PATTERNS[pat * 2u + 1u];

                let col_a = group_idx * 4u + pos_a;
                let col_b = group_idx * 4u + pos_b;

                // Sign bits: bit 0 = sign of first non-zero, bit 1 = sign of second
                let sign_a: f32 = select(1.0, -1.0, (signs & 1u) != 0u);
                let sign_b: f32 = select(1.0, -1.0, (signs & 2u) != 0u);

                if (col_a < in_cols) {
                    sum = sum + sign_a * input[input_base + col_a];
                }
                if (col_b < in_cols) {
                    sum = sum + sign_b * input[input_base + col_b];
                }
            }
        }

        output[seq_idx * params.out_rows + row_idx] = sum * scales[row_idx];
    }
"#;

pub struct SparseTernaryMatMulOp {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl SparseTernaryMatMulOp {
    pub fn new(device: &Arc<Device>, _queue: &Arc<Queue>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sparse Ternary MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_TERNARY_MATMUL_WGSL.into()),
        });

        let read_storage = |binding: u32| BindGroupLayoutEntry {
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

        let uniform_entry = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sparse Ternary MatMul BGL"),
            entries: &[
                read_storage(0),  // patterns (packed u32)
                read_storage(1),  // scales (f32 per row)
                read_storage(2),  // input (f32)
                rw_storage(3),    // output (f32)
                uniform_entry(4), // params
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sparse Ternary MatMul Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sparse Ternary MatMul Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("sparse_ternary_matmul_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            bind_group_layout: bgl,
            device: Arc::clone(device),
        })
    }

    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        patterns: &GpuBuffer,
        scales: &GpuBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
        seq_len: u32,
        in_cols: u32,
        out_rows: u32,
    ) -> Result<()> {
        let groups_per_row = (in_cols + 3) / 4;
        let params_data: [u32; 4] = [seq_len, in_cols, out_rows, groups_per_row];

        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Ternary MatMul Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&params_data));
        params_buffer.unmap();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparse Ternary MatMul Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: patterns.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scales.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: input.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        let total_threads = seq_len * out_rows;
        let wg_count = (total_threads + 63) / 64;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Sparse Ternary MatMul Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_count, 1, 1);

        drop(pass);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// Pack sparse ternary data: each group of 4 weights has exactly 2 non-zero.
    /// Returns packed u32 array and per-row scales.
    fn pack_sparse_ternary(
        weights: &[i8], // {-1, 0, +1}, exactly 2 non-zero per group of 4
        rows: usize,
        cols: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let groups_per_row = (cols + 3) / 4;
        let groups_per_word = 6; // 6 groups × 5 bits = 30 bits per u32
        let words_per_row = (groups_per_row + groups_per_word - 1) / groups_per_word;

        let mut packed = vec![0u32; rows * words_per_row];
        let mut scales = vec![1.0f32; rows];

        for r in 0..rows {
            let mut row_sum = 0.0f32;
            let mut row_count = 0usize;

            for g in 0..groups_per_row {
                let base = g * 4;
                // Find which positions are non-zero
                let mut nonzero = Vec::new();
                for i in 0..4 {
                    let col = base + i;
                    if col < cols {
                        let v = weights[r * cols + col];
                        if v != 0 {
                            nonzero.push((i, v));
                            row_sum += (v as f32).abs();
                            row_count += 1;
                        }
                    }
                }

                assert_eq!(nonzero.len(), 2, "Each group must have exactly 2 non-zero values");

                // Determine pattern index
                let (pos_a, pos_b) = (nonzero[0].0, nonzero[1].0);
                let pat_idx = match (pos_a, pos_b) {
                    (0, 1) => 0u32,
                    (0, 2) => 1u32,
                    (0, 3) => 2u32,
                    (1, 2) => 3u32,
                    (1, 3) => 4u32,
                    (2, 3) => 5u32,
                    _ => panic!("Invalid positions"),
                };

                // Sign bits: bit 0 = first non-zero sign, bit 1 = second
                let sign_a = if nonzero[0].1 < 0 { 1u32 } else { 0u32 };
                let sign_b = if nonzero[1].1 < 0 { 1u32 } else { 0u32 };
                let signs = sign_a | (sign_b << 1);

                let group_packed = pat_idx | (signs << 3);

                let word_idx = g / groups_per_word;
                let bit_offset = (g % groups_per_word) * 5;
                packed[r * words_per_row + word_idx] |= group_packed << bit_offset;
            }

            if row_count > 0 {
                scales[r] = row_sum / row_count as f32; // absmean
            }
        }

        (packed, scales)
    }

    /// CPU reference: sparse ternary matmul from packed data.
    fn sparse_ternary_matmul_ref(
        packed: &[u32],
        scales: &[f32],
        input: &[f32],
        seq_len: usize,
        cols: usize,
        rows: usize,
    ) -> Vec<f32> {
        let groups_per_row = (cols + 3) / 4;
        let groups_per_word = 6;
        let words_per_row = (groups_per_row + groups_per_word - 1) / groups_per_word;

        let patterns = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];
        let mut output = vec![0.0f32; seq_len * rows];

        for s in 0..seq_len {
            for r in 0..rows {
                let mut sum = 0.0f32;
                for g in 0..groups_per_row {
                    let word_idx = g / groups_per_word;
                    let bit_offset = (g % groups_per_word) * 5;
                    let group_packed = (packed[r * words_per_row + word_idx] >> bit_offset) & 0x1F;
                    let pat = (group_packed & 7).min(5) as usize;
                    let signs = (group_packed >> 3) & 3;

                    let (pos_a, pos_b) = patterns[pat];
                    let col_a = g * 4 + pos_a;
                    let col_b = g * 4 + pos_b;

                    let sign_a: f32 = if signs & 1 != 0 { -1.0 } else { 1.0 };
                    let sign_b: f32 = if signs & 2 != 0 { -1.0 } else { 1.0 };

                    if col_a < cols { sum += sign_a * input[s * cols + col_a]; }
                    if col_b < cols { sum += sign_b * input[s * cols + col_b]; }
                }
                output[s * rows + r] = sum * scales[r];
            }
        }
        output
    }

    #[test]
    fn test_sparse_ternary_identity() {
        // 4×4 identity with 2:4 sparsity
        // Row 0: [1,1,0,0], Row 1: [1,0,1,0], Row 2: [0,1,0,1], Row 3: [0,0,1,1]
        let weights: Vec<i8> = vec![
            1, 1, 0, 0,  // row 0
            1, 0, 1, 0,  // row 1
            0, 1, 0, 1,  // row 2
            0, 0, 1, 1,  // row 3
        ];
        let (packed, scales) = pack_sparse_ternary(&weights, 4, 4);
        let input = vec![1.0f32, 2.0, 3.0, 4.0];

        let output = sparse_ternary_matmul_ref(&packed, &scales, &input, 1, 4, 4);

        // Row 0: 1*1 + 1*2 = 3.0
        assert!((output[0] - 3.0).abs() < 1e-5, "row 0 = {}", output[0]);
        // Row 1: 1*1 + 1*3 = 4.0
        assert!((output[1] - 4.0).abs() < 1e-5, "row 1 = {}", output[1]);
        // Row 2: 1*2 + 1*4 = 6.0
        assert!((output[2] - 6.0).abs() < 1e-5, "row 2 = {}", output[2]);
        // Row 3: 1*3 + 1*4 = 7.0
        assert!((output[3] - 7.0).abs() < 1e-5, "row 3 = {}", output[3]);
    }

    #[test]
    fn test_sparse_ternary_negate() {
        let weights: Vec<i8> = vec![
            -1, -1, 0, 0,
            -1, 0, -1, 0,
        ];
        let (packed, scales) = pack_sparse_ternary(&weights, 2, 4);
        let input = vec![1.0f32, 2.0, 3.0, 4.0];

        let output = sparse_ternary_matmul_ref(&packed, &scales, &input, 1, 4, 2);

        // Row 0: -1*1 + -1*2 = -3.0
        assert!((output[0] - (-3.0)).abs() < 1e-5);
        // Row 1: -1*1 + -1*3 = -4.0
        assert!((output[1] - (-4.0)).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_ternary_all_patterns() {
        // Test all 6 patterns in one row (8 columns = 2 groups)
        // Group 0: pattern 0 (pos 0,1) = [1, -1, 0, 0]
        // Group 1: pattern 5 (pos 2,3) = [0, 0, 1, -1]
        let weights: Vec<i8> = vec![
            1, -1, 0, 0,  // group 0: pattern 0
            0, 0, 1, -1,  // group 1: pattern 5
        ];
        let (packed, scales) = pack_sparse_ternary(&weights, 1, 8);
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let output = sparse_ternary_matmul_ref(&packed, &scales, &input, 1, 8, 1);

        // 1*1 + (-1)*2 + 1*7 + (-1)*8 = 1 - 2 + 7 - 8 = -2.0
        assert!((output[0] - (-2.0)).abs() < 1e-5, "output = {}", output[0]);
    }

    #[test]
    fn test_sparse_ternary_batch() {
        let weights: Vec<i8> = vec![1, 1, 0, 0];
        let (packed, scales) = pack_sparse_ternary(&weights, 1, 4);
        let input = vec![
            1.0f32, 2.0, 0.0, 0.0,  // pos 0
            3.0f32, 4.0, 0.0, 0.0,  // pos 1
        ];

        let output = sparse_ternary_matmul_ref(&packed, &scales, &input, 2, 4, 1);

        // pos 0: 1+2 = 3.0, pos 1: 3+4 = 7.0
        assert!((output[0] - 3.0).abs() < 1e-5);
        assert!((output[1] - 7.0).abs() < 1e-5);
    }
}
