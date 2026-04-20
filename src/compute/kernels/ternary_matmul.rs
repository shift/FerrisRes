//! WGSL GPU kernel for ternary matrix multiplication.
//!
//! Ternary weights are {-1, 0, +1} packed as 2-bit values (4 per byte / 16 per u32).
//! The matmul eliminates all hardware multipliers — only conditional add/subtract.
//!
//! Layout:
//!   ternary: [rows * cols_packed] where cols_packed = ceil(cols / 16) u32s
//!            Each u32 holds 16 × 2-bit ternary values (LSB first)
//!   scale:   per-row absmean scale factor
//!   input:   [seq_len * cols] f32
//!   output:  [seq_len * rows] f32

use std::sync::Arc;
use wgpu::{Device, Queue, CommandEncoder, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

const TERNARY_MATMUL_WGSL: &str = r#"
    struct Params {
        seq_len: u32,
        in_cols: u32,
        out_rows: u32,
        cols_packed: u32,  // ceil(in_cols / 16)
    }

    @group(0) @binding(0) var<storage, read> ternary: array<u32>;
    @group(0) @binding(1) var<storage, read> scales: array<f32>;
    @group(0) @binding(2) var<storage, read> input: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;

    @compute @workgroup_size(64)
    fn ternary_matmul_main(
        @builtin(global_invocation_id) gid: vec3<u32>,
    ) {
        let seq_idx = gid.x / params.out_rows;
        let row_idx = gid.x % params.out_rows;

        if (seq_idx >= params.seq_len) || (row_idx >= params.out_rows) {
            return;
        }

        let in_cols = params.in_cols;
        let cols_packed = params.cols_packed;
        let row_base = row_idx * cols_packed;
        let input_base = seq_idx * in_cols;

        var sum: f32 = 0.0;

        for (var p = 0u; p < cols_packed; p = p + 1u) {
            let packed = ternary[row_base + p];
            let col_base = p * 16u;

            // Unpack 16 × 2-bit values and accumulate
            for (var b = 0u; b < 16u; b = b + 1u) {
                let col = col_base + b;
                if (col >= in_cols) {
                    break;
                }

                let code = (packed >> (b * 2u)) & 3u;
                let x = input[input_base + col];

                // code: 0 = -1, 1 = 0, 2 = +1
                if (code == 0u) {
                    sum = sum - x;      // -1
                } else if (code == 2u) {
                    sum = sum + x;      // +1
                }
                // code == 1: 0, skip
            }
        }

        output[seq_idx * params.out_rows + row_idx] = sum * scales[row_idx];
    }
"#;

pub struct TernaryMatMulOp {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl TernaryMatMulOp {
    pub fn new(device: &Arc<Device>, _queue: &Arc<Queue>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ternary MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(TERNARY_MATMUL_WGSL.into()),
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
            label: Some("Ternary MatMul BGL"),
            entries: &[
                read_storage(0),  // ternary (packed u32)
                read_storage(1),  // scales (f32 per row)
                read_storage(2),  // input (f32)
                rw_storage(3),    // output (f32)
                uniform_entry(4), // params
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ternary MatMul Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ternary MatMul Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("ternary_matmul_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            bind_group_layout: bgl,
            device: Arc::clone(device),
        })
    }

    /// Dispatch ternary matmul: output = ternary_weights × input × scale
    ///
    /// # Arguments
    /// * `ternary` — Packed 2-bit ternary weights [rows × cols_packed] u32
    /// * `scales` — Per-row absmean scale factors [rows] f32
    /// * `input` — Input activations [seq_len × cols] f32
    /// * `output` — Output buffer [seq_len × rows] f32
    /// * `seq_len` — Sequence length
    /// * `in_cols` — Input column dimension (unpacked)
    /// * `out_rows` — Output row dimension
    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        ternary: &GpuBuffer,
        scales: &GpuBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
        seq_len: u32,
        in_cols: u32,
        out_rows: u32,
    ) -> Result<()> {
        let cols_packed = (in_cols + 15) / 16;
        let params_data: [u32; 4] = [seq_len, in_cols, out_rows, cols_packed];

        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ternary MatMul Params"),
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
            label: Some("Ternary MatMul Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ternary.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scales.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: input.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        let total_threads = seq_len * out_rows;
        let wg_count = (total_threads + 63) / 64;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Ternary MatMul Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_count, 1, 1);

        drop(pass);

        tracing::debug!(
            "TernaryMatMul dispatched: seq={} cols={} rows={} packed={}",
            seq_len, in_cols, out_rows, cols_packed
        );

        Ok(())
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {

    /// CPU reference: ternary matmul with 2-bit packed values.
    fn ternary_matmul_ref(
        packed: &[u32],
        scales: &[f32],
        input: &[f32],
        seq_len: usize,
        in_cols: usize,
        out_rows: usize,
    ) -> Vec<f32> {
        let cols_packed = (in_cols + 15) / 16;
        let mut output = vec![0.0f32; seq_len * out_rows];

        for s in 0..seq_len {
            for r in 0..out_rows {
                let mut sum = 0.0f32;
                for p in 0..cols_packed {
                    let word = packed[r * cols_packed + p];
                    for b in 0..16 {
                        let col = p * 16 + b;
                        if col >= in_cols { break; }
                        let code = (word >> (b * 2)) & 3;
                        let x = input[s * in_cols + col];
                        match code {
                            0 => sum -= x,  // -1
                            2 => sum += x,  // +1
                            _ => {}         // 0 (code 1 or 3)
                        }
                    }
                }
                output[s * out_rows + r] = sum * scales[r];
            }
        }
        output
    }

    /// Pack a slice of i8 ternary values {-1, 0, +1} into u32 words.
    fn pack_ternary(values: &[i8], in_cols: usize) -> Vec<u32> {
        let cols_packed = (in_cols + 15) / 16;
        let rows = values.len() / in_cols;
        let mut packed = vec![0u32; rows * cols_packed];

        for r in 0..rows {
            for p in 0..cols_packed {
                let mut word = 0u32;
                for b in 0..16 {
                    let col = p * 16 + b;
                    if col >= in_cols { break; }
                    let val = values[r * in_cols + col];
                    let code = match val {
                        -1 => 0u32,
                         0 => 1u32,
                         1 => 2u32,
                         _ => 3u32, // shouldn't happen
                    };
                    word |= code << (b * 2);
                }
                packed[r * cols_packed + p] = word;
            }
        }
        packed
    }

    #[test]
    fn test_ternary_matmul_identity() {
        // 2×2 identity matrix as ternary
        let values: Vec<i8> = vec![1, 0, 0, 1]; // row 0: [+1,0], row 1: [0,+1]
        let packed = pack_ternary(&values, 2);
        let scales = vec![1.0f32, 1.0f32];
        let input = vec![3.0f32, 5.0f32]; // [1, 2]

        let output = ternary_matmul_ref(&packed, &scales, &input, 1, 2, 2);

        assert!((output[0] - 3.0).abs() < 1e-5, "output[0] = {}", output[0]);
        assert!((output[1] - 5.0).abs() < 1e-5, "output[1] = {}", output[1]);
    }

    #[test]
    fn test_ternary_matmul_negate() {
        // 2×2 negate matrix
        let values: Vec<i8> = vec![-1, 0, 0, -1];
        let packed = pack_ternary(&values, 2);
        let scales = vec![1.0f32, 1.0f32];
        let input = vec![3.0f32, 5.0f32];

        let output = ternary_matmul_ref(&packed, &scales, &input, 1, 2, 2);

        assert!((output[0] - (-3.0)).abs() < 1e-5);
        assert!((output[1] - (-5.0)).abs() < 1e-5);
    }

    #[test]
    fn test_ternary_matmul_scaled() {
        let values: Vec<i8> = vec![1, -1, 1, 1];
        let packed = pack_ternary(&values, 2);
        let scales = vec![2.0f32, 0.5f32];
        let input = vec![1.0f32, 2.0f32];

        let output = ternary_matmul_ref(&packed, &scales, &input, 1, 2, 2);

        // Row 0: (1*1 + (-1)*2) * 2.0 = (1-2) * 2 = -2.0
        assert!((output[0] - (-2.0)).abs() < 1e-5);
        // Row 1: (1*1 + 1*2) * 0.5 = 3 * 0.5 = 1.5
        assert!((output[1] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_ternary_matmul_batch() {
        // 2×2 matrix, 3 positions
        let values: Vec<i8> = vec![1, 0, 0, -1];
        let packed = pack_ternary(&values, 2);
        let scales = vec![1.0f32, 1.0f32];
        let input = vec![
            1.0f32, 2.0f32,  // pos 0
            3.0f32, 4.0f32,  // pos 1
            5.0f32, 6.0f32,  // pos 2
        ];

        let output = ternary_matmul_ref(&packed, &scales, &input, 3, 2, 2);

        // pos 0: [1, -2]
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - (-2.0)).abs() < 1e-5);
        // pos 1: [3, -4]
        assert!((output[2] - 3.0).abs() < 1e-5);
        assert!((output[3] - (-4.0)).abs() < 1e-5);
        // pos 2: [5, -6]
        assert!((output[4] - 5.0).abs() < 1e-5);
        assert!((output[5] - (-6.0)).abs() < 1e-5);
    }

    #[test]
    fn test_pack_ternary_roundtrip() {
        let values: Vec<i8> = vec![1, 0, -1, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 0, 1]; // 17 values (1 full word + 1 partial)
        let packed = pack_ternary(&values, 17);
        assert_eq!(packed.len(), 2); // ceil(17/16) = 2

        // Unpack and verify
        for (i, &v) in values.iter().enumerate() {
            let p = i / 16;
            let b = i % 16;
            let code = (packed[p] >> (b * 2)) & 3;
            let expected = match v {
                -1 => 0u32,
                 0 => 1u32,
                 1 => 2u32,
                 _ => 3u32,
            };
            assert_eq!(code, expected, "value {} at index {} packed wrong", v, i);
        }
    }
}
