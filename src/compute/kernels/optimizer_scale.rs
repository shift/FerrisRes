//! WGSL GPU kernels for SCALE optimizer (edge-optimized).
//!
//! SCALE (Stochastic Column-Adaptive Learning for Edges) uses column-norm
//! gradient scaling instead of Adam's element-wise moments. This gives
//! 12.1 MB state vs Adam's 49.84 GB for the same model.
//!
//! Three kernels:
//! 1. column_norms: Compute sqrt(Σ_i g_ij²) per column
//! 2. scale_update: W -= lr * g / column_norms (column-wise scaling)
//! 3. scale_momentum: m = β₁m + g_tilde (for output layers only)

use std::sync::Arc;
use wgpu::{Device, Queue, CommandEncoder, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

/// Compute column norms: for each column j, compute sqrt(Σ_i g_ij²).
/// Input: gradient [rows × cols] f32
/// Output: column_norms [cols] f32
pub const COLUMN_NORMS_WGSL: &str = r#"
    struct Params {
        rows: u32,
        cols: u32,
    }

    @group(0) @binding(0) var<storage, read> gradient: array<f32>;
    @group(0) @binding(1) var<storage, read_write> norms: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;

    @compute @workgroup_size(256)
    fn column_norms_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let col = gid.x;
        if (col >= params.cols) {
            return;
        }

        var sum_sq: f32 = 0.0;
        for (var row = 0u; row < params.rows; row = row + 1u) {
            let val = gradient[row * params.cols + col];
            sum_sq = sum_sq + val * val;
        }
        norms[col] = sqrt(sum_sq);
    }
"#;

/// Scale update: W -= lr * g / max(column_norm, epsilon)
/// Input: weights [rows × cols], gradient [rows × cols], column_norms [cols]
/// Output: weights updated in-place
pub const SCALE_UPDATE_WGSL: &str = r#"
    struct Params {
        rows: u32,
        cols: u32,
        lr: u32,           // bitcast<f32>
        epsilon: u32,      // bitcast<f32>
    }

    @group(0) @binding(0) var<storage, read_write> weights: array<f32>;
    @group(0) @binding(1) var<storage, read> gradient: array<f32>;
    @group(0) @binding(2) var<storage, read> norms: array<f32>;
    @group(0) @binding(3) var<uniform> params: Params;

    @compute @workgroup_size(256)
    fn scale_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        let total = params.rows * params.cols;
        if (idx >= total) {
            return;
        }

        let col = idx % params.cols;
        let lr = bitcast<f32>(params.lr);
        let epsilon = bitcast<f32>(params.epsilon);
        let norm = norms[col];
        let scale = lr / max(norm, epsilon);

        weights[idx] = weights[idx] - scale * gradient[idx];
    }
"#;

/// Momentum update: m = β₁ * m + g (for output layers only)
/// Input: momentum [rows × cols], gradient [rows × cols]
/// Output: momentum updated in-place
pub const SCALE_MOMENTUM_WGSL: &str = r#"
    struct Params {
        total: u32,
        beta1: u32,        // bitcast<f32>
    }

    @group(0) @binding(0) var<storage, read_write> momentum: array<f32>;
    @group(0) @binding(1) var<storage, read> gradient: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;

    @compute @workgroup_size(256)
    fn scale_momentum_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx >= params.total) {
            return;
        }

        let beta1 = bitcast<f32>(params.beta1);
        momentum[idx] = beta1 * momentum[idx] + gradient[idx];
    }
"#;

pub struct ScaleGpuOps {
    column_norms_pipeline: wgpu::ComputePipeline,
    column_norms_bgl: wgpu::BindGroupLayout,
    scale_update_pipeline: wgpu::ComputePipeline,
    scale_update_bgl: wgpu::BindGroupLayout,
    momentum_pipeline: wgpu::ComputePipeline,
    momentum_bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl ScaleGpuOps {
    pub fn new(device: &Arc<Device>, _queue: &Arc<Queue>) -> Result<Self> {
        let read = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let rw = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let uniform = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // --- Column norms ---
        let cn_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SCALE Column Norms Shader"),
            source: wgpu::ShaderSource::Wgsl(COLUMN_NORMS_WGSL.into()),
        });
        let cn_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SCALE Column Norms BGL"),
            entries: &[read(0), rw(1), uniform(2)],
        });
        let cn_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SCALE Column Norms Layout"),
            bind_group_layouts: &[Some(&cn_bgl)],
            immediate_size: 0,
        });
        let cn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SCALE Column Norms Pipeline"),
            layout: Some(&cn_layout),
            module: &cn_shader,
            entry_point: Some("column_norms_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // --- Scale update ---
        let su_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SCALE Update Shader"),
            source: wgpu::ShaderSource::Wgsl(SCALE_UPDATE_WGSL.into()),
        });
        let su_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SCALE Update BGL"),
            entries: &[rw(0), read(1), read(2), uniform(3)],
        });
        let su_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SCALE Update Layout"),
            bind_group_layouts: &[Some(&su_bgl)],
            immediate_size: 0,
        });
        let su_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SCALE Update Pipeline"),
            layout: Some(&su_layout),
            module: &su_shader,
            entry_point: Some("scale_update_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // --- Momentum ---
        let mo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SCALE Momentum Shader"),
            source: wgpu::ShaderSource::Wgsl(SCALE_MOMENTUM_WGSL.into()),
        });
        let mo_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SCALE Momentum BGL"),
            entries: &[rw(0), read(1), uniform(2)],
        });
        let mo_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SCALE Momentum Layout"),
            bind_group_layouts: &[Some(&mo_bgl)],
            immediate_size: 0,
        });
        let mo_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SCALE Momentum Pipeline"),
            layout: Some(&mo_layout),
            module: &mo_shader,
            entry_point: Some("scale_momentum_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            column_norms_pipeline: cn_pipeline,
            column_norms_bgl: cn_bgl,
            scale_update_pipeline: su_pipeline,
            scale_update_bgl: su_bgl,
            momentum_pipeline: mo_pipeline,
            momentum_bgl: mo_bgl,
            device: Arc::clone(device),
        })
    }

    /// Compute column norms: norms[j] = sqrt(Σ_i gradient[i*cols+j]²)
    pub fn dispatch_column_norms(
        &self,
        encoder: &mut CommandEncoder,
        gradient: &GpuBuffer,
        norms: &GpuBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        let params: [u32; 2] = [rows, cols];
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SCALE Column Norms Params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&params));
        params_buf.unmap();

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SCALE Column Norms BG"),
            layout: &self.column_norms_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gradient.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: norms.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SCALE Column Norms Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.column_norms_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((cols + 255) / 256, 1, 1);
        drop(pass);

        Ok(())
    }

    /// Scale update: weights -= lr * gradient / max(column_norms, epsilon)
    pub fn dispatch_scale_update(
        &self,
        encoder: &mut CommandEncoder,
        weights: &GpuBuffer,
        gradient: &GpuBuffer,
        norms: &GpuBuffer,
        rows: u32,
        cols: u32,
        lr: f32,
        epsilon: f32,
    ) -> Result<()> {
        let params: [u32; 4] = [rows, cols, lr.to_bits(), epsilon.to_bits()];
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SCALE Update Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&params));
        params_buf.unmap();

        let total = rows * cols;
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SCALE Update BG"),
            layout: &self.scale_update_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: weights.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gradient.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: norms.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SCALE Update Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.scale_update_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((total + 255) / 256, 1, 1);
        drop(pass);

        Ok(())
    }

    /// Momentum update: momentum = beta1 * momentum + gradient
    pub fn dispatch_momentum(
        &self,
        encoder: &mut CommandEncoder,
        momentum: &GpuBuffer,
        gradient: &GpuBuffer,
        total: u32,
        beta1: f32,
    ) -> Result<()> {
        let params: [u32; 2] = [total, beta1.to_bits()];
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SCALE Momentum Params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&params));
        params_buf.unmap();

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SCALE Momentum BG"),
            layout: &self.momentum_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: momentum.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gradient.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SCALE Momentum Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.momentum_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((total + 255) / 256, 1, 1);
        drop(pass);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    fn column_norms_ref(gradient: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut norms = vec![0.0f32; cols];
        for r in 0..rows {
            for c in 0..cols {
                let v = gradient[r * cols + c];
                norms[c] += v * v;
            }
        }
        for n in &mut norms {
            *n = n.sqrt();
        }
        norms
    }

    fn scale_update_ref(
        weights: &mut [f32],
        gradient: &[f32],
        norms: &[f32],
        rows: usize,
        cols: usize,
        lr: f32,
        epsilon: f32,
    ) {
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let scale = lr / norms[c].max(epsilon);
                weights[idx] -= scale * gradient[idx];
            }
        }
    }

    fn momentum_ref(momentum: &mut [f32], gradient: &[f32], beta1: f32) {
        for i in 0..momentum.len() {
            momentum[i] = beta1 * momentum[i] + gradient[i];
        }
    }

    #[test]
    fn test_column_norms_basic() {
        // 2×3 matrix: [[1, 2, 3], [4, 5, 6]]
        let gradient = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let norms = column_norms_ref(&gradient, 2, 3);
        // col 0: sqrt(1+16) = sqrt(17)
        assert!((norms[0] - 17.0f32.sqrt()).abs() < 1e-5);
        // col 1: sqrt(4+25) = sqrt(29)
        assert!((norms[1] - 29.0f32.sqrt()).abs() < 1e-5);
        // col 2: sqrt(9+36) = sqrt(45)
        assert!((norms[2] - 45.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_column_norms_zero() {
        let gradient = vec![0.0f32; 6];
        let norms = column_norms_ref(&gradient, 2, 3);
        assert!(norms.iter().all(|&n| n == 0.0));
    }

    #[test]
    fn test_scale_update_basic() {
        let mut weights = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]; // 2×3
        let gradient = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let norms = column_norms_ref(&gradient, 2, 3);
        let lr = 0.1f32;
        let epsilon = 1e-8f32;

        scale_update_ref(&mut weights, &gradient, &norms, 2, 3, lr, epsilon);

        // col 0: scale = 0.1/sqrt(17) ≈ 0.02426
        let scale0 = lr / norms[0].max(epsilon);
        assert!((weights[0] - (10.0 - scale0 * 1.0)).abs() < 1e-4);
        assert!((weights[3] - (40.0 - scale0 * 4.0)).abs() < 1e-4);
    }

    #[test]
    fn test_scale_update_zero_norm() {
        // When column norm is zero, epsilon prevents division by zero
        let mut weights = vec![10.0f32, 20.0];
        let gradient = vec![0.0f32, 1.0];
        let norms = vec![0.0f32, 1.0];
        let lr = 0.1f32;
        let epsilon = 1e-8f32;

        scale_update_ref(&mut weights, &gradient, &norms, 1, 2, lr, epsilon);

        // col 0: norm=0, scale = lr/epsilon = very large, but gradient=0 so no change
        assert!((weights[0] - 10.0).abs() < 1e-4);
        // col 1: scale = 0.1/1.0 = 0.1
        assert!((weights[1] - 19.9).abs() < 1e-4);
    }

    #[test]
    fn test_momentum_basic() {
        let mut momentum = vec![1.0f32, 2.0, 3.0];
        let gradient = vec![0.1f32, 0.2, 0.3];
        let beta1 = 0.9f32;

        momentum_ref(&mut momentum, &gradient, beta1);

        assert!((momentum[0] - (0.9 * 1.0 + 0.1)).abs() < 1e-5);
        assert!((momentum[1] - (0.9 * 2.0 + 0.2)).abs() < 1e-5);
        assert!((momentum[2] - (0.9 * 3.0 + 0.3)).abs() < 1e-5);
    }

    #[test]
    fn test_momentum_decay() {
        // After many steps with zero gradient, momentum should decay to zero
        let mut momentum = vec![10.0f32];
        let gradient = vec![0.0f32];
        for _ in 0..100 {
            momentum_ref(&mut momentum, &gradient, 0.9);
        }
        assert!(momentum[0].abs() < 1e-2);
    }
}
