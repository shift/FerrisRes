use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::kernels::matmul::MatMulOp;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::compute::GpuBuffer;
use crate::error::Result;

pub struct Linear {
    weight: GpuBuffer,
    bias: Option<GpuBuffer>,
    in_features: usize,
    out_features: usize,
    matmul_op: MatMulOp,
    elementwise_op: ElementWiseOp,
}

impl Linear {
    pub fn new(
        device: Arc<Device>,
        _queue: Arc<Queue>,
        in_features: usize,
        out_features: usize,
        use_bias: bool,
    ) -> Result<Self> {
        tracing::info!(
            "Creating Linear layer: in_features={} out_features={} use_bias={}",
            in_features, out_features, use_bias
        );

        let weight_bytes = out_features * in_features * std::mem::size_of::<f32>();
        let weight = GpuBuffer::zeros(&device, weight_bytes, Some("Linear Weight"))?;

        let bias = if use_bias {
            let bias_bytes = out_features * std::mem::size_of::<f32>();
            Some(GpuBuffer::zeros(&device, bias_bytes, Some("Linear Bias"))?)
        } else {
            None
        };

        let matmul_op = MatMulOp::new(&device);
        let elementwise_op = ElementWiseOp::new(&device);

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            matmul_op,
            elementwise_op,
        })
    }

    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        output: &GpuBuffer,
        batch_size: u32,
    ) -> Result<()> {
        let m = batch_size;
        let k = self.in_features as u32;
        let n = self.out_features as u32;

        tracing::debug!(
            "Linear::forward batch_size={} in_features={} out_features={}",
            m, k, n
        );

        self.matmul_op.dispatch(encoder, input, &self.weight, output, m, k, n)?;

        if let Some(ref bias) = self.bias {
            let numel = m * n;
            self.elementwise_op.dispatch_add_with_stride(
                encoder,
                output,
                bias,
                output,
                numel,
                n,
            )?;
        }

        Ok(())
    }

    pub fn weight(&self) -> &GpuBuffer {
        &self.weight
    }

    pub fn bias(&self) -> Option<&GpuBuffer> {
        self.bias.as_ref()
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}
