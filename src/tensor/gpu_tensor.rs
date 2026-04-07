use std::marker::PhantomData;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable, cast_slice};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

use crate::error::{FerrisResError, Result};

pub struct GpuTensor<T: Pod + Zeroable> {
    buffer: Buffer,
    shape: Vec<usize>,
    numel: usize,
    device: Arc<Device>,
    queue: Arc<Queue>,
    _marker: PhantomData<T>,
}

impl<T: Pod + Zeroable> GpuTensor<T> {
    pub fn from_data(
        device: Arc<Device>,
        queue: Arc<Queue>,
        shape: Vec<usize>,
        data: &[T],
    ) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(FerrisResError::Shape(format!(
                "data length {} does not match shape {:?} (product {})",
                data.len(),
                shape,
                numel
            )));
        }

        let byte_size = (numel * std::mem::size_of::<T>()) as u64;
        tracing::debug!(
            "GpuTensor::from_data shape={:?} numel={} byte_size={}",
            shape,
            numel,
            byte_size
        );

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("GpuTensor"),
            size: byte_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(cast_slice(data));
        buffer.unmap();

        Ok(Self {
            buffer,
            shape,
            numel,
            device,
            queue,
            _marker: PhantomData,
        })
    }

    pub fn zeros(device: Arc<Device>, queue: Arc<Queue>, shape: Vec<usize>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let byte_size = (numel * std::mem::size_of::<T>()) as u64;

        tracing::debug!(
            "GpuTensor::zeros shape={:?} numel={} byte_size={}",
            shape,
            numel,
            byte_size
        );

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("GpuTensor(zeros)"),
            size: byte_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped.copy_from_slice(bytemuck::cast_slice(&vec![T::zeroed(); numel]));
        drop(mapped);
        buffer.unmap();

        Ok(Self {
            buffer,
            shape,
            numel,
            device,
            queue,
            _marker: PhantomData,
        })
    }

    pub fn read_back(&self, output: &mut [T]) -> Result<()> {
        if output.len() != self.numel {
            return Err(FerrisResError::Shape(format!(
                "output length {} does not match tensor numel {}",
                output.len(),
                self.numel
            )));
        }

        let byte_size = self.buffer.size();
        if byte_size == 0 {
            return Ok(());
        }

        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("GpuTensor Staging"),
            size: byte_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuTensor Readback"),
            });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| FerrisResError::Gpu(format!("device poll failed: {}", e)))?;
        rx.recv()
            .unwrap()
            .map_err(|e| FerrisResError::Gpu(format!("buffer map failed: {}", e)))?;

        let mapped = slice.get_mapped_range();
        output.copy_from_slice(cast_slice(&mapped));
        drop(mapped);
        staging.unmap();

        Ok(())
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}
