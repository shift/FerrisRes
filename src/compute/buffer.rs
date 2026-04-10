use wgpu::{Buffer, BufferDescriptor, BufferUsages};
use crate::error::Result;

pub struct GpuBuffer {
    buffer: Buffer,
    size: usize,
}

impl GpuBuffer {
    pub fn new(
        device: &wgpu::Device,
        size: usize,
        label: Option<&str>,
    ) -> Result<Self> {
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self { buffer, size })
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn zeros(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: usize,
        label: Option<&str>,
    ) -> Result<Self> {
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &vec![0u8; size]);
        Ok(Self { buffer, size })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn from_existing(buffer: wgpu::Buffer, size: usize) -> Self {
        Self { buffer, size }
    }
}
