use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, util::DeviceExt};
use crate::error::Result;

pub struct GpuBuffer {
    buffer: Buffer,
    size: usize,
}

impl GpuBuffer {
    /// Create a GPU buffer with STORAGE + COPY_SRC + COPY_DST.
    /// Note: may end up in host-visible memory on some drivers.
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

    /// Create a device-local buffer and upload data via staging.
    /// Uses only STORAGE usage (no COPY_DST), forcing device-local VRAM allocation.
    /// Upload goes through a temporary staging buffer + copy command.
    pub fn new_device_local(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        label: Option<&str>,
    ) -> Result<Self> {
        let size = data.len();

        // Staging buffer: host-visible, temporary
        let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|l| format!("{}_staging", l)).as_deref(),
            contents: data,
            usage: BufferUsages::COPY_SRC,
        });

        // Destination: device-local (STORAGE + COPY_DST for the initial copy only)
        let dest = device.create_buffer(&BufferDescriptor {
            label,
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Copy staging → device-local
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("upload_device_local"),
        });
        enc.copy_buffer_to_buffer(&staging, 0, &dest, 0, size as u64);
        queue.submit(std::iter::once(enc.finish()));

        // Poll to free staging buffer immediately
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        Ok(Self { buffer: dest, size })
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
