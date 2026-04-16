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

        // Drop staging — freed when GPU processes the copy
        drop(staging);

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

    /// Read buffer contents from GPU to CPU.
    /// Uses a staging buffer approach.
    pub fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue, output: &mut [u8]) -> Result<()> {
        // Create staging buffer (host-visible, can be read after map)
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("read_staging"),
            size: self.size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy GPU buffer → staging
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("read_copy"),
        });
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size as u64);
        queue.submit(std::iter::once(enc.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let _ = rx.recv().map_err(|_| crate::error::FerrisResError::Device("read failed".into()))?;

        let data = slice.get_mapped_range();
        output.copy_from_slice(&data);
        drop(data);
        staging.unmap();

        Ok(())
    }
}
