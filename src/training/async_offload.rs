use std::collections::VecDeque;
use std::sync::Arc;

use wgpu::Device;

use crate::compute::buffer::GpuBuffer;
use crate::device::profile::DeviceProfile;
use crate::error::Result;

/// Async gradient offload engine for CPU-Offload compute mode.
///
/// Implements a triple-buffer (or double-buffer) pipeline where:
/// - While the GPU writes gradient batch **N** into the active staging buffer,
/// - the CPU is simultaneously reading gradient batch **N-1** from the
///   previously-filled staging buffer.
///
/// This hides GPU→CPU transfer latency behind useful computation.
///
/// # Buffering depth
/// The `buffering_depth` controls how many GPU staging buffers are held in the
/// pool at once:
/// - `3` (HighEnd)           — full triple-buffer pipeline
/// - `2` (MidRange, or low-end with < 85% memory usage)
/// - `1` (Integrated / LowEnd when mem_usage_pct > 85%)  — minimal staging
pub struct AsyncGradientOffload {
    /// Rotating pool of GPU → CPU staging buffers.
    staging_pool: VecDeque<wgpu::Buffer>,
    /// Number of staging buffers in the pool.
    pub buffering_depth: usize,
    /// Index of the current write slot (0..buffering_depth).
    write_slot: usize,
    /// Size in bytes of each gradient buffer.
    grad_size: usize,
    #[allow(dead_code)]
    /// Shared device reference.
    device: Arc<Device>,
}

impl AsyncGradientOffload {
    /// Create a new `AsyncGradientOffload`.
    ///
    /// # Arguments
    /// * `device`          – wgpu device.
    /// * `grad_size`       – byte size of each gradient buffer that will be
    ///                       passed to [`offload_batch`].
    /// * `buffering_depth` – number of staging buffers to pre-allocate.
    ///                       Use [`buffering_depth_for_profile`] to derive
    ///                       this from a [`DeviceProfile`].
    pub fn new(device: Arc<Device>, grad_size: usize, buffering_depth: usize) -> Result<Self> {
        assert!(buffering_depth >= 1, "buffering_depth must be at least 1");
        let depth = buffering_depth.max(1);

        let mut staging_pool = VecDeque::with_capacity(depth);
        for i in 0..depth {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("async_offload_staging_{}", i)),
                size: grad_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            staging_pool.push_back(buf);
        }

        tracing::info!(
            "AsyncGradientOffload: created with buffering_depth={} grad_size={}",
            depth,
            grad_size
        );

        Ok(Self {
            staging_pool,
            buffering_depth: depth,
            write_slot: 0,
            grad_size,
            device,
        })
    }

    /// Submit gradient batch `grad_buf` for async CPU readback.
    ///
    /// The method:
    /// 1. Copies `grad_buf` into the **current** staging buffer (GPU-side copy).
    /// 2. Submits the copy command.
    /// 3. If `buffering_depth >= 2`, non-blockingly attempts to read the
    ///    **previous** staging buffer (batch N-1) from CPU memory while the
    ///    GPU writes batch N.  If that slot is not yet ready the function
    ///    falls back to polling the current slot.
    /// 4. Returns the CPU-side `Vec<f32>` for the oldest pending gradient.
    ///
    /// On the first call there is no previous batch, so the function always
    /// polls the current copy.
    pub async fn offload_batch(
        &mut self,
        grad_buf: &GpuBuffer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<f32> {
        // ── Step 1: copy grad_buf into the current write slot ──────────────
        let current_staging = self.staging_pool.pop_front().expect("staging pool empty");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("async_offload_copy"),
        });
        let copy_size = self.grad_size.min(grad_buf.size());
        encoder.copy_buffer_to_buffer(
            grad_buf.buffer(),
            0,
            &current_staging,
            0,
            copy_size as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Push the now-filled staging buffer to the back of the deque so the
        // *front* of the deque always holds the oldest filled slot.
        self.staging_pool.push_back(current_staging);
        self.write_slot = (self.write_slot + 1) % self.buffering_depth;

        // ── Step 2: read back the oldest staging buffer ────────────────────
        // For depth ≥ 2 the front of the deque is the N-1 slot that was
        // submitted on the previous call and is likely already mapped.
        // For depth 1 the front IS the slot we just submitted (block until
        // ready).
        let read_buf = self.staging_pool.front().expect("staging pool empty");

        let (tx, rx) = tokio::sync::oneshot::channel::<
            std::result::Result<(), wgpu::BufferAsyncError>,
        >();

        read_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });

        // Poll until the mapping is ready.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        // Await the oneshot; if the channel was dropped (device lost) return zeros.
        let _ = rx.await;

        let numel = self.grad_size / std::mem::size_of::<f32>();
        let mapped = read_buf.slice(..).get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);

        // Unmap the buffer so it can be written to again next iteration.
        let read_buf_mut = self.staging_pool.pop_front().expect("staging pool empty");
        read_buf_mut.unmap();
        self.staging_pool.push_front(read_buf_mut);

        tracing::debug!(
            "AsyncGradientOffload::offload_batch: returned {} f32 values (depth={})",
            numel,
            self.buffering_depth
        );

        result
    }

    /// Drain all pending staging buffers and return the last batch of
    /// gradient values.  Call at end of training to flush the pipeline.
    pub async fn flush(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<Vec<f32>> {
        if self.grad_size == 0 {
            return None;
        }

        // Map every buffer in the pool and return the first non-empty result.
        for _ in 0..self.buffering_depth {
            let buf = self.staging_pool.pop_front()?;
            let (tx, rx) = tokio::sync::oneshot::channel::<
                std::result::Result<(), wgpu::BufferAsyncError>,
            >();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let _ = rx.await;
            let mapped = buf.slice(..).get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
            drop(mapped);
            buf.unmap();
            self.staging_pool.push_back(buf);
            return Some(result);
        }
        None
    }

    /// Return the recommended buffering depth for a given device profile and
    /// current memory usage.
    ///
    /// | Profile                     | Condition                  | Depth |
    /// |-----------------------------|----------------------------|-------|
    /// | `HighEnd`                   | any                        | 3     |
    /// | `MidRange`                  | any                        | 2     |
    /// | `Integrated` or `LowEnd`    | `mem_usage_pct` > 0.85     | 1     |
    /// | `Integrated` or `LowEnd`    | `mem_usage_pct` ≤ 0.85     | 2     |
    pub fn buffering_depth_for_profile(profile: DeviceProfile, mem_usage_pct: f32) -> usize {
        match profile {
            DeviceProfile::HighEnd => 3,
            DeviceProfile::MidRange => 2,
            DeviceProfile::Integrated | DeviceProfile::LowEnd => {
                if mem_usage_pct > 0.85 {
                    1
                } else {
                    2
                }
            }
        }
    }

    /// Byte size of each gradient tensor this offloader was created for.
    pub fn grad_size(&self) -> usize {
        self.grad_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffering_depth_for_profile() {
        assert_eq!(
            AsyncGradientOffload::buffering_depth_for_profile(DeviceProfile::HighEnd, 0.5),
            3
        );
        assert_eq!(
            AsyncGradientOffload::buffering_depth_for_profile(DeviceProfile::MidRange, 0.9),
            2
        );
        assert_eq!(
            AsyncGradientOffload::buffering_depth_for_profile(DeviceProfile::Integrated, 0.9),
            1
        );
        assert_eq!(
            AsyncGradientOffload::buffering_depth_for_profile(DeviceProfile::Integrated, 0.5),
            2
        );
        assert_eq!(
            AsyncGradientOffload::buffering_depth_for_profile(DeviceProfile::LowEnd, 0.86),
            1
        );
        assert_eq!(
            AsyncGradientOffload::buffering_depth_for_profile(DeviceProfile::LowEnd, 0.84),
            2
        );
    }
}
