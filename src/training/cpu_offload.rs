use std::collections::HashMap;
use std::sync::Arc;
use wgpu::Device;
use crate::compute::buffer::GpuBuffer;
use crate::error::{FerrisResError, Result};

/// A CPU-side buffer that holds gradient data downloaded from the GPU.
///
/// Used in `CpuOffload` compute mode where the GPU has insufficient VRAM to
/// keep all gradient tensors live simultaneously.  Gradients are read back to
/// CPU memory via a staging buffer, processed on the CPU (or uploaded back for
/// the Adam update), and the GPU staging memory is reclaimed.
///
/// ## Multi-parameter accumulation
///
/// This struct supports two usage modes:
///
/// **Single-buffer mode** (backwards compatible): construct with `new(numel)`,
/// access `data` directly.
///
/// **Accumulation mode**: construct with `new_accumulator()`, then call
/// `register`, `accumulate`, `normalize`, `reset` across multiple micro-batches.
/// This mode fixes the divide-by-K gap documented in engram context 0dbab45c:
/// the Adam optimizer does **not** divide gradients by the accumulation count
/// internally, so the caller must normalise before passing to Adam.
pub struct CpuGradientBuffer {
    /// The gradient values as a flat `f32` vector (single-buffer mode).
    pub data: Vec<f32>,
    /// Number of elements (not bytes) in `data` (single-buffer mode).
    pub numel: usize,
    /// Per-parameter accumulated gradients (accumulation mode).
    accumulators: HashMap<String, Vec<f32>>,
    /// How many micro-batches have been accumulated since the last reset.
    micro_batch_count: u32,
}

impl CpuGradientBuffer {
    /// Create an empty gradient buffer with `numel` zeroed f32 elements.
    /// (Single-buffer mode — backwards compatible.)
    pub fn new(numel: usize) -> Self {
        Self {
            data: vec![0.0_f32; numel],
            numel,
            accumulators: HashMap::new(),
            micro_batch_count: 0,
        }
    }

    /// Create a `CpuGradientBuffer` intended for multi-parameter accumulation
    /// across micro-batches.  Use `register`, `accumulate`, `normalize`, and
    /// `reset` to drive the accumulation loop.
    pub fn new_accumulator() -> Self {
        Self {
            data: Vec::new(),
            numel: 0,
            accumulators: HashMap::new(),
            micro_batch_count: 0,
        }
    }

    // ── Accumulation-mode API ─────────────────────────────────────────────────

    /// Register a parameter with the given number of elements.
    /// Subsequent `accumulate` calls for this parameter must match `numel`.
    pub fn register(&mut self, name: &str, numel: usize) {
        self.accumulators.insert(name.to_string(), vec![0.0f32; numel]);
        tracing::debug!("CpuGradientBuffer: registered '{}' numel={}", name, numel);
    }

    /// Add `grad` element-wise into the CPU accumulation buffer for `name`.
    ///
    /// # Errors
    /// Returns an error if `name` was not registered or if `grad.len()` does
    /// not match the registered size.
    pub fn accumulate(&mut self, name: &str, grad: &[f32]) -> Result<()> {
        let buf = self.accumulators.get_mut(name).ok_or_else(|| {
            FerrisResError::Device(format!("CpuGradientBuffer: '{}' not registered", name))
        })?;
        if buf.len() != grad.len() {
            return Err(FerrisResError::Device(format!(
                "CpuGradientBuffer: size mismatch for '{}': expected {}, got {}",
                name,
                buf.len(),
                grad.len()
            )));
        }
        for (dst, src) in buf.iter_mut().zip(grad.iter()) {
            *dst += src;
        }
        tracing::debug!("CpuGradientBuffer: accumulated '{}' ({} elements)", name, grad.len());
        Ok(())
    }

    /// Increment the micro-batch counter.  Call once per micro-batch after all
    /// parameter gradients have been offloaded and accumulated for that batch.
    pub fn increment_micro_batch(&mut self) {
        self.micro_batch_count += 1;
        tracing::debug!("CpuGradientBuffer: micro_batch_count={}", self.micro_batch_count);
    }

    /// Returns the current micro-batch accumulation count.
    pub fn micro_batch_count(&self) -> u32 {
        self.micro_batch_count
    }

    /// Divides every accumulated gradient element by `accumulation_steps`.
    ///
    /// **This is the fix for the divide-by-K bug** (engram context 0dbab45c):
    /// `AdamOptimizer::step` receives a raw gradient and does not internally
    /// divide by the accumulation count.  The caller must normalise the
    /// accumulated sum before passing it to Adam.
    ///
    /// This method is a no-op when `accumulation_steps <= 1`.
    pub fn normalize(&mut self, accumulation_steps: u32) {
        if accumulation_steps <= 1 {
            return;
        }
        let inv_k = 1.0 / accumulation_steps as f32;
        for (name, buf) in self.accumulators.iter_mut() {
            for v in buf.iter_mut() {
                *v *= inv_k;
            }
            tracing::debug!(
                "CpuGradientBuffer: normalized '{}' by 1/{} = {:.6}",
                name, accumulation_steps, inv_k
            );
        }
    }

    /// Returns a reference to the accumulated (and possibly normalised) gradient
    /// data for `name`, or `None` if not registered.
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.accumulators.get(name).map(|v| v.as_slice())
    }

    /// Reset all accumulation buffers to zero and the micro-batch counter to 0.
    pub fn reset(&mut self) {
        for buf in self.accumulators.values_mut() {
            for v in buf.iter_mut() {
                *v = 0.0;
            }
        }
        self.micro_batch_count = 0;
        tracing::debug!("CpuGradientBuffer: reset");
    }

    // ── Single-buffer async upload/download (original API) ───────────────────

    /// Asynchronously read a `GpuBuffer` back to CPU memory.
    ///
    /// Creates a MAP_READ staging buffer, copies `src` into it, submits the
    /// command buffer, and polls the device until the mapping completes.
    /// Returns the gradient values as a `Vec<f32>`.
    pub async fn from_gpu_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &GpuBuffer,
    ) -> Result<Self> {
        let size = src.size();
        let numel = size / std::mem::size_of::<f32>();

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cpu_grad_staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cpu_offload_readback"),
        });
        encoder.copy_buffer_to_buffer(src.buffer(), 0, &staging, 0, size as u64);
        queue.submit(std::iter::once(encoder.finish()));

        // Use a oneshot channel to drive the async map.
        let (tx, rx) = tokio::sync::oneshot::channel::<std::result::Result<(), wgpu::BufferAsyncError>>();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

        device.poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| crate::error::FerrisResError::Gpu(format!("poll error: {:?}", e)))?;

        rx.await
            .map_err(|_| crate::error::FerrisResError::Gpu("oneshot channel dropped".to_string()))?
            .map_err(|e| crate::error::FerrisResError::Gpu(format!("map_async error: {:?}", e)))?;

        let mapped = staging.slice(..).get_mapped_range();
        let data: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();

        Ok(Self {
            data,
            numel,
            accumulators: HashMap::new(),
            micro_batch_count: 0,
        })
    }

    /// Upload the CPU gradient data back to a new GPU buffer suitable for
    /// use as the `grad` argument to `AdamOptimizer::step`.
    pub fn to_gpu_buffer(&self, device: &Arc<Device>, queue: &wgpu::Queue) -> Result<GpuBuffer> {
        let bytes: &[u8] = bytemuck::cast_slice(&self.data);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cpu_grad_upload"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytes);
        Ok(GpuBuffer::from_existing(buf, bytes.len()))
    }
}

// ── Standalone helpers ────────────────────────────────────────────────────────

/// Read back the contents of a [`GpuBuffer`] to a CPU `Vec<f32>`.
///
/// Creates a temporary `MAP_READ` staging buffer, copies the GPU buffer into it,
/// submits the work, polls the device to completion, and returns the data.
///
/// This is the synchronous variant suitable for the training loop on
/// `DeviceProfile::Integrated` devices where the training loop is not itself
/// async.  For pipelined async readback see [`AsyncGradientOffload`].
pub fn offload_gradients_to_cpu(
    device: &Arc<Device>,
    queue: &Arc<wgpu::Queue>,
    grad_buffer: &GpuBuffer,
) -> Result<Vec<f32>> {
    let size = grad_buffer.size();
    let numel = size / std::mem::size_of::<f32>();

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_offload_staging"),
        size: size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cpu_offload_readback"),
    });
    encoder.copy_buffer_to_buffer(grad_buffer.buffer(), 0, &staging, 0, size as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| {
        FerrisResError::Device(format!("offload_gradients_to_cpu: poll failed: {:?}", e))
    })?;
    rx.recv()
        .map_err(|e| FerrisResError::Device(format!("offload_gradients_to_cpu: channel error: {}", e)))?
        .map_err(|e| FerrisResError::Device(format!("offload_gradients_to_cpu: map_async error: {:?}", e)))?;

    let mapped = slice.get_mapped_range();
    let data: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped)
        .iter()
        .copied()
        .collect();
    drop(mapped);
    staging.unmap();

    tracing::debug!("offload_gradients_to_cpu: read {} f32 values", numel);
    Ok(data)
}

/// Upload CPU gradient data back into a [`GpuBuffer`].
///
/// Uses `queue.write_buffer` which performs a host→device copy without requiring
/// a staging buffer (wgpu internally manages the upload).
///
/// # Errors
/// Returns an error if the byte size of `data` does not match `target.size()`.
pub fn upload_gradients_from_cpu(
    queue: &Arc<wgpu::Queue>,
    data: &[f32],
    target: &GpuBuffer,
) -> Result<()> {
    let byte_len = data.len() * std::mem::size_of::<f32>();
    if byte_len != target.size() {
        return Err(FerrisResError::Device(format!(
            "upload_gradients_from_cpu: size mismatch: data={} bytes, target={} bytes",
            byte_len,
            target.size()
        )));
    }
    queue.write_buffer(target.buffer(), 0, bytemuck::cast_slice(data));
    tracing::debug!("upload_gradients_from_cpu: wrote {} f32 values", data.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_gradient_buffer_accumulate_and_normalize() {
        let mut buf = CpuGradientBuffer::new_accumulator();
        buf.register("w", 4);

        // Accumulate two micro-batches
        buf.accumulate("w", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        buf.increment_micro_batch();
        buf.accumulate("w", &[3.0, 2.0, 1.0, 0.0]).unwrap();
        buf.increment_micro_batch();

        assert_eq!(buf.micro_batch_count(), 2);
        assert_eq!(buf.get("w").unwrap(), &[4.0, 4.0, 4.0, 4.0]);

        // Normalize by K=2 — this is the divide-by-K fix
        buf.normalize(2);
        let result = buf.get("w").unwrap();
        for v in result {
            assert!((v - 2.0).abs() < 1e-6, "expected 2.0, got {}", v);
        }
    }

    #[test]
    fn test_cpu_gradient_buffer_reset() {
        let mut buf = CpuGradientBuffer::new_accumulator();
        buf.register("w", 2);
        buf.accumulate("w", &[5.0, 10.0]).unwrap();
        buf.increment_micro_batch();
        buf.reset();
        assert_eq!(buf.micro_batch_count(), 0);
        assert_eq!(buf.get("w").unwrap(), &[0.0, 0.0]);
    }

    #[test]
    fn test_cpu_gradient_buffer_normalize_k1_noop() {
        let mut buf = CpuGradientBuffer::new_accumulator();
        buf.register("w", 3);
        buf.accumulate("w", &[6.0, 9.0, 12.0]).unwrap();
        buf.normalize(1); // should be a no-op
        assert_eq!(buf.get("w").unwrap(), &[6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_cpu_gradient_buffer_size_mismatch_error() {
        let mut buf = CpuGradientBuffer::new_accumulator();
        buf.register("w", 4);
        let result = buf.accumulate("w", &[1.0, 2.0]); // wrong size
        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_gradient_buffer_new_single_buffer_mode() {
        // Verify backward-compatible single-buffer mode still works
        let buf = CpuGradientBuffer::new(4);
        assert_eq!(buf.numel, 4);
        assert_eq!(buf.data, vec![0.0f32; 4]);
    }

    #[test]
    fn test_cpu_gradient_buffer_multi_param() {
        let mut buf = CpuGradientBuffer::new_accumulator();
        buf.register("w1", 2);
        buf.register("w2", 3);

        buf.accumulate("w1", &[1.0, 2.0]).unwrap();
        buf.accumulate("w2", &[3.0, 4.0, 5.0]).unwrap();
        buf.increment_micro_batch();

        buf.accumulate("w1", &[1.0, 2.0]).unwrap();
        buf.accumulate("w2", &[3.0, 4.0, 5.0]).unwrap();
        buf.increment_micro_batch();

        // Sum over K=2 micro-batches, then normalize
        buf.normalize(2);

        let w1 = buf.get("w1").unwrap();
        let w2 = buf.get("w2").unwrap();
        assert!((w1[0] - 1.0).abs() < 1e-6, "w1[0]={}", w1[0]);
        assert!((w1[1] - 2.0).abs() < 1e-6, "w1[1]={}", w1[1]);
        assert!((w2[0] - 3.0).abs() < 1e-6, "w2[0]={}", w2[0]);
        assert!((w2[1] - 4.0).abs() < 1e-6, "w2[1]={}", w2[1]);
        assert!((w2[2] - 5.0).abs() < 1e-6, "w2[2]={}", w2[2]);
    }
}

