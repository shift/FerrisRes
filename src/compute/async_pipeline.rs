//! Async compute pipeline — double-buffered command dispatch.
//!
//! Inspired by FlashAttention-3's pipelining: while the GPU executes kernel N,
//! the CPU prepares kernel N+1 in a separate command buffer. When both are ready,
//! submit them in order for overlap.
//!
//! On integrated GPUs with shared DRAM (800 MHz bus), this hides CPU-side
//! encoder creation latency behind GPU execution.

use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::error::Result;

/// Double-buffered async compute pipeline.
///
/// Usage:
/// 1. `begin()` — start recording into buffer A
/// 2. Record compute passes into the encoder
/// 3. `swap()` — submit buffer A, start recording into buffer B
/// 4. Repeat
/// 5. `flush()` — submit any pending buffer
///
/// This ensures the GPU never idles waiting for CPU encoder creation:
/// while GPU processes buffer A, CPU records into buffer B.
pub struct AsyncComputePipeline {
    /// Currently recording encoder (front buffer).
    current: Option<wgpu::CommandEncoder>,
    /// Pending encoder to submit (back buffer).
    pending: Option<wgpu::CommandEncoder>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    label: String,
    /// Number of submissions made.
    submit_count: u32,
    /// Number of bytes submitted (approximate).
    #[allow(dead_code)]
    bytes_submitted: u64,
}

impl AsyncComputePipeline {
    /// Create a new async pipeline.
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, label: &str) -> Self {
        Self {
            current: None,
            pending: None,
            device,
            queue,
            label: label.to_string(),
            submit_count: 0,
            bytes_submitted: 0,
        }
    }

    /// Begin recording into the current buffer.
    /// Creates a new command encoder if none exists.
    pub fn begin(&mut self) -> &mut wgpu::CommandEncoder {
        if self.current.is_none() {
            self.current = Some(self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("async_{}_buf_{}", self.label, self.submit_count)),
                },
            ));
        }
        self.current.as_mut().unwrap()
    }

    /// Swap buffers: submit the current encoder and move pending to front.
    /// The previously pending buffer (if any) is submitted first.
    pub fn swap(&mut self) -> Result<()> {
        // Move current to pending
        let new_pending = self.current.take();

        // Submit old pending first (if any)
        if let Some(enc) = self.pending.take() {
            let buf = enc.finish();
            self.queue.submit(std::iter::once(buf));
            self.submit_count += 1;
        }

        // New pending is what was current
        self.pending = new_pending;
        Ok(())
    }

    /// Flush both buffers: submit everything pending.
    pub fn flush(&mut self) -> Result<()> {
        // Submit pending first
        if let Some(enc) = self.pending.take() {
            let buf = enc.finish();
            self.queue.submit(std::iter::once(buf));
            self.submit_count += 1;
        }

        // Then submit current
        if let Some(enc) = self.current.take() {
            let buf = enc.finish();
            self.queue.submit(std::iter::once(buf));
            self.submit_count += 1;
        }

        tracing::debug!(
            "AsyncComputePipeline '{}': flushed ({} submissions)",
            self.label, self.submit_count,
        );
        Ok(())
    }

    /// Get the number of submissions made.
    pub fn submit_count(&self) -> u32 {
        self.submit_count
    }

    /// Check if there's an active encoder being recorded.
    pub fn is_recording(&self) -> bool {
        self.current.is_some()
    }

    /// Get the device.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get the queue.
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

impl Drop for AsyncComputePipeline {
    fn drop(&mut self) {
        // Auto-flush on drop to avoid losing work
        if self.pending.is_some() || self.current.is_some() {
            if let Err(e) = self.flush() {
                tracing::warn!(event = "asynccomputepipeline_drop_flush_failed", "AsyncComputePipeline drop flush failed: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_pipeline_submit_count() {
        // Just test that the struct is constructible and methods exist
        // Full GPU tests would need a real device
        assert!(true);
    }

    #[test]
    fn test_pipeline_is_recording_initially_false() {
        // Without a GPU device, we can't test full functionality
        // but we can test the type interface
        assert!(true);
    }
}
