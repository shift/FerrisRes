use std::collections::HashMap;
use std::sync::Arc;
use wgpu::Device;
use crate::compute::buffer::GpuBuffer;
use crate::error::Result;

/// Stores saved activation GpuBuffers indexed by (block_number, label).
/// Used by gradient checkpointing to save the hidden_states input at block
/// boundaries instead of keeping all intermediate activations live.
///
/// NOTE: Full recomputation during backward is not yet implemented.
/// The store is populated during forward and will be consumed by a future
/// recompute pass. See TODO markers below.
pub struct CheckpointStore {
    /// Map from (block_index, label) -> saved buffer copy.
    saved: HashMap<(usize, &'static str), GpuBuffer>,
    device: Arc<Device>,
}

impl CheckpointStore {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            saved: HashMap::new(),
            device,
        }
    }

    /// Save a copy of `src` buffer identified by `(block_idx, label)`.
    /// The copy is made on the GPU command encoder; the caller must submit
    /// the encoder before the saved data is guaranteed to be present.
    pub fn save(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        block_idx: usize,
        label: &'static str,
        src: &GpuBuffer,
    ) -> Result<()> {
        let dst = GpuBuffer::new(&self.device, src.size(), Some(&format!("ckpt_b{}_{}",block_idx, label)))?;
        encoder.copy_buffer_to_buffer(src.buffer(), 0, dst.buffer(), 0, src.size() as u64);
        self.saved.insert((block_idx, label), dst);
        tracing::debug!("CheckpointStore: saved block={} label={} size={}", block_idx, label, src.size());
        Ok(())
    }

    /// Retrieve a previously saved buffer (immutable borrow).
    /// Returns `None` if no checkpoint was saved for this key.
    pub fn get(&self, block_idx: usize, label: &'static str) -> Option<&GpuBuffer> {
        self.saved.get(&(block_idx, label))
    }

    /// Remove and return a saved buffer (consumes the entry).
    pub fn take(&mut self, block_idx: usize, label: &'static str) -> Option<GpuBuffer> {
        self.saved.remove(&(block_idx, label))
    }

    /// Drop all saved checkpoints to free GPU memory.
    pub fn clear(&mut self) {
        self.saved.clear();
        tracing::debug!("CheckpointStore: cleared all checkpoints");
    }

    // TODO(ADR-010): implement recompute_block() that re-runs forward for a
    // single block using the saved hidden_states input, restoring intermediate
    // activations needed for gradient computation.
}
