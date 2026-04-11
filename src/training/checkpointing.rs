use std::collections::HashMap;
use std::sync::Arc;
use wgpu::Device;
use crate::compute::buffer::GpuBuffer;
use crate::compute::turboquant::{TurboQuantConfig, TurboQuantEngine};
use crate::error::Result;

/// Configuration for TurboQuant compression in checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointCompressionConfig {
    /// Whether compression is enabled for checkpoint storage
    pub enabled: bool,
    /// Bit width for compression (2, 3, or 4)
    pub bit_width: Option<u32>,
    /// Memory savings ratio (computed from bit_width)
    pub compression_ratio: f32,
}

impl Default for CheckpointCompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bit_width: None,
            compression_ratio: 1.0,
        }
    }
}

impl CheckpointCompressionConfig {
    /// Enable checkpoint compression with specified bit width
    pub fn new(bit_width: u32) -> Self {
        let ratio = 32.0 / bit_width as f32;
        Self {
            enabled: true,
            bit_width: Some(bit_width),
            compression_ratio: ratio,
        }
    }
    
    /// Enable 2-bit compression (16x reduction)
    pub fn two_bit() -> Self {
        Self::new(2)
    }
    
    /// Enable 3-bit compression (10.7x reduction)
    pub fn three_bit() -> Self {
        Self::new(3)
    }
}

/// Stores saved activation GpuBuffers indexed by (block_number, label).
/// Used by gradient checkpointing to save the hidden_states input at block
/// boundaries instead of keeping all intermediate activations live.
///
/// NOTE: Full recomputation during backward is not yet implemented.
/// The store is populated during forward and will be consumed by a future
/// recompute pass. See TODO markers below.
///
/// TurboQuant compression can be enabled to reduce checkpoint memory.
pub struct CheckpointStore {
    /// Map from (block_index, label) -> saved buffer copy.
    saved: HashMap<(usize, &'static str), GpuBuffer>,
    /// Optional compressed checkpoint storage
    #[allow(dead_code)]
    compressed: HashMap<(usize, &'static str), GpuBuffer>,
    /// TurboQuant configuration for compression
    compression_config: CheckpointCompressionConfig,
    /// TurboQuant engine (for compression/decompression)
    #[allow(dead_code)]
    turboquant: Option<TurboQuantEngine>,
    device: Arc<Device>,
}

impl CheckpointStore {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            saved: HashMap::new(),
            compressed: HashMap::new(),
            compression_config: CheckpointCompressionConfig::default(),
            turboquant: None,
            device,
        }
    }
    
    /// Create checkpoint store with TurboQuant compression enabled
    #[allow(dead_code)]
    pub fn with_compression(
        device: Arc<Device>,
        compression: CheckpointCompressionConfig,
    ) -> std::result::Result<Self, crate::error::FerrisResError> {
        let turboquant = if compression.enabled {
            let bit_width = compression.bit_width.unwrap_or(2);
            let tq_config = TurboQuantConfig::new(bit_width, 512, false);
            match TurboQuantEngine::new(tq_config) {
                Ok(engine) => Some(engine),
                Err(e) => return Err(crate::error::FerrisResError::Device(format!(
                    "Failed to create TurboQuant engine: {}", e
                ))),
            }
        } else {
            None
        };
        
        Ok(Self {
            saved: HashMap::new(),
            compressed: HashMap::new(),
            compression_config: compression,
            turboquant,
            device,
        })
    }
    
    /// Get memory savings from compression
    pub fn compression_ratio(&self) -> f32 {
        self.compression_config.compression_ratio
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
