use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use wgpu::Device;

use crate::compute::buffer::GpuBuffer;
use crate::compute::cache::BlockCache;
use crate::device::capability::{Capability, GpuKind};
use crate::device::profile::{ComputeMode, DeviceProfile};
use crate::error::{FerrisResError, Result};

/// Memory coalescing configuration based on GPU type
#[derive(Debug, Clone)]
pub struct MemoryCoalescingConfig {
    /// Preferred alignment in bytes for memory transfers
    pub alignment_bytes: u32,
    /// Whether to prefer sequential access patterns
    pub prefer_sequential: bool,
    /// Whether to use double buffering
    pub use_double_buffering: bool,
    /// Chunk size for batched transfers
    pub chunk_size: u32,
}

impl MemoryCoalescingConfig {
    /// Create config based on GPU kind
    pub fn for_gpu_kind(kind: GpuKind) -> Self {
        match kind {
            GpuKind::Discrete => Self {
                alignment_bytes: 256,
                prefer_sequential: true,
                use_double_buffering: true,
                chunk_size: 4096,
            },
            GpuKind::Integrated => Self {
                // Integrated GPUs benefit from simpler access patterns
                alignment_bytes: 64,
                prefer_sequential: true,
                use_double_buffering: false,
                chunk_size: 1024,
            },
            GpuKind::Other => Self::default(),
        }
    }
}

impl Default for MemoryCoalescingConfig {
    fn default() -> Self {
        Self {
            alignment_bytes: 64,
            prefer_sequential: true,
            use_double_buffering: false,
            chunk_size: 1024,
        }
    }
}

/// Maximum tile size in bytes for integrated / non-discrete GPUs.
/// This caps the tile budget so that a single tile does not consume all available VRAM,
/// leaving room for model parameters, optimizer state, and activations.
/// Derived from ADR research (41e79f09): 4 tiles of 512 MB = 2 GB coverage with
/// ~1.5 GB headroom for a 125M-parameter model on DDR4-2666 iGPU.
const INTEGRATED_MAX_TILE_BYTES: u64 = 512 * 1024 * 1024;

// ─── DeviceMemoryPhase ────────────────────────────────────────────────────────

/// Tracks whether the runtime is currently in inference or training mode.
/// Used by [`BorrowedBufferPool`] to gate KV-cache buffer reuse for gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryPhase {
    /// Normal inference: KV-cache buffers are owned by [`BlockCache`].
    Inference,
    /// Backward-pass / optimizer step: KV-cache buffers are available for
    /// gradient accumulation, returned when training completes.
    Training,
}

// ─── BorrowedBufferPool ───────────────────────────────────────────────────────

/// Buffer-borrow strategy for integrated GPU profiles.
///
/// On integrated GPUs, CPU and GPU share DRAM, so we can reuse KV-cache
/// buffers as gradient scratch space during the training phase instead of
/// allocating a second set of equally-sized buffers.  Only enabled when
/// [`DeviceProfile::Integrated`] is in use.
///
/// # Lifecycle
/// ```text
/// inference  ──transition_to_training()──▶  training
///                (kv buffers extracted)      (kv buffers used as grad bufs)
/// training   ──transition_to_inference()──▶  inference
///                (grad bufs returned)         (kv cache restored)
/// ```
pub struct BorrowedBufferPool {
    phase: DeviceMemoryPhase,
    /// KV-cache buffers borrowed for gradient use during training.
    borrowed_kv_buffers: Vec<GpuBuffer>,
    /// Whether this pool is active (only for Integrated profile).
    enabled: bool,
}

impl BorrowedBufferPool {
    /// Create a new pool.  Pass `enabled = true` only for
    /// [`DeviceProfile::Integrated`].
    pub fn new(profile: &DeviceProfile) -> Self {
        Self {
            phase: DeviceMemoryPhase::Inference,
            borrowed_kv_buffers: Vec::new(),
            enabled: *profile == DeviceProfile::Integrated,
        }
    }

    /// Returns `true` if the borrowed-buffer strategy is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the current memory phase.
    pub fn phase(&self) -> DeviceMemoryPhase {
        self.phase
    }

    /// Transition from inference to training.
    ///
    /// Extracts the underlying [`GpuBuffer`] from the KV cache so it can be
    /// reused as gradient scratch space.  After calling this function the
    /// [`BlockCache`] is consumed; the caller must not use it until
    /// [`transition_to_inference`] restores it.
    ///
    /// If the pool is disabled (non-integrated profile) this is a no-op and
    /// returns an empty `Vec`.
    ///
    /// # Panics
    /// Panics if called while already in [`DeviceMemoryPhase::Training`].
    pub fn transition_to_training(&mut self, kv_cache: BlockCache) -> Vec<GpuBuffer> {
        assert_eq!(
            self.phase,
            DeviceMemoryPhase::Inference,
            "BorrowedBufferPool: already in Training phase"
        );
        self.phase = DeviceMemoryPhase::Training;

        if !self.enabled {
            // Return the backing buffer without consuming the cache so callers
            // can still treat the result as "no borrowed buffers".
            drop(kv_cache);
            return Vec::new();
        }

        // Extract the backing GPU buffer from the cache and hand it to the
        // gradient accumulator.
        let buf = kv_cache.into_buffer();
        self.borrowed_kv_buffers.push(buf);
        self.borrowed_kv_buffers.drain(..).collect()
    }

    /// Transition from training back to inference.
    ///
    /// Restores gradient buffers to KV-cache use by reconstructing a
    /// [`BlockCache`] from the returned buffers.
    ///
    /// If the pool is disabled this reconstructs a fresh cache from the
    /// provided parameters.
    ///
    /// # Panics
    /// Panics if called while already in [`DeviceMemoryPhase::Inference`].
    pub fn transition_to_inference(
        &mut self,
        gradient_bufs: Vec<GpuBuffer>,
        device: Arc<Device>,
        hidden_dim: usize,
        cache_capacity: usize,
    ) -> Result<BlockCache> {
        assert_eq!(
            self.phase,
            DeviceMemoryPhase::Training,
            "BorrowedBufferPool: already in Inference phase"
        );
        self.phase = DeviceMemoryPhase::Inference;

        if self.enabled && !gradient_bufs.is_empty() {
            // Reconstruct the BlockCache by handing back the first buffer.
            // Remaining buffers (if any) are dropped — the cache only needs one.
            let mut bufs = gradient_bufs;
            let backing = bufs.remove(0);
            Ok(BlockCache::from_buffer(
                device,
                backing,
                hidden_dim,
                cache_capacity,
            ))
        } else {
            // Disabled path or no buffers returned: allocate a fresh cache.
            BlockCache::new(device, hidden_dim, cache_capacity)
        }
    }
}

pub struct MemoryBudget {
    total_vram_bytes: u64,
    reserved_bytes: u64,
    allocated_bytes: AtomicU64,
    device: Arc<Device>,
}

impl MemoryBudget {
    pub fn new(device: Arc<Device>, capability: &Capability) -> Self {
        let total_vram_bytes = capability.effective_vram_mb() * 1024 * 1024;
        let reserved_bytes = total_vram_bytes / 10;
        Self {
            total_vram_bytes,
            reserved_bytes,
            allocated_bytes: AtomicU64::new(0),
            device,
        }
    }

    pub fn allocate(&self, size: u64) -> Result<u64> {
        loop {
            let current = self.allocated_bytes.load(Ordering::SeqCst);
            let new = current.checked_add(size).ok_or_else(|| {
                FerrisResError::OOM(format!(
                    "Requested {} bytes overflows allocation tracker (current: {})",
                    size, current
                ))
            })?;

            let usable = self.total_vram_bytes.saturating_sub(self.reserved_bytes);
            if new > usable {
                return Err(FerrisResError::OOM(format!(
                    "Cannot allocate {} bytes: {} bytes already used of {} usable ({} reserved from {} total)",
                    size, current, usable, self.reserved_bytes, self.total_vram_bytes
                )));
            }

            if self
                .allocated_bytes
                .compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Ok(current);
            }
        }
    }

    pub fn deallocate(&self, size: u64) {
        let current = self.allocated_bytes.load(Ordering::SeqCst);
        let new = current.saturating_sub(size);
        self.allocated_bytes.store(new, Ordering::SeqCst);
    }

    pub fn available(&self) -> u64 {
        let usable = self.total_vram_bytes.saturating_sub(self.reserved_bytes);
        usable.saturating_sub(self.allocated_bytes.load(Ordering::SeqCst))
    }

    pub fn utilization(&self) -> f64 {
        let usable = self.total_vram_bytes.saturating_sub(self.reserved_bytes);
        if usable == 0 {
            return 0.0;
        }
        let allocated = self.allocated_bytes.load(Ordering::SeqCst);
        allocated as f64 / usable as f64
    }

    pub fn total_vram_bytes(&self) -> u64 {
        self.total_vram_bytes
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes.load(Ordering::SeqCst)
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

pub struct MemoryPool {
    budget: MemoryBudget,
    pools: Mutex<HashMap<usize, Vec<GpuBuffer>>>,
    device: Arc<Device>,
}

impl MemoryPool {
    pub fn new(device: Arc<Device>, capability: &Capability) -> Self {
        let budget = MemoryBudget::new(device.clone(), capability);
        Self {
            budget,
            pools: Mutex::new(HashMap::new()),
            device,
        }
    }

    pub fn get(&self, size: usize) -> Result<GpuBuffer> {
        let mut pools = self.pools.lock().map_err(|e| {
            FerrisResError::Device(format!("Memory pool lock poisoned: {}", e))
        })?;

        if let Some(bucket) = pools.get_mut(&size) {
            if let Some(buffer) = bucket.pop() {
                return Ok(buffer);
            }
        }

        self.budget.allocate(size as u64)?;
        let buffer = GpuBuffer::new(&self.device, size, Some("pooled_buffer"))?;
        Ok(buffer)
    }

    pub fn return_buffer(&self, buffer: GpuBuffer) {
        let size = buffer.size();
        let mut pools = match self.pools.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        pools.entry(size).or_default().push(buffer);
    }

    pub fn clear(&self) {
        let mut pools = match self.pools.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        pools.clear();
    }

    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledCompute {
    tile_size_bytes: u64,
    max_tiles: u32,
}

impl TiledCompute {
    pub fn new(capability: &Capability) -> Self {
        let effective_vram = capability.effective_vram_mb() * 1024 * 1024;
        let raw_tile = effective_vram / 4;

        // Bug fix (ADR-013 / task b2965655): for integrated / non-discrete GPUs
        // `effective_vram` reflects shared system RAM (e.g. 8 GB), so the
        // uncapped formula produces a 2 GB tile that consumes the entire compute
        // budget.  Cap at 512 MB for non-discrete devices so that at least four
        // tiles fit simultaneously alongside model parameters and optimizer state.
        let tile_size_bytes = match capability.gpu_kind {
            GpuKind::Discrete => raw_tile,
            _ => raw_tile.min(INTEGRATED_MAX_TILE_BYTES),
        };

        let max_tiles = 64;
        Self {
            tile_size_bytes: tile_size_bytes.max(1),
            max_tiles,
        }
    }

    pub fn tile_count(&self, total_bytes: u64) -> u32 {
        if total_bytes == 0 {
            return 0;
        }
        let count = (total_bytes + self.tile_size_bytes - 1) / self.tile_size_bytes;
        count.min(self.max_tiles as u64) as u32
    }

    pub fn tile_dims(
        &self,
        rows: u32,
        cols: u32,
        elem_size: u32,
    ) -> Vec<(u32, u32)> {
        let total_bytes = rows as u64 * cols as u64 * elem_size as u64;
        let tiles = self.tile_count(total_bytes);
        if tiles <= 1 {
            return vec![(rows, cols)];
        }

        let mut dims = Vec::with_capacity(tiles as usize);
        let mut remaining_rows = rows;
        let rows_per_tile = (rows as u64 + tiles as u64 - 1) / tiles as u64;

        while remaining_rows > 0 {
            let chunk = remaining_rows.min(rows_per_tile as u32);
            dims.push((chunk, cols));
            remaining_rows -= chunk;
        }

        dims
    }

    pub fn tile_size_bytes(&self) -> u64 {
        self.tile_size_bytes
    }

    pub fn max_tiles(&self) -> u32 {
        self.max_tiles
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDispatcher {
    FullGpu,
    Tiled(TiledCompute),
    CpuOffload,
}

impl ComputeDispatcher {
    pub fn from_profile(profile: &DeviceProfile) -> Self {
        match profile.compute_mode() {
            ComputeMode::FullGpu => ComputeDispatcher::FullGpu,
            ComputeMode::Tiled => ComputeDispatcher::Tiled(TiledCompute {
                tile_size_bytes: 0,
                max_tiles: 64,
            }),
            ComputeMode::CpuOffload => ComputeDispatcher::CpuOffload,
        }
    }

    pub fn from_profile_with_capability(
        profile: &DeviceProfile,
        capability: &Capability,
    ) -> Self {
        match profile.compute_mode() {
            ComputeMode::FullGpu => ComputeDispatcher::FullGpu,
            ComputeMode::Tiled => ComputeDispatcher::Tiled(TiledCompute::new(capability)),
            ComputeMode::CpuOffload => ComputeDispatcher::CpuOffload,
        }
    }

    pub fn is_tiled(&self) -> bool {
        matches!(self, ComputeDispatcher::Tiled(_))
    }

    pub fn is_cpu_offload(&self) -> bool {
        matches!(self, ComputeDispatcher::CpuOffload)
    }
}
