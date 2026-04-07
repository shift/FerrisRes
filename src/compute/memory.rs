use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use wgpu::Device;

use crate::compute::buffer::GpuBuffer;
use crate::device::capability::Capability;
use crate::device::profile::{ComputeMode, DeviceProfile};
use crate::error::{FerrisResError, Result};

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
        let tile_size_bytes = effective_vram / 4;
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
