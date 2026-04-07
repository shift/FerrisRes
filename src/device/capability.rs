pub struct Capability {
    pub vram_mb: u64,
    pub shared_ram_mb: u64,
    pub max_compute_workgroup_size: u32,
    pub max_storage_buffer_range: u64,
}

impl Capability {
    pub fn new(vram_mb: u64, shared_ram_mb: u64) -> Self {
        Self {
            vram_mb,
            shared_ram_mb,
            max_compute_workgroup_size: 0,
            max_storage_buffer_range: 0,
        }
    }

    pub fn effective_vram_mb(&self) -> u64 {
        if self.vram_mb > 0 {
            self.vram_mb
        } else {
            self.shared_ram_mb
        }
    }
}
