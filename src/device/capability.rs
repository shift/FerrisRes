use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuKind {
    Discrete,
    Integrated,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub vram_mb: u64,
    pub shared_ram_mb: u64,
    pub gpu_kind: GpuKind,
    pub max_compute_workgroup_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_storage_buffer_range: u64,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_bind_groups: u32,
    pub backend: String,
    pub adapter_name: String,
}

impl Capability {
    pub fn detect() -> Self {
        let mut cap = Self {
            vram_mb: 0,
            shared_ram_mb: 0,
            gpu_kind: GpuKind::Other,
            max_compute_workgroup_size: 0,
            max_compute_invocations_per_workgroup: 0,
            max_storage_buffer_range: 0,
            max_storage_buffers_per_shader_stage: 0,
            max_bind_groups: 0,
            backend: String::new(),
            adapter_name: String::new(),
        };

        match detect_vram_ash() {
            Some((vram, name, kind)) => {
                cap.vram_mb = vram;
                cap.adapter_name = name;
                cap.gpu_kind = kind;
                info!(vram_mb = vram, kind = ?kind, method = "ash", "Detected GPU via Vulkan");
            }
            None => {
                warn!("ash VRAM detection failed, trying sysfs fallback");
                match detect_vram_sysfs() {
                    Some((vram, name)) => {
                        cap.vram_mb = vram;
                        cap.adapter_name = name;
                        info!(vram_mb = vram, method = "sysfs", "Detected GPU VRAM via sysfs");
                    }
                    None => {
                        warn!("No GPU VRAM detected via any method");
                    }
                }
            }
        }

        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();
        cap.shared_ram_mb = sys.total_memory() / (1024 * 1024);

        cap
    }

    pub fn with_adapter_limits(mut self, limits: &wgpu::Limits, adapter_info: &wgpu::AdapterInfo) -> Self {
        self.max_compute_workgroup_size = limits.max_compute_workgroup_size_x
            .max(limits.max_compute_workgroup_size_y)
            .max(limits.max_compute_workgroup_size_z);
        self.max_compute_invocations_per_workgroup = limits.max_compute_invocations_per_workgroup;
        self.max_storage_buffer_range = limits.max_storage_buffer_binding_size;
        self.max_storage_buffers_per_shader_stage = limits.max_storage_buffers_per_shader_stage;
        self.max_bind_groups = limits.max_bind_groups;
        self.backend = format!("{:?}", adapter_info.backend);
        if self.adapter_name.is_empty() {
            self.adapter_name = adapter_info.name.clone();
        }
        self
    }

    pub fn effective_vram_mb(&self) -> u64 {
        match self.gpu_kind {
            GpuKind::Discrete if self.vram_mb > 0 => self.vram_mb,
            _ => self.shared_ram_mb,
        }
    }
}

fn detect_vram_ash() -> Option<(u64, String, GpuKind)> {
    let entry = unsafe { ash::Entry::load().ok()? };

    let mut app_info = ash::vk::ApplicationInfo::default();
    app_info.api_version = ash::vk::API_VERSION_1_0;

    let extension_names: Vec<*const i8> = vec![ash::ext::debug_utils::NAME.as_ptr()];

    let create_info = ash::vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names);

    let instance = unsafe { entry.create_instance(&create_info, None).ok()? };

    let physical_devices = unsafe { instance.enumerate_physical_devices().ok()? };

    let mut best: Option<(u64, String, GpuKind)> = None;

    for &pd in &physical_devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };

        let name = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        if name.contains("llvmpipe") || name.contains("lavapipe") || name.contains("swrast") {
            warn!(adapter = %name, "Skipping software Vulkan renderer");
            continue;
        }

        let gpu_kind = match props.device_type {
            ash::vk::PhysicalDeviceType::DISCRETE_GPU => GpuKind::Discrete,
            ash::vk::PhysicalDeviceType::INTEGRATED_GPU => GpuKind::Integrated,
            _ => GpuKind::Other,
        };

        let mut total_vram: u64 = 0;
        for i in 0..mem_props.memory_heap_count {
            let heap = mem_props.memory_heaps[i as usize];
            if heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL) {
                total_vram += heap.size;
            }
        }

        let vram_mb = total_vram / (1024 * 1024);

        if gpu_kind != GpuKind::Discrete {
            continue;
        }

        if best.as_ref().map_or(true, |(prev, _, _)| vram_mb > *prev) {
            best = Some((vram_mb, name, gpu_kind));
        }
    }

    if best.is_none() {
        for &pd in &physical_devices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let name = unsafe {
                std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };

            if name.contains("llvmpipe") || name.contains("lavapipe") || name.contains("swrast") {
                continue;
            }

            let gpu_kind = match props.device_type {
                ash::vk::PhysicalDeviceType::INTEGRATED_GPU => GpuKind::Integrated,
                _ => GpuKind::Other,
            };

            best = Some((0, name, gpu_kind));
            break;
        }
    }

    unsafe { instance.destroy_instance(None) };
    best
}

fn detect_vram_sysfs() -> Option<(u64, String)> {
    let card_dirs = std::fs::read_dir("/sys/class/drm/card0/device")
        .or_else(|_| std::fs::read_dir("/sys/class/drm/card1/device"))
        .ok()?;

    for entry in card_dirs.flatten() {
        let path = entry.path();
        let name_path = path.join("vendor");
        if !name_path.exists() {
            continue;
        }

        if let Ok(vram_str) = std::fs::read_to_string(path.join("mem_info_vram_total")) {
            if let Ok(vram_kb) = vram_str.trim().parse::<u64>() {
                let name = std::fs::read_to_string(path.join("vendor"))
                    .map(|v| format!("vendor:{}", v.trim()))
                    .unwrap_or_else(|_| "unknown".into());
                return Some((vram_kb / 1024, name));
            }
        }

        if let Ok(vram_str) = std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_vram_total") {
            if let Ok(vram_kb) = vram_str.trim().parse::<u64>() {
                return Some((vram_kb / 1024, "GPU (sysfs)".into()));
            }
        }
    }

    None
}
