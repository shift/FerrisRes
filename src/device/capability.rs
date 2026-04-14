use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuKind {
    Discrete,
    Integrated,
    Other,
}

/// GPU vendor identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Qualcomm,
    Unknown,
}

impl GpuVendor {
    pub fn from_adapter_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();
        if name_lower.contains("nvidia") || name_lower.contains("geforce") || name_lower.contains("rtx") {
            GpuVendor::Nvidia
        } else if name_lower.contains("amd") || name_lower.contains("radeon") {
            GpuVendor::Amd
        } else if name_lower.contains("intel") || name_lower.contains("uhd") {
            GpuVendor::Intel
        } else if name_lower.contains("apple") || name_lower.contains("metal") {
            GpuVendor::Apple
        } else if name_lower.contains("qualcomm") || name_lower.contains("adreno") {
            GpuVendor::Qualcomm
        } else {
            GpuVendor::Unknown
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub vram_mb: u64,
    pub shared_ram_mb: u64,
    pub gpu_kind: GpuKind,
    pub vendor: GpuVendor,
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
            vendor: GpuVendor::Unknown,
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
                cap.adapter_name = name.clone();
                cap.gpu_kind = kind;
                cap.vendor = GpuVendor::from_adapter_name(&name);
                info!(event = "gpu_detected", vram_mb = vram, kind = ?kind, vendor = ?cap.vendor, method = "ash", "detected GPU via Vulkan");
            }
            None => {
                warn!(event = "ash_vram_detection_failed_trying_sysfs", "ash VRAM detection failed, trying sysfs fallback");
                match detect_vram_sysfs() {
                    Some((vram, name)) => {
                        cap.vram_mb = vram;
                        cap.adapter_name = name.clone();
                        cap.vendor = GpuVendor::from_adapter_name(&name);
                        info!(event = "gpu_detected", vram_mb = vram, vendor = ?cap.vendor, method = "sysfs", "detected GPU VRAM via sysfs");
                    }
                    None => {
                        warn!(event = "no_gpu_vram_detected_via_any", "No GPU VRAM detected via any method");
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
    
    /// Get vendor-specific compute tuning hints
    pub fn vendor_tuning(&self) -> VendorTuning {
        match self.vendor {
            GpuVendor::Nvidia => VendorTuning {
                prefer_warps: true,
                warp_size: 32,
                align_to_warp: true,
                prefer_fp32_atomic: false,
                l2_benefits_from_padding: true,
            },
            GpuVendor::Amd => VendorTuning {
                prefer_warps: true,
                warp_size: 32,
                align_to_warp: true,
                prefer_fp32_atomic: true,
                l2_benefits_from_padding: false,
            },
            GpuVendor::Intel => VendorTuning {
                prefer_warps: false,
                warp_size: 32,
                align_to_warp: false,
                prefer_fp32_atomic: false,
                l2_benefits_from_padding: true,
            },
            GpuVendor::Apple => VendorTuning {
                prefer_warps: true,
                warp_size: 32,
                align_to_warp: true,
                prefer_fp32_atomic: false,
                l2_benefits_from_padding: true,
            },
            GpuVendor::Qualcomm => VendorTuning {
                prefer_warps: true,
                warp_size: 32,
                align_to_warp: true,
                prefer_fp32_atomic: false,
                l2_benefits_from_padding: false,
            },
            GpuVendor::Unknown => VendorTuning::default(),
        }
    }
}

/// Vendor-specific GPU tuning parameters
#[derive(Debug, Clone)]
pub struct VendorTuning {
    /// Whether the GPU benefits from warp-aligned access patterns
    pub prefer_warps: bool,
    /// Hardware warp size
    pub warp_size: u32,
    /// Whether to align memory accesses to warp boundaries
    pub align_to_warp: bool,
    /// Whether to prefer FP32 atomics (AMD performs better with these)
    pub prefer_fp32_atomic: bool,
    /// Whether L2 cache benefits from memory padding
    pub l2_benefits_from_padding: bool,
}

impl Default for VendorTuning {
    fn default() -> Self {
        Self {
            prefer_warps: false,
            warp_size: 32,
            align_to_warp: false,
            prefer_fp32_atomic: false,
            l2_benefits_from_padding: false,
        }
    }
}

fn detect_vram_ash() -> Option<(u64, String, GpuKind)> {
    let entry = unsafe { ash::Entry::load().ok()? };

    let mut app_info = ash::vk::ApplicationInfo::default();
    app_info.api_version = ash::vk::API_VERSION_1_0;

    let extension_names: Vec<*const std::os::raw::c_char> = vec![ash::ext::debug_utils::NAME.as_ptr()];

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
            warn!(event = "skipping_software_renderer", adapter = %name, "skipping software Vulkan renderer");
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
