use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// GPU feature capabilities detected from wgpu device.
///
/// Queried at device creation time from `wgpu::Device::features()` and `wgpu::Device::limits()`,
/// stored alongside the existing `Capability` struct. Used by dispatch to select optimal
/// kernel variants (cooperative matrix vs basic matmul, subgroup vs basic FlashDecode, etc).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapabilities {
    // ── Feature flags ────────────────────────────────────────────────────
    /// F64 (double precision) floating point.
    /// Intel iGPUs support this; Apple/Mobile/RPi do not.
    pub float64: bool,

    /// F16 (half precision) floating point.
    /// Most desktop GPUs support this.
    pub float16: bool,

    /// Subgroup operations (shuffle, broadcast, reduce).
    /// Critical for FlashDecode: subgroup-based reduction avoids shared memory.
    /// Intel Gen9+, NVIDIA, AMD support; Apple/Mobile do not.
    pub subgroups: bool,

    /// Cooperative matrix (matrix multiply with implicit tiling).
    /// Intel Xe-HPG+, NVIDIA Tensor Cores, AMD CDNA support.
    pub cooperative_matrix: bool,

    /// Push constants (small fast uniform data).
    /// Nearly universal, but some mobile GPUs have limited support.
    pub immediates: bool,

    /// Shader primitives (shader_f16, shader_f64).
    /// Enables using these types directly in WGSL.
    pub shader_f16: bool,
    pub shader_f64: bool,

    /// Storage resource binding array (dynamic indexing into texture/buffer arrays).
    pub storage_resource_binding_array: bool,

    /// 16-bit norm formats for textures.
    pub norm16: bool,

    // ── Limits that affect kernel selection ──────────────────────────────
    /// Max compute workgroup invocations (e.g., 256 on most GPUs).
    pub max_workgroup_invocations: u32,

    /// Max storage buffer binding size (e.g., 128MB on Intel, 2GB on NVIDIA).
    pub max_storage_buffer_size: u64,

    /// Max compute workgroup size per dimension.
    pub max_workgroup_size_x: u32,
    pub max_workgroup_size_y: u32,
    pub max_workgroup_size_z: u32,

    /// Subgroup size (if subgroups supported). Typically 8, 16, or 32.
    /// 0 if subgroups not supported.
    pub subgroup_size: u32,

    /// Max bindings per bind group.
    pub max_bind_groups: u32,

    // ── Derived capabilities ────────────────────────────────────────────
    /// Whether this GPU can run the optimized "cooperative" matmul kernel.
    /// Requires: cooperative_matrix + subgroups + sufficient workgroup size.
    pub can_cooperative_matmul: bool,

    /// Whether this GPU can run subgroup-optimized FlashDecode.
    /// Requires: subgroups + subgroup_size >= 8.
    pub can_subgroup_flash_decode: bool,

    /// Whether F16 accumulation is available for matmuls.
    /// Requires: shader_f16 + float16.
    pub can_f16_matmul: bool,

    /// Backend type (Vulkan, Metal, DX12, GL, WebGPU).
    pub backend: String,

    /// Adapter name (e.g., "Intel(R) HD Graphics 530").
    pub adapter_name: String,
}

impl GpuCapabilities {
    /// Detect GPU capabilities from wgpu adapter and device.
    ///
    /// Call after `adapter.request_device()` with the features that were actually enabled.
    pub fn from_device(device: &wgpu::Device, adapter_info: &wgpu::AdapterInfo) -> Self {
        let features = device.features();
        let limits = device.limits();

        let float64 = features.contains(wgpu::Features::SHADER_F64);
        let float16 = features.contains(wgpu::Features::SHADER_F16);
        let subgroups = features.contains(wgpu::Features::SUBGROUP);
        let cooperative_matrix = features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
        let immediates = features.contains(wgpu::Features::IMMEDIATES);
        let shader_f16 = features.contains(wgpu::Features::SHADER_F16);
        let shader_f64 = features.contains(wgpu::Features::SHADER_F64);
        let storage_resource_binding_array = features.contains(wgpu::Features::BUFFER_BINDING_ARRAY);
        let norm16 = features.contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);

        let max_workgroup_invocations = limits.max_compute_invocations_per_workgroup;
        let max_storage_buffer_size = limits.max_storage_buffer_binding_size as u64;
        let max_workgroup_size_x = limits.max_compute_workgroup_size_x;
        let max_workgroup_size_y = limits.max_compute_workgroup_size_y;
        let max_workgroup_size_z = limits.max_compute_workgroup_size_z;
        // Subgroup size is not directly available from wgpu limits;
        // use a heuristic based on vendor/adapter
        let subgroup_size = if subgroups { 32u32 } else { 0 };
        let max_bind_groups = limits.max_bind_groups;

        // Derived capabilities
        let can_cooperative_matmul = cooperative_matrix && subgroups && max_workgroup_invocations >= 64;
        let can_subgroup_flash_decode = subgroups && subgroup_size >= 8;
        let can_f16_matmul = shader_f16 && float16;

        let caps = GpuCapabilities {
            float64,
            float16,
            subgroups,
            cooperative_matrix,
            immediates,
            shader_f16,
            shader_f64,
            storage_resource_binding_array,
            norm16,
            max_workgroup_invocations,
            max_storage_buffer_size,
            max_workgroup_size_x,
            max_workgroup_size_y,
            max_workgroup_size_z,
            subgroup_size,
            max_bind_groups,
            can_cooperative_matmul,
            can_subgroup_flash_decode,
            can_f16_matmul,
            backend: format!("{:?}", adapter_info.backend),
            adapter_name: adapter_info.name.clone(),
        };

        info!(
            event = "gpu_capabilities",
            f64 = float64,
            f16 = float16,
            subgroups = subgroups,
            subgroup_size = subgroup_size,
            coop_matrix = cooperative_matrix,
            immediates = immediates,
            max_invocations = max_workgroup_invocations,
            max_storage_mb = max_storage_buffer_size / (1024 * 1024),
            can_cooperative_matmul = can_cooperative_matmul,
            can_subgroup_flash_decode = can_subgroup_flash_decode,
            can_f16_matmul = can_f16_matmul,
            backend = %caps.backend,
            adapter = %caps.adapter_name,
            "GPU capabilities detected"
        );

        caps
    }

    /// Create a CPU-only capabilities struct (no GPU features).
    pub fn cpu_only() -> Self {
        GpuCapabilities {
            float64: true, // CPU always has F64
            float16: false,
            subgroups: false,
            cooperative_matrix: false,
            immediates: false,
            shader_f16: false,
            shader_f64: true,
            storage_resource_binding_array: false,
            norm16: false,
            max_workgroup_invocations: 0,
            max_storage_buffer_size: 0,
            max_workgroup_size_x: 0,
            max_workgroup_size_y: 0,
            max_workgroup_size_z: 0,
            subgroup_size: 0,
            max_bind_groups: 0,
            can_cooperative_matmul: false,
            can_subgroup_flash_decode: false,
            can_f16_matmul: false,
            backend: "CPU".to_string(),
            adapter_name: "CPU".to_string(),
        }
    }

    /// Check if any GPU features are available.
    pub fn has_gpu(&self) -> bool {
        self.max_workgroup_invocations > 0
    }

    /// Recommend a matmul kernel type based on capabilities.
    pub fn recommend_matmul_kernel(&self) -> MatmulKernelType {
        if self.can_cooperative_matmul {
            MatmulKernelType::CooperativeMatrix
        } else if self.can_f16_matmul {
            MatmulKernelType::F16Tiled
        } else {
            MatmulKernelType::Basic
        }
    }

    /// Recommend a FlashDecode kernel type based on capabilities.
    pub fn recommend_flash_decode_kernel(&self) -> FlashDecodeKernelType {
        if self.can_subgroup_flash_decode {
            FlashDecodeKernelType::SubgroupOptimized
        } else {
            FlashDecodeKernelType::Basic
        }
    }
}

/// Matmul kernel variant selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatmulKernelType {
    /// Basic tiled matmul (works everywhere).
    Basic,
    /// F16 accumulation tiled matmul.
    F16Tiled,
    /// Cooperative matrix multiply (Intel/AMD/NVIDIA tensor cores).
    CooperativeMatrix,
}

/// FlashDecode kernel variant selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlashDecodeKernelType {
    /// Basic decode (sequential per-head).
    Basic,
    /// Subgroup-optimized decode (parallel reduction within subgroup).
    SubgroupOptimized,
}

// ── Existing Capability struct ───────────────────────────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_only_capabilities() {
        let caps = GpuCapabilities::cpu_only();
        assert!(caps.float64);
        assert!(caps.shader_f64);
        assert!(!caps.has_gpu());
        assert!(!caps.subgroups);
        assert!(!caps.cooperative_matrix);
        assert!(!caps.can_cooperative_matmul);
        assert!(!caps.can_subgroup_flash_decode);
        assert!(!caps.can_f16_matmul);
        assert_eq!(caps.recommend_matmul_kernel(), MatmulKernelType::Basic);
        assert_eq!(caps.recommend_flash_decode_kernel(), FlashDecodeKernelType::Basic);
    }

    #[test]
    fn test_vendor_from_name() {
        assert_eq!(GpuVendor::from_adapter_name("NVIDIA GeForce RTX 4090"), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_adapter_name("AMD Radeon RX 7900 XTX"), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_adapter_name("Intel(R) HD Graphics 530"), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_adapter_name("Apple M2"), GpuVendor::Apple);
        assert_eq!(GpuVendor::from_adapter_name("Qualcomm Adreno 740"), GpuVendor::Qualcomm);
        assert_eq!(GpuVendor::from_adapter_name("Mesa Intel"), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_adapter_name("Unknown GPU"), GpuVendor::Unknown);
    }

    #[test]
    fn test_kernel_selection_with_subgroups() {
        let caps = GpuCapabilities {
            float64: false,
            float16: true,
            subgroups: true,
            cooperative_matrix: false,
            immediates: true,
            shader_f16: true,
            shader_f64: false,
            storage_resource_binding_array: false,
            norm16: false,
            max_workgroup_invocations: 256,
            max_storage_buffer_size: 128 * 1024 * 1024,
            max_workgroup_size_x: 256,
            max_workgroup_size_y: 256,
            max_workgroup_size_z: 64,
            subgroup_size: 32,
            max_bind_groups: 4,
            can_cooperative_matmul: false,
            can_subgroup_flash_decode: true,
            can_f16_matmul: true,
            backend: "Vulkan".to_string(),
            adapter_name: "Test GPU".to_string(),
        };

        assert_eq!(caps.recommend_matmul_kernel(), MatmulKernelType::F16Tiled);
        assert_eq!(caps.recommend_flash_decode_kernel(), FlashDecodeKernelType::SubgroupOptimized);
    }

    #[test]
    fn test_kernel_selection_with_cooperative_matrix() {
        let caps = GpuCapabilities {
            float64: true,
            float16: true,
            subgroups: true,
            cooperative_matrix: true,
            immediates: true,
            shader_f16: true,
            shader_f64: true,
            storage_resource_binding_array: true,
            norm16: true,
            max_workgroup_invocations: 256,
            max_storage_buffer_size: 2 * 1024 * 1024 * 1024,
            max_workgroup_size_x: 1024,
            max_workgroup_size_y: 1024,
            max_workgroup_size_z: 64,
            subgroup_size: 32,
            max_bind_groups: 8,
            can_cooperative_matmul: true,
            can_subgroup_flash_decode: true,
            can_f16_matmul: true,
            backend: "Vulkan".to_string(),
            adapter_name: "Intel Arc A770".to_string(),
        };

        assert_eq!(caps.recommend_matmul_kernel(), MatmulKernelType::CooperativeMatrix);
        assert_eq!(caps.recommend_flash_decode_kernel(), FlashDecodeKernelType::SubgroupOptimized);
    }

    #[test]
    fn test_capability_detect() {
        // Just test that detect() doesn't panic — it may or may not find a GPU
        let _cap = Capability::detect();
    }

    #[test]
    fn test_effective_vram_discrete() {
        let cap = Capability {
            vram_mb: 8192,
            shared_ram_mb: 16384,
            gpu_kind: GpuKind::Discrete,
            vendor: GpuVendor::Nvidia,
            max_compute_workgroup_size: 1024,
            max_compute_invocations_per_workgroup: 1024,
            max_storage_buffer_range: 128 << 20,
            max_storage_buffers_per_shader_stage: 8,
            max_bind_groups: 4,
            backend: "Vulkan".to_string(),
            adapter_name: "RTX 4060".to_string(),
        };
        assert_eq!(cap.effective_vram_mb(), 8192);
    }

    #[test]
    fn test_effective_vram_integrated() {
        let cap = Capability {
            vram_mb: 0,
            shared_ram_mb: 8192,
            gpu_kind: GpuKind::Integrated,
            vendor: GpuVendor::Intel,
            max_compute_workgroup_size: 256,
            max_compute_invocations_per_workgroup: 256,
            max_storage_buffer_range: 128 << 20,
            max_storage_buffers_per_shader_stage: 8,
            max_bind_groups: 4,
            backend: "Vulkan".to_string(),
            adapter_name: "Intel HD 530".to_string(),
        };
        assert_eq!(cap.effective_vram_mb(), 8192); // Falls back to shared RAM
    }
}
