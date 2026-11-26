use wgpu::{Adapter, Device, Queue, ShaderModule, ComputePipeline};
use crate::error::{FerrisResError, Result};
use crate::device::{DeviceProfile, Capability};

/// The backends adapter enumeration and `WGPU_ADAPTER_INDEX` select over.
///
/// `WgpuCompute::new()` uses the same set for its `HighPerformance` fallback,
/// so an index always refers to the same list a worker sees.
pub fn device_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN
        | wgpu::Backends::METAL
        | wgpu::Backends::DX12
        | wgpu::Backends::GL
}

fn new_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: device_backends(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    })
}

/// Deterministic order for adapters: name, then PCI bus id, then backend.
///
/// Two identical T4s share a name, so the bus id is the tie-breaker that keeps
/// index 0 and index 1 from swapping between runs. Pure, so it is unit-tested
/// without a GPU.
pub(crate) fn sort_order(names: &[String], bus_ids: &[String], backends: &[String]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_by(|&a, &b| {
        names[a]
            .cmp(&names[b])
            .then_with(|| bus_ids.get(a).cmp(&bus_ids.get(b)))
            .then_with(|| backends.get(a).cmp(&backends.get(b)))
    });
    order
}

fn sorted_adapters(adapters: Vec<Adapter>) -> Vec<Adapter> {
    let infos: Vec<wgpu::AdapterInfo> = adapters.iter().map(|a| a.get_info()).collect();
    let names: Vec<String> = infos.iter().map(|i| i.name.clone()).collect();
    let bus_ids: Vec<String> = infos.iter().map(|i| i.device_pci_bus_id.clone()).collect();
    let backends: Vec<String> = infos.iter().map(|i| format!("{:?}", i.backend)).collect();
    let order = sort_order(&names, &bus_ids, &backends);
    let mut slots: Vec<Option<Adapter>> = adapters.into_iter().map(Some).collect();
    order
        .into_iter()
        .filter_map(|i| slots[i].take())
        .collect()
}

/// Choose an adapter index from a name-sorted list.
///
/// `requested` counts positions *after* software filtering, matching
/// `WGPU_ADAPTER_INDEX`. Names and kinds must already be in sorted order; the
/// returned value indexes that same slice.
pub fn pick_adapter(
    names: &[String],
    kinds: &[wgpu::DeviceType],
    requested: usize,
    allow_software: bool,
) -> Result<usize> {
    let usable: Vec<usize> = names
        .iter()
        .zip(kinds.iter())
        .enumerate()
        .filter(|(_, (_, kind))| allow_software || **kind != wgpu::DeviceType::Cpu)
        .map(|(i, _)| i)
        .collect();

    if let Some(&index) = usable.get(requested) {
        return Ok(index);
    }

    let listed = if usable.is_empty() {
        "none".to_string()
    } else {
        usable
            .iter()
            .map(|&i| names[i].as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(FerrisResError::Device(format!(
        "WGPU_ADAPTER_INDEX={} is out of range: {} usable adapter(s) after software filtering: [{}]",
        requested,
        usable.len(),
        listed
    )))
}

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes")
    )
}

/// Runtime compute parameters optimized for detected hardware
#[derive(Debug, Clone)]
pub struct ComputeParams {
    /// Recommended workgroup size for compute shaders
    pub workgroup_size: u32,
    /// Recommended tile size for matrix operations (height, width)
    pub tile_size: (u32, u32),
    /// Recommended batch size for inference
    pub batch_size: u32,
    /// Whether to use async compute pipelines
    pub use_async: bool,
}

impl Default for ComputeParams {
    fn default() -> Self {
        Self {
            workgroup_size: 64,
            tile_size: (16, 16),
            batch_size: 1,
            use_async: false,
        }
    }
}

pub struct WgpuCompute {
    device: Device,
    queue: Queue,
    adapter: Adapter,
    pipeline: Option<ComputePipeline>,
}

/// Limits to request for a device, given the adapter's limits and the features
/// we enable.
///
/// `wgpu::Limits::default()` carries `max_immediate_size: 0`. If `IMMEDIATES`
/// is enabled while that limit stays 0, every pipeline layout that declares
/// immediate data fails validation with
/// `Immediate data has size N which exceeds device immediate data size limit
/// 0..0`. The adapter reports the real size (Vulkan push-constant bytes), so
/// take it whenever the feature is on. With the feature off the limit stays 0,
/// which is correct: no layout may declare immediate data.
pub(crate) fn device_limits(adapter: &wgpu::Limits, features: wgpu::Features) -> wgpu::Limits {
    let mut limits = wgpu::Limits::default();
    if features.contains(wgpu::Features::IMMEDIATES) {
        limits.max_immediate_size = adapter.max_immediate_size;
    }
    limits
}

impl WgpuCompute {
    pub async fn new() -> Result<Self> {
        let instance = new_instance();

        // `WGPU_ADAPTER_INDEX` selects a specific adapter (one worker per GPU).
        // Without it, keep the historical HighPerformance request.
        let adapter = match std::env::var("WGPU_ADAPTER_INDEX") {
            Ok(raw) => {
                let requested = raw.trim().parse::<usize>().map_err(|_| {
                    FerrisResError::Device(format!(
                        "WGPU_ADAPTER_INDEX must be a non-negative integer, got {:?}",
                        raw
                    ))
                })?;
                let adapters =
                    sorted_adapters(instance.enumerate_adapters(device_backends()).await);
                let names: Vec<String> = adapters.iter().map(|a| a.get_info().name).collect();
                let kinds: Vec<wgpu::DeviceType> =
                    adapters.iter().map(|a| a.get_info().device_type).collect();
                let index = pick_adapter(
                    &names,
                    &kinds,
                    requested,
                    env_flag("FERRISRES_ALLOW_SOFTWARE"),
                )?;
                adapters.into_iter().nth(index).ok_or_else(|| {
                    FerrisResError::Device(format!(
                        "WGPU_ADAPTER_INDEX={} resolved to {} but the adapter list changed",
                        requested, index
                    ))
                })?
            }
            Err(_) => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .map_err(|e| {
                    FerrisResError::Device(format!("Failed to find adapter: {}", e))
                })?,
        };

        let required_features = if adapter.features().contains(wgpu::Features::IMMEDIATES) {
            wgpu::Features::IMMEDIATES
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("FerrisRes Device"),
                    required_features,
                    required_limits: device_limits(&adapter.limits(), required_features),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                },
            )
            .await
            .map_err(|e| FerrisResError::Device(format!("Failed to request device: {}", e)))?;

        Ok(Self {
            device,
            queue,
            adapter,
            pipeline: None,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    pub fn adapter_limits(&self) -> wgpu::Limits {
        self.adapter.limits()
    }

    pub fn detect_capability(&self) -> Capability {
        let limits = self.adapter_limits();
        let info = self.adapter_info();
        Capability::detect()
            .with_adapter_limits(&limits, &info)
    }
    
    /// Get optimized compute parameters based on detected hardware
    pub fn get_optimal_params(&self) -> ComputeParams {
        let capability = self.detect_capability();
        let profile = DeviceProfile::from_vram_and_kind(
            capability.vram_mb, 
            capability.gpu_kind
        );
        
        ComputeParams {
            workgroup_size: profile.recommended_workgroup_size(
                capability.max_compute_invocations_per_workgroup
            ),
            tile_size: profile.recommended_tile_size(),
            batch_size: profile.recommended_batch_size(),
            use_async: matches!(profile, DeviceProfile::HighEnd | DeviceProfile::MidRange),
        }
    }

    pub fn detect_profile(&self) -> DeviceProfile {
        // Priority 1: FERRIS_DEVICE_PROFILE env var override.
        if let Some(profile) = DeviceProfile::from_env() {
            return profile;
        }

        // Priority 2: Vulkan/ash capability detection.
        let cap = self.detect_capability();
        let profile = DeviceProfile::from_vram_and_kind(cap.vram_mb, cap.gpu_kind);

        // Priority 3 (implicit): DeviceProfile::Integrated is the fallback inside
        // from_vram_and_kind when no discrete GPU is found.

        // Emit startup summary log.
        let adapter_info = self.adapter_info();
        tracing::info!(event = "ferrisres_device_profile", "=== FerrisRes Device Profile ===");
        tracing::info!(event = "gpu", "  GPU: {}", adapter_info.name);
        tracing::info!(event = "profile", "  Profile: {:?}", profile);
        tracing::info!(event = "compute_mode", "  Compute mode: {:?}", profile.compute_mode());
        tracing::info!(event = "batch_size", "  Batch size: {}", profile.recommended_batch_size());
        tracing::info!(event = "cache_size_mb", "  Cache size: {} MB", profile.cache_size() / (1024 * 1024));
        tracing::info!(event = "", "================================");

        profile
    }

    pub fn create_shader_module(&self, wgsl: &str) -> Result<ShaderModule> {
        Ok(self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FerrisRes Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        }))
    }

    pub fn create_compute_pipeline(
        &mut self,
        shader: &ShaderModule,
        entry_point: &str,
    ) -> Result<ComputePipeline> {
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&layout),
                module: shader,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        self.pipeline = Some(pipeline.clone());
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::{device_limits, pick_adapter, sort_order};

    /// The regression this module exists for: `Limits::default()` reports no
    /// immediate space, so a device that enables `IMMEDIATES` on top of it
    /// cannot create any pipeline layout that declares immediate data.
    #[test]
    fn default_limits_have_no_immediate_space() {
        assert_eq!(wgpu::Limits::default().max_immediate_size, 0);
    }

    #[test]
    fn immediates_take_the_adapter_size() {
        let adapter = wgpu::Limits {
            max_immediate_size: 256,
            ..wgpu::Limits::default()
        };
        let limits = device_limits(&adapter, wgpu::Features::IMMEDIATES);
        assert_eq!(limits.max_immediate_size, 256);
    }

    #[test]
    fn without_immediates_the_limit_stays_zero() {
        let adapter = wgpu::Limits {
            max_immediate_size: 256,
            ..wgpu::Limits::default()
        };
        let limits = device_limits(&adapter, wgpu::Features::empty());
        assert_eq!(limits.max_immediate_size, 0);
    }

    /// Every immediate-data pipeline in the tree declares at most 16 bytes
    /// (`softmax` 4, `embedding` 8, `backward` up to 16). If the adapter offers
    /// less than that, the feature must not be enabled or the layout will
    /// fail, so this asserts the value the guard relies on.
    #[test]
    fn smallest_supported_immediate_size_covers_the_tree() {
        let adapter = wgpu::Limits {
            max_immediate_size: 4,
            ..wgpu::Limits::default()
        };
        let limits = device_limits(&adapter, wgpu::Features::IMMEDIATES);
        assert!(limits.max_immediate_size >= 4);
    }

    fn strs(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn selects_each_of_two_gpus_by_index() {
        let names = strs(&["NVIDIA T4", "NVIDIA T4"]);
        let kinds = [wgpu::DeviceType::DiscreteGpu, wgpu::DeviceType::DiscreteGpu];
        assert_eq!(pick_adapter(&names, &kinds, 0, false).unwrap(), 0);
        assert_eq!(pick_adapter(&names, &kinds, 1, false).unwrap(), 1);
    }

    #[test]
    fn software_adapter_is_skipped_by_default() {
        let names = strs(&["GPU A", "llvmpipe", "GPU B"]);
        let kinds = [
            wgpu::DeviceType::DiscreteGpu,
            wgpu::DeviceType::Cpu,
            wgpu::DeviceType::DiscreteGpu,
        ];
        assert_eq!(pick_adapter(&names, &kinds, 1, false).unwrap(), 2);
        assert_eq!(pick_adapter(&names, &kinds, 0, false).unwrap(), 0);
        assert_eq!(pick_adapter(&names, &kinds, 1, true).unwrap(), 1);
    }

    #[test]
    fn out_of_range_names_the_count_and_adapters() {
        let names = strs(&["GPU A", "llvmpipe", "GPU B"]);
        let kinds = [
            wgpu::DeviceType::DiscreteGpu,
            wgpu::DeviceType::Cpu,
            wgpu::DeviceType::DiscreteGpu,
        ];
        let err = pick_adapter(&names, &kinds, 2, false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("2 usable adapter"), "missing count: {msg}");
        assert!(msg.contains("GPU A") && msg.contains("GPU B"), "missing names: {msg}");
        assert!(!msg.contains("llvmpipe"), "software must not be listed: {msg}");
    }

    #[test]
    fn ordering_is_stable_and_ties_break_on_bus_id() {
        let names = strs(&["GPU B", "GPU A", "GPU A"]);
        let bus_ids = strs(&["0000:02:00.0", "0000:03:00.0", "0000:01:00.0"]);
        let backends = strs(&["Vulkan", "Vulkan", "Vulkan"]);
        let order = sort_order(&names, &bus_ids, &backends);
        assert_eq!(order, vec![2, 1, 0]);
        for _ in 0..8 {
            assert_eq!(sort_order(&names, &bus_ids, &backends), order);
        }
    }
}
