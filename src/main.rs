use ferrisres::WgpuCompute;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    info!("Initializing FerrisRes...");

    let compute = WgpuCompute::new().await?;
    info!("GPU compute initialized successfully");

    let capability = compute.detect_capability();
    let profile = ferrisres::DeviceProfile::from_vram_and_kind(
        capability.vram_mb,
        capability.gpu_kind,
    );

    info!("Adapter: {} ({})", capability.adapter_name, capability.backend);
    info!("GPU kind: {:?}", capability.gpu_kind);
    info!("Dedicated VRAM: {} MB", capability.vram_mb);
    info!("System RAM: {} MB", capability.shared_ram_mb);
    info!("Effective VRAM: {} MB", capability.effective_vram_mb());
    info!("Max workgroup size: {}", capability.max_compute_workgroup_size);
    info!("Max invocations/workgroup: {}", capability.max_compute_invocations_per_workgroup);
    info!("Max storage buffer: {} MB", capability.max_storage_buffer_range / (1024 * 1024));
    info!("Max storage buffers/stage: {}", capability.max_storage_buffers_per_shader_stage);
    info!("Max bind groups: {}", capability.max_bind_groups);
    info!("Device profile: {:?}", profile);
    info!("Compute mode: {:?}", profile.compute_mode());
    info!("Recommended batch size: {}", profile.recommended_batch_size());
    info!("Cache size: {} blocks", profile.cache_size());

    info!("FerrisRes ready!");

    Ok(())
}
