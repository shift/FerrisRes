use ferrisres::{WgpuCompute, DeviceProfile};
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    info!("Initializing FerrisRes...");
    
    let compute = WgpuCompute::new().await?;
    
    info!("GPU Compute initialized successfully");
    
    let adapter_info = compute.device().get_info();
    info!("Adapter: {} ({:?})", adapter_info.name, adapter_info.backend);
    
    let profile = DeviceProfile::from_vram_mb(0);
    info!("Device profile: {:?}", profile);
    info!("Compute mode: {:?}", profile.compute_mode());
    info!("Recommended batch size: {}", profile.recommended_batch_size());
    
    info!("FerrisRes ready!");
    
    Ok(())
}
