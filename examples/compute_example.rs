use ferrisres::WgpuCompute;

const ADD_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read> a: array<f32>;

@group(0) @binding(1)
var<storage, read> b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    output[index] = a[index] + b[index];
}
"#;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("Initializing FerrisRes compute example...");
    
    let mut compute = WgpuCompute::new().await?;
    println!("GPU initialized");
    
    let shader = compute.create_shader_module(ADD_WGSL)?;
    println!("Shader compiled");
    
    let pipeline = compute.create_compute_pipeline(&shader, "main")?;
    println!("Pipeline created: {:?}", pipeline);
    
    println!("\nCompute example complete!");
    println!("This demonstrates:");
    println!("  1. WGPU Vulkan backend initialization");
    println!("  2. WGSL shader compilation");
    println!("  3. Compute pipeline creation");
    
    Ok(())
}
