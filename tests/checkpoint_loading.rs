//! AnyModel must not pretend that metadata-only construction loads weights.
use std::path::PathBuf;
use std::sync::Arc;
use ferrisres::compute::WgpuCompute;
use ferrisres::error::FerrisResError;
use ferrisres::model::dispatcher::{AnyModel, ArchitectureHint};
use ferrisres::model::safetensors::{load_safetensors, write_safetensors, SafeDtype, TensorToWrite};

struct Fixture(PathBuf);
impl Drop for Fixture { fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); } }
fn fixture(extension: &str) -> Fixture {
    Fixture(std::env::temp_dir().join(format!("ferrisres-loader-{}.{extension}",std::process::id())))
}
fn rejects_unimplemented_import(result: ferrisres::error::Result<AnyModel>) {
    match result {
        Err(FerrisResError::Unsupported(message)) => assert!(message.contains("checkpoint"), "{message}"),
        Err(error) => panic!("expected an explicit unsupported-checkpoint error, got {error}"),
        Ok(_) => panic!("checkpoint tensors were silently discarded while reporting load success"),
    }
}

#[test]
fn safetensors_import_cannot_report_success_without_loading_weights() {
    let file=fixture("safetensors");
    write_safetensors(&file.0,&[
        TensorToWrite { name:"model.embed_tokens.weight".into(),shape:vec![4,8],dtype:SafeDtype::F32,data_f32:vec![0.25;32] },
        TensorToWrite { name:"model.layers.0.self_attn.q_proj.weight".into(),shape:vec![8,8],dtype:SafeDtype::F32,data_f32:vec![0.75;64] },
    ]).unwrap();
    assert_eq!(load_safetensors(&file.0).unwrap().get("model.layers.0.self_attn.q_proj.weight").unwrap().data[0],0.75);
    let compute=pollster::block_on(WgpuCompute::new()).unwrap();
    let device=Arc::new(compute.device().clone()); let queue=Arc::new(compute.queue().clone());
    rejects_unimplemented_import(AnyModel::from_safetensors(&file.0,device.clone(),queue.clone(),&ArchitectureHint::Auto));
    rejects_unimplemented_import(AnyModel::from_path(&file.0,device.clone(),queue.clone(),&ArchitectureHint::Standard));
    rejects_unimplemented_import(AnyModel::from_path(&file.0,device,queue,&ArchitectureHint::BlockAttnRes));
}

#[test]
fn gguf_import_cannot_report_success_without_loading_weights() {
    let file=fixture("gguf");
    let mut data=b"GGUF".to_vec();
    data.extend(3u32.to_le_bytes()); data.extend(1u64.to_le_bytes()); data.extend(2u64.to_le_bytes());
    for (key,value) in [("llama.embedding_length",8u32),("llama.attention.head_count",2u32)] {
        data.extend((key.len() as u64).to_le_bytes()); data.extend(key.as_bytes());
        data.extend(4u32.to_le_bytes()); data.extend(value.to_le_bytes());
    }
    let name="token_embd.weight";
    data.extend((name.len() as u64).to_le_bytes()); data.extend(name.as_bytes());
    data.extend(2u32.to_le_bytes()); data.extend(8u64.to_le_bytes()); data.extend(4u64.to_le_bytes());
    data.extend(0u32.to_le_bytes()); data.extend(0u64.to_le_bytes()); // F32, relative offset zero
    data.resize(data.len().div_ceil(32)*32,0);
    for _ in 0..32 { data.extend(0.25f32.to_le_bytes()); }
    std::fs::write(&file.0,data).unwrap();
    let parsed=ferrisres::model::gguf::load_gguf(&file.0).unwrap();
    assert!(parsed.load_tensor(name).is_ok(),"fixture must contain readable tensor data");
    let compute=pollster::block_on(WgpuCompute::new()).unwrap();
    let device=Arc::new(compute.device().clone()); let queue=Arc::new(compute.queue().clone());
    rejects_unimplemented_import(AnyModel::from_gguf(&file.0,device.clone(),queue.clone(),&ArchitectureHint::Auto));
    rejects_unimplemented_import(AnyModel::from_path(&file.0,device,queue,&ArchitectureHint::Standard));
}
