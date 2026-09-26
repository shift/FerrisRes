use std::sync::Arc;
use ferrisres::compute::{GpuBuffer,WgpuCompute};
use ferrisres::inference::unified_generator::UnifiedTokenGenerator;
use ferrisres::model::dispatcher::AnyModel;
use ferrisres::model::io_weights::{HeadWeights,ModelIo};
use ferrisres::model::standard_transformer::StandardTransformerConfig;

const EMBED: [f32;6]=[1.0,2.0,3.0,-1.0,-2.0,0.5];
const HEAD: [f32;6]=[1.0,-2.0,0.5,3.0,-4.0,1.0];
fn read(device:&wgpu::Device,queue:&wgpu::Queue,source:&GpuBuffer)->Vec<f32> {
    let staging=device.create_buffer(&wgpu::BufferDescriptor {label:None,size:source.size() as u64,
        usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
    let mut encoder=device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source.buffer(),0,&staging,0,source.size() as u64);
    queue.submit([encoder.finish()]);let(tx,rx)=std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read,move|r|tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();rx.recv().unwrap().unwrap();
    bytemuck::cast_slice::<u8,f32>(&staging.slice(..).get_mapped_range()).to_vec()
}
fn check_logits(tied:bool) {
    let compute=pollster::block_on(WgpuCompute::new()).unwrap();
    let device=Arc::new(compute.device().clone());let queue=Arc::new(compute.queue().clone());
    let head=if tied {HeadWeights::TiedToEmbedding} else {HeadWeights::Untied(&HEAD)};
    let io=ModelIo::from_weights(device.clone(),queue.clone(),3,2,&EMBED,head).unwrap();
    let ids=GpuBuffer::new(&device,8,None).unwrap();let hidden=GpuBuffer::new(&device,16,None).unwrap();
    let logits=GpuBuffer::new(&device,24,None).unwrap();
    queue.write_buffer(ids.buffer(),0,bytemuck::cast_slice(&[2u32,0]));
    let mut encoder=device.create_command_encoder(&Default::default());
    io.embedding().forward(&mut encoder,&ids,&hidden,2).unwrap();
    io.lm_head().forward(&mut encoder,&hidden,&logits,2).unwrap();
    queue.submit([encoder.finish()]);
    let head=if tied {&EMBED} else {&HEAD};
    let expected:Vec<f32>=[2usize,0].iter().flat_map(|&t|(0..3).map(move|v|
        EMBED[t*2]*head[v*2]+EMBED[t*2+1]*head[v*2+1])).collect();
    for(a,b) in read(&device,&queue,&logits).iter().zip(expected) {assert!((a-b).abs()<1e-5,"{a} vs {b}");}
}
#[test] fn untied_checkpoint_layout_reaches_gpu_logits(){check_logits(false);}
#[test] fn explicitly_tied_values_reach_gpu_logits(){check_logits(true);}

#[test]
fn malformed_weights_never_produce_a_bundle() {
    let compute=pollster::block_on(WgpuCompute::new()).unwrap();
    let d=Arc::new(compute.device().clone());let q=Arc::new(compute.queue().clone());
    for(v,h,e,head) in [(0,2,EMBED.to_vec(),HEAD.to_vec()),(3,0,EMBED.to_vec(),HEAD.to_vec()),
        (usize::MAX,2,EMBED.to_vec(),HEAD.to_vec()),(3,2,vec![1.0;5],HEAD.to_vec()),
        (3,2,EMBED.to_vec(),vec![1.0;5]),(3,2,vec![f32::NAN;6],HEAD.to_vec()),
        (3,2,EMBED.to_vec(),vec![f32::INFINITY;6])] {
        assert!(ModelIo::from_weights(d.clone(),q.clone(),v,h,&e,HeadWeights::Untied(&head)).is_err());
    }
}

#[test]
fn generator_accepts_only_matching_model_dimensions() {
    let compute=pollster::block_on(WgpuCompute::new()).unwrap();
    let d=Arc::new(compute.device().clone());let q=Arc::new(compute.queue().clone());
    for(hidden,vocab,valid) in [(2,3,true),(4,3,false),(2,4,false)] {
        let io=ModelIo::from_weights(d.clone(),q.clone(),3,2,&EMBED,HeadWeights::Untied(&HEAD)).unwrap();
        let mut config=StandardTransformerConfig::new(hidden,1,0);config.vocab_size=vocab;
        let model=AnyModel::new_standard(d.clone(),q.clone(),config).unwrap();
        assert_eq!(UnifiedTokenGenerator::from_model_io(model,io,8).is_ok(),valid);
    }
}

#[test]
fn generator_rejects_different_gpu_context() {
    let first=pollster::block_on(WgpuCompute::new()).unwrap();
    let second=pollster::block_on(WgpuCompute::new()).unwrap();
    let io=ModelIo::from_weights(Arc::new(first.device().clone()),Arc::new(first.queue().clone()),3,2,&EMBED,HeadWeights::TiedToEmbedding).unwrap();
    let mut config=StandardTransformerConfig::new(2,1,0);config.vocab_size=3;
    let model=AnyModel::new_standard(Arc::new(second.device().clone()),Arc::new(second.queue().clone()),config).unwrap();
    assert!(UnifiedTokenGenerator::from_model_io(model,io,8).is_err());
}
