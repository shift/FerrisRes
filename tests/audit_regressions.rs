//! Regressions from the 2026-09-15 bug audit; these assert intended behavior.
//! Unresolved defects remain ignored; repaired regressions run by default.
//! Run unresolved reproductions with:
//! nix develop --command cargo test --test audit_regressions -- --ignored --test-threads=1
//! Remove each ignore after its corresponding defect is fixed.
use std::io::{BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};
use ferrisres::inference::host_tools::{execute_file_read, execute_shell_exec, execute_web_fetch};
use ferrisres::inference::tool_search::ToolCall;
use ferrisres::server::{ChatCompletionResponse, ChatMessage, Choice, HttpRequest, Usage};

#[test]
#[ignore = "audit: file_read slices inside UTF-8"]
fn file_read_unicode_limit_does_not_panic() {
    let path = std::env::temp_dir().join(format!("ferrisres-audit-unicode-{}", std::process::id()));
    std::fs::write(&path, "é").unwrap();
    let args = serde_json::json!({"path": path, "max_bytes": 1}).to_string();
    let result = std::panic::catch_unwind(|| execute_file_read(&ToolCall::new("file_read", args)));
    std::fs::remove_file(path).unwrap();
    assert!(result.is_ok(), "byte limit must not panic on valid UTF-8");
}

#[test]
#[ignore = "audit: shell_exec ignores timeout_secs"]
fn shell_timeout_is_enforced() {
    let start = Instant::now();
    let result = execute_shell_exec(&ToolCall::new("shell_exec", r#"{"command":"sleep 2","timeout_secs":1}"#));
    assert!(!result.success, "timed-out command must not report success");
    assert!(start.elapsed() < Duration::from_millis(1800));
}

#[test]
#[ignore = "audit: web_fetch ignores max_bytes"]
fn web_fetch_obeys_byte_limit() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let worker = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut request = [0u8; 4096];
        let _ = stream.read(&mut request).unwrap();
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabcdef").unwrap();
    });
    let args = serde_json::json!({"url": format!("http://{addr}"), "max_bytes": 1}).to_string();
    let result = execute_web_fetch(&ToolCall::new("web_fetch", args));
    worker.join().unwrap();
    assert!(result.success, "local fixture fetch failed: {}", result.output);
    assert!(result.output.len() <= 1, "received {} bytes despite limit 1", result.output.len());
}

#[test]
#[ignore = "audit: HTTP parser accepts truncated body"]
fn truncated_http_body_is_rejected() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    client.write_all(b"POST /v1/completions HTTP/1.1\r\nContent-Length: 5\r\n\r\nabc").unwrap();
    client.shutdown(Shutdown::Write).unwrap();
    let (stream, _) = listener.accept().unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    let parsed = HttpRequest::from_stream(&mut BufReader::new(stream));
    assert!(parsed.is_none(), "incomplete body must not be accepted with zero padding");
}

#[test]
#[ignore = "audit: response serializer fails to escape control characters"]
fn response_with_control_character_is_valid_json() {
    let response = ChatCompletionResponse {
        id: "audit".into(), object: "chat.completion".into(), created: 0, model: "audit".into(),
        choices: vec![Choice { index: 0, message: ChatMessage::assistant("hello\u{0001}"), finish_reason: "stop".into() }],
        usage: Usage { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    };
    serde_json::from_str::<serde_json::Value>(&response.to_json()).expect("response must be valid JSON");
}

fn tiny_block_model() -> ferrisres::model::cpu_block_attn_res::CpuBlockAttnResModel {
    use ferrisres::model::cpu_block_attn_res::{BlockConfig, CpuBlockAttnResLayer, CpuBlockAttnResModel};
    CpuBlockAttnResModel {
        layers: vec![CpuBlockAttnResLayer::new(2, 1, 1, 2, 2)],
        embed_tokens: vec![1.0, 0.0, 0.0, 1.0, 0.0, -1.0],
        lm_head: vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        final_norm: vec![1.0; 2], hidden_dim: 2, vocab_size: 3, num_layers: 1,
        final_logit_softcapping: None, ple_model_projection: None,
        ple_projection_norm: None, embed_tokens_per_layer: None,
        hidden_size_per_layer_input: 0, num_kv_shared_layers: 0,
        block_config: BlockConfig { num_blocks: 1, layers_per_block: 1,
            boundary_layers: vec![0], attn_res_proj: vec![1.0, 0.0, 0.0, 1.0],
            attn_res_norm: vec![1.0; 2] },
        lora_manager: None,
    }
}

#[test]
fn cpu_forward_prefix_logits_are_causal() {
    let model = tiny_block_model();
    let a = model.forward(&[0, 1]);
    let b = model.forward(&[0, 2]);
    assert!(a[..3].iter().zip(&b[..3]).all(|(x, y)| (x-y).abs() < 1e-5),
        "same first token has different logits: {:?} vs {:?}", &a[..3], &b[..3]);
}

#[test]
fn cpu_training_prefix_logits_are_causal() {
    let model = tiny_block_model();
    let a = model.forward_train(&[0, 1]).logits;
    let b = model.forward_train(&[0, 2]).logits;
    assert!(a[..3].iter().zip(&b[..3]).all(|(x, y)| (x-y).abs() < 1e-5),
        "same first token has different training logits: {:?} vs {:?}", &a[..3], &b[..3]);
}

#[test]
#[ignore = "audit: GGUF zero alignment divides by zero"]
fn gguf_zero_alignment_returns_error_without_panic() {
    let path = std::env::temp_dir().join(format!("ferrisres-audit-alignment-{}.gguf", std::process::id()));
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes()); // zero tensors
    bytes.extend(1u64.to_le_bytes()); // one metadata field
    let key = b"general.alignment";
    bytes.extend((key.len() as u64).to_le_bytes());
    bytes.extend(key);
    bytes.extend(4u32.to_le_bytes()); // GGUF UINT32
    bytes.extend(0u32.to_le_bytes());
    std::fs::write(&path, bytes).unwrap();
    let result = std::panic::catch_unwind(|| ferrisres::model::gguf::load_gguf(&path));
    std::fs::remove_file(path).unwrap();
    assert!(result.is_ok(), "invalid GGUF must return Err, not panic");
    assert!(result.unwrap().is_err(), "zero alignment must be rejected");
}

#[test]
fn gpu_cross_entropy_dispatch_is_valid() {
    use ferrisres::compute::{GpuBuffer, WgpuCompute};
    use ferrisres::training::CrossEntropyLoss;
    let compute = pollster::block_on(WgpuCompute::new()).expect("GPU required for this explicit probe");
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    let loss = CrossEntropyLoss::new(device.clone());
    let logits = GpuBuffer::zeros(&device, &queue, 8, Some("audit logits")).unwrap();
    let targets = GpuBuffer::zeros(&device, &queue, 4, Some("audit targets")).unwrap();
    let output = GpuBuffer::zeros(&device, &queue, 4, Some("audit loss")).unwrap();
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    loss.compute(&mut encoder, &logits, &targets, 1, 2, &output).unwrap();
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let error = pollster::block_on(scope.pop());
    assert!(error.is_none(), "cross entropy dispatch validation error: {error:?}");
}
