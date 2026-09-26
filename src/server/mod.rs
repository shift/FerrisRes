//! OpenAI-compatible API server for FerrisRes.
//!
//! Implements HTTP endpoints compatible with the OpenAI API:
//! - POST /v1/chat/completions
//! - POST /v1/completions
//! - GET /v1/models
//! - GET /health
//!
//! Uses a simple HTTP server (no external deps — pure std net).
//! For production use, this would be replaced with axum/actix.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use crate::security::armor::{ArmorLayer, SecurityVerdict};

// ---------------------------------------------------------------------------
// Request/Response types
// ---------------------------------------------------------------------------

/// Chat completion request (OpenAI-compatible).
#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    /// Model name.
    pub model: String,
    /// Chat messages.
    pub messages: Vec<ChatMessage>,
    /// Max tokens to generate.
    pub max_tokens: usize,
    /// Temperature.
    pub temperature: f32,
    /// Top-p.
    pub top_p: f32,
    /// Whether to stream.
    pub stream: bool,
}

/// A chat message.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: &str) -> Self {
        Self { role: "system".to_string(), content: content.to_string() }
    }
    pub fn user(content: &str) -> Self {
        Self { role: "user".to_string(), content: content.to_string() }
    }
    pub fn assistant(content: &str) -> Self {
        Self { role: "assistant".to_string(), content: content.to_string() }
    }
}

/// Chat completion response.
#[derive(Debug, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// A completion choice.
#[derive(Debug, Clone)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Token usage.
#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Text completion request.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stream: bool,
}

/// Model info.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

impl ChatCompletionResponse {
    pub fn to_json(&self) -> String {
        let choices_json: Vec<String> = self.choices.iter().map(|c| {
            format!(
                r#"{{"index":{},"message":{{"role":"{}","content":"{}"}},"finish_reason":"{}"}}"#,
                c.index, c.message.role, escape_json(&c.message.content), c.finish_reason
            )
        }).collect();

        format!(
            r#"{{"id":"{}","object":"{}","created":{},"model":"{}","choices":[{}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
            self.id, self.object, self.created, self.model,
            choices_json.join(","),
            self.usage.prompt_tokens, self.usage.completion_tokens, self.usage.total_tokens
        )
    }
}

impl ModelInfo {
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"id":"{}","object":"{}","owned_by":"{}"}}"#,
            self.id, self.object, self.owned_by
        )
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
     .replace('"', "\\\"")
     .replace('\n', "\\n")
     .replace('\r', "\\r")
     .replace('\t', "\\t")
}

// ---------------------------------------------------------------------------
// SSE streaming chunk
// ---------------------------------------------------------------------------

/// SSE chunk for streaming responses.
pub fn sse_chunk(data: &str) -> String {
    format!("data: {}\n\n", data)
}

/// SSE done signal.
pub fn sse_done() -> String {
    "data: [DONE]\n\n".to_string()
}

// ---------------------------------------------------------------------------
// Simple HTTP parsing
// ---------------------------------------------------------------------------

/// Parsed HTTP request.
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: String,
    pub headers: HashMap<String, String>,
}

impl HttpRequest {
    /// Parse from a raw TCP stream.
    pub fn from_stream(stream: &mut BufReader<TcpStream>) -> Option<Self> {
        let mut first_line = String::new();
        stream.read_line(&mut first_line).ok()?;
        let parts: Vec<&str> = first_line.trim().split(' ').collect();
        if parts.len() < 2 { return None; }

        let method = parts[0].to_string();
        let path = parts[1].to_string();

        let mut headers = HashMap::new();
        let mut header_line = String::new();
        loop {
            header_line.clear();
            stream.read_line(&mut header_line).ok()?;
            if header_line.trim().is_empty() { break; }
            if let Some((key, value)) = header_line.trim().split_once(':') {
                headers.insert(key.trim().to_lowercase(), value.trim().to_string());
            }
        }

        let content_length: usize = headers.get("content-length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let mut body_bytes = vec![0u8; content_length];
        if content_length > 0 {
            stream.fill_buf().ok()?;
            let _ = stream.read_exact(&mut body_bytes).ok();
        }
        let body = String::from_utf8_lossy(&body_bytes).into_owned();

        Some(Self { method, path, body, headers })
    }
}

// ---------------------------------------------------------------------------
// API Server
// ---------------------------------------------------------------------------

/// The API server configuration.
pub struct ApiServerConfig {
    pub host: String,
    pub port: u16,
    pub model_name: String,
}

impl Default for ApiServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            model_name: "ferrisres".to_string(),
        }
    }
}

/// The API server.
pub struct ApiServer {
    config: ApiServerConfig,
    handler: Arc<Mutex<Box<dyn ApiHandler + Send>>>,
    armor: Option<Mutex<ArmorLayer>>,
}

/// Trait for handling API requests.
pub trait ApiHandler {
    /// Handle a chat completion request.
    fn chat_completion(&mut self, req: &ChatCompletionRequest) -> ChatCompletionResponse;
    /// Handle a text completion request.
    fn completion(&mut self, req: &CompletionRequest) -> ChatCompletionResponse;
    /// List available models.
    fn list_models(&self) -> Vec<ModelInfo>;
}

impl ApiServer {
    pub fn new(config: ApiServerConfig, handler: Box<dyn ApiHandler + Send>) -> Self {
        Self {
            config,
            handler: Arc::new(Mutex::new(handler)),
            armor: None,
        }
    }

    /// Enable Armor request (L0/L1) and response (L3) filtering.
    /// Hidden-state L2 probing requires model-level hooks and is not provided here.
    pub fn with_armor(mut self, armor: ArmorLayer) -> Self {
        self.armor = Some(Mutex::new(armor));
        self
    }

    fn input_allowed(&self, prompt: &str) -> bool {
        match &self.armor {
            None => true,
            Some(armor) => match armor.lock() {
                Ok(mut armor) => matches!(armor.verify_input(prompt), SecurityVerdict::Allow),
                Err(_) => false, // A failed security layer must never fail open.
            },
        }
    }

    fn completion_response(&self, mut response: ChatCompletionResponse, stream: bool) -> String {
        if let Some(armor) = &self.armor {
            let armor = match armor.lock() {
                Ok(armor) => armor,
                Err(_) => return http_forbidden(),
            };
            for choice in &mut response.choices {
                match armor.sanitize_output(&choice.message.content) {
                    SecurityVerdict::Allow => {},
                    SecurityVerdict::Redact(text) => choice.message.content = text,
                    SecurityVerdict::Block(_) => return http_forbidden(),
                }
            }
        }
        if stream {
            sse_response(response)
        } else {
            http_ok(response.to_json(), "application/json")
        }
    }

    /// Run the server (blocking).
    pub fn serve(&self) -> std::io::Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr)?;
        println!("FerrisRes API server listening on {}", addr);

        for stream in listener.incoming() {
            let mut stream = stream?;
            self.handle_connection(&mut stream);
        }
        Ok(())
    }

    fn handle_connection(&self, stream: &mut TcpStream) {
        let mut reader = BufReader::new(stream.try_clone().unwrap_or_else(|e| {
            eprintln!("Stream clone error: {}", e);
            stream.try_clone().unwrap()
        }));

        let request = match HttpRequest::from_stream(&mut reader) {
            Some(r) => r,
            None => {
                let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\n\r\n");
                return;
            }
        };

        let response = self.route(&request);
        let _ = stream.write_all(response.as_bytes());
    }

    fn route(&self, req: &HttpRequest) -> String {
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/health") => http_ok(r#"{"status":"ok"}"#.to_string(), "application/json"),
            ("GET", "/v1/models") => {
                let handler = self.handler.lock().unwrap();
                let models = handler.list_models();
                let data: Vec<String> = models.iter().map(|m| m.to_json()).collect();
                let body = format!(r#"{{"object":"list","data":[{}]}}"#, data.join(","));
                http_ok(body, "application/json")
            }
            ("POST", "/v1/chat/completions") => {
                let chat_req = self.parse_chat_request(&req.body);
                match chat_req {
                    Some(chat_req) => {
                        let prompt = chat_req.messages.iter()
                            .map(|message| format!("{}: {}", message.role, message.content))
                            .collect::<Vec<_>>()
                            .join("\n");
                        if !self.input_allowed(&prompt) {
                            return http_forbidden();
                        }
                        let mut handler = self.handler.lock().unwrap();
                        let resp = handler.chat_completion(&chat_req);
                        drop(handler);
                        self.completion_response(resp, chat_req.stream)
                    }
                    None => http_bad("Invalid request body"),
                }
            }
            ("POST", "/v1/completions") => {
                let comp_req = self.parse_completion_request(&req.body);
                match comp_req {
                    Some(comp_req) => {
                        if !self.input_allowed(&comp_req.prompt) {
                            return http_forbidden();
                        }
                        let mut handler = self.handler.lock().unwrap();
                        let resp = handler.completion(&comp_req);
                        drop(handler);
                        self.completion_response(resp, comp_req.stream)
                    }
                    None => http_bad("Invalid request body"),
                }
            }
            _ => http_not_found(),
        }
    }

    fn parse_chat_request(&self, body: &str) -> Option<ChatCompletionRequest> {
        // Simple JSON parsing (no serde dependency)
        let model = extract_json_string(body, "model").unwrap_or_else(|| "ferrisres".to_string());
        let max_tokens = extract_json_number(body, "max_tokens").unwrap_or(128.0) as usize;
        let temperature = extract_json_number(body, "temperature").unwrap_or(1.0) as f32;
        let top_p = extract_json_number(body, "top_p").unwrap_or(1.0) as f32;
        let stream = body.contains("\"stream\":true");

        // Parse messages array (simplified)
        let messages = parse_messages(body).unwrap_or_default();

        Some(ChatCompletionRequest {
            model,
            messages,
            max_tokens,
            temperature,
            top_p,
            stream,
        })
    }

    fn parse_completion_request(&self, body: &str) -> Option<CompletionRequest> {
        let model = extract_json_string(body, "model").unwrap_or_else(|| "ferrisres".to_string());
        let prompt = extract_json_string(body, "prompt").unwrap_or_default();
        let max_tokens = extract_json_number(body, "max_tokens").unwrap_or(128.0) as usize;
        let temperature = extract_json_number(body, "temperature").unwrap_or(1.0) as f32;
        let stream = body.contains("\"stream\":true");

        Some(CompletionRequest {
            model,
            prompt,
            max_tokens,
            temperature,
            stream,
        })
    }
}

// ---------------------------------------------------------------------------
// HTTP response helpers
// ---------------------------------------------------------------------------

fn http_ok(body: String, content_type: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        content_type, body.len(), body
    )
}

fn http_bad(body: &str) -> String {
    format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    )
}

fn http_forbidden() -> String {
    // Never reflect the blocked prompt, generated content or scanner reason.
    let body = r#"{"error":{"message":"Content blocked by Armor","type":"content_filter","code":"content_filter"}}"#;
    format!(
        "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        body.len(), body
    )
}

fn http_not_found() -> String {
    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_string()
}

fn sse_response(resp: ChatCompletionResponse) -> String {
    let chunk = sse_chunk(&resp.to_json());
    let done = sse_done();
    let body = format!("{}{}", chunk, done);
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    )
}

// ---------------------------------------------------------------------------
// Simple JSON helpers
// ---------------------------------------------------------------------------

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = &after_pos(after_key, colon_pos + 1);

    // Skip whitespace
    let trimmed = after_colon.trim_start();
    if !trimmed.starts_with('"') { return None; }
    let str_start = 1;
    let str_end = trimmed[1..].find('"')?;
    Some(trimmed[str_start..str_start + str_end].to_string())
}

fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = &after_pos(after_key, colon_pos + 1);
    let trimmed = after_colon.trim_start();

    let end = trimmed.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').unwrap_or(trimmed.len());
    trimmed[..end].parse().ok()
}

fn after_pos(s: &str, pos: usize) -> &str {
    if pos < s.len() { &s[pos..] } else { "" }
}

fn parse_messages(json: &str) -> Option<Vec<ChatMessage>> {
    let mut messages = Vec::new();
    let messages_start = json.find("\"messages\"")?;
    let array_start = json[messages_start..].find('[')?;
    let array_content = &json[messages_start + array_start..];

    // Find each message object
    let mut pos = 0;
    loop {
        let obj_start = array_content[pos..].find("{\"role\"")?;
        let obj_end = array_content[pos + obj_start..].find("}").unwrap_or(array_content.len() - pos - obj_start);
        let obj = &array_content[pos + obj_start..pos + obj_start + obj_end];
        let role = extract_json_string(obj, "role").unwrap_or_default();
        let content = extract_json_string(obj, "content").unwrap_or_default();
        messages.push(ChatMessage { role, content });
        pos += obj_start + obj_end + 1;
        if pos >= array_content.len() || !array_content[pos..].contains("{\"role\"") {
            break;
        }
    }

    if messages.is_empty() { None } else { Some(messages) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct SensitiveHandler(Arc<AtomicUsize>);

    impl ApiHandler for SensitiveHandler {
        fn chat_completion(&mut self, _req: &ChatCompletionRequest) -> ChatCompletionResponse {
            self.0.fetch_add(1, Ordering::SeqCst);
            ChatCompletionResponse {
                id: "test".into(), object: "chat.completion".into(), created: 0,
                model: "test".into(),
                choices: ["Contact alice@example.com", "Contact bob@example.com"].iter()
                    .enumerate().map(|(index, text)| Choice {
                        index, message: ChatMessage::assistant(text), finish_reason: "stop".into(),
                    }).collect(),
                usage: Usage { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            }
        }
        fn completion(&mut self, _req: &CompletionRequest) -> ChatCompletionResponse {
            self.chat_completion(&ChatCompletionRequest {
                model: "test".into(), messages: vec![], max_tokens: 1,
                temperature: 1.0, top_p: 1.0, stream: false,
            })
        }
        fn list_models(&self) -> Vec<ModelInfo> { vec![] }
    }

    fn armor_server() -> (ApiServer, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let server = ApiServer::new(ApiServerConfig::default(), Box::new(SensitiveHandler(calls.clone())))
            .with_armor(ArmorLayer::new());
        (server, calls)
    }

    fn http_request(server: &ApiServer, path: &str, body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        client.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let (mut connection, _) = listener.accept().unwrap();
                connection.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                server.handle_connection(&mut connection);
            });
            write!(client, "POST {path} HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
            let mut response = String::new();
            client.read_to_string(&mut response).unwrap();
            response
        })
    }

    #[test]
    fn armor_redacts_every_choice_before_json_or_sse_delivery() {
        for path in ["/v1/chat/completions", "/v1/completions"] {
            for stream in [false, true] {
                let (server, calls) = armor_server();
                let body = format!(r#"{{"messages":[{{"role":"user","content":"Hello"}}],"prompt":"Hello","stream":{stream}}}"#);
                let response = http_request(&server, path, &body);
                assert!(response.starts_with("HTTP/1.1 200"), "{response}");
                assert_eq!(calls.load(Ordering::SeqCst), 1);
                assert!(response.contains("Contact"));
                assert!(!response.contains("alice@example.com"), "{response}");
                assert!(!response.contains("bob@example.com"), "{response}");
                if stream { assert!(response.contains("data: [DONE]")); }
            }
        }
    }

    #[test]
    fn armor_rejects_input_before_handler_execution() {
        let (server, calls) = armor_server();
        for path in ["/v1/chat/completions", "/v1/completions"] {
            let response = http_request(&server, path,
                r#"{"messages":[{"role":"system","content":"alice@example.com"},{"role":"user","content":"Hello"}],"prompt":"alice@example.com"}"#);
            assert!(response.starts_with("HTTP/1.1 403"), "{response}");
            assert!(!response.contains("alice@example.com"));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_chat_message_constructors() {
        let sys = ChatMessage::system("You are helpful");
        assert_eq!(sys.role, "system");
        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, "user");
        let asst = ChatMessage::assistant("Hi there");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn test_chat_response_json() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "ferrisres".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage::assistant("Hello!"),
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };
        let json = resp.to_json();
        assert!(json.contains("chatcmpl-123"));
        assert!(json.contains("Hello!"));
        assert!(json.contains("prompt_tokens\":10"));
    }

    #[test]
    fn test_model_info_json() {
        let info = ModelInfo {
            id: "ferrisres".to_string(),
            object: "model".to_string(),
            owned_by: "ferrisres".to_string(),
        };
        let json = info.to_json();
        assert!(json.contains("ferrisres"));
    }

    #[test]
    fn test_sse_chunk() {
        let chunk = sse_chunk("{\"text\":\"hi\"}");
        assert!(chunk.starts_with("data: "));
        assert!(chunk.ends_with("\n\n"));
    }

    #[test]
    fn test_sse_done() {
        let done = sse_done();
        assert_eq!(done, "data: [DONE]\n\n");
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello \"world\""), "hello \\\"world\\\"");
        assert_eq!(escape_json("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_extract_json_string() {
        let json = r#"{"model":"ferrisres","max_tokens":128}"#;
        assert_eq!(extract_json_string(json, "model"), Some("ferrisres".to_string()));
    }

    #[test]
    fn test_extract_json_number() {
        let json = r#"{"max_tokens":128,"temperature":0.7}"#;
        assert_eq!(extract_json_number(json, "max_tokens"), Some(128.0));
        assert!((extract_json_number(json, "temperature").unwrap() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_parse_messages() {
        let json = r#"{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"}]}"#;
        let msgs = parse_messages(json).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "Hello");
        assert_eq!(msgs[1].role, "assistant");
    }

    #[test]
    fn test_parse_messages_system() {
        let json = r#"{"messages":[{"role":"system","content":"Be helpful"},{"role":"user","content":"Hi"}]}"#;
        let msgs = parse_messages(json).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
    }

    #[test]
    fn test_http_ok_response() {
        let resp = http_ok("{}".to_string(), "application/json");
        assert!(resp.starts_with("HTTP/1.1 200 OK"));
        assert!(resp.contains("Content-Type: application/json"));
        assert!(resp.contains("Access-Control-Allow-Origin: *"));
    }

    #[test]
    fn test_http_not_found() {
        let resp = http_not_found();
        assert!(resp.starts_with("HTTP/1.1 404"));
    }

    #[test]
    fn test_api_config_default() {
        let config = ApiServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.model_name, "ferrisres");
    }

    #[test]
    fn test_usage() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        assert_eq!(usage.total_tokens, usage.prompt_tokens + usage.completion_tokens);
    }

    #[test]
    fn test_extract_missing_key() {
        let json = r#"{"foo":"bar"}"#;
        assert!(extract_json_string(json, "model").is_none());
    }
}
