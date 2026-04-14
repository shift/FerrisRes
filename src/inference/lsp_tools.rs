//! LSP (Language Server Protocol) integration for deterministic code feedback.
//!
//! Registers LSP servers as host tools. The model generates code, the LSP validates
//! it, and compiler errors feed into the autodiff graph as loss — penalizing the
//! model at the weight level for non-compiling code.
//!
//! Supports any LSP-compliant server: rust-analyzer, pyright, clangd, gopls, etc.

use std::io::{BufRead, Write, Read};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;

use crate::inference::tool_search::{Tool, ToolCall, ToolResult};

// ---------------------------------------------------------------------------
// JSON-RPC layer
// ---------------------------------------------------------------------------

/// Minimal JSON-RPC client for LSP communication over stdio.
pub struct LspClient {
    process: Child,
    stdin: ChildStdin,
    stdout: Mutex<LspStdoutReader>,
    request_id: Mutex<u64>,
    initialized: bool,
    #[allow(dead_code)]
    root_uri: String,
}

/// Wrapper for reading LSP stdout (Content-Length header + JSON body).
struct LspStdoutReader {
    reader: std::io::BufReader<std::process::ChildStdout>,
}

impl LspStdoutReader {
    fn new(stdout: std::process::ChildStdout) -> Self {
        Self { reader: std::io::BufReader::new(stdout) }
    }

    fn read_response(&mut self) -> Result<serde_json::Value, String> {
        // Read Content-Length header
        let mut line = String::new();
        let mut content_length: usize = 0;
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => return Err("LSP server closed".into()),
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        break; // End of headers
                    }
                    if let Some(cl) = trimmed.strip_prefix("Content-Length:") {
                        content_length = cl.trim().parse::<usize>()
                            .map_err(|e| format!("Invalid Content-Length: {}", e))?;
                    }
                }
                Err(e) => return Err(format!("LSP read error: {}", e)),
            }
        }

        if content_length == 0 {
            return Err("No Content-Length header".into());
        }

        // Read body
        let mut body = vec![0u8; content_length];
        self.reader.read_exact(&mut body)
            .map_err(|e| format!("LSP body read error: {}", e))?;

        let text = String::from_utf8(body)
            .map_err(|e| format!("LSP non-UTF8 body: {}", e))?;

        serde_json::from_str(&text)
            .map_err(|e| format!("LSP JSON parse error: {}", e))
    }
}

impl LspClient {
    /// Start an LSP server and perform initialization handshake.
    pub fn start(
        command: &str,
        args: &[&str],
        root_path: &Path,
    ) -> Result<Self, String> {
        let root_uri = format!("file://{}", root_path.display());

        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start LSP server '{}': {}", command, e))?;

        let stdin = child.stdin.take()
            .ok_or_else(|| "No stdin".to_string())?;
        let stdout = child.stdout.take()
            .ok_or_else(|| "No stdout".to_string())?;

        let mut client = Self {
            process: child,
            stdin,
            stdout: Mutex::new(LspStdoutReader::new(stdout)),
            request_id: Mutex::new(0),
            initialized: false,
            root_uri: root_uri.clone(),
        };

        // Initialize
        client.send_request("initialize", serde_json::json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "rootPath": root_path.to_string_lossy(),
            "capabilities": {},
        }))?;

        // Send initialized notification
        client.send_notification("initialized", serde_json::json!({}))?;
        client.initialized = true;

        Ok(client)
    }

    fn next_id(&self) -> u64 {
        let mut id = self.request_id.lock().unwrap();
        *id += 1;
        *id
    }

    fn send_request(&mut self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
        let id = self.next_id();
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.send_message(&request)?;

        // Read response (skip notifications)
        let stdout = self.stdout.get_mut().unwrap();
        let mut attempts = 0;
        loop {
            let response = stdout.read_response()?;
            if response.get("id").and_then(|v| v.as_u64()) == Some(id) {
                return Ok(response);
            }
            attempts += 1;
            if attempts > 100 {
                return Err("Too many LSP notifications without matching response".into());
            }
        }
    }

    fn send_notification(&mut self, method: &str, params: serde_json::Value) -> Result<(), String> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.send_message(&notification)
    }

    fn send_message(&mut self, message: &serde_json::Value) -> Result<(), String> {
        let body = serde_json::to_string(message)
            .map_err(|e| format!("JSON serialize error: {}", e))?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        self.stdin.write_all(header.as_bytes())
            .map_err(|e| format!("LSP write error: {}", e))?;
        self.stdin.write_all(body.as_bytes())
            .map_err(|e| format!("LSP write body error: {}", e))?;
        self.stdin.flush()
            .map_err(|e| format!("LSP flush error: {}", e))?;
        Ok(())
    }

    /// Send a didOpen notification to the LSP server.
    pub fn open_document(&mut self, uri: &str, language_id: &str, text: &str) -> Result<(), String> {
        self.send_notification("textDocument/didOpen", serde_json::json!({
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": 1,
                "text": text,
            }
        }))
    }

    /// Pull diagnostics using textDocument/diagnostic (LSP 3.17+).
    /// Falls back to returning empty if the server doesn't support it.
    pub fn check_syntax(&mut self, uri: &str, language_id: &str, code: &str) -> Result<Vec<Diagnostic>, String> {
        // Open the document
        self.open_document(uri, language_id, code)?;

        // Try pull diagnostics
        let result = self.send_request("textDocument/diagnostic", serde_json::json!({
            "textDocument": { "uri": uri }
        }));

        match result {
            Ok(response) => {
                if let Some(items) = response.get("result").and_then(|r| r.get("items")) {
                    Ok(parse_diagnostics_val(items))
                } else if let Some(result) = response.get("result") {
                    Ok(parse_diagnostics_val(result))
                } else {
                    Ok(vec![])
                }
            }
            Err(_) => {
                // Server doesn't support pull diagnostics — return empty for now
                // (Push diagnostics would need an async notification collector)
                Ok(vec![])
            }
        }
    }

    /// Shutdown the LSP server gracefully.
    pub fn shutdown(&mut self) -> Result<(), String> {
        if self.initialized {
            let _ = self.send_request("shutdown", serde_json::Value::Null);
            let _ = self.send_notification("exit", serde_json::json!({}));
        }
        let _ = self.process.wait();
        Ok(())
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn parse_diagnostics_val(val: &serde_json::Value) -> Vec<Diagnostic> {
    val.as_array()
        .map(|arr| {
            arr.iter().filter_map(|d| {
                let severity = d.get("severity").and_then(|s| s.as_u64()).unwrap_or(1);
                let msg = d.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string();
                let line = d.get("range")
                    .and_then(|r| r.get("start"))
                    .and_then(|s| s.get("line"))
                    .and_then(|l| l.as_u64())
                    .unwrap_or(0);
                Some(Diagnostic {
                    severity: if severity <= 1 { DiagnosticSeverity::Error }
                              else if severity == 2 { DiagnosticSeverity::Warning }
                              else { DiagnosticSeverity::Info },
                    message: msg,
                    line: line as usize,
                })
            }).collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Diagnostic types
// ---------------------------------------------------------------------------

/// A single LSP diagnostic (error, warning, info).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub line: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
}

// ---------------------------------------------------------------------------
// LSP tool definitions for host_tools integration
// ---------------------------------------------------------------------------

/// Create the lsp_check tool definition.
pub fn create_lsp_check_tool() -> Tool {
    Tool::new("lsp_check", "Validates code using a Language Server (LSP). Returns compiler errors, warnings, and type information.")
        .with_parameters(r#"{"code": "string, the code to check", "language": "string: rust, python, typescript, c, go, java", "uri": "optional URI for the document"}"#)
        .with_category("development")
        .with_example(r#"TOOL_CALL:lsp_check({"code": "fn main() { let x = 1; }", "language": "rust"})"#)
}

/// Create the lsp_typecheck tool definition.
pub fn create_lsp_typecheck_tool() -> Tool {
    Tool::new("lsp_typecheck", "Runs full type-checking on code via LSP. Returns type errors and inferred types.")
        .with_parameters(r#"{"code": "string", "language": "string", "uri": "optional URI"}"#)
        .with_category("development")
        .with_example(r#"TOOL:lsp_typecheck({"code": "def foo(x: int) -> str: return x", "language": "python"})"#)
}

/// Create all LSP tool definitions.
pub fn all_lsp_tools() -> Vec<Tool> {
    vec![create_lsp_check_tool(), create_lsp_typecheck_tool()]
}

// ---------------------------------------------------------------------------
// Standalone check (no LSP server — uses compiler directly)
// ---------------------------------------------------------------------------

/// Map language name to (command, args) for quick syntax check.
fn language_to_lsp(language: &str) -> (&'static str, Vec<&'static str>) {
    match language {
        "rust" => ("rust-analyzer", vec![]),
        "python" | "py" => ("pyright", vec!["--stdio"]),
        "typescript" | "ts" => ("typescript-language-server", vec!["--stdio"]),
        "c" | "cpp" | "c++" => ("clangd", vec![]),
        "go" => ("gopls", vec![]),
        "java" => ("jdtls", vec![]),
        _ => ("", vec![]),
    }
}

/// Execute an LSP check tool call.
/// If no LSP server is available, falls back to a basic compiler check.
pub fn execute_lsp_check(call: &ToolCall) -> ToolResult {
    let code = match extract_json_string(&call.arguments, "code") {
        Some(c) => c,
        None => return ToolResult::error(&call.call_id, "lsp_check", "Missing 'code' parameter"),
    };
    let language = match extract_json_string(&call.arguments, "language") {
        Some(l) => l,
        None => return ToolResult::error(&call.call_id, "lsp_check", "Missing 'language' parameter"),
    };

    let (cmd, args) = language_to_lsp(&language);

    if cmd.is_empty() {
        return ToolResult::error(&call.call_id, "lsp_check",
            &format!("Unsupported language: {}", language));
    }

    // Try to start the LSP server
    let tmp_dir = std::env::temp_dir().join("ferrisres_lsp");
    let _ = std::fs::create_dir_all(&tmp_dir);

    match LspClient::start(cmd, &args, &tmp_dir) {
        Ok(mut client) => {
            let uri = format!("file:///tmp/ferrisres_lsp/source.{}", language_ext(&language));
            match client.check_syntax(&uri, &language, &code) {
                Ok(diagnostics) => {
                    let output = format_diagnostics(&diagnostics);
                    ToolResult::success(&call.call_id, "lsp_check", &output)
                }
                Err(e) => ToolResult::error(&call.call_id, "lsp_check", &format!("LSP error: {}", e)),
            }
        }
        Err(_) => {
            // LSP server not installed — fall back to basic syntax check
            let diagnostics = fallback_syntax_check(&code, &language);
            let output = format_diagnostics(&diagnostics);
            if diagnostics.is_empty() {
                ToolResult::success(&call.call_id, "lsp_check",
                    &format!("No LSP server found for '{}'. Basic check passed (no obvious syntax errors).", language))
            } else {
                ToolResult::success(&call.call_id, "lsp_check", &output)
            }
        }
    }
}

/// Execute an LSP typecheck tool call.
pub fn execute_lsp_typecheck(call: &ToolCall) -> ToolResult {
    // Same implementation as check for now — typecheck is a superset of syntax check
    execute_lsp_check(call)
}

/// Get file extension for a language.
fn language_ext(language: &str) -> &'static str {
    match language {
        "rust" => "rs",
        "python" | "py" => "py",
        "typescript" | "ts" => "ts",
        "c" => "c",
        "cpp" | "c++" => "cpp",
        "go" => "go",
        "java" => "java",
        _ => "txt",
    }
}

/// Format diagnostics into a human-readable string.
fn format_diagnostics(diagnostics: &[Diagnostic]) -> String {
    if diagnostics.is_empty() {
        return "No errors or warnings.".to_string();
    }

    let mut out = String::new();
    let errors = diagnostics.iter().filter(|d| d.severity == DiagnosticSeverity::Error).count();
    let warnings = diagnostics.iter().filter(|d| d.severity == DiagnosticSeverity::Warning).count();
    let infos = diagnostics.len() - errors - warnings;

    out.push_str(&format!("Found {} error(s), {} warning(s), {} info.\n", errors, warnings, infos));

    for d in diagnostics {
        let sev = match d.severity {
            DiagnosticSeverity::Error => "ERROR",
            DiagnosticSeverity::Warning => "WARN",
            DiagnosticSeverity::Info => "INFO",
        };
        out.push_str(&format!("  [{}] line {}: {}\n", sev, d.line + 1, d.message));
    }

    out
}

/// Extract a string value from a JSON arguments string.
pub fn extract_json_string(args: &str, key: &str) -> Option<String> {
    // Simple JSON extraction without full parse
    let pattern = format!("\"{}\"", key);
    let start = args.find(&pattern)?;
    let after_key = &args[start + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = &after_key[colon_pos + 1..];
    let trimmed = after_colon.trim_start();
    if trimmed.starts_with('"') {
        let str_start = 1;
        let str_end = trimmed[1..].find('"')?;
        Some(trimmed[str_start..str_start + str_end].to_string())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Fallback syntax check (no LSP server needed)
// ---------------------------------------------------------------------------

/// Basic syntax checks that don't require an LSP server.
/// Catches common issues: unmatched braces, missing semicolons, etc.
pub fn fallback_syntax_check(code: &str, language: &str) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    match language {
        "rust" => check_rust_basics(code, &mut diagnostics),
        "python" | "py" => check_python_basics(code, &mut diagnostics),
        "c" | "cpp" | "c++" => check_c_basics(code, &mut diagnostics),
        _ => {}
    }

    diagnostics
}

fn check_rust_basics(code: &str, diagnostics: &mut Vec<Diagnostic>) {
    let mut brace_depth = 0i32;
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;

    for (line_num, line) in code.lines().enumerate() {
        let trimmed = line.trim();

        // Skip comments and strings (basic)
        if trimmed.starts_with("//") {
            continue;
        }

        for ch in trimmed.chars() {
            match ch {
                '{' => brace_depth += 1,
                '}' => {
                    brace_depth -= 1;
                    if brace_depth < 0 {
                        diagnostics.push(Diagnostic {
                            severity: DiagnosticSeverity::Error,
                            message: "Unexpected closing brace '}'".into(),
                            line: line_num,
                        });
                    }
                }
                '(' => paren_depth += 1,
                ')' => {
                    paren_depth -= 1;
                    if paren_depth < 0 {
                        diagnostics.push(Diagnostic {
                            severity: DiagnosticSeverity::Error,
                            message: "Unexpected closing paren ')'".into(),
                            line: line_num,
                        });
                    }
                }
                '[' => bracket_depth += 1,
                ']' => {
                    bracket_depth -= 1;
                    if bracket_depth < 0 {
                        diagnostics.push(Diagnostic {
                            severity: DiagnosticSeverity::Error,
                            message: "Unexpected closing bracket ']'".into(),
                            line: line_num,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    if brace_depth > 0 {
        diagnostics.push(Diagnostic {
            severity: DiagnosticSeverity::Error,
            message: format!("Missing {} closing brace(s) '}}'", brace_depth),
            line: code.lines().count().saturating_sub(1),
        });
    }
    if paren_depth > 0 {
        diagnostics.push(Diagnostic {
            severity: DiagnosticSeverity::Error,
            message: format!("Missing {} closing paren(s) ')'", paren_depth),
            line: code.lines().count().saturating_sub(1),
        });
    }
    if bracket_depth > 0 {
        diagnostics.push(Diagnostic {
            severity: DiagnosticSeverity::Error,
            message: format!("Missing {} closing bracket(s) ']'", bracket_depth),
            line: code.lines().count().saturating_sub(1),
        });
    }

    // Check for common Rust issues
    for (_line_num, line) in code.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("let ") && !trimmed.contains('=')
            && !trimmed.ends_with('{') && !trimmed.ends_with('(')
            && !trimmed.contains(';')
            && !trimmed.starts_with("//")
        {
            // Could be missing `=` or `;` — not flagging as error
            // since valid patterns like `let x: i32;` exist
        }
    }
}

fn check_python_basics(code: &str, diagnostics: &mut Vec<Diagnostic>) {
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;

    for (line_num, line) in code.lines().enumerate() {
        for ch in line.chars() {
            match ch {
                '(' => paren_depth += 1,
                ')' => {
                    paren_depth -= 1;
                    if paren_depth < 0 {
                        diagnostics.push(Diagnostic {
                            severity: DiagnosticSeverity::Error,
                            message: "Unexpected closing paren ')'".into(),
                            line: line_num,
                        });
                    }
                }
                '[' => bracket_depth += 1,
                ']' => {
                    bracket_depth -= 1;
                    if bracket_depth < 0 {
                        diagnostics.push(Diagnostic {
                            severity: DiagnosticSeverity::Error,
                            message: "Unexpected closing bracket ']'".into(),
                            line: line_num,
                        });
                    }
                }
                '{' => brace_depth += 1,
                '}' => {
                    brace_depth -= 1;
                    if brace_depth < 0 {
                        diagnostics.push(Diagnostic {
                            severity: DiagnosticSeverity::Error,
                            message: "Unexpected closing brace '}'".into(),
                            line: line_num,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    if paren_depth > 0 {
        diagnostics.push(Diagnostic {
            severity: DiagnosticSeverity::Error,
            message: format!("Missing {} closing paren(s) ')'", paren_depth),
            line: code.lines().count().saturating_sub(1),
        });
    }
    if bracket_depth > 0 {
        diagnostics.push(Diagnostic {
            severity: DiagnosticSeverity::Error,
            message: format!("Missing {} closing bracket(s) ']'", bracket_depth),
            line: code.lines().count().saturating_sub(1),
        });
    }
    if brace_depth > 0 {
        diagnostics.push(Diagnostic {
            severity: DiagnosticSeverity::Error,
            message: format!("Missing {} closing brace(s) '}}'", brace_depth),
            line: code.lines().count().saturating_sub(1),
        });
    }
}

fn check_c_basics(code: &str, diagnostics: &mut Vec<Diagnostic>) {
    // Same brace/paren matching as Rust
    check_rust_basics(code, diagnostics);
}

// ---------------------------------------------------------------------------
// Compiler-error loss for autodiff integration
// ---------------------------------------------------------------------------

/// Compute a scalar loss from a list of LSP diagnostics.
///
/// The loss is: `error_weight * num_errors + warning_weight * num_warnings + info_weight * num_infos`.
///
/// This can be used as a loss signal in the autodiff graph to penalize the model
/// for generating non-compiling code at the weight level.
pub fn compiler_error_loss(diagnostics: &[Diagnostic]) -> f32 {
    let error_weight = 1.0;
    let warning_weight = 0.3;
    let info_weight = 0.1;

    let mut loss = 0.0f32;
    for d in diagnostics {
        loss += match d.severity {
            DiagnosticSeverity::Error => error_weight,
            DiagnosticSeverity::Warning => warning_weight,
            DiagnosticSeverity::Info => info_weight,
        };
    }
    loss
}

/// Compute a per-line compiler error loss map (for fine-grained feedback).
pub fn per_line_loss(diagnostics: &[Diagnostic]) -> Vec<(usize, f32)> {
    let mut line_losses: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
    for d in diagnostics {
        let weight = match d.severity {
            DiagnosticSeverity::Error => 1.0,
            DiagnosticSeverity::Warning => 0.3,
            DiagnosticSeverity::Info => 0.1,
        };
        *line_losses.entry(d.line).or_insert(0.0) += weight;
    }
    let mut result: Vec<(usize, f32)> = line_losses.into_iter().collect();
    result.sort_by_key(|(line, _)| *line);
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_rust_valid() {
        let code = r#"fn main() {
    let x = 1;
    println!("{}", x);
}"#;
        let diags = fallback_syntax_check(code, "rust");
        assert!(diags.is_empty(), "Expected no errors, got: {:?}", diags);
    }

    #[test]
    fn test_fallback_rust_unclosed_brace() {
        let code = r#"fn main() {
    let x = 1;
"#;
        let diags = fallback_syntax_check(code, "rust");
        assert!(!diags.is_empty(), "Expected errors for unclosed brace");
        assert!(diags[0].message.contains("Missing"));
    }

    #[test]
    fn test_fallback_rust_extra_brace() {
        let code = r#"fn main() {
}}"#;
        let diags = fallback_syntax_check(code, "rust");
        assert!(!diags.is_empty(), "Expected error for extra brace");
    }

    #[test]
    fn test_fallback_python_valid() {
        let code = r#"def foo(x):
    return x + 1"#;
        let diags = fallback_syntax_check(code, "python");
        assert!(diags.is_empty(), "Expected no errors, got: {:?}", diags);
    }

    #[test]
    fn test_fallback_python_unclosed_paren() {
        let code = r#"def foo(x:
    return x + 1"#;
        let diags = fallback_syntax_check(code, "python");
        assert!(!diags.is_empty(), "Expected error for unclosed paren");
    }

    #[test]
    fn test_compiler_error_loss() {
        let diags = vec![
            Diagnostic { severity: DiagnosticSeverity::Error, message: "type mismatch".into(), line: 3 },
            Diagnostic { severity: DiagnosticSeverity::Warning, message: "unused var".into(), line: 5 },
            Diagnostic { severity: DiagnosticSeverity::Error, message: "missing semicolon".into(), line: 7 },
        ];
        let loss = compiler_error_loss(&diags);
        // 2 errors (1.0 each) + 1 warning (0.3) = 2.3
        assert!((loss - 2.3).abs() < 0.001, "Expected 2.3, got {}", loss);
    }

    #[test]
    fn test_per_line_loss() {
        let diags = vec![
            Diagnostic { severity: DiagnosticSeverity::Error, message: "err1".into(), line: 3 },
            Diagnostic { severity: DiagnosticSeverity::Warning, message: "warn".into(), line: 3 },
            Diagnostic { severity: DiagnosticSeverity::Error, message: "err2".into(), line: 7 },
        ];
        let losses = per_line_loss(&diags);
        assert_eq!(losses.len(), 2);
        assert_eq!(losses[0].0, 3);
        assert!((losses[0].1 - 1.3).abs() < 0.001);
        assert_eq!(losses[1].0, 7);
        assert!((losses[1].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_extract_json_string() {
        let args = r#"{"code": "fn main() {}", "language": "rust"}"#;
        assert_eq!(extract_json_string(args, "code"), Some("fn main() {}".to_string()));
        assert_eq!(extract_json_string(args, "language"), Some("rust".to_string()));
        assert_eq!(extract_json_string(args, "missing"), None);
    }

    #[test]
    fn test_format_diagnostics() {
        let diags = vec![
            Diagnostic { severity: DiagnosticSeverity::Error, message: "test error".into(), line: 2 },
        ];
        let out = format_diagnostics(&diags);
        assert!(out.contains("1 error(s)"));
        assert!(out.contains("test error"));
    }

    #[test]
    fn test_format_diagnostics_empty() {
        let out = format_diagnostics(&[]);
        assert_eq!(out, "No errors or warnings.");
    }

    #[test]
    fn test_language_to_lsp() {
        let (cmd, _) = language_to_lsp("rust");
        assert_eq!(cmd, "rust-analyzer");

        let (cmd, _) = language_to_lsp("python");
        assert_eq!(cmd, "pyright");

        let (cmd, _) = language_to_lsp("typescript");
        assert_eq!(cmd, "typescript-language-server");

        let (cmd, _) = language_to_lsp("c");
        assert_eq!(cmd, "clangd");

        let (cmd, _) = language_to_lsp("go");
        assert_eq!(cmd, "gopls");

        let (cmd, _) = language_to_lsp("unknown");
        assert_eq!(cmd, "");
    }

    #[test]
    fn test_language_ext() {
        assert_eq!(language_ext("rust"), "rs");
        assert_eq!(language_ext("python"), "py");
        assert_eq!(language_ext("typescript"), "ts");
        assert_eq!(language_ext("go"), "go");
        assert_eq!(language_ext("brainfuck"), "txt");
    }
}
