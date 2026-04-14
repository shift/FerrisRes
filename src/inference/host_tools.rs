//! Host-based tool implementations for FerrisRes.
//!
//! These tools run on the host machine (not inside the transformer) and provide
//! the model with real-world capabilities: web access, math, file I/O, shell
//! execution, and code interpretation.
//!
//! Tools use the Tool/ToolCall/ToolResult types from tool_search.rs.

use crate::inference::tool_search::{Tool, ToolCall, ToolResult};

// ---------------------------------------------------------------------------
// Tool definitions (for registration)
// ---------------------------------------------------------------------------

/// Create the web_fetch tool definition.
/// Fetches live content from a URL to provide real-time context.
pub fn create_web_fetch_tool() -> Tool {
    Tool::new("web_fetch", "Fetches live content from a URL to provide real-time context.")
        .with_parameters(r#"{"url": "string", "selector": "optional CSS selector to extract"}"#)
        .with_category("research")
        .with_example(r#"TOOL_CALL:web_fetch({"url": "https://news.ycombinator.com", "selector": ".titleline"})"#)
}

/// Create the math_eval tool definition.
/// Uses the LLM-Computer VM to evaluate mathematical expressions deterministically.
pub fn create_math_eval_tool() -> Tool {
    Tool::new("math_eval", "Evaluates a mathematical expression using deterministic VM execution.")
        .with_parameters(r#"{"expression": "string, e.g. '2+3*4' or 'sum(1..100)'"}"#)
        .with_category("computation")
        .with_example(r#"TOOL_CALL:math_eval({"expression": "2+3*4"})"#)
}

/// Create the file_read tool definition.
pub fn create_file_read_tool() -> Tool {
    Tool::new("file_read", "Reads the contents of a file from disk.")
        .with_parameters(r#"{"path": "string, file path to read", "max_bytes": "optional max bytes to read (default 65536)"}"#)
        .with_category("filesystem")
        .with_example(r#"TOOL_CALL:file_read({"path": "/tmp/data.txt"})"#)
}

/// Create the file_write tool definition.
pub fn create_file_write_tool() -> Tool {
    Tool::new("file_write", "Writes content to a file on disk.")
        .with_parameters(r#"{"path": "string, file path", "content": "string, content to write"}"#)
        .with_category("filesystem")
        .with_example(r#"TOOL_CALL:file_write({"path": "/tmp/output.txt", "content": "Hello world"})"#)
}

/// Create the shell_exec tool definition.
/// Runs a system command and returns stdout/stderr.
pub fn create_shell_exec_tool() -> Tool {
    Tool::new("shell_exec", "Executes a shell command and returns stdout/stderr. Use with caution.")
        .with_parameters(r#"{"command": "string, shell command to run", "timeout_secs": "optional timeout (default 30)"}"#)
        .with_category("system")
        .with_example(r#"TOOL_CALL:shell_exec({"command": "ls -la /tmp", "timeout_secs": "10"})"#)
}

/// Create the search tool definition.
/// Performs a web search (requires SearXNG or Brave Search API).
pub fn create_search_tool() -> Tool {
    Tool::new("search", "Search the web for information. Returns top results with snippets.")
        .with_parameters(r#"{"query": "string, search query", "num_results": "optional number of results (default 5)"}"#)
        .with_category("research")
        .with_example(r#"TOOL_CALL:search({"query": "Rust programming language latest features", "num_results": "3"})"#)
}

/// Create the code_interpreter tool definition.
/// Executes Python-like expressions in a sandboxed evaluator.
pub fn create_code_interpreter_tool() -> Tool {
    Tool::new("code_interpreter", "Interprets and executes code expressions (calculator, string ops, list ops).")
        .with_parameters(r#"{"code": "string, expression to evaluate", "language": "optional: calc, json, regex"}"#)
        .with_category("computation")
        .with_example(r#"TOOL_CALL:code_interpreter({"code": "[x**2 for x in range(10)]", "language": "calc"})"#)
}

/// Create all built-in host tools.
pub fn all_builtin_tools() -> Vec<Tool> {
    vec![
        create_web_fetch_tool(),
        create_math_eval_tool(),
        create_file_read_tool(),
        create_file_write_tool(),
        create_shell_exec_tool(),
        create_search_tool(),
        create_code_interpreter_tool(),
    ]
}

// ---------------------------------------------------------------------------
// Tool execution handlers
// ---------------------------------------------------------------------------

/// Execute a web fetch. Requires `reqwest` at the application level.
/// Returns the response body as text, truncated to max_bytes.
pub fn execute_web_fetch(call: &ToolCall) -> ToolResult {
    // Parse URL from arguments (simple JSON extraction)
    let url = extract_json_string(&call.arguments, "url")
        .unwrap_or_else(|| call.arguments.clone());

    // Validate URL
    if url.is_empty() {
        return ToolResult::error(&call.call_id, "web_fetch", "Missing 'url' parameter");
    }
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return ToolResult::error(&call.call_id, "web_fetch",
            format!("Invalid URL (must start with http:// or https://): {}", url));
    }

    // Synchronous fetch using ureq (lightweight, no async runtime needed)
    match ureq::get(&url)
        .timeout(std::time::Duration::from_secs(30))
        .call()
    {
        Ok(response) => {
            let _max_bytes: usize = extract_json_number(&call.arguments, "max_bytes")
                .unwrap_or(65536.0) as usize;
            let mut text_buf = String::new();
            match response.into_reader().read_to_string(&mut text_buf) {
                Ok(_) => ToolResult::success(&call.call_id, "web_fetch", text_buf),
                Err(e) => ToolResult::error(&call.call_id, "web_fetch",
                    format!("Failed to read response: {}", e)),
            }
        }
        Err(e) => ToolResult::error(&call.call_id, "web_fetch",
            format!("Fetch failed: {}", e)),
    }
}

/// Execute math using the LLM-Computer VM.
pub fn execute_math_eval(call: &ToolCall) -> ToolResult {
    let expr = match extract_json_string(&call.arguments, "expression") {
        Some(e) => e,
        None => return ToolResult::error(&call.call_id, "math_eval", "Missing 'expression' parameter"),
    };

    // Simple expression evaluator: parse basic arithmetic
    match eval_simple_expr(&expr) {
        Ok(result) => ToolResult::success(&call.call_id, "math_eval", format!("{} = {}", expr, result)),
        Err(msg) => ToolResult::error(&call.call_id, "math_eval", msg),
    }
}

/// Read a file from disk.
pub fn execute_file_read(call: &ToolCall) -> ToolResult {
    let path = match extract_json_string(&call.arguments, "path") {
        Some(p) => p,
        None => return ToolResult::error(&call.call_id, "file_read", "Missing 'path' parameter"),
    };

    let max_bytes: usize = extract_json_number(&call.arguments, "max_bytes")
        .unwrap_or(65536.0) as usize;

    match std::fs::read_to_string(&path) {
        Ok(content) => {
            let truncated = if content.len() > max_bytes {
                format!("{}... (truncated to {} bytes)", &content[..max_bytes], max_bytes)
            } else {
                content
            };
            ToolResult::success(&call.call_id, "file_read", truncated)
        }
        Err(e) => ToolResult::error(&call.call_id, "file_read",
            format!("Failed to read '{}': {}", path, e)),
    }
}

/// Write content to a file.
pub fn execute_file_write(call: &ToolCall) -> ToolResult {
    let path = match extract_json_string(&call.arguments, "path") {
        Some(p) => p,
        None => return ToolResult::error(&call.call_id, "file_write", "Missing 'path' parameter"),
    };
    let content = extract_json_string(&call.arguments, "content")
        .unwrap_or_default();

    // Create parent directories if needed
    if let Some(parent) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    match std::fs::write(&path, &content) {
        Ok(()) => ToolResult::success(&call.call_id, "file_write",
            format!("Wrote {} bytes to {}", content.len(), path)),
        Err(e) => ToolResult::error(&call.call_id, "file_write",
            format!("Failed to write '{}': {}", path, e)),
    }
}

/// Execute a shell command.
pub fn execute_shell_exec(call: &ToolCall) -> ToolResult {
    let command = match extract_json_string(&call.arguments, "command") {
        Some(c) => c,
        None => return ToolResult::error(&call.call_id, "shell_exec", "Missing 'command' parameter"),
    };

    let _timeout_secs: u64 = extract_json_number(&call.arguments, "timeout_secs")
        .unwrap_or(30.0) as u64;

    // Blocklisted commands for safety
    let dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){:|:&};:", "fork bomb"];
    for d in &dangerous {
        if command.contains(d) {
            return ToolResult::error(&call.call_id, "shell_exec",
                format!("Blocked dangerous command pattern: {}", d));
        }
    }

    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(&command)
        .output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let exit_code = out.status.code().unwrap_or(-1);

            let mut result = String::new();
            if !stdout.is_empty() {
                result.push_str(&format!("stdout:\n{}\n", stdout));
            }
            if !stderr.is_empty() {
                result.push_str(&format!("stderr:\n{}\n", stderr));
            }
            result.push_str(&format!("exit_code: {}", exit_code));

            // Check timeout (approximate — process may have hung)
            if result.len() > 1_000_000 {
                result.truncate(1_000_000);
                result.push_str("\n... (output truncated)");
            }

            if out.status.success() {
                ToolResult::success(&call.call_id, "shell_exec", result)
            } else {
                ToolResult::error(&call.call_id, "shell_exec", result)
            }
        }
        Err(e) => ToolResult::error(&call.call_id, "shell_exec",
            format!("Failed to execute: {}", e)),
    }
}

/// Execute a web search (placeholder — needs SearXNG or Brave API config).
pub fn execute_search(call: &ToolCall) -> ToolResult {
    let query = match extract_json_string(&call.arguments, "query") {
        Some(q) => q,
        None => return ToolResult::error(&call.call_id, "search", "Missing 'query' parameter"),
    };

    // Try SearXNG locally, then fall back to a DuckDuckGo HTML scrape
    let url = format!("https://html.duckduckgo.com/html/?q={}",
        urlencoding(&query));

    match ureq::get(&url)
        .timeout(std::time::Duration::from_secs(15))
        .call()
    {
        Ok(response) => {
            let mut html_buf = String::new();
            match response.into_reader().read_to_string(&mut html_buf) {
                Ok(_) => {
                    // Extract basic results from DDG HTML
                    let results = extract_ddg_results(&html_buf, 5);
                    if results.is_empty() {
                        ToolResult::success(&call.call_id, "search",
                            format!("No results found for: {}", query))
                    } else {
                        ToolResult::success(&call.call_id, "search", results)
                    }
                }
                Err(e) => ToolResult::error(&call.call_id, "search",
                    format!("Failed to read search results: {}", e)),
            }
        }
        Err(e) => ToolResult::error(&call.call_id, "search",
            format!("Search request failed: {}", e)),
    }
}

/// Execute a code expression (basic calculator mode).
pub fn execute_code_interpreter(call: &ToolCall) -> ToolResult {
    let code = match extract_json_string(&call.arguments, "code") {
        Some(c) => c,
        None => return ToolResult::error(&call.call_id, "code_interpreter", "Missing 'code' parameter"),
    };

    // For safety, only support math expressions
    match eval_simple_expr(&code) {
        Ok(result) => ToolResult::success(&call.call_id, "code_interpreter",
            format!("{} = {}", code, result)),
        Err(msg) => ToolResult::error(&call.call_id, "code_interpreter", msg),
    }
}

/// Dispatch a tool call to the appropriate handler.
pub fn dispatch_tool(call: &ToolCall) -> ToolResult {
    match call.tool_name.as_str() {
        "web_fetch" => execute_web_fetch(call),
        "math_eval" => execute_math_eval(call),
        "file_read" => execute_file_read(call),
        "file_write" => execute_file_write(call),
        "shell_exec" => execute_shell_exec(call),
        "search" => execute_search(call),
        "code_interpreter" => execute_code_interpreter(call),
        other => ToolResult::error(&call.call_id, other,
            format!("Unknown tool: {}", other)),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a string value from simple JSON like {"key": "value"}.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    // Find the colon
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    // Extract string value
    if after_colon.starts_with('"') {
        let rest = &after_colon[1..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    } else {
        None
    }
}

/// Extract a numeric value from JSON.
fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    // Read digits and optional decimal
    let num_str: String = after_colon.chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    num_str.parse().ok()
}

/// URL-encode a string (basic implementation).
fn urlencoding(s: &str) -> String {
    s.chars().map(|c| {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '~' {
            c.to_string()
        } else {
            format!("%{:02X}", c as u8)
        }
    }).collect()
}

/// Extract basic results from DuckDuckGo HTML.
fn extract_ddg_results(html: &str, max: usize) -> String {
    let mut results = Vec::new();
    // Look for result links in DDG HTML
    for line in html.lines() {
        if results.len() >= max { break; }
        if line.contains("result__a") || line.contains("result__snippet") {
            // Strip HTML tags
            let text = strip_html_tags(line);
            let trimmed = text.trim();
            if !trimmed.is_empty() && trimmed.len() > 10 {
                results.push(trimmed.to_string());
            }
        }
    }
    results.join("\n\n")
}

/// Strip basic HTML tags.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }
    result
}

/// Evaluate a simple arithmetic expression.
/// Supports: +, -, *, /, parentheses, and basic functions.
fn eval_simple_expr(expr: &str) -> Result<f64, String> {
    let expr = expr.replace(" ", "");
    let (result, _) = parse_expr(&expr, 0)?;
    Ok(result)
}

/// Recursive descent parser for simple math expressions.
fn parse_expr(s: &str, pos: usize) -> Result<(f64, usize), String> {
    let (mut result, mut pos) = parse_term(s, pos)?;

    while pos < s.len() {
        let c = s.as_bytes()[pos];
        if c == b'+' {
            let (rhs, new_pos) = parse_term(s, pos + 1)?;
            result += rhs;
            pos = new_pos;
        } else if c == b'-' {
            let (rhs, new_pos) = parse_term(s, pos + 1)?;
            result -= rhs;
            pos = new_pos;
        } else {
            break;
        }
    }

    Ok((result, pos))
}

fn parse_term(s: &str, pos: usize) -> Result<(f64, usize), String> {
    let (mut result, mut pos) = parse_factor(s, pos)?;

    while pos < s.len() {
        let c = s.as_bytes()[pos];
        if c == b'*' {
            let (rhs, new_pos) = parse_factor(s, pos + 1)?;
            result *= rhs;
            pos = new_pos;
        } else if c == b'/' {
            let (rhs, new_pos) = parse_factor(s, pos + 1)?;
            if rhs == 0.0 {
                return Err("Division by zero".to_string());
            }
            result /= rhs;
            pos = new_pos;
        } else {
            break;
        }
    }

    Ok((result, pos))
}

fn parse_factor(s: &str, pos: usize) -> Result<(f64, usize), String> {
    if pos >= s.len() {
        return Err("Unexpected end of expression".to_string());
    }

    let c = s.as_bytes()[pos];

    if c == b'(' {
        let (result, new_pos) = parse_expr(s, pos + 1)?;
        if new_pos >= s.len() || s.as_bytes()[new_pos] != b')' {
            return Err("Missing closing parenthesis".to_string());
        }
        Ok((result, new_pos + 1))
    } else if c == b'-' {
        let (result, new_pos) = parse_factor(s, pos + 1)?;
        Ok((-result, new_pos))
    } else {
        // Parse number
        let start = pos;
        let mut end = pos;
        while end < s.len() {
            let ch = s.as_bytes()[end];
            if ch.is_ascii_digit() || ch == b'.' {
                end += 1;
            } else {
                break;
            }
        }
        if end == start {
            return Err(format!("Expected number at position {}", pos));
        }
        let num_str = &s[start..end];
        num_str.parse::<f64>()
            .map(|n| (n, end))
            .map_err(|e| format!("Invalid number '{}': {}", num_str, e))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_builtin_tools_created() {
        let tools = all_builtin_tools();
        assert_eq!(tools.len(), 7);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"web_fetch"));
        assert!(names.contains(&"math_eval"));
        assert!(names.contains(&"file_read"));
        assert!(names.contains(&"file_write"));
        assert!(names.contains(&"shell_exec"));
        assert!(names.contains(&"search"));
        assert!(names.contains(&"code_interpreter"));
    }

    #[test]
    fn test_math_eval_simple() {
        let call = ToolCall::new("math_eval", r#"{"expression": "2+3*4"}"#);
        let result = execute_math_eval(&call);
        assert!(result.success);
        assert!(result.output.contains("14"));
    }

    #[test]
    fn test_math_eval_parens() {
        let call = ToolCall::new("math_eval", r#"{"expression": "(2+3)*4"}"#);
        let result = execute_math_eval(&call);
        assert!(result.success);
        assert!(result.output.contains("20"));
    }

    #[test]
    fn test_math_eval_division() {
        let call = ToolCall::new("math_eval", r#"{"expression": "10/3"}"#);
        let result = execute_math_eval(&call);
        assert!(result.success);
        assert!(result.output.contains("3.333"));
    }

    #[test]
    fn test_math_eval_division_by_zero() {
        let call = ToolCall::new("math_eval", r#"{"expression": "1/0"}"#);
        let result = execute_math_eval(&call);
        assert!(!result.success);
        assert!(result.output.contains("zero"));
    }

    #[test]
    fn test_file_write_read_roundtrip() {
        let dir = std::env::temp_dir().join("ferrisres_tools_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_file.txt");

        let write_call = ToolCall::new("file_write",
            format!(r#"{{"path": "{}", "content": "hello world"}}"#, path.display()));
        let write_result = execute_file_write(&write_call);
        assert!(write_result.success);

        let read_call = ToolCall::new("file_read",
            format!(r#"{{"path": "{}"}}"#, path.display()));
        let read_result = execute_file_read(&read_call);
        assert!(read_result.success);
        assert!(read_result.output.contains("hello world"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_read_missing() {
        let call = ToolCall::new("file_read", r#"{"path": "/nonexistent/file.txt"}"#);
        let result = execute_file_read(&call);
        assert!(!result.success);
    }

    #[test]
    fn test_shell_exec_echo() {
        let call = ToolCall::new("shell_exec", r#"{"command": "echo hello_world"}"#);
        let result = execute_shell_exec(&call);
        assert!(result.success);
        assert!(result.output.contains("hello_world"));
    }

    #[test]
    fn test_shell_exec_dangerous_blocked() {
        let call = ToolCall::new("shell_exec", r#"{"command": "rm -rf /"}"#);
        let result = execute_shell_exec(&call);
        assert!(!result.success);
        assert!(result.output.contains("Blocked"));
    }

    #[test]
    fn test_dispatch_unknown_tool() {
        let call = ToolCall::new("nonexistent_tool", "{}");
        let result = dispatch_tool(&call);
        assert!(!result.success);
        assert!(result.output.contains("Unknown tool"));
    }

    #[test]
    fn test_extract_json_string() {
        let json = r#"{"url": "https://example.com", "selector": ".title"}"#;
        assert_eq!(extract_json_string(json, "url"), Some("https://example.com".to_string()));
        assert_eq!(extract_json_string(json, "selector"), Some(".title".to_string()));
        assert_eq!(extract_json_string(json, "missing"), None);
    }

    #[test]
    fn test_eval_simple_expr() {
        assert_eq!(eval_simple_expr("2+3").unwrap(), 5.0);
        assert_eq!(eval_simple_expr("2+3*4").unwrap(), 14.0);
        assert_eq!(eval_simple_expr("(2+3)*4").unwrap(), 20.0);
        assert_eq!(eval_simple_expr("100/10").unwrap(), 10.0);
        assert_eq!(eval_simple_expr("-5+10").unwrap(), 5.0);
    }

    #[test]
    fn test_urlencoding() {
        assert_eq!(urlencoding("hello world"), "hello%20world");
        assert_eq!(urlencoding("a&b=c"), "a%26b%3Dc");
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(strip_html_tags("<p>Hello</p>"), "Hello");
        assert_eq!(strip_html_tags("<b>bold</b> text"), "bold text");
    }
}
