//! Tool Search Registry for agentic workflows.
//!
//! Dynamically discovers and loads relevant tools (3-8) from a larger library
//! (50-100+) using embedding similarity search. Cuts prompt costs by ~50%
//! compared to including all tools in every prompt.
//!
//! Based on research task 1c6f7edf: Tool Search (2026 commercial approach).

use std::collections::HashMap;

/// A tool that can be invoked by the model.
#[derive(Debug, Clone)]
pub struct Tool {
    /// Unique tool identifier.
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// JSON schema for the tool's parameters.
    pub parameters_schema: String,
    /// Semantic embedding of the tool's description (for search).
    pub embedding: Option<Vec<f32>>,
    /// Tool category for filtering.
    pub category: String,
    /// Example usage (few-shot prompt fragment).
    pub example_usage: Option<String>,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters_schema: String::new(),
            embedding: None,
            category: "general".to_string(),
            example_usage: None,
        }
    }

    pub fn with_parameters(mut self, schema: impl Into<String>) -> Self {
        self.parameters_schema = schema.into();
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }

    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example_usage = Some(example.into());
        self
    }
}

/// A tool that has been selected for inclusion in a prompt.
#[derive(Debug, Clone)]
pub struct SelectedTool {
    pub tool: Tool,
    pub relevance_score: f32,
    pub rank: usize,
}

/// Configuration for tool search.
#[derive(Debug, Clone)]
pub struct ToolSearchConfig {
    /// Maximum number of tools to include per query.
    pub max_tools: usize,
    /// Minimum relevance score to include a tool.
    pub min_relevance: f32,
    /// Whether to include example usage in the formatted output.
    pub include_examples: bool,
    /// Category filter: only search within these categories (empty = all).
    pub category_filter: Vec<String>,
}

impl Default for ToolSearchConfig {
    fn default() -> Self {
        Self {
            max_tools: 6,
            min_relevance: 0.3,
            include_examples: true,
            category_filter: Vec::new(),
        }
    }
}

/// Tool calling result.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: String,
    pub call_id: String,
}

impl ToolCall {
    pub fn new(tool_name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments: arguments.into(),
            call_id: format!("call_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()),
        }
    }
}

/// Tool execution result.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub call_id: String,
    pub tool_name: String,
    pub output: String,
    pub success: bool,
}

impl ToolResult {
    pub fn success(call_id: impl Into<String>, tool_name: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            output: output.into(),
            success: true,
        }
    }

    pub fn error(call_id: impl Into<String>, tool_name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            output: error.into(),
            success: false,
        }
    }
}

/// Keyword-indexed tool search (fallback when embeddings are unavailable).
struct KeywordIndex {
    /// term → set of tool names that contain the term in their description.
    index: HashMap<String, Vec<String>>,
}

impl KeywordIndex {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    fn add_tool(&mut self, name: &str, description: &str) {
        let terms = Self::tokenize(description);
        for term in terms {
            self.index
                .entry(term)
                .or_default()
                .push(name.to_string());
        }
    }

    fn search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let query_terms = Self::tokenize(query);
        let mut scores: HashMap<String, f32> = HashMap::new();

        for term in &query_terms {
            if let Some(tool_names) = self.index.get(term) {
                for name in tool_names {
                    *scores.entry(name.clone()).or_insert(0.0) += 1.0;
                }
            }
        }

        // Normalize by query length
        for (_, score) in scores.iter_mut() {
            *score /= query_terms.len() as f32;
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(String::from)
            .collect()
    }
}

/// Tool search registry with embedding-based and keyword-based search.
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
    keyword_index: KeywordIndex,
    config: ToolSearchConfig,
}

impl ToolRegistry {
    /// Create a new tool registry.
    pub fn new(config: ToolSearchConfig) -> Self {
        Self {
            tools: HashMap::new(),
            keyword_index: KeywordIndex::new(),
            config,
        }
    }

    /// Create with default config.
    pub fn default_registry() -> Self {
        Self::new(ToolSearchConfig::default())
    }

    /// Register a tool.
    pub fn register(&mut self, tool: Tool) {
        self.keyword_index.add_tool(&tool.name, &tool.description);
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Register multiple tools.
    pub fn register_all(&mut self, tools: Vec<Tool>) {
        for tool in tools {
            self.register(tool);
        }
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    /// Get total number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Search for relevant tools using embedding similarity.
    pub fn search_by_embedding(&self, query_embedding: &[f32], top_k: usize) -> Vec<SelectedTool> {
        let mut results: Vec<SelectedTool> = self.tools.values()
            .filter_map(|tool| {
                // Apply category filter
                if !self.config.category_filter.is_empty()
                    && !self.config.category_filter.contains(&tool.category) {
                    return None;
                }

                tool.embedding.as_ref().map(|emb| {
                    let score = cosine_similarity(query_embedding, emb);
                    SelectedTool {
                        tool: tool.clone(),
                        relevance_score: score,
                        rank: 0,
                    }
                })
            })
            .filter(|s| s.relevance_score >= self.config.min_relevance)
            .collect();

        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k.min(self.config.max_tools));

        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i + 1;
        }

        results
    }

    /// Search for relevant tools using keyword matching.
    pub fn search_by_keywords(&self, query: &str, top_k: usize) -> Vec<SelectedTool> {
        let keyword_results = self.keyword_index.search(query, top_k * 2);

        let mut results: Vec<SelectedTool> = keyword_results.into_iter()
            .filter_map(|(name, score)| {
                self.tools.get(&name).map(|tool| {
                    // Apply category filter
                    if !self.config.category_filter.is_empty()
                        && !self.config.category_filter.contains(&tool.category) {
                        return None;
                    }
                    Some(SelectedTool {
                        tool: tool.clone(),
                        relevance_score: score,
                        rank: 0,
                    })
                })
            })
            .flatten()
            .filter(|s| s.relevance_score >= self.config.min_relevance)
            .collect();

        results.truncate(top_k.min(self.config.max_tools));

        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i + 1;
        }

        results
    }

    /// Hybrid search: combine embedding and keyword search.
    pub fn search(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
    ) -> Vec<SelectedTool> {
        let top_k = self.config.max_tools;

        let mut combined: HashMap<String, (f32, Tool)> = HashMap::new();

        // Keyword search
        let keyword_results = self.search_by_keywords(query, top_k);
        let max_kw = keyword_results.iter().map(|r| r.relevance_score).fold(0.001, f32::max);
        for r in keyword_results {
            let normalized = r.relevance_score / max_kw;
            let entry = combined.entry(r.tool.name.clone()).or_insert_with(|| (0.0, r.tool.clone()));
            entry.0 += normalized * 0.3;
        }

        // Embedding search
        if let Some(emb) = query_embedding {
            let emb_results = self.search_by_embedding(emb, top_k);
            let max_emb = emb_results.iter().map(|r| r.relevance_score).fold(0.001, f32::max);
            for r in emb_results {
                let normalized = r.relevance_score / max_emb;
                let entry = combined.entry(r.tool.name.clone()).or_insert_with(|| (0.0, r.tool.clone()));
                entry.0 += normalized * 0.7;
            }
        }

        let mut results: Vec<SelectedTool> = combined.into_iter()
            .map(|(_, (score, tool))| SelectedTool {
                tool,
                relevance_score: score,
                rank: 0,
            })
            .filter(|s| s.relevance_score >= self.config.min_relevance)
            .collect();

        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i + 1;
        }

        results
    }

    /// Format selected tools as a system prompt fragment.
    pub fn format_tools_prompt(&self, tools: &[SelectedTool]) -> String {
        let mut prompt = String::from("Available tools:\n\n");

        for selected in tools {
            prompt.push_str(&format!(
                "## {} (relevance: {:.2})\n{}\n",
                selected.tool.name, selected.relevance_score, selected.tool.description
            ));

            if !selected.tool.parameters_schema.is_empty() {
                prompt.push_str(&format!("Parameters: {}\n", selected.tool.parameters_schema));
            }

            if self.config.include_examples {
                if let Some(ref example) = selected.tool.example_usage {
                    prompt.push_str(&format!("Example: {}\n", example));
                }
            }

            prompt.push('\n');
        }

        prompt.push_str("To call a tool, respond with: TOOL_CALL:<tool_name>(<json_arguments>)\n");
        prompt
    }

    /// Parse a tool call from model output.
    pub fn parse_tool_call(&self, output: &str) -> Option<ToolCall> {
        // Try TOOL_CALL:name(args) format
        if let Some(start) = output.find("TOOL_CALL:") {
            let rest = &output[start + "TOOL_CALL:".len()..];
            if let Some(paren_pos) = rest.find('(') {
                let name = rest[..paren_pos].trim().to_string();
                // Find matching closing paren
                let args_start = paren_pos + 1;
                if let Some(end) = rest[args_start..].find(')') {
                    let args = rest[args_start..args_start + end].trim().to_string();
                    if self.tools.contains_key(&name) {
                        return Some(ToolCall::new(name, args));
                    }
                }
            }
        }
        None
    }

    /// Format a tool result for inclusion in the conversation.
    pub fn format_tool_result(&self, result: &ToolResult) -> String {
        if result.success {
            format!("TOOL_RESULT:{}:{}:{}", result.call_id, result.tool_name, result.output)
        } else {
            format!("TOOL_ERROR:{}:{}:{}", result.call_id, result.tool_name, result.output)
        }
    }

    /// List all registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// List all categories.
    pub fn categories(&self) -> Vec<&str> {
        let mut cats: Vec<&str> = self.tools.values()
            .map(|t| t.category.as_str())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }
}

/// Cosine similarity (shared with RAG module, duplicated here to avoid coupling).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(name: &str, desc: &str, emb: Vec<f32>) -> Tool {
        Tool::new(name, desc).with_embedding(emb)
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = ToolRegistry::default_registry();
        registry.register(Tool::new("calculator", "Performs arithmetic calculations"));
        assert!(registry.get("calculator").is_some());
        assert!(registry.get("nonexistent").is_none());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_keyword_search() {
        let mut registry = ToolRegistry::default_registry();
        registry.register(Tool::new("web_search", "Search the web for information"));
        registry.register(Tool::new("calculator", "Perform math calculations"));
        registry.register(Tool::new("file_read", "Read files from disk"));

        let results = registry.search_by_keywords("search the internet", 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].tool.name, "web_search");
    }

    #[test]
    fn test_embedding_search() {
        let mut registry = ToolRegistry::default_registry();
        registry.register(make_tool("weather", "Get weather forecasts", vec![1.0, 0.0]));
        registry.register(make_tool("calculator", "Math operations", vec![0.0, 1.0]));
        registry.register(make_tool("news", "Get weather news", vec![0.9, 0.1]));

        let query = vec![1.0, 0.0]; // weather-like
        let results = registry.search_by_embedding(&query, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tool.name, "weather");
    }

    #[test]
    fn test_hybrid_search() {
        let mut registry = ToolRegistry::default_registry();
        registry.register(make_tool("weather", "Get weather forecasts", vec![1.0, 0.0]));
        registry.register(make_tool("calculator", "Math operations", vec![0.0, 1.0]));

        let query_emb = vec![1.0, 0.0];
        let results = registry.search("weather today", Some(&query_emb));
        assert!(!results.is_empty());
        assert_eq!(results[0].tool.name, "weather");
    }

    #[test]
    fn test_category_filter() {
        let config = ToolSearchConfig {
            category_filter: vec!["math".to_string()],
            ..Default::default()
        };
        let mut registry = ToolRegistry::new(config);
        registry.register(Tool::new("calc", "Calculator").with_category("math"));
        registry.register(Tool::new("search", "Web search").with_category("web"));

        let results = registry.search_by_keywords("search calculate", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool.name, "calc");
    }

    #[test]
    fn test_max_tools_limit() {
        let config = ToolSearchConfig {
            max_tools: 2,
            ..Default::default()
        };
        let mut registry = ToolRegistry::new(config);
        for i in 0..10 {
            registry.register(Tool::new(format!("tool_{}", i), format!("Tool number {}", i)));
        }

        let results = registry.search_by_keywords("tool", 10);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_format_tools_prompt() {
        let registry = ToolRegistry::default_registry();
        let tools = vec![
            SelectedTool {
                tool: Tool::new("calc", "Calculator").with_parameters("{\"expr\": \"string\"}"),
                relevance_score: 0.95,
                rank: 1,
            },
        ];
        let prompt = registry.format_tools_prompt(&tools);
        assert!(prompt.contains("calc"));
        assert!(prompt.contains("Calculator"));
        assert!(prompt.contains("TOOL_CALL"));
    }

    #[test]
    fn test_parse_tool_call() {
        let mut registry = ToolRegistry::default_registry();
        registry.register(Tool::new("calculator", "Math"));

        let output = "I'll calculate that. TOOL_CALL:calculator({"a": 1, "b": 2})";
        let call = registry.parse_tool_call(output).unwrap();
        assert_eq!(call.tool_name, "calculator");
        assert!(call.arguments.contains("\"a\""));
    }

    #[test]
    fn test_parse_tool_call_unknown() {
        let registry = ToolRegistry::default_registry();
        let output = "TOOL_CALL:unknown_tool({})";
        assert!(registry.parse_tool_call(output).is_none());
    }

    #[test]
    fn test_format_tool_result() {
        let registry = ToolRegistry::default_registry();
        let result = ToolResult::success("call_123", "calculator", "42");
        let formatted = registry.format_tool_result(&result);
        assert!(formatted.contains("TOOL_RESULT:"));
        assert!(formatted.contains("42"));

        let error = ToolResult::error("call_123", "calculator", "division by zero");
        let formatted = registry.format_tool_result(&error);
        assert!(formatted.contains("TOOL_ERROR:"));
    }

    #[test]
    fn test_tool_categories() {
        let mut registry = ToolRegistry::default_registry();
        registry.register(Tool::new("calc", "Math").with_category("math"));
        registry.register(Tool::new("search", "Web").with_category("web"));
        registry.register(Tool::new("plot", "Charts").with_category("math"));

        let cats = registry.categories();
        assert!(cats.contains(&"math"));
        assert!(cats.contains(&"web"));
        assert_eq!(cats.len(), 2);
    }

    #[test]
    fn test_tool_with_example() {
        let tool = Tool::new("calc", "Math")
            .with_example("TOOL_CALL:calc({\"expr\": \"2+2\"}) → 4");
        assert!(tool.example_usage.is_some());
    }
}
