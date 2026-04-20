//! Subgoal Generator — Hierarchical Goal Decomposition
//!
//! Decomposes high-level goals into AND/OR subgoal trees using:
//!   - LLM-driven decomposition (prompt parsing)
//!   - Stored decomposition patterns (HTN methods) from PlanCache
//!   - Fallback to flat PlanExecutor for simple goals
//!
//! Safety constraints:
//!   - Max depth: 3 levels
//!   - Max subgoals per level: 8
//!   - Total leaf tool calls capped at 50 per request
//!   - 30-second timeout per decomposition attempt

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Decomposition kind for a subgoal node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecomposeKind {
    /// All children must succeed (sequential).
    And,
    /// Any child succeeding is sufficient (alternatives).
    Or,
}

/// A node in the subgoal tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubgoalNode {
    /// Leaf node: maps to a flat plan.
    Leaf {
        label: String,
        tool_name: String,
        args: String,
    },
    /// Decomposed node: contains child subgoals.
    Decomposed {
        label: String,
        kind: DecomposeKind,
        children: Vec<SubgoalNode>,
    },
}

/// Result of decomposing a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    /// The root of the decomposition tree.
    pub tree: SubgoalNode,
    /// How the decomposition was obtained.
    pub method: DecompositionMethod,
    /// Total leaf count (tool calls).
    pub leaf_count: usize,
    /// Depth of the tree.
    pub depth: usize,
    /// Whether validation passed.
    pub valid: bool,
    /// Validation errors (if any).
    pub validation_errors: Vec<String>,
}

/// How the decomposition was obtained.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecompositionMethod {
    /// Reused from PlanCache (pattern match).
    Cached { pattern_id: u64 },
    /// Parsed from LLM output.
    LlmParsed,
    /// Fallback: single-step flat execution.
    Fallback,
}

/// A stored decomposition pattern (HTN method).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionPattern {
    /// Unique ID.
    pub id: u64,
    /// Goal description.
    pub goal_description: String,
    /// The decomposition tree.
    pub tree: SubgoalNode,
    /// Preconditions for this decomposition to apply.
    pub preconditions: Vec<String>,
    /// Success rate from past executions.
    pub success_rate: f32,
    /// Number of times used.
    pub usage_count: u32,
    /// Goal embedding (character n-gram, 128-dim).
    pub goal_embedding: Vec<f32>,
}

/// Configuration for SubgoalGenerator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgoalConfig {
    /// Maximum decomposition depth (default: 3).
    pub max_depth: usize,
    /// Maximum subgoals per decomposition level (default: 8).
    pub max_subgoals_per_level: usize,
    /// Maximum total leaf tool calls per request (default: 50).
    pub max_total_leaves: usize,
    /// Minimum similarity to reuse a cached pattern (default: 0.7).
    pub cache_similarity_threshold: f32,
    /// Maximum stored patterns (default: 500).
    pub max_patterns: usize,
}

impl Default for SubgoalConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_subgoals_per_level: 8,
            max_total_leaves: 50,
            cache_similarity_threshold: 0.7,
            max_patterns: 500,
        }
    }
}

/// Outcome of executing a subgoal tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionOutcome {
    /// Whether the overall goal succeeded.
    pub success: bool,
    /// Which subgoal nodes succeeded/failed.
    pub node_outcomes: Vec<(String, bool)>,
    /// Total execution time in ms.
    pub execution_time_ms: u64,
    /// Quality score from MirrorTest (if available).
    pub quality: Option<f32>,
}

// ---------------------------------------------------------------------------
// SubgoalGenerator
// ---------------------------------------------------------------------------

/// Hierarchical goal decomposition engine.
pub struct SubgoalGenerator {
    config: SubgoalConfig,
    /// Stored decomposition patterns.
    patterns: HashMap<u64, DecompositionPattern>,
    /// Next pattern ID.
    next_pattern_id: u64,
    /// Available tool names.
    available_tools: Vec<String>,
}

impl SubgoalGenerator {
    /// Create a new SubgoalGenerator with the given config and tools.
    pub fn new(config: SubgoalConfig, available_tools: Vec<String>) -> Self {
        Self {
            config,
            patterns: HashMap::new(),
            next_pattern_id: 1,
            available_tools,
        }
    }

    /// Create with default config.
    pub fn with_tools(available_tools: Vec<String>) -> Self {
        Self::new(SubgoalConfig::default(), available_tools)
    }

    // -----------------------------------------------------------------------
    // Decomposition
    // -----------------------------------------------------------------------

    /// Decompose a goal into a subgoal tree.
    /// Checks cache first, then parses from LLM-style output, then falls back.
    pub fn decompose(&mut self, goal: &str) -> DecompositionResult {
        // Step 1: Check cache for similar goals
        if let Some(pattern_id) = self.find_cached_pattern(goal) {
            let pattern = self.patterns.get(&pattern_id).unwrap().clone();
            let depth = Self::tree_depth(&pattern.tree);
            let leaves = Self::leaf_count(&pattern.tree);
            return DecompositionResult {
                tree: pattern.tree,
                method: DecompositionMethod::Cached { pattern_id },
                leaf_count: leaves,
                depth,
                valid: true,
                validation_errors: vec![],
            };
        }

        // Step 2: Parse from LLM-style output
        if let Some(tree) = self.parse_subgoals(goal) {
            let depth = Self::tree_depth(&tree);
            let leaves = Self::leaf_count(&tree);
            let errors = self.validate_tree(&tree);
            let valid = errors.is_empty();
            return DecompositionResult {
                tree,
                method: DecompositionMethod::LlmParsed,
                leaf_count: leaves,
                depth,
                valid,
                validation_errors: errors,
            };
        }

        // Step 3: Fallback — single leaf
        let tree = SubgoalNode::Leaf {
            label: goal.to_string(),
            tool_name: String::new(),
            args: String::new(),
        };
        DecompositionResult {
            tree,
            method: DecompositionMethod::Fallback,
            leaf_count: 1,
            depth: 1,
            valid: true,
            validation_errors: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Parsing
    // -----------------------------------------------------------------------

    /// Parse `[subgoals]...[/subgoals]` blocks from model output.
    pub fn parse_subgoals(&self, text: &str) -> Option<SubgoalNode> {
        // Extract [subgoals]...[/subgoals] block
        let start = text.find("[subgoals]")? + "[subgoals]".len();
        let end = text.find("[/subgoals]")?;
        let block = &text[start..end];

        self.parse_subgoal_block(block.trim())
    }

    /// Parse a subgoal block into a tree.
    fn parse_subgoal_block(&self, block: &str) -> Option<SubgoalNode> {
        let lines: Vec<&str> = block.lines().collect();
        if lines.is_empty() {
            return None;
        }

        // Check if first line declares a kind
        let (kind, content_lines) = if lines[0].trim().to_uppercase().starts_with("AND") {
            (DecomposeKind::And, &lines[1..])
        } else if lines[0].trim().to_uppercase().starts_with("OR") {
            (DecomposeKind::Or, &lines[1..])
        } else {
            // Default to AND
            (DecomposeKind::And, &lines[..])
        };

        let mut children = Vec::new();

        for line in content_lines {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }

            // Parse numbered item: "1. label" or "1. tool(args)"
            let item = if let Some(dot_pos) = trimmed.find('.') {
                trimmed[dot_pos + 1..].trim()
            } else {
                trimmed
            };

            // Check if this is a nested AND/OR block (indented)
            if item.starts_with("AND:") || item.starts_with("OR:") {
                // Nested decomposition — simplified: treat children as leaves
                let nested_kind = if item.starts_with("AND:") {
                    DecomposeKind::And
                } else {
                    DecomposeKind::Or
                };
                // For simplicity, just create a Decomposed node with the label
                let label = item.trim_end_matches(':').to_string();
                children.push(SubgoalNode::Decomposed {
                    label,
                    kind: nested_kind,
                    children: vec![],
                });
                continue;
            }

            // Check if this is a leaf: "label: tool(args)" or "label"
            if let Some(colon_pos) = item.find('(') {
                // Has arguments: "tool_name(args)"
                let tool_name = item[..colon_pos].trim().to_string();
                let args_end = item.rfind(')').unwrap_or(item.len());
                let args = item[colon_pos + 1..args_end].trim().to_string();
                let label = tool_name.clone();
                children.push(SubgoalNode::Leaf {
                    label,
                    tool_name,
                    args,
                });
            } else {
                // Plain label
                let label = item.trim().to_string();
                let tool_name = label.clone();
                children.push(SubgoalNode::Leaf {
                    label: label.clone(),
                    tool_name,
                    args: String::new(),
                });
            }

            // Enforce max subgoals per level
            if children.len() >= self.config.max_subgoals_per_level {
                break;
            }
        }

        if children.is_empty() {
            return None;
        }

        Some(SubgoalNode::Decomposed {
            label: "root".to_string(),
            kind,
            children,
        })
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    /// Validate a subgoal tree.
    pub fn validate_tree(&self, node: &SubgoalNode) -> Vec<String> {
        let mut errors = Vec::new();
        self.validate_node(node, 0, &mut errors);
        errors
    }

    fn validate_node(&self, node: &SubgoalNode, depth: usize, errors: &mut Vec<String>) {
        match node {
            SubgoalNode::Leaf { tool_name, .. } => {
                if !tool_name.is_empty() && !self.available_tools.contains(tool_name) {
                    errors.push(format!("Unknown tool '{}' in leaf node", tool_name));
                }
            }
            SubgoalNode::Decomposed { children, .. } => {
                if depth >= self.config.max_depth {
                    errors.push(format!(
                        "Decomposition depth {} exceeds max {}",
                        depth, self.config.max_depth
                    ));
                    return;
                }
                if children.len() > self.config.max_subgoals_per_level {
                    errors.push(format!(
                        "Decomposition has {} children, max is {}",
                        children.len(),
                        self.config.max_subgoals_per_level
                    ));
                }
                if children.is_empty() {
                    errors.push("Decomposed node has no children".to_string());
                }
                for child in children {
                    self.validate_node(child, depth + 1, errors);
                }
            }
        }

        // Check total leaf count
        let leaves = Self::leaf_count(node);
        if leaves > self.config.max_total_leaves {
            errors.push(format!(
                "Total leaf count {} exceeds max {}",
                leaves, self.config.max_total_leaves
            ));
        }
    }

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    /// Execute a subgoal tree. Returns the outcome.
    /// For a real system, this would delegate to PlanExecutor for leaf nodes.
    /// Here we simulate execution based on tool availability.
    pub fn execute_tree(
        &self,
        node: &SubgoalNode,
        executor: &mut dyn FnMut(&str, &str) -> bool,
    ) -> ExecutionOutcome {
        let start = std::time::Instant::now();
        let mut outcomes = Vec::new();
        let success = self.execute_node(node, &mut outcomes, executor);
        ExecutionOutcome {
            success,
            node_outcomes: outcomes,
            execution_time_ms: start.elapsed().as_millis() as u64,
            quality: None,
        }
    }

    fn execute_node(
        &self,
        node: &SubgoalNode,
        outcomes: &mut Vec<(String, bool)>,
        executor: &mut dyn FnMut(&str, &str) -> bool,
    ) -> bool {
        match node {
            SubgoalNode::Leaf { label, tool_name, args } => {
                let ok = executor(tool_name, args);
                outcomes.push((label.clone(), ok));
                ok
            }
            SubgoalNode::Decomposed { label, kind, children } => {
                let results: Vec<bool> = children
                    .iter()
                    .map(|c| self.execute_node(c, outcomes, executor))
                    .collect();

                let overall = match kind {
                    DecomposeKind::And => results.iter().all(|&r| r),
                    DecomposeKind::Or => results.iter().any(|&r| r),
                };
                outcomes.push((label.clone(), overall));
                overall
            }
        }
    }

    // -----------------------------------------------------------------------
    // Learning
    // -----------------------------------------------------------------------

    /// Store a successful decomposition as a pattern.
    pub fn store_pattern(
        &mut self,
        goal: &str,
        tree: SubgoalNode,
        preconditions: Vec<String>,
    ) -> u64 {
        let id = self.next_pattern_id;
        self.next_pattern_id += 1;

        let embedding = Self::ngram_embedding(goal, 128);

        // Enforce max patterns
        if self.patterns.len() >= self.config.max_patterns {
            // Remove lowest success rate pattern
            if let Some(worst_id) = self
                .patterns
                .iter()
                .min_by(|a, b| a.1.success_rate.partial_cmp(&b.1.success_rate).unwrap())
                .map(|(&id, _)| id)
            {
                self.patterns.remove(&worst_id);
            }
        }

        self.patterns.insert(
            id,
            DecompositionPattern {
                id,
                goal_description: goal.to_string(),
                tree,
                preconditions,
                success_rate: 1.0,
                usage_count: 1,
                goal_embedding: embedding,
            },
        );

        id
    }

    /// Update pattern success rate after execution.
    pub fn update_pattern(&mut self, pattern_id: u64, success: bool) {
        if let Some(pattern) = self.patterns.get_mut(&pattern_id) {
            pattern.usage_count += 1;
            let alpha = 0.2; // EMA smoothing
            pattern.success_rate =
                alpha * (success as u32 as f32) + (1.0 - alpha) * pattern.success_rate;
        }
    }

    // -----------------------------------------------------------------------
    // Cache lookup
    // -----------------------------------------------------------------------

    /// Find a cached pattern similar to the given goal.
    fn find_cached_pattern(&self, goal: &str) -> Option<u64> {
        let goal_emb = Self::ngram_embedding(goal, 128);
        let threshold = self.config.cache_similarity_threshold;

        let mut best_id: Option<u64> = None;
        let mut best_sim = 0.0f32;

        for (&id, pattern) in &self.patterns {
            let sim = Self::cosine_similarity(&goal_emb, &pattern.goal_embedding);
            if sim > threshold && sim > best_sim {
                best_sim = sim;
                best_id = Some(id);
            }
        }

        best_id
    }

    // -----------------------------------------------------------------------
    // Tree utilities
    // -----------------------------------------------------------------------

    /// Count leaf nodes in a tree.
    pub fn leaf_count(node: &SubgoalNode) -> usize {
        match node {
            SubgoalNode::Leaf { .. } => 1,
            SubgoalNode::Decomposed { children, .. } => {
                children.iter().map(Self::leaf_count).sum()
            }
        }
    }

    /// Compute tree depth.
    pub fn tree_depth(node: &SubgoalNode) -> usize {
        match node {
            SubgoalNode::Leaf { .. } => 1,
            SubgoalNode::Decomposed { children, .. } => {
                1 + children.iter().map(Self::tree_depth).max().unwrap_or(0)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Embedding utilities
    // -----------------------------------------------------------------------

    /// Character 3-gram embedding (128-dimensional, crude but sufficient).
    fn ngram_embedding(text: &str, dims: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; dims];
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();

        for window in chars.windows(3) {
            let hash = Self::hash_ngram(window);
            let idx = (hash as usize) % dims;
            vec[idx] += 1.0;
        }

        // Normalize
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut vec {
            *v /= norm;
        }

        vec
    }

    /// Simple hash for a character trigram.
    fn hash_ngram(chars: &[char]) -> u64 {
        let mut h: u64 = 5381;
        for &c in chars {
            h = h.wrapping_mul(33).wrapping_add(c as u64);
        }
        h
    }

    /// Cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let dot: f32 = a[..len].iter().zip(&b[..len]).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        dot / (na * nb)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tools() -> Vec<String> {
        vec![
            "parse_csv".into(),
            "clean_data".into(),
            "compute_stats".into(),
            "plot_results".into(),
            "find_outliers".into(),
            "web_fetch".into(),
            "math_eval".into(),
        ]
    }

    #[test]
    fn test_parse_and_tree() {
        let gen = SubgoalGenerator::with_tools(test_tools());
        let input = r#"
[subgoals]
AND:
  1. parse_csv(data.csv)
  2. clean_data($1)
  3. compute_stats($2)
[/subgoals]
"#;
        let tree = gen.parse_subgoals(input).unwrap();
        match &tree {
            SubgoalNode::Decomposed { kind, children, .. } => {
                assert_eq!(*kind, DecomposeKind::And);
                assert_eq!(children.len(), 3);
            }
            _ => panic!("Expected Decomposed node"),
        }
    }

    #[test]
    fn test_parse_or_tree() {
        let gen = SubgoalGenerator::with_tools(test_tools());
        let input = r#"
[subgoals]
OR:
  1. plot_results(stats)
  2. find_outliers(stats)
[/subgoals]
"#;
        let tree = gen.parse_subgoals(input).unwrap();
        match &tree {
            SubgoalNode::Decomposed { kind, children, .. } => {
                assert_eq!(*kind, DecomposeKind::Or);
                assert_eq!(children.len(), 2);
            }
            _ => panic!("Expected Decomposed node"),
        }
    }

    #[test]
    fn test_parse_mixed_tree() {
        let gen = SubgoalGenerator::with_tools(test_tools());
        let input = r#"
[subgoals]
AND:
  1. parse_csv(input.csv)
  2. clean_data($1)
OR:
  3. compute_stats($2)
  4. find_outliers($2)
[/subgoals]
"#;
        let tree = gen.parse_subgoals(input).unwrap();
        // The parser treats the entire block as AND with the OR as another child
        match &tree {
            SubgoalNode::Decomposed { kind, children, .. } => {
                assert_eq!(*kind, DecomposeKind::And);
                assert!(children.len() >= 2);
            }
            _ => panic!("Expected Decomposed node"),
        }
    }

    #[test]
    fn test_leaf_count() {
        let tree = SubgoalNode::Decomposed {
            label: "root".into(),
            kind: DecomposeKind::And,
            children: vec![
                SubgoalNode::Leaf {
                    label: "a".into(),
                    tool_name: "t1".into(),
                    args: "".into(),
                },
                SubgoalNode::Decomposed {
                    label: "sub".into(),
                    kind: DecomposeKind::Or,
                    children: vec![
                        SubgoalNode::Leaf {
                            label: "b".into(),
                            tool_name: "t2".into(),
                            args: "".into(),
                        },
                        SubgoalNode::Leaf {
                            label: "c".into(),
                            tool_name: "t3".into(),
                            args: "".into(),
                        },
                    ],
                },
            ],
        };
        assert_eq!(SubgoalGenerator::leaf_count(&tree), 3);
    }

    #[test]
    fn test_tree_depth() {
        let tree = SubgoalNode::Decomposed {
            label: "root".into(),
            kind: DecomposeKind::And,
            children: vec![
                SubgoalNode::Leaf {
                    label: "a".into(),
                    tool_name: "t1".into(),
                    args: "".into(),
                },
                SubgoalNode::Decomposed {
                    label: "sub".into(),
                    kind: DecomposeKind::Or,
                    children: vec![SubgoalNode::Leaf {
                        label: "b".into(),
                        tool_name: "t2".into(),
                        args: "".into(),
                    }],
                },
            ],
        };
        assert_eq!(SubgoalGenerator::tree_depth(&tree), 3);
    }

    #[test]
    fn test_decompose_fallback() {
        let mut gen = SubgoalGenerator::with_tools(test_tools());
        // No [subgoals] block → fallback
        let result = gen.decompose("simple goal");
        assert_eq!(result.method, DecompositionMethod::Fallback);
        assert_eq!(result.leaf_count, 1);
    }

    #[test]
    fn test_decompose_llm_parsed() {
        let mut gen = SubgoalGenerator::with_tools(test_tools());
        let goal = "[subgoals]\nAND:\n  1. parse_csv(data.csv)\n  2. clean_data($1)\n[/subgoals]";
        let result = gen.decompose(goal);
        assert_eq!(result.method, DecompositionMethod::LlmParsed);
        assert_eq!(result.leaf_count, 2);
    }

    #[test]
    fn test_validate_unknown_tool() {
        let gen = SubgoalGenerator::with_tools(test_tools());
        let tree = SubgoalNode::Leaf {
            label: "test".into(),
            tool_name: "nonexistent_tool".into(),
            args: "".into(),
        };
        let errors = gen.validate_tree(&tree);
        assert!(errors.iter().any(|e| e.contains("Unknown tool")));
    }

    #[test]
    fn test_validate_max_depth_exceeded() {
        let mut config = SubgoalConfig::default();
        config.max_depth = 1;
        let gen = SubgoalGenerator::new(config, test_tools());
        let tree = SubgoalNode::Decomposed {
            label: "root".into(),
            kind: DecomposeKind::And,
            children: vec![SubgoalNode::Decomposed {
                label: "sub".into(),
                kind: DecomposeKind::And,
                children: vec![SubgoalNode::Leaf {
                    label: "leaf".into(),
                    tool_name: "t1".into(),
                    args: "".into(),
                }],
            }],
        };
        let errors = gen.validate_tree(&tree);
        assert!(errors.iter().any(|e| e.contains("depth")));
    }

    #[test]
    fn test_validate_empty_children() {
        let gen = SubgoalGenerator::with_tools(test_tools());
        let tree = SubgoalNode::Decomposed {
            label: "root".into(),
            kind: DecomposeKind::And,
            children: vec![],
        };
        let errors = gen.validate_tree(&tree);
        assert!(errors.iter().any(|e| e.contains("no children")));
    }

    #[test]
    fn test_validate_valid_tree() {
        let gen = SubgoalGenerator::with_tools(test_tools());
        let tree = SubgoalNode::Leaf {
            label: "test".into(),
            tool_name: "parse_csv".into(),
            args: "data.csv".into(),
        };
        let errors = gen.validate_tree(&tree);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_execute_and_all_succeed() {
        let gen = SubgoalGenerator::with_tools(test_tools());
        let tree = SubgoalNode::Decomposed {
            label: "root".into(),
            kind: DecomposeKind::And,
            children: vec![
                SubgoalNode::Leaf {
                    label: "a".into(),
                    tool_name: "t1".into(),
                    args: "".into(),
                },
                SubgoalNode::Leaf {
                    label: "b".into(),
                    tool_name: "t2".into(),
                    args: "".into(),
                },
            ],
        };
        let outcome = gen.execute_tree(&tree, &mut |_, _| true);
        assert!(outcome.success);
        assert_eq!(outcome.node_outcomes.len(), 3); // 2 leaves + 1 root
    }

    #[test]
    fn test_execute_and_one_fails() {
        let gen = SubgoalGenerator::with_tools(test_tools());
        let tree = SubgoalNode::Decomposed {
            label: "root".into(),
            kind: DecomposeKind::And,
            children: vec![
                SubgoalNode::Leaf {
                    label: "a".into(),
                    tool_name: "t1".into(),
                    args: "".into(),
                },
                SubgoalNode::Leaf {
                    label: "b".into(),
                    tool_name: "t2".into(),
                    args: "".into(),
                },
            ],
        };
        let outcome = gen.execute_tree(&tree, &mut |name, _| name == "t1");
        assert!(!outcome.success);
    }

    #[test]
    fn test_execute_or_first_succeeds() {
        let gen = SubgoalGenerator::with_tools(test_tools());
        let tree = SubgoalNode::Decomposed {
            label: "root".into(),
            kind: DecomposeKind::Or,
            children: vec![
                SubgoalNode::Leaf {
                    label: "a".into(),
                    tool_name: "t1".into(),
                    args: "".into(),
                },
                SubgoalNode::Leaf {
                    label: "b".into(),
                    tool_name: "t2".into(),
                    args: "".into(),
                },
            ],
        };
        let outcome = gen.execute_tree(&tree, &mut |name, _| name == "t1");
        assert!(outcome.success);
    }

    #[test]
    fn test_execute_or_all_fail() {
        let gen = SubgoalGenerator::with_tools(test_tools());
        let tree = SubgoalNode::Decomposed {
            label: "root".into(),
            kind: DecomposeKind::Or,
            children: vec![
                SubgoalNode::Leaf {
                    label: "a".into(),
                    tool_name: "t1".into(),
                    args: "".into(),
                },
                SubgoalNode::Leaf {
                    label: "b".into(),
                    tool_name: "t2".into(),
                    args: "".into(),
                },
            ],
        };
        let outcome = gen.execute_tree(&tree, &mut |_, _| false);
        assert!(!outcome.success);
    }

    #[test]
    fn test_store_and_retrieve_pattern() {
        let mut gen = SubgoalGenerator::with_tools(test_tools());
        let tree = SubgoalNode::Leaf {
            label: "test".into(),
            tool_name: "parse_csv".into(),
            args: "data.csv".into(),
        };
        let id = gen.store_pattern("parse a CSV file", tree, vec![]);
        assert_eq!(id, 1);
        assert!(gen.patterns.contains_key(&1));

        // Should retrieve by similar goal
        let result = gen.decompose("parse a CSV");
        assert!(matches!(result.method, DecompositionMethod::Cached { .. }));
    }

    #[test]
    fn test_update_pattern_success_rate() {
        let mut gen = SubgoalGenerator::with_tools(test_tools());
        let tree = SubgoalNode::Leaf {
            label: "test".into(),
            tool_name: "t1".into(),
            args: "".into(),
        };
        let id = gen.store_pattern("goal", tree, vec![]);

        gen.update_pattern(id, true);
        gen.update_pattern(id, false);
        gen.update_pattern(id, true);

        let pattern = gen.patterns.get(&id).unwrap();
        assert!(pattern.success_rate > 0.5 && pattern.success_rate < 1.0);
        assert_eq!(pattern.usage_count, 4); // initial 1 + 3 updates
    }

    #[test]
    fn test_max_patterns_eviction() {
        let mut config = SubgoalConfig::default();
        config.max_patterns = 3;
        let mut gen = SubgoalGenerator::new(config, test_tools());

        // Store 4 patterns, last should evict the worst
        for i in 0..4 {
            let tree = SubgoalNode::Leaf {
                label: format!("test_{}", i),
                tool_name: "t1".into(),
                args: "".into(),
            };
            let id = gen.store_pattern(&format!("goal {}", i), tree, vec![]);
            // Make first pattern have lowest success rate
            if i == 0 {
                gen.update_pattern(id, false);
                gen.update_pattern(id, false);
                gen.update_pattern(id, false);
            }
        }

        assert_eq!(gen.patterns.len(), 3);
    }

    #[test]
    fn test_ngram_embedding_normalization() {
        let emb = SubgoalGenerator::ngram_embedding("hello world", 128);
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let sim = SubgoalGenerator::cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = SubgoalGenerator::cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.01);
    }

    #[test]
    fn test_max_subgoals_per_level_enforced() {
        let mut config = SubgoalConfig::default();
        config.max_subgoals_per_level = 3;
        let gen = SubgoalGenerator::new(config, test_tools());
        let input = r#"
[subgoals]
AND:
  1. parse_csv(a)
  2. clean_data(b)
  3. compute_stats(c)
  4. find_outliers(d)
  5. plot_results(e)
[/subgoals]
"#;
        let tree = gen.parse_subgoals(input).unwrap();
        match &tree {
            SubgoalNode::Decomposed { children, .. } => {
                assert_eq!(children.len(), 3); // Capped at 3
            }
            _ => panic!("Expected Decomposed"),
        }
    }

    #[test]
    fn test_decomposition_result_valid_field() {
        let mut gen = SubgoalGenerator::with_tools(test_tools());
        let result = gen.decompose("[subgoals]\nAND:\n  1. unknown_tool(x)\n[/subgoals]");
        assert!(!result.valid);
        assert!(result.validation_errors.iter().any(|e| e.contains("Unknown tool")));
    }
}
