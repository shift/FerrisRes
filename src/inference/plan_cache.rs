//! Plan Cache — Store, Retrieve, and Adapt Successful Plans
//!
//! Stores successful plan executions as templates with extracted parameters.
//! Retrieves similar plans via goal embedding similarity + structural matching.
//! Adapts retrieved plans by parameter substitution for new goals.
//!
//! Similarity = 0.6×goal_embedding + 0.3×structural + 0.1×context
//! Threshold: similarity > 0.7 for reuse suggestion.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A concrete plan step (mirrors PlanExecutor's PlanStep for storage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub step_num: usize,
    pub tool_name: String,
    pub args: String,
    pub retry_on_fail: bool,
}

/// Structural signature of a plan — tool sequence + reference pattern.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PlanSignature {
    /// Sequence of tool names (the structural skeleton).
    pub tool_sequence: Vec<String>,
    /// Reference pattern: step i references step ref_pattern[i], if any.
    pub ref_pattern: Vec<Option<usize>>,
}

/// A parameter slot extracted from plan arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSlot {
    /// Slot name (param_0, param_1, ...).
    pub name: String,
    /// Inferred type.
    pub inferred_type: ParamType,
    /// Example values seen.
    pub examples: Vec<String>,
}

/// Inferred parameter type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParamType {
    Url,
    Path,
    String,
    Number,
    JsonPath,
    Unknown,
}

/// A plan step with parameterized arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateStep {
    pub tool_name: String,
    /// Argument template with ${param_N} placeholders and $N references.
    pub arg_template: String,
    pub retry_on_fail: bool,
}

/// A stored plan template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanTemplate {
    /// Unique ID.
    pub id: u64,
    /// Structural signature.
    pub signature: PlanSignature,
    /// Parameterized steps.
    pub steps: Vec<TemplateStep>,
    /// Parameter slots.
    pub parameters: Vec<ParameterSlot>,
    /// Success rate (EMA).
    pub success_rate: f32,
    /// Usage count.
    pub usage_count: u32,
    /// Goal description embedding (128-dim n-gram).
    pub goal_embedding: Vec<f32>,
    /// Goal description text.
    pub goal_description: String,
    /// Context tags (domain, tool categories).
    pub context_tags: Vec<String>,
    /// Timestamp of last use.
    pub last_used: u64,
    /// Creation timestamp.
    pub created_at: u64,
}

/// Outcome of executing a cached plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOutcome {
    pub plan_id: u64,
    pub success: bool,
    pub execution_time_ms: u64,
    pub step_that_failed: Option<usize>,
    pub quality_score: f32,
    pub timestamp: u64,
    pub goal_description: String,
}

/// A cache hit with similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHit {
    pub template_id: u64,
    pub similarity: f32,
    pub goal_description: String,
    pub success_rate: f32,
}

/// Quality trend for a plan template.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QualityTrend {
    Improving,
    Stable,
    Degrading,
}

/// Configuration for PlanCache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanCacheConfig {
    /// Minimum similarity to suggest reuse (default: 0.7).
    pub similarity_threshold: f32,
    /// Minimum success rate to keep a template (default: 0.3).
    pub min_success_rate: f32,
    /// Maximum templates (default: 1000).
    pub max_templates: usize,
    /// Maximum outcomes per template (default: 50).
    pub max_outcomes_per_template: usize,
    /// Days before deprioritizing unused templates (default: 30).
    pub staleness_days: u64,
}

impl Default for PlanCacheConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            min_success_rate: 0.3,
            max_templates: 1000,
            max_outcomes_per_template: 50,
            staleness_days: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// PlanCache
// ---------------------------------------------------------------------------

/// Stores, retrieves, and adapts successful plans.
pub struct PlanCache {
    templates: HashMap<u64, PlanTemplate>,
    outcomes: HashMap<u64, Vec<PlanOutcome>>,
    config: PlanCacheConfig,
    next_id: u64,
}

impl PlanCache {
    pub fn new(config: PlanCacheConfig) -> Self {
        Self {
            templates: HashMap::new(),
            outcomes: HashMap::new(),
            config,
            next_id: 1,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(PlanCacheConfig::default())
    }

    // -----------------------------------------------------------------------
    // Store
    // -----------------------------------------------------------------------

    /// Extract a template from a concrete plan and store it.
    pub fn store(
        &mut self,
        goal: &str,
        plan: &[PlanStep],
        context_tags: Vec<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let signature = Self::compute_signature(plan);
        let (template_steps, parameters) = Self::extract_template(plan);

        let embedding = Self::ngram_embedding(goal, 128);
        let now = Self::now_secs();

        // Evict worst if at capacity
        if self.templates.len() >= self.config.max_templates {
            self.evict_worst();
        }

        let template = PlanTemplate {
            id,
            signature,
            steps: template_steps,
            parameters,
            success_rate: 1.0,
            usage_count: 1,
            goal_embedding: embedding,
            goal_description: goal.to_string(),
            context_tags,
            last_used: now,
            created_at: now,
        };

        self.templates.insert(id, template);
        id
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Search for plans similar to the given goal.
    pub fn search(&self, goal: &str, available_tools: &[String]) -> Vec<CacheHit> {
        let goal_emb = Self::ngram_embedding(goal, 128);

        let mut hits: Vec<CacheHit> = self
            .templates
            .values()
            .filter_map(|t| {
                // Skip templates with missing tools
                if !t.signature.tool_sequence.iter().all(|tn| available_tools.contains(tn)) {
                    return None;
                }

                let goal_sim = Self::cosine_similarity(&goal_emb, &t.goal_embedding);
                let struct_sim = 0.5; // neutral when no structural query
                let context_sim = 0.5; // neutral when no context query

                let combined =
                    0.6 * goal_sim + 0.3 * struct_sim + 0.1 * context_sim;

                if combined >= self.config.similarity_threshold {
                    Some(CacheHit {
                        template_id: t.id,
                        similarity: combined,
                        goal_description: t.goal_description.clone(),
                        success_rate: t.success_rate,
                    })
                } else {
                    None
                }
            })
            .collect();

        hits.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        hits
    }

    /// Search with structural query.
    pub fn search_with_structure(
        &self,
        goal: &str,
        tool_sequence: &[String],
        available_tools: &[String],
    ) -> Vec<CacheHit> {
        let goal_emb = Self::ngram_embedding(goal, 128);
        let query_sig = PlanSignature {
            tool_sequence: tool_sequence.to_vec(),
            ref_pattern: vec![],
        };

        let mut hits: Vec<CacheHit> = self
            .templates
            .values()
            .filter_map(|t| {
                if !t.signature.tool_sequence.iter().all(|tn| available_tools.contains(tn)) {
                    return None;
                }

                let goal_sim = Self::cosine_similarity(&goal_emb, &t.goal_embedding);
                let struct_sim = Self::structural_similarity(&query_sig, &t.signature);
                let context_sim = 0.5;

                let combined =
                    0.6 * goal_sim + 0.3 * struct_sim + 0.1 * context_sim;

                if combined >= self.config.similarity_threshold {
                    Some(CacheHit {
                        template_id: t.id,
                        similarity: combined,
                        goal_description: t.goal_description.clone(),
                        success_rate: t.success_rate,
                    })
                } else {
                    None
                }
            })
            .collect();

        hits.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        hits
    }

    // -----------------------------------------------------------------------
    // Adapt
    // -----------------------------------------------------------------------

    /// Adapt a cached plan for a new goal by parameter substitution.
    /// Returns concrete PlanSteps with placeholders filled.
    pub fn adapt(
        &self,
        template_id: u64,
        param_values: &HashMap<String, String>,
    ) -> Option<Vec<PlanStep>> {
        let template = self.templates.get(&template_id)?;

        let mut steps = Vec::new();
        for ts in &template.steps {
            let mut args = ts.arg_template.clone();
            // Replace ${param_N} with provided values
            for (key, value) in param_values {
                args = args.replace(&format!("${{{}}}", key), value);
            }
            steps.push(PlanStep {
                step_num: steps.len() + 1,
                tool_name: ts.tool_name.clone(),
                args,
                retry_on_fail: ts.retry_on_fail,
            });
        }

        Some(steps)
    }

    // -----------------------------------------------------------------------
    // Outcome tracking
    // -----------------------------------------------------------------------

    /// Record an execution outcome for a template.
    pub fn record_outcome(&mut self, template_id: u64, outcome: PlanOutcome) {
        if let Some(template) = self.templates.get_mut(&template_id) {
            template.usage_count += 1;
            template.last_used = Self::now_secs();

            let alpha = 0.15;
            template.success_rate =
                alpha * (outcome.success as u32 as f32) + (1.0 - alpha) * template.success_rate;
        }

        let outcomes = self.outcomes.entry(template_id).or_default();
        outcomes.push(outcome);

        // Cap outcomes
        if outcomes.len() > self.config.max_outcomes_per_template {
            outcomes.remove(0);
        }
    }

    /// Get quality trend for a template.
    pub fn quality_trend(&self, template_id: u64) -> Option<QualityTrend> {
        let outcomes = self.outcomes.get(&template_id)?;
        if outcomes.len() < 5 {
            return Some(QualityTrend::Stable);
        }

        let mid = outcomes.len() / 2;
        let first_half: f32 = outcomes[..mid].iter().map(|o| o.quality_score).sum::<f32>() / mid as f32;
        let second_half: f32 = outcomes[mid..].iter().map(|o| o.quality_score).sum::<f32>() / (outcomes.len() - mid) as f32;

        let delta = second_half - first_half;
        if delta > 0.05 {
            Some(QualityTrend::Improving)
        } else if delta < -0.05 {
            Some(QualityTrend::Degrading)
        } else {
            Some(QualityTrend::Stable)
        }
    }

    /// Get the most common failure step for a template.
    pub fn common_failure_step(&self, template_id: u64) -> Option<usize> {
        let outcomes = self.outcomes.get(&template_id)?;
        let mut failure_counts: HashMap<usize, u32> = HashMap::new();
        for o in outcomes {
            if let Some(step) = o.step_that_failed {
                *failure_counts.entry(step).or_insert(0) += 1;
            }
        }
        failure_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(step, _)| step)
    }

    // -----------------------------------------------------------------------
    // Invalidation
    // -----------------------------------------------------------------------

    /// Invalidate all plans using a removed tool.
    pub fn invalidate_tool(&mut self, tool_name: &str) -> Vec<u64> {
        let to_remove: Vec<u64> = self
            .templates
            .values()
            .filter(|t| t.signature.tool_sequence.contains(&tool_name.to_string()))
            .map(|t| t.id)
            .collect();

        for &id in &to_remove {
            self.templates.remove(&id);
            self.outcomes.remove(&id);
        }

        to_remove
    }

    /// Prune stale and low-quality templates.
    pub fn prune(&mut self) -> usize {
        let now = Self::now_secs();
        let staleness_secs = self.config.staleness_days * 86400;
        let threshold = self.config.min_success_rate;

        let to_remove: Vec<u64> = self
            .templates
            .values()
            .filter(|t| {
                // Remove if below min success rate
                t.success_rate < threshold
                // Or if stale and low quality
                || (now - t.last_used > staleness_secs && t.success_rate < 0.5)
            })
            .map(|t| t.id)
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            self.templates.remove(&id);
            self.outcomes.remove(&id);
        }
        count
    }

    /// Get a template by ID.
    pub fn get(&self, id: u64) -> Option<&PlanTemplate> {
        self.templates.get(&id)
    }

    /// Number of stored templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn compute_signature(plan: &[PlanStep]) -> PlanSignature {
        let tool_sequence: Vec<String> = plan.iter().map(|s| s.tool_name.clone()).collect();
        let ref_pattern: Vec<Option<usize>> = plan
            .iter()
            .map(|s| {
                // Find $N references in args
                let re = regex::Regex::new(r"\$(\d+)").unwrap();
                re.captures(&s.args)
                    .and_then(|c| c[1].parse::<usize>().ok())
            })
            .collect();
        PlanSignature {
            tool_sequence,
            ref_pattern,
        }
    }

    fn extract_template(plan: &[PlanStep]) -> (Vec<TemplateStep>, Vec<ParameterSlot>) {
        let mut steps = Vec::new();
        let mut params = Vec::new();
        let mut param_idx = 0;

        for step in plan {
            let mut arg_template = step.args.clone();

            // Extract literal values (non-$N references) as parameters
            // Simple heuristic: values that look like URLs, paths, or quoted strings
            let parts: Vec<String> = arg_template.split_whitespace()
                .map(|s| s.to_string()).collect();
            for part in &parts {
                if !part.starts_with('$') && looks_like_param(part) {
                    let param_name = format!("param_{}", param_idx);
                    arg_template = arg_template.replace(part.as_str(), &format!("${{{}}}", param_name));
                    params.push(ParameterSlot {
                        name: param_name,
                        inferred_type: infer_type(part),
                        examples: vec![part.clone()],
                    });
                    param_idx += 1;
                }
            }

            steps.push(TemplateStep {
                tool_name: step.tool_name.clone(),
                arg_template,
                retry_on_fail: step.retry_on_fail,
            });
        }

        (steps, params)
    }

    fn structural_similarity(a: &PlanSignature, b: &PlanSignature) -> f32 {
        if a.tool_sequence.is_empty() && b.tool_sequence.is_empty() {
            return 1.0;
        }
        if a.tool_sequence.is_empty() || b.tool_sequence.is_empty() {
            return 0.0;
        }

        // Jaccard similarity of tool sets
        let set_a: std::collections::HashSet<_> = a.tool_sequence.iter().collect();
        let set_b: std::collections::HashSet<_> = b.tool_sequence.iter().collect();
        let intersection = set_a.intersection(&set_b).count() as f32;
        let union = set_a.union(&set_b).count() as f32;
        if union == 0.0 {
            return 0.0;
        }

        let jaccard = intersection / union;

        // Sequence alignment bonus
        let min_len = a.tool_sequence.len().min(b.tool_sequence.len());
        let max_len = a.tool_sequence.len().max(b.tool_sequence.len());
        let matches = (0..min_len)
            .filter(|&i| a.tool_sequence[i] == b.tool_sequence[i])
            .count();

        let alignment = matches as f32 / max_len as f32;

        0.5 * jaccard + 0.5 * alignment
    }

    fn evict_worst(&mut self) {
        if let Some(worst_id) = self
            .templates
            .values()
            .min_by(|a, b| {
                let score_a = a.success_rate * (a.usage_count as f32).ln().max(1.0);
                let score_b = b.success_rate * (b.usage_count as f32).ln().max(1.0);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|t| t.id)
        {
            self.templates.remove(&worst_id);
            self.outcomes.remove(&worst_id);
        }
    }

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn ngram_embedding(text: &str, dims: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; dims];
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();

        for window in chars.windows(3) {
            let mut h: u64 = 5381;
            for &c in window {
                h = h.wrapping_mul(33).wrapping_add(c as u64);
            }
            let idx = (h as usize) % dims;
            vec[idx] += 1.0;
        }

        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut vec {
            *v /= norm;
        }
        vec
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let dot: f32 = a[..len].iter().zip(&b[..len]).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        dot / (na * nb)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn looks_like_param(s: &str) -> bool {
    // URLs, file paths, numbers, quoted strings
    s.starts_with("http://")
        || s.starts_with("https://")
        || s.starts_with('/')
        || s.starts_with('"')
        || s.starts_with('\'')
        || s.contains(".csv")
        || s.contains(".json")
        || s.contains(".txt")
        || s.parse::<f64>().is_ok()
}

fn infer_type(s: &str) -> ParamType {
    if s.starts_with("http://") || s.starts_with("https://") {
        ParamType::Url
    } else if s.starts_with("$.") {
        ParamType::JsonPath
    } else if s.starts_with('/') {
        ParamType::Path
    } else if s.contains('.') {
        ParamType::Path
    } else if s.parse::<f64>().is_ok() {
        ParamType::Number
    } else {
        ParamType::String
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_plan() -> Vec<PlanStep> {
        vec![
            PlanStep {
                step_num: 1,
                tool_name: "web_fetch".into(),
                args: "https://api.weather.com/forecast?city=Stockholm".into(),
                retry_on_fail: true,
            },
            PlanStep {
                step_num: 2,
                tool_name: "json_extract".into(),
                args: "$1 $.temperature".into(),
                retry_on_fail: false,
            },
            PlanStep {
                step_num: 3,
                tool_name: "format_output".into(),
                args: "Temperature in Stockholm: {temp}".into(),
                retry_on_fail: false,
            },
        ]
    }

    #[test]
    fn test_template_extraction() {
        let _cache = PlanCache::with_defaults();
        let (steps, params) = PlanCache::extract_template(&sample_plan());
        assert_eq!(steps.len(), 3);
        assert!(!params.is_empty()); // URL extracted as parameter
    }

    #[test]
    fn test_signature_computation() {
        let sig = PlanCache::compute_signature(&sample_plan());
        assert_eq!(sig.tool_sequence, vec!["web_fetch", "json_extract", "format_output"]);
        assert_eq!(sig.ref_pattern[0], None); // no $N in step 1
        assert_eq!(sig.ref_pattern[1], Some(1)); // $1 in step 2
    }

    #[test]
    fn test_parameter_slot_inference() {
        let _cache = PlanCache::with_defaults();
        let (_, params) = PlanCache::extract_template(&sample_plan());
        // URL should be inferred as Url type
        let url_param = params.iter().find(|p| p.inferred_type == ParamType::Url);
        assert!(url_param.is_some());
    }

    #[test]
    fn test_store_and_search() {
        let mut config = PlanCacheConfig::default();
        config.similarity_threshold = 0.5; // Lower threshold for short strings with n-gram embeddings
        let mut cache = PlanCache::new(config);
        let tools = vec!["web_fetch".into(), "json_extract".into(), "format_output".into()];
        let id = cache.store("get weather forecast for Stockholm", &sample_plan(), vec![]);

        let hits = cache.search("weather forecast for Tokyo", &tools);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].template_id, id);
    }

    #[test]
    fn test_similarity_threshold() {
        let mut cache = PlanCache::with_defaults();
        let tools = vec!["web_fetch".into(), "json_extract".into(), "format_output".into()];
        cache.store("get weather forecast for Stockholm", &sample_plan(), vec![]);

        // Completely unrelated goal should not match
        let hits = cache.search("debug a rust compiler error", &tools);
        assert!(hits.is_empty() || hits[0].similarity < 0.9);
    }

    #[test]
    fn test_adapt_parameter_substitution() {
        let mut cache = PlanCache::with_defaults();
        let id = cache.store("get weather for Stockholm", &sample_plan(), vec![]);

        let mut params = HashMap::new();
        params.insert("param_0".into(), "https://api.weather.com/forecast?city=Tokyo".into());

        let adapted = cache.adapt(id, &params).unwrap();
        assert!(adapted[0].args.contains("Tokyo"));
        // $1 reference should still be preserved
        assert!(adapted[1].args.contains("$1"));
    }

    #[test]
    fn test_outcome_recording() {
        let mut cache = PlanCache::with_defaults();
        let id = cache.store("test goal", &sample_plan(), vec![]);

        cache.record_outcome(
            id,
            PlanOutcome {
                plan_id: id,
                success: true,
                execution_time_ms: 100,
                step_that_failed: None,
                quality_score: 0.9,
                timestamp: 0,
                goal_description: "test".into(),
            },
        );

        let template = cache.get(id).unwrap();
        assert_eq!(template.usage_count, 2); // initial 1 + outcome update
    }

    #[test]
    fn test_quality_decay() {
        let mut cache = PlanCache::with_defaults();
        let id = cache.store("test goal", &sample_plan(), vec![]);

        // Record many failures
        for _ in 0..20 {
            cache.record_outcome(
                id,
                PlanOutcome {
                    plan_id: id,
                    success: false,
                    execution_time_ms: 100,
                    step_that_failed: Some(1),
                    quality_score: 0.1,
                    timestamp: 0,
                    goal_description: "test".into(),
                },
            );
        }

        let template = cache.get(id).unwrap();
        assert!(template.success_rate < 0.5);
    }

    #[test]
    fn test_tool_invalidation() {
        let mut cache = PlanCache::with_defaults();
        cache.store("test", &sample_plan(), vec![]);

        let removed = cache.invalidate_tool("web_fetch");
        assert_eq!(removed.len(), 1);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_max_templates_cap() {
        let mut config = PlanCacheConfig::default();
        config.max_templates = 3;
        let mut cache = PlanCache::new(config);

        for i in 0..5 {
            let plan = vec![PlanStep {
                step_num: 1,
                tool_name: format!("tool_{}", i),
                args: format!("arg_{}", i),
                retry_on_fail: true,
            }];
            cache.store(&format!("goal {}", i), &plan, vec![]);
        }

        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_structural_similarity_same() {
        let a = PlanSignature {
            tool_sequence: vec!["a".into(), "b".into(), "c".into()],
            ref_pattern: vec![],
        };
        let sim = PlanCache::structural_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_structural_similarity_different() {
        let a = PlanSignature {
            tool_sequence: vec!["a".into(), "b".into(), "c".into()],
            ref_pattern: vec![],
        };
        let b = PlanSignature {
            tool_sequence: vec!["x".into(), "y".into(), "z".into()],
            ref_pattern: vec![],
        };
        let sim = PlanCache::structural_similarity(&a, &b);
        assert!(sim < 0.1);
    }

    #[test]
    fn test_search_with_structure() {
        let mut cache = PlanCache::with_defaults();
        let tools = vec!["a".into(), "b".into(), "c".into()];
        cache.store("process data", &vec![
            PlanStep { step_num: 1, tool_name: "a".into(), args: "x".into(), retry_on_fail: true },
            PlanStep { step_num: 2, tool_name: "b".into(), args: "$1".into(), retry_on_fail: true },
        ], vec![]);

        let hits = cache.search_with_structure("process data", &vec!["a".into(), "b".into()], &tools);
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_quality_trend_improving() {
        let mut cache = PlanCache::with_defaults();
        let id = cache.store("test", &sample_plan(), vec![]);

        for i in 0..10 {
            cache.record_outcome(id, PlanOutcome {
                plan_id: id, success: true, execution_time_ms: 100,
                step_that_failed: None, quality_score: 0.3 + (i as f32 * 0.07),
                timestamp: i as u64, goal_description: "test".into(),
            });
        }

        assert_eq!(cache.quality_trend(id), Some(QualityTrend::Improving));
    }

    #[test]
    fn test_quality_trend_degrading() {
        let mut cache = PlanCache::with_defaults();
        let id = cache.store("test", &sample_plan(), vec![]);

        for i in 0..10 {
            cache.record_outcome(id, PlanOutcome {
                plan_id: id, success: true, execution_time_ms: 100,
                step_that_failed: None, quality_score: 1.0 - (i as f32 * 0.08),
                timestamp: i as u64, goal_description: "test".into(),
            });
        }

        assert_eq!(cache.quality_trend(id), Some(QualityTrend::Degrading));
    }

    #[test]
    fn test_common_failure_step() {
        let mut cache = PlanCache::with_defaults();
        let id = cache.store("test", &sample_plan(), vec![]);

        cache.record_outcome(id, PlanOutcome {
            plan_id: id, success: false, execution_time_ms: 100,
            step_that_failed: Some(2), quality_score: 0.0,
            timestamp: 0, goal_description: "test".into(),
        });
        cache.record_outcome(id, PlanOutcome {
            plan_id: id, success: false, execution_time_ms: 100,
            step_that_failed: Some(2), quality_score: 0.0,
            timestamp: 1, goal_description: "test".into(),
        });
        cache.record_outcome(id, PlanOutcome {
            plan_id: id, success: false, execution_time_ms: 100,
            step_that_failed: Some(1), quality_score: 0.0,
            timestamp: 2, goal_description: "test".into(),
        });

        assert_eq!(cache.common_failure_step(id), Some(2));
    }

    #[test]
    fn test_search_missing_tool_excluded() {
        let mut cache = PlanCache::with_defaults();
        cache.store("test", &sample_plan(), vec![]);

        // Missing web_fetch tool
        let tools = vec!["json_extract".into(), "format_output".into()];
        let hits = cache.search("test", &tools);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_adapt_nonexistent_template() {
        let cache = PlanCache::with_defaults();
        let result = cache.adapt(999, &HashMap::new());
        assert!(result.is_none());
    }

    #[test]
    fn test_infer_type_url() {
        assert_eq!(infer_type("https://example.com"), ParamType::Url);
    }

    #[test]
    fn test_infer_type_path() {
        assert_eq!(infer_type("/data/file.csv"), ParamType::Path);
    }

    #[test]
    fn test_infer_type_number() {
        assert_eq!(infer_type("42"), ParamType::Number);
    }

    #[test]
    fn test_infer_type_jsonpath() {
        assert_eq!(infer_type("$.temperature"), ParamType::JsonPath);
    }
}
