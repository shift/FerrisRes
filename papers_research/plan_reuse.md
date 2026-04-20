# Research: Plan Reuse — Storing, Retrieving, and Adapting Successful Plans

## Summary

Research into plan persistence and reuse for FerrisRes. The current PlanExecutor creates flat plans from model output but doesn't store them or search for similar past plans. This research covers plan embedding, similarity metrics, plan adaptation (parameter substitution), cache invalidation, and quality tracking.

## Existing Architecture

Relevant existing modules:
- **PlanExecutor**: Creates and executes flat plans. Steps have `$N` references. No persistence.
- **ConceptMemory**: Stores concepts with embeddings, cosine similarity retrieval, JSON persistence. Could host plan patterns.
- **EpisodicMemory**: Stores episodes with outcomes. Plans could be linked to episodes.
- **ToolUsageTracker**: Per-tool quality tracking. Could extend to per-plan quality.
- **AbstractionEngine**: Clusters concepts into meta-concepts. Could cluster similar plans.

## Core Design Questions

### Q1: How to represent a plan for storage?

A plan is a sequence of `(tool_name, args_template)` where args may contain `$N` references and literal values. For storage, we need to separate the structure from the specific values.

**Approach: Plan Templates**

A plan template extracts the structural skeleton:
```
Original plan:
  Step 1: web_fetch("https://api.weather.com/forecast?city=Stockholm")
  Step 2: json_extract($1, "$.temperature")
  Step 3: format_output("Temperature in Stockholm: {temp}", temp=$2)

Template:
  Step 1: web_fetch(url: ${param_1})
  Step 2: json_extract($1, path: ${param_2})
  Step 3: format_output(template: ${param_3}, temp=$2)

Extracted parameters:
  param_1 = "https://api.weather.com/forecast?city=Stockholm"
  param_2 = "$.temperature"
  param_3 = "Temperature in Stockholm: {temp}"
```

**Template structure**:
```rust
struct PlanTemplate {
    /// Structurally similar plans share this signature.
    signature: PlanSignature,
    /// Steps with parameterized arguments.
    steps: Vec<TemplateStep>,
    /// Parameter slots extracted from args.
    parameters: Vec<ParameterSlot>,
    /// Quality metrics.
    success_rate: f32,
    usage_count: u32,
    /// Embedding of the goal description.
    goal_embedding: Vec<f32>,
}

struct PlanSignature {
    /// Sequence of tool names (the structural skeleton).
    tool_sequence: Vec<String>,
    /// Reference pattern: which steps reference which.
    ref_pattern: Vec<Option<usize>>, // step i references step ref_pattern[i]
}

struct TemplateStep {
    tool_name: String,
    /// Argument template with ${param_N} placeholders and $N references.
    arg_template: String,
    retry_on_fail: bool,
}

struct ParameterSlot {
    name: String,
    /// Inferred type (url, path, string, number).
    inferred_type: ParamType,
    /// Example values seen.
    examples: Vec<String>,
}
```

### Q2: How to measure plan similarity?

Plan similarity has two dimensions:

1. **Structural similarity**: Same tool sequence + reference pattern. Exact match on `PlanSignature`.
2. **Semantic similarity**: Similar goal (embedding cosine > 0.85) → may use different tools but achieve same outcome.

**Scoring**:
```
similarity(query, cached) = 
  0.6 * cosine_similarity(query.goal_embedding, cached.goal_embedding) +
  0.3 * structural_similarity(query.signature, cached.signature) +
  0.1 * context_overlap(query.context_tags, cached.context_tags)
```

Where:
- `cosine_similarity`: goal description embeddings (captures intent)
- `structural_similarity`: Jaccard similarity of tool sets + reference pattern alignment
- `context_overlap`: domain tags, tool categories

**Threshold**: similarity > 0.7 to suggest reuse. Below that, decompose from scratch.

### Q3: How to adapt a retrieved plan?

When a cached plan is retrieved for a new query, the arguments need adaptation:

1. **Parameter extraction from new goal**: Parse the new goal to extract parameters that fill the template slots. E.g., "weather in Tokyo" → `param_1 = "...city=Tokyo"`.
2. **LLM-assisted adaptation**: If automatic extraction fails, ask the model to fill the template:
   ```
   Adapt this plan template for the new goal:
   Template: web_fetch(${param_1}) → json_extract($1, ${param_2}) → format_output(${param_3})
   New goal: "What's the weather in Tokyo?"
   Fill in: param_1, param_2, param_3
   ```
3. **Validation**: After adaptation, validate that all referenced tools exist and all `$N` references are valid.

**Adaptation safety**: If adapted plan fails on first execution, fall back to fresh plan generation.

### Q4: When to invalidate cached plans?

Cache invalidation triggers:
1. **Tool removed**: If a tool in the cached plan is no longer registered → invalidate.
2. **Quality decay**: If success rate drops below 50% over last 10 uses → deprioritize (not delete, may recover).
3. **Environment change**: If domain context shifts significantly (detected by embedding drift) → re-evaluate.
4. **Version change**: Tool API changed → invalidate plans using that tool. Track tool versions.
5. **Time decay**: Plans not used for 30+ days get lower priority in retrieval.

### Q5: How to track plan quality over time?

Each plan execution records an outcome:
```rust
struct PlanOutcome {
    plan_id: u64,
    success: bool,
    execution_time_ms: u64,
    step_that_failed: Option<usize>,
    quality_score: f32, // from MirrorTest or user feedback
    timestamp: u64,
    goal_description: String,
}
```

Aggregated metrics:
- **Success rate**: EMA over last N executions (N=20)
- **Average execution time**: For cost estimation
- **Common failure point**: Which step fails most → target for tool improvement
- **Quality trend**: Improving / stable / degrading

## Recommended Architecture: PlanCache Module

```rust
/// Stores, retrieves, and adapts successful plans.
pub struct PlanCache {
    /// Stored plan templates, indexed by signature hash.
    templates: HashMap<u64, PlanTemplate>,
    /// FAISS-like flat index for goal embedding similarity search.
    embedding_index: Vec<(u64, Vec<f32>)>, // (template_id, embedding)
    /// Execution outcomes per template.
    outcomes: HashMap<u64, Vec<PlanOutcome>>,
    /// Configuration.
    config: PlanCacheConfig,
    /// Persistence path.
    persist_path: Option<PathBuf>,
}

pub struct PlanCacheConfig {
    /// Minimum similarity to suggest plan reuse.
    pub similarity_threshold: f32, // default: 0.7
    /// Minimum success rate to keep a template.
    pub min_success_rate: f32, // default: 0.3
    /// Maximum templates to store.
    pub max_templates: usize, // default: 1000
    /// Maximum outcomes per template.
    pub max_outcomes_per_template: usize, // default: 50
    /// Days before unused template is deprioritized.
    pub staleness_days: u64, // default: 30
}
```

### Key Methods

```rust
impl PlanCache {
    /// Store a successful plan as a template.
    fn store(&mut self, goal: &str, plan: &[PlanStep], outcome: &PlanOutcome) -> Result<u64>;
    
    /// Search for similar plans.
    fn search(&self, goal: &str, available_tools: &[String]) -> Vec<CacheHit>;
    
    /// Adapt a cached plan for a new goal.
    fn adapt(&self, template_id: u64, new_goal: &str) -> Result<Vec<PlanStep>>;
    
    /// Record an execution outcome.
    fn record_outcome(&mut self, template_id: u64, outcome: PlanOutcome);
    
    /// Invalidate plans using a removed tool.
    fn invalidate_tool(&mut self, tool_name: &str);
    
    /// Prune stale/low-quality templates.
    fn prune(&mut self);
    
    /// Persist to disk.
    fn save(&self) -> Result<()>;
    
    /// Load from disk.
    fn load(path: &Path) -> Result<Self>;
}
```

### Integration with SubgoalGenerator

The PlanCache is queried BEFORE decomposition:
```
SubgoalGenerator::decompose(goal)
  1. cache.search(goal) → found? → cache.adapt(id, goal) → return adapted plan
  2. not found → LLM decomposition → execute → if success: cache.store(goal, plan, outcome)
```

### Integration with Existing Modules

- **ConceptMemory**: Plan templates stored as `ConceptContent::Algorithm` with embeddings. PlanCache can delegate storage to ConceptMemory for unified persistence.
- **EpisodicMemory**: Each plan execution is an episode. `PlanOutcome` links to the episode for full context retrieval.
- **ToolUsageTracker**: Plan-level quality feeds into tool-level quality. If a plan's step 2 consistently fails, that tool's quality score reflects it.
- **AbstractionEngine**: Cluster similar plan templates into meta-templates. E.g., 5 "weather query" plans → 1 generalized weather template.
- **ProactiveController**: When a plan's success rate drops, ProactiveController can flag it for investigation.

## Plan Embedding Strategy

Since FerrisRes doesn't have a real embedding model, use a **bag-of-tools + goal hash** approach:

1. **Goal embedding**: Hash the goal description into a fixed-size vector using character n-grams (3-grams, dimension 128). Fast, no model needed.
2. **Structural fingerprint**: Tool sequence as a string, hashed to u64.
3. **Combined**: Concatenate goal embedding (128-dim) + structural one-hot (max 256 tools) = 384-dim vector. Cosine similarity for retrieval.

This is crude but sufficient for plan retrieval. Real embedding models can be plugged in later.

## Test Plan

1. `test_template_extraction` — extract template from concrete plan
2. `test_signature_computation` — same tool sequence → same signature
3. `test_parameter_slot_inference` — URL, path, string types inferred
4. `test_store_and_search` — store plan, find it by similar goal
5. `test_similarity_threshold` — below threshold not returned
6. `test_adapt_parameter_substitution` — fill template with new params
7. `test_adapt_validation` — invalid adapted plan rejected
8. `test_outcome_recording` — success rate tracked
9. `test_quality_decay` — success rate below 0.3 deprioritized
10. `test_tool_invalidation` — remove tool → plans invalidated
11. `test_staleness_pruning` — old unused plans deprioritized
12. `test_max_templates_cap` — evict lowest-quality when full
13. `test_persistence_roundtrip` — save/load preserves all data
14. `test_embedding_similarity` — similar goals retrieve same plans
15. `test_structural_similarity` — same tools different args still match
16. `test_multi_param_adaptation` — adapt plan with 3+ parameters
17. `test_adaptation_fallback` — failed adaptation → fresh plan
18. `test_plan_quality_trend` — track improving/stable/degrading
19. `test_failure_point_tracking` — common failure step identified
20. `test_abstraction_integration` — similar plans clustered

## References

- Schank, R. & Abelson, R. (1977). "Scripts, Plans, Goals and Understanding" — plan as cognitive schema
- Hoffmann, J. (2001). "FF: The Fast-Forward Planning System" — heuristic plan retrieval
- Riesbeck, C. & Schank, R. (1989). "Inside Case-Based Reasoning" — case-based plan reuse
- Bergmann, R. et al. (2005). "Developing Industrial Case-Based Reasoning Applications" — INRECA methodology
- Ontañón, S. et al. (2020). "A Survey of Deep Learning for Case-Based Reasoning" — neural plan retrieval
