# Research: Self-Bootstrapping — Recursive Tool Composition

## Summary

Research into recursive tool composition for FerrisRes. The current ToolCreationPipeline creates single tools, but created tools cannot compose into better tools. This research covers tool composition metrics, bootstrapping safety (preventing divergence), quality tracking per tool version, replace-vs-extend heuristics, and tool versioning with rollback.

## Existing Architecture

- **ToolCreationPipeline**: Model generates `[tool_create]` specs → WasmSandbox validates → MirrorTest evaluates → register in ToolRegistry. Max 3 refinement iterations.
- **ToolRegistry** (in `cognitive_pipeline.rs`): HashMap of registered tools with `execute_tool()` dispatch.
- **PlanExecutor**: Chains tools sequentially with `$N` references. No tool composition.
- **ToolUsageTracker**: Per-tool EMA quality tracking, contextual bandit for best-tool recommendation.
- **WasmSandbox**: Validates tool code. Could be used for composability testing.
- **MirrorTest**: Generates test cases for tool validation. Key for bootstrap quality gates.

## Core Concept: Tool Composition Levels

```
Level 0: Host tools (built-in: web_fetch, math_eval, code_run, ...)
Level 1: Model-created tools (single-purpose: parse_csv, clean_data)
Level 2: Composed tools (Level 1 tools combined: analyze_dataset = parse_csv + clean_data + stats)
Level 3: Meta-tools (tools that create tools: dataset_analyzer_generator)
```

Bootstrapping is the process of composing lower-level tools into higher-level tools, then using those as building blocks for even higher-level tools.

## Key Design Decisions

### D1: How does tool composition work?

**Approach: Macro Tools (Composed Plans as Tools)**

A composed tool is a stored plan template that behaves like a regular tool:

```rust
struct ComposedTool {
    /// Tool name (appears in ToolRegistry).
    name: String,
    /// Description (for retrieval and model prompting).
    description: String,
    /// The plan template that implements this tool.
    plan_template: PlanTemplate,
    /// Input parameter mapping: which plan params come from tool args.
    input_mapping: Vec<(String, String)>, // (tool_param, plan_param)
    /// Output extraction: which step's output is the tool's return value.
    output_step: usize,
    /// Composition level (0=host, 1=model-created, 2=composed, 3=meta).
    level: u32,
    /// Version of this tool.
    version: u32,
}
```

When executed, a ComposedTool:
1. Maps input args → plan parameters
2. Runs the plan through PlanExecutor
3. Returns the output from `output_step`

This makes composed tools indistinguishable from regular tools — they can be used in any plan, including other composed tools.

### D2: What triggers bootstrapping?

Bootstrapping is triggered when the system detects a **repeated pattern**:

1. **Pattern detection**: ToolUsageTracker notices that tools A, B, C are always called in sequence for a certain class of problems. IntrinsicMotivation flags this as a "comprehension gap" (we keep doing the same multi-step process).
2. **Composition proposal**: ProactiveController (at SemiAutonomous or higher) proposes: "Tools A, B, C are called together 15 times with 80% success. Compose into tool ABC?"
3. **Composition creation**: ToolCreationPipeline creates a ComposedTool with the plan template. MirrorTest validates it against past successful executions.
4. **Registration**: If quality > 0.7, register in ToolRegistry. Track as Level 2.

Alternatively, the model can explicitly propose composition:
```
[tool_create]
name: analyze_dataset
description: Parse, clean, and analyze a CSV dataset
composed_from: parse_csv → clean_data → compute_stats
input: csv_path (string)
output: statistics (object)
[/tool_create]
```

### D3: How to prevent divergence?

Divergence = composed tools getting worse over time (bad tools composing into worse tools).

**Safety layers**:

1. **Quality gate**: Every composed tool must pass MirrorTest with quality > 0.7 before registration. This is the same gate as single-tool creation.
2. **Monotonicity check**: A composed tool must be at least as good as the best of its constituent tools. If `quality(composed) < max(quality(parts))`, reject.
3. **Composition budget**: Max 5 new composed tools per session. Prevents runaway composition.
4. **Level limit**: Max composition level = 3. No level-4+ tools. Prevents deep recursion.
5. **Divergence detection**: Track rolling average quality per level. If level-2 quality drops below level-1 quality for 10 consecutive uses, freeze level-2 tool creation and alert.
6. **Sandbox testing**: New composed tools are tested against the last 10 successful executions of the constituent plan. Must match or exceed success rate.
7. **Human gate**: At SemiAutonomous or lower, composed tools require human approval. Only FullyAutonomous can auto-register.

```rust
struct BootstrapSafetyConfig {
    /// Min quality for a composed tool to be registered.
    pub min_quality: f32, // default: 0.7
    /// Max composed tools created per session.
    pub max_compositions_per_session: u32, // default: 5
    /// Max composition level.
    pub max_level: u32, // default: 3
    /// Min quality delta (composed must be >= best constituent - delta).
    pub min_quality_delta: f32, // default: -0.05
    /// Number of past executions to test against.
    pub sandbox_test_count: usize, // default: 10
    /// Autonomy level required for auto-registration.
    pub auto_register_level: AutonomyLevel, // FullyAutonomous
}
```

### D4: Quality tracking per tool version

Tools evolve: `parse_csv v1` → `parse_csv v2` (handles edge cases) → `parse_csv v3` (faster).

```rust
struct ToolVersion {
    tool_name: String,
    version: u32,
    created_at: u64,
    quality: f32,
    success_count: u32,
    failure_count: u32,
    /// What changed from previous version.
    change_description: String,
    /// The composed plan or code for this version.
    implementation: ToolImpl,
}

enum ToolImpl {
    /// Built-in host tool.
    Host(String),
    /// Model-generated WASM code.
    Generated { code: String, spec: ToolSpec },
    /// Composed from other tools.
    Composed(ComposedTool),
}
```

**Version quality tracking**:
- Each version has independent quality tracking (EMA over last 20 uses)
- New versions start with the parent's quality as prior (not from zero)
- If a new version's quality drops below parent by > 0.1 over 5 uses → auto-rollback to parent
- Version comparison: if v2 quality > v1 quality by > 0.05 sustained over 10 uses → v2 becomes default

### D5: Replace vs extend heuristics

When a tool has a bug or limitation:

**Replace** (overwrite existing tool):
- Bug fix (same interface, better implementation)
- Performance improvement (same output, faster)
- Minor edge case handling
- Condition: v2 must pass all v1's test cases

**Extend** (create new tool, keep old):
- New interface (different params or return type)
- Different use case (e.g., parse_csv → parse_tsv)
- Experimental feature (not yet validated)
- Condition: old tool still useful on its own

**Heuristic**:
```
if new_tool.signature == old_tool.signature && new_tool.passes_old_tests:
    replace(old_tool, new_tool)
elif new_tool.use_case != old_tool.use_case:
    register(new_tool)  // separate tool, old stays
else:
    extend(old_tool, new_tool)  // new version, old preserved
```

### D6: Tool versioning and rollback

```rust
struct ToolRegistry {
    tools: HashMap<String, Vec<ToolVersion>>,  // name → all versions
    active_version: HashMap<String, u32>,       // name → current version number
    max_versions_per_tool: usize,               // default: 5
}
```

**Rollback triggers**:
1. Quality drops > 0.1 below parent version (automatic)
2. Failure rate > 50% over last 10 uses (automatic)
3. Human explicitly requests rollback
4. Divergence detected at this tool's composition level

**Rollback procedure**:
1. Set `active_version[name]` back to previous stable version
2. Mark failed version as `quarantined`
3. Log rollback event with quality evidence
4. IntrinsicMotivation creates practice goal: "Why did v{N} fail?"

## Recommended Architecture: ToolBootstrapper Module

```rust
/// Orchestrates recursive tool composition and bootstrapping.
pub struct ToolBootstrapper {
    safety: BootstrapSafetyConfig,
    /// Compositions created this session.
    session_compositions: u32,
    /// Quality tracking per composition level.
    level_quality: HashMap<u32, EmaTracker>,
    /// Reference to ToolRegistry for registration.
    registry: ToolRegistry,
}

impl ToolBootstrapper {
    /// Detect repeated patterns and propose compositions.
    fn detect_pattern(&self, tracker: &ToolUsageTracker) -> Vec<CompositionCandidate>;
    
    /// Create a composed tool from a plan template.
    fn compose(&mut self, name: &str, plan: PlanTemplate, level: u32) -> Result<ComposedTool>;
    
    /// Validate a composed tool against past executions.
    fn validate(&self, tool: &ComposedTool, past_executions: &[PlanOutcome]) -> bool;
    
    /// Register a composed tool (with safety checks).
    fn register(&mut self, tool: ComposedTool) -> Result<()>;
    
    /// Check for divergence at any composition level.
    fn check_divergence(&self) -> Option<DivergenceAlert>;
    
    /// Roll back a tool to its previous version.
    fn rollback(&mut self, tool_name: &str);
}
```

### Integration with Existing Modules

- **ToolCreationPipeline**: Extended to handle `composed_from:` directive in tool specs. Single tools still go through WASM validation.
- **PlanExecutor**: Composed tools use PlanExecutor internally. No changes needed.
- **PlanCache**: Composed tools are stored as plan templates in PlanCache for retrieval.
- **ToolUsageTracker**: Pattern detection (`detect_pattern()`) queries tracker for co-occurrence statistics.
- **IntrinsicMotivation**: Repeated multi-step patterns flagged as composition opportunities.
- **ProactiveController**: Proposes composition at appropriate autonomy levels.
- **MirrorTest**: Validates composed tools against historical execution data.
- **EmergenceBenchmark**: Tool composition count and quality tracked as "Self-Extension" metric.

## Test Plan

1. `test_composed_tool_creation` — create from plan template
2. `test_composed_tool_execution` — runs plan and returns output
3. `test_level_assignment` — host=0, model=1, composed=2, meta=3
4. `test_pattern_detection` — repeated A→B→C sequence detected
5. `test_quality_gate_pass` — quality > 0.7 registers
6. `test_quality_gate_fail` — quality < 0.7 rejected
7. `test_monotonicity_check` — composed worse than parts rejected
8. `test_composition_budget` — max 5 per session
9. `test_level_limit` — level > 3 rejected
10. `test_divergence_detection` — level-2 quality drops below level-1
11. `test_sandbox_validation` — test against past 10 executions
12. `test_version_tracking` — v1 → v2 quality tracked separately
13. `test_auto_rollback` — quality drop triggers rollback
14. `test_replace_vs_extend` — same signature → replace, different → extend
15. `test_max_versions` — old versions pruned at limit
16. `test_human_approval_gate` — SemiAutonomous requires approval
17. `test_quarantine` — failed version marked, not deleted
18. `test_bootstrap_prompt_parsing` — `[tool_create] composed_from:` parsed
19. `test_recursive_composition` — level-2 tool uses level-1 tools
20. `test_meta_tool` — level-3 tool creates level-1 tools

## References

- Wang, L. et al. (2023). "Voyager: An Open-Ended Embodied Agent with Large Language Models" — skill library + composition
- Qian, C. et al. (2023). "Experiential Co-Learning of Software-Developing Agents" — recursive tool creation
- Cai, T. et al. (2023). "Large Language Models as Tool Makers (LATM)" — tool creation + reuse cycle
- Rana, C. et al. (2023). "CREATOR: Disentangling Abstraction and Reasoning" — tool creation with composability
- Schmidhuber, J. (2013). "PowerPlay: Training an Increasingly General Problem Solver" — self-improving system safety
- Stanley, K. & Lehman, J. (2015). "Why Greatness Cannot Be Planned" — open-endedness vs convergence in bootstrapping
