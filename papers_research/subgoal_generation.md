# Research: Subgoal Generation — Hierarchical Goal Decomposition

## Summary

Research into hierarchical goal decomposition for FerrisRes's PlanExecutor. The current PlanExecutor handles flat plans (sequential tool calls with `$N` references) but cannot decompose a high-level goal into subgoals. This research covers AND/OR goal trees, Hierarchical Task Networks (HTN), Goal-Oriented Action Planning (GOAP), decomposition-vs-execution heuristics, and pattern storage/retrieval.

## Existing Architecture

PlanExecutor currently supports:
- Flat plans: `[plan] Step 1: tool(args) → Step 2: tool($1) [/plan]`
- Sequential execution with `$N` output references
- Failure handling: retry with different params or replan from failed step
- Max replanning attempts (default 3)
- Inference-side plan validation (tools exist, references valid)
- Step conditions: `$N.success`, `$N.output.contains('x')`

ToolCreationPipeline adds:
- Model-generated tool specs parsed from `[tool_create]...[/tool_create]`
- WasmSandbox validation, MirrorTest quality check
- Max 3 refinement iterations

## Approaches

### 1. AND/OR Goal Trees

**Structure**: Each goal node is either an AND-node (all children must succeed) or an OR-node (any child succeeding suffices).

```
Goal: "Analyze this dataset"
├── AND: "Load + Clean"
│   ├── "Parse CSV" (leaf: parse_csv tool)
│   └── "Remove nulls" (leaf: clean_data tool)
├── AND: "Analyze"
│   ├── "Compute statistics" (leaf: stats_tool)
│   └── "Find outliers" (leaf: outlier_detect)
└── OR: "Visualize"
    ├── "Generate plot" (leaf: plot_tool)
    └── "Generate table" (leaf: table_tool)
```

**Pros**:
- Simple to implement and reason about
- Naturally maps to existing PlanStep structure (AND = sequential, OR = alternative)
- Failure semantics are clear: AND fails if any child fails, OR fails only if all children fail
- Supports partial success reporting

**Cons**:
- No ordering constraints beyond parent-child
- No resource/state modeling
- Manual decomposition (model must do the work)

**Implementation fit**: Very good. PlanStep already has conditions and retry logic. An AND/OR tree would be `Vec<SubgoalNode>` where each node is either `Leaf(PlanStep)` or `Decomposed { kind: And|Or, children: Vec<SubgoalNode> }`.

### 2. Hierarchical Task Networks (HTN)

**Structure**: Tasks are decomposed using methods (recipes). Each method has preconditions and a set of subtasks. The planner searches for applicable methods.

```
Task: "Analyze dataset"
Method 1 (precondition: has_csv_path):
  → Subtasks: [parse_csv, clean_data, compute_stats]
Method 2 (precondition: has_database_url):
  → Subtasks: [query_db, clean_data, compute_stats]
```

**Pros**:
- Domain knowledge encoded in methods (reusable decomposition patterns)
- Preconditions enable context-aware decomposition
- Well-studied in robotics and game AI (SHOP, SHOP2)
- Methods are essentially stored plan templates → natural fit with PlanCache concept

**Cons**:
- Requires a method library (bootstrapping problem: who writes the methods?)
- More complex implementation than AND/OR trees
- Over-constrained if methods are too specific

**Implementation fit**: Good. Methods could be stored as `ConceptEntry` in ConceptMemory, giving automatic persistence and retrieval. A method = a named decomposition pattern with preconditions.

### 3. Goal-Oriented Action Planning (GOAP)

**Structure**: Define a world state as key-value pairs. Each action has preconditions (required state) and effects (state changes). Planner searches backward from goal state to current state using A*.

```
Current state: { raw_data: true, cleaned: false, analyzed: false }
Goal state: { analyzed: true, visualized: true }

Actions:
  clean_data: requires { raw_data: true } → sets { cleaned: true }
  compute_stats: requires { cleaned: true } → sets { analyzed: true }
  plot_results: requires { analyzed: true } → sets { visualized: true }
```

**Pros**:
- Very flexible — plans emerge from state, not templates
- No method library needed
- Naturally handles replanning (state changed → re-search)
- A* finds optimal plan given cost function

**Cons**:
- Requires formal world state representation (hard for open-ended LLM tasks)
- Action definitions are rigid — doesn't handle fuzzy LLM outputs well
- Search can be expensive for large action spaces
- Doesn't map naturally to FerrisRes's tool-centric architecture

**Implementation fit**: Poor for direct use. The world-state abstraction doesn't match LLM tool outputs (which are natural language, not key-value pairs). However, the backward-chaining idea (start from goal, work backward to current capabilities) is useful for the decomposition heuristic.

### 4. LLM-Based Decomposition (Prompt-Driven)

**Structure**: Ask the model to decompose the goal. No formal planning structure — rely on the model's ability to break problems into steps.

```
Prompt: "Decompose this goal into subgoals:
Goal: Analyze this dataset
Available tools: parse_csv, clean_data, compute_stats, plot_results, find_outliers

Format:
[subgoals]
AND:
  1. Load and clean the data
  2. Run statistical analysis
OR:
  3a. Generate visualization
  3b. Generate summary table
[/subgoals]"
```

**Pros**:
- Zero implementation complexity
- Handles open-ended goals that formal planners can't
- Naturally leverages LLM reasoning ability
- Can combine with stored patterns ("here's how we solved similar problems before")

**Cons**:
- No formal correctness guarantees
- Decomposition quality depends on model capability
- Hard to validate subgoal feasibility
- No learning from decomposition failures (unless explicitly tracked)

**Implementation fit**: Excellent for bootstrapping. The model already emits `[plan]...[/plan]` blocks. Extending to `[subgoals]...[/subgoals]` is natural. Combined with AND/OR tree parsing, this gives us LLM-driven decomposition with structured output.

## Recommended Approach: Hybrid AND/OR + LLM Decomposition + HTN Patterns

### Architecture

```
SubgoalGenerator
├── decompose(goal, available_tools, context) → SubgoalTree
│   ├── 1. Check PlanCache for similar past goals → reuse if found
│   ├── 2. Ask model to decompose (prompt-driven)
│   ├── 3. Parse output into AND/OR tree
│   ├── 4. Validate: all leaf tools exist, references valid
│   └── 5. Store successful decomposition as HTN method in ConceptMemory
├── execute_tree(tree) → Result
│   ├── AND nodes: execute all children, fail if any fails
│   ├── OR nodes: try children in order, succeed if any succeeds
│   └── Leaf nodes: delegate to PlanExecutor (existing flat execution)
└── learn_from_execution(tree, outcome)
    ├── Store decomposition pattern in PlanCache
    └── Update quality scores for subgoal patterns
```

### Data Structures

```rust
/// Decomposition kind for a subgoal node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecomposeKind {
    /// All children must succeed (sequential or parallel).
    And,
    /// Any child succeeding is sufficient (try in order).
    Or,
}

/// A node in the subgoal tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubgoalNode {
    /// Leaf: maps to a flat plan for PlanExecutor.
    Leaf {
        label: String,
        plan: Vec<PlanStep>,
    },
    /// Decomposed: contains child subgoals.
    Decomposed {
        label: String,
        kind: DecomposeKind,
        children: Vec<SubgoalNode>,
        /// Maximum depth for recursive decomposition.
        max_depth: usize,
    },
}

/// A stored decomposition pattern (HTN method).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionPattern {
    /// Goal description (used for retrieval).
    pub goal_embedding: Vec<f32>,
    /// The decomposition tree.
    pub tree: SubgoalNode,
    /// Preconditions for this decomposition to apply.
    pub preconditions: Vec<String>,
    /// Success rate from past executions.
    pub success_rate: f32,
    /// Number of times used.
    pub usage_count: u32,
}
```

### Decomposition Heuristic

**When to decompose vs execute directly**:
1. **Tool coverage**: If the goal can be achieved by a single existing tool → execute directly (no decomposition)
2. **Goal complexity**: If the goal contains "and", "then", multiple verbs → decompose
3. **Past patterns**: If PlanCache has a matching pattern with >70% success rate → reuse
4. **Failure trigger**: If flat execution fails after max retries → decompose and retry

**Max depth**: Default 3 levels. Prevents infinite decomposition. At max depth, force leaf nodes.

### Prompt Template for Decomposition

```text
Decompose this goal into subgoals. Use AND (all must succeed) and OR (alternatives).

Goal: {goal}
Available tools: {tool_list}
Context: {recent_episodes}
Past patterns: {cached_patterns}

Format:
[subgoals]
AND:
  1. {subgoal_1}
  2. {subgoal_2}
OR:
  3. {alternative_a} | {alternative_b}
[/subgoals]

For each subgoal, specify the tool to use and arguments.
```

### Integration with Existing Modules

- **PlanExecutor**: Leaf nodes delegate to existing `PlanExecutor::execute_plan()`. No changes needed.
- **ConceptMemory**: Decomposition patterns stored as `ConceptContent::Algorithm` entries with embeddings.
- **EpisodicMemory**: Subgoal outcomes logged as episodes for future retrieval.
- **IntrinsicMotivation**: Failed decompositions create practice goals for planning skills.
- **ToolCreationPipeline**: If no tool matches a subgoal, trigger tool creation.
- **EmergenceBenchmark**: Track decomposition quality over time (plan depth, success rate).

## Safety Constraints

1. **Max depth**: 3 levels of decomposition. Prevents infinite recursion.
2. **Max subgoals per level**: 8. Prevents combinatorial explosion.
3. **Timeout**: Each decomposition attempt has a 30-second timeout.
4. **Fallback**: If decomposition fails, fall back to flat PlanExecutor execution.
5. **Resource limit**: Total leaf tool calls across all subgoals capped at 50 per request.

## Test Plan

1. `test_parse_and_tree` — parse AND subgoal block
2. `test_parse_or_tree` — parse OR subgoal block
3. `test_nested_decomposition` — AND → OR → AND nesting
4. `test_max_depth_enforcement` — depth > 3 becomes leaf
5. `test_leaf_execution` — leaf delegates to PlanExecutor
6. `test_and_node_partial_failure` — one child fails, parent fails
7. `test_or_node_fallback` — first fails, second succeeds
8. `test_decomposition_pattern_storage` — store pattern in ConceptMemory
9. `test_pattern_retrieval` — find similar past decomposition
10. `test_pattern_reuse` — reuse cached pattern instead of decomposing
11. `test_goal_complexity_heuristic` — complex goals trigger decomposition
12. `test_tool_coverage_heuristic` — single-tool goals skip decomposition
13. `test_failure_triggered_decomposition` — flat failure → decompose
14. `test_resource_limit` — 50 leaf cap
15. `test_timeout` — decomposition timeout

## References

- Erol, K., Hendler, J., & Nau, D. (1994). "HTN Planning: Complexity and Expressivity"
- Fikes, R. & Nilsson, N. (1971). "STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving"
- Orkin, J. (2004). "Applying Goal-Oriented Action Planning to Games" (GOAP in F.E.A.R.)
- Yao, S. et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models"
- Wang, L. et al. (2023). "Voyager: An Open-Ended Embodied Agent with Large Language Models" (skill library decomposition)
