# Research: Multi-Step Planning & Tool Chaining

## Task ID: 69d3a8ad-8c72-43be-9181-dd249c64d74d

## Key Papers & Techniques

### 1. ReAct (Reasoning + Acting)
- **Paper**: Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in Language Models" [arXiv:2210.03629]
- **Core idea**: Interleave reasoning traces (Thought) with action execution (Action/Observation).
- **Pattern**: Thought → Action → Observation → Thought → Action → ...
- **Key insight**: Reasoning helps plan, actions ground reasoning in reality. Together they outperform either alone.
- **Application to FerrisRes**: Model generates a reasoning trace before each tool call. The trace serves as the "plan" and can be stored as a concept for reuse.

### 2. Plan-and-Solve Prompting
- **Paper**: Wang et al. (2023) "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning" [arXiv:2305.04091]
- **Core idea**: Two-phase approach: (1) Generate a plan (sub-tasks), (2) Execute each sub-task.
- **Plan format**: "Step 1: ... Step 2: ... Step 3: ..."
- **Application**: Model emits a full plan before executing any tools. Plan stored in ConceptMap. On failure, only re-plan from the failed step.

### 3. Toolformer Tool Chaining
- **Paper**: Schick et al. (2023) [arXiv:2302.04761]
- **Core idea**: Model learns to compose tool calls: output of tool A becomes input of tool B.
- **Chaining format**: [QA("What is X?")] → [Calculator(result)] → [Summarize(result)]
- **Application**: Pipeline supports sequential tool calls where output references are resolved ($1, $2, etc).

### 4. LangChain / AutoGPT Agent Patterns
- **LangChain**: Chase (2022) — Framework for chaining LLM calls with tools
- **AutoGPT**: Richards (2023) — Autonomous GPT-4 agent with planning loop
- **Core pattern**: Agent maintains a scratchpad of (thought, action, observation) triples. Each step appends to scratchpad. Agent decides when done.
- **Application**: CognitivePipeline already has the scratchpad pattern (generate → tool call → result). Extend to multi-step by looping until "done" signal.

### 5. Reflexion
- **Paper**: Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning" [arXiv:2303.11366]
- **Core idea**: Agent attempts task, fails, generates verbal reflection on failure, retries with reflection in context. Reflections accumulate across episodes.
- **Application**: After tool chain failure, model generates a reflection (why it failed, what to change). Reflection stored in ConceptMap. Next attempt uses reflection.

### 6. CALM Programs as Plans
- **Direct application**: CALM VM already has a sequential instruction format. A "plan" IS a CALM program:
  - Step 1: LookUp relevant concepts
  - Step 2: Compute intermediate result
  - Step 3: Call external tool
  - Step 4: Transform result
  - Step 5: Return answer
- The LlmComputer can execute plans natively. We just need the model to generate plan-programs.

## Recommended Approach for FerrisRes

### Plan Representation
```rust
struct ToolPlan {
    steps: Vec<PlanStep>,
    // References between steps: $N refers to output of step N
}

struct PlanStep {
    tool: String,
    args: String,  // Can contain $1, $2, etc for references
    condition: Option<String>,  // Execute only if condition met
    retry_on_fail: bool,
    max_retries: usize,
}
```

### Plan Execution Engine
```
1. Model emits plan in structured format:
   [plan]
   Step 1: concept_lookup(prompt)
   Step 2: calm_execute(sum $1.count 0)
   Step 3: mirror_test($2)
   [/plan]

2. PlanExecutor validates plan:
   - All referenced tools exist in registry
   - All $N references are valid (step N exists, hasn't failed)
   - No circular references

3. Execute steps sequentially:
   - Resolve $N references from previous outputs
   - Execute tool via cognitive pipeline (sandbox, mirror test)
   - On failure: if retry_on_fail, retry with different params
   - On persistent failure: replan from failed step

4. Store successful plans:
   - Plan → ConceptMap as Algorithm concept
   - Embedding: prompt embedding
   - Tags: ["plan", tool_names...]
```

### Replanning on Failure
When a step fails:
1. Generate reflection on failure (Reflexion pattern)
2. Store reflection in ConceptMap
3. Re-plan from the failed step, incorporating reflection
4. Max 2 replans before giving up

## Key References
1. [arXiv:2210.03629] Yao 2023 - ReAct
2. [arXiv:2305.04091] Wang 2023 - Plan-and-Solve
3. [arXiv:2303.11366] Shinn 2023 - Reflexion
4. [arXiv:2302.04761] Schick 2023 - Toolformer
