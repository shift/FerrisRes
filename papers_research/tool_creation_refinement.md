# Research: Tool Creation/Refinement Loop — Model Generates Its Own Tools

## Task ID: 59606c3d-f99d-47a6-a422-b79a9775428f

## Key Papers & Techniques

### 1. Voyager (MineDojo) — Skill Library
- **Paper**: Wang et al. (2023) "Voyager: An Open-Ended Embodied Agent with Large Language Models" [arXiv:2305.16291]
- **Core architecture**: LLM agent in Minecraft that:
  1. Proposes tasks based on state
  2. Writes JavaScript code to complete tasks
  3. Stores successful programs as "skills" in a skill library
  4. Retrieves relevant skills for new tasks via embedding similarity
- **Skill format**: {task_description, code} pairs indexed by task embedding
- **Key insight**: Skills accumulate over time, making the agent more capable. New tasks build on existing skills.
- **Application to FerrisRes**: Direct mapping — tool = skill, ToolRegistry = skill library, ConceptMap = skill index. The model generates WASM/DSL tool specs, validates in sandbox, stores in registry.

### 2. Toolformer
- **Paper**: Schick et al. (2023) "Toolformer: Language Models Can Teach Themselves to Use Tools" [arXiv:2302.04761]
- **Core idea**: Self-supervised tool learning. The model:
  1. Generates potential API call positions in text
  2. Executes the API calls
  3. Compares loss with/without API results
  4. Keeps calls that reduce perplexity
- **Application**: Model learns WHEN to create tools by self-supervision. Not just using tools, but deciding they're needed.

### 3. CodeGen / StarCoder Self-Repair
- **CodeGen**: Nijkamp et al. (2023) [arXiv:2203.13474]
- **Self-Repair**: Chen et al. (2023) "Teaching Large Language Models to Self-Debug" [arXiv:2304.05128]
- **Core idea**: Model generates code, executes it, reads error messages, fixes bugs. Iterates until tests pass.
- **Application**: Tool refinement loop — model generates tool, MirrorTest evaluates, model reads failure, rewrites tool, repeats.

### 4. Program Synthesis with LLMs
- **AlphaCode**: Li et al. (2022) [arXiv:2203.07814]
- **Core approach**: Generate many candidate programs, cluster by output, submit most common. Massively sample, filter, select.
- **Application**: When creating a new tool, generate multiple candidates, validate each in WasmSandbox, select the one that passes MirrorTest.

### 5. LATM (Large Language Models as Tool Makers)
- **Paper**: Cai et al. (2023) "Large Language Models as Tool Makers" [arXiv:2305.17126]
- **Core idea**: Two-phase framework:
  - Tool maker: LLM creates reusable tools from task descriptions
  - Tool user: LLM uses the created tools
- **Verification**: Tool maker tests the tool on example inputs before releasing
- **Application**: Direct mapping to FerrisRes architecture. The model is both tool maker and tool user. WasmSandbox validates. MirrorTest verifies.

### 6. CREATOR (Tool Creation via Disentangling)
- **Paper**: Qian et al. (2023) "CREATOR: Tool Creation for Disentangling Abstraction and Reasoning" [arXiv:2305.14318]
- **Core idea**: Disentangle tool creation from tool usage. Creation phase: analyze problem, create tool. Reasoning phase: use tool to solve.
- **Application**: Cognitive pipeline already has this separation — process_generation creates tools, execute_tool uses them.

## Recommended Approach for FerrisRes

### Tool Creation Pipeline

```
1. Model encounters problem it can't solve directly
2. Model emits [tool_create] specification:
   {
     name: "parse_csv",
     description: "Parse CSV string into structured data",
     parameters: "input: string",
     returns: "array of objects",
     dsl_code: "fn parse_csv(s) { ... }"  // DSL or WASM
   }
3. WasmSandbox compiles and validates the tool code
4. MirrorTest generates test cases and validates:
   - Input/output format matches spec
   - No crashes on edge cases (empty, null, overflow)
   - Performance within bounds
5. If validation passes → register in ToolRegistry + store in ConceptMap
6. If validation fails → feed error back to model for repair
7. Model retries (max 3 attempts)
```

### Tool DSL Design
Tools should be written in a restricted DSL (not full Rust/WASM) for safety:
- Input: typed parameters (string, number, array, object)
- Operations: string manipulation, arithmetic, array ops, lookups
- Output: typed return value
- No filesystem, no network, no unsafe operations
- Compiled to WASM for sandboxed execution

### Tool Refinement Loop
1. Track tool usage: (tool_name, context, success_rate)
2. When success_rate drops below threshold:
   - Retrieve the tool's concept entry (includes original problem + solution)
   - Analyze failure patterns from MirrorTest results
   - Generate improved version of the tool
   - Validate via WasmSandbox + MirrorTest
   - If improved: replace tool in registry, update concept
   - If not improved: keep original, log failure for future analysis

## Key References
1. [arXiv:2305.16291] Wang 2023 - Voyager
2. [arXiv:2302.04761] Schick 2023 - Toolformer
3. [arXiv:2304.05128] Chen 2023 - Self-Debug
4. [arXiv:2305.17126] Cai 2023 - LATM (LLMs as Tool Makers)
5. [arXiv:2305.14318] Qian 2023 - CREATOR
6. [arXiv:2203.07814] Li 2022 - AlphaCode
7. [arXiv:2203.13474] Nijkamp 2023 - CodeGen
