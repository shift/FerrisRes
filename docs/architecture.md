# FerrisRes Architecture

## Block AttnRes: O(n) Transformer

Standard transformers use quadratic attention: every token attends to every other token. Block AttnRes reduces this to linear time through a two-level hierarchy.

### Level 1: Intra-block attention

The token sequence is divided into fixed-size blocks (default: 8 tokens). Within each block, standard multi-head self-attention runs with RoPE positional encoding. This is O(block_size²) — constant per block.

### Level 2: Inter-block attention

Block summaries (mean-pooled representations of each block) attend across all blocks. Since there are only `n / block_size` blocks, this is O(n / block_size) — linear in the total sequence length.

### Distillation

Standard transformer models (Gemma 4) are converted to Block AttnRes through structural linearization:

```
Teacher (Gemma 4, O(n²))
    ↓  KL divergence loss
Student (Block AttnRes, O(n))
```

The teacher's attention patterns are preserved via KL divergence loss during training. Quality is 95-99% of the teacher, measured by perplexity.

## System Architecture

```
┌─────────────────────────────────────────────┐
│                  CLI / API Server            │
├─────────────────────────────────────────────┤
│             Inference Pipeline               │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │Generator │ │ Logit    │ │ Sampling     │ │
│  │(prefill+ │ │Processor │ │ (top-k/top-p)│ │
│  │ decode)  │ │ Chain    │ │              │ │
│  └─────────┘ └──────────┘ └──────────────┘ │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ RAG     │ │ Tools    │ │ WASM Sandbox │ │
│  │ Store   │ │ Registry │ │ (zero-trust) │ │
│  └─────────┘ └──────────┘ └──────────────┘ │
├─────────────────────────────────────────────┤
│              Model Layer                     │
│  ┌─────────────┐ ┌───────────────────────┐  │
│  │BlockAttnRes │ │ Standard Transformer  │  │
│  │ O(n)        │ │ O(n²) compatibility  │  │
│  └─────────────┘ └───────────────────────┘  │
│  ┌─────────────┐ ┌───────────────────────┐  │
│  │ Vision      │ │ Audio (EnCodec)       │  │
│  │ (Implicit   │ │ (RVQ codebooks)       │  │
│  │  GEMM)      │ │                       │  │
│  └─────────────┘ └───────────────────────┘  │
├─────────────────────────────────────────────┤
│            Compute Layer (wgpu)              │
│  Vulkan │ Metal │ DX12 │ WebGPU             │
├─────────────────────────────────────────────┤
│          Device Adaptation                   │
│  Integrated │ Low-End │ Mid-Range │ High-End │
└─────────────────────────────────────────────┘
```

## Self-Improvement Loop

FerrisRes implements a closed-loop self-correction system:

1. **Model generates code** → WASM sandbox validates in <1ms
2. **LSP-as-Oracle** provides deterministic compiler feedback
3. **Mirror Test** — model writes tests for its own code
4. **Test failures → loss signal → backprop** at the weight level
5. **Concept Memory** persists learned patterns across sessions

## Cognitive Architecture

FerrisRes includes a 5-layer cognitive architecture for self-improving AI:

```
Layer 0: Pipeline Wiring    — orchestrates all cognitive components
Layer 1: Memory & Learning  — episodic memory, differentiable execution, LoRA weight updates
Layer 2: Autonomy            — tool creation, multi-step planning, usage tracking
Layer 3: Self-Improvement    — abstraction, intrinsic motivation, proactive behavior
Layer 4: Emergence           — quantitative emergence measurement (6 categories)
```

### Layer 0: Cognitive Pipeline
The `CognitivePipeline` orchestrates all cognitive components through a unified
`process_generation()` entry point:
- **ConceptMap**: embedding-based learned pattern retrieval with quality scoring
- **LlmComputer**: CALM virtual machine (LookUp → Compute → BranchIf)
- **MirrorTest**: recursive self-verification — generate code → test → loss
- **HullKVCache**: 2D convex hull attention with O(log n) lookups
- **WasmSandbox**: zero-trust tool execution with wasmi runtime

### Layer 1: Memory & Learning
- **EpisodicMemory**: Stores *experiences* (prompt, tool traces, outcome, importance),
  not raw tokens. Content-based retrieval via cosine similarity + recency bias.
  Importance = surprise × uncertainty × outcome_magnitude. Compression merges
  similar episodes (cosine > 0.85) into generalizations.
- **DifferentiableLlmComputer**: Makes CALM VM differentiable via Gumbel-Softmax
  op selection with Straight-Through Estimator (STE). NTM-style DiffMemoryBank
  for gradient flow through memory. Temperature annealing (τ: 1.0 → 0.1).
- **ToolTriggeredLora**: On-the-fly LoRA weight updates from the `learn` tool.
  Elastic Weight Consolidation (Fisher diagonal) prevents catastrophic forgetting.
  Progressive adapter stacking — new adapter per learning event.

### Layer 2: Autonomy
- **ToolCreationPipeline**: Model generates tool specs via `[tool_create]` blocks.
  6-stage validation (name, code size, structure, safety, syntax, semantics).
  Bans unsafe code (filesystem, network, process spawning). Refinement loop.
- **PlanExecutor**: Multi-step `[plan]` execution with `$N` reference resolution.
  Condition evaluation (`$1.success`), retry on failure, replanning.
- **ToolUsageTracker**: Per-tool + per-context EMA quality tracking.
  Contextual bandit for best-tool recommendation. JSON persistence.

### Layer 3: Self-Improvement
- **AbstractionEngine**: Scans concepts for similarity clusters (cosine > 0.8),
  computes centroid meta-concepts. Hierarchical levels:
  Instance → Pattern → Principle → MetaPrinciple.
- **IntrinsicMotivation**: Per-concept uncertainty (entropy + quality + distance).
  Zone of Proximal Development goal selection. Mastery detection.
- **ProactiveController**: 4 autonomy levels (Reactive → Suggestive →
  SemiAutonomous → FullyAutonomous). 6 initiative signals. Action logging + rollback.

### Layer 4: Emergence Measurement
- **EmergenceBenchmark**: 6 measurement categories:
  - Skill Acquisition (improvement rate), Self-Correction (error recurrence),
    Self-Extension (tools × reuse × depth), Cognitive Scaffolding (concepts vs diversity),
    Planning (success rate), Abstraction (compression ratio).
  - Compares baseline (no pipeline) vs augmented (with pipeline).
  - Trend analysis and composite emergence index (0.0–1.0).

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Pure Rust (no Python) | Single build system, no FFI overhead |
| wgpu (not CUDA) | Runs on NVIDIA, AMD, Intel, Apple, Qualcomm |
| Block AttnRes | O(n) attention without quality loss |
| WASM sandbox | Zero-trust tool execution |
| Memory-mapped weights | 10GB model fits in 16GB RAM |
| JIT GPU uploads | Scales from 256MB to multi-GB buffers |
