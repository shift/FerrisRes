# FerrisRes API Reference

> **Version 0.2.0** — API is unstable and may change before 1.0.0.

## Module Stability Tiers

| Tier | Modules | Guarantee |
|---|---|---|
| **Stable** | `inference::sampling`, `inference::prompt_templates`, `model::config` | No breaking changes |
| **Beta** | `inference::generator`, `inference::rag`, `model::block_attn_res`, `model::safetensors` | Minor breaking changes possible |
| **Unstable** | `compute::*`, `device::*`, `model::gpu_forward`, `training::*` | May change any time |

## Core Types

### Inference

```rust
// Sampling
pub fn sample_top_k(logits: &[f32], k: usize, temperature: f32) -> u32;
pub fn sample_top_p(logits: &[f32], p: f32, temperature: f32) -> u32;
pub fn sample_argmax(logits: &[f32]) -> u32;

// Prompt Templates
pub enum TemplateKind { ChatML, Llama2, Mistral, Alpaca, Raw }
pub fn apply_template(template: TemplateKind, messages: &[Message]) -> String;

// RAG
pub struct RagStore { /* dense/sparse/hybrid retrieval */ }
impl RagStore {
    pub fn add_document(&mut self, text: &str, metadata: HashMap<String, String>);
    pub fn retrieve(&self, query: &str, top_k: usize) -> Vec<RagResult>;
}
```

### Model Loading

```rust
// Safetensors
pub fn load_safetensors(path: &Path) -> Result<LoadedWeights>;
pub struct MmapedSafetensors { /* memory-mapped, zero-copy */ }

// GGUF
pub fn load_gguf(path: &Path) -> Result<GgufFile>;

// Architecture detection
pub enum DetectedArchitecture { Llama, Mistral, Gemma4, Gemma4Mm, Unknown }
pub fn detect_architecture_from_names(names: &[String]) -> DetectedArchitecture;
```

### Self-Improvement

```rust
// WASM Sandbox
pub struct WasmRuntime { /* wasmi engine, fuel limits */ }
impl WasmRuntime {
    pub fn execute_parse_from_bytes(&self, wasm: &[u8], code: &str) -> Result<WasmParseResult>;
}

// Mirror Test
pub struct MirrorTestRunner { /* code→test→execute→loss */ }
impl MirrorTestRunner {
    pub fn evaluate(&self, code: &str, test_code: &str, language: &str) -> MirrorTestResult;
}

// Concept Memory
pub struct ConceptMap { /* embedding-based retrieval */ }
impl ConceptMap {
    pub fn store(&mut self, name: String, embedding: Vec<f32>, content: ConceptContent, ...) -> ConceptId;
    pub fn retrieve(&mut self, query: &[f32], top_k: usize) -> Vec<RetrievedConcept>;
    pub fn save(&self, path: &Path) -> Result<()>;
    pub fn load(path: &Path) -> Result<Self>;
}
```

### Training

```rust
// Distillation
pub fn distillation_kl_loss(
    teacher_logits: &[f32], student_logits: &[f32],
    temperature: f32, vocab_size: usize,
) -> f32;

// LoRA
pub struct LoraManager { /* hot-swap adapters */ }
impl LoraManager {
    pub fn apply(&mut self, adapter: &LoraAdapter);
    pub fn merge_all(&mut self);
}

// Autodiff
pub struct ComputationGraph { /* reverse-mode AD */ }
impl ComputationGraph {
    pub fn backward(&mut self) -> Vec<f32>;
}
```

## CLI

```
ferrisres <command> [options]

Commands:
  infer       Run inference
  train       Train a model
  distill     Distill a teacher model to Block AttnRes
  serve       Start OpenAI-compatible API server
  benchmark   Run benchmarks
  info        Show device info

Infer options:
  --prompt <TEXT>        Input prompt
  --template <TEMPLATE>  Prompt template (chatml|llama2|mistral|alpaca|raw)
  --max-tokens <N>       Max tokens to generate
  --image <PATH>         Input image for multimodal
  --yarn-scale <F>       YaRN context extension scale

Distill options:
  --model-path <PATH>    Path to safetensors model
  --config <CONFIG>      Model config (e2b|e4b|27b-mm)
  --steps <N>            Training steps
  --seq-len <N>          Sequence length
  --learning-rate <F>    Learning rate
  --temperature <F>      Distillation temperature
  nable GPU acceleration
  --resume <PATH>        Resume from checkpoint

Serve options:
  --port <PORT>          Server port (default: 8080)
```

## Error Handling

All fallible operations return `Result<T, FerrisResError>`:

```rust
pub enum FerrisResError {
    Shape(String),       // Tensor shape mismatch
    Compute(String),     // GPU compute error
    Io(String),          // I/O error
    Model(String),       // Model loading/forward error
    Tokenizer(String),   // Tokenization error
}
```
