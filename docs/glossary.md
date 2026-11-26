# Glossary

One line per term, with the document that owns it. Alphabetical.

| term | meaning | doc |
|---|---|---|
| **active parameters** | parameters used for one token in a MoE; what decode pays for | `model-tiers.md` |
| **Adam** | the optimizer used by the f32 trainer; `m`, `v`, step | `train-and-export.md` |
| **Armor L1** | the injection/PII classifier; a filter, not the model's judgement | `security.md` |
| **BF16** | 16-bit brain float; the T4 training format the f32 path does not use | `hardware-shapes.md` |
| **Block AttnRes** | block summaries carried across layers; `boundary_layers` marks the blocks | `block-attnres-causal.md` |
| **BPE** | byte-pair encoding; the tokenizer algorithm | `tokenizer.md` |
| **CE** | cross-entropy loss; next-token training objective | `train-and-export.md` |
| **checkpoint** | model + optimizer + step, written by `train` (`FRTRAIN1`) | `train-and-export.md` |
| **context extension** | YaRN-style widening of the trained window; a minor version | `model-tiers.md` |
| **coverage** | in text-aligned KD, the teacher probability mass that maps to student tokens | `text-aligned-kd.md` |
| **decode** | generating one token at a time; bandwidth-bound | `hardware-shapes.md` |
| **distillation** | training a student on a teacher's outputs, logits or text | `distillation.md` |
| **embedding head** | a 768→256 adapter that gives a generative model an embedding mode | `embedding-head.md` |
| **fertility** | tokens per byte for a tokenizer on a given language/code | `tokenizer.md` |
| **FIM** | fill-in-the-middle; prefix/suffix/middle tokens for code infill | `tokenizer.md` |
| **FRAL** | the engine's own binary artefact format (e.g. Armor weights) | `naming.md` |
| **FRTRAIN1** | the training checkpoint format (config, tensors, Adam state) | `train-and-export.md` |
| **GGUF** | a quantised inference format and its metadata | `naming.md` |
| **GQA / MQA** | grouped-query / multi-query attention; fewer KV heads than query heads | `hardware-shapes.md` |
| **KV cache** | keys and values kept for earlier tokens; 3-bit in the tiers | `quantization.md` |
| **local SGD** | each rank trains its shard, weights are averaged every k steps | `tasks/multi-gpu.md` |
| **logits** | unnormalised next-token scores | `distillation.md` |
| **loop core** | the shared layers traversed multiple times in a looped model | `model-tiers.md` |
| **Loop KV** | MoLT's strict-prefix memory KV read from the previous loop (`j < t`) | `molt.md` |
| **loop forcing** | training loops in parallel rather than sequentially | `molt.md` |
| **LoRA** | low-rank adapters; the blocked-config gradient is the open defect | `distillation.md` |
| **Matryoshka** | embeddings trained so a prefix of the vector is usable | `embedding-head.md` |
| **MFU** | model FLOPs utilisation; measured, never assumed | `hardware-shapes.md` |
| **MLA** | multi-head latent attention; compressed KV for long context | `hardware-shapes.md` |
| **MoE** | mixture of experts; storage for capacity at fixed active bytes | `model-tiers.md` |
| **MoLA** | Mixture-of-Loops Attention; KV as memory read across loops | `molt.md` |
| **MoL-Hybrid-k** | the schedule that skips decoder traversals without changing predictions | `molt.md` |
| **NF4** | 4-bit normal-float weights; the planned head/weight format | `quantization.md` |
| **PLE** | per-layer embeddings; an auxiliary input in the block model | `architecture.md` |
| **prefill** | the first forward over the prompt; compute-bound | `hardware-shapes.md` |
| **principal** | the identity a run acts for; owns its project and quiet hours | `persona-and-principals.md` |
| **Q4_K_M / Q8_0** | GGUF quantisation levels; artefact properties, not model ids | `naming.md` |
| **rank / world size** | this process's index and the number of processes in a run | `tasks/multi-gpu.md` |
| **risk class** | Read / WriteLocal / WriteExternal / Destructive / Privileged on a tool | `security.md` |
| **safetensors** | the tensor file format used for weights and export | `naming.md` |
| **SeqKD** | sequence-level distillation: train on teacher-generated text | `text-aligned-kd.md` |
| **soul** | the committed identity/persona an assistant speaks with; data, not weights | `soul-training.md` |
| **student / teacher** | the model being trained / the model supplying targets | `distillation.md` |
| **ternary** | 1.58-bit weights (`-1,0,+1`); bandwidth and compute, trained from scratch | `quantization.md` |
| **text-aligned KD** | soft targets only where teacher and student byte boundaries coincide | `text-aligned-kd.md` |
| **tokenizer** | text → token ids; the 32k code-aware BPE is the shipping design | `tokenizer.md` |
| **top-k** | the k highest-probability tokens kept for a target or a sample | `distillation.md` |
| **TTFT** | time to first token; the instant-mode latency metric | `response-modes.md` |
| **U-MoLT** | MoLT with a looping encoder and a cross-only decoder | `molt.md` |
| **u16 shard** | packed token ids for a 32k vocab; half the I/O of u32 | `corpus-builder.md` |
| **vocab** | the token set; 32k for Nano/Small, 151k for Base/Large | `tokenizer.md` |
| **workflow / state / grant** | the `.workflow` grammar; `Tools:` is the authorization | `harness-integration.md` |
| **YaRN** | the context-extension method behind a minor version bump | `model-tiers.md` |
