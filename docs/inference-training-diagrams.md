# FerrisRes: inference and training

High-level text-model flows. Dashed arrows denote optional configuration, not features enabled in every entry point.

## 1. Inference — use learned weights to generate output

```mermaid
flowchart TD
    A[User prompt] --> B[Prompt template and tokenizer]
    W[Load model weights and configuration] --> M[Model ready: BlockAttnRes or standard transformer]
    M --> P
    B --> P[Prefill: embed and process full prompt]
    P --> K[(Per-layer KV cache)]
    P --> H[LM head: next-token logits]
    H --> S[Penalties, temperature, top-k / top-p sampling]
    S --> O[Emit token / stream decoded text]
    O --> E{EOS or token limit?}
    E -->|Yes| F[Finished response]
    E -->|No| D[Decode: embed latest token and run model]
    K --> D
    D -->|Append new keys and values| K
    D --> H
    L[Optional LoRA adapter] -.-> M
    HW[Hardware profile and compute backend] -.-> P
    HW -.-> D
```

**Model core:** the native model combines intra-block processing with inter-block representation attention; MoE variants route to selected experts. Standard-transformer compatibility uses its own layer implementation. Model weights stay fixed during ordinary inference; the KV cache changes as tokens are generated.

Optional vision/audio encoders and modality-specific output heads are separate paths, omitted here to keep the text generation loop clear. Retrieval, tool execution, and Armor depend on the selected application path; they are not universal stages of `UnifiedTokenGenerator`.

## 2. Training — update parameters from prediction error

```mermaid
flowchart TD
    A[Training text / examples] --> B[Tokenize and construct batches]
    B --> X[Input tokens]
    B --> Y[Next-token targets]
    W[Initialize or load model weights] --> F[Forward pass: embeddings, model core, LM head]
    X --> F
    F --> P[Predicted token logits]
    P --> L[Cross-entropy loss]
    Y --> L
    L --> G[Backward pass: compute gradients]
    G --> AC[Accumulate gradients across microbatches / tiles]
    AC --> U[Optimizer updates trainable parameters]
    U --> N{More training steps?}
    N -->|Yes: updated weights, next batch| F
    N -->|No| C[Save trained weights or adapters]
    C --> I[Load for inference]
    MODE[Full training: model parameters\nLoRA / QLoRA: adapters, frozen base] -.-> U
    MEM[Optional activation checkpointing\nand CPU / async gradient offload] -.-> G
    HW[Device profile selects compute and optimizer strategy] -.-> U
```

This is the conceptual supervised next-token training flow, not a claim that every training entry point uses the same orchestrator. Batch input and targets advance on each loop iteration. Optimizer implementations include SGD, Adam, SCALE, and AdaMeM; selection depends on the path and configuration. QLoRA keeps the quantized base frozen and trains low-rank adapters. Distillation is an alternative training path using teacher supervision, not a required stage of this diagram.

## Source references

- `src/inference/unified_generator.rs`: prefill, LM head, sampling, cache updates, decode and stop conditions.
- `src/model/model.rs`: native intra-block and inter-block forward processing.
- `src/model/dispatcher.rs`: model-specific dispatch.
- `src/training/mod.rs`: training state, optimizer exports, checkpointing and offload configuration.
- `src/training/optimizer.rs`: optimizer and cross-entropy implementations.
- `src/training/lora.rs`, `src/training/qlora.rs`: adapter training components.
- `README.md`: broader architecture and optional modalities.
