# Research: Weight-Updating Tools — Continual Learning at Inference Time

## Task ID: 71ef9a3a-d03b-4075-a221-b0a6753c1907

## Key Papers & Techniques

### 1. Elastic Weight Consolidation (EWC)
- **Paper**: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks" [PNAS 114(13)]
- **Core idea**: After learning task A, compute Fisher information matrix diagonal for each parameter. When learning task B, add penalty term: L_total = L_B + λ/2 Σ F_i (θ_i - θ*_A,i)²
- **Fisher information**: F_i = E[∂²L/∂θ_i²] — measures how important parameter i is for task A
- **Application to FerrisRes**: When a "learn" tool triggers weight update, compute Fisher for affected parameters. Penalize changes to important weights. Store Fisher diagonals in checkpoint.

### 2. Progressive Neural Networks
- **Paper**: Rusu et al. (2016) "Progressive Neural Networks" [arXiv:1606.04671]
- **Core idea**: For each new task, add a new column (copy of network). Lateral connections from previous columns. No forgetting because old weights are frozen.
- **Application to FerrisRes**: Each "learn" call could add a thin adapter column. Model grows but never forgets. Memory cost: O(tasks × params).

### 3. LoRA / QLoRA (Already implemented)
- **Paper**: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models" [arXiv:2106.09685]
- **Dettmers et al. (2023)**: QLoRA [arXiv:2305.14314]
- **Already in FerrisRes**: src/training/lora.rs, src/training/qlora.rs
- **Application**: Tool-triggered LoRA = when "learn" tool fires, create a new LoRA adapter for relevant layers, compute gradient on the fly, apply update. Adapters accumulate over time.

### 4. Memory-Augmented Fine-Tuning (MAM)
- **Paper**: Not a single paper, but related to:
  - RAG (Lewis et al. 2020) [arXiv:2005.11401]
  - RETRO (Borgeaud et al. 2022) [arXiv:2112.04426]
  - Memorizing Transformers (Wu et al. 2022) [arXiv:2203.08932]
- **Core idea**: Instead of modifying weights, store knowledge in external memory and attend to it during inference. Memorizing Transformers use kNN-augmented attention over a growing key-value store.
- **Application to FerrisRes**: ConceptMap already does this! Retrieve relevant concepts, inject as context. This is weight-free continual learning. The model's "knowledge" grows without touching weights.

### 5. MAML (Model-Agnostic Meta-Learning)
- **Paper**: Finn et al. (2017) [arXiv:1703.03400]
- **Core idea**: Learn initial weights θ such that few gradient steps on a new task produce good performance. θ' = θ - α ∇L_task(θ). Meta-optimize θ for fast adaptation.
- **Application to FerrisRes**: Train the model with MAML objective so it's predisposed to quick adaptation when the "learn" tool triggers a gradient step.

### 6. Reptile (Simpler MAML)
- **Paper**: Nichol et al. (2018) "On First-Order Meta-Learning Algorithms" [arXiv:1803.02999]
- **Core idea**: θ ← θ + ε(θ_task - θ) after each task. Much simpler than MAML, same effect.
- **Application**: Use Reptile during distillation so student is predisposed to quick LoRA adaptation.

### 7. Compositional Pattern Producing Networks (CPPN)
- **Paper**: Stanley (2007) "Compositional Pattern Producing Networks"
- **Application**: Tool could generate CPPN-like weight perturbations — compact encoding of weight changes via procedural generation.

### 8. 8-bit Optimizers (bitsandbytes)
- **Paper**: Dettmers et al. (2022) "8-bit Optimizers via Block-wise Quantization" [arXiv:2110.02861]
- **Application**: When doing on-device LoRA updates, use 8-bit Adam to minimize memory footprint. Critical for Intel iGPU (256MB buffer limit).

## Recommended Approach for FerrisRes

### Strategy: Hybrid Memory-Augmented + Tool-Triggered LoRA

**Tier 1 (now)**: Memory-augmented (ConceptMap)
- Already works via cognitive pipeline
- Zero weight modification
- Scales infinitely (disk-backed)
- Limitation: no permanent skill change, only context augmentation

**Tier 2 (near-term)**: Tool-triggered LoRA patching
- "learn" tool receives (input, correct_output)
- Runs forward pass, computes loss
- Applies single LoRA gradient step to relevant layers
- Saves LoRA adapter alongside concept
- On next inference, loads accumulated LoRA adapters
- EWC penalty prevents catastrophic forgetting

**Tier 3 (long-term)**: Progressive adapter stacking
- Each major learning event adds a new LoRA adapter (rank 4-16)
- Old adapters frozen, new adapter trainable
- Model grows ~0.1% per learning event
- Memory: manageable with adapter offloading to CPU/disk

## Architecture: ToolTriggeredLora

```rust
struct ToolTriggeredLora {
    // LoRA layers for quick adaptation
    adapters: Vec<LoraLayer>,
    // Fisher information for EWC
    fisher_diag: Vec<f32>,
    // Learning rate for on-the-fly updates
    lr: f32,
    // Maximum adapters before consolidation
    max_adapters: usize,
}

impl ToolTriggeredLora {
    fn learn(&mut self, model: &mut BlockAttnResModel, input: &[f32], target: &[f32]) -> f32 {
        // 1. Forward pass
        let output = model.forward(input);
        // 2. Compute loss
        let loss = mse(&output, target);
        // 3. Compute gradients (single step)
        let grads = backward(&loss);
        // 4. Apply EWC penalty
        let penalized_grads = apply_ewc(grads, &self.fisher_diag);
        // 5. Update LoRA adapters only (not base weights)
        self.adapters.last_mut().unwrap().update(penalized_grads, self.lr);
        // 6. Return loss for quality tracking
        loss
    }
}
```

## Key References
1. [PNAS 114(13)] Kirkpatrick 2017 - Elastic Weight Consolidation
2. [arXiv:1606.04671] Rusu 2016 - Progressive Neural Networks
3. [arXiv:2106.09685] Hu 2021 - LoRA
4. [arXiv:2305.14314] Dettmers 2023 - QLoRA
5. [arXiv:1703.03400] Finn 2017 - MAML
6. [arXiv:1803.02999] Nichol 2018 - Reptile
7. [arXiv:2203.08932] Wu 2022 - Memorizing Transformers
8. [arXiv:2110.02861] Dettmers 2022 - 8-bit Optimizers
