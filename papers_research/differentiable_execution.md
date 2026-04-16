# Research: Differentiable Execution Through LLM-Computer

## Task ID: 5f860dbb-2e79-408b-b4e3-d814734fa7e4

## Key Papers & Techniques

### 1. Straight-Through Estimator (STE)
- **Paper**: Bengio et al. (2013) "Estimating or Propagating Gradients Through Stochastic Neurons" [arXiv:1308.3432]
- **Core idea**: In forward pass, use hard threshold (argmax). In backward pass, pretend the threshold was the identity function.
- **Gradient**: ∂L/∂z ≈ ∂L/∂hard_threshold(z) × 1 (identity derivative)
- **Application to CALM**: When the model selects a CALM op via argmax over op logits, STE passes gradient through as if the selection was soft. The model learns to select the right op.
- **Pros**: Simple, fast, no variance
- **Cons**: Biased gradient estimate

### 2. Gumbel-Softmax (Concrete Distribution)
- **Paper**: Jang et al. (2016) "Categorical Reparameterization with Gumbel-Softmax" [arXiv:1611.01144]
- **Core idea**: Replace argmax with softmax over logits + Gumbel noise, with temperature τ → 0 annealing to discrete.
- **Formula**: y_i = exp((log π_i + g_i) / τ) / Σ_j exp((log π_j + g_j) / τ)
- **Application to CALM**: Op selection becomes differentiable — each op gets a soft weight, memory read/write becomes weighted sum over all table entries.
- **Pros**: Reparameterizable, low variance, anneals to true discrete
- **Cons**: Still biased for any fixed τ > 0

### 3. REINFORCE / Score Function Estimator
- **Paper**: Williams (1992) "Simple statistical gradient-following algorithms"
- **Core idea**: ∇_θ E[f(z)] = E[f(z) ∇_θ log p(z|θ)]
- **Application to CALM**: Sample CALM programs, execute them, use reward (loss reduction) as f(z), backprop through program selection probabilities.
- **Pros**: Unbiased
- **Cons**: High variance, needs many samples

### 4. REBAR / RELAX (Variance Reduction)
- **REBAR**: Tucker et al. (2017) [arXiv:1703.07370] — Uses a control variate from the continuous relaxation to reduce REINFORCE variance.
- **RELAX**: Grathwohl et al. (2017) [arXiv:1711.00123] — "Backpropagation through the Void" — learns the control variate neural network.
- **Application to CALM**: Use Gumbel-Softmax as control variate for REINFORCE. Get unbiased + low-variance gradient estimates for discrete CALM program selection.

### 5. Neural Turing Machines (NTM)
- **Paper**: Graves et al. (2014) [arXiv:1410.5401]
- **Core architecture**: Controller network (LSTM/feedforward) + external memory matrix + differentiable read/write heads via content-based + location-based attention.
- **Read**: r_t = Σ_i w_t(i) M_t(i) — weighted sum over memory rows
- **Write**: M_t(i) = M_{t-1}(i) × (1 - w_t(i) e_t) + w_t(i) a_t — erase + add
- **Application to FerrisRes**: Replace CALM's discrete table_lookup/table_set with NTM-style soft read/write. The LlmComputer's tables become differentiable memory banks accessed via attention.

### 6. Differentiable Neural Computer (DNC)
- **Paper**: Graves et al. (2016) "Hybrid computing using a neural network with dynamic external memory" [Nature 538]
- **Improvements over NTM**: Dynamic memory allocation, temporal linking (tracks write order), robust to memory size changes.
- **Application**: LlmComputer could use DNC-style memory instead of fixed tables.

## Recommended Approach for FerrisRes

### Phase 1: STE + Gumbel-Softmax (low effort, immediate benefit)
1. Add op_logits field to LlmComputer — model emits logits for each CALM op
2. Forward: Gumbel-Softmax with τ annealing to select ops
3. Backward: STE for hard selections during training
4. This makes CALM program generation differentiable end-to-end

### Phase 2: NTM-style Memory (medium effort)
1. Replace table_lookup/table_set with attention-based soft read/write
2. Tables become memory matrices, keys become query vectors
3. Fully differentiable — gradients flow through memory access patterns

### Phase 3: REINFORCE + RELAX (high effort, best quality)
1. Sample full CALM programs from policy
2. Execute programs, compute reward (task loss reduction)
3. Use RELAX control variate for low-variance policy gradient
4. This is the gold standard but requires more compute

## Architecture: DifferentiableLlmComputer

```rust
struct DiffLlmComputer {
    // Op selection: logits over CALM ops
    op_selector: Linear(hidden_dim, num_ops),
    // Memory: NTM-style soft read/write
    memory: DiffMemoryBank,
    // Temperature for Gumbel-Softmax
    temperature: f32,
}

impl DiffLlmComputer {
    fn forward(&self, hidden: &[f32]) -> (VmState, Vec<f32>) {
        // 1. Select op via Gumbel-Softmax
        let op_logits = self.op_selector.forward(hidden);
        let op_weights = gumbel_softmax(&op_logits, self.temperature);
        let selected_op = argmax(&op_weights);

        // 2. Execute op (STE: forward is discrete, backward is soft)
        let result = execute_op_ste(selected_op, op_weights, &mut state);

        // 3. Return state + logits for loss computation
        (state, op_logits)
    }
}
```

## Key References
1. [arXiv:1308.3432] Bengio 2013 - Straight-Through Estimator
2. [arXiv:1611.01144] Jang 2016 - Gumbel-Softmax
3. [arXiv:1703.07370] Tucker 2017 - REBAR
4. [arXiv:1711.00123] Grathwohl 2017 - RELAX
5. [arXiv:1410.5401] Graves 2014 - Neural Turing Machines
6. [Nature 538, 2016] Graves 2016 - Differentiable Neural Computer
