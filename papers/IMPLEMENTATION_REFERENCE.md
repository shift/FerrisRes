# Tokenization Implementation Reference
# Based on 2026 Research Papers

## 1. BLT Entropy Patching Algorithm (Primary)

### Core Equation
```
H(x_i) = Σ_v p_e(x_i = v | x_<i) * log(p_e(x_i = v | x_<i))
```
Where p_e is a small byte-level language model trained on the training data.

### Patch Boundary Detection
Two methods:
1. **Global Constraint**: H(x_t) > θ_g (entropy exceeds global threshold)
2. **Approx. Monotonic Constraint**: H(x_t) - H(x_t-1) > θ_r

### Implementation Steps
1. Train small byte-LM on training data (e.g., 2-layer transformer)
2. Compute next-byte entropies for each position
3. Identify patch boundaries where entropy exceeds threshold
4. Minimum one non-space-like byte per patch required

### Architecture Components
- **Local Encoder**: lightweight transformer (l_E << l_G layers) - maps bytes to patch representations
- **Latent Transformer**: large autoregressive transformer over patches
- **Local Decoder**: decodes patch representations back to bytes
- Uses cross-attention after each layer to pool byte representations

### Inference FLOP Reduction
- BLT achieves up to 50% fewer FLOPs vs BPE tokenization
- Model size can scale while maintaining fixed inference budget
- Patch size ps=6 or ps=8 recommended (BLT Entropy models)

---

## 2. Learnable End-to-End Tokenization (Secondary)

### Score Function Estimator
Directly optimizes discrete token boundaries to minimize loss:
```
∇_boundary E[loss] ≈ E[score_function_estimator]
```

### Key Techniques
- Time discounting (from RL) to reduce variance
- Autoregressive U-net architecture
- Straight-through estimators as baseline

### Comparison to BLT
- BLT uses pre-computed entropy thresholds
- Learnable Tokenization learns boundaries during training
- More computationally expensive but potentially better boundaries

---

## 3. SMILES Pair Encoding (Domain-Specific)

### Key Findings
- 16,800 specialized molecular tokens
- 45% token reduction for chemical structures
- Near-zero to 2,500+ exact matches improvement

### Implementation
- Extract common substructures from training corpus
- Add to vocabulary as specialized tokens
- Extend without modifying core tokenizer

---

## 4. MergeDNA Dynamic Tokenization (Domain-Specific)

### Approach
- Context-aware token merging using differentiable operations
- Variable chunk sizes: 15bp (dense) to 320bp (repetitive)
- Uses local similarity for chunk boundary decisions

---

## FerrisRes Implementation Priority

### Phase 1: BPE Subword Tokenizer
- Replace SimpleTokenizer with BPE
- Configurable vocab size (16K-64K)
- See: src/model/tokenizer.rs

### Phase 2: Entropy-Based Patching (BLT-Aligned)
- Implement entropy thresholding
- Dynamic patch boundaries
- Integrate with Block AttnRes intra-block attention

### Phase 3: Vocabulary Extension API
- Domain-specific token injection
- SMILES-PE style extensions
- genomics extensions

### Phase 4: Learnable Tokenization (Future)
- RL-based boundary learning
- Score function estimators

---

## Key Hyperparameters

### BLT Recommended Settings
- Patch size (ps): 6-8 bytes
- Local Encoder: 2-4 layers
- Latent Transformer: matches target model size
- Entropy threshold θ_g: tunable (typically 2-3 bits)
- n-gram embeddings: 2-4 bytes

### For FerrisRes Integration
- Block size currently: 8 tokens
- Replace with entropy-based boundary detection
- Local Encoder maps to existing intra-block pass
- Latent Transformer maps to inter-block pass

---

## References
- BLT Paper: arxiv.org/abs/2412.09871
- Learnable Tokenization: arxiv.org/abs/2602.13940
- SMILES-PE: arxiv.org/abs/2511.14365
- MergeDNA: arxiv.org/abs/2511.14806

Extracted from: papers/blt_extracted.md, papers/learnable_tokenization_extracted.md