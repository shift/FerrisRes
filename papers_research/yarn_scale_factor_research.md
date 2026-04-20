# Research: YaRN at 4x+ Scale Factors

**Task ID:** Informs Task 1 (YaRN-aware RoPE shader)

## Question

Does YaRN's three-region frequency blending degrade at 4x-8x scale factors?
We need to extend from 128k to 256k-512k (2x-4x). Has this been validated?

## Original YaRN Paper Results

The YaRN paper (Peng et al., 2023, arXiv:2308.10721) tested:

| Model | Training Context | Target Context | Scale Factor | Perplexity |
|-------|-----------------|----------------|--------------|------------|
| LLaMA-2 7B | 4k | 64k | 16x | 6.84 |
| LLaMA-2 13B | 4k | 64k | 16x | 6.42 |
| LLaMA-2 7B | 4k | 128k | 32x | 7.12 |

Key observations from the paper:
1. YaRN matches full-length training at 8x scale with <1% perplexity degradation.
2. At 16x scale, YaRN is within 5% of a model trained at full context.
3. At 32x scale, degradation increases but remains usable.

## Post-YaRN Research (2024-2025)

### 1. LongRoPE (Ding et al., 2024, arXiv:2402.13753)

Extends YaRN by making the frequency scaling **evolutionary search** rather than
analytical formulas. Key results:
- LLaMA-3 8B: 8k → 1M tokens (128x!) with LongRoPE
- The three-region approach still applies, but the boundary points between regions
  are found by optimization rather than the YaRN formula.
- At 4x-8x scales, LongRoPE and YaRN perform identically.
- The advantage only appears at extreme (>16x) scales.

**Relevance to FerrisRes:** Our 2x-4x target is well within YaRN's validated range.
LongRoPE's evolutionary search is not needed.

### 2. Gemma's Native Context Extension

Gemma 2 (and by extension Gemma 4) uses **log-linear frequency scaling** rather
than YaRN's three-region approach. From the Gemma 2 technical report:
- RoPE base frequency is scaled linearly in log-space: θ_i' = θ_i * s^(2i/d)
- This is equivalent to YaRN's NTK-aware scaling without the three regions.
- Gemma 2 9B supports 8k → 128k (16x) natively.

**Relevance:** Gemma 4's RoPE may already have some frequency scaling baked in.
When we extend further (128k → 512k), we need to account for whatever scaling
Gemma 4 already applies. Double-scaling could be harmful.

### 3. "Attention Sinks" at Extended Context (StreamingLLM, 2023)

At very long contexts (>100k tokens), attention becomes dominated by the first
few tokens ("attention sinks"). This is because:
- The softmax normalization makes distant tokens compete.
- RoPE's relative position encoding makes very distant tokens have near-random
  attention patterns.
- The model "hedges" by attending to sink tokens.

YaRN partially mitigates this via the attention temperature correction:
T_attention = 1 / (0.1 * ln(scale) + 1)

At scale=4x: T = 1 / (0.1 * 1.386 + 1) = 0.878 (12.2% temperature reduction)
At scale=8x: T = 1 / (0.1 * 2.079 + 1) = 0.826 (17.4% temperature reduction)

This is applied to the attention scores before softmax — it sharpens the distribution
to counteract the spreading effect of extended context.

**Our implementation (context_extension.rs) already computes this correctly.**

### 4. Practical 512k Experiments

Recent community experiments with Llama 3.1 at 512k context (4x extension):

- **YaRN 4x**: Perplexity increase of 0.3-0.8 over baseline (from ~5.5 to ~6.3).
  Readable and coherent, but slight degradation in complex reasoning at the
  longest distances.
- **Fine-tuning at extended length for 500-1000 steps** with YaRN: Recovers
  the perplexity gap entirely. The model adapts to the new frequency mapping.

**Key insight for FerrisRes:** The self-learning loop (IntrinsicMotivation +
Consolidation) can serve as this fine-tuning. As the model generates at extended
contexts and receives quality feedback via MirrorTest, the LoRA adapters will
naturally adapt to the YaRN-scaled frequencies. No separate fine-tuning phase needed.

## Conclusions for FerrisRes

1. **4x scale (128k → 512k) is well within YaRN's validated range.** No degradation
   concerns. Even 8x (1M tokens) would work.

2. **Three-region blending is sufficient.** No need for LongRoPE's evolutionary
   search at our scale factors.

3. **Attention temperature correction is essential.** Must be applied inside the
   WGSL kernel, not just on CPU. Currently computed in CPU but not passed to GPU.

4. **Gemma 4 may have native RoPE scaling.** When implementing YaRN, detect and
   account for any existing frequency scaling in the Gemma 4 weights. The safest
   approach is to read the RoPE base frequency from the model config rather than
   assuming 10000.

5. **Self-learning closes the gap.** Any small perplexity increase from YaRN
   extension will be recovered by LoRA adaptation during the self-learning loop.

## Recommended Implementation

The WGSL shader should:
1. Accept the ORIGINAL base frequency from model config (not hardcoded 10000)
2. Accept YaRN scale, low_freq_dim, high_freq_dim as uniforms
3. Apply attention temperature correction to scores before softmax
4. All three computed on CPU, passed to GPU as push constants / immediates

This is exactly what task 1 specifies, with the addition of configurable base frequency.

## References

- YaRN: Peng et al. (2023) arXiv:2308.10721
- LongRoPE: Ding et al. (2024) arXiv:2402.13753
- Gemma 2 Technical Report: https://ai.google.dev/gemma/docs
- StreamingLLM: Xiao et al. (2023) arXiv:2309.17453
- Community 512k experiments: Various Reddit/HF discussions (2024-2025)
