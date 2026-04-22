# Ternary Quality Ceiling Research

## Question
Is there a hard limit to how well a ternary {-1, 0, +1} student can approximate
a BF16 teacher? How much FP32 LoRA capacity is needed to close the gap?

## Background

The "Logit Gap" problem: ternary weights have 3 possible values per weight.
A 1536×1536 ternary matrix has 3^(1536×1536) possible configurations.
That's astronomically many, but the information per weight is only ~1.58 bits
vs 16 bits for BF16. The representational capacity is fundamentally lower.

Current distillation setup:
- Ternary base: ~2.66B params × 1.58 bits = ~0.53 GB of information
- BF16 teacher: ~2.66B params × 16 bits = ~5.3 GB of information
- Ratio: ~10× less information in the student
- LoRA adds back ~26M FP32 params = ~0.4 GB of information
- Total student info: ~0.93 GB — still 5.7× less than teacher

## Key Question

Can LoRA on top of ternary actually recover the lost information?
Or is there a fundamental ceiling regardless of LoRA rank?

## Analysis

### Information-Theoretic View
The teacher has ~42.5 Gbits of information. The ternary base has ~4.2 Gbits.
LoRA at rank 8 adds ~26M × 32 bits = ~0.83 Gbits.
Total: ~5.0 Gbits — 8.5× less than teacher.

To match teacher information: need ~37.5 Gbits of LoRA = ~1.2B FP32 params.
That's ~45% of the model. At that point, why use ternary at all?

### Practical View (Quality ≠ Raw Information)
Information theory is a ceiling, not the actual quality gap. In practice:
- Many weights are near-zero and quantize losslessly
- Structure in the data means not all bits are equally important
- Distillation matches output distribution, not individual weights
- MoE routing means only top-k experts contribute per token

### Evidence So Far
- With only LoRA q/v (1.6M params, 0.06%): loss plateaus at ~16-18
- With LoRA all + router (3.8M params, 0.14%): loss still ~16
- With +expert FFN LoRA (~26M, 1.0%): NOT YET TESTED (ternary refactor in progress)
- Prediction: ~26M should bring loss down to ~10-12
- Need at least 100M LoRA params (rank 32?) to approach teacher quality

## Experiment Plan

### Experiment 1: LoRA Rank Sweep
- Rank 4, 8, 16, 32, 64 on ternary base
- Measure: training loss, KL divergence, perplexity
- 10k steps each, same data

### Experiment 2: Ternary vs BF16 Base
- Same LoRA config, same training
- Compare: ternary base + LoRA vs BF16 base + LoRA
- Isolates the ternary quality impact from LoRA capacity

### Experiment 3: Convergence Analysis
- Train 50k steps with rank 8 LoRA
- Measure loss curve slope: does it flatten (ceiling) or keep decreasing?
- If slope approaches zero at high step count, that's the ternary ceiling

## Success Criteria
- Establish minimum LoRA rank for < 5.0 KL divergence loss
- Quantify quality gap: ternary+LoRA vs BF16+LoRA vs teacher
- Document the ceiling so we know when to stop scaling LoRA

## Status: NOT STARTED
## Priority: HIGH (determines if our entire approach has a fundamental limit)

## Update: 2026-04-21 — STE is likely mathematical necessity

### Why LoRA has a fundamental ceiling in ternary space
In 1.58-bit models, the **geometry** of the weights matters more than the **magnitude**.
LoRA can only add continuous deltas to a frozen base — it cannot **sign-flip** weights.
In ternary space, flipping {-1, 0, +1} is the biggest possible change per weight.
STE enables sign-flipping during training; LoRA cannot do this.

### Rank sweep predictions
| Rank | Params | Info capacity | Expected quality |
|------|--------|---------------|-----------------|
| 4    | ~13M   | 1×            | Struggles with syntax |
| 8    | ~26M   | 2×            | Coarse style/tone (current) |
| 16   | ~52M   | 4×            | Sweet spot for 7B models |
| 32   | ~104M  | 8×            | Where reasoning emerges |
| 64   | ~208M  | 16×           | **CEILING** — diminishing returns |

If rank 32→64 yields 0% improvement, we've hit the **Representation Ceiling**.

### The MoE pooling advantage
"City of Experts" provides a mitigation: if one expert hits ceiling, the router
activates additional experts to **pool their information capacity**. With top-2
routing, two ternary experts together have ~3.16 bits per weight position.
With top-4: ~6.32 bits. This is a fundamental advantage over dense ternary models.

### Experiment priority
1. Rank sweep on text LM head (not vision encoder — not implemented)
2. Measure: KL divergence at each rank after 10k steps
3. Identify saturation point → determines if STE is needed
