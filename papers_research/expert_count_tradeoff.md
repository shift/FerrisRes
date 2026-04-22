# Expert Count × Precision Tradeoff Research

## Question
For ternary MoE distillation, what's the optimal expert count and per-expert
size? 4 full-size experts (6144/12288 inter_dim) vs 16 small experts (1536/3072
inter_dim) vs 128 tiny experts — all with same total parameter budget.

## Background

Current setup: 2 or 4 full-size experts per layer. Each expert has the same
intermediate dimension as the original dense FFN (6144 for layers 0-14,
12288 for layers 15-34). This means:
- 4 experts: ~7.3B params, ~3.7 GB ternary
- 2 experts: ~4.2B params, ~2.1 GB ternary

The user asked about 128 experts to compensate for ternary precision loss.
But 128 full-size experts would be ~234B params — impossible. So 128 experts
requires much smaller per-expert intermediate dims.

## Parameter Budget Analysis

For a single layer with inter_dim=12288, hidden_dim=1536:
- Dense FFN: 12288 × 1536 × 3 = 56.6M params
- 4 full experts: 56.6M × 4 = 226M params
- 16 half experts: (6144 × 1536 × 3) × 16 = 226M params (same total!)
- 128 quarter experts: (3072 × 1536 × 3) × 128 = 226M params (same total!)

So same parameter budget, very different routing behavior.

## Key Tradeoffs

| Factor | Few Large Experts | Many Small Experts |
|--------|------------------|-------------------|
| Specialization | Coarse (4 domains) | Fine (128 micro-domains) |
| Routing entropy | Lower (easier to collapse) | Higher (more choices) |
| Ternary quality | Each expert has full precision path | Each expert is very lossy |
| Top-k overlap | top-2 = 50% of experts | top-2 = 1.6% of experts |
| Memory per expert | Large (~7MB ternary) | Small (~0.2MB ternary) |
| Expert mmap viable | ~7ms on USB SSD | ~0.2ms on USB SSD |
| Load balance | Easier (4 buckets) | Harder (128 buckets) |

## Unknown / Needs Investigation

1. **Ternary quality vs expert size**: A 1536×3072 ternary matrix has very low
   information density (~12KB of actual information). Can it represent meaningful
   transformations? Or does ternary need larger matrices to average out noise?

2. **Routing with 128 experts**: With top-2 of 128, each token uses 1.6% of
   experts. This is extremely sparse. Does this cause training instability?
   Do we need higher top-k (top-4 or top-8)?

3. **Load balance at scale**: The balance loss `ne × Σ(f_i × P_i)` becomes
   harder to optimize with 128 experts. Do we need per-expert learning rates?
   Or auxiliary losses beyond balance?

4. **Distillation quality**: Does the teacher's knowledge distribute better
   across many small experts or few large ones? Hypothesis: many small experts
   specialize in narrow patterns (syntax, specific domains), while few large
   experts capture broader patterns (general reasoning).

## Experiment Plan

### Experiment 1: Quality vs Count (Fixed Budget)
- Train 3 student configurations: 4×12288, 16×3072, 64×768
- All same total params (~226M per layer)
- Measure: KL loss, routing entropy, downstream quality
- On Colab T4 with 10k steps

### Experiment 2: Mmap Viability
- Benchmark expert loading: 4×7MB vs 128×0.2MB on USB SSD and NVMe
- Measure: per-token latency with mmap expert loading
- On Raspberry Pi 4 with USB SSD

### Experiment 3: Routing Dynamics
- Visualize expert specialization at 4 vs 16 vs 64 experts
- Measure: expert overlap, routing stability across similar inputs
- Use cosine similarity of expert outputs as specialization metric

## Status: NOT STARTED
## Priority: MEDIUM (current 2-4 expert setup works, this is optimization)
