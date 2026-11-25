# Loop policy: passes chosen by hardware and workload

Status: **designed.** The loop is currently a fixed config value
(`prelude/core/passes/coda`); nothing selects it at runtime.

## Two knobs, not one

- **Core size** is a training decision. It is the weights: a 2-layer 1024-wide
  core is ~5.9 MB and cannot be made smaller at load time. The hardware choice
  is *which tier* to load: Nano's core (~1.5 MB) fits a Pi 5's 2 MB L3; Small's
  (~4.6 MB) fits the i5's 6 MB; Base's (~5.9 MB) fits the i5 only.
- **Passes** are a runtime decision. The same checkpoint can run the core once,
  twice or three times. That is the knob this document is about.

## Why passes are not free

Each pass re-reads the core from memory unless it is cache-resident. At 6.3 GB/s
a 5.9 MB core costs ~0.94 ms per pass; if it fits L3, passes 2 and 3 are
near-free and only add compute. If it does not fit, every pass is a full read
and the loop is slower than more unique layers.

The NeurIPS 2026 analysis adds the second constraint: a looped layer converges
to a fixed point, and passes after convergence repeat the same computation. So
the useful pass count is bounded by both cache and convergence.

## Inputs

| input | source |
|---|---|
| core bytes | checkpoint config: core layers × per-layer bytes |
| L3 bytes | **measured at startup** by a pointer-chase probe (wgpu does not expose cache size), or a table by adapter name |
| bandwidth | `DeviceProfile` / a streaming probe |
| compute | `DeviceProfile` (FP32/FP16 estimate) |
| latency budget | the request: interactive chat, background task, batch |
| quality floor | the tier's minimum passes from evaluation |

A pointer-chase probe costs milliseconds and is more honest than a hard-coded
table. Record the measured value in the device evidence.

## Policy

```
fits_l3   = core_bytes <= 0.8 * l3_bytes
p_cache   = if fits_l3 { max_passes } else { 1 or 2 }
p_latency = passes_that_meet(latency_budget)
p_quality = min_passes_for(tier)
passes    = clamp(min(p_cache, p_latency), p_quality, max_passes)
```

Then, if the checkpoint declares a convergence exit, run up to `passes` and
stop early when `‖h_p − h_{p−1}‖ / ‖h_{p−1}‖ < ε`. The exit is deterministic
for a given input, and the actual count is recorded.

Examples:

| device | core | L3 | passes |
|---|---:|---:|---|
| Pi 5 | 1.5 MB (Nano) | 2 MB | 2–3, cache-resident |
| Pi 5 | 5.9 MB (Base) | 2 MB | 1 (do not load Base for latency) |
| i5-6500T | 5.9 MB (Base) | 6 MB | 3 (fits, marginal) |
| Strix Halo | 5.9 MB (Base) | 32 MB+ | 3, cache-resident; compute-bound |
| DGX Spark | 5.9 MB (Base) | large | 3 |

## Training requirement

A pass count chosen at runtime is only safe if the model was trained for it:

- sample passes per batch from `1..=max_passes` (or a curriculum that starts at
  max and anneals), so every count is in-distribution;
- a per-pass step embedding or gate, so the model knows which pass it is on;
- evaluate the quality curve at 1, 2 and 3 passes; require 1 pass not to
  collapse, and 3 passes to be no worse than 2 beyond noise.

Without this, running fewer passes is a gamble. The ICML 2026 result that
inference-time looping helps even untrained models is encouraging, not a
substitute for training the knob.

## Capability matrix

`ModelFeatures` gains `loop: { core_layers, max_passes, min_passes, step_embedding, exit }`.
`resolve(model, device, workload)` returns the pass count and the reason:

```
loop_passes: 2 (core 5.9 MB > 0.8 × L3 2.0 MB; bandwidth 6.3 GB/s)
```

The `capabilities` command prints the recommendation for the detected device,
so the policy is observable rather than buried in a scheduler.

## Evidence and determinism

- The chosen passes, the reason, the measured L3 and the convergence exit are
  recorded per run.
- A device-based count is deterministic per machine; a convergence exit is
  deterministic per input. Both are reproducible.
- Changing the pass count changes the output distribution; the evidence record
  is what makes a comparison fair.

## Tests

| test | asserts |
|---|---|
| policy table | the five device/core rows above resolve to the stated passes |
| cache probe | measured L3 is within 2× of the adapter's known value on this host |
| clamp | latency and quality floors cannot invert the range |
| training | sampled-pass training produces a monotone-ish quality curve; 1 pass does not collapse |
| evidence | the pass count and reason are written to the run record |
