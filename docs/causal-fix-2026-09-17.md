# Causal BlockAttnRes repair — 2026-09-17

Audit task **A01**: `1f821f56-be94-4e67-9176-8bb665be0236`.
Related cached-generation fixes: `f8fc88f9-b882-41f5-a831-7e127e801df6`.
Decision: **ADR-026**, `14aadb21-7318-4329-a81b-699885ec5492`.

## What changed

CPU BlockAttnRes previously averaged hidden states over the entire sequence and broadcast one block residual to all positions. Future tokens therefore changed earlier logits, including during training.

The new internal `BlockResidualState` preserves `[sequence, hidden]` rows in each depth-block snapshot. Each token attends only to its own depth history. Layer averages use the actual number of layers since the last boundary, including unequal-depth blocks. The initial embedding is a separate snapshot, not part of the first block's layer average.

The shared helper is used by full inference, training forward, routing collection, GPU-assisted full forward, prefill, all four model decode variants, and the standalone CPU token generator. Each decode step starts new depth state for the new token; temporal information remains in the KV cache. Legacy public cache summary fields remain for compatibility but are no longer used by model execution.

`inter_block_attention()` now requires each block snapshot to have shape `[seq, hidden]` and returns that same shape, rather than a single broadcast vector. External callers using the old pooled representation must migrate. Single-token depth attention delegates to the same implementation.

## Cached-generation defects exposed and fixed

- `cpu_attention_raw()` now treats queries as the suffix of cached keys. A one-token decode attends the whole allowed prefix, rather than only key zero. Full-sequence causal masking is preserved.
- `forward_prefill()` clears the previous request's cache before populating it.
- `CpuTokenGenerator` samples only the final prompt position's logits. Previously it could return an ID outside the vocabulary by sampling flattened multi-position logits.

## Regression coverage

`tests/causal_blocks.rs` uses a three-layer, nonzero Q/K/V/O fixture with two unequal-depth blocks. Its five tests check:

1. Full/training/routing prefix invariance and agreement.
2. Prefill plus repeated decode against full-forward logits.
3. Reusing a cache for a new request.
4. Attention over all cached prefix values.
5. Greedy standalone CPU generation against a full-forward reference.

All five failed before the fix and pass afterward. The original two minimal causal-leakage audit tests are now enabled by default.

## Validation

Inside the Nix development shell:

- `cargo build -q --message-format short` — passed.
- `cargo test --lib --tests -- --test-threads=1` — **1,664 passed**, six unrelated known-bug audit reproductions ignored.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `git diff --check` — passed.

Earlier A02 GPU-parameter and A04 Armor fixes remain intact. No commits were made.

## Compatibility and limits

**Numerical behavior necessarily changes.** Checkpoints trained using future-token leakage should be reevaluated and may require retraining; matching old leaked outputs is not a compatibility goal.

Nonzero cached/full parity tests cover the base CPU path, without optional PLE, LoRA or KV-sharing configurations. GPU-assisted variants use the same corrected block-state helper, but their complete projection/adapter parity is not certified by these tests. Independent gradient-math, optional-path and model-loading defects remain separate work.

**A03 (discarded pretrained weights in `AnyModel` loaders) is still the remaining open critical audit item.** This repair is not a production-readiness certification.
