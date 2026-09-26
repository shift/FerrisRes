# Final output normalization and GPU embedding repairs

Checkpoint representation parent: `09197a72-e77e-43f3-abab-acb71503dd6c` (still unfinished).

## Final model RMSNorm

Task `519a628a-524e-45fb-9ac6-b60748338268` is complete.

`StandardTransformerModel` now owns a final RMSNorm scale initialized to ones, supports validated scale upload/export, and normalizes output with its configured epsilon. Zero-layer models also validate epsilon and dimensions. Output rows, byte sizes and device buffer limits are checked before allocation.

`AnyModel::finalize_hidden_states` applies this normalization for Standard models and preserves BlockAttnRes identity behavior. Unified generation invokes it exactly once before the LM head in all four paths: synchronous/streaming prefill/decode. Prefill first extracts the final token position.

Five GPU tests cover one/multiple rows, signed learned scales, invalid-upload non-mutation, invalid epsilon and row sizes, and BlockAttnRes identity. All three initial standard-model tests failed before implementation. These tests isolate the finalization API; complete trained-generator parity is not yet certified.

## Token embedding kernel

Task `048e6f92-f823-42f4-b76b-68de05949ef0` is complete.

The shader previously read vocabulary size and hidden dimension from uninitialized private variables (both zero), while the host wrote unused immediates. Uploaded nonzero embeddings therefore produced no output, and devices without IMMEDIATES could not create the pipeline.

Dimensions and batch size now arrive through a distinct uniform buffer per dispatch. Invocations beyond the batch return before accessing token IDs. Out-of-vocabulary IDs produce zero rows rather than stale output; callers requiring strict token validation must still reject invalid IDs. Constructors/dispatches validate dimensions, arithmetic, buffers and device limits. A zero batch is a no-op.

Three regressions reproduce the original failures and now pass, including a device without IMMEDIATES, signed uploaded weights, two batch sizes in one encoder, invalid IDs and untouched output beyond the batch.

## Validation and remaining work

Nix build, **1,695 passing tests** (six unrelated audit reproductions ignored), all-target Clippy with warnings denied, and diff whitespace checks passed. No commits were made.

**A03 remains unfinished and generic AnyModel imports remain Unsupported.** Next: embedding/LM-head ownership and validated weight handoff, complete architecture settings and checkpoint mapping, then reference checkpoint/generation parity. The unified readback helper's `Poll` followed by blocking receive also needs investigation before relying on end-to-end generation tests.
