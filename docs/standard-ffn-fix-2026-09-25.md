# Standard-transformer gated FFN repair

Audit **A17**: `e67822f4-dba7-4de7-ad01-9794dd3cf968`. Decision: ADR-028.

Prefill and decode previously computed `ReLU(gate) + up`. Both now share a helper computing `activation(gate) * up` before the down projection, using separate buffers to avoid read/write aliasing.

`StandardTransformerConfig::ffn_activation` explicitly selects:

- `GatedActivation::Silu`: SwiGLU, default in all constructors, including LLaMA/Mistral presets.
- `GatedActivation::GeluTanh`: gated GELU using the tanh approximation, not exact-erf GELU.

External configuration struct literals must supply the new field. Numerical results necessarily differ from the incorrect old formula. Constructors preserve their signatures.

Five tests in `tests/standard_ffn.rs` cover constructor defaults and both activations through multi-token prefill and repeated decode, compared with independent CPU layer formulas. Fixtures use signed, non-square, nonsymmetric FFN matrices. Attention weights are deliberately zeroed to isolate the FFN; this is **not** full attention or pretrained checkpoint parity certification. Both original SwiGLU tests failed before the repair.

Validation in the Nix dev shell: build passed; **1,671 tests passed**, six unrelated known-bug audit reproductions ignored; all-target Clippy with warnings denied passed; diff whitespace check passed. No commits made.

**A03 remains unfinished.** Generic AnyModel checkpoint imports still return `Unsupported`. Completing FFN semantics removes one prerequisite, but checkpoint-owned model representation, metadata validation, complete tensor mapping and reference checkpoint parity are still required. See [checkpoint mitigation](checkpoint-mitigation-2026-09-24.md).
