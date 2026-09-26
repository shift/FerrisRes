# Learned standard-transformer layer normalization

Subtask: `b6426ceb-6c6d-4172-8bfb-e12fb572bc04`, under checkpoint representation `09197a72-e77e-43f3-abab-acb71503dd6c`. Decision: ADR-030.

Standard layers now own learned attention and FFN RMSNorm scale buffers. Both prefill and decode use those scales and `StandardTransformerConfig::norm_eps`.

- Scales initialize to one, preserving prior explicit-constructor behavior.
- `set_norm_weights(attention, ffn)` validates both arrays before writing either. Lengths must match hidden size; values must be finite. Signed and zero scales are allowed.
- `norm_weights()` exposes both buffers for readback/export.
- `norm_eps` must be positive, normal and finite. Subnormal epsilon is rejected because GPU arithmetic may flush it to zero.
- Existing standard constructors and dimension-only metadata conversion retain `1e-5`. **Checkpoint import must supply the actual checkpoint epsilon.** External config struct literals need the new field.
- Scales are effective multipliers. Architecture-specific offsets such as `1 + weight` must be applied explicitly by a future importer.

The shared weighted GPU kernel now takes a distinct epsilon uniform per dispatch, validates dimensions/buffer sizes, and retains `1e-6` through its legacy dispatch API.

Seven tests in `tests/standard_norm.rs` cover nonzero causal attention and FFN output against CPU references, signed/nonunit scales, tiny-input epsilon sensitivity, prefill/repeated decode, invalid upload non-mutation, invalid epsilon, and different epsilon dispatches in one encoder. All four initial behavior tests failed before implementation.

Validation: Nix build, **1,687 passing tests** (six unrelated audit reproductions ignored), all-target Clippy with warnings denied, and diff whitespace checks passed. No commits made.

**Remaining:** final model normalization, embedding/LM-head ownership, complete architecture settings and checkpoint tensor mapping/reference parity. A03 generic checkpoint imports remain `Unsupported`; this subtask does not complete the representation parent or restore loading.
