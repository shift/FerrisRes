# Attention metadata prerequisite (A16)

Task: `a8800f37-3493-4c7d-bc38-1ce8fa936ef9`. Decision: ADR-029.

The old dispatcher guessed head counts from projection width (a square 4096-wide Q projection became 4096 heads). Those helpers were removed with the A03 fail-closed mitigation. New `model::attention_metadata::AttentionMetadata` provides a validated replacement API for future import:

- Read HF `config.json` or an explicitly selected JSON text configuration.
- Read GGUF keys under the actual `general.architecture` prefix.
- Require positive hidden size and attention head count; do not guess from tensors.
- Preserve explicit KV-head counts and independent head dimensions.
- Default *absent* KV count to MHA; derive absent head dimension only by exact division.
- Reject malformed/present invalid values, dimension overflow and unequal GGUF key/value head dimensions.
- Validate canonical `[out_features, in_features]` Q/K/V tensor shapes separately. Reverse GGUF dimension order first; native GPU Linear weights use `[in, out]` and still require proper import conversion.

`standard_config()` only converts representable equal-width MHA attention into the current GPU configuration. GQA/MQA, independent query widths and odd RoPE head dimensions return `Unsupported`, rather than silently losing metadata. The caller supplies FFN settings and must separately validate biases, normalization, positional settings and the complete architecture.

Nine tests cover square Q matrices, HF file reading, GQA/MQA, explicit dimensions, malformed metadata, public-field revalidation, GGUF defaults and shape/layout mismatches. Final Nix build, full suite (**1,680 passed**, six unrelated ignored), all-target Clippy and whitespace checks passed. An initial combined validation timed out; a rerun completed. Clippy caught a newer-than-MSRV helper, replaced with validated modulo checks.

**This is not restored checkpoint loading.** Generic `AnyModel` imports remain `Unsupported`; A03 still needs checkpoint-complete model representation, integration of metadata validation, complete tensor mapping and numerical reference parity. No commits were made.
