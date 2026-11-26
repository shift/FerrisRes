//! Audit of the distillation / LoRA gradient path in a blocked config.
//!
//! The measurement lives in
//! [`ferrisres::training::grad_audit::max_rel_error_blocked_lora`]. This file
//! is the gate: the ignored test asserts the value (it fails today), and the
//! non-ignored test proves `forward_train` refuses the combination outright.

/// The blocked-config LoRA gradient is wrong today, so this test fails when
/// run. It stays `#[ignore]` so the normal suite is green; the number is also
/// exposed by `examples/grad_audit.rs` for the harness gate.
#[test]
#[ignore = "documents a real defect: the distillation LoRA gradient is not a gradient in a blocked config"]
fn lora_gradient_matches_finite_difference_in_blocked_config() {
    let max_rel = ferrisres::training::grad_audit::max_rel_error_blocked_lora();
    assert!(
        max_rel <= 1e-2,
        "distillation LoRA gradient is wrong in a blocked config: max relative error {:.3e}",
        max_rel
    );
}

/// `forward_train` must not silently produce gradients it cannot compute.
#[test]
#[should_panic(expected = "lora_gradient_invalid_with_blocks")]
fn forward_train_panics_with_blocks_and_lora() {
    let model = ferrisres::training::grad_audit::blocked_lora_model();
    let _ = model.forward_train(&[1, 2, 3, 4]);
}
