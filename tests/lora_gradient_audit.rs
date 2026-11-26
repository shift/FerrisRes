//! Audit of the distillation / LoRA gradient path in a blocked config.
//!
//! The measurement lives in
//! [`ferrisres::training::grad_audit::max_rel_error_blocked_lora`]. The ignored
//! test is the tight gate; the non-ignored test proves the blocked + LoRA
//! forward trains and that its gradient stays inside the audit tolerance.

/// The blocked-config LoRA gradient must match a central finite difference.
///
/// Kept `#[ignore]` because it runs the finite-difference audit, which is too
/// slow for the default suite; the number is also exposed by
/// `examples/grad_audit.rs` for the harness gate.
#[test]
#[ignore = "runs the finite-difference audit; also measured by examples/grad_audit.rs"]
fn lora_gradient_matches_finite_difference_in_blocked_config() {
    let max_rel = ferrisres::training::grad_audit::max_rel_error_blocked_lora();
    assert!(
        max_rel <= 1e-2,
        "distillation LoRA gradient is wrong in a blocked config: max relative error {:.3e}",
        max_rel
    );
}

/// `forward_train` must run (no panic) when blocks and LoRA are both active,
/// and the gradient it is measured against must be within tolerance.
#[test]
fn forward_train_trains_with_blocks_and_lora() {
    let model = ferrisres::training::grad_audit::blocked_lora_model();
    let out = model.forward_train(&[1, 2, 3, 4]);
    assert_eq!(out.logits.len(), 4 * 8);
    assert!(out.activations.len() == model.num_layers);

    let max_rel = ferrisres::training::grad_audit::max_rel_error_blocked_lora();
    assert!(
        max_rel <= 1e-2,
        "blocked-config LoRA gradient above tolerance: max relative error {:.3e}",
        max_rel
    );
}
