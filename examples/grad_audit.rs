//! Print the blocked-config LoRA gradient audit as a gate metric.
//!
//! The harness gate consumes higher-is-better metrics. The audit passes when
//! the max relative error against a central finite difference is at most
//! [`TOLERANCE`]; a passing audit reports `grad_quality` 1 and a failing one
//! reports 0. The raw error goes to stderr so a failure names the number.
//! The process exits 0 regardless of the value: the *number* is the signal,
//! not the exit status.

/// The blocked-config gradient is accepted when it is within 1% of the
/// central finite difference; the same bound is asserted by
/// `tests/lora_gradient_audit.rs`.
const TOLERANCE: f32 = 1e-2;

fn main() {
    let max_rel_error = ferrisres::training::grad_audit::max_rel_error_blocked_lora();
    let quality = if max_rel_error <= TOLERANCE { 1.0 } else { 0.0 };

    println!("metric=grad_quality value={quality}");
    eprintln!(
        "distillation LoRA gradient in a blocked config: max_rel_error={max_rel_error:.6e} (tolerance {TOLERANCE:.0e})"
    );
}
