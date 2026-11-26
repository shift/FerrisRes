//! Print the blocked-config LoRA gradient audit as a gate metric.
//!
//! The harness gate consumes higher-is-better metrics, so stdout carries the
//! quality `1 - min(error, 1)` and the raw error goes to stderr. The process
//! exits 0 regardless of the value: the *number* is the signal, not the exit
//! status.

fn main() {
    let max_rel_error = ferrisres::training::grad_audit::max_rel_error_blocked_lora();
    let quality = 1.0 - max_rel_error.min(1.0);

    println!("metric=grad_quality value={}", quality as f64);
    eprintln!(
        "distillation LoRA gradient in a blocked config: max_rel_error={:.6e}",
        max_rel_error
    );
}
