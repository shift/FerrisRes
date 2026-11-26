//! The CLI training path (`training::train_loop::run`) must actually learn.
//!
//! The scaffold this replaces allocated zeroed logits, computed a loss and
//! never called backward or `optimizer.step`, so a green compile and a finite
//! loss proved nothing. These tests drive the same public function `cmd_train`
//! calls, on a fixed sequence, with a fixed seed.

use ferrisres::training::f32_trainer::F32Config;
use ferrisres::training::train_loop::{run, TrainOptions};
use std::path::PathBuf;

fn tiny_config() -> F32Config {
    F32Config {
        hidden: 16,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 8,
        intermediate: 32,
        vocab: 32,
        max_seq: 16,
        rope_theta: 10_000.0,
        eps: 1e-6,
    }
}

/// Period-4 pattern: the next token is a deterministic function of the current
/// one, so a correct loop can drive the loss well below the initial value.
fn fixed_tokens() -> Vec<u32> {
    // seq_len + 1 tokens: one window of 16 inputs and 16 shifted targets.
    (0..17).map(|t| ((t % 4) + 1) as u32).collect()
}

fn temp_checkpoint(name: &str) -> PathBuf {
    std::env::temp_dir()
        .join(format!("ferrisres-train-learns-{}-{}", std::process::id(), name))
        .join("checkpoint.bin")
}

fn options(epochs: u32, output: PathBuf) -> TrainOptions {
    TrainOptions {
        config: tiny_config(),
        seq_len: 16,
        batch_size: 1,
        epochs,
        learning_rate: 1e-2,
        seed: 42,
        checkpoint_every: 0,
        output,
        resume: None,
        log_every: 0,
    }
}

#[test]
fn train_loop_reduces_loss_and_changes_parameters() {
    let output = temp_checkpoint("learns");
    let report = run(&options(300, output.clone()), &fixed_tokens()).expect("training failed");

    println!(
        "initial_loss={:.6} final_loss={:.6} steps={}",
        report.initial_loss, report.final_loss, report.steps_run
    );

    assert!(report.initial_loss.is_finite(), "initial loss is not finite");
    assert!(report.final_loss.is_finite(), "final loss is not finite");
    assert!(
        report.final_loss < 0.8 * report.initial_loss,
        "loss did not fall below 0.8x initial: initial={} final={}",
        report.initial_loss,
        report.final_loss
    );
    assert!(report.params_changed, "no parameter changed");
    assert_eq!(report.steps_run, 300);
    assert!(report.checkpoint.exists(), "final checkpoint was not written");

    let _ = std::fs::remove_dir_all(output.parent().expect("checkpoint has a parent"));
}

#[test]
fn resume_restores_the_recorded_step() {
    let output = temp_checkpoint("resume");
    let first = run(&options(5, output.clone()), &fixed_tokens()).expect("first run failed");
    assert_eq!(first.total_step, 5);

    let mut resumed_options = options(5, output.clone());
    resumed_options.resume = Some(output.clone());
    // Batch of two exercises gradient accumulation, which a single-window
    // step never reaches.
    resumed_options.batch_size = 2;
    let resumed = run(&resumed_options, &fixed_tokens()).expect("resume failed");

    println!(
        "resumed_from={} total_step={} initial_loss={:.6} final_loss={:.6}",
        resumed.resumed_from, resumed.total_step, resumed.initial_loss, resumed.final_loss
    );
    assert_eq!(resumed.resumed_from, 5, "resume did not report the saved step");
    assert_eq!(resumed.steps_run, 5);
    assert_eq!(resumed.total_step, 10);
    assert!(resumed.initial_loss.is_finite());

    let _ = std::fs::remove_dir_all(output.parent().expect("checkpoint has a parent"));
}
