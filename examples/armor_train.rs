//! Train the Armor L1 encoder on the generated licensed corpus and report
//! honest dev/test numbers.
//!
//! ```bash
//! cargo run --release --example armor_train -- \
//!   --train assets/armor/build/train.jsonl \
//!   --dev assets/armor/build/dev.jsonl \
//!   --test assets/armor/build/test.jsonl \
//!   --epochs 1 --max-tokens 128 --out /tmp/opencode/armor-full
//! ```
//!
//! The operating point is chosen on **dev** at FPR <= 1 % and then applied
//! unchanged to **test**; test is scored once. `--out` is a directory; the
//! FRAL artefact is `<out>/armor-l1.fral` and the metrics are
//! `<out>/report.json`.
//!
//! `--train-limit` / `--dev-limit` take a deterministic shuffled prefix. They
//! exist for a fast plumbing read; the full run uses every example.
//!
//! `--fixed-length` trains every batch at `--max-tokens` with a length mask
//! instead of sampling a shorter scan. The sampled-length default is why the
//! first full run collapsed at 128 tokens: the model learned a short-prefix cue
//! that mean pooling diluted at the evaluation length.
//!
//! `--probe` prints train and dev AUROC in memory and after the FRAL
//! round-trip; `--skip-test` leaves test unevaluated when dev does not clear
//! the gate.

use std::path::PathBuf;
use std::time::Instant;

use ferrisres::security::armor_l1::{ArmorGuardConfig, L1Scanner, L1TokenizerSpec};
use ferrisres::training::armor_jsonl::load_labeled;
use ferrisres::training::{
    operating_point, score_items_at, ArmorEncoder, ArmorTrainerConfig, OperatingPoint,
    SplitMix64,
};

struct Args {
    train: PathBuf,
    dev: PathBuf,
    test: PathBuf,
    epochs: usize,
    max_tokens: usize,
    out: PathBuf,
    train_limit: Option<usize>,
    dev_limit: Option<usize>,
    probe: bool,
    fixed_length: bool,
    skip_test: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage: armor_train --train <jsonl> --dev <jsonl> --test <jsonl> \
         [--epochs N] [--max-tokens N] [--out DIR] [--train-limit N] [--dev-limit N]          [--probe] [--fixed-length] [--skip-test]"
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut args = Args {
        train: PathBuf::new(),
        dev: PathBuf::new(),
        test: PathBuf::new(),
        epochs: 1,
        max_tokens: 128,
        out: PathBuf::from("armor-out"),
        train_limit: None,
        dev_limit: None,
        probe: false,
        fixed_length: false,
        skip_test: false,
    };
    fn take(it: &mut std::vec::IntoIter<String>) -> String {
        it.next().unwrap_or_else(|| usage())
    }
    let mut it = std::env::args().skip(1).collect::<Vec<_>>().into_iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--train" => args.train = PathBuf::from(take(&mut it)),
            "--dev" => args.dev = PathBuf::from(take(&mut it)),
            "--test" => args.test = PathBuf::from(take(&mut it)),
            "--epochs" => args.epochs = take(&mut it).parse().unwrap_or_else(|_| usage()),
            "--max-tokens" => {
                args.max_tokens = take(&mut it).parse().unwrap_or_else(|_| usage())
            }
            "--out" => args.out = PathBuf::from(take(&mut it)),
            "--train-limit" => {
                args.train_limit = Some(take(&mut it).parse().unwrap_or_else(|_| usage()))
            }
            "--dev-limit" => {
                args.dev_limit = Some(take(&mut it).parse().unwrap_or_else(|_| usage()))
            }
            "--probe" => args.probe = true,
            "--fixed-length" => args.fixed_length = true,
            "--skip-test" => args.skip_test = true,
            _ => usage(),
        }
    }
    if args.train.as_os_str().is_empty()
        || args.dev.as_os_str().is_empty()
        || (args.test.as_os_str().is_empty() && !args.skip_test)
    {
        usage();
    }
    args
}

/// Deterministic shuffled prefix, so a limited run is not dominated by the
/// first family in file order.
fn take_limit(items: Vec<(String, u8)>, limit: Option<usize>, seed: u64) -> Vec<(String, u8)> {
    match limit {
        Some(n) if n < items.len() => {
            let mut items = items;
            SplitMix64::new(seed).shuffle(&mut items);
            items.truncate(n);
            items
        }
        _ => items,
    }
}

fn tokenize_split(
    encoder: &ArmorEncoder,
    items: &[(String, u8)],
    max_tokens: usize,
) -> Vec<(Vec<u32>, u8)> {
    items
        .iter()
        .filter_map(|(text, label)| {
            let toks = encoder.tokenize(text, max_tokens);
            if toks.is_empty() {
                None
            } else {
                Some((toks, *label))
            }
        })
        .collect()
}

/// Confusion counts at a threshold chosen elsewhere (threshold is not tuned
/// here).
fn confusion_at(scores: &[(f32, u8)], threshold: f32) -> OperatingPoint {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_count = 0usize;
    for &(s, y) in scores {
        match (s >= threshold, y) {
            (true, 1) => tp += 1,
            (true, _) => fp += 1,
            (false, 1) => fn_count += 1,
            (false, _) => tn += 1,
        }
    }
    let n_pos = tp + fn_count;
    let n_neg = tn + fp;
    OperatingPoint {
        threshold,
        fpr: if n_neg > 0 { fp as f32 / n_neg as f32 } else { 0.0 },
        fnr: if n_pos > 0 {
            fn_count as f32 / n_pos as f32
        } else {
            0.0
        },
        accuracy: (tp + tn) as f32 / scores.len().max(1) as f32,
        true_positives: tp,
        false_positives: fp,
        true_negatives: tn,
        false_negatives: fn_count,
    }
}

fn op_json(op: &OperatingPoint, eval_tokens: usize) -> serde_json::Value {
    serde_json::json!({
        "threshold": op.threshold,
        "fnr": op.fnr,
        "fpr": op.fpr,
        "accuracy": op.accuracy,
        "true_positives": op.true_positives,
        "false_positives": op.false_positives,
        "true_negatives": op.true_negatives,
        "false_negatives": op.false_negatives,
        "n": op.true_positives + op.false_positives + op.true_negatives + op.false_negatives,
        "eval_tokens": eval_tokens,
    })
}

/// Rank AUROC with ties counted as 0.5.
fn auroc_parts(pos: &[f32], neg: &[f32]) -> f32 {
    if pos.is_empty() || neg.is_empty() {
        return f32::NAN;
    }
    let mut all: Vec<(f32, u8)> =
        pos.iter().map(|&s| (s, 1)).chain(neg.iter().map(|&s| (s, 0))).collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut rank_sum = 0.0f64;
    let mut i = 0usize;
    while i < all.len() {
        let mut j = i + 1;
        while j < all.len() && all[j].0 == all[i].0 {
            j += 1;
        }
        let avg = ((i + 1) as f64 + j as f64) / 2.0;
        for item in &all[i..j] {
            if item.1 == 1 {
                rank_sum += avg;
            }
        }
        i = j;
    }
    let n_pos = pos.len() as f64;
    let n_neg = neg.len() as f64;
    ((rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)) as f32
}

fn auroc_scores(scores: &[(f32, u8)]) -> f32 {
    let pos: Vec<f32> = scores.iter().filter(|(_, y)| *y == 1).map(|(s, _)| *s).collect();
    let neg: Vec<f32> = scores.iter().filter(|(_, y)| *y == 0).map(|(s, _)| *s).collect();
    auroc_parts(&pos, &neg)
}

fn bce_at(scores: &[(f32, u8)]) -> f32 {
    if scores.is_empty() {
        return f32::NAN;
    }
    let mut sum = 0.0f64;
    for &(s, y) in scores {
        let p = (s as f64).clamp(1e-6, 1.0 - 1e-6);
        sum += if y == 1 { -p.ln() } else { -(1.0 - p).ln() };
    }
    (sum / scores.len() as f64) as f32
}

fn counts(items: &[(String, u8)]) -> (usize, usize) {
    let pos = items.iter().filter(|(_, y)| *y == 1).count();
    (pos, items.len() - pos)
}

fn print_row(name: &str, op: &OperatingPoint) {
    println!(
        "{name}: n={} threshold={:.6} FNR={:.4} ({}/{}) FPR={:.4} ({}/{}) acc={:.4}",
        op.true_positives + op.false_negatives + op.true_negatives + op.false_positives,
        op.threshold,
        op.fnr,
        op.false_negatives,
        op.true_positives + op.false_negatives,
        op.fpr,
        op.false_positives,
        op.true_negatives + op.false_positives,
        op.accuracy,
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let started = Instant::now();

    let train_all = load_labeled(&args.train)?;
    let dev_all = load_labeled(&args.dev)?;
    let test = if args.skip_test {
        Vec::new()
    } else {
        load_labeled(&args.test)?
    };
    let (train_pos, train_neg) = counts(&train_all);
    let (dev_pos_all, dev_neg_all) = counts(&dev_all);
    let (test_pos, test_neg) = counts(&test);
    println!(
        "loaded train={} (inj {} / safe {}) dev={} (inj {} / safe {}) test={} (inj {} / safe {})",
        train_all.len(),
        train_pos,
        train_neg,
        dev_all.len(),
        dev_pos_all,
        dev_neg_all,
        test.len(),
        test_pos,
        test_neg
    );

    let train = take_limit(train_all, args.train_limit, 0xa4_0001);
    let dev = take_limit(dev_all, args.dev_limit, 0xa4_0002);
    let (train_pos, train_neg) = counts(&train);
    let (dev_pos, dev_neg) = counts(&dev);
    println!(
        "using train={} (inj {} / safe {}) dev={} (inj {} / safe {}) test={} (inj {} / safe {})",
        train.len(),
        train_pos,
        train_neg,
        dev.len(),
        dev_pos,
        dev_neg,
        test.len(),
        test_pos,
        test_neg
    );

    let mut cfg = ArmorTrainerConfig {
        epochs: args.epochs,
        max_tokens: args.max_tokens,
        ..ArmorTrainerConfig::default()
    };
    cfg.fixed_length = args.fixed_length;
    cfg.model = ArmorGuardConfig::default();
    cfg.tokenizer = L1TokenizerSpec::default();
    let eval_tokens = args.max_tokens.min(cfg.model.max_seq_len);

    println!(
        "config epochs={} max_tokens={} batch={} lr={} seed={} model hidden={} heads={} ffn={} layers={} seq={}",
        cfg.epochs,
        cfg.max_tokens,
        cfg.batch_size,
        cfg.learning_rate,
        cfg.seed,
        cfg.model.hidden_dim,
        cfg.model.num_heads,
        cfg.model.ffn_dim,
        cfg.model.num_layers,
        cfg.model.max_seq_len
    );

    let mut encoder = ArmorEncoder::new(&cfg);
    let report = encoder.train(&train, &dev, &cfg);
    let train_seconds = started.elapsed().as_secs_f64();
    println!("training finished in {train_seconds:.1} s");

    // In-memory probe: can the trained model separate *train* at the eval
    // length? If train AUROC is ~0.5 the loop is broken; if train is high and
    // dev is ~0.5 the corpus or the length mismatch is the cause.
    let probe_train: Vec<(String, u8)> =
        train.iter().step_by(8).cloned().collect();
    let probe_dev: Vec<(String, u8)> = dev.iter().step_by(4).cloned().collect();
    if args.probe {
        let ptoks = tokenize_split(&encoder, &probe_train, eval_tokens);
        let dtoks = tokenize_split(&encoder, &probe_dev, eval_tokens);
        let ps = score_items_at(&encoder, &ptoks, eval_tokens);
        let ds = score_items_at(&encoder, &dtoks, eval_tokens);
        let (p_auc, d_auc) = (auroc_scores(&ps), auroc_scores(&ds));
        let bce = bce_at(&ps);
        println!(
            "probe in-memory @{eval_tokens} tokens: train n={} AUROC={p_auc:.4} BCE={bce:.4} | dev n={} AUROC={d_auc:.4}",
            ps.len(),
            ds.len()
        );
    }

    // Final dev evaluation at the configured scan length. The trainer's
    // per-epoch dev pass uses a cheap 12-token scan for early stopping; these
    // numbers are the ones the operating point is chosen from.
    let dev_tokens = tokenize_split(&encoder, &dev, eval_tokens);
    let dev_scores = score_items_at(&encoder, &dev_tokens, eval_tokens);
    let dev_op = operating_point(&dev_scores, 0.01);
    let test_op = if args.skip_test {
        None
    } else {
        let test_tokens = tokenize_split(&encoder, &test, eval_tokens);
        Some(confusion_at(
            &score_items_at(&encoder, &test_tokens, eval_tokens),
            dev_op.threshold,
        ))
    };
    let total_seconds = started.elapsed().as_secs_f64();

    println!();
    for e in &report.epochs {
        println!(
            "epoch {}: train_loss={:.6} dev12 threshold={:.6} FNR={:.4} FPR={:.4} acc={:.4}",
            e.epoch, e.train_loss, e.dev.threshold, e.dev.fnr, e.dev.fpr, e.dev.accuracy
        );
    }
    print_row("dev  (max_tokens)", &dev_op);
    if let Some(op) = &test_op {
        print_row("test (dev threshold, scored once)", op);
    } else {
        println!("test: not evaluated (--skip-test)");
    }
    println!(
        "wall={total_seconds:.1} s (train {train_seconds:.1} s) best_epoch={} stopped_early={}",
        report.best_epoch, report.stopped_early
    );

    std::fs::create_dir_all(&args.out)?;
    let fral = args.out.join("armor-l1.fral");
    let scanner = encoder.to_scanner();
    scanner.save(&fral)?;
    println!("wrote {}", fral.display());
    if args.probe {
        // Export round-trip: the saved artefact must score the same as memory.
        let loaded = L1Scanner::load(&fral)?;
        let spec = *loaded.tokenizer_spec();
        let (mut pos, mut neg) = (Vec::new(), Vec::new());
        for (text, label) in &probe_dev {
            let toks = ferrisres::security::armor_l1::tokenize_text(&spec, text, eval_tokens);
            if toks.is_empty() {
                continue;
            }
            let s = loaded.forward_tokens(&toks);
            if *label == 1 { pos.push(s); } else { neg.push(s); }
        }
        println!("probe reloaded artefact on dev: AUROC={:.4}", auroc_parts(&pos, &neg));
    }

    let epochs: Vec<serde_json::Value> = report
        .epochs
        .iter()
        .map(|e| {
            serde_json::json!({
                "epoch": e.epoch,
                "train_loss": e.train_loss,
                // The trainer early-stops on a 12-token dev scan.
                "dev_early_stop_scan": op_json(&e.dev, 12),
            })
        })
        .collect();
    let doc = serde_json::json!({
        "config": {
            "train": args.train.display().to_string(),
            "dev": args.dev.display().to_string(),
            "test": args.test.display().to_string(),
            "epochs": cfg.epochs,
            "max_tokens": cfg.max_tokens,
            "eval_tokens": eval_tokens,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "seed": cfg.seed,
            "early_stop_fpr": cfg.early_stop_fpr,
            "train_limit": args.train_limit,
            "dev_limit": args.dev_limit,
            "fixed_length": cfg.fixed_length,
            "model": {
                "vocab_size": cfg.model.vocab_size,
                "hidden_dim": cfg.model.hidden_dim,
                "num_heads": cfg.model.num_heads,
                "ffn_dim": cfg.model.ffn_dim,
                "num_layers": cfg.model.num_layers,
                "max_seq_len": cfg.model.max_seq_len,
            },
        },
        "data": {
            "train": train.len(),
            "train_injection": train_pos,
            "train_safe": train_neg,
            "dev": dev.len(),
            "dev_injection": dev_pos,
            "dev_safe": dev_neg,
            "test": test.len(),
            "test_injection": test_pos,
            "test_safe": test_neg,
        },
        "epochs": epochs,
        "dev": op_json(&dev_op, eval_tokens),
        "test": test_op.as_ref().map(|op| op_json(op, eval_tokens)),
        "test_threshold_source": "dev",
        "wall_seconds": total_seconds,
        "train_seconds": train_seconds,
        "best_epoch": report.best_epoch,
        "stopped_early": report.stopped_early,
        "fral": fral.display().to_string(),
    });
    let report_path = args.out.join("report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&doc)?)?;
    println!("wrote {}", report_path.display());
    Ok(())
}
