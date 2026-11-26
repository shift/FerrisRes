//! Diagnostic for a saved FRAL: score distribution, AUROC and per-family dev
//! FNR. Not part of the training path; it reads an artefact and prints numbers.
//!
//! ```bash
//! cargo run --release --example armor_diag -- /tmp/opencode/armor-real/armor-l1.fral
//! ```
//!
//! Train is scored on a deterministic stride sample (default every 8th
//! example) because scoring all 10 268 at 128 positions costs ~25 minutes on
//! this CPU; dev is scored in full.

use ferrisres::security::armor_l1::{tokenize_text, L1Scanner};
use std::collections::BTreeMap;
use std::path::Path;

struct Row {
    text: String,
    label: u8,
    source: String,
}

fn load_rows(path: &str) -> Vec<Row> {
    let data = std::fs::read_to_string(path).expect("corpus file");
    data.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).expect("json line");
            Row {
                text: v["text"].as_str().expect("text").to_string(),
                label: if v["label"].as_str() == Some("injection") { 1 } else { 0 },
                source: v["source"].as_str().unwrap_or("?").to_string(),
            }
        })
        .collect()
}

fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = ((sorted.len() - 1) as f32 * q).round() as usize;
    sorted[idx]
}

/// Rank AUROC with ties counted as 0.5.
fn auroc(pos: &[f32], neg: &[f32]) -> f32 {
    if pos.is_empty() || neg.is_empty() {
        return f32::NAN;
    }
    let mut all: Vec<(f32, u8)> = pos.iter().map(|&s| (s, 1)).chain(neg.iter().map(|&s| (s, 0))).collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // Average ranks for ties.
    let mut rank_sum_pos = 0.0f64;
    let mut i = 0usize;
    while i < all.len() {
        let mut j = i + 1;
        while j < all.len() && all[j].0 == all[i].0 {
            j += 1;
        }
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0; // 1-based ranks i+1..j
        for item in &all[i..j] {
            if item.1 == 1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j;
    }
    let n_pos = pos.len() as f64;
    let n_neg = neg.len() as f64;
    ((rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)) as f32
}

fn summarise(name: &str, pos: &[f32], neg: &[f32]) {
    let mut p = pos.to_vec();
    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut n = neg.to_vec();
    n.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{name:<22} inj n={:<5} p50={:.6} p95={:.6} max={:.6} | safe n={:<5} p50={:.6} p95={:.6} max={:.6} | AUROC={:.4}",
        p.len(),
        quantile(&p, 0.50),
        quantile(&p, 0.95),
        p.last().copied().unwrap_or(f32::NAN),
        n.len(),
        quantile(&n, 0.50),
        quantile(&n, 0.95),
        n.last().copied().unwrap_or(f32::NAN),
        auroc(&p, &n),
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "/tmp/opencode/armor-real/armor-l1.fral".to_string());
    let stride: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let scanner = L1Scanner::load(Path::new(&path)).expect("FRAL loads");
    let spec = *scanner.tokenizer_spec();
    println!("model {path} (stride {stride})");

    if std::env::args().nth(3).as_deref() == Some("lengths") {
        let train = load_rows("assets/armor/build/train.jsonl");
        let mut pos: Vec<&Row> = train.iter().filter(|r| r.label == 1).step_by(7).take(300).collect();
        let mut neg: Vec<&Row> = train.iter().filter(|r| r.label == 0).step_by(7).take(300).collect();
        pos.append(&mut neg);
        for length in [8usize, 12, 16, 24, 32, 48, 64, 96, 128, 256] {
            let (mut p, mut n) = (Vec::new(), Vec::new());
            for r in &pos {
                let toks = tokenize_text(&spec, &r.text, length);
                if toks.is_empty() {
                    continue;
                }
                let s = scanner.forward_tokens(&toks);
                if r.label == 1 {
                    p.push(s);
                } else {
                    n.push(s);
                }
            }
            p.sort_by(|a, b| a.partial_cmp(b).unwrap());
            n.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "train len {length:<3} inj p50={:.6} p95={:.6} | safe p50={:.6} p95={:.6} | AUROC={:.4}",
                quantile(&p, 0.50),
                quantile(&p, 0.95),
                quantile(&n, 0.50),
                quantile(&n, 0.95),
                auroc(&p, &n),
            );
        }
        return;
    }

    if std::env::args().nth(3).as_deref() == Some("stats") {
        for (name, v) in scanner.tensors() {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
            let max = v.iter().fold(0f32, |m, x| m.max(x.abs()));
            if name.contains("classifier") || name.contains("ln") || !norm.is_finite() {
                println!("{name:<16} n={:<6} norm={norm:.6} mean={mean:.6} max={max:.6}", v.len());
            }
        }
        return;
    }

    let mut score = |rows: &[Row]| {
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for r in rows {
            let toks = tokenize_text(&spec, &r.text, 128);
            if toks.is_empty() {
                continue;
            }
            let s = scanner.forward_tokens(&toks);
            if r.label == 1 {
                pos.push(s);
            } else {
                neg.push(s);
            }
        }
        (pos, neg)
    };

    let dev = load_rows("assets/armor/build/dev.jsonl");
    let (dev_pos, dev_neg) = score(&dev);
    summarise("dev (full)", &dev_pos, &dev_neg);

    let train = load_rows("assets/armor/build/train.jsonl");
    let train_sample: Vec<Row> = train.into_iter().step_by(stride).collect();
    let (tr_pos, tr_neg) = score(&train_sample);
    summarise("train (stride)", &tr_pos, &tr_neg);

    // Dev-selected operating point is degenerate when every finite threshold
    // breaks the 1 % FPR budget; use the best finite threshold for the table.
    let mut all_neg: Vec<f32> = dev_neg.clone();
    all_neg.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = *all_neg.last().expect("dev safe scores");

    let mut per_family: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in &dev {
        if r.label != 1 {
            continue;
        }
        let toks = tokenize_text(&spec, &r.text, 128);
        if toks.is_empty() {
            continue;
        }
        let hit = scanner.forward_tokens(&toks) >= threshold;
        let e = per_family.entry(r.source.clone()).or_insert((0, 0));
        e.1 += 1;
        if !hit {
            e.0 += 1;
        }
    }
    println!("\ndev per-family injection FNR at t={threshold:.6} (best finite: dev safe max)");
    let mut fams: Vec<_> = per_family.into_iter().collect();
    fams.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    for (family, (miss, total)) in fams {
        println!("{family:<45} {miss}/{total} missed FNR={:.4}", miss as f32 / total as f32);
    }
}
