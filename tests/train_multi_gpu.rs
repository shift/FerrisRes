//! Two-process CPU gate for the f32 trainer's multi-GPU coordination layer.
//!
//! The exchange, barrier and launcher are CPU mechanisms, so this test runs the
//! real `ferrisres workers --cpu` launcher and two real `ferrisres train`
//! children. It needs no Vulkan ICD and no GPU. What it does *not* prove is GPU
//! training: there is no wgpu training loop, and the evidence files say
//! `"adapter": "cpu"`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ferrisres::training::train_loop::inspect_checkpoint;

const WEIGHT_MAGIC: &[u8; 8] = b"FRSWGT01";
const WEIGHT_HEADER_LEN: usize = 40;

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "ferrisres-multi-gpu-gate-{}-{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_ferrisres")
}

/// 200 ASCII bytes. With `--seq-len 16` that is 12 windows, which divides
/// across a world size of two, so every rank runs six whole steps.
fn tiny_corpus(dir: &Path) -> PathBuf {
    let path = dir.join("corpus.txt");
    let line = "the quick brown fox jumps over the lazy dog. ";
    let mut text = String::new();
    while text.len() < 200 {
        text.push_str(line);
    }
    text.truncate(200);
    fs::write(&path, text).expect("write corpus");
    path
}

fn train_flags(data: &Path, exchange: &Path, output: &Path) -> Vec<String> {
    [
        "train",
        "--data",
        &data.display().to_string(),
        "--hidden-dim",
        "16",
        "--num-blocks",
        "2",
        "--block-size",
        "4",
        "--seq-len",
        "16",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--exchange-dir",
        &exchange.display().to_string(),
        "--exchange-every",
        "5",
        "--output",
        &output.display().to_string(),
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

#[test]
fn two_cpu_processes_train_exchange_and_write_rank_evidence() {
    let temp = TempDir::new("two-process");
    let data = tiny_corpus(temp.path());
    let exchange = temp.path().join("exchange");
    let output = temp.path().join("rank");

    let mut command = Command::new(binary());
    command.arg("workers").arg("--cpu").arg("--world-size").arg("2").arg("--");
    command.arg(binary());
    command.args(train_flags(&data, &exchange, &output));

    let result = command.output().expect("spawn workers");
    let stderr = String::from_utf8_lossy(&result.stderr);
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        result.status.success(),
        "workers failed: status={:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        result.status
    );

    // Two rank evidence files, each naming its own rank.
    let mut ranks = Vec::new();
    for rank in 0..2 {
        let path = exchange.join(format!("rank-{rank}.json"));
        let bytes = fs::read(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["rank"], serde_json::json!(rank));
        assert_eq!(value["world_size"], serde_json::json!(2));
        assert_eq!(value["adapter"], serde_json::json!("cpu"));
        assert_eq!(value["steps"], serde_json::json!(6));
        for field in ["initial_loss", "final_loss", "loss"] {
            let loss = value[field].as_f64().expect("loss is a number");
            assert!(loss.is_finite(), "{field} is not finite: {loss}");
        }
        assert!(value["tokens_per_second"].as_f64().unwrap() > 0.0);
        ranks.push(value["rank"].as_u64().unwrap());
    }
    assert_ne!(ranks[0], ranks[1], "evidence must name distinct ranks");

    // Both per-rank checkpoints load and record world size 2.
    let mut param_counts = Vec::new();
    for rank in 0..2 {
        let path = PathBuf::from(format!("{}.rank-{rank}", output.display()));
        let info = inspect_checkpoint(&path)
            .unwrap_or_else(|err| panic!("checkpoint {}: {err}", path.display()));
        assert_eq!(info.parallelism.world_size, 2);
        assert_eq!(info.parallelism.rank, rank);
        assert_eq!(info.step, 6);
        param_counts.push(info.param_count);
    }
    assert_eq!(param_counts[0], param_counts[1]);

    // The averaged parameter file that every rank loaded back.
    let averages: Vec<PathBuf> = fs::read_dir(&exchange)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("weights-") && name.ends_with("-avg.bin"))
                .unwrap_or(false)
        })
        .collect();
    assert!(!averages.is_empty(), "no averaged weight file in {}", exchange.display());
    // The last step always exchanges, so the step-6 average must exist.
    let final_average = exchange.join("weights-6-avg.bin");
    assert!(final_average.is_file(), "missing {}", final_average.display());

    let bytes = fs::read(&final_average).unwrap();
    assert_eq!(&bytes[..8], WEIGHT_MAGIC, "wrong average file magic");
    let count = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
    assert_eq!(count, param_counts[0], "average element count");
    assert_eq!(bytes.len(), WEIGHT_HEADER_LEN + 4 * count, "average payload length");
}

#[test]
fn resuming_a_world_size_two_checkpoint_at_world_size_one_is_refused() {
    let temp = TempDir::new("resume-refusal");
    let data = tiny_corpus(temp.path());
    let exchange = temp.path().join("exchange");
    let output = temp.path().join("rank");

    let mut command = Command::new(binary());
    command.arg("workers").arg("--cpu").arg("--world-size").arg("2").arg("--");
    command.arg(binary());
    command.args(train_flags(&data, &exchange, &output));
    let run = command.output().expect("spawn workers");
    assert!(run.status.success(), "setup run failed: {}", String::from_utf8_lossy(&run.stderr));

    let checkpoint = PathBuf::from(format!("{}.rank-0", output.display()));
    let mut resume = Command::new(binary());
    resume
        .arg("train")
        .arg("--data")
        .arg(&data)
        .arg("--hidden-dim")
        .arg("16")
        .arg("--num-blocks")
        .arg("2")
        .arg("--block-size")
        .arg("4")
        .arg("--seq-len")
        .arg("16")
        .arg("--epochs")
        .arg("1")
        .arg("--batch-size")
        .arg("1")
        .arg("--world-size")
        .arg("1")
        .arg("--resume")
        .arg(&checkpoint)
        .arg("--output")
        .arg(temp.path().join("resumed"));

    let refused = resume.output().expect("spawn resume");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&refused.stdout),
        String::from_utf8_lossy(&refused.stderr)
    );
    assert!(!refused.status.success(), "a world-size change must be refused, got success:\n{text}");
    assert!(text.contains("world_size=2"), "must name the recorded size: {text}");
    assert!(text.contains("world_size=1"), "must name the request: {text}");
    assert!(text.contains("refusing"), "must say it refused: {text}");
}
