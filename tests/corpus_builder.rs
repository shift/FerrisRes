//! Corpus builder v1 integration tests. Fixtures are local temp directories.

use ferrisres::corpus::{build, classify_split, sample, stats, verify, Split};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(name: &str) -> Self {
        let mut path = std::env::temp_dir();
        path.push(format!("ferrisres-corpus-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn write(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, content).expect("write fixture");
}

fn write_plan(path: &Path, plan: &serde_json::Value) {
    write(path, &serde_json::to_string_pretty(plan).expect("plan json"));
}

fn source(id: &str, path: &Path) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "path": path.to_string_lossy(),
        "licence": "MIT OR Apache-2.0",
        "revision": "abc123",
        "weight": 1.0
    })
}

fn find_content_for_split(target: Split, seq_len: usize) -> (String, String) {
    for i in 0..10_000 {
        let content = format!("document-{i:04}-{}-end", "token-sequence-".repeat(6));
        if content.len() < seq_len {
            continue;
        }
        let hash = sha256_hex(content.trim().as_bytes());
        if classify_split(&hash, 0.0, 0.5) == target {
            return (content, hash);
        }
    }
    panic!("no fixture content classified as {}", target.as_str());
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|window| window == needle)
}

fn read_shard_slice(dir: &Path, file: &str, start: u64, end: u64) -> Vec<u8> {
    let data = fs::read(dir.join(file)).expect("read shard");
    data[(start * 2) as usize..(end * 2) as usize].to_vec()
}

#[test]
fn build_is_deterministic() {
    let temp = TempDir::new("deterministic");
    let source_a = temp.path().join("source-a");
    let source_b = temp.path().join("source-b");
    write(
        &source_a.join("alpha.md"),
        "# Alpha\n\nThe first source has enough text to fill several windows.\nRepeated line.\n",
    );
    write(
        &source_a.join("data.jsonl"),
        "{\"text\":\"first jsonl record with corpus text\"}\n{\"text\":\"second jsonl record\"}\n",
    );
    write(
        &source_b.join("beta.rs"),
        "pub fn beta() -> &'static str { \"a rust source document for the corpus\" }\n",
    );
    write(
        &source_b.join("nested/gamma.txt"),
        "nested text document\nwith a second line that is long enough to pack\n",
    );

    let plan = serde_json::json!({
        "name": "determinism",
        "version": "0.1.0",
        "seq_len": 24,
        "seed": 42,
        "tokenizer": {"kind": "byte"},
        "sources": [
            source("zeta", &source_b),
            source("alpha", &source_a)
        ],
        "split": {"val": 0.15, "heldout": 0.15}
    });
    let plan_path = temp.path().join("plan.json");
    write_plan(&plan_path, &plan);

    let out_one = temp.path().join("out-one");
    let out_two = temp.path().join("out-two");
    let first = build(&plan_path, &out_one, false).expect("first build");
    let second = build(&plan_path, &out_two, false).expect("second build");

    assert_eq!(first.manifest_sha256, second.manifest_sha256);
    assert_eq!(first.shards, second.shards);
    assert!(first.tokens > 0, "fixture must produce tokens");
    assert!(!first.resumed);
    for shard in &first.shards {
        assert_eq!(shard.tokens * 2, shard.bytes);
    }

    let first_manifest = fs::read(out_one.join("manifest.json")).expect("manifest one");
    let second_manifest = fs::read(out_two.join("manifest.json")).expect("manifest two");
    assert_eq!(sha256_hex(&first_manifest), first.manifest_sha256);
    assert_eq!(sha256_hex(&second_manifest), first.manifest_sha256);

    verify(&out_one).expect("first build verifies");
    verify(&out_two).expect("second build verifies");
}

#[test]
fn heldout_never_in_train() {
    let temp = TempDir::new("heldout");
    let source_dir = temp.path().join("source");
    let seq_len = 32;
    let (heldout_text, heldout_hash) = find_content_for_split(Split::Heldout, seq_len);
    let (train_text, train_hash) = find_content_for_split(Split::Train, seq_len);
    write(&source_dir.join("heldout.txt"), &heldout_text);
    write(&source_dir.join("train.txt"), &train_text);

    let plan = serde_json::json!({
        "name": "heldout",
        "version": "0.1.0",
        "seq_len": seq_len,
        "seed": 7,
        "tokenizer": {"kind": "byte"},
        "sources": [source("fixture", &source_dir)],
        "split": {"val": 0.0, "heldout": 0.5}
    });
    let plan_path = temp.path().join("plan.json");
    write_plan(&plan_path, &plan);
    let out = temp.path().join("out");
    build(&plan_path, &out, false).expect("build");

    let index: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("index.json")).expect("index"))
            .expect("parse index");
    let shards = index["shards"].as_array().expect("index shards");

    let heldout_doc = shards
        .iter()
        .filter(|shard| shard["split"].as_str() == Some("heldout"))
        .flat_map(|shard| {
            shard["documents"]
                .as_array()
                .expect("documents")
                .iter()
                .map(move |doc| (shard["file"].as_str().expect("file"), doc.clone()))
        })
        .find(|(_, doc)| doc["sha256"].as_str() == Some(heldout_hash.as_str()))
        .expect("heldout fixture appears in a heldout shard");
    let train_doc = shards
        .iter()
        .filter(|shard| shard["split"].as_str() == Some("train"))
        .flat_map(|shard| {
            shard["documents"]
                .as_array()
                .expect("documents")
                .iter()
                .map(move |doc| (shard["file"].as_str().expect("file"), doc.clone()))
        })
        .find(|(_, doc)| doc["sha256"].as_str() == Some(train_hash.as_str()))
        .expect("train fixture appears in a train shard");

    let heldout_file = heldout_doc.0;
    let heldout_seq = read_shard_slice(
        &out,
        heldout_file,
        heldout_doc.1["token_start"].as_u64().expect("start"),
        heldout_doc.1["token_end"].as_u64().expect("end"),
    );
    let train_file = train_doc.0;
    let train_seq = read_shard_slice(
        &out,
        train_file,
        train_doc.1["token_start"].as_u64().expect("start"),
        train_doc.1["token_end"].as_u64().expect("end"),
    );
    assert!(!heldout_seq.is_empty());
    assert!(!train_seq.is_empty());
    assert!(
        !contains_subslice(&train_seq, &heldout_seq),
        "held-out token sequence must not appear in a train shard"
    );

    let train_bytes = fs::read(out.join(train_file)).expect("train shard");
    assert!(
        !contains_subslice(&train_bytes, &heldout_seq),
        "held-out token sequence must not appear anywhere in a train shard"
    );
}

#[test]
fn licence_missing_is_refused() {
    let temp = TempDir::new("licence");
    let source_dir = temp.path().join("source");
    write(&source_dir.join("doc.txt"), "some licensed text long enough to tokenize\n");
    let out = temp.path().join("out");

    let mut without_licence = source("no-licence", &source_dir);
    without_licence.as_object_mut().expect("object").remove("licence");
    let plan = serde_json::json!({
        "name": "licence",
        "version": "0.1.0",
        "seq_len": 16,
        "seed": 1,
        "tokenizer": {"kind": "byte"},
        "sources": [without_licence],
        "split": {"val": 0.0, "heldout": 0.0}
    });
    let plan_path = temp.path().join("plan.json");
    write_plan(&plan_path, &plan);
    let error = build(&plan_path, &out, false).expect_err("missing licence must fail");
    let message = format!("{error:#}");
    assert!(message.contains("licence"), "error must name the licence gate: {message}");
    assert!(message.contains("no-licence"), "error must name the source: {message}");

    let mut without_revision = source("no-revision", &source_dir);
    without_revision.as_object_mut().expect("object").remove("revision");
    let plan = serde_json::json!({
        "name": "revision",
        "version": "0.1.0",
        "seq_len": 16,
        "seed": 1,
        "tokenizer": {"kind": "byte"},
        "sources": [without_revision],
        "split": {"val": 0.0, "heldout": 0.0}
    });
    write_plan(&plan_path, &plan);
    let error = build(&plan_path, &out, false).expect_err("missing revision must fail");
    let message = format!("{error:#}");
    assert!(message.contains("revision"), "error must name the revision gate: {message}");
    assert!(message.contains("no-revision"), "error must name the source: {message}");
}

#[test]
fn secret_is_dropped_and_counted() {
    let temp = TempDir::new("secret");
    let source_dir = temp.path().join("source");
    let key = "AKIAIOSFODNN7EXAMPLE";
    let leak = format!(
        "credentials\nAWS_ACCESS_KEY={key}\n{}",
        "filler text that would otherwise be a corpus document\n".repeat(4)
    );
    let clean = "a clean document with no credentials and enough bytes to pack\n".repeat(4);
    write(&source_dir.join("leak.txt"), &leak);
    write(&source_dir.join("clean.txt"), &clean);

    let plan = serde_json::json!({
        "name": "secret",
        "version": "0.1.0",
        "seq_len": 16,
        "seed": 3,
        "tokenizer": {"kind": "byte"},
        "sources": [source("fixture", &source_dir)],
        "split": {"val": 0.0, "heldout": 0.0}
    });
    let plan_path = temp.path().join("plan.json");
    write_plan(&plan_path, &plan);
    let out = temp.path().join("out");
    let summary = build(&plan_path, &out, false).expect("build");
    assert_eq!(summary.drop_reasons.get("secret"), Some(&1));

    let report = stats(&out).expect("stats");
    assert_eq!(report.drop_reasons.get("secret"), Some(&1));
    assert_eq!(report.secret_files_dropped, 1);
    assert!(report.secret_findings >= 1);
    assert!(report.tokens > 0, "clean document must survive");

    for shard in ["train-00000.bin", "val-00000.bin", "heldout-00000.bin"] {
        let bytes = fs::read(out.join("shards").join(shard)).expect("shard");
        assert!(
            !contains_subslice(&bytes, key.as_bytes()),
            "secret bytes must be absent from {shard}"
        );
    }
    verify(&out).expect("build verifies");
}

#[test]
fn verify_detects_tamper() {
    let temp = TempDir::new("tamper");
    let source_dir = temp.path().join("source");
    write(
        &source_dir.join("doc.txt"),
        "a document long enough to produce at least one full token window\n",
    );
    let plan = serde_json::json!({
        "name": "tamper",
        "version": "0.1.0",
        "seq_len": 16,
        "seed": 5,
        "tokenizer": {"kind": "byte"},
        "sources": [source("fixture", &source_dir)],
        "split": {"val": 0.0, "heldout": 0.0}
    });
    let plan_path = temp.path().join("plan.json");
    write_plan(&plan_path, &plan);
    let out = temp.path().join("out");
    let summary = build(&plan_path, &out, false).expect("build");
    verify(&out).expect("untampered build verifies");

    let shard = summary.shards.iter().find(|shard| shard.tokens > 0).expect("a non-empty shard");
    let path = out.join(&shard.file);
    let mut bytes = fs::read(&path).expect("read shard");
    bytes[0] ^= 0xff;
    fs::write(&path, &bytes).expect("tamper shard");

    let error = verify(&out).expect_err("tampered shard must fail verification");
    let message = format!("{error:#}");
    assert!(
        message.contains(shard.file.split('/').next_back().unwrap_or(&shard.file)),
        "error must name the tampered shard: {message}"
    );
    assert!(message.contains("mismatch"), "error must name the mismatch: {message}");
}

#[test]
fn sample_decodes() {
    let temp = TempDir::new("sample");
    let text = "corpus sample round-trip";
    let seq_len = text.len();
    let source_dir = temp.path().join("source");
    write(&source_dir.join("doc.txt"), text);
    let plan = serde_json::json!({
        "name": "sample",
        "version": "0.1.0",
        "seq_len": seq_len,
        "seed": 11,
        "tokenizer": {"kind": "byte"},
        "sources": [source("fixture", &source_dir)],
        "split": {"val": 0.0, "heldout": 0.0}
    });
    let plan_path = temp.path().join("plan.json");
    write_plan(&plan_path, &plan);
    let out = temp.path().join("out");
    build(&plan_path, &out, false).expect("build");

    let output = sample(&out, text.len(), Some(Split::Train)).expect("sample");
    assert_eq!(output.offset, 0);
    assert_eq!(output.ids.len(), text.len());
    assert_eq!(output.text, text);
}
