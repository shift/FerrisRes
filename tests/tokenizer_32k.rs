//! The 32k tokenizer: round trip, determinism, artefacts and fertility.
//!
//! The corpus is authored here rather than downloaded: a real sample is
//! recorded in `artifacts/frs-tokenizer-32k/0.1.0/sample-manifest.jsonl`, and
//! these tests must not depend on the network.

use std::path::{Path, PathBuf};

use ferrisres::tokenizer::train::{train, TrainConfig};
use ferrisres::tokenizer::{fertility, inspect, sha256_hex, SPECIAL_TOKENS};

/// A small source-like fixture: Rust, Nix, JSON and prose, with tabs, trailing
/// spaces and literal `\n` escapes.
const FIXTURE_FILES: [(&str, &str); 6] = [
    (
        "src/lib.rs",
        "use std::collections::HashMap;\n\
         \n\
         /// A cache keyed by name.\n\
         pub struct Cache {\n\
         \tentries: HashMap<String, Vec<u8>>,\n\
         }\n\
         \n\
         impl Cache {\n\
         \tpub fn insert(&mut self, key: &str, value: Vec<u8>) {\n\
         \t\tself.entries.insert(key.to_string(), value);\n\
         \t}\n\
         \tpub fn get(&self, key: &str) -> Option<&Vec<u8>> {\n\
         \t\tself.entries.get(key)\n\
         \t}\n\
         }\n\
         \n\
         fn main() {\n\
         \tlet mut cache = Cache { entries: HashMap::new() };\n\
         \tcache.insert(\"answer\", b\"forty-two\".to_vec());\n\
         \tprintln!(\"escaped newline: \\\\n\");\n\
         \tprintln!(\"trailing spaces end here   \");\n\
         }\n",
    ),
    (
        "nix/module.nix",
        "{ config, lib, pkgs, ... }:\n\
         let\n\
         \tcfg = config.services.ferrisres;\n\
         in\n\
         {\n\
         \toptions.services.ferrisres = {\n\
         \t\tenable = lib.mkEnableOption \"the FerrisRes runtime\";\n\
         \t\tport = lib.mkOption {\n\
         \t\t\ttype = lib.types.port;\n\
         \t\t\tdefault = 8080;\n\
         \t\t\tdescription = \"Port to listen on.\";\n\
         \t\t};\n\
         \t};\n\
         \tconfig = lib.mkIf cfg.enable {\n\
         \t\tsystemd.services.ferrisres = {\n\
         \t\t\twantedBy = [ \"multi-user.target\" ];\n\
         \t\t\tserviceConfig.ExecStart = \"${pkgs.ferrisres}/bin/ferrisres serve --port ${toString cfg.port}\";\n\
         \t\t};\n\
         \t};\n\
         }\n",
    ),
    (
        "docs/guide.md",
        "# Guide\n\
         \n\
         The tokenizer preserves whitespace, indentation and case. Rust and Nix\n\
         are case-sensitive languages, so no normalisation is applied.\n\
         \n\
         ```rust\n\
         let x = \"hello\";\n\
         ```\n\
         \n\
         Trailing spaces stay: three here   \n",
    ),
    (
        "data/schema.json",
        "{\n\
         \t\"name\": \"ferrisres\",\n\
         \t\"version\": \"0.1.0\",\n\
         \t\"tools\": [\n\
         \t\t{ \"name\": \"read_file\", \"description\": \"Read a UTF-8 file.\" },\n\
         \t\t{ \"name\": \"run_tests\", \"description\": \"Run the test suite.\" }\n\
         \t],\n\
         \t\"messages\": [\n\
         \t\t{ \"role\": \"user\", \"content\": \"why is the build slow?\" },\n\
         \t\t{ \"role\": \"assistant\", \"content\": \"Because four crates compile twice.\" }\n\
         \t]\n\
         }\n",
    ),
    (
        "notes/german.txt",
        "Die Quelle bleibt unverändert. Groß- und Kleinschreibung ist wichtig,\n\
         und Umlaute dürfen nicht normalisiert werden: ä ö ü ß.\n\
         \n\
         Ein Werkzeug liest eine Datei und gibt den Inhalt zurück.\n",
    ),
    (
        "notes/whitespace.txt",
        "tabs\tbetween\twords\n\
         trailing spaces after this:   \n\
         literal backslash-n: \\n\n\
         empty line follows:\n\
         \n\
         end\n",
    ),
];

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "ferrisres-tokenizer-32k-{}-{}",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_dir_all(&root);
    root
}

/// Write the authored fixture corpus and return its directory.
fn write_fixture(name: &str) -> PathBuf {
    let corpus = temp_root(name).join("corpus");
    for (relative, contents) in FIXTURE_FILES {
        let path = corpus.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, contents).unwrap();
    }
    corpus
}

/// The fixture is small, so the vocabulary saturates well below 32768. That is
/// fine for the round-trip and determinism gates; the production run uses 32768.
const FIXTURE_VOCAB: usize = 512;

#[test]
fn tokenizer_32k_round_trips_and_is_deterministic() {
    tokenizer_round_trips_and_is_deterministic();
}

#[test]
fn tokenizer_round_trips_and_is_deterministic() {
    let corpus = write_fixture("determinism");
    let root = corpus.parent().unwrap().to_path_buf();

    let first = train(&TrainConfig::new(&corpus, FIXTURE_VOCAB, root.join("out-a"))).unwrap();
    let second = train(&TrainConfig::new(&corpus, FIXTURE_VOCAB, root.join("out-b"))).unwrap();

    assert_eq!(
        first.tokenizer_sha256, second.tokenizer_sha256,
        "two trainings on the same sorted corpus must produce the same tokenizer.json"
    );
    assert_eq!(first.vocab_size, second.vocab_size);

    let tokenizer_path = root.join("out-a").join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();

    let cases = [
        "plain ASCII text",
        "UTF-8: ä ö ü ß 日本語 🦀 – “quotes”",
        "tabs\tbetween\tfields",
        "trailing spaces live here   ",
        "escaped newline in a Rust literal: \\n",
        "a real newline\nstays a newline\n",
        "  indented with spaces\n\tindented with a tab\n",
        "",
    ];
    for text in cases {
        let encoding = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(encoding.get_ids(), false).unwrap();
        assert_eq!(decoded, text, "round trip failed for {text:?}");
    }

    // A fertility number exists and is positive for non-empty text.
    let sample = root.join("sample.txt");
    std::fs::write(&sample, "fn main() { println!(\"hello\"); }\n").unwrap();
    let report = fertility(&tokenizer_path, &sample).unwrap();
    assert!(report.tokens > 0);
    assert!(report.bytes > 0);
    assert!(
        report.tokens_per_byte > 0.0,
        "tokens/byte must be positive, got {}",
        report.tokens_per_byte
    );
}

#[test]
fn tokenizer_32k_writes_the_artefacts_and_the_sample_manifest() {
    let corpus = write_fixture("artefacts");
    let root = corpus.parent().unwrap().to_path_buf();
    let out = root.join("out");

    let report = train(&TrainConfig::new(&corpus, FIXTURE_VOCAB, &out)).unwrap();

    for name in ["tokenizer.json", "special_tokens.json", "sample-manifest.jsonl", "SHA256SUMS"] {
        assert!(out.join(name).is_file(), "missing artefact {name}");
    }

    // Every fixture file is in the manifest, with its own hash, in path order.
    let manifest = std::fs::read_to_string(out.join("sample-manifest.jsonl")).unwrap();
    let lines: Vec<serde_json::Value> =
        manifest.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(lines.len(), FIXTURE_FILES.len());

    let mut paths: Vec<&str> = lines.iter().map(|v| v["path"].as_str().unwrap()).collect();
    let mut sorted = paths.clone();
    sorted.sort_unstable();
    assert_eq!(paths, sorted, "manifest must be sorted by path");
    paths.clear();

    for line in &lines {
        let path = line["path"].as_str().unwrap();
        let expected = sha256_hex(std::fs::read(corpus.join(path)).unwrap().as_slice());
        assert_eq!(line["sha256"].as_str().unwrap(), expected);
        assert_eq!(line["licence"].as_str().unwrap(), "unspecified");
    }

    // The special block is exactly the required one, and every token has an id.
    let specials: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("special_tokens.json")).unwrap()).unwrap();
    let object = specials.as_object().unwrap();
    assert_eq!(object.len(), SPECIAL_TOKENS.len());
    let mut ids: Vec<u64> = Vec::new();
    for token in SPECIAL_TOKENS {
        let id = object
            .get(token)
            .unwrap_or_else(|| panic!("special token {token} missing"))
            .as_u64()
            .unwrap();
        ids.push(id);
    }
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), SPECIAL_TOKENS.len(), "special ids must be unique");

    // Inspect reports the same hash training reported, and the SHA256SUMS file
    // agrees with the actual bytes.
    let inspected = inspect(&out.join("tokenizer.json")).unwrap();
    assert_eq!(inspected.sha256, report.tokenizer_sha256);
    assert!(inspected.vocab_size >= 256 + SPECIAL_TOKENS.len());
    let sums = std::fs::read_to_string(out.join("SHA256SUMS")).unwrap();
    for name in ["sample-manifest.jsonl", "special_tokens.json", "tokenizer.json"] {
        let expected = sha256_hex(&std::fs::read(out.join(name)).unwrap());
        assert!(
            sums.lines().any(|line| line == format!("{expected}  {name}")),
            "SHA256SUMS is missing {name}"
        );
    }
}

#[test]
fn tokenizer_32k_licence_map_labels_the_manifest() {
    let corpus = write_fixture("licence");
    let root = corpus.parent().unwrap().to_path_buf();
    let out = root.join("out");

    let mut config = TrainConfig::new(&corpus, FIXTURE_VOCAB, &out);
    config.default_licence = "MIT".to_string();
    config.licence_map.insert("src/".to_string(), "MIT OR Apache-2.0".to_string());
    train(&config).unwrap();

    let manifest = std::fs::read_to_string(out.join("sample-manifest.jsonl")).unwrap();
    for line in manifest.lines() {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        let path = value["path"].as_str().unwrap();
        let expected = if path.starts_with("src/") { "MIT OR Apache-2.0" } else { "MIT" };
        assert_eq!(value["licence"].as_str().unwrap(), expected, "{path}");
    }
}

#[test]
fn tokenizer_32k_cli_trains_inspects_and_measures_fertility() {
    use std::process::Command;

    let corpus = write_fixture("cli");
    let root = corpus.parent().unwrap().to_path_buf();
    let out = root.join("cli-out");
    let bin = env!("CARGO_BIN_EXE_ferrisres");

    let trained = Command::new(bin)
        .args(["tokenizer", "train", "--corpus"])
        .arg(&corpus)
        .args(["--vocab", &FIXTURE_VOCAB.to_string(), "--out"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(trained.status.success(), "train failed: {}", String::from_utf8_lossy(&trained.stderr));
    let stdout = String::from_utf8_lossy(&trained.stdout);
    assert!(stdout.contains("sha256="), "stdout was {stdout}");

    let inspected = Command::new(bin)
        .args(["tokenizer", "inspect", "--tokenizer"])
        .arg(out.join("tokenizer.json"))
        .output()
        .unwrap();
    assert!(inspected.status.success());
    let stdout = String::from_utf8_lossy(&inspected.stdout);
    assert!(stdout.contains("vocab_size"), "stdout was {stdout}");
    assert!(stdout.contains("sha256"), "stdout was {stdout}");

    let sample = root.join("cli-text.txt");
    std::fs::write(&sample, "The build is slow because four crates compile twice.\n").unwrap();
    let measured = Command::new(bin)
        .args(["tokenizer", "fertility", "--tokenizer"])
        .arg(out.join("tokenizer.json"))
        .arg("--text")
        .arg(&sample)
        .output()
        .unwrap();
    assert!(measured.status.success());
    let stdout = String::from_utf8_lossy(&measured.stdout);
    assert!(stdout.contains("tokens/byte"), "stdout was {stdout}");
}

#[test]
fn tokenizer_32k_refuses_a_vocabulary_smaller_than_the_byte_alphabet() {
    let corpus = write_fixture("small-vocab");
    let root = corpus.parent().unwrap().to_path_buf();
    let error = train(&TrainConfig::new(&corpus, 100, root.join("out"))).unwrap_err();
    assert!(error.to_string().contains("too small"), "{error}");
    assert!(!Path::new(&root.join("out")).exists());
}

/// The committed artefact is frozen: its hash is the contract the model config
/// records. Retraining the tokenizer must update this constant and bump the
/// major version.
const ARTEFACT_SHA256: &str = "98aa17a1e7d02e033647dd171d26a73872a2ae49b341bb39d1ce5387963feaf8";

#[test]
fn tokenizer_32k_committed_artefact_is_frozen_and_round_trips() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts")
        .join("frs-tokenizer-32k")
        .join("0.1.0");
    let tokenizer_path = dir.join("tokenizer.json");
    if !tokenizer_path.is_file() {
        // The artefact is committed with the branch; skip if it was removed.
        return;
    }

    let bytes = std::fs::read(&tokenizer_path).unwrap();
    assert_eq!(sha256_hex(&bytes), ARTEFACT_SHA256);

    let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes).unwrap();
    assert_eq!(tokenizer.get_vocab_size(true), 32768);
    for token in SPECIAL_TOKENS {
        assert!(
            tokenizer.token_to_id(token).is_some(),
            "artefact is missing special token {token}"
        );
    }

    for text in [
        "fn main() { println!(\"hello\"); }\n",
        "  \tindented\twith a tab \n",
        "trailing spaces stay   ",
        "escaped newline literal: \\n",
        "UTF-8: ä ö ü ß 日本語 🦀 – “quotes”",
    ] {
        let encoding = tokenizer.encode(text, false).unwrap();
        assert_eq!(
            tokenizer.decode(encoding.get_ids(), false).unwrap(),
            text,
            "artefact round trip failed for {text:?}"
        );
    }

    // SHA256SUMS must describe the committed bytes.
    let sums = std::fs::read_to_string(dir.join("SHA256SUMS")).unwrap();
    for name in ["sample-manifest.jsonl", "special_tokens.json", "tokenizer.json"] {
        let expected = sha256_hex(&std::fs::read(dir.join(name)).unwrap());
        assert!(
            sums.lines().any(|line| line == format!("{expected}  {name}")),
            "SHA256SUMS is missing {name}"
        );
    }
}
