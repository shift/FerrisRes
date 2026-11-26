//! Corpus source adapter tests.
//!
//! Every fixture is local: a temporary store directory, a `file://` git remote,
//! and a loopback TCP listener that must stay untouched. No test uses a real
//! network remote.

use ferrisres::corpus::plan::{Plan, SourceFileSpec, SourceKind, SourceSpec};
use ferrisres::corpus::sources::acquire;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::ErrorKind;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(name: &str) -> Self {
        let mut path = std::env::temp_dir();
        path.push(format!("ferrisres-corpus-sources-{}-{}", name, std::process::id()));
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

fn hf_source(id: &str, store: &Path, files: Vec<SourceFileSpec>) -> SourceSpec {
    SourceSpec {
        id: id.to_string(),
        kind: SourceKind::Hf,
        path: format!("file://{}", store.display()),
        licence: Some("MIT".to_string()),
        revision: Some("snapshot-abc123".to_string()),
        offline: false,
        gated: false,
        files,
        weight: 1.0,
    }
}

fn named_file(store: &Path, rel: &str) -> SourceFileSpec {
    let bytes = fs::read(store.join(rel)).expect("fixture file");
    SourceFileSpec { path: rel.to_string(), sha256: sha256_hex(&bytes) }
}

/// Create a local git repository with two commits. Returns the repo path and
/// the first commit id.
fn git_fixture(root: &Path) -> (PathBuf, String) {
    let repo = root.join("repo");
    fs::create_dir_all(&repo).expect("create repo");
    run(&repo, &["init", "-q"]);
    write(&repo.join("doc.txt"), "version one of the document\n");
    run(&repo, &["add", "doc.txt"]);
    commit(&repo, "first");
    let first = String::from_utf8(run(&repo, &["rev-parse", "HEAD"]))
        .expect("utf8 commit")
        .trim()
        .to_string();

    write(&repo.join("doc.txt"), "version two of the document\n");
    run(&repo, &["add", "doc.txt"]);
    commit(&repo, "second");
    (repo, first)
}

fn commit(repo: &Path, message: &str) {
    let status = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid"])
        .args(["commit", "-q", "-m", message])
        .status()
        .expect("run git commit");
    assert!(status.success(), "git commit failed");
}

fn run(repo: &Path, args: &[&str]) -> Vec<u8> {
    let output = Command::new("git").arg("-C").arg(repo).args(args).output().expect("run git");
    assert!(
        output.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

// The VERIFY filter is `corpus_sources`, which Cargo applies to test *names*.
// Keeping the tests in this module means every one is selected by that filter.
mod corpus_sources {
    use super::*;

    #[test]
    fn unpinned_revision_is_refused() {
        let temp = TempDir::new("unpinned");
        let (repo, _commit) = git_fixture(temp.path());

        let plan = serde_json::json!({
            "name": "unpinned",
            "version": "0.1.0",
            "seq_len": 16,
            "seed": 1,
            "tokenizer": {"kind": "byte"},
            "sources": [{
                "id": "no-revision",
                "kind": "git",
                "path": format!("file://{}", repo.display()),
                "licence": "MIT"
            }],
            "split": {"val": 0.0, "heldout": 0.0}
        });
        let plan_path = temp.path().join("plan.json");
        write_plan(&plan_path, &plan);

        let error = Plan::load(&plan_path).expect_err("a git source without a revision must fail");
        let message = format!("{error:#}");
        assert!(message.contains("revision"), "error must name the revision gate: {message}");
        assert!(message.contains("no-revision"), "error must name the source: {message}");
        assert!(message.contains("git"), "error must name the kind: {message}");
    }

    #[test]
    fn offline_refuses_network() {
        let temp = TempDir::new("offline");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
        listener.set_nonblocking(true).expect("nonblocking listener");
        let port = listener.local_addr().expect("listener address").port();

        let source = SourceSpec {
            id: "gated-hf".to_string(),
            kind: SourceKind::Hf,
            // A loopback URL would be reached immediately if the offline guard did
            // not run first. The listener below proves nothing connected.
            path: format!("http://127.0.0.1:{port}/datasets/org/name"),
            licence: Some("MIT".to_string()),
            revision: Some("snapshot-abc123".to_string()),
            offline: true,
            gated: false,
            files: vec![SourceFileSpec {
                path: "data/train.txt".to_string(),
                sha256: sha256_hex(b"never fetched"),
            }],
            weight: 1.0,
        };

        let cache = temp.path().join("cache");
        let error = acquire(&source, &cache).expect_err("offline hf source must fail");
        let message = format!("{error:#}");
        assert!(message.contains("offline"), "error must name the offline gate: {message}");
        assert!(message.contains("gated-hf"), "error must name the source: {message}");
        assert!(message.contains("hf"), "error must name the kind: {message}");

        match listener.accept() {
            Err(error) if error.kind() == ErrorKind::WouldBlock => {}
            other => panic!("offline source opened a socket: {other:?}"),
        }
        assert!(!cache.exists(), "offline source must not create a cache directory");
    }

    #[test]
    fn git_source_checks_out_the_pinned_commit() {
        let temp = TempDir::new("git-pinned");
        let (repo, first) = git_fixture(temp.path());
        let source = SourceSpec {
            id: "fixture-git".to_string(),
            kind: SourceKind::Git,
            path: format!("file://{}", repo.display()),
            licence: Some("MIT".to_string()),
            revision: Some(first.clone()),
            offline: false,
            gated: false,
            files: Vec::new(),
            weight: 1.0,
        };

        let cache = temp.path().join("cache");
        let acquired = acquire(&source, &cache).expect("clone the local git fixture");
        assert_eq!(acquired.kind, SourceKind::Git);
        assert_eq!(acquired.revision, first);
        let text = fs::read_to_string(acquired.root.join("doc.txt")).expect("checked out file");
        assert_eq!(text, "version one of the document\n", "must read the pinned commit, not HEAD");
    }

    #[test]
    fn git_revision_that_is_a_branch_is_refused() {
        let temp = TempDir::new("git-branch");
        let (repo, _commit) = git_fixture(temp.path());
        let source = SourceSpec {
            id: "moving-git".to_string(),
            kind: SourceKind::Git,
            path: format!("file://{}", repo.display()),
            licence: Some("MIT".to_string()),
            revision: Some("main".to_string()),
            offline: true,
            gated: false,
            files: Vec::new(),
            weight: 1.0,
        };

        let error = acquire(&source, &temp.path().join("cache"))
            .expect_err("a branch is not a pinned revision");
        let message = format!("{error:#}");
        assert!(message.contains("moving-git"), "error must name the source: {message}");
        assert!(message.contains("commit"), "error must say a commit is required: {message}");
    }

    #[test]
    fn hf_reads_only_the_named_files_and_verifies_sha256() {
        let temp = TempDir::new("hf-local");
        let store = temp.path().join("store");
        write(&store.join("data/train.txt"), "the named training document\n");
        write(&store.join("data/other.txt"), "a file the plan did not name\n");

        let source = hf_source("fixture-hf", &store, vec![named_file(&store, "data/train.txt")]);
        let acquired = acquire(&source, &temp.path().join("cache")).expect("read the local store");
        let files = acquired.files.expect("hf source lists its files");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].rel_path, "data/train.txt");
        assert_eq!(
            fs::read_to_string(&files[0].path).expect("read named file"),
            "the named training document\n"
        );

        // Only the named file is read. Tampering with an unnamed file changes
        // nothing, which is how we know the adapter never opened it.
        write(&store.join("data/other.txt"), "tampered but unlisted\n");
        acquire(&source, &temp.path().join("cache")).expect("unlisted file is not read");

        write(&store.join("data/train.txt"), "tampered named file\n");
        let error =
            acquire(&source, &temp.path().join("cache")).expect_err("a sha256 mismatch must fail");
        let message = format!("{error:#}");
        assert!(message.contains("fixture-hf"), "error must name the source: {message}");
        assert!(message.contains("sha256"), "error must name the check: {message}");
        assert!(message.contains("mismatch"), "error must name the mismatch: {message}");
        assert!(message.contains("data/train.txt"), "error must name the file: {message}");
    }

    #[test]
    fn gated_source_without_a_token_is_refused() {
        let temp = TempDir::new("gated");
        let store = temp.path().join("store");
        write(&store.join("secret.txt"), "gated bytes\n");

        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");

        let mut source = hf_source("gated-hf", &store, vec![named_file(&store, "secret.txt")]);
        source.gated = true;
        let error = acquire(&source, &temp.path().join("cache"))
            .expect_err("a gated source without a token must fail");
        let message = format!("{error:#}");
        assert!(message.contains("gated-hf"), "error must name the source: {message}");
        assert!(message.contains("gated"), "error must name the gate: {message}");
        assert!(
            message.contains("HF_TOKEN"),
            "error must say where the token comes from: {message}"
        );
    }

    #[test]
    fn credentials_are_redacted_from_errors() {
        let temp = TempDir::new("credentials");
        let source = SourceSpec {
            id: "private-remote".to_string(),
            kind: SourceKind::Git,
            path: "https://user:secret@example.invalid/org/repo.git".to_string(),
            licence: Some("MIT".to_string()),
            revision: Some("0123456789abcdef0123456789abcdef01234567".to_string()),
            offline: true,
            gated: false,
            files: Vec::new(),
            weight: 1.0,
        };

        let error =
            acquire(&source, &temp.path().join("cache")).expect_err("offline remote must fail");
        let message = format!("{error:#}");
        assert!(!message.contains("secret"), "credential leaked into an error: {message}");
        assert!(message.contains("***@example.invalid"), "URL is redacted, not dropped: {message}");
    }

    #[test]
    fn build_reads_an_hf_source_by_name() {
        let temp = TempDir::new("hf-build");
        let store = temp.path().join("store");
        write(
            &store.join("corpus.txt"),
            "a hugging face document with enough text to fill one window\n",
        );
        write(&store.join("ignored.txt"), "this file is not named by the plan\n");

        let plan = serde_json::json!({
            "name": "hf-build",
            "version": "0.1.0",
            "seq_len": 16,
            "seed": 9,
            "tokenizer": {"kind": "byte"},
            "sources": [{
                "id": "fixture-hf",
                "kind": "hf",
                "path": format!("file://{}", store.display()),
                "licence": "MIT",
                "revision": "snapshot-abc123",
                "files": [{
                    "path": "corpus.txt",
                    "sha256": sha256_hex(b"a hugging face document with enough text to fill one window\n")
                }]
            }],
            "split": {"val": 0.0, "heldout": 0.0}
        });
        let plan_path = temp.path().join("plan.json");
        write_plan(&plan_path, &plan);

        let out = temp.path().join("out");
        let summary =
            ferrisres::corpus::build(&plan_path, &out, false).expect("build the hf source");
        assert_eq!(summary.files_seen, 1, "only the named file is seen");
        assert!(summary.tokens > 0);

        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("manifest.json")).expect("manifest"))
                .expect("manifest json");
        assert_eq!(manifest["sources"][0]["kind"], "hf");
        assert_eq!(manifest["sources"][0]["revision"], "snapshot-abc123");
        assert_eq!(manifest["sources"][0]["files_seen"], 1);
        assert_eq!(manifest["files"].as_array().expect("files").len(), 1);
        ferrisres::corpus::verify(&out).expect("built corpus verifies");
    }
} // mod corpus_sources
