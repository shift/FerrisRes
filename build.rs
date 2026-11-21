//! Build script: embed every `architectures/*.json` descriptor into the binary.
//!
//! Adding a new architecture is a data change: drop a descriptor in
//! `architectures/` and the next build embeds it. No Rust file lists the
//! descriptors, which is the whole point of the declarative registry.
//!
//! An external directory can still be loaded at runtime with
//! `FERRISRES_ARCHITECTURES_DIR` (see `src/model/architectures.rs`).

use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let dir = manifest.join("architectures");
    println!("cargo:rerun-if-changed={}", dir.display());

    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                files.push(path);
            }
        }
    }
    files.sort();

    let mut code = String::from("pub const EMBEDDED_ARCHITECTURES: &[(&str, &str)] = &[\n");
    for path in files {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown.json".to_string());
        writeln!(code, "    ({:?}, include_str!(r###\"{}\"###)),", name, path.display())
            .expect("write embedded architecture entry");
    }
    code.push_str("];\n");

    let out =
        PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("architectures_embedded.rs");
    std::fs::write(out, code).expect("write architectures_embedded.rs");
}
