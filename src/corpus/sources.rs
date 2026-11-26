//! Corpus source adapters.
//!
//! A source is resolved into a local directory that the builder may read. The
//! three kinds are:
//!
//! - `local`: a directory or file already on this machine.
//! - `git`: a remote cloned once into a cache and checked out at an exact
//!   commit. `offline: true` refuses a clone when the cache is cold.
//! - `hf`: a Hugging Face dataset store. Only the files named in the plan are
//!   read, and each is refused unless its sha256 matches. `offline: true`
//!   refuses a remote store before a socket is opened.
//!
//! Every kind records a pinned revision. A source without one is refused by
//! name in [`crate::corpus::plan::Plan::validate`].
//!
//! Credentials never enter the plan, a log line or the manifest. The HF token
//! is read from `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for the duration of
//! one call and is only placed in an HTTP header. URLs written to errors and
//! the manifest have their userinfo redacted.

use crate::corpus::plan::{sha256_hex, SourceFileSpec, SourceKind, SourceSpec};
use anyhow::{bail, Context, Result};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A source that has been resolved to a local directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcquiredSource {
    pub id: String,
    pub kind: SourceKind,
    /// The revision actually used. For git this is the resolved commit hash.
    pub revision: String,
    /// Directory the builder reads from.
    pub root: PathBuf,
    /// When present, the only files that may be read. Relative paths are
    /// relative to `root`. `None` means discover supported files under `root`.
    pub files: Option<Vec<AcquiredFile>>,
}

/// One verified file inside an [`AcquiredSource`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcquiredFile {
    pub rel_path: String,
    pub path: PathBuf,
    pub sha256: String,
}

/// A corpus source adapter.
pub trait Source {
    fn id(&self) -> &str;
    fn kind(&self) -> SourceKind;
    fn revision(&self) -> &str;
    /// Resolve this source into a local directory. `cache` is writable and
    /// may be reused across builds.
    fn acquire(&self, cache: &Path) -> Result<AcquiredSource>;
}

/// Construct the adapter for a plan source and acquire it.
pub fn acquire(spec: &SourceSpec, cache: &Path) -> Result<AcquiredSource> {
    let revision =
        spec.revision.as_deref().map(str::trim).filter(|value| !value.is_empty()).with_context(
            || {
                format!(
                    "source '{}' ({}) is missing a revision; every source must pin one",
                    spec.id,
                    spec.kind.as_str()
                )
            },
        )?;

    match spec.kind {
        SourceKind::Local => LocalSource {
            id: spec.id.clone(),
            path: PathBuf::from(&spec.path),
            revision: revision.to_string(),
        }
        .acquire(cache),
        SourceKind::Git => GitSource {
            id: spec.id.clone(),
            url: spec.path.clone(),
            revision: revision.to_string(),
            offline: spec.offline,
        }
        .acquire(cache),
        SourceKind::Hf => HfSource {
            id: spec.id.clone(),
            location: spec.path.clone(),
            revision: revision.to_string(),
            offline: spec.offline,
            gated: spec.gated,
            files: spec.files.clone(),
        }
        .acquire(cache),
    }
}

/// Cache directory for git clones and fetched HF files.
pub fn default_cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("FERRISRES_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("ferrisres").join("corpus-sources");
    }
    std::env::temp_dir().join("ferrisres-corpus-sources")
}

/// Remove `user:password@` from a URL before it is shown or recorded.
pub fn redact_credentials(input: &str) -> String {
    let Some(scheme_end) = input.find("://") else {
        return input.to_string();
    };
    let authority_start = scheme_end + 3;
    let rest = &input[authority_start..];
    let authority_end = rest.find('/').unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    match authority.rfind('@') {
        Some(at) => format!("{}://***@{}", &input[..scheme_end], &rest[at + 1..]),
        None => input.to_string(),
    }
}

// ---------------------------------------------------------------------------
// local
// ---------------------------------------------------------------------------

struct LocalSource {
    id: String,
    path: PathBuf,
    revision: String,
}

impl Source for LocalSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> SourceKind {
        SourceKind::Local
    }

    fn revision(&self) -> &str {
        &self.revision
    }

    fn acquire(&self, _cache: &Path) -> Result<AcquiredSource> {
        if !self.path.exists() {
            bail!("local source '{}' path does not exist: {}", self.id, self.path.display());
        }
        Ok(AcquiredSource {
            id: self.id.clone(),
            kind: SourceKind::Local,
            revision: self.revision.clone(),
            root: self.path.clone(),
            files: None,
        })
    }
}

// ---------------------------------------------------------------------------
// git
// ---------------------------------------------------------------------------

struct GitSource {
    id: String,
    url: String,
    revision: String,
    offline: bool,
}

impl Source for GitSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> SourceKind {
        SourceKind::Git
    }

    fn revision(&self) -> &str {
        &self.revision
    }

    fn acquire(&self, cache: &Path) -> Result<AcquiredSource> {
        // A branch or tag is not a pin. Refuse anything that is not a commit
        // hash before touching the network, so "revision: main" cannot slip in.
        if !is_commit_hash(&self.revision) {
            bail!(
                "git source '{}' revision '{}' is not a commit hash; pin an exact commit, never a branch",
                self.id,
                self.revision
            );
        }

        let dir = cache.join(format!("git-{}-{}", slug(&self.id), slug(&self.revision)));

        if !dir.join(".git").exists() {
            if self.offline {
                bail!(
                    "offline: git source '{}' has no cached clone at {}; refusing network access to {}",
                    self.id,
                    dir.display(),
                    redact_credentials(&self.url)
                );
            }
            clone_into_cache(&self.id, &self.url, &self.revision, cache, &dir)?;
        }

        checkout(&self.id, &dir, &self.revision)?;
        let head = git_output(&self.id, Some(&dir), &["rev-parse", "HEAD"])?;
        if !head.starts_with(&self.revision.to_ascii_lowercase()) {
            bail!(
                "git source '{}' checked out {} but requested revision {}",
                self.id,
                head,
                self.revision
            );
        }

        Ok(AcquiredSource {
            id: self.id.clone(),
            kind: SourceKind::Git,
            revision: head,
            root: dir,
            files: None,
        })
    }
}

fn clone_into_cache(id: &str, url: &str, revision: &str, cache: &Path, dir: &Path) -> Result<()> {
    fs::create_dir_all(cache)
        .with_context(|| format!("cannot create source cache {}", cache.display()))?;
    let nonce =
        SystemTime::now().duration_since(UNIX_EPOCH).map(|value| value.as_nanos()).unwrap_or(0);
    let tmp = cache.join(format!(".tmp-git-{}-{}-{nonce}", slug(id), std::process::id()));
    let _ = fs::remove_dir_all(&tmp);

    let result = (|| -> Result<()> {
        run_git(
            id,
            None,
            &["clone", "--quiet", "--no-checkout", "--", url, &tmp.to_string_lossy()],
        )?;
        checkout(id, &tmp, revision)?;
        Ok(())
    })();

    if let Err(error) = result {
        let _ = fs::remove_dir_all(&tmp);
        return Err(error);
    }

    if dir.exists() {
        fs::remove_dir_all(dir)
            .with_context(|| format!("cannot replace stale clone {}", dir.display()))?;
    }
    fs::rename(&tmp, dir).with_context(|| format!("cannot move clone into {}", dir.display()))?;
    Ok(())
}

fn checkout(id: &str, dir: &Path, revision: &str) -> Result<()> {
    run_git(id, Some(dir), &["checkout", "--quiet", "--detach", revision])
}

fn run_git(id: &str, dir: Option<&Path>, args: &[&str]) -> Result<()> {
    let output = git_command(id, dir, args)?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    bail!(
        "git source '{}' failed to run git {}: {}",
        id,
        args.first().copied().unwrap_or(""),
        redact_credentials(stderr.trim())
    )
}

fn git_output(id: &str, dir: Option<&Path>, args: &[&str]) -> Result<String> {
    let output = git_command(id, dir, args)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "git source '{}' failed to run git {}: {}",
            id,
            args.first().copied().unwrap_or(""),
            redact_credentials(stderr.trim())
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_command(id: &str, dir: Option<&Path>, args: &[&str]) -> Result<std::process::Output> {
    let mut command = Command::new("git");
    if let Some(dir) = dir {
        command.arg("-C").arg(dir);
    }
    command.args(args).output().with_context(|| format!("git source '{}' cannot run git", id))
}

fn is_commit_hash(value: &str) -> bool {
    let len = value.len();
    (4..=40).contains(&len) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

// ---------------------------------------------------------------------------
// hf
// ---------------------------------------------------------------------------

struct HfSource {
    id: String,
    location: String,
    revision: String,
    offline: bool,
    gated: bool,
    files: Vec<SourceFileSpec>,
}

impl Source for HfSource {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> SourceKind {
        SourceKind::Hf
    }

    fn revision(&self) -> &str {
        &self.revision
    }

    fn acquire(&self, cache: &Path) -> Result<AcquiredSource> {
        if self.files.is_empty() {
            bail!("hf source '{}' names no files; list every file and its sha256", self.id);
        }

        // A gated source needs a credential even before the location is known.
        // The token is held in this local, never stored on the struct and never
        // formatted into an error.
        let token = hf_token();
        if self.gated && token.is_none() {
            bail!(
                "hf source '{}' is gated and no token was supplied in HF_TOKEN or HUGGING_FACE_HUB_TOKEN",
                self.id
            );
        }

        match location_kind(&self.location) {
            Location::Local(root) => self.read_local(&root),
            Location::Remote(base) => {
                if self.offline {
                    bail!(
                        "offline: hf source '{}' refuses network access to {}",
                        self.id,
                        redact_credentials(&base)
                    );
                }
                self.read_remote(cache, &base, token.as_deref())
            }
            Location::Unsupported(scheme) => bail!(
                "hf source '{}' location {} uses unsupported scheme '{}'; use a local path, file://, http:// or https://",
                self.id,
                redact_credentials(&self.location),
                scheme
            ),
        }
    }
}

impl HfSource {
    fn read_local(&self, root: &Path) -> Result<AcquiredSource> {
        let mut acquired = Vec::new();
        for file in &self.files {
            let rel = relative_path(&self.id, &file.path)?;
            let path = root.join(&rel);
            let bytes = fs::read(&path).with_context(|| {
                format!(
                    "hf source '{}' cannot read named file '{}' at {}",
                    self.id,
                    file.path,
                    path.display()
                )
            })?;
            let got = sha256_hex(&bytes);
            if !got.eq_ignore_ascii_case(&file.sha256) {
                bail!(
                    "hf source '{}' file '{}' sha256 mismatch: expected {}, got {}",
                    self.id,
                    file.path,
                    file.sha256,
                    got
                );
            }
            acquired.push(AcquiredFile { rel_path: file.path.clone(), path, sha256: got });
        }
        Ok(AcquiredSource {
            id: self.id.clone(),
            kind: SourceKind::Hf,
            revision: self.revision.clone(),
            root: root.to_path_buf(),
            files: Some(acquired),
        })
    }

    fn read_remote(&self, cache: &Path, base: &str, token: Option<&str>) -> Result<AcquiredSource> {
        let dir = cache.join(format!("hf-{}-{}", slug(&self.id), slug(&self.revision)));

        // A warm cache is verified, so offline builds can use it and a corrupt
        // file is never accepted. A cache miss is not an error until offline.
        if let Some(files) = cached_files(&self.id, &dir, &self.files)? {
            return Ok(AcquiredSource {
                id: self.id.clone(),
                kind: SourceKind::Hf,
                revision: self.revision.clone(),
                root: dir,
                files: Some(files),
            });
        }

        if self.offline {
            bail!(
                "offline: hf source '{}' has no verified cache at {}; refusing network access to {}",
                self.id,
                dir.display(),
                redact_credentials(base)
            );
        }

        fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create hf cache {}", dir.display()))?;

        let mut acquired = Vec::new();
        for file in &self.files {
            let rel = relative_path(&self.id, &file.path)?;
            let dest = dir.join(&rel);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("cannot create {}", parent.display()))?;
            }
            let url = format!(
                "{}/resolve/{}/{}",
                base.trim_end_matches('/'),
                self.revision,
                file.path.trim_start_matches('/')
            );
            let bytes = http_get(&self.id, &url, token)?;
            let got = sha256_hex(&bytes);
            if !got.eq_ignore_ascii_case(&file.sha256) {
                bail!(
                    "hf source '{}' file '{}' sha256 mismatch: expected {}, got {}",
                    self.id,
                    file.path,
                    file.sha256,
                    got
                );
            }
            fs::write(&dest, &bytes).with_context(|| format!("cannot write {}", dest.display()))?;
            acquired.push(AcquiredFile { rel_path: file.path.clone(), path: dest, sha256: got });
        }

        Ok(AcquiredSource {
            id: self.id.clone(),
            kind: SourceKind::Hf,
            revision: self.revision.clone(),
            root: dir,
            files: Some(acquired),
        })
    }
}

enum Location {
    Local(PathBuf),
    Remote(String),
    Unsupported(String),
}

fn location_kind(location: &str) -> Location {
    if let Some(path) = location.strip_prefix("file://") {
        return Location::Local(PathBuf::from(path));
    }
    if location.starts_with("http://") || location.starts_with("https://") {
        return Location::Remote(location.trim_end_matches('/').to_string());
    }
    if let Some((scheme, _rest)) = location.split_once("://") {
        return Location::Unsupported(scheme.to_string());
    }
    Location::Local(PathBuf::from(location))
}

fn cached_files(
    id: &str,
    dir: &Path,
    files: &[SourceFileSpec],
) -> Result<Option<Vec<AcquiredFile>>> {
    let mut acquired = Vec::new();
    for file in files {
        let rel = relative_path(id, &file.path)?;
        let path = dir.join(&rel);
        let Ok(bytes) = fs::read(&path) else {
            return Ok(None);
        };
        let got = sha256_hex(&bytes);
        if !got.eq_ignore_ascii_case(&file.sha256) {
            return Ok(None);
        }
        acquired.push(AcquiredFile { rel_path: file.path.clone(), path, sha256: got });
    }
    Ok(Some(acquired))
}

fn http_get(id: &str, url: &str, token: Option<&str>) -> Result<Vec<u8>> {
    let mut request = ureq::get(url).timeout(Duration::from_secs(120));
    if let Some(token) = token {
        // The token exists only in this header, never in an error or a file.
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = request.call().map_err(|error| {
        let status = match &error {
            ureq::Error::Status(status, _) => Some(*status),
            _ => None,
        };
        if matches!(status, Some(401) | Some(403)) {
            anyhow::anyhow!(
                "hf source '{}' was refused by {} with status {}; the dataset is gated and the supplied token was rejected",
                id,
                redact_credentials(url),
                status.unwrap_or(0)
            )
        } else {
            anyhow::anyhow!(
                "hf source '{}' cannot fetch {}: {}",
                id,
                redact_credentials(url),
                redact_credentials(&error.to_string())
            )
        }
    })?;

    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .with_context(|| format!("hf source '{}' cannot read response body", id))?;
    Ok(bytes)
}

fn hf_token() -> Option<String> {
    for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(value) = std::env::var(name) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn relative_path(id: &str, raw: &str) -> Result<PathBuf> {
    if raw.trim().is_empty() {
        bail!("hf source '{}' names a file with an empty path", id);
    }
    let mut path = PathBuf::new();
    for component in Path::new(raw).components() {
        match component {
            Component::Normal(part) => path.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("hf source '{}' file '{}' must stay inside its store", id, raw)
            }
        }
    }
    if path.as_os_str().is_empty() {
        bail!("hf source '{}' names a file with an empty path", id);
    }
    Ok(path)
}

/// Filesystem-safe cache segment. Never contains a credential.
fn slug(value: &str) -> String {
    let mut out = String::new();
    for byte in value.chars() {
        if byte.is_ascii_alphanumeric() || byte == '-' || byte == '_' || byte == '.' {
            out.push(byte);
        } else {
            out.push('-');
        }
        if out.len() >= 64 {
            break;
        }
    }
    if out.is_empty() {
        out.push_str("source");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_are_redacted() {
        assert_eq!(
            redact_credentials("https://user:secret@example.com/a/b"),
            "https://***@example.com/a/b"
        );
        assert_eq!(
            redact_credentials("ssh://git@github.com/org/repo.git"),
            "ssh://***@github.com/org/repo.git"
        );
        assert_eq!(redact_credentials("https://example.com/a"), "https://example.com/a");
        assert_eq!(redact_credentials("file:///tmp/store"), "file:///tmp/store");
    }

    #[test]
    fn relative_paths_stay_inside_the_store() {
        assert!(relative_path("s", "a/b.txt").is_ok());
        assert!(relative_path("s", "../escape").is_err());
        assert!(relative_path("s", "/etc/passwd").is_err());
        assert!(relative_path("s", "").is_err());
    }
}
