//! Process-per-GPU launcher.
//!
//! The launcher discovers adapters, refuses to start fewer workers than the
//! requested world size, spawns one child per adapter with
//! `WGPU_ADAPTER_INDEX`, and records per-rank evidence. It is split so the core
//! ([`launch`]) takes an explicit adapter list and an arbitrary child command;
//! the GPU enumeration lives only in [`run_workers`], which the CLI calls.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::compute::pipeline::{list_adapters, AdapterListing};
use crate::multi_gpu::error::MultiGpuError;

const POLL_INTERVAL: Duration = Duration::from_millis(20);
const TERM_GRACE: Duration = Duration::from_millis(200);

/// One adapter a worker may take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterSlot {
    /// Position among *usable* adapters, which is what `WGPU_ADAPTER_INDEX`
    /// selects after `WgpuCompute::pick_adapter` filters software adapters.
    /// It is not necessarily [`AdapterListing::index`].
    pub selector: usize,
    pub name: String,
    /// PCI bus id, when the backend exposed one. Two cards of the same model
    /// share `name`, so this is what tells their evidence apart.
    pub bus_id: Option<String>,
}

/// The adapters a launch may use, in the same order `WGPU_ADAPTER_INDEX`
/// counts them.
pub fn usable_adapters(listings: &[AdapterListing], allow_software: bool) -> Vec<AdapterSlot> {
    listings
        .iter()
        .filter(|adapter| allow_software || !adapter.software)
        .enumerate()
        .map(|(selector, adapter)| AdapterSlot {
            selector,
            name: adapter.name.clone(),
            bus_id: adapter.bus_id.clone(),
        })
        .collect()
}

/// The synthetic slots a `workers --cpu` run uses.
///
/// There is no GPU per rank on the f32 path, so no adapter is enumerated and
/// no `WGPU_ADAPTER_INDEX` is exported. The slot exists only so the launcher's
/// capacity check, evidence and report keep one shape for both paths.
pub fn cpu_adapters(world_size: usize) -> Vec<AdapterSlot> {
    (0..world_size)
        .map(|rank| AdapterSlot {
            selector: rank,
            name: "cpu".to_string(),
            bus_id: None,
        })
        .collect()
}

/// Refuse a world size the host cannot satisfy, naming the count and adapters.
pub fn check_capacity(
    world_size: usize,
    adapters: &[AdapterSlot],
) -> Result<(), MultiGpuError> {
    if world_size == 0 {
        return Err(MultiGpuError::ZeroWorldSize);
    }
    if adapters.len() < world_size {
        let names = if adapters.is_empty() {
            "none".to_string()
        } else {
            adapters
                .iter()
                .map(|adapter| adapter.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(MultiGpuError::NotEnoughAdapters {
            requested: world_size,
            available: adapters.len(),
            adapters: names,
        });
    }
    Ok(())
}

/// Where evidence goes and how long children may run.
#[derive(Debug, Clone)]
pub struct LaunchOptions {
    pub evidence_dir: PathBuf,
    /// `None` means no launcher-side deadline; the barrier in each worker and
    /// the training budget are the real limits. Set one for scripted runs.
    pub child_timeout: Option<Duration>,
    /// Export `WGPU_ADAPTER_INDEX`. False on the CPU path, where no child may
    /// be pointed at an adapter it does not have.
    pub export_adapter_index: bool,
}

impl LaunchOptions {
    pub fn new(evidence_dir: impl Into<PathBuf>) -> Self {
        Self {
            evidence_dir: evidence_dir.into(),
            child_timeout: None,
            export_adapter_index: true,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.child_timeout = Some(timeout);
        self
    }

    pub fn with_adapter_index(mut self, export: bool) -> Self {
        self.export_adapter_index = export;
        self
    }
}

/// What one rank reported.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerEvidence {
    pub rank: usize,
    /// Position among usable adapters; `WGPU_ADAPTER_INDEX` for this child.
    pub adapter_index: usize,
    pub adapter_name: String,
    /// PCI bus id of the adapter this rank took, when one was available.
    pub adapter_bus_id: Option<String>,
    pub world_size: usize,
    /// Exit code; `None` when the process was signalled or never started.
    pub exit_status: Option<i32>,
    pub success: bool,
    pub elapsed_ms: u64,
    /// Read from `rank-<rank>-tokens-per-second.txt` when the child wrote it.
    pub tokens_per_second: Option<f64>,
}

/// Every rank's evidence plus the launch summary.
#[derive(Debug, Clone, Serialize)]
pub struct LaunchReport {
    pub world_size: usize,
    pub adapters: Vec<String>,
    pub workers: Vec<WorkerEvidence>,
}

impl LaunchReport {
    /// How many different adapter names this run used. A run that claims two
    /// GPUs must report two.
    pub fn distinct_adapter_names(&self) -> usize {
        let mut names: Vec<&str> = self
            .workers
            .iter()
            .map(|worker| worker.adapter_name.as_str())
            .collect();
        names.sort_unstable();
        names.dedup();
        names.len()
    }

    pub fn worker(&self, rank: usize) -> Option<&WorkerEvidence> {
        self.workers.iter().find(|worker| worker.rank == rank)
    }
}

struct ChildSlot {
    rank: usize,
    adapter: AdapterSlot,
    started: Instant,
    child: Child,
}

/// Spawn one child per adapter and collect evidence.
///
/// `program` and `args` are the child command; `--rank <i> --world-size <n>`
/// are appended and `WGPU_ADAPTER_INDEX` is set from the slot's selector. The
/// first child to fail ends the run: the rest are terminated by process group,
/// and the error names the failed rank.
pub fn launch(
    adapters: &[AdapterSlot],
    world_size: usize,
    program: &OsStr,
    args: &[OsString],
    options: &LaunchOptions,
) -> Result<LaunchReport, MultiGpuError> {
    check_capacity(world_size, adapters)?;
    let dir = &options.evidence_dir;
    fs::create_dir_all(dir).map_err(|source| MultiGpuError::Io {
        path: dir.display().to_string(),
        source,
    })?;

    // One id for the whole launch scopes the init-identity reference file, so
    // a reused exchange directory cannot feed a rank a previous run's hash.
    let run_id = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0)
    );

    let mut children: Vec<ChildSlot> = Vec::with_capacity(world_size);
    for (rank, adapter) in adapters.iter().take(world_size).enumerate() {
        let mut command = Command::new(program);
        command
            .args(args)
            .arg("--rank")
            .arg(rank.to_string())
            .arg("--world-size")
            .arg(world_size.to_string())
            .env("FERRISRES_RUN_ID", &run_id)
            .env("FERRISRES_RANK", rank.to_string())
            .env("FERRISRES_WORLD_SIZE", world_size.to_string())
            .env("FERRISRES_EVIDENCE_DIR", dir)
            .env("FERRISRES_TOKENS_FILE", tokens_path(dir, rank))
            .stdin(Stdio::null());
        if options.export_adapter_index {
            command.env("WGPU_ADAPTER_INDEX", adapter.selector.to_string());
        }
        // A worker gets its own process group so that a launcher-side terminate
        // also reaches a shell script's grandchildren, not only the shell.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }

        match command.spawn() {
            Ok(child) => children.push(ChildSlot {
                rank,
                adapter: adapter.clone(),
                started: Instant::now(),
                child,
            }),
            Err(source) => {
                for slot in &mut children {
                    terminate_and_reap(&mut slot.child);
                }
                return Err(MultiGpuError::SpawnFailed {
                    rank,
                    program: program.to_string_lossy().into_owned(),
                    source,
                });
            }
        }
    }

    let deadline = options.child_timeout.map(|timeout| Instant::now() + timeout);
    let launch_started = Instant::now();
    let mut exit_info: Vec<Option<(Option<i32>, bool)>> = vec![None; world_size];
    let mut failure: Option<(usize, String)> = None;

    loop {
        let mut running = 0usize;
        for slot in children.iter_mut() {
            if exit_info[slot.rank].is_some() {
                continue;
            }
            match slot.child.try_wait() {
                Ok(Some(status)) => {
                    exit_info[slot.rank] = Some((status.code(), status.success()));
                    if !status.success() && failure.is_none() {
                        failure = Some((slot.rank, status.to_string()));
                    }
                }
                Ok(None) => running += 1,
                Err(source) => {
                    exit_info[slot.rank] = Some((None, false));
                    if failure.is_none() {
                        failure = Some((slot.rank, format!("wait failed: {source}")));
                    }
                }
            }
        }

        if failure.is_some() || running == 0 {
            break;
        }

        if let Some(deadline) = deadline {
            if Instant::now() >= deadline {
                let mut still_running = Vec::new();
                for slot in children.iter_mut() {
                    if exit_info[slot.rank].is_none() {
                        still_running.push(slot.rank);
                        exit_info[slot.rank] = terminate_and_reap(&mut slot.child)
                            .map(|status| (status.code(), status.success()))
                            .or(Some((None, false)));
                    }
                }
                let report = build_report(world_size, adapters, &children, &exit_info, dir);
                let _ = write_evidence(dir, &report);
                return Err(MultiGpuError::LaunchTimeout {
                    waited_ms: launch_started.elapsed().as_millis(),
                    running: still_running,
                    evidence_dir: dir.display().to_string(),
                });
            }
        }
        std::thread::sleep(POLL_INTERVAL);
    }

    let mut killed = Vec::new();
    for slot in children.iter_mut() {
        if exit_info[slot.rank].is_none() {
            killed.push(slot.rank);
            exit_info[slot.rank] = terminate_and_reap(&mut slot.child)
                .map(|status| (status.code(), status.success()))
                .or(Some((None, false)));
        }
    }

    let report = build_report(world_size, adapters, &children, &exit_info, dir);
    write_evidence(dir, &report)?;

    match failure {
        Some((rank, status)) => Err(MultiGpuError::WorkerFailed {
            rank,
            status,
            killed,
            evidence_dir: dir.display().to_string(),
        }),
        None => Ok(report),
    }
}

/// CLI entry point: enumerate adapters, then [`launch`] the given command.
///
/// Prints one line per rank on success and returns the report; every refusal
/// travels up as the named error so `main` exits non-zero.
pub async fn run_workers(
    cpu: bool,
    world_size: usize,
    allow_software: bool,
    evidence_dir: Option<String>,
    command: Vec<String>,
) -> Result<LaunchReport, MultiGpuError> {
    if command.is_empty() {
        return Err(MultiGpuError::EmptyCommand);
    }
    let (adapters, export_adapter_index) = if cpu {
        // No GPU is involved on this path, so enumerating adapters (and
        // needing a Vulkan ICD) would refuse a run the host can perform.
        (cpu_adapters(world_size), false)
    } else {
        let listings = list_adapters()
            .await
            .map_err(|err| MultiGpuError::Enumeration(err.to_string()))?;
        (usable_adapters(&listings, allow_software), true)
    };
    check_capacity(world_size, &adapters)?;

    let dir = evidence_dir.map(PathBuf::from).unwrap_or_else(|| {
        std::env::temp_dir().join(format!("ferrisres-evidence-{}", std::process::id()))
    });
    let (program, args) = command_parts(&command)?;
    let report = launch(
        &adapters,
        world_size,
        &program,
        &args,
        &LaunchOptions::new(&dir).with_adapter_index(export_adapter_index),
    )?;

    for worker in &report.workers {
        let tokens = worker
            .tokens_per_second
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let bus = worker.adapter_bus_id.as_deref().unwrap_or("-");
        println!(
            "rank {}/{} adapter {} ({}, bus {}) exit {} elapsed {} ms tokens/s {}",
            worker.rank,
            worker.world_size,
            worker.adapter_index,
            worker.adapter_name,
            bus,
            worker
                .exit_status
                .map(|code| code.to_string())
                .unwrap_or_else(|| "signal".to_string()),
            worker.elapsed_ms,
            tokens
        );
    }
    println!("evidence: {}", dir.display());
    Ok(report)
}

/// Split the CLI's `-- <program> [args...]` vector.
fn command_parts(command: &[String]) -> Result<(OsString, Vec<OsString>), MultiGpuError> {
    let (program, args) = command.split_first().ok_or(MultiGpuError::EmptyCommand)?;
    Ok((
        OsString::from(program),
        args.iter().map(OsString::from).collect(),
    ))
}

fn tokens_path(dir: &Path, rank: usize) -> PathBuf {
    dir.join(format!("rank-{}-tokens-per-second.txt", rank))
}

fn read_tokens(dir: &Path, rank: usize) -> Option<f64> {
    let text = fs::read_to_string(tokens_path(dir, rank)).ok()?;
    text.trim().parse().ok()
}

fn build_report(
    world_size: usize,
    adapters: &[AdapterSlot],
    children: &[ChildSlot],
    exit_info: &[Option<(Option<i32>, bool)>],
    dir: &Path,
) -> LaunchReport {
    let workers = children
        .iter()
        .map(|slot| {
            let (exit_status, success) = exit_info[slot.rank].unwrap_or((None, false));
            WorkerEvidence {
                rank: slot.rank,
                adapter_index: slot.adapter.selector,
                adapter_name: slot.adapter.name.clone(),
                adapter_bus_id: slot.adapter.bus_id.clone(),
                world_size,
                exit_status,
                success,
                elapsed_ms: slot.started.elapsed().as_millis() as u64,
                tokens_per_second: read_tokens(dir, slot.rank),
            }
        })
        .collect();
    LaunchReport {
        world_size,
        adapters: adapters
            .iter()
            .take(world_size)
            .map(|adapter| adapter.name.clone())
            .collect(),
        workers,
    }
}

fn write_evidence(dir: &Path, report: &LaunchReport) -> Result<(), MultiGpuError> {
    for worker in &report.workers {
        let path = dir.join(format!("rank-{}.json", worker.rank));
        let bytes =
            serde_json::to_vec_pretty(worker).map_err(|err| MultiGpuError::Json(err.to_string()))?;
        fs::write(&path, bytes).map_err(|source| MultiGpuError::Io {
            path: path.display().to_string(),
            source,
        })?;
    }
    let path = dir.join("launch.json");
    let bytes =
        serde_json::to_vec_pretty(report).map_err(|err| MultiGpuError::Json(err.to_string()))?;
    fs::write(&path, bytes).map_err(|source| MultiGpuError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(())
}

/// Stop a child's whole process group, then reap it.
///
/// SIGTERM first so a trainer can flush, SIGKILL after a short grace. The group
/// matters because the child is usually a shell or a wrapper whose real work is
/// a grandchild.
fn terminate_and_reap(child: &mut Child) -> Option<ExitStatus> {
    #[cfg(unix)]
    {
        signal_group(child.id(), 15);
        let deadline = Instant::now() + TERM_GRACE;
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(status)) => return Some(status),
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(_) => break,
            }
        }
        signal_group(child.id(), 9);
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
    child.wait().ok()
}

#[cfg(unix)]
fn signal_group(pid: u32, signal: i32) {
    // No libc crate: one symbol is enough, and `kill` with a negated pid is the
    // only way std exposes group delivery (it does not). `extern` without
    // `unsafe` keeps the declared 1.75 MSRV; edition 2021 accepts it.
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    unsafe {
        let _ = kill(-(pid as i32), signal);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        check_capacity, launch, usable_adapters, AdapterSlot, LaunchOptions,
    };
    use crate::compute::pipeline::AdapterListing;
    use crate::multi_gpu::error::MultiGpuError;
    use std::ffi::{OsStr, OsString};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "ferrisres-launcher-{}-{}",
                name,
                std::process::id()
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

    fn slot(selector: usize, name: &str) -> AdapterSlot {
        AdapterSlot {
            selector,
            name: name.to_string(),
            bus_id: None,
        }
    }

    fn listing(index: usize, name: &str, software: bool) -> AdapterListing {
        AdapterListing {
            index,
            name: name.to_string(),
            kind: "Discrete".to_string(),
            backend: "vulkan".to_string(),
            max_buffer: 0,
            max_immediate: 0,
            software,
            dedicated_vram: None,
            shared_memory: None,
            bus_id: None,
        }
    }

    #[test]
    fn selector_counts_usable_adapters_not_listing_indexes() {
        let listings = vec![
            listing(0, "GPU A", false),
            listing(1, "llvmpipe", true),
            listing(2, "GPU B", false),
        ];
        let usable = usable_adapters(&listings, false);
        assert_eq!(usable, vec![slot(0, "GPU A"), slot(1, "GPU B")]);
        let with_software = usable_adapters(&listings, true);
        assert_eq!(with_software.len(), 3);
        assert_eq!(with_software[1].name, "llvmpipe");
    }

    #[test]
    fn world_three_with_two_adapters_is_refused_by_name() {
        let adapters = vec![slot(0, "FakeGPU-A"), slot(1, "FakeGPU-B")];
        let err = check_capacity(3, &adapters).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("world size 3"), "must name the request: {text}");
        assert!(text.contains("2 usable adapter"), "must name the count: {text}");
        assert!(text.contains("FakeGPU-A") && text.contains("FakeGPU-B"));
        assert!(matches!(
            err,
            MultiGpuError::NotEnoughAdapters {
                requested: 3,
                available: 2,
                ..
            }
        ));
    }

    #[cfg(unix)]
    #[test]
    fn two_fake_adapters_spawn_two_children_and_aggregate_evidence() {
        let temp = TempDir::new("two");
        let adapters = vec![slot(0, "FakeGPU-A"), slot(1, "FakeGPU-B")];
        let script = r#"echo "$WGPU_ADAPTER_INDEX $1 $3" > "$FERRISRES_EVIDENCE_DIR/seen-$1.txt""#;
        let args = [OsString::from("-c"), OsString::from(script)];

        let report = launch(
            &adapters,
            2,
            OsStr::new("/bin/sh"),
            &args,
            &LaunchOptions::new(temp.path()),
        )
        .expect("launch two fake adapters");

        assert_eq!(report.workers.len(), 2);
        assert_eq!(report.distinct_adapter_names(), 2);
        for worker in &report.workers {
            assert!(worker.success, "rank {} failed", worker.rank);
            assert_eq!(worker.exit_status, Some(0));
            assert_eq!(worker.world_size, 2);
        }
        assert_eq!(report.worker(0).unwrap().adapter_name, "FakeGPU-A");
        assert_eq!(report.worker(1).unwrap().adapter_name, "FakeGPU-B");

        // The children saw their own adapter index, rank and world size.
        let seen0 = fs::read_to_string(temp.path().join("seen-0.txt")).unwrap();
        let seen1 = fs::read_to_string(temp.path().join("seen-1.txt")).unwrap();
        assert_eq!(seen0.trim(), "0 0 2");
        assert_eq!(seen1.trim(), "1 1 2");

        assert!(temp.path().join("rank-0.json").is_file());
        assert!(temp.path().join("rank-1.json").is_file());
        assert!(temp.path().join("launch.json").is_file());
    }

    #[cfg(unix)]
    #[test]
    fn one_failing_child_fails_the_run_and_kills_the_rest() {
        let temp = TempDir::new("fail");
        let adapters = vec![slot(0, "FakeGPU-A"), slot(1, "FakeGPU-B")];
        let script = r#"if [ "$1" = "1" ]; then exit 7; fi; sleep 30"#;
        let args = [OsString::from("-c"), OsString::from(script)];

        let started = Instant::now();
        let err = launch(
            &adapters,
            2,
            OsStr::new("/bin/sh"),
            &args,
            &LaunchOptions::new(temp.path()),
        )
        .expect_err("rank 1 must fail the run");
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "the surviving rank must be killed, not awaited: {elapsed:?}"
        );

        match err {
            MultiGpuError::WorkerFailed { rank, killed, .. } => {
                assert_eq!(rank, 1, "must name the failed rank");
                assert_eq!(killed, vec![0], "must name the terminated rank");
            }
            other => panic!("wrong refusal: {other}"),
        }
        // Evidence is still written, with the failure recorded.
        let rank1: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(temp.path().join("rank-1.json")).unwrap())
                .unwrap();
        assert_eq!(rank1["success"], serde_json::json!(false));
        assert_eq!(rank1["exit_status"], serde_json::json!(7));
    }

    #[cfg(unix)]
    #[test]
    fn tokens_per_second_is_read_when_the_child_writes_it() {
        let temp = TempDir::new("tokens");
        let adapters = vec![slot(0, "FakeGPU-A"), slot(1, "FakeGPU-B")];
        let script = r#"echo 123.5 > "$FERRISRES_TOKENS_FILE""#;
        let args = [OsString::from("-c"), OsString::from(script)];

        let report = launch(
            &adapters,
            2,
            OsStr::new("/bin/sh"),
            &args,
            &LaunchOptions::new(temp.path()),
        )
        .unwrap();

        for worker in &report.workers {
            assert_eq!(worker.tokens_per_second, Some(123.5));
        }
    }

    #[cfg(unix)]
    #[test]
    fn cpu_mode_spawns_children_without_an_adapter_index() {
        let temp = TempDir::new("cpu");
        let adapters = super::cpu_adapters(2);
        let script = r#"echo "${WGPU_ADAPTER_INDEX:-none} $1 $3" > "$FERRISRES_EVIDENCE_DIR/seen-$1.txt""#;
        let args = [OsString::from("-c"), OsString::from(script)];
        let options = LaunchOptions::new(temp.path()).with_adapter_index(false);

        let report = launch(&adapters, 2, OsStr::new("/bin/sh"), &args, &options)
            .expect("launch two cpu ranks");

        assert_eq!(report.workers.len(), 2);
        for worker in &report.workers {
            assert_eq!(worker.adapter_name, "cpu");
            assert_eq!(worker.adapter_bus_id, None);
        }
        for rank in 0..2 {
            let seen = fs::read_to_string(temp.path().join(format!("seen-{rank}.txt"))).unwrap();
            assert_eq!(seen.trim(), format!("none {rank} 2"));
        }
    }

    #[cfg(unix)]
    #[test]
    fn bus_id_is_carried_from_the_listing_into_the_evidence() {
        let temp = TempDir::new("bus");
        let mut a = listing(0, "NVIDIA T4", false);
        a.bus_id = Some("0000:3b:00.0".to_string());
        let mut b = listing(1, "NVIDIA T4", false);
        b.bus_id = Some("0000:5e:00.0".to_string());
        let adapters = usable_adapters(&[a, b], false);
        assert_eq!(adapters[0].name, adapters[1].name);
        assert_eq!(adapters[0].bus_id.as_deref(), Some("0000:3b:00.0"));
        assert_eq!(adapters[1].bus_id.as_deref(), Some("0000:5e:00.0"));

        let args = [OsString::from("-c"), OsString::from("true")];
        let report = launch(&adapters, 2, OsStr::new("/bin/sh"), &args, &LaunchOptions::new(temp.path()))
            .unwrap();
        assert_eq!(report.worker(0).unwrap().adapter_bus_id.as_deref(), Some("0000:3b:00.0"));
        assert_eq!(report.worker(1).unwrap().adapter_bus_id.as_deref(), Some("0000:5e:00.0"));

        let written: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(temp.path().join("rank-1.json")).unwrap())
                .unwrap();
        assert_eq!(written["adapter_bus_id"], serde_json::json!("0000:5e:00.0"));
    }

    #[cfg(unix)]
    #[test]
    fn a_launcher_timeout_terminates_children() {
        let temp = TempDir::new("timeout");
        let adapters = vec![slot(0, "FakeGPU-A")];
        let args = [
            OsString::from("-c"),
            OsString::from("sleep 30"),
        ];
        let options = LaunchOptions::new(temp.path()).with_timeout(Duration::from_millis(200));

        let started = Instant::now();
        let err = launch(&adapters, 1, OsStr::new("/bin/sh"), &args, &options)
            .expect_err("must time out");
        assert!(started.elapsed() < Duration::from_secs(5));
        match err {
            MultiGpuError::LaunchTimeout { running, .. } => assert_eq!(running, vec![0]),
            other => panic!("wrong refusal: {other}"),
        }
    }

    #[test]
    fn an_empty_command_is_refused() {
        assert!(matches!(
            super::command_parts(&[]),
            Err(MultiGpuError::EmptyCommand)
        ));
        let (program, args) =
            super::command_parts(&["ferrisres".to_string(), "train".to_string()]).unwrap();
        assert_eq!(program, OsString::from("ferrisres"));
        assert_eq!(args, vec![OsString::from("train")]);
    }

    /// Run manually on a host with a Vulkan ICD:
    ///
    /// ```text
    /// VK_ICD_FILENAMES=... cargo test --lib multi_gpu -- --ignored --nocapture
    /// ```
    ///
    /// This host has one Intel adapter, so it can only prove enumeration, not a
    /// world-size-2 launch.
    #[test]
    #[ignore = "needs a Vulkan ICD; this host has one adapter, not two"]
    fn launcher_enumerates_this_hosts_adapters() {
        let listings =
            pollster::block_on(super::list_adapters()).expect("enumerate adapters");
        let adapters = usable_adapters(&listings, false);
        for adapter in &adapters {
            eprintln!("#{} {} selector={}", adapter.selector, adapter.name, adapter.selector);
        }
        assert!(!adapters.is_empty(), "no usable adapter enumerated");
        assert!(adapters.iter().any(|a| a.name.contains("Intel")));
    }
}
