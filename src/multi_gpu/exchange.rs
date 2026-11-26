//! Local-SGD weight exchange through files in a shared directory.
//!
//! A rank publishes `weights-<step>-<rank>.bin`, waits (with a deadline) until
//! every rank's file exists, and rank 0 writes `weights-<step>-avg.bin`. Every
//! rank reads the average back and verifies its header and checksum.
//!
//! Files, not sockets, because the topology is one host and `/dev/shm`: no TCP
//! all-reduce, no NCCL, no CUDA IPC. Each publish goes to a temporary file in
//! the same directory and is renamed into place, so a reader never observes a
//! half-written file.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::multi_gpu::error::MultiGpuError;
use crate::multi_gpu::fnv1a;

/// The barrier default. One minute is long enough for a 0.6B fp32 vector over
/// tmpfs on a loaded host, and short enough that a dead rank does not cost the
/// rest of the quota hour.
pub const DEFAULT_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(60);

const MAGIC: [u8; 8] = *b"FRSWGT01";
const HEADER_LEN: usize = 40;
/// The average file is written by rank 0 but is not rank 0's private weights,
/// so it carries a sentinel rank that no real rank can claim.
const AVERAGE_RANK: u32 = u32::MAX;
const POLL_INTERVAL: Duration = Duration::from_millis(2);

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Reads and writes one run's weight files in one directory.
#[derive(Debug, Clone)]
pub struct WeightExchange {
    dir: PathBuf,
    world_size: usize,
    rank: usize,
    timeout: Duration,
}

impl WeightExchange {
    pub fn new(
        dir: impl Into<PathBuf>,
        world_size: usize,
        rank: usize,
        timeout: Duration,
    ) -> Result<Self, MultiGpuError> {
        if world_size == 0 {
            return Err(MultiGpuError::ZeroWorldSize);
        }
        if rank >= world_size {
            return Err(MultiGpuError::RankOutOfRange { rank, world_size });
        }
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(|source| MultiGpuError::Io {
            path: dir.display().to_string(),
            source,
        })?;
        Ok(Self {
            dir,
            world_size,
            rank,
            timeout,
        })
    }

    pub fn with_default_timeout(
        dir: impl Into<PathBuf>,
        world_size: usize,
        rank: usize,
    ) -> Result<Self, MultiGpuError> {
        Self::new(dir, world_size, rank, DEFAULT_EXCHANGE_TIMEOUT)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn rank_path(&self, step: u64, rank: usize) -> PathBuf {
        self.dir.join(format!("weights-{}-{}.bin", step, rank))
    }

    pub fn average_path(&self, step: u64) -> PathBuf {
        self.dir.join(format!("weights-{}-avg.bin", step))
    }

    /// Write this rank's weights for `step`, atomically.
    pub fn publish(&self, step: u64, weights: &[f32]) -> Result<(), MultiGpuError> {
        let path = self.rank_path(step, self.rank);
        self.write_atomic(&path, step, self.rank as u32, weights)
    }

    /// Wait until every rank has published for `step`.
    ///
    /// On timeout the error names the missing ranks; it never waits forever.
    pub fn wait_for_step(&self, step: u64) -> Result<(), MultiGpuError> {
        let started = Instant::now();
        loop {
            let missing: Vec<usize> = (0..self.world_size)
                .filter(|rank| !self.rank_path(step, *rank).is_file())
                .collect();
            if missing.is_empty() {
                return Ok(());
            }
            if started.elapsed() >= self.timeout {
                return Err(MultiGpuError::BarrierTimeout {
                    step,
                    waited_ms: started.elapsed().as_millis(),
                    missing,
                });
            }
            let remaining = self.timeout.saturating_sub(started.elapsed());
            std::thread::sleep(remaining.min(POLL_INTERVAL));
        }
    }

    /// Wait for all ranks, average element-wise, and write the average.
    ///
    /// Rank 0 only; other ranks call [`WeightExchange::wait_for_average`].
    pub fn average_step(&self, step: u64, count: usize) -> Result<Vec<f32>, MultiGpuError> {
        if self.rank != 0 {
            return Err(MultiGpuError::NotCoordinator { rank: self.rank });
        }
        self.wait_for_step(step)?;
        let mut sum = vec![0.0f32; count];
        let expected = count as u64;
        for rank in 0..self.world_size {
            let path = self.rank_path(step, rank);
            let weights = self.read_file(&path, Some(rank as u32), step, expected)?;
            for (total, value) in sum.iter_mut().zip(weights) {
                *total += value;
            }
        }
        let divisor = self.world_size as f32;
        for value in &mut sum {
            *value /= divisor;
        }
        let path = self.average_path(step);
        self.write_atomic(&path, step, AVERAGE_RANK, &sum)?;
        Ok(sum)
    }

    /// Wait until rank 0 has written the average for `step`.
    pub fn wait_for_average(&self, step: u64) -> Result<(), MultiGpuError> {
        let path = self.average_path(step);
        let started = Instant::now();
        loop {
            if path.is_file() {
                return Ok(());
            }
            if started.elapsed() >= self.timeout {
                return Err(MultiGpuError::AverageTimeout {
                    step,
                    waited_ms: started.elapsed().as_millis(),
                });
            }
            let remaining = self.timeout.saturating_sub(started.elapsed());
            std::thread::sleep(remaining.min(POLL_INTERVAL));
        }
    }

    /// Read and verify the average file written by rank 0.
    pub fn load_average(&self, step: u64, count: usize) -> Result<Vec<f32>, MultiGpuError> {
        let path = self.average_path(step);
        self.read_file(&path, None, step, count as u64)
    }

    /// One local-SGD exchange: publish, barrier, rank 0 averages, read back.
    ///
    /// The returned vector is identical on every rank, or the call fails.
    pub fn round(&self, step: u64, local: &[f32]) -> Result<Vec<f32>, MultiGpuError> {
        self.publish(step, local)?;
        if self.rank == 0 {
            self.average_step(step, local.len())?;
        } else {
            self.wait_for_step(step)?;
        }
        self.wait_for_average(step)?;
        self.load_average(step, local.len())
    }

    /// Remove this step's files. Best effort: a missing file is not an error.
    pub fn cleanup_step(&self, step: u64) -> Result<(), MultiGpuError> {
        let mut paths: Vec<PathBuf> = (0..self.world_size)
            .map(|rank| self.rank_path(step, rank))
            .collect();
        paths.push(self.average_path(step));
        for path in paths {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => {
                    return Err(MultiGpuError::Io {
                        path: path.display().to_string(),
                        source,
                    })
                }
            }
        }
        Ok(())
    }

    fn write_atomic(
        &self,
        path: &Path,
        step: u64,
        rank: u32,
        weights: &[f32],
    ) -> Result<(), MultiGpuError> {
        let mut data = Vec::with_capacity(weights.len() * 4);
        for weight in weights {
            data.extend_from_slice(&weight.to_le_bytes());
        }
        let header = encode_header(
            self.world_size as u32,
            rank,
            step,
            weights.len() as u64,
            fnv1a(&data),
        );
        let mut bytes = Vec::with_capacity(HEADER_LEN + data.len());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);

        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("weights.bin");
        let temp = self.dir.join(format!(
            ".{}.tmp-{}-{}",
            file_name,
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));

        let write_result = (|| -> std::io::Result<()> {
            let mut file = fs::File::create(&temp)?;
            file.write_all(&bytes)?;
            // tmpfs keeps the bytes in memory; sync_all still orders the rename
            // after the content for a concurrent reader.
            file.sync_all()?;
            fs::rename(&temp, path)
        })();

        if let Err(source) = write_result {
            let _ = fs::remove_file(&temp);
            return Err(MultiGpuError::Io {
                path: path.display().to_string(),
                source,
            });
        }
        Ok(())
    }

    /// Read one weight file and verify every header field plus the checksum.
    ///
    /// `expected_rank == None` means the average file, which carries
    /// [`AVERAGE_RANK`].
    fn read_file(
        &self,
        path: &Path,
        expected_rank: Option<u32>,
        expected_step: u64,
        expected_count: u64,
    ) -> Result<Vec<f32>, MultiGpuError> {
        let mut file = fs::File::open(path).map_err(|source| MultiGpuError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|source| MultiGpuError::Io {
                path: path.display().to_string(),
                source,
            })?;
        let display = path.display().to_string();

        if bytes.len() < HEADER_LEN {
            return Err(MultiGpuError::ShortFile {
                path: display,
                len: bytes.len() as u64,
                header: HEADER_LEN,
            });
        }
        if &bytes[..8] != MAGIC.as_slice() {
            return Err(MultiGpuError::BadMagic { path: display });
        }
        let world_size = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let rank = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let step = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let count = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let checksum = u64::from_le_bytes(bytes[32..40].try_into().unwrap());

        let check = |field: &'static str,
                     expected: String,
                     actual: String|
         -> Result<(), MultiGpuError> {
            if expected == actual {
                Ok(())
            } else {
                Err(MultiGpuError::HeaderMismatch {
                    path: display.clone(),
                    field,
                    expected,
                    actual,
                })
            }
        };
        check(
            "world_size",
            self.world_size.to_string(),
            world_size.to_string(),
        )?;
        let expected_rank = expected_rank.unwrap_or(AVERAGE_RANK);
        check("rank", expected_rank.to_string(), rank.to_string())?;
        check("step", expected_step.to_string(), step.to_string())?;
        check("element count", expected_count.to_string(), count.to_string())?;

        let data = &bytes[HEADER_LEN..];
        if data.len() % 4 != 0 {
            return Err(MultiGpuError::ElementCountMismatch {
                path: display,
                expected: expected_count,
                actual: (data.len() / 4) as u64,
            });
        }
        if data.len() as u64 != count.saturating_mul(4) {
            return Err(MultiGpuError::ElementCountMismatch {
                path: display,
                expected: count,
                actual: (data.len() / 4) as u64,
            });
        }
        let actual_checksum = fnv1a(data);
        if actual_checksum != checksum {
            return Err(MultiGpuError::ChecksumMismatch {
                path: display,
                expected: checksum,
                actual: actual_checksum,
            });
        }

        Ok(data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }
}

fn encode_header(
    world_size: u32,
    rank: u32,
    step: u64,
    count: u64,
    checksum: u64,
) -> [u8; HEADER_LEN] {
    let mut header = [0u8; HEADER_LEN];
    header[..8].copy_from_slice(&MAGIC);
    header[8..12].copy_from_slice(&world_size.to_le_bytes());
    header[12..16].copy_from_slice(&rank.to_le_bytes());
    header[16..24].copy_from_slice(&step.to_le_bytes());
    header[24..32].copy_from_slice(&count.to_le_bytes());
    header[32..40].copy_from_slice(&checksum.to_le_bytes());
    header
}

#[cfg(test)]
mod tests {
    use super::WeightExchange;
    use crate::multi_gpu::error::MultiGpuError;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "ferrisres-multi-gpu-{}-{}",
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

    fn exchange(dir: &Path, world_size: usize, rank: usize, timeout: Duration) -> WeightExchange {
        WeightExchange::new(dir, world_size, rank, timeout).expect("construct exchange")
    }

    #[test]
    fn average_equals_the_arithmetic_mean_within_1e_6() {
        let temp = TempDir::new("average");
        let world_size = 4;
        let local: Vec<Vec<f32>> = (0..world_size)
            .map(|rank| {
                (0..64)
                    .map(|i| rank as f32 * 1.5 + i as f32 * 0.25 - 7.0)
                    .collect()
            })
            .collect();

        let results: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..world_size)
                .map(|rank| {
                    let weights = &local[rank];
                    let dir = temp.path();
                    scope.spawn(move || {
                        exchange(dir, world_size, rank, Duration::from_secs(10))
                            .round(3, weights)
                            .expect("exchange round")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("rank thread"))
                .collect()
        });

        for i in 0..64 {
            let expected: f32 =
                (0..world_size).map(|rank| local[rank][i]).sum::<f32>() / world_size as f32;
            for (rank, row) in results.iter().enumerate() {
                assert!(
                    (row[i] - expected).abs() <= 1e-6,
                    "rank {rank} element {i}: {} != {}",
                    row[i],
                    expected
                );
            }
        }
        // Every rank read the same file, so every rank must agree bit for bit.
        for row in &results[1..] {
            assert_eq!(row, &results[0]);
        }
    }

    #[test]
    fn a_missing_rank_times_out_and_is_named() {
        let temp = TempDir::new("missing");
        let world_size = 3;
        let rank0 = exchange(temp.path(), world_size, 0, Duration::from_millis(150));

        std::thread::scope(|scope| {
            let dir = temp.path();
            scope.spawn(move || {
                let rank1 = exchange(dir, world_size, 1, Duration::from_millis(150));
                rank1.publish(0, &[1.0, 2.0]).expect("rank 1 publish");
            });

            let started = Instant::now();
            let err = rank0.round(0, &[0.5, 0.5]).unwrap_err();
            let elapsed = started.elapsed();
            assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
            match err {
                MultiGpuError::BarrierTimeout { missing, .. } => {
                    assert_eq!(missing, vec![2], "must name only the missing rank");
                }
                other => panic!("wrong refusal: {other}"),
            }
        });
    }

    #[test]
    fn a_corrupt_checksum_is_refused() {
        let temp = TempDir::new("checksum");
        let rank0 = exchange(temp.path(), 2, 0, Duration::from_secs(5));
        let rank1 = exchange(temp.path(), 2, 1, Duration::from_secs(5));
        rank0.publish(1, &[1.0, 2.0, 3.0]).unwrap();
        rank1.publish(1, &[4.0, 5.0, 6.0]).unwrap();

        let path = rank1.rank_path(1, 1);
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        fs::write(&path, &bytes).unwrap();

        match rank0.average_step(1, 3).unwrap_err() {
            MultiGpuError::ChecksumMismatch { expected, actual, .. } => {
                assert_ne!(expected, actual);
            }
            other => panic!("wrong refusal: {other}"),
        }
    }

    #[test]
    fn an_element_count_mismatch_is_refused() {
        let temp = TempDir::new("count");
        let rank0 = exchange(temp.path(), 2, 0, Duration::from_secs(5));
        let rank1 = exchange(temp.path(), 2, 1, Duration::from_secs(5));
        rank0.publish(2, &[1.0, 2.0, 3.0]).unwrap();
        rank1.publish(2, &[1.0, 2.0]).unwrap();

        let text = rank0.average_step(2, 3).unwrap_err().to_string();
        assert!(
            text.contains("element count") || text.contains("elements"),
            "must name the count: {text}"
        );
    }

    #[test]
    fn publish_leaves_no_temporary_files() {
        let temp = TempDir::new("atomic");
        let rank0 = exchange(temp.path(), 1, 0, Duration::from_secs(5));
        let avg = rank0.round(0, &[0.25, 0.5, 0.75]).unwrap();
        assert_eq!(avg, vec![0.25, 0.5, 0.75]);

        let leftovers: Vec<_> = fs::read_dir(temp.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with('.') || name.contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "temporary files left: {leftovers:?}");
    }

    #[test]
    fn cleanup_removes_the_step_and_tolerates_missing_files() {
        let temp = TempDir::new("cleanup");
        let rank0 = exchange(temp.path(), 1, 0, Duration::from_secs(5));
        rank0.round(4, &[1.0]).unwrap();
        rank0.cleanup_step(4).unwrap();
        assert!(!rank0.rank_path(4, 0).exists());
        assert!(!rank0.average_path(4).exists());
        // A second cleanup is a no-op, not an error.
        rank0.cleanup_step(4).unwrap();
    }

    #[test]
    fn out_of_range_rank_is_refused() {
        let temp = TempDir::new("range");
        assert!(matches!(
            WeightExchange::new(temp.path(), 2, 2, Duration::from_secs(1)),
            Err(MultiGpuError::RankOutOfRange { .. })
        ));
    }
}
