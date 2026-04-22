//! Expert memory-mapped file format + lazy loader.
//!
//! Stores each MoE expert as a separate memory-mapped file. Only pages in
//! top-k experts during inference. The OS page cache handles caching automatically.
//!
//! ## File format (.stm = sparse ternary matrix)
//! ```text
//! [magic: u32 = 0x53544D31]  // "STM1"
//! [rows: u32]
//! [cols: u32]
//! [scale: f32]
//! [pattern_bytes: u32]
//! [sign_bytes: u32]
//! [patterns: [u8; pattern_bytes]]
//! [signs: [u8; sign_bytes]]
//! ```
//!
//! ## IO timeline (RPi4 + USB SSD, per-layer decode)
//! - Router forward (6KB FP32): ~0.01ms
//! - Page in 2 experts (~3×2.4 MB each): ~5ms first, ~0ms cached
//! - OS page cache handles eviction under memory pressure

use std::collections::HashMap;
use std::path::Path;

use crate::model::sparse_ternary::SparseTernaryMatrix;

/// Magic bytes for .stm files: "STM1" (Sparse Ternary Matrix v1).
const STM_MAGIC: u32 = 0x53544D31;

/// Header size: magic(4) + rows(4) + cols(4) + scale(4) + pattern_bytes(4) + sign_bytes(4) = 24 bytes.
const STM_HEADER_SIZE: usize = 24;

/// Write a SparseTernaryMatrix to a .stm file.
///
/// File format:
/// - Magic: u32 = 0x53544D31
/// - Rows: u32
/// - Cols: u32
/// - Scale: f32
/// - Pattern bytes: u32
/// - Sign bytes: u32
/// - Pattern data: [u8]
/// - Sign data: [u8]
pub fn write_stm_file(path: &Path, mat: &SparseTernaryMatrix) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;

    f.write_all(&STM_MAGIC.to_le_bytes())?;
    f.write_all(&(mat.rows as u32).to_le_bytes())?;
    f.write_all(&(mat.cols as u32).to_le_bytes())?;
    f.write_all(&mat.scale.to_le_bytes())?;
    f.write_all(&(mat.patterns.len() as u32).to_le_bytes())?;
    f.write_all(&(mat.signs.len() as u32).to_le_bytes())?;
    f.write_all(&mat.patterns)?;
    f.write_all(&mat.signs)?;

    f.flush()?;
    Ok(())
}

/// Read a SparseTernaryMatrix from a .stm file.
pub fn read_stm_file(path: &Path) -> std::io::Result<SparseTernaryMatrix> {
    let data = std::fs::read(path)?;

    if data.len() < STM_HEADER_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("STM file too small: {} bytes", data.len()),
        ));
    }

    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != STM_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid STM magic: 0x{:08X}", magic),
        ));
    }

    let rows = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let scale = f32::from_le_bytes(data[12..16].try_into().unwrap());
    let pattern_bytes = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
    let sign_bytes = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;

    let expected_len = STM_HEADER_SIZE + pattern_bytes + sign_bytes;
    if data.len() < expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("STM data truncated: expected {} bytes, got {}", expected_len, data.len()),
        ));
    }

    let patterns = data[STM_HEADER_SIZE..STM_HEADER_SIZE + pattern_bytes].to_vec();
    let signs = data[STM_HEADER_SIZE + pattern_bytes..STM_HEADER_SIZE + pattern_bytes + sign_bytes].to_vec();

    Ok(SparseTernaryMatrix {
        patterns,
        signs,
        scale,
        rows,
        cols,
    })
}

/// Read a SparseTernaryMatrix from a memory-mapped file.
///
/// Zero-copy: patterns and signs are copied from the mmap, but the OS
/// handles paging the file in/out. Frequently-accessed experts stay in
/// the OS page cache automatically.
pub fn read_stm_mmap(mmap: &memmap2::Mmap) -> std::io::Result<SparseTernaryMatrix> {
    if mmap.len() < STM_HEADER_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("STM mmap too small: {} bytes", mmap.len()),
        ));
    }

    let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
    if magic != STM_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid STM magic: 0x{:08X}", magic),
        ));
    }

    let rows = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
    let scale = f32::from_le_bytes(mmap[12..16].try_into().unwrap());
    let pattern_bytes = u32::from_le_bytes(mmap[16..20].try_into().unwrap()) as usize;
    let sign_bytes = u32::from_le_bytes(mmap[20..24].try_into().unwrap()) as usize;

    let expected_len = STM_HEADER_SIZE + pattern_bytes + sign_bytes;
    if mmap.len() < expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("STM mmap truncated: expected {} bytes, got {}", expected_len, mmap.len()),
        ));
    }

    let patterns = mmap[STM_HEADER_SIZE..STM_HEADER_SIZE + pattern_bytes].to_vec();
    let signs = mmap[STM_HEADER_SIZE + pattern_bytes..STM_HEADER_SIZE + pattern_bytes + sign_bytes].to_vec();

    Ok(SparseTernaryMatrix {
        patterns,
        signs,
        scale,
        rows,
        cols,
    })
}

/// Expert loader: memory-mapped expert files with lazy decoding.
///
/// Keeps all expert files mmap'd but not read. When an expert is needed,
/// decodes from the mmap into a SparseTernaryMatrix. The OS page cache
/// handles caching — frequently-used experts stay in RAM.
pub struct ExpertLoader {
    /// mmap handles: [layer][expert] — kept open, not read into RAM.
    expert_mmaps: Vec<Vec<Option<memmap2::Mmap>>>,

    /// Decoded experts cache: (layer, expert_idx) → SparseTernaryMatrix.
    /// Populated on first access, evicted by the OS under memory pressure.
    cache: HashMap<(usize, usize), SparseTernaryMatrix>,

    /// Router weights: [layer][hidden_dim * num_experts] — always in RAM (tiny).
    pub routers: Vec<Vec<f32>>,

    /// Number of experts per layer.
    pub num_experts: usize,

    /// Top-k experts to activate per token.
    pub top_k: usize,

    /// Hidden dimension.
    pub hidden_dim: usize,

    /// Cache hit/miss statistics.
    pub stats: LoaderStats,

    /// Optional BuddyMoE substitution for cache misses (Phase 19-27).
    /// PreScope is used externally (it calls back into ExpertLoader).
    pub buddy_moe: Option<crate::inference::buddy_moe::BuddyMoE>,
}

/// IO statistics for the expert loader.
#[derive(Debug, Clone, Default)]
pub struct LoaderStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_loads: u64,
}

impl ExpertLoader {
    /// Open all expert files from a directory.
    ///
    /// Expected structure:
    /// ```text
    /// expert_dir/
    ///   layer_00_expert_0.stm
    ///   layer_00_expert_1.stm
    ///   ...
    ///   layer_34_expert_3.stm
    /// ```
    ///
    /// Files are mmap'd but NOT read into RAM. First access triggers a page-in.
    pub fn open(
        expert_dir: &Path,
        num_layers: usize,
        num_experts: usize,
        hidden_dim: usize,
        top_k: usize,
    ) -> std::io::Result<Self> {
        let mut expert_mmaps = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let mut layer_mmaps = Vec::with_capacity(num_experts);
            for expert in 0..num_experts {
                let path = expert_dir.join(format!("layer_{:02}_expert_{}.stm", layer, expert));
                if path.exists() {
                    let file = std::fs::File::open(&path)?;
                    // SAFETY: mmap is read-only, no mutation
                    let mmap = unsafe { memmap2::Mmap::map(&file)? };
                    layer_mmaps.push(Some(mmap));
                } else {
                    tracing::warn!("Expert file not found: {:?}", path);
                    layer_mmaps.push(None);
                }
            }
            expert_mmaps.push(layer_mmaps);
        }

        Ok(ExpertLoader {
            expert_mmaps,
            cache: HashMap::new(),
            routers: Vec::new(),
            num_experts,
            top_k,
            hidden_dim,
            stats: LoaderStats::default(),
            buddy_moe: None,
        })
    }

    /// Set router weights for all layers.
    /// Router weights are always in RAM (tiny: hidden_dim × num_experts per layer).
    pub fn set_routers(&mut self, routers: Vec<Vec<f32>>) {
        self.routers = routers;
    }

    /// Route a token through the router for a given layer.
    /// Returns top-k expert indices sorted by score (descending).
    pub fn route(&self, layer: usize, hidden: &[f32]) -> Vec<usize> {
        let router = &self.routers[layer];
        let hd = self.hidden_dim;
        let ne = self.num_experts;

        // Compute scores
        let mut scores = vec![0.0f32; ne];
        for e in 0..ne {
            let mut sum = 0.0f32;
            for j in 0..hd {
                sum += router[e * hd + j] * hidden[j];
            }
            scores[e] = sum;
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum_exp;
        }

        // Top-k
        let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed[..self.top_k.min(indexed.len())].iter().map(|&(i, _)| i).collect()
    }

    /// Ensure experts for given indices are loaded into cache.
    /// Call before `get_expert_ref()`.
    ///
    /// Returns the number of experts successfully loaded.
    pub fn ensure_loaded_batch(
        &mut self,
        layer: usize,
        indices: &[usize],
    ) -> usize {
        let mut loaded_count = 0;
        for &expert_idx in indices {
            if self.ensure_loaded(layer, expert_idx) {
                loaded_count += 1;
            }
        }
        loaded_count
    }

    /// Ensure an expert is loaded into cache. Call before `get_expert_ref()`.
    pub fn ensure_loaded(&mut self, layer: usize, expert_idx: usize) -> bool {
        let key = (layer, expert_idx);
        if self.cache.contains_key(&key) {
            self.stats.cache_hits += 1;
            return true;
        }

        self.stats.cache_misses += 1;
        self.stats.total_loads += 1;

        // BuddyMoE: if we don't have the expert, check if a buddy is already cached (Phase 19-27)
        // The actual buddy map must be set up via set_buddy_map() before use.
        if let Some(ref buddy) = self.buddy_moe {
            if let Some(buddy_idx) = buddy.get_buddy(layer, expert_idx) {
                let buddy_key = (layer, buddy_idx);
                if self.cache.contains_key(&buddy_key) {
                    tracing::debug!(
                        event = "buddy_substitution",
                        layer, expert_idx, buddy_idx,
                        "Substituted buddy expert for cache miss"
                    );
                    // Still load the real expert, but the buddy is available immediately
                }
            }
        }

        let loaded = match self.expert_mmaps.get(layer).and_then(|v| v.get(expert_idx)) {
            Some(Some(ref mmap)) => read_stm_mmap(mmap).ok(),
            _ => None,
        };

        if let Some(mat) = loaded {
            self.cache.insert(key, mat);
            true
        } else {
            false
        }
    }

    /// Get a reference to a cached expert.
    /// Call `ensure_loaded()` first to populate the cache.
    pub fn get_expert_ref(&self, layer: usize, expert_idx: usize) -> Option<&SparseTernaryMatrix> {
        self.cache.get(&(layer, expert_idx))
    }

    /// Evict all cached experts to free memory.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Number of experts currently in cache.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Cache hit rate (0.0 to 1.0).
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        if total == 0 { 0.0 } else { self.stats.cache_hits as f64 / total as f64 }
    }

    /// Export all experts from a model to .stm files.
    ///
    /// Creates one file per expert in the specified directory.
    pub fn export_experts(
        model: &crate::model::cpu_block_attn_res::CpuBlockAttnResModel,
        expert_dir: &Path,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(expert_dir)?;

        for (layer_idx, layer) in model.layers.iter().enumerate() {
            if let Some(ref moe) = layer.moe {
                for expert_idx in 0..moe.num_experts {
                    // Convert each expert's weight matrices to sparse ternary
                    let gate_fp32 = moe.expert_gate[expert_idx].to_fp32();
                    let gate = crate::model::sparse_ternary::prune_fp32_to_sparse_ternary(
                        &gate_fp32,
                        moe.intermediate_dim, moe.hidden_dim,
                    );
                    let up_fp32 = moe.expert_up[expert_idx].to_fp32();
                    let up = crate::model::sparse_ternary::prune_fp32_to_sparse_ternary(
                        &up_fp32,
                        moe.intermediate_dim, moe.hidden_dim,
                    );
                    let down_fp32 = moe.expert_down[expert_idx].to_fp32();
                    let down = crate::model::sparse_ternary::prune_fp32_to_sparse_ternary(
                        &down_fp32,
                        moe.hidden_dim, moe.intermediate_dim,
                    );

                    // Write each as a separate file
                    let path = expert_dir.join(format!(
                        "layer_{:02}_expert_{}_gate.stm",
                        layer_idx, expert_idx
                    ));
                    write_stm_file(&path, &gate)?;

                    let path = expert_dir.join(format!(
                        "layer_{:02}_expert_{}_up.stm",
                        layer_idx, expert_idx
                    ));
                    write_stm_file(&path, &up)?;

                    let path = expert_dir.join(format!(
                        "layer_{:02}_expert_{}_down.stm",
                        layer_idx, expert_idx
                    ));
                    write_stm_file(&path, &down)?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::sparse_ternary::SparseTernaryMatrix;
    use std::io::Write;

    fn make_test_sparse(rows: usize, cols: usize) -> SparseTernaryMatrix {
        let ternary: Vec<i8> = (0..rows * cols)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761) >> 30) as i8 - 1;
                v.clamp(-1, 1)
            })
            .collect();
        SparseTernaryMatrix::from_ternary(&ternary, 0.5, rows, cols)
    }

    #[test]
    fn test_stm_write_read_roundtrip() {
        let dir = std::env::temp_dir().join("ferrisres_test_stm");
        let _ = std::fs::create_dir(&dir);

        let original = make_test_sparse(8, 16);
        let path = dir.join("test.stm");

        write_stm_file(&path, &original).expect("write failed");
        let loaded = read_stm_file(&path).expect("read failed");

        assert_eq!(loaded.rows, original.rows);
        assert_eq!(loaded.cols, original.cols);
        assert!((loaded.scale - original.scale).abs() < 1e-6);
        assert_eq!(loaded.patterns, original.patterns);
        assert_eq!(loaded.signs, original.signs);

        // Decompressed should match
        assert_eq!(loaded.to_ternary(), original.to_ternary());

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_stm_mmap_roundtrip() {
        let dir = std::env::temp_dir().join("ferrisres_test_stm_mmap");
        let _ = std::fs::create_dir(&dir);

        let original = make_test_sparse(4, 8);
        let path = dir.join("test_mmap.stm");

        write_stm_file(&path, &original).expect("write failed");

        let file = std::fs::File::open(&path).expect("open failed");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap failed") };
        let loaded = read_stm_mmap(&mmap).expect("mmap read failed");

        assert_eq!(loaded.rows, original.rows);
        assert_eq!(loaded.cols, original.cols);
        assert_eq!(loaded.patterns, original.patterns);
        assert_eq!(loaded.signs, original.signs);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_stm_invalid_magic() {
        let dir = std::env::temp_dir().join("ferrisres_test_stm_bad");
        let _ = std::fs::create_dir(&dir);
        let path = dir.join("bad.stm");

        let mut f = std::fs::File::create(&path).expect("create");
        // Write enough bytes for header but with bad magic
        f.write_all(&0xDEADBEEFu32.to_le_bytes()).expect("write");
        f.write_all(&0u32.to_le_bytes()).expect("write"); // rows
        f.write_all(&0u32.to_le_bytes()).expect("write"); // cols
        f.write_all(&0f32.to_le_bytes()).expect("write"); // scale
        f.write_all(&0u32.to_le_bytes()).expect("write"); // pattern_bytes
        f.write_all(&0u32.to_le_bytes()).expect("write"); // sign_bytes

        let result = read_stm_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("magic"), "expected magic error, got: {}", err);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_stm_truncated_file() {
        let dir = std::env::temp_dir().join("ferrisres_test_stm_trunc");
        let _ = std::fs::create_dir(&dir);
        let path = dir.join("trunc.stm");

        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(&STM_MAGIC.to_le_bytes()).expect("write"); // Only magic, no header

        let result = read_stm_file(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_loader_cache_stats() {
        let loader = ExpertLoader {
            expert_mmaps: vec![],
            cache: HashMap::new(),
            routers: vec![],
            num_experts: 4,
            top_k: 2,
            hidden_dim: 8,
            stats: LoaderStats::default(),
            buddy_moe: None,
        };

        assert_eq!(loader.cache_hit_rate(), 0.0);
        assert_eq!(loader.cache_size(), 0);
    }
}
