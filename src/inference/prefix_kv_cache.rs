//! Prefix KV cache: reuse the attention KV of an unchanged token prefix.
//!
//! This is lever 1 of `docs/prefill-performance.md`. A turn usually extends a
//! stable prefix (soul, tools, skill text, earlier conversation) with a short
//! suffix. Re-running `CpuBlockAttnResModel::forward_prefill` over the whole
//! prompt throws that work away; this module keeps a serialisable
//! [`ModelKVCache`] snapshot for a prefix and lets the next turn start from it.
//!
//! # What is cached
//!
//! The key is a SHA-256 over:
//!
//! * the exact token ids, in order;
//! * a model fingerprint supplied by the caller (weights + architecture hash);
//! * a tokenizer/config version supplied by the caller;
//! * the optional principal/project scope.
//!
//! No key, no hit. A different model, tokenizer or scope is a miss even when
//! the text is identical. Nothing is guessed from a text prefix: the token ids
//! are compared exactly.
//!
//! The value is a [`KvSnapshot`]: every layer's K/V arrays, the causal
//! inter-block running sums, the PLE prefix and the processed token ids, plus
//! the number of tokens the snapshot covers. It restores into a
//! [`ModelKVCache`] without touching the model maths.
//!
//! # Suffix prefill
//!
//! [`prefill_with_prefix`] looks up the longest cached prefix and then feeds
//! only the remaining tokens through the model's existing
//! [`CpuBlockAttnResModel::forward_decode`] path, which attends over the
//! restored KV and appends the new K/V to it. That is the same computation the
//! decode loop already performs, so no prefill kernel is changed. The result is
//! inserted as the next prefix.
//!
//! An exact-length match cannot answer the "what is the next token" question
//! without a forward pass, so the helper treats it as a miss and prefills the
//! prompt; the snapshot itself is still returned by [`PrefixKvCache::longest_prefix`].
//!
//! # Budget and eviction
//!
//! The in-memory store is LRU-bounded by [`PrefixKvConfig::byte_budget`]. Bytes
//! are the actual K/V/ids/logit buffer lengths, not capacities. Every eviction
//! is counted in [`PrefixKvStats::evictions`].
//!
//! # Disk persistence
//!
//! With [`PrefixKvConfig::disk_dir`] set, every insert writes one file per
//! entry: magic, the key digest, the payload length, the payload and a SHA-256
//! checksum over the payload. The write is `write temp -> rename`, so a reader
//! never observes a partial file. A corrupt, truncated or mismatched file is a
//! miss, never a wrong answer. Disk files are not removed by memory eviction;
//! they form a second tier that is read back on a miss.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::inference::student_kv_cache::{LayerKVCache, ModelKVCache};
use crate::model::cpu_block_attn_res::CpuBlockAttnResModel;

/// On-disk file magic. Changes only with an incompatible format.
const DISK_MAGIC: &[u8; 8] = b"FRKVKVC1";
/// Layout version for the snapshot payload.
const SNAPSHOT_VERSION: u32 = 1;
/// Domain separation for the key hash.
const KEY_DOMAIN: &[u8] = b"ferrisres/prefix-kv/v1";

/// Configuration for a [`PrefixKvCache`].
#[derive(Debug, Clone)]
pub struct PrefixKvConfig {
    /// In-memory byte budget. The store evicts LRU entries until it fits.
    pub byte_budget: usize,
    /// Identity of the weights and architecture. Two models with the same
    /// token ids but a different fingerprint must not share an entry.
    pub model_fingerprint: String,
    /// Tokenizer and tokenizer-config version. Bumped when tokenisation changes.
    pub tokenizer_version: String,
    /// Optional directory for persistent snapshots.
    pub disk_dir: Option<PathBuf>,
}

impl Default for PrefixKvConfig {
    fn default() -> Self {
        Self {
            byte_budget: 64 * 1024 * 1024,
            model_fingerprint: String::new(),
            tokenizer_version: String::new(),
            disk_dir: None,
        }
    }
}

/// Counters describing what the cache did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefixKvStats {
    /// Lookups that found an in-memory entry.
    pub hits: u64,
    /// Lookups that found a valid on-disk entry and loaded it.
    pub disk_hits: u64,
    /// Lookups with no usable entry.
    pub misses: u64,
    /// Entries dropped to stay inside the byte budget.
    pub evictions: u64,
    /// Disk files that failed magic, checksum, length or key validation.
    pub disk_corrupt: u64,
    /// Disk writes that failed (permissions, IO).
    pub persist_errors: u64,
    /// Bytes currently held in memory.
    pub bytes: usize,
    /// Entries currently held in memory.
    pub entries: usize,
}

/// A cache hit: the snapshot restored to a [`ModelKVCache`] plus how many
/// prompt tokens it covers.
pub struct CacheHit {
    /// Number of leading prompt tokens whose K/V the cache already holds.
    pub covered_len: usize,
    /// Restored cache. Call `forward_decode` to extend it.
    pub cache: ModelKVCache,
}

/// Result of [`prefill_with_prefix`].
pub struct PrefixPrefill {
    /// Logits for the last position of the prompt (next-token logits).
    pub logits: Vec<f32>,
    /// Number of prompt tokens served from the cache.
    pub covered_len: usize,
    /// Number of prompt tokens actually run through the model.
    pub prefilled_tokens: usize,
}

/// One layer of a serialisable snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSnapshot {
    pub hidden_dim: usize,
    pub kv_dim: usize,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Serialisable image of a [`ModelKVCache`] for one token prefix.
#[derive(Debug, Clone, PartialEq)]
pub struct KvSnapshot {
    pub covered_len: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub ple_dim: usize,
    pub layers_per_block: usize,
    pub block_token_count: usize,
    pub layers: Vec<LayerSnapshot>,
    pub block_reps: Vec<Vec<f32>>,
    pub ple_prefix: Option<Vec<f32>>,
    pub cached_token_ids: Vec<u32>,
    pub shared_kv_sources: Vec<(usize, usize)>,
}

impl KvSnapshot {
    /// Copy a populated cache into a snapshot.
    pub fn from_cache(cache: &ModelKVCache) -> Self {
        let mut shared_kv_sources: Vec<(usize, usize)> =
            cache.shared_kv_sources.iter().map(|(&k, &v)| (k, v)).collect();
        shared_kv_sources.sort_unstable();
        Self {
            covered_len: cache.seq_len(),
            hidden_dim: cache.hidden_dim,
            num_layers: cache.num_layers,
            ple_dim: cache.ple_dim,
            layers_per_block: cache.layers_per_block,
            block_token_count: cache.block_token_count,
            layers: cache
                .layers
                .iter()
                .map(|l| LayerSnapshot {
                    hidden_dim: l.hidden_dim,
                    kv_dim: l.kv_dim,
                    k: l.k.clone(),
                    v: l.v.clone(),
                })
                .collect(),
            block_reps: cache.block_reps.clone(),
            ple_prefix: cache.ple_prefix.clone(),
            cached_token_ids: cache.cached_token_ids.clone(),
            shared_kv_sources,
        }
    }

    /// Build a fresh [`ModelKVCache`] from the snapshot.
    pub fn restore(&self) -> ModelKVCache {
        let kv_dims: Vec<usize> = self.layers.iter().map(|l| l.kv_dim).collect();
        let mut cache = ModelKVCache::new(
            self.num_layers,
            self.hidden_dim,
            &kv_dims,
            self.layers_per_block,
            self.ple_dim,
        );
        cache.layers = self
            .layers
            .iter()
            .map(|l| LayerKVCache {
                k: l.k.clone(),
                v: l.v.clone(),
                len: self.covered_len,
                hidden_dim: l.hidden_dim,
                kv_dim: l.kv_dim,
            })
            .collect();
        cache.block_reps = self.block_reps.clone();
        cache.block_token_count = self.block_token_count;
        cache.shared_kv_sources = self.shared_kv_sources.iter().copied().collect();
        cache.ple_prefix = self.ple_prefix.clone();
        cache.cached_token_ids = self.cached_token_ids.clone();
        cache
    }

    /// Bytes held by the snapshot's buffers.
    pub fn memory_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        for l in &self.layers {
            bytes += std::mem::size_of::<LayerSnapshot>() + (l.k.len() + l.v.len()) * 4;
        }
        for r in &self.block_reps {
            bytes += std::mem::size_of::<Vec<f32>>() + r.len() * 4;
        }
        if let Some(p) = &self.ple_prefix {
            bytes += std::mem::size_of::<Vec<f32>>() + p.len() * 4;
        }
        bytes += self.cached_token_ids.len() * 4;
        bytes += self.shared_kv_sources.len() * std::mem::size_of::<(usize, usize)>();
        bytes
    }

    fn encode(&self, w: &mut Writer) {
        w.u32(SNAPSHOT_VERSION);
        w.u64(self.covered_len as u64);
        w.u64(self.hidden_dim as u64);
        w.u64(self.num_layers as u64);
        w.u64(self.ple_dim as u64);
        w.u64(self.layers_per_block as u64);
        w.u64(self.block_token_count as u64);

        w.u64(self.layers.len() as u64);
        for l in &self.layers {
            w.u64(l.hidden_dim as u64);
            w.u64(l.kv_dim as u64);
            w.u64(l.k.len() as u64);
            w.f32s(&l.k);
            w.u64(l.v.len() as u64);
            w.f32s(&l.v);
        }

        w.u64(self.block_reps.len() as u64);
        for r in &self.block_reps {
            w.u64(r.len() as u64);
            w.f32s(r);
        }

        match &self.ple_prefix {
            Some(p) => {
                w.u8(1);
                w.u64(p.len() as u64);
                w.f32s(p);
            }
            None => w.u8(0),
        }

        w.u64(self.cached_token_ids.len() as u64);
        for &id in &self.cached_token_ids {
            w.u32(id);
        }

        w.u64(self.shared_kv_sources.len() as u64);
        for &(k, v) in &self.shared_kv_sources {
            w.u64(k as u64);
            w.u64(v as u64);
        }
    }

    fn decode(r: &mut Reader<'_>) -> Option<Self> {
        if r.u32()? != SNAPSHOT_VERSION {
            return None;
        }
        let covered_len = r.u64()? as usize;
        let hidden_dim = r.u64()? as usize;
        let num_layers = r.u64()? as usize;
        let ple_dim = r.u64()? as usize;
        let layers_per_block = r.u64()? as usize;
        let block_token_count = r.u64()? as usize;

        let layer_count = r.u64()? as usize;
        if layer_count > 4096 {
            return None;
        }
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let l_hidden = r.u64()? as usize;
            let kv_dim = r.u64()? as usize;
            let k_len = r.u64()? as usize;
            let k = r.f32s(k_len)?;
            let v_len = r.u64()? as usize;
            let v = r.f32s(v_len)?;
            layers.push(LayerSnapshot { hidden_dim: l_hidden, kv_dim, k, v });
        }

        let rep_count = r.u64()? as usize;
        if rep_count > 4096 {
            return None;
        }
        let mut block_reps = Vec::with_capacity(rep_count);
        for _ in 0..rep_count {
            let len = r.u64()? as usize;
            block_reps.push(r.f32s(len)?);
        }

        let ple_prefix = match r.u8()? {
            0 => None,
            1 => {
                let len = r.u64()? as usize;
                Some(r.f32s(len)?)
            }
            _ => return None,
        };

        let id_count = r.u64()? as usize;
        if id_count > (1 << 24) {
            return None;
        }
        let mut cached_token_ids = Vec::with_capacity(id_count);
        for _ in 0..id_count {
            cached_token_ids.push(r.u32()?);
        }

        let shared_count = r.u64()? as usize;
        if shared_count > 4096 {
            return None;
        }
        let mut shared_kv_sources = Vec::with_capacity(shared_count);
        for _ in 0..shared_count {
            shared_kv_sources.push((r.u64()? as usize, r.u64()? as usize));
        }

        if cached_token_ids.len() != covered_len || layers.len() != num_layers {
            return None;
        }

        Some(Self {
            covered_len,
            hidden_dim,
            num_layers,
            ple_dim,
            layers_per_block,
            block_token_count,
            layers,
            block_reps,
            ple_prefix,
            cached_token_ids,
            shared_kv_sources,
        })
    }
}

struct Entry {
    prefix_ids: Vec<u32>,
    scope: Option<String>,
    snapshot: KvSnapshot,
    bytes: usize,
    last_used: u64,
    digest: [u8; 32],
}

/// Bounded, optionally persistent prefix KV store.
pub struct PrefixKvCache {
    config: PrefixKvConfig,
    entries: HashMap<[u8; 32], Entry>,
    clock: u64,
    stats: PrefixKvStats,
    disk_counter: u64,
}

impl PrefixKvCache {
    /// Create a cache. Creates the disk directory when one is configured.
    pub fn new(config: PrefixKvConfig) -> std::io::Result<Self> {
        if let Some(dir) = &config.disk_dir {
            fs::create_dir_all(dir)?;
        }
        Ok(Self {
            config,
            entries: HashMap::new(),
            clock: 0,
            stats: PrefixKvStats::default(),
            disk_counter: 0,
        })
    }

    /// Current counters.
    pub fn stats(&self) -> &PrefixKvStats {
        &self.stats
    }

    /// The configuration this cache was built with.
    pub fn config(&self) -> &PrefixKvConfig {
        &self.config
    }

    /// Number of entries currently held in memory.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no entry is held in memory.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every in-memory entry. Disk files are kept.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.bytes = 0;
        self.stats.entries = 0;
    }

    /// Find the longest cached prefix of `ids` for `scope`.
    ///
    /// Unlike [`Self::lookup`] this returns an exact-length match too.
    pub fn longest_prefix(&mut self, ids: &[u32], scope: Option<&str>) -> Option<CacheHit> {
        if ids.is_empty() {
            self.stats.misses += 1;
            return None;
        }
        let digests = self.prefix_digests(ids, scope);
        for k in (1..=ids.len()).rev() {
            let digest = digests[k];
            if let Some(entry) = self.entries.get_mut(&digest) {
                if entry.scope.as_deref() != scope || entry.prefix_ids.as_slice() != &ids[..k] {
                    continue;
                }
                self.clock += 1;
                entry.last_used = self.clock;
                self.stats.hits += 1;
                return Some(CacheHit { covered_len: k, cache: entry.snapshot.restore() });
            }
            if let Some(entry) = self.load_from_disk(&digest, &ids[..k], scope) {
                self.stats.disk_hits += 1;
                let hit = CacheHit { covered_len: k, cache: entry.snapshot.restore() };
                self.admit(digest, entry);
                return Some(hit);
            }
        }
        self.stats.misses += 1;
        None
    }

    /// Find the longest cached prefix strictly shorter than `ids`, leaving the
    /// suffix for the caller to run. This is the lookup [`prefill_with_prefix`]
    /// uses: an exact match cannot supply next-token logits by itself.
    pub fn lookup(&mut self, ids: &[u32], scope: Option<&str>) -> Option<CacheHit> {
        if ids.len() < 2 {
            self.stats.misses += 1;
            return None;
        }
        let digests = self.prefix_digests(ids, scope);
        for k in (1..ids.len()).rev() {
            let digest = digests[k];
            if let Some(entry) = self.entries.get_mut(&digest) {
                if entry.scope.as_deref() != scope || entry.prefix_ids.as_slice() != &ids[..k] {
                    continue;
                }
                self.clock += 1;
                entry.last_used = self.clock;
                self.stats.hits += 1;
                return Some(CacheHit { covered_len: k, cache: entry.snapshot.restore() });
            }
            if let Some(entry) = self.load_from_disk(&digest, &ids[..k], scope) {
                self.stats.disk_hits += 1;
                let hit = CacheHit { covered_len: k, cache: entry.snapshot.restore() };
                self.admit(digest, entry);
                return Some(hit);
            }
        }
        self.stats.misses += 1;
        None
    }

    /// Insert the cache produced by a prefill of exactly `ids`.
    ///
    /// `cache.cached_token_ids` must equal `ids`; otherwise the snapshot would
    /// claim a prefix it does not hold, so it is rejected.
    pub fn insert(
        &mut self,
        ids: &[u32],
        cache: &ModelKVCache,
        scope: Option<&str>,
    ) -> Result<(), PrefixKvError> {
        if ids.is_empty() {
            return Ok(());
        }
        if cache.seq_len() != ids.len() || cache.cached_token_ids != ids {
            return Err(PrefixKvError::PrefixMismatch { ids: ids.len(), cached: cache.seq_len() });
        }
        let snapshot = KvSnapshot::from_cache(cache);
        if snapshot.covered_len != ids.len() {
            return Err(PrefixKvError::PrefixMismatch {
                ids: ids.len(),
                cached: snapshot.covered_len,
            });
        }
        let digest = self.digest_for(ids, scope);
        let bytes = snapshot.memory_bytes();
        let mut entry = Entry {
            prefix_ids: ids.to_vec(),
            scope: scope.map(str::to_owned),
            snapshot,
            bytes,
            last_used: 0,
            digest,
        };

        if let Some(dir) = self.config.disk_dir.clone() {
            if self.write_disk(&dir, &entry).is_err() {
                self.stats.persist_errors += 1;
            }
        }

        self.clock += 1;
        entry.last_used = self.clock;
        self.admit(digest, entry);
        Ok(())
    }

    /// Insert a snapshot directly. Used by tests and by callers that restore a
    /// cache from elsewhere; the same identity checks apply.
    pub fn insert_snapshot(
        &mut self,
        ids: &[u32],
        snapshot: KvSnapshot,
        scope: Option<&str>,
    ) -> Result<(), PrefixKvError> {
        if ids.is_empty() {
            return Ok(());
        }
        if snapshot.covered_len != ids.len() || snapshot.cached_token_ids != ids {
            return Err(PrefixKvError::PrefixMismatch {
                ids: ids.len(),
                cached: snapshot.covered_len,
            });
        }
        let digest = self.digest_for(ids, scope);
        let bytes = snapshot.memory_bytes();
        let mut entry = Entry {
            prefix_ids: ids.to_vec(),
            scope: scope.map(str::to_owned),
            snapshot,
            bytes,
            last_used: 0,
            digest,
        };
        if let Some(dir) = self.config.disk_dir.clone() {
            if self.write_disk(&dir, &entry).is_err() {
                self.stats.persist_errors += 1;
            }
        }
        self.clock += 1;
        entry.last_used = self.clock;
        self.admit(digest, entry);
        Ok(())
    }

    /// Store a prefilled cache and return the digest that was used.
    pub fn digest_for(&self, ids: &[u32], scope: Option<&str>) -> [u8; 32] {
        let mut hasher = self.base_hasher(scope);
        for &id in ids {
            hasher.update(id.to_le_bytes());
        }
        hasher.update((ids.len() as u64).to_le_bytes());
        hasher.finalize().into()
    }

    fn base_hasher(&self, scope: Option<&str>) -> Sha256 {
        let mut hasher = Sha256::new();
        hasher.update(KEY_DOMAIN);
        hash_str(&mut hasher, &self.config.model_fingerprint);
        hash_str(&mut hasher, &self.config.tokenizer_version);
        match scope {
            None => hasher.update([0u8]),
            Some(s) => {
                hasher.update([1u8]);
                hash_str(&mut hasher, s);
            }
        }
        hasher
    }

    /// Digest of every non-empty prefix, indexed by prefix length.
    ///
    /// The running hasher is cloned per length, so the whole ladder costs
    /// O(tokens) token absorptions plus one finalise each.
    fn prefix_digests(&self, ids: &[u32], scope: Option<&str>) -> Vec<[u8; 32]> {
        let mut out = vec![[0u8; 32]; ids.len() + 1];
        let mut running = self.base_hasher(scope);
        for (i, &id) in ids.iter().enumerate() {
            running.update(id.to_le_bytes());
            let mut h = running.clone();
            h.update(((i + 1) as u64).to_le_bytes());
            out[i + 1] = h.finalize().into();
        }
        out
    }

    fn admit(&mut self, digest: [u8; 32], entry: Entry) {
        if let Some(old) = self.entries.insert(digest, entry) {
            self.stats.bytes = self.stats.bytes.saturating_sub(old.bytes);
        }
        let added = self.entries.get(&digest).map_or(0, |e| e.bytes);
        self.stats.bytes += added;
        self.stats.entries = self.entries.len();
        self.evict_to_budget();
    }

    fn evict_to_budget(&mut self) {
        while self.stats.bytes > self.config.byte_budget && !self.entries.is_empty() {
            let victim = self.entries.iter().min_by_key(|(_, e)| e.last_used).map(|(d, _)| *d);
            let Some(victim) = victim else { break };
            if let Some(entry) = self.entries.remove(&victim) {
                self.stats.bytes = self.stats.bytes.saturating_sub(entry.bytes);
                self.stats.evictions += 1;
            }
        }
        self.stats.entries = self.entries.len();
    }

    fn file_name(digest: &[u8; 32]) -> String {
        format!("{}.frkv", hex(digest))
    }

    fn write_disk(&mut self, dir: &Path, entry: &Entry) -> std::io::Result<()> {
        let name = Self::file_name(&entry.digest);
        let final_path = dir.join(&name);
        self.disk_counter += 1;
        let tmp_path =
            dir.join(format!(".{}.tmp-{}-{}", name, std::process::id(), self.disk_counter));

        let mut writer = Writer::default();
        writer.u64(entry.prefix_ids.len() as u64);
        for &id in &entry.prefix_ids {
            writer.u32(id);
        }
        match &entry.scope {
            Some(s) => {
                writer.u8(1);
                writer.bytes(s.as_bytes());
            }
            None => writer.u8(0),
        }
        entry.snapshot.encode(&mut writer);
        let payload = writer.buf;

        let checksum: [u8; 32] = Sha256::digest(&payload).into();
        let mut file = Vec::with_capacity(8 + 32 + 8 + payload.len() + 32);
        file.extend_from_slice(DISK_MAGIC);
        file.extend_from_slice(&entry.digest);
        file.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        file.extend_from_slice(&payload);
        file.extend_from_slice(&checksum);

        fs::write(&tmp_path, &file)?;
        if let Err(err) = fs::rename(&tmp_path, &final_path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(err);
        }
        Ok(())
    }

    fn load_from_disk(
        &mut self,
        digest: &[u8; 32],
        ids: &[u32],
        scope: Option<&str>,
    ) -> Option<Entry> {
        let dir = self.config.disk_dir.as_ref()?;
        let path = dir.join(Self::file_name(digest));
        let raw = match fs::read(&path) {
            Ok(raw) => raw,
            Err(_) => return None,
        };
        let entry = match Self::decode_disk(&raw, digest, ids, scope) {
            Some(entry) => entry,
            None => {
                self.stats.disk_corrupt += 1;
                return None;
            }
        };
        Some(entry)
    }

    fn decode_disk(
        raw: &[u8],
        digest: &[u8; 32],
        ids: &[u32],
        scope: Option<&str>,
    ) -> Option<Entry> {
        if raw.len() < 8 + 32 + 8 + 32 || &raw[..8] != DISK_MAGIC {
            return None;
        }
        let stored_digest: [u8; 32] = raw[8..40].try_into().ok()?;
        if &stored_digest != digest {
            return None;
        }
        let payload_len = u64::from_le_bytes(raw[40..48].try_into().ok()?) as usize;
        let payload_end = 48usize.checked_add(payload_len)?;
        let checksum_start = payload_end;
        let checksum_end = checksum_start.checked_add(32)?;
        if checksum_end != raw.len() {
            return None;
        }
        let payload = &raw[48..payload_end];
        let checksum: [u8; 32] = raw[checksum_start..checksum_end].try_into().ok()?;
        let computed: [u8; 32] = Sha256::digest(payload).into();
        if computed != checksum {
            return None;
        }

        let mut r = Reader { buf: payload, pos: 0 };
        let id_count = r.u64()? as usize;
        if id_count > (1 << 24) {
            return None;
        }
        let mut prefix_ids = Vec::with_capacity(id_count);
        for _ in 0..id_count {
            prefix_ids.push(r.u32()?);
        }
        let stored_scope = match r.u8()? {
            0 => None,
            1 => {
                let bytes = r.bytes()?;
                Some(String::from_utf8(bytes.to_vec()).ok()?)
            }
            _ => return None,
        };
        if prefix_ids != ids || stored_scope.as_deref() != scope {
            return None;
        }
        let snapshot = KvSnapshot::decode(&mut r)?;
        if r.pos != payload.len() {
            return None;
        }
        if snapshot.cached_token_ids != prefix_ids || snapshot.covered_len != ids.len() {
            return None;
        }
        let bytes = snapshot.memory_bytes();
        Some(Entry {
            prefix_ids,
            scope: stored_scope,
            snapshot,
            bytes,
            last_used: 0,
            digest: *digest,
        })
    }
}

/// Errors from an insert that would not describe the prefix it claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixKvError {
    /// The cache holds a different number of tokens than `ids`.
    PrefixMismatch { ids: usize, cached: usize },
}

impl std::fmt::Display for PrefixKvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrefixMismatch { ids, cached } => {
                write!(f, "cache covers {cached} tokens but the prefix has {ids}")
            }
        }
    }
}

impl std::error::Error for PrefixKvError {}

/// Prefill only the part of `token_ids` that the cache does not already cover.
///
/// The cached prefix is restored, then each remaining token goes through
/// [`CpuBlockAttnResModel::forward_decode`], which attends over the restored
/// K/V and appends to it. The full prompt is stored for the next turn.
/// The last returned logits are the next-token logits.
pub fn prefill_with_prefix(
    model: &CpuBlockAttnResModel,
    token_ids: &[u32],
    store: &mut PrefixKvCache,
    scope: Option<&str>,
) -> PrefixPrefill {
    if token_ids.is_empty() {
        return PrefixPrefill { logits: Vec::new(), covered_len: 0, prefilled_tokens: 0 };
    }

    let hit = store.lookup(token_ids, scope);
    let (mut cache, covered_len) = match hit {
        Some(hit) => (hit.cache, hit.covered_len),
        None => (empty_cache_for(model), 0),
    };

    if covered_len > 0 {
        extend_ple_for_suffix(model, &token_ids[covered_len..], &mut cache, covered_len);
    }

    let mut logits = Vec::new();
    for &tok in &token_ids[covered_len..] {
        logits = model.forward_decode(tok, &mut cache);
    }

    // Store the full prompt as the next prefix. A persist failure is counted
    // in the stats and must not fail the generation.
    let _ = store.insert(token_ids, &cache, scope);

    PrefixPrefill { logits, covered_len, prefilled_tokens: token_ids.len() - covered_len }
}

/// PLE is computed from the initial embedding of each token, so it is a
/// per-token function and can be extended for a suffix without seeing the
/// prefix. `forward_decode` looks PLE up by absolute position, so the suffix
/// rows must be appended before the decode loop or those tokens silently lose
/// their PLE input and the answer changes.
fn extend_ple_for_suffix(
    model: &CpuBlockAttnResModel,
    suffix: &[u32],
    cache: &mut ModelKVCache,
    covered_len: usize,
) {
    let ple_dim = model.hidden_size_per_layer_input;
    if ple_dim == 0 || suffix.is_empty() {
        return;
    }
    let stride = model.num_layers * ple_dim;
    let Some(prefix) = cache.ple_prefix.as_ref() else {
        return;
    };
    if prefix.len() != covered_len * stride {
        return;
    }

    let hd = model.hidden_dim;
    let mut hidden = vec![0.0f32; suffix.len() * hd];
    for (t, &tid) in suffix.iter().enumerate() {
        let id = tid as usize;
        if id * hd + hd <= model.embed_tokens.len() {
            hidden[t * hd..(t + 1) * hd]
                .copy_from_slice(&model.embed_tokens[id * hd..(id + 1) * hd]);
        }
    }
    let scale = (hd as f32).sqrt();
    for h in hidden.iter_mut() {
        *h *= scale;
    }

    if let Some(ple) = model.precompute_ple(&hidden, suffix, suffix.len(), hd, ple_dim) {
        if let Some(prefix) = cache.ple_prefix.as_mut() {
            prefix.extend_from_slice(&ple);
        }
    }
}

/// An empty [`ModelKVCache`] shaped for `model`.
pub fn empty_cache_for(model: &CpuBlockAttnResModel) -> ModelKVCache {
    let kv_dims: Vec<usize> = model.layers.iter().map(|l| l.num_kv_heads * l.head_dim).collect();
    ModelKVCache::new(
        model.num_layers,
        model.hidden_dim,
        &kv_dims,
        model.block_config.layers_per_block.max(1),
        model.hidden_size_per_layer_input,
    )
}

fn hash_str(hasher: &mut Sha256, s: &str) {
    hasher.update((s.len() as u64).to_le_bytes());
    hasher.update(s.as_bytes());
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0x0f) as usize] as char);
    }
    out
}

#[derive(Default)]
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.buf.extend_from_slice(v);
    }
    fn f32s(&mut self, v: &[f32]) {
        for &x in v {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        if end > self.buf.len() {
            return None;
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Some(slice)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }
    fn u32(&mut self) -> Option<u32> {
        self.take(4).map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
    }
    fn u64(&mut self) -> Option<u64> {
        self.take(8).map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
    }
    fn bytes(&mut self) -> Option<&'a [u8]> {
        let len = self.u64()? as usize;
        self.take(len)
    }
    fn f32s(&mut self, len: usize) -> Option<Vec<f32>> {
        let raw = self.take(len.checked_mul(4)?)?;
        let mut out = Vec::with_capacity(len);
        for chunk in raw.chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)")));
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cpu_block_attn_res::{BlockConfig, CpuBlockAttnResLayer};
    use crate::model::cpu_linear::{CpuLinear, CpuRmsNorm};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn build_model(hidden: usize, vocab: usize, layers: usize, seed: f32) -> CpuBlockAttnResModel {
        let hd = hidden;
        let num_heads = 2;
        let head_dim = hd / num_heads;
        let num_kv_heads = 1;
        let inter = hd * 2;

        fn weight(n: usize, seed: f32, mul: f32) -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = i as f32 * 0.017 * mul + seed;
                    (x.sin() + 0.1 * (x * 1.7).cos()) * 0.1
                })
                .collect()
        }

        let mk_layer = |idx: usize| CpuBlockAttnResLayer {
            layer_number: idx,
            hidden_dim: hd,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim: inter,
            attn_norm: CpuRmsNorm::from_weight(vec![1.0f32; hd], 1e-6),
            q_proj: CpuLinear::from_weight(
                weight(hd * num_heads * head_dim, seed + idx as f32 * 0.3, 1.0),
                hd,
                num_heads * head_dim,
            ),
            k_proj: CpuLinear::from_weight(
                weight(hd * num_kv_heads * head_dim, seed + 1.0, 1.1),
                hd,
                num_kv_heads * head_dim,
            ),
            v_proj: CpuLinear::from_weight(
                weight(hd * num_kv_heads * head_dim, seed + 2.0, 1.2),
                hd,
                num_kv_heads * head_dim,
            ),
            out_proj: CpuLinear::from_weight(
                weight(num_heads * head_dim * hd, seed + 3.0, 1.3),
                num_heads * head_dim,
                hd,
            ),
            q_norm: CpuRmsNorm::from_weight(vec![1.0f32; head_dim], 1e-6),
            k_norm: CpuRmsNorm::from_weight(vec![1.0f32; head_dim], 1e-6),
            v_norm: CpuRmsNorm::from_weight(vec![1.0f32; head_dim], 1e-6),
            post_attn_norm: CpuRmsNorm::from_weight(vec![1.0f32; hd], 1e-6),
            pre_ffn_norm: CpuRmsNorm::from_weight(vec![1.0f32; hd], 1e-6),
            ffn_gate: Some(CpuLinear::from_weight(weight(hd * inter, seed + 4.0, 1.0), hd, inter)),
            ffn_up: Some(CpuLinear::from_weight(weight(hd * inter, seed + 5.0, 1.0), hd, inter)),
            ffn_down: Some(CpuLinear::from_weight(weight(inter * hd, seed + 6.0, 1.0), inter, hd)),
            moe: None,
            post_ffn_norm: CpuRmsNorm::from_weight(vec![1.0f32; hd], 1e-6),
            layer_scalar: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
            use_gelu: true,
            kv_shared: false,
        };

        CpuBlockAttnResModel {
            layers: (0..layers).map(mk_layer).collect(),
            embed_tokens: weight(vocab * hd, seed + 7.0, 1.0),
            lm_head: weight(hd * vocab, seed + 8.0, 1.0),
            final_norm: vec![1.0f32; hd],
            hidden_dim: hd,
            vocab_size: vocab,
            num_layers: layers,
            final_logit_softcapping: None,
            ple_model_projection: None,
            ple_projection_norm: None,
            embed_tokens_per_layer: None,
            hidden_size_per_layer_input: 0,
            num_kv_shared_layers: 0,
            block_config: BlockConfig {
                num_blocks: 1,
                layers_per_block: layers.max(1),
                boundary_layers: if layers > 1 { vec![layers - 1] } else { Vec::new() },
                attn_res_proj: weight(hd * hd, seed + 9.0, 1.0),
                attn_res_norm: vec![1.0f32; hd],
            },
            lora_manager: None,
        }
    }

    fn tiny_model() -> CpuBlockAttnResModel {
        build_model(8, 16, 2, 0.3)
    }

    /// Tiny model with a PLE pathway enabled, so the suffix path has to extend
    /// the per-token PLE rows exactly like a full prefill would.
    fn ple_model() -> CpuBlockAttnResModel {
        fn w(n: usize, seed: f32) -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = i as f32 * 0.017 + seed;
                    (x.sin() + 0.1 * (x * 1.7).cos()) * 0.1
                })
                .collect()
        }

        let mut model = build_model(8, 16, 2, 0.6);
        let hd = model.hidden_dim;
        let ple_dim = 4;
        let ple_total = ple_dim * model.num_layers;
        for layer in &mut model.layers {
            layer.ple_input_gate = Some(CpuLinear::from_weight(w(hd * ple_dim, 3.0), hd, ple_dim));
            layer.ple_projection = Some(CpuLinear::from_weight(w(ple_dim * hd, 4.0), ple_dim, hd));
            layer.ple_post_norm = Some(CpuRmsNorm::from_weight(vec![1.0f32; hd], 1e-6));
        }
        model.ple_model_projection = Some(w(hd * ple_total, 5.0));
        model.ple_projection_norm = Some(vec![1.0f32; ple_dim]);
        model.embed_tokens_per_layer = Some(w(model.vocab_size * ple_total, 6.0));
        model.hidden_size_per_layer_input = ple_dim;
        model
    }

    fn prompt(n: usize, seed: u32) -> Vec<u32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                x % 16
            })
            .collect()
    }

    fn cache(budget: usize) -> PrefixKvCache {
        PrefixKvCache::new(PrefixKvConfig {
            byte_budget: budget,
            model_fingerprint: "test-model-weights-abc".into(),
            tokenizer_version: "tok-v1".into(),
            disk_dir: None,
        })
        .expect("in-memory cache")
    }

    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let n = TMP_COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "ferrisres-prefix-kv-{}-{}-{}",
                tag,
                std::process::id(),
                n
            ));
            fs::create_dir_all(&path).expect("temp dir");
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

    #[test]
    fn miss_then_hit_prefills_only_the_suffix() {
        let model = tiny_model();
        let mut store = cache(16 * 1024 * 1024);
        let mut ids = prompt(12, 1);
        ids.push(3);

        let first = prefill_with_prefix(&model, &ids, &mut store, Some("alice"));
        assert_eq!(first.covered_len, 0);
        assert_eq!(first.prefilled_tokens, ids.len());

        ids.push(7);
        let second = prefill_with_prefix(&model, &ids, &mut store, Some("alice"));
        assert_eq!(second.covered_len, ids.len() - 1);
        assert_eq!(second.prefilled_tokens, 1);
        assert!(store.stats().hits >= 1);
    }

    #[test]
    fn longest_prefix_shares_the_common_part() {
        let model = tiny_model();
        let mut store = cache(16 * 1024 * 1024);
        let a = prompt(10, 2);
        let mut ab = a.clone();
        ab.extend(prompt(5, 3));
        let mut ac = a.clone();
        ac.extend(prompt(5, 4));

        let prefill = |ids: &[u32], store: &mut PrefixKvCache| {
            let mut c = empty_cache_for(&model);
            model.forward_prefill(ids, &mut c);
            store.insert(ids, &c, Some("alice")).unwrap();
        };
        prefill(&a, &mut store);
        prefill(&ab, &mut store);

        let hit = store.lookup(&ac, Some("alice")).expect("A must be a cached prefix of A+C");
        assert_eq!(hit.covered_len, a.len());

        let hit_b = store.lookup(&ab, Some("alice")).expect("A+B lookup");
        // Exact match is returned by longest_prefix, not by lookup.
        assert_eq!(hit_b.covered_len, a.len());
        assert_eq!(store.longest_prefix(&ab, Some("alice")).unwrap().covered_len, ab.len());
    }

    #[test]
    fn wrong_model_or_tokenizer_is_a_miss() {
        let model = tiny_model();
        let dir = TempDir::new("identity");
        let ids = prompt(8, 5);
        let mut c = empty_cache_for(&model);
        model.forward_prefill(&ids, &mut c);

        // Probe with one extra token so an exact match cannot mask the key.
        let mut probe = ids.clone();
        probe.push(2);

        let mut base = PrefixKvCache::new(cache_config_with_dir(dir.path())).unwrap();
        base.insert(&ids, &c, None).unwrap();
        assert!(base.lookup(&probe, None).is_some());

        // Same disk directory, different model fingerprint: no shared entry.
        let mut other_model = PrefixKvCache::new(PrefixKvConfig {
            byte_budget: 16 * 1024 * 1024,
            model_fingerprint: "test-model-weights-xyz".into(),
            tokenizer_version: "tok-v1".into(),
            disk_dir: Some(dir.path().to_path_buf()),
        })
        .unwrap();
        assert!(other_model.lookup(&probe, None).is_none());
        assert!(other_model.longest_prefix(&ids, None).is_none());

        // Different tokenizer version: no shared entry either.
        let mut other_tok = PrefixKvCache::new(PrefixKvConfig {
            byte_budget: 16 * 1024 * 1024,
            model_fingerprint: "test-model-weights-abc".into(),
            tokenizer_version: "tok-v2".into(),
            disk_dir: Some(dir.path().to_path_buf()),
        })
        .unwrap();
        assert!(other_tok.lookup(&probe, None).is_none());
        assert!(other_tok.longest_prefix(&ids, None).is_none());

        // Same model and tokenizer, but different token ids: a miss too.
        let mut wrong_ids = ids.clone();
        wrong_ids[3] = (wrong_ids[3] + 1) % 16;
        assert!(base.lookup(&wrong_ids, None).is_none());
    }

    #[test]
    fn insert_rejects_a_cache_for_a_different_prefix() {
        let model = tiny_model();
        let ids = prompt(6, 6);
        let mut c = empty_cache_for(&model);
        model.forward_prefill(&ids[..4], &mut c);
        let err = cache(1024).insert(&ids, &c, None).unwrap_err();
        assert_eq!(err, PrefixKvError::PrefixMismatch { ids: 6, cached: 4 });
    }

    #[test]
    fn eviction_respects_the_byte_budget() {
        let model = tiny_model();
        let mut store = cache(0);
        let ids = prompt(8, 7);
        let mut c = empty_cache_for(&model);
        model.forward_prefill(&ids, &mut c);
        store.insert(&ids, &c, None).unwrap();

        assert_eq!(store.len(), 0);
        assert_eq!(store.stats().evictions, 1);
        assert_eq!(store.stats().bytes, 0);

        // A budget that fits exactly two entries but not three.
        let mut first = cache(16 * 1024 * 1024);
        let mut one = empty_cache_for(&model);
        model.forward_prefill(&ids, &mut one);
        first.insert(&ids, &one, None).unwrap();
        let entry_bytes = first.stats().bytes;
        assert!(entry_bytes > 0);

        let mut store = cache(entry_bytes * 2 + entry_bytes / 2);
        for seed in 0..4u32 {
            let ids = prompt(8, 100 + seed);
            let mut c = empty_cache_for(&model);
            model.forward_prefill(&ids, &mut c);
            store.insert(&ids, &c, None).unwrap();
            assert!(store.stats().bytes <= store.config().byte_budget);
        }
        assert_eq!(store.len(), 2);
        assert!(store.stats().evictions >= 2);
    }

    #[test]
    fn corrupt_disk_entry_is_a_miss() {
        let model = tiny_model();
        let dir = TempDir::new("corrupt");
        let ids = prompt(9, 8);

        let config = PrefixKvConfig {
            byte_budget: 16 * 1024 * 1024,
            model_fingerprint: "test-model-weights-abc".into(),
            tokenizer_version: "tok-v1".into(),
            disk_dir: Some(dir.path().to_path_buf()),
        };
        let mut store = PrefixKvCache::new(config.clone()).unwrap();
        let mut c = empty_cache_for(&model);
        model.forward_prefill(&ids, &mut c);
        store.insert(&ids, &c, None).unwrap();

        let digest = store.digest_for(&ids, None);
        let file = dir.path().join(PrefixKvCache::file_name(&digest));
        assert!(file.exists());

        // Fresh store, cold memory: the disk entry must load. `lookup` skips an
        // exact match by design, so use `longest_prefix` here.
        let mut cold = PrefixKvCache::new(config.clone()).unwrap();
        assert!(cold.longest_prefix(&ids, None).is_some());

        // Flip one payload byte: checksum mismatch, never a wrong answer.
        let mut raw = fs::read(&file).unwrap();
        let mid = raw.len() / 2;
        raw[mid] ^= 0xff;
        fs::write(&file, &raw).unwrap();
        let mut cold = PrefixKvCache::new(config.clone()).unwrap();
        assert!(cold.longest_prefix(&ids, None).is_none());
        assert_eq!(cold.stats().disk_corrupt, 1);

        // Truncation is also a miss.
        fs::write(&file, &raw[..raw.len() - 7]).unwrap();
        let mut cold = PrefixKvCache::new(config).unwrap();
        assert!(cold.longest_prefix(&ids, None).is_none());
        assert_eq!(cold.stats().disk_corrupt, 1);
    }

    #[test]
    fn disk_snapshot_round_trips_exactly() {
        let model = tiny_model();
        let dir = TempDir::new("roundtrip");
        let config = PrefixKvConfig {
            byte_budget: 16 * 1024 * 1024,
            model_fingerprint: "m".into(),
            tokenizer_version: "t".into(),
            disk_dir: Some(dir.path().to_path_buf()),
        };
        let ids = prompt(11, 9);
        let mut store = PrefixKvCache::new(config.clone()).unwrap();
        let mut c = empty_cache_for(&model);
        model.forward_prefill(&ids, &mut c);
        store.insert(&ids, &c, Some("p1")).unwrap();

        let mut cold = PrefixKvCache::new(config).unwrap();
        let hit = cold.longest_prefix(&ids, Some("p1")).expect("disk hit");
        assert_eq!(hit.covered_len, ids.len());
        assert_eq!(hit.cache.seq_len(), c.seq_len());
        for (a, b) in hit.cache.layers.iter().zip(c.layers.iter()) {
            assert_eq!(a.k, b.k);
            assert_eq!(a.v, b.v);
        }
        assert_eq!(hit.cache.block_reps, c.block_reps);
        assert_eq!(hit.cache.cached_token_ids, c.cached_token_ids);
        assert_eq!(cold.stats().disk_hits, 1);

        // A different scope must not read the same file.
        let mut cold = PrefixKvCache::new(cache_config_with_dir(dir.path())).unwrap();
        assert!(cold.longest_prefix(&ids, Some("p2")).is_none());
    }

    fn cache_config_with_dir(dir: &Path) -> PrefixKvConfig {
        PrefixKvConfig {
            byte_budget: 16 * 1024 * 1024,
            model_fingerprint: "m".into(),
            tokenizer_version: "t".into(),
            disk_dir: Some(dir.to_path_buf()),
        }
    }

    #[test]
    fn scope_isolation_never_shares_a_hit() {
        let model = tiny_model();
        let mut store = cache(16 * 1024 * 1024);
        let ids = prompt(10, 10);
        let mut c = empty_cache_for(&model);
        model.forward_prefill(&ids, &mut c);
        store.insert(&ids, &c, Some("alice")).unwrap();

        let mut probe = ids.clone();
        probe.push(4);

        assert!(store.lookup(&probe, Some("bob")).is_none());
        assert!(store.lookup(&probe, None).is_none());
        assert!(store.lookup(&probe, Some("alice")).is_some());
        assert_eq!(store.longest_prefix(&ids, Some("alice")).unwrap().covered_len, ids.len());
        assert!(store.stats().hits >= 1);
        assert!(store.stats().misses >= 2);
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn answer_equality_prefix_plus_suffix_matches_full_prefill() {
        let model = build_model(16, 32, 3, 0.9);
        let mut store = cache(64 * 1024 * 1024);

        let prefix = prompt(14, 11);
        let mut full = prefix.clone();
        full.push(5);

        // Warm the cache with the prefix only.
        let mut warm = empty_cache_for(&model);
        model.forward_prefill(&prefix, &mut warm);
        store.insert(&prefix, &warm, Some("alice")).unwrap();

        let cached = prefill_with_prefix(&model, &full, &mut store, Some("alice"));
        assert_eq!(cached.covered_len, prefix.len());
        assert_eq!(cached.prefilled_tokens, 1);

        let reference_all = model.forward_prefill(&full, &mut empty_cache_for(&model));
        let vs = model.vocab_size;
        let reference = &reference_all[(full.len() - 1) * vs..];

        let diff = max_abs_diff(&cached.logits, reference);
        eprintln!("prefix+suffix vs full prefill: max |dlogit| = {diff:e}");
        assert!(diff <= 1e-5, "cached answer changed the logits: max diff {diff:e} > 1e-5");

        // The uncached default path must agree with the reference too.
        let plain = model.forward(&full);
        let plain_last = &plain[(full.len() - 1) * vs..];
        let plain_diff = max_abs_diff(plain_last, reference);
        eprintln!("forward vs forward_prefill: max |dlogit| = {plain_diff:e}");
        assert!(plain_diff <= 1e-4, "baseline kernels disagree: max diff {plain_diff:e}");
    }

    #[test]
    fn answer_equality_with_ple() {
        let model = ple_model();
        let mut store = cache(64 * 1024 * 1024);
        let prefix = prompt(10, 13);
        let mut full = prefix.clone();
        full.push(6);

        let mut warm = empty_cache_for(&model);
        model.forward_prefill(&prefix, &mut warm);
        assert!(warm.ple_prefix.is_some(), "PLE path must be active");
        store.insert(&prefix, &warm, None).unwrap();

        let cached = prefill_with_prefix(&model, &full, &mut store, None);
        assert_eq!(cached.covered_len, prefix.len());

        let reference_all = model.forward_prefill(&full, &mut empty_cache_for(&model));
        let vs = model.vocab_size;
        let reference = &reference_all[(full.len() - 1) * vs..];
        let diff = max_abs_diff(&cached.logits, reference);
        eprintln!("PLE equality: max |dlogit| = {diff:e}");
        assert!(diff <= 1e-5, "PLE suffix changed the answer: {diff:e}");
    }

    #[test]
    fn long_prefix_equality_holds() {
        let model = build_model(12, 24, 2, 1.4);
        let mut store = cache(64 * 1024 * 1024);
        let prefix = prompt(30, 12);
        let mut full = prefix.clone();
        full.extend([1u32, 2, 3, 4]);

        let mut warm = empty_cache_for(&model);
        model.forward_prefill(&prefix, &mut warm);
        store.insert(&prefix, &warm, None).unwrap();

        let cached = prefill_with_prefix(&model, &full, &mut store, None);
        assert_eq!(cached.covered_len, prefix.len());
        assert_eq!(cached.prefilled_tokens, 4);

        let reference_all = model.forward_prefill(&full, &mut empty_cache_for(&model));
        let vs = model.vocab_size;
        let reference = &reference_all[(full.len() - 1) * vs..];
        let diff = max_abs_diff(&cached.logits, reference);
        eprintln!("long prefix equality: max |dlogit| = {diff:e}");
        assert!(diff <= 1e-5, "max diff {diff:e}");
    }

    /// Timing benchmark. `cargo test -p ferrisres --lib prefix_kv_cache::tests::benchmark -- --ignored --nocapture`
    #[test]
    #[ignore = "timing benchmark, run explicitly"]
    fn benchmark_two_k_prompt_with_shared_prefix() {
        use std::time::Instant;

        let model = build_model(32, 64, 4, 2.1);
        let vocab = model.vocab_size;
        let shared = 1500usize;
        let total = 2000usize;
        let mut prompt_ids: Vec<u32> =
            (0..shared).map(|i| (i as u32 * 7919 + 13) % vocab as u32).collect();
        prompt_ids.extend((0..total - shared).map(|i| (i as u32 * 104729 + 7) % vocab as u32));
        assert_eq!(prompt_ids.len(), total);

        let mut warm_store = cache(512 * 1024 * 1024);
        let mut warm_cache = empty_cache_for(&model);
        let t_pre = Instant::now();
        model.forward_prefill(&prompt_ids[..shared], &mut warm_cache);
        warm_store.insert(&prompt_ids[..shared], &warm_cache, Some("bench")).unwrap();
        let warm_insert_time = t_pre.elapsed();

        // Cold turn: an empty store, no cache for this prompt.
        let mut cold_store = cache(512 * 1024 * 1024);
        let t0 = Instant::now();
        let cold = prefill_with_prefix(&model, &prompt_ids, &mut cold_store, Some("bench"));
        let cold_time = t0.elapsed();

        // Warm turn: the same 2k prompt, 1.5k of it already cached.
        let t1 = Instant::now();
        let warm = prefill_with_prefix(&model, &prompt_ids, &mut warm_store, Some("bench"));
        let warm_time = t1.elapsed();

        // Reference: full prefill with no cache at all.
        let t2 = Instant::now();
        model.forward_prefill(&prompt_ids, &mut empty_cache_for(&model));
        let full_time = t2.elapsed();

        eprintln!(
            "prompt {total} tokens, shared prefix {shared}:\n  cold (no cache): prefilled {} tokens in {:?}\n  warm (prefix hit): prefilled {} tokens in {:?}\n  full prefill:    prefilled {} tokens in {:?}\n  warming insert ({} tokens): {:?}",
            cold.prefilled_tokens,
            cold_time,
            warm.prefilled_tokens,
            warm_time,
            total,
            full_time,
            shared,
            warm_insert_time
        );
        assert_eq!(cold.prefilled_tokens, total);
        assert_eq!(warm.covered_len, shared);
        assert_eq!(warm.prefilled_tokens, total - shared);
        assert!(warm_time < full_time);
    }
}
