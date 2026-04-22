//! KV cache for CpuBlockAttnResModel inference.
//!
//! Enables O(1) per-token generation instead of O(n) by caching K/V projections
//! from previous positions. On the test machine (Skylake, 32GB):
//!   - Without cache: 217s for seq=42 tokens
//!   - With cache:    ~42s for seq=42 tokens (only new token computed)
//!
//! Handles:
//!   - Per-layer K/V caching
//!   - KV-shared layers (later layers reuse earlier layers' K/V)
//!   - PLE per-layer inputs
//!   - Inter-block attention block representations
//!   - RoPE position tracking

/// KV cache for a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// Cached K projections: [max_seq × kv_dim].
    /// Grows as new tokens are appended.
    pub k: Vec<f32>,
    /// Cached V projections: [max_seq × kv_dim].
    pub v: Vec<f32>,
    /// Number of cached positions.
    pub len: usize,
    /// Hidden dim for this layer.
    pub hidden_dim: usize,
    /// KV dim for this layer (num_kv_heads × head_dim).
    pub kv_dim: usize,
}

impl LayerKVCache {
    pub fn new(hidden_dim: usize, kv_dim: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
            hidden_dim,
            kv_dim,
        }
    }

    /// Pre-allocate for a known sequence length.
    pub fn with_capacity(seq_len: usize, kv_dim: usize, hidden_dim: usize) -> Self {
        Self {
            k: vec![0.0f32; seq_len * kv_dim],
            v: vec![0.0f32; seq_len * kv_dim],
            len: 0,
            hidden_dim,
            kv_dim,
        }
    }

    /// Append K/V for one token position.
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        let kv_dim = self.kv_dim;
        if self.k.len() < (self.len + 1) * kv_dim {
            self.k.extend_from_slice(k);
            self.v.extend_from_slice(v);
        } else {
            let offset = self.len * kv_dim;
            self.k[offset..offset + kv_dim].copy_from_slice(k);
            self.v[offset..offset + kv_dim].copy_from_slice(v);
        }
        self.len += 1;
    }

    /// Append K/V for multiple token positions (prefill).
    pub fn append_batch(&mut self, k: &[f32], v: &[f32], seq: usize) {
        let kv_dim = self.kv_dim;
        let total = seq * kv_dim;
        if self.k.len() < (self.len + seq) * kv_dim {
            self.k.resize((self.len + seq) * kv_dim, 0.0);
            self.v.resize((self.len + seq) * kv_dim, 0.0);
        }
        let offset = self.len * kv_dim;
        self.k[offset..offset + total].copy_from_slice(k);
        self.v[offset..offset + total].copy_from_slice(v);
        self.len += seq;
    }

    /// Get cached K/V up to current position.
    pub fn get(&self) -> (&[f32], &[f32]) {
        let end = self.len * self.kv_dim;
        (&self.k[..end], &self.v[..end])
    }

    /// Number of cached positions.
    pub fn seq_len(&self) -> usize {
        self.len
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.k.capacity() * 4 + self.v.capacity() * 4
    }
}

/// Full model KV cache across all layers.
pub struct ModelKVCache {
    /// Per-layer caches.
    pub layers: Vec<LayerKVCache>,
    /// Cached block representations for inter-block attention.
    pub block_reps: Vec<Vec<f32>>,
    /// Partial sum accumulator for current block.
    pub partial_sum: Vec<f32>,
    /// Current block token count (for averaging).
    pub block_token_count: usize,
    /// Shared KV mapping: layer_idx → source_layer_idx.
    /// For KV-shared layers, we point to the source layer's cache.
    pub shared_kv_sources: std::collections::HashMap<usize, usize>,
    /// PLE precomputed values for the full prefix.
    pub ple_prefix: Option<Vec<f32>>,
    /// Token IDs that have been processed (for PLE computation).
    pub cached_token_ids: Vec<u32>,
    /// Hidden dim.
    pub hidden_dim: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// PLE dim.
    pub ple_dim: usize,
    /// Block config.
    pub layers_per_block: usize,
}

impl ModelKVCache {
    pub fn new(
        num_layers: usize,
        hidden_dim: usize,
        kv_dims: &[usize],
        layers_per_block: usize,
        ple_dim: usize,
    ) -> Self {
        let layers = kv_dims.iter().map(|&kv_dim| {
            LayerKVCache::new(hidden_dim, kv_dim)
        }).collect();

        Self {
            layers,
            block_reps: Vec::new(),
            partial_sum: vec![0.0f32; hidden_dim],
            block_token_count: 0,
            shared_kv_sources: std::collections::HashMap::new(),
            ple_prefix: None,
            cached_token_ids: Vec::new(),
            hidden_dim,
            num_layers,
            ple_dim,
            layers_per_block,
        }
    }

    /// Memory usage across all layers.
    pub fn total_memory_bytes(&self) -> usize {
        let kv_mem: usize = self.layers.iter().map(|l| l.memory_bytes()).sum();
        let block_mem: usize = self.block_reps.iter().map(|b| b.len() * 4).sum();
        let ple_mem = self.ple_prefix.as_ref().map_or(0, |p| p.len() * 4);
        kv_mem + block_mem + ple_mem
    }

    /// Total cached sequence length (from layer 0).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.seq_len())
    }

    /// Clear all caches.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.block_reps.clear();
        self.partial_sum = vec![0.0f32; self.hidden_dim];
        self.block_token_count = 0;
        self.ple_prefix = None;
        self.cached_token_ids.clear();
    }
}
