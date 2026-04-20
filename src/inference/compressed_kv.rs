//! Compressed KV cache trait for inference.
//!
//! Unifies TurboQuantKVCache (3-bit quantized) and RecurrentKVCache
//! (block summaries) behind a common trait, allowing the CPU generator
//! to use either transparently.
//!
//! Also provides a basic uncompressed KV cache for reference/testing.


/// Trait for KV caches that can be used in the CPU decode loop.
///
/// All implementations provide per-layer K/V storage with append,
/// range retrieval, and memory tracking.
pub trait KVCacheBackend {
    /// Append a single token's K/V to the given layer.
    fn append(&mut self, layer: usize, k: &[f32], v: &[f32]);

    /// Get K/V for a range of positions in the given layer.
    /// Returns (keys, values) where each is [range_len * kv_dim].
    fn get_range(&self, layer: usize, start: usize, end: usize) -> (Vec<f32>, Vec<f32>);

    /// Get K/V for a single position in the given layer.
    /// Returns (key, value) where each is [kv_dim].
    fn get(&self, layer: usize, pos: usize) -> (Vec<f32>, Vec<f32>);

    /// Current sequence length (number of positions stored).
    fn len(&self) -> usize;

    /// Whether the cache is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total memory used by the cache in bytes.
    fn memory_bytes(&self) -> usize;

    /// Number of layers in the cache.
    fn num_layers(&self) -> usize;

    /// KV dimension per position per layer.
    fn kv_dim(&self) -> usize;

    /// Reset the cache.
    fn reset(&mut self);
}

/// Basic uncompressed KV cache for reference and testing.
///
/// Stores K/V as Vec<f32> per layer. Simple but uses full BF16 memory.
#[derive(Clone, Debug)]
pub struct BasicKVCache {
    /// Per-layer keys: [seq_len * kv_dim]
    keys: Vec<Vec<f32>>,
    /// Per-layer values: [seq_len * kv_dim]
    values: Vec<Vec<f32>>,
    /// KV dimension.
    kv_dim: usize,
    /// Current sequence length.
    seq_len: usize,
}

impl BasicKVCache {
    pub fn new(num_layers: usize, kv_dim: usize) -> Self {
        Self {
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
            kv_dim,
            seq_len: 0,
        }
    }
}

impl KVCacheBackend for BasicKVCache {
    fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_dim);
        assert_eq!(v.len(), self.kv_dim);
        self.keys[layer].extend_from_slice(k);
        self.values[layer].extend_from_slice(v);
        if layer == 0 {
            self.seq_len += 1;
        }
    }

    fn get_range(&self, layer: usize, start: usize, end: usize) -> (Vec<f32>, Vec<f32>) {
        let kv_dim = self.kv_dim;
        let keys = self.keys[layer][start * kv_dim..end * kv_dim].to_vec();
        let values = self.values[layer][start * kv_dim..end * kv_dim].to_vec();
        (keys, values)
    }

    fn get(&self, layer: usize, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let kv_dim = self.kv_dim;
        let k = self.keys[layer][pos * kv_dim..(pos + 1) * kv_dim].to_vec();
        let v = self.values[layer][pos * kv_dim..(pos + 1) * kv_dim].to_vec();
        (k, v)
    }

    fn len(&self) -> usize {
        self.seq_len
    }

    fn memory_bytes(&self) -> usize {
        let per_layer = self.seq_len * self.kv_dim * std::mem::size_of::<f32>();
        per_layer * self.keys.len() * 2 // keys + values
    }

    fn num_layers(&self) -> usize {
        self.keys.len()
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn reset(&mut self) {
        for layer in &mut self.keys {
            layer.clear();
        }
        for layer in &mut self.values {
            layer.clear();
        }
        self.seq_len = 0;
    }
}

/// Adapter: TurboQuantKVCache implements KVCacheBackend.
/// Since TurboQuantKVCache has its own API, we wrap it here.
pub struct TurboQuantAdapter {
    inner: crate::inference::turboquant_kv::TurboQuantKVCache,
}

impl TurboQuantAdapter {
    pub fn new(
        num_layers: usize,
        kv_dim: usize,
        max_seq_len: usize,
        group_size: usize,
    ) -> Self {
        Self {
            inner: crate::inference::turboquant_kv::TurboQuantKVCache::new(
                num_layers, kv_dim, max_seq_len, group_size,
            ),
        }
    }

    pub fn inner(&self) -> &crate::inference::turboquant_kv::TurboQuantKVCache {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut crate::inference::turboquant_kv::TurboQuantKVCache {
        &mut self.inner
    }
}

impl KVCacheBackend for TurboQuantAdapter {
    fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.inner.append(layer, k, v);
    }

    fn get_range(&self, layer: usize, start: usize, end: usize) -> (Vec<f32>, Vec<f32>) {
        self.inner.get_range(layer, start, end)
    }

    fn get(&self, layer: usize, pos: usize) -> (Vec<f32>, Vec<f32>) {
        self.inner.get(layer, pos)
    }

    fn len(&self) -> usize {
        self.inner.seq_len()
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn kv_dim(&self) -> usize {
        self.inner.kv_dim()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Adapter: RecurrentKVCache implements KVCacheBackend.
pub struct RecurrentKVAdapter {
    inner: crate::inference::recurrent_kv::RecurrentKVCache,
}

impl RecurrentKVAdapter {
    pub fn new(
        num_layers: usize,
        hidden_dim: usize,
        kv_dim: usize,
        window_size: usize,
        layers_per_block: usize,
        boundary_layers: Vec<usize>,
    ) -> Self {
        Self {
            inner: crate::inference::recurrent_kv::RecurrentKVCache::new(
                num_layers, hidden_dim, kv_dim, window_size, layers_per_block, boundary_layers,
            ),
        }
    }

    pub fn inner(&self) -> &crate::inference::recurrent_kv::RecurrentKVCache {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut crate::inference::recurrent_kv::RecurrentKVCache {
        &mut self.inner
    }
}

impl KVCacheBackend for RecurrentKVAdapter {
    fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.inner.append(layer, k, v);
    }

    fn get_range(&self, layer: usize, start: usize, end: usize) -> (Vec<f32>, Vec<f32>) {
        // RecurrentKV only stores recent positions; get recent range
        self.inner.get_recent(layer, start, end)
    }

    fn get(&self, layer: usize, pos: usize) -> (Vec<f32>, Vec<f32>) {
        self.inner.get_recent_pos(layer, pos)
    }

    fn len(&self) -> usize {
        self.inner.recent_len()
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn kv_dim(&self) -> usize {
        self.inner.kv_dim()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_kv_cache_roundtrip() {
        let mut cache = BasicKVCache::new(2, 4);
        assert!(cache.is_empty());
        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.kv_dim(), 4);

        cache.append(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        cache.append(0, &[10.0, 20.0, 30.0, 40.0], &[50.0, 60.0, 70.0, 80.0]);

        assert_eq!(cache.len(), 2);

        let (k, v) = cache.get(0, 0);
        assert_eq!(k, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v, &[5.0, 6.0, 7.0, 8.0]);

        let (k, v) = cache.get(0, 1);
        assert_eq!(k, &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(v, &[50.0, 60.0, 70.0, 80.0]);

        let (k, v) = cache.get_range(0, 0, 2);
        assert_eq!(k.len(), 8);
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn test_basic_kv_cache_reset() {
        let mut cache = BasicKVCache::new(1, 2);
        cache.append(0, &[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(cache.len(), 1);
        cache.reset();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_basic_kv_cache_memory() {
        let mut cache = BasicKVCache::new(2, 4);
        cache.append(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        // 1 position × 4 dims × 4 bytes × 2 (K+V) × 2 layers = 64 bytes
        assert_eq!(cache.memory_bytes(), 64);
    }
}
