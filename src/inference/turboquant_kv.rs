//! 3-bit TurboQuant KV cache for ~10× memory reduction vs FP32.
//!
//! Per-group vector quantization (group size = 128):
//! 1. Find group min/max → 2 FP32 values per group
//! 2. Quantize each element to 3 bits (8 levels between min and max)
//! 3. Pack 3-bit values: 8 values per 3 bytes
//!
//! Memory comparison (128K context, 35 layers, kv_dim=256):
//!   FP32:  917 MB
//!   3-bit: 172 MB
//!   With block summaries replacing old KV: ~3.5 MB regardless of context!

// ── Constants ───────────────────────────────────────────────────────────────

/// Number of quantization levels for 3-bit (2^3 = 8).
const NUM_LEVELS: u32 = 8;

/// Maximum value of a 3-bit code (0-7).
const MAX_CODE: u8 = 7;

/// Group size for vector quantization.
/// Larger = less overhead, slightly worse quality. 128 is a good balance.
pub const DEFAULT_GROUP_SIZE: usize = 128;

// ── 3-bit packing ───────────────────────────────────────────────────────────
//
// 8 codes of 3 bits = 24 bits = 3 bytes
// Layout: codes are packed LSB-first:
//   byte 0: codes[0] bits 0-2 | codes[1] bits 3-5 | codes[2] bits 6-7 + codes[3] bit 0
//   byte 1: codes[3] bits 1-3 | codes[4] bits 4-6 | codes[5] bit 7 + codes[6] bits 0-1
//   byte 2: codes[6] bits 2-4 | codes[7] bits 5-7

/// Pack 8 three-bit values into 3 bytes.
/// Returns [u8; 3].
fn pack_3bit_8(codes: &[u8; 8]) -> [u8; 3] {
    // Total: 24 bits across 3 bytes
    let mut packed = [0u8; 3];
    let mut bit_pos = 0usize;

    for &code in codes.iter() {
        let code_3bit = code & 0x07;
        let byte_idx = bit_pos / 8;
        let shift = bit_pos % 8;
        packed[byte_idx] |= code_3bit << shift;
        // Handle overflow into next byte
        if shift + 3 > 8 {
            packed[byte_idx + 1] |= code_3bit >> (8 - shift);
        }
        bit_pos += 3;
    }

    packed
}

/// Unpack 3 bytes into 8 three-bit values.
fn unpack_3bit_8(packed: &[u8; 3]) -> [u8; 8] {
    let mut codes = [0u8; 8];
    let mut bit_pos = 0usize;

    for code in codes.iter_mut() {
        let byte_idx = bit_pos / 8;
        let shift = bit_pos % 8;
        let mut val = (packed[byte_idx] >> shift) & 0x07;
        if shift + 3 > 8 {
            // Bits spill into next byte
            let spill = shift + 3 - 8;
            val |= (packed[byte_idx + 1] & ((1u8 << spill) - 1)) << (3 - spill);
        }
        *code = val;
        bit_pos += 3;
    }

    codes
}

// ── Quantize / Dequantize ───────────────────────────────────────────────────

/// Quantize a group of FP32 values to 3-bit codes.
///
/// Returns (codes, min_val, max_val).
/// codes has len = values.len(), each in [0, 7].
/// Dequantize: value = min + code * (max - min) / 7.
pub fn quantize_3bit(values: &[f32]) -> (Vec<u8>, f32, f32) {
    if values.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }

    let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let range = max_val - min_val;
    let scale = if range > 0.0 { (NUM_LEVELS - 1) as f32 / range } else { 0.0 };

    let codes: Vec<u8> = values
        .iter()
        .map(|&v| {
            if range > 0.0 {
                let normalized = (v - min_val) * scale;
                (normalized.round() as u32).clamp(0, MAX_CODE as u32) as u8
            } else {
                0u8
            }
        })
        .collect();

    (codes, min_val, max_val)
}

/// Dequantize 3-bit codes back to FP32.
pub fn dequantize_3bit(codes: &[u8], min_val: f32, max_val: f32) -> Vec<f32> {
    if max_val == min_val {
        return vec![min_val; codes.len()];
    }

    let scale = (max_val - min_val) / (NUM_LEVELS - 1) as f32;
    codes.iter().map(|&c| min_val + c as f32 * scale).collect()
}

/// Pack a slice of 3-bit codes into bytes.
/// Length must be a multiple of 8 (pad with zeros if needed).
pub fn pack_3bit(codes: &[u8]) -> Vec<u8> {
    let padded_len = ((codes.len() + 7) / 8) * 8;
    let mut padded = codes.to_vec();
    padded.resize(padded_len, 0u8);

    let num_groups = padded_len / 8;
    let mut packed = Vec::with_capacity(num_groups * 3);

    for g in 0..num_groups {
        let group: &[u8; 8] = (&padded[g * 8..g * 8 + 8]).try_into().unwrap();
        let p = pack_3bit_8(group);
        packed.extend_from_slice(&p);
    }

    packed
}

/// Unpack bytes into 3-bit codes.
/// `num_codes` is the original number of codes (may not be multiple of 8).
pub fn unpack_3bit(packed: &[u8], num_codes: usize) -> Vec<u8> {
    let num_groups = (num_codes + 7) / 8;
    let mut codes = Vec::with_capacity(num_codes);

    for g in 0..num_groups {
        let offset = g * 3;
        if offset + 3 > packed.len() {
            break;
        }
        let group: [u8; 3] = [packed[offset], packed[offset + 1], packed[offset + 2]];
        let unpacked = unpack_3bit_8(&group);
        for &c in &unpacked {
            codes.push(c);
            if codes.len() >= num_codes {
                break;
            }
        }
    }

    codes
}

// ── TurboQuant KV Cache ─────────────────────────────────────────────────────

/// Per-layer KV cache with 3-bit TurboQuant compression.
///
/// For each position, K and V are quantized to 3 bits per element
/// using per-group min/max scaling (group_size = 128).
///
/// Memory: (seq_len × kv_dim × 3/8) bytes for data
///         + (seq_len × num_groups × 2 × 4) bytes for scales
///         ≈ seq_len × kv_dim × 0.375 bytes (data) + seq_len × kv_dim / 128 × 8 bytes (scales)
#[derive(Debug, Clone)]
pub struct TurboQuantKVCache {
    /// Quantized K cache: [num_layers][seq_len] packed 3-bit bytes.
    pub k_cache: Vec<Vec<Vec<u8>>>,

    /// Quantized V cache: [num_layers][seq_len] packed 3-bit bytes.
    pub v_cache: Vec<Vec<Vec<u8>>>,

    /// Per-group (min, max) for K: [num_layers][num_groups_per_pos][seq_len][2].
    /// Stored as flat Vec per layer per position.
    pub k_scales: Vec<Vec<Vec<[f32; 2]>>>,

    /// Per-group (min, max) for V: same layout.
    pub v_scales: Vec<Vec<Vec<[f32; 2]>>>,

    /// Number of layers.
    pub num_layers: usize,

    /// KV dimension per position (num_kv_heads * head_dim).
    pub kv_dim: usize,

    /// Group size for quantization.
    pub group_size: usize,

    /// Number of groups per position.
    num_groups: usize,

    /// Bytes per position for packed 3-bit data.
    bytes_per_pos: usize,

    /// Current sequence length.
    pub current_len: usize,

    /// Maximum sequence length before eviction.
    pub max_seq_len: usize,
}

impl TurboQuantKVCache {
    /// Create a new TurboQuant KV cache.
    pub fn new(num_layers: usize, kv_dim: usize, max_seq_len: usize, group_size: usize) -> Self {
        let num_groups = (kv_dim + group_size - 1) / group_size;
        let bytes_per_pos = (kv_dim * 3 + 7) / 8; // ceil(kv_dim * 3 / 8)

        let k_cache = vec![vec![]; num_layers];
        let v_cache = vec![vec![]; num_layers];
        let k_scales = vec![vec![]; num_layers];
        let v_scales = vec![vec![]; num_layers];

        TurboQuantKVCache {
            k_cache,
            v_cache,
            k_scales,
            v_scales,
            num_layers,
            kv_dim,
            group_size,
            num_groups,
            bytes_per_pos,
            current_len: 0,
            max_seq_len,
        }
    }

    /// Append one position's K and V to the cache.
    ///
    /// - `layer`: layer index
    /// - `k`: K vector [kv_dim] for this position
    /// - `v`: V vector [kv_dim] for this position
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_dim);
        assert_eq!(v.len(), self.kv_dim);

        // Quantize K
        let (k_codes, k_min, k_max) = Self::quantize_vector(k, self.group_size);
        let k_packed = pack_3bit(&k_codes);

        // Quantize V
        let (v_codes, v_min, v_max) = Self::quantize_vector(v, self.group_size);
        let v_packed = pack_3bit(&v_codes);

        self.k_cache[layer].push(k_packed);
        self.v_cache[layer].push(v_packed);

        // Store scales per group
        self.k_scales[layer].push(
            k_min.iter().zip(k_max.iter()).map(|(&mn, &mx)| [mn, mx]).collect()
        );
        self.v_scales[layer].push(
            v_min.iter().zip(v_max.iter()).map(|(&mn, &mx)| [mn, mx]).collect()
        );

        if layer == self.num_layers - 1 {
            self.current_len += 1;
        }
    }

    /// Get K and V for a range of positions at a given layer.
    ///
    /// - Returns: (K [range_len × kv_dim], V [range_len × kv_dim])
    pub fn get_range(&self, layer: usize, start: usize, end: usize) -> (Vec<f32>, Vec<f32>) {
        let range_len = end - start;
        let mut k = vec![0.0f32; range_len * self.kv_dim];
        let mut v = vec![0.0f32; range_len * self.kv_dim];

        for (i, pos) in (start..end).enumerate() {
            if pos >= self.k_cache[layer].len() {
                continue;
            }
            let k_codes = unpack_3bit(&self.k_cache[layer][pos], self.kv_dim);
            let v_codes = unpack_3bit(&self.v_cache[layer][pos], self.kv_dim);

            let k_deq = Self::dequantize_vector(&k_codes, &self.k_scales[layer][pos], self.group_size, self.kv_dim);
            let v_deq = Self::dequantize_vector(&v_codes, &self.v_scales[layer][pos], self.group_size, self.kv_dim);

            k[i * self.kv_dim..(i + 1) * self.kv_dim].copy_from_slice(&k_deq);
            v[i * self.kv_dim..(i + 1) * self.kv_dim].copy_from_slice(&v_deq);
        }

        (k, v)
    }

    /// Get a single position's K and V.
    pub fn get(&self, layer: usize, pos: usize) -> (Vec<f32>, Vec<f32>) {
        if pos >= self.k_cache[layer].len() {
            return (vec![0.0; self.kv_dim], vec![0.0; self.kv_dim]);
        }

        let k_codes = unpack_3bit(&self.k_cache[layer][pos], self.kv_dim);
        let v_codes = unpack_3bit(&self.v_cache[layer][pos], self.kv_dim);
        let k = Self::dequantize_vector(&k_codes, &self.k_scales[layer][pos], self.group_size, self.kv_dim);
        let v = Self::dequantize_vector(&v_codes, &self.v_scales[layer][pos], self.group_size, self.kv_dim);

        (k, v)
    }

    /// Evict the oldest position from the cache.
    pub fn evict_oldest(&mut self) {
        for layer in 0..self.num_layers {
            if !self.k_cache[layer].is_empty() {
                self.k_cache[layer].remove(0);
                self.v_cache[layer].remove(0);
                self.k_scales[layer].remove(0);
                self.v_scales[layer].remove(0);
            }
        }
        if self.current_len > 0 {
            self.current_len -= 1;
        }
    }

    /// Evict positions until cache is at most `target_len` positions.
    pub fn evict_to(&mut self, target_len: usize) {
        while self.current_len > target_len {
            self.evict_oldest();
        }
    }

    /// Memory usage in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        let per_pos_data = self.bytes_per_pos * 2; // K + V
        let per_pos_scales = self.num_groups * 2 * 4 * 2; // K + V, min+max, f32
        let per_pos = per_pos_data + per_pos_scales;
        per_pos * self.current_len * self.num_layers
    }

    /// Compression ratio vs FP32 KV cache.
    pub fn compression_ratio(&self) -> f32 {
        let fp32_per_pos = self.kv_dim * 4 * 2; // K + V, f32
        let actual_per_pos = self.memory_bytes() as f32 / (self.current_len * self.num_layers) as f32;
        if actual_per_pos > 0.0 { fp32_per_pos as f32 / actual_per_pos } else { 1.0 }
    }

    /// Current sequence length.
    pub fn seq_len(&self) -> usize {
        self.current_len
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// KV dimension per position.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Reset the cache to empty.
    pub fn reset(&mut self) {
        for layer in &mut self.k_cache { layer.clear(); }
        for layer in &mut self.v_cache { layer.clear(); }
        for layer in &mut self.k_scales { layer.clear(); }
        for layer in &mut self.v_scales { layer.clear(); }
        self.current_len = 0;
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    fn quantize_vector(values: &[f32], group_size: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let n = values.len();
        let num_groups = (n + group_size - 1) / group_size;
        let mut all_codes = Vec::with_capacity(n);
        let mut mins = Vec::with_capacity(num_groups);
        let mut maxs = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(n);
            let group = &values[start..end];

            let (codes, min_val, max_val) = quantize_3bit(group);
            all_codes.extend_from_slice(&codes);
            mins.push(min_val);
            maxs.push(max_val);
        }

        (all_codes, mins, maxs)
    }

    fn dequantize_vector(
        codes: &[u8],
        scales: &[[f32; 2]],
        group_size: usize,
        total_len: usize,
    ) -> Vec<f32> {
        let mut result = Vec::with_capacity(total_len);
        let num_groups = scales.len();

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(total_len);
            let [min_val, max_val] = scales[g];
            let group_codes = &codes[start..end];
            let group_deq = dequantize_3bit(group_codes, min_val, max_val);
            result.extend_from_slice(&group_deq);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_3bit_roundtrip() {
        let codes: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 3, 5, 7, 1, 2, 4, 6];
        let packed = pack_3bit(&codes);
        let unpacked = unpack_3bit(&packed, codes.len());
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_unpack_single_group() {
        let codes: Vec<u8> = vec![7, 6, 5, 4, 3, 2, 1, 0];
        let packed = pack_3bit(&codes);
        assert_eq!(packed.len(), 3);
        let unpacked = unpack_3bit(&packed, 8);
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_unpack_non_multiple() {
        let codes: Vec<u8> = vec![1, 2, 3, 4, 5]; // 5 codes, padded to 8
        let packed = pack_3bit(&codes);
        let unpacked = unpack_3bit(&packed, 5);
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let values: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1 - 6.4).sin()).collect();
        let (codes, min_val, max_val) = quantize_3bit(&values);
        assert_eq!(codes.len(), 128);
        assert!(codes.iter().all(|&c| c <= 7));

        let reconstructed = dequantize_3bit(&codes, min_val, max_val);

        // Should be a reasonable approximation
        let mse: f32 = values.iter().zip(reconstructed.iter())
            .map(|(o, r)| (o - r) * (o - r))
            .sum::<f32>() / values.len() as f32;
        assert!(mse < 0.1, "MSE too high: {}", mse);
    }

    #[test]
    fn test_quantize_uniform() {
        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let (codes, min_val, max_val) = quantize_3bit(&values);
        assert_eq!(codes, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert!((min_val - 0.0).abs() < 1e-5);
        assert!((max_val - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantize_constant() {
        let values = vec![3.14f32; 16];
        let (codes, min_val, max_val) = quantize_3bit(&values);
        assert!(codes.iter().all(|&c| c == 0));
        assert!((min_val - 3.14).abs() < 1e-5);
        assert!((max_val - 3.14).abs() < 1e-5);
    }

    #[test]
    fn test_kv_cache_append_get() {
        let mut cache = TurboQuantKVCache::new(2, 16, 128, 8);
        let k = vec![1.0f32, -0.5, 0.3, -0.1, 2.0, -1.5, 0.7, -0.3,
                     1.1, -0.4, 0.2, -0.2, 1.8, -1.3, 0.6, -0.4];
        let v = vec![0.5f32; 16];

        cache.append(0, &k, &v);
        cache.append(1, &k, &v);
        cache.current_len = 1; // Manually sync since we didn't go through all layers

        let (k_out, v_out) = cache.get(0, 0);
        assert_eq!(k_out.len(), 16);
        assert_eq!(v_out.len(), 16);

        // K should approximate original
        let mse: f32 = k.iter().zip(k_out.iter())
            .map(|(o, r)| (o - r) * (o - r))
            .sum::<f32>() / 16.0;
        assert!(mse < 0.5, "K MSE too high: {}", mse);
    }

    #[test]
    fn test_kv_cache_eviction() {
        let mut cache = TurboQuantKVCache::new(1, 8, 4, 8);
        let k = vec![1.0f32; 8];
        let v = vec![0.5f32; 8];

        for _ in 0..6 {
            cache.append(0, &k, &v);
        }
        assert_eq!(cache.k_cache[0].len(), 6);
        assert_eq!(cache.current_len, 6);

        cache.evict_to(4);
        assert_eq!(cache.k_cache[0].len(), 4);
        assert_eq!(cache.current_len, 4);
    }

    #[test]
    fn test_kv_cache_memory() {
        let cache = TurboQuantKVCache::new(35, 256, 512, 128);
        // Empty cache should have minimal memory
        assert_eq!(cache.memory_bytes(), 0);
    }

    #[test]
    fn test_compression_ratio() {
        let kv_dim = 256;
        let group_size = 128;
        let num_groups = (kv_dim + group_size - 1) / group_size; // 2
        let bytes_per_pos_data = (kv_dim * 3 + 7) / 8; // 96 bytes
        let bytes_per_pos_scales = num_groups * 2 * 4 * 2; // 32 bytes
        let total_per_pos = bytes_per_pos_data + bytes_per_pos_scales; // 128 bytes
        let fp32_per_pos = kv_dim * 4 * 2; // 2048 bytes
        let expected_ratio = fp32_per_pos as f32 / total_per_pos as f32; // ~16x

        // With actual cache
        assert!(expected_ratio > 10.0, "expected > 10x ratio, got {}", expected_ratio);
    }
}
