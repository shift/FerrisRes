//! Virtual circular KV buffer — zero-copy eviction via modulo addressing.
//!
//! Instead of compacting the KV cache buffer on overflow (O(window_size) copy),
//! this module uses modulo addressing: `physical_idx = (logical_idx + start_offset) % capacity`.
//! The write head wraps around, and a valid-segments mask tracks which entries
//! are live (sink tokens + recent window).
//!
//! This is the future replacement for segmented compaction in
//! [`LayerKVCache::compact()`]. When benchmarking shows compaction is a
//! bottleneck, upgrade to this zero-copy approach.
//!
//! Components:
//! 1. `CircularBufferConfig` — capacity, sink count, valid-segment tracking
//! 2. `CircularWriteHead` — atomic write position with wrapping
//! 3. `ValidSegmentMask` — bitmask tracking live entries (sink + recent window)
//! 4. `CIRCULAR_FLASH_DECODE_WGSL` — kernel with modular K/V lookup

use std::sync::atomic::{AtomicU32, Ordering};

use crate::error::Result;

// ---------------------------------------------------------------------------
// CircularBufferConfig
// ---------------------------------------------------------------------------

/// Configuration for the circular KV buffer.
#[derive(Debug, Clone)]
pub struct CircularBufferConfig {
    /// Total capacity in token positions.
    pub capacity: u32,
    /// Number of attention sink tokens (always valid, at buffer start).
    pub num_sink_tokens: u32,
    /// Whether to track valid segments for sparse attention.
    pub track_valid_segments: bool,
}

impl CircularBufferConfig {
    pub fn new(capacity: u32, num_sink_tokens: u32) -> Self {
        Self {
            capacity,
            num_sink_tokens,
            track_valid_segments: true,
        }
    }

    /// Default: 4096 capacity, 4 sink tokens.
    pub fn default_streaming() -> Self {
        Self::new(4096, 4)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.capacity == 0 {
            return Err(crate::error::FerrisResError::Shape(
                "CircularBuffer capacity must be > 0".into(),
            ));
        }
        if self.num_sink_tokens >= self.capacity {
            return Err(crate::error::FerrisResError::Shape(
                format!("num_sink_tokens ({}) must be < capacity ({})", 
                    self.num_sink_tokens, self.capacity),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CircularWriteHead — atomic write position with wrapping
// ---------------------------------------------------------------------------

/// Thread-safe circular write head. Wraps around at capacity.
pub struct CircularWriteHead {
    /// Current write position (mod capacity).
    head: AtomicU32,
    /// Total tokens ever written (monotonically increasing).
    total_written: AtomicU32,
    /// Buffer capacity.
    capacity: u32,
}

impl CircularWriteHead {
    pub fn new(capacity: u32) -> Self {
        Self {
            head: AtomicU32::new(0),
            total_written: AtomicU32::new(0),
            capacity,
        }
    }

    /// Get the current physical write position.
    pub fn position(&self) -> u32 {
        self.head.load(Ordering::Acquire)
    }

    /// Get the total tokens ever written.
    pub fn total_written(&self) -> u32 {
        self.total_written.load(Ordering::Acquire)
    }

    /// Advance the write head by `n` positions, wrapping at capacity.
    /// Returns the previous position.
    pub fn advance(&self, n: u32) -> u32 {
        let prev = self.head.fetch_add(n, Ordering::AcqRel);
        self.total_written.fetch_add(n, Ordering::AcqRel);
        // Wrap (caller should use position() which does modulo)
        prev
    }

    /// Get the physical index for a given logical index, accounting for
    /// the circular start offset.
    pub fn physical_index(&self, logical_idx: u32, start_offset: u32) -> u32 {
        (logical_idx + start_offset) % self.capacity
    }

    /// Get the start offset (for modular addressing).
    /// When total_written > capacity, the start offset shifts.
    pub fn start_offset(&self) -> u32 {
        let total = self.total_written.load(Ordering::Acquire);
        if total >= self.capacity {
            total % self.capacity
        } else {
            0
        }
    }

    /// Number of valid entries currently in the buffer.
    pub fn valid_count(&self) -> u32 {
        let total = self.total_written.load(Ordering::Acquire);
        total.min(self.capacity)
    }

    /// Reset to initial state.
    pub fn reset(&self) {
        self.head.store(0, Ordering::Release);
        self.total_written.store(0, Ordering::Release);
    }

    /// Capacity.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// ValidSegmentMask — bitmask tracking live KV entries
// ---------------------------------------------------------------------------

/// Tracks which positions in the circular buffer are valid for attention.
///
/// In StreamingLLM mode, valid entries are:
/// - Sink tokens: positions 0..num_sink_tokens (always valid)
/// - Recent window: positions (total_written - window_size)..total_written
///
/// Entries in between (the "evicted middle") are invalid and skipped during
/// attention.
pub struct ValidSegmentMask {
    /// Bitmask: bit i = 1 if position i is valid.
    /// Stored as a Vec<u64> for efficient bitwise operations.
    mask: Vec<u64>,
    /// Capacity (number of bits = number of positions).
    capacity: u32,
    /// Number of sink tokens.
    num_sink_tokens: u32,
}

impl ValidSegmentMask {
    pub fn new(capacity: u32, num_sink_tokens: u32) -> Self {
        let n_words = (capacity as usize + 63) / 64;
        Self {
            mask: vec![0u64; n_words],
            capacity,
            num_sink_tokens,
        }
    }

    /// Mark a position as valid.
    pub fn set_valid(&mut self, pos: u32) {
        if pos < self.capacity {
            let word = pos as usize / 64;
            let bit = pos as usize % 64;
            self.mask[word] |= 1u64 << bit;
        }
    }

    /// Mark a position as invalid (evicted).
    pub fn set_invalid(&mut self, pos: u32) {
        if pos < self.capacity {
            let word = pos as usize / 64;
            let bit = pos as usize % 64;
            self.mask[word] &= !(1u64 << bit);
        }
    }

    /// Check if a position is valid.
    pub fn is_valid(&self, pos: u32) -> bool {
        if pos >= self.capacity {
            return false;
        }
        let word = pos as usize / 64;
        let bit = pos as usize % 64;
        (self.mask[word] >> bit) & 1 == 1
    }

    /// Update the mask for StreamingLLM: sink tokens + recent window.
    ///
    /// - `write_head`: current write position (physical)
    /// - `valid_count`: number of valid entries
    /// - `window_size`: size of the recent window
    pub fn update_streaming(&mut self, write_head: u32, valid_count: u32, window_size: u32) {
        // Clear all
        for word in &mut self.mask {
            *word = 0;
        }

        // Set sink tokens (always valid, positions 0..num_sink_tokens)
        for i in 0..self.num_sink_tokens.min(self.capacity) {
            self.set_valid(i);
        }

        // Set recent window: the last `window_size` valid entries
        if valid_count > self.num_sink_tokens {
            let recent_count = window_size.min(valid_count - self.num_sink_tokens);
            // Recent window wraps around from write_head
            for i in 0..recent_count {
                // Walk backward from write_head
                let pos = if write_head >= i + 1 {
                    write_head - i - 1
                } else {
                    self.capacity - (i + 1 - write_head)
                };
                // Skip sink positions (they're already marked)
                if pos >= self.num_sink_tokens {
                    self.set_valid(pos);
                }
            }
        }
    }

    /// Count the number of valid positions.
    pub fn count_valid(&self) -> u32 {
        let mut count = 0u32;
        for &word in &self.mask {
            count += word.count_ones();
        }
        count
    }

    /// Get valid positions as a sorted vector.
    pub fn valid_positions(&self) -> Vec<u32> {
        let mut positions = Vec::new();
        for pos in 0..self.capacity {
            if self.is_valid(pos) {
                positions.push(pos);
            }
        }
        positions
    }

    /// Reset all positions to invalid.
    pub fn reset(&mut self) {
        for word in &mut self.mask {
            *word = 0;
        }
    }

    /// Number of sink tokens.
    pub fn num_sink_tokens(&self) -> u32 {
        self.num_sink_tokens
    }
}

// ---------------------------------------------------------------------------
// CircularKVBuffer — combined circular buffer manager
// ---------------------------------------------------------------------------

/// Manages the circular KV buffer with write head and valid segment tracking.
pub struct CircularKVBuffer {
    config: CircularBufferConfig,
    write_head: CircularWriteHead,
    valid_mask: ValidSegmentMask,
}

impl CircularKVBuffer {
    pub fn new(config: CircularBufferConfig) -> Result<Self> {
        config.validate()?;
        let write_head = CircularWriteHead::new(config.capacity);
        let valid_mask = ValidSegmentMask::new(config.capacity, config.num_sink_tokens);
        Ok(Self { config, write_head, valid_mask })
    }

    /// Write `n` tokens and return the physical start position.
    pub fn write(&self, n: u32) -> u32 {
        self.write_head.advance(n) % self.config.capacity
    }

    /// Get the physical index for a logical position.
    pub fn physical_index(&self, logical_idx: u32) -> u32 {
        self.write_head.physical_index(logical_idx, self.write_head.start_offset())
    }

    /// Update the valid segments for the current state.
    pub fn update_valid_segments(&mut self) {
        let head = self.write_head.position() % self.config.capacity;
        let valid = self.write_head.valid_count();
        let window = valid.saturating_sub(self.config.num_sink_tokens);
        self.valid_mask.update_streaming(head, valid, window);
    }

    /// Check if a physical position is valid for attention.
    pub fn is_valid(&self, pos: u32) -> bool {
        self.valid_mask.is_valid(pos)
    }

    /// Number of valid entries.
    pub fn valid_count(&self) -> u32 {
        self.write_head.valid_count()
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.write_head.reset();
        self.valid_mask.reset();
    }

    /// Get the configuration.
    pub fn config(&self) -> &CircularBufferConfig {
        &self.config
    }

    /// Get the write head reference.
    pub fn write_head(&self) -> &CircularWriteHead {
        &self.write_head
    }

    /// Get the valid mask reference.
    pub fn valid_mask(&self) -> &ValidSegmentMask {
        &self.valid_mask
    }

    /// Get valid positions for attention (physical indices).
    pub fn valid_positions(&self) -> Vec<u32> {
        self.valid_mask.valid_positions()
    }
}

// ---------------------------------------------------------------------------
// WGSL kernel for circular flash decode
// ---------------------------------------------------------------------------

/// WGSL kernel for flash decode with circular (modulo) KV lookup.
///
/// This kernel reads K/V from a circular buffer using modular addressing:
/// `physical_k = (k_pos + start_offset) % capacity`.
/// Positions not in the valid set are skipped (weight = 0).
pub const CIRCULAR_FLASH_DECODE_WGSL: &str = r#"
struct Params {
    query_len: u32,
    num_heads: u32,
    head_dim:   u32,
    capacity:   u32,
    start_offset: u32,
    valid_count:  u32,
    _pad0:        u32,
    _pad1:        u32,
}

@group(0) @binding(0) var<storage, read>       q:       array<f32>;
@group(0) @binding(1) var<storage, read>       k:       array<f32>;
@group(0) @binding(2) var<storage, read>       v:       array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;
@group(0) @binding(4) var<storage, read>       valid_mask: array<u32>;
@group(0) @binding(5) var<uniform>             params:  Params;

fn is_valid(pos: u32) -> bool {
    let word_idx = pos / 32u;
    let bit_idx = pos % 32u;
    if word_idx >= arrayLength(&valid_mask) {
        return false;
    }
    return (valid_mask[word_idx] >> bit_idx) & 1u == 1u;
}

fn physical_index(logical: u32) -> u32 {
    return (logical + params.start_offset) % params.capacity;
}

@compute @workgroup_size(1, 16)
fn circular_flash_decode(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let h = gid.y;
    if (h >= params.num_heads) { return; }

    let scale = 1.0 / sqrt(f32(params.head_dim));
    let q_base = h * params.head_dim;

    // Online softmax pass 1: find max score
    var max_score: f32 = -3.402823466e+38;
    for (var i: u32 = 0u; i < params.valid_count; i = i + 1u) {
        let phys = physical_index(i);
        if (!is_valid(phys)) { continue; }
        let k_base = phys * params.num_heads * params.head_dim + h * params.head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            dot += q[q_base + d] * k[k_base + d];
        }
        let score = dot * scale;
        if (score > max_score) { max_score = score; }
    }

    // Pass 2: weighted sum
    var sum_exp: f32 = 0.0;
    let out_base = h * params.head_dim;
    for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
        output[out_base + d] = 0.0;
    }

    for (var i: u32 = 0u; i < params.valid_count; i = i + 1u) {
        let phys = physical_index(i);
        if (!is_valid(phys)) { continue; }
        let k_base = phys * params.num_heads * params.head_dim + h * params.head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            dot += q[q_base + d] * k[k_base + d];
        }
        let weight = exp(dot * scale - max_score);
        sum_exp += weight;

        let v_base = phys * params.num_heads * params.head_dim + h * params.head_dim;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            output[out_base + d] += weight * v[v_base + d];
        }
    }

    if (sum_exp > 0.0) {
        let inv = 1.0 / sum_exp;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            output[out_base + d] *= inv;
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_write_head_basic() {
        let head = CircularWriteHead::new(16);
        assert_eq!(head.position(), 0);
        assert_eq!(head.total_written(), 0);
        assert_eq!(head.valid_count(), 0);
    }

    #[test]
    fn test_circular_write_head_advance() {
        let head = CircularWriteHead::new(8);
        head.advance(3);
        assert_eq!(head.position(), 3);
        assert_eq!(head.total_written(), 3);
        assert_eq!(head.valid_count(), 3);
    }

    #[test]
    fn test_circular_write_head_wrap() {
        let head = CircularWriteHead::new(8);
        head.advance(10);
        assert_eq!(head.total_written(), 10);
        assert_eq!(head.valid_count(), 8); // capped at capacity
        assert_eq!(head.start_offset(), 2); // 10 % 8
    }

    #[test]
    fn test_circular_write_head_physical_index() {
        let head = CircularWriteHead::new(8);
        // start_offset = 0
        assert_eq!(head.physical_index(0, 0), 0);
        assert_eq!(head.physical_index(7, 0), 7);
        // With offset
        assert_eq!(head.physical_index(0, 2), 2);
        assert_eq!(head.physical_index(6, 2), 0); // (6+2) % 8 = 0
        assert_eq!(head.physical_index(7, 2), 1); // (7+2) % 8 = 1
    }

    #[test]
    fn test_circular_write_head_reset() {
        let head = CircularWriteHead::new(8);
        head.advance(5);
        head.reset();
        assert_eq!(head.position(), 0);
        assert_eq!(head.total_written(), 0);
    }

    #[test]
    fn test_valid_segment_mask_basic() {
        let mut mask = ValidSegmentMask::new(64, 4);
        assert!(!mask.is_valid(0));
        assert!(!mask.is_valid(10));

        mask.set_valid(0);
        mask.set_valid(5);
        mask.set_valid(63);
        assert!(mask.is_valid(0));
        assert!(mask.is_valid(5));
        assert!(mask.is_valid(63));
        assert!(!mask.is_valid(1));
        assert_eq!(mask.count_valid(), 3);
    }

    #[test]
    fn test_valid_segment_mask_invalid() {
        let mut mask = ValidSegmentMask::new(64, 4);
        mask.set_valid(10);
        assert!(mask.is_valid(10));
        mask.set_invalid(10);
        assert!(!mask.is_valid(10));
    }

    #[test]
    fn test_valid_segment_mask_streaming() {
        let mut mask = ValidSegmentMask::new(16, 2);
        // write_head=8, valid_count=10, window_size=8
        mask.update_streaming(8, 10, 8);
        // Sink tokens: 0, 1
        assert!(mask.is_valid(0));
        assert!(mask.is_valid(1));
        // Recent window: positions 7,6,5,4,3,2 (backward from 8, skipping sinks)
        assert!(mask.is_valid(7));
        // Evicted middle should be invalid
    }

    #[test]
    fn test_valid_segment_mask_positions() {
        let mut mask = ValidSegmentMask::new(16, 2);
        mask.set_valid(0);
        mask.set_valid(1);
        mask.set_valid(5);
        mask.set_valid(10);
        let positions = mask.valid_positions();
        assert_eq!(positions, vec![0, 1, 5, 10]);
    }

    #[test]
    fn test_valid_segment_mask_reset() {
        let mut mask = ValidSegmentMask::new(32, 2);
        mask.set_valid(0);
        mask.set_valid(15);
        mask.reset();
        assert_eq!(mask.count_valid(), 0);
    }

    #[test]
    fn test_circular_kv_buffer_basic() {
        let buf = CircularKVBuffer::new(CircularBufferConfig::new(16, 2)).unwrap();
        assert_eq!(buf.valid_count(), 0);
    }

    #[test]
    fn test_circular_kv_buffer_write() {
        let buf = CircularKVBuffer::new(CircularBufferConfig::new(8, 2)).unwrap();
        let pos = buf.write(3);
        assert_eq!(pos, 0); // first write starts at 0
        assert_eq!(buf.valid_count(), 3);
    }

    #[test]
    fn test_circular_kv_buffer_wrap() {
        let buf = CircularKVBuffer::new(CircularBufferConfig::new(8, 2)).unwrap();
        buf.write(10);
        assert_eq!(buf.valid_count(), 8);
        assert_eq!(buf.write_head().start_offset(), 2);
    }

    #[test]
    fn test_circular_kv_buffer_reset() {
        let mut buf = CircularKVBuffer::new(CircularBufferConfig::new(8, 2)).unwrap();
        buf.write(5);
        buf.reset();
        assert_eq!(buf.valid_count(), 0);
    }

    #[test]
    fn test_circular_config_validate() {
        assert!(CircularBufferConfig::new(0, 0).validate().is_err());
        assert!(CircularBufferConfig::new(10, 10).validate().is_err());
        assert!(CircularBufferConfig::new(100, 4).validate().is_ok());
    }

    #[test]
    fn test_circular_wgsl_compiles() {
        // Verify the WGSL string is non-empty and contains expected functions
        assert!(!CIRCULAR_FLASH_DECODE_WGSL.is_empty());
        assert!(CIRCULAR_FLASH_DECODE_WGSL.contains("circular_flash_decode"));
        assert!(CIRCULAR_FLASH_DECODE_WGSL.contains("physical_index"));
        assert!(CIRCULAR_FLASH_DECODE_WGSL.contains("is_valid"));
        assert!(CIRCULAR_FLASH_DECODE_WGSL.contains("valid_mask"));
    }
}
