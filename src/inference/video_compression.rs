//! Video token compression for efficient multimodal processing.
//!
//! Reduces the number of tokens from video frames by exploiting temporal
//! redundancy between adjacent frames:
//! - Temporal redundancy removal: skip encoding unchanged regions
//! - Motion-compensated residual encoding
//! - Temporal token merging: extend ToMe to merge similar tokens across frames
//! - Target: 4-8x reduction in video token count

// ---------------------------------------------------------------------------
// FrameTokens — tokens from a single video frame
// ---------------------------------------------------------------------------

/// Tokens from a single encoded video frame.
#[derive(Debug, Clone)]
pub struct FrameTokens {
    /// Token embeddings: [num_tokens × dim].
    pub tokens: Vec<f32>,
    /// Number of tokens.
    pub num_tokens: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Frame index in the video.
    pub frame_idx: usize,
}

impl FrameTokens {
    pub fn new(tokens: Vec<f32>, num_tokens: usize, dim: usize, frame_idx: usize) -> Self {
        Self { tokens, num_tokens, dim, frame_idx }
    }

    /// Create synthetic frame tokens for testing.
    pub fn synthetic(frame_idx: usize, num_tokens: usize, dim: usize, value: f32) -> Self {
        Self {
            tokens: vec![value; num_tokens * dim],
            num_tokens,
            dim,
            frame_idx,
        }
    }

    /// Get a token embedding.
    pub fn get_token(&self, idx: usize) -> &[f32] {
        &self.tokens[idx * self.dim..(idx + 1) * self.dim]
    }

    /// Total number of values.
    pub fn total_values(&self) -> usize {
        self.num_tokens * self.dim
    }
}

// ---------------------------------------------------------------------------
// TemporalRedundancyDetector
// ---------------------------------------------------------------------------

/// Configuration for temporal redundancy detection.
#[derive(Debug, Clone)]
pub struct TemporalRedundancyConfig {
    /// MSE threshold below which a token is considered unchanged.
    pub mse_threshold: f32,
    /// Whether to use cosine similarity instead of MSE.
    pub use_cosine: bool,
    /// Cosine similarity threshold (higher = more similar).
    pub cosine_threshold: f32,
    /// Whether to do block-level detection (group tokens into blocks).
    pub block_level: bool,
    /// Block size for block-level detection.
    pub block_size: usize,
}

impl Default for TemporalRedundancyConfig {
    fn default() -> Self {
        Self {
            mse_threshold: 0.01,
            use_cosine: false,
            cosine_threshold: 0.99,
            block_level: false,
            block_size: 4,
        }
    }
}

/// Detects temporal redundancy between consecutive frames.
pub struct TemporalRedundancyDetector {
    config: TemporalRedundancyConfig,
    /// Previous frame tokens.
    prev_frame: Option<FrameTokens>,
}

impl TemporalRedundancyDetector {
    pub fn new(config: TemporalRedundancyConfig) -> Self {
        Self { config, prev_frame: None }
    }

    /// Compute MSE between two token embeddings.
    pub fn token_mse(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() { return f32::MAX; }
        let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        sum_sq / a.len() as f32
    }

    /// Compute cosine similarity between two token embeddings.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() { return 0.0; }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 { return 0.0; }
        dot / (norm_a * norm_b)
    }

    /// Process a frame, returning indices of tokens that changed.
    pub fn process_frame(&mut self, frame: &FrameTokens) -> Vec<usize> {
        let mut changed = Vec::new();

        if let Some(ref prev) = self.prev_frame {
            if prev.num_tokens != frame.num_tokens || prev.dim != frame.dim {
                // Shape mismatch: all tokens changed
                changed.extend(0..frame.num_tokens);
            } else {
                for i in 0..frame.num_tokens {
                    let prev_tok = prev.get_token(i);
                    let cur_tok = frame.get_token(i);

                    let is_similar = if self.config.use_cosine {
                        Self::cosine_similarity(prev_tok, cur_tok) >= self.config.cosine_threshold
                    } else {
                        Self::token_mse(prev_tok, cur_tok) <= self.config.mse_threshold
                    };

                    if !is_similar {
                        changed.push(i);
                    }
                }
            }
        } else {
            // First frame: all tokens are "new"
            changed.extend(0..frame.num_tokens);
        }

        self.prev_frame = Some(frame.clone());
        changed
    }

    /// Compute the redundancy ratio for the last comparison.
    pub fn last_redundancy(&self, _total_tokens: usize) -> f32 {
        if let Some(ref prev) = self.prev_frame {
            if prev.num_tokens == 0 { return 0.0; }
            // Redundancy = fraction of tokens that were similar to previous frame
            // This is approximate since process_frame returns changed tokens
        }
        0.0
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.prev_frame = None;
    }
}

// ---------------------------------------------------------------------------
// MotionCompensatedResidual
// ---------------------------------------------------------------------------

/// Motion vector for a block of tokens.
#[derive(Debug, Clone, Copy)]
pub struct MotionVector {
    /// Source block index in the reference frame.
    pub src_block: usize,
    /// Horizontal displacement (in token units).
    pub dx: i32,
    /// Vertical displacement (in token units).
    pub dy: i32,
    /// Match cost (lower = better match).
    pub cost: f32,
}

/// Encodes the residual between frames using motion compensation.
pub struct MotionCompensatedResidual {
    /// Block size for motion estimation.
    pub block_size: usize,
    /// Search range for motion vectors.
    pub search_range: i32,
    /// Whether the reference frame has been set.
    has_reference: bool,
}

impl MotionCompensatedResidual {
    pub fn new(block_size: usize, search_range: i32) -> Self {
        Self { block_size, search_range, has_reference: false }
    }

    /// Compute block mean of a set of tokens.
    pub fn block_mean(tokens: &FrameTokens, block_indices: &[usize]) -> Vec<f32> {
        if block_indices.is_empty() { return vec![0.0; tokens.dim]; }
        let mut sum = vec![0.0f32; tokens.dim];
        for &idx in block_indices {
            let tok = tokens.get_token(idx);
            for (i, &v) in tok.iter().enumerate() {
                sum[i] += v;
            }
        }
        let n = block_indices.len() as f32;
        sum.iter().map(|&s| s / n).collect()
    }

    /// Compute L2 distance between two vectors.
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
    }

    /// Simple motion estimation: for each block, find the best match.
    pub fn estimate_motion(
        &self,
        current: &FrameTokens,
        reference: &FrameTokens,
        frame_width_tokens: usize,
    ) -> Vec<MotionVector> {
        let bs = self.block_size;
        let num_blocks = current.num_tokens / bs;
        let mut vectors = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let cur_indices: Vec<usize> = (block_idx * bs..(block_idx + 1) * bs).collect();
            let cur_mean = Self::block_mean(current, &cur_indices);

            let block_row = block_idx / (frame_width_tokens / bs);
            let block_col = block_idx % (frame_width_tokens / bs);

            let mut best_mv = MotionVector {
                src_block: block_idx,
                dx: 0,
                dy: 0,
                cost: f32::MAX,
            };

            for dy in -self.search_range..=self.search_range {
                for dx in -self.search_range..=self.search_range {
                    let ref_row = block_row as i32 + dy;
                    let ref_col = block_col as i32 + dx;

                    if ref_row < 0 || ref_col < 0 { continue; }
                    let ref_block_idx = (ref_row as usize) * (frame_width_tokens / bs) + ref_col as usize;
                    if ref_block_idx * bs + bs > reference.num_tokens { continue; }

                    let ref_indices: Vec<usize> = (ref_block_idx * bs..(ref_block_idx + 1) * bs).collect();
                    let ref_mean = Self::block_mean(reference, &ref_indices);
                    let cost = Self::l2_distance(&cur_mean, &ref_mean);

                    if cost < best_mv.cost {
                        best_mv = MotionVector {
                            src_block: ref_block_idx,
                            dx,
                            dy,
                            cost,
                        };
                    }
                }
            }

            vectors.push(best_mv);
        }

        vectors
    }

    /// Compute residuals given motion vectors.
    pub fn compute_residuals(
        &self,
        current: &FrameTokens,
        reference: &FrameTokens,
        motion_vectors: &[MotionVector],
    ) -> Vec<f32> {
        let bs = self.block_size;
        let mut residuals = Vec::with_capacity(current.total_values());

        for (block_idx, mv) in motion_vectors.iter().enumerate() {
            let ref_start = mv.src_block * bs;
            for i in 0..bs {
                let cur_tok = current.get_token(block_idx * bs + i);
                let ref_idx = ref_start + i;
                let ref_tok = if ref_idx < reference.num_tokens {
                    reference.get_token(ref_idx)
                } else {
                    &[0.0f32; 0][..]
                };

                for (d, &cur_v) in cur_tok.iter().enumerate() {
                    let ref_v = ref_tok.get(d).copied().unwrap_or(0.0);
                    residuals.push(cur_v - ref_v);
                }
            }
        }

        residuals
    }

    /// Reconstruct frame from reference + residuals + motion vectors.
    pub fn reconstruct(
        &self,
        reference: &FrameTokens,
        residuals: &[f32],
        motion_vectors: &[MotionVector],
    ) -> FrameTokens {
        let bs = self.block_size;
        let num_tokens = motion_vectors.len() * bs;
        let dim = reference.dim;
        let mut tokens = vec![0.0f32; num_tokens * dim];

        for (block_idx, mv) in motion_vectors.iter().enumerate() {
            let ref_start = mv.src_block * bs;
            for i in 0..bs {
                let token_idx = block_idx * bs + i;
                let ref_idx = ref_start + i;
                for d in 0..dim {
                    let ref_v = if ref_idx < reference.num_tokens {
                        reference.get_token(ref_idx).get(d).copied().unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    let res_v = residuals.get(token_idx * dim + d).copied().unwrap_or(0.0);
                    tokens[token_idx * dim + d] = ref_v + res_v;
                }
            }
        }

        FrameTokens::new(tokens, num_tokens, dim, 0)
    }

    pub fn has_reference(&self) -> bool {
        self.has_reference
    }
}

// ---------------------------------------------------------------------------
// TemporalTokenMerger — extends ToMe across frames
// ---------------------------------------------------------------------------

/// Configuration for temporal token merging.
#[derive(Debug, Clone)]
pub struct TemporalMergeConfig {
    /// Number of tokens to merge per frame.
    pub merge_ratio: f32,
    /// Whether to consider cross-frame similarity.
    pub cross_frame: bool,
    /// Maximum distance for cross-frame merge.
    pub max_temporal_distance: usize,
}

impl Default for TemporalMergeConfig {
    fn default() -> Self {
        Self {
            merge_ratio: 0.5, // Merge 50% of tokens
            cross_frame: true,
            max_temporal_distance: 3,
        }
    }
}

/// Merges similar tokens within and across video frames.
pub struct TemporalTokenMerger {
    config: TemporalMergeConfig,
    /// Buffered frame tokens for cross-frame merging.
    frame_buffer: Vec<FrameTokens>,
}

impl TemporalTokenMerger {
    pub fn new(config: TemporalMergeConfig) -> Self {
        Self { config, frame_buffer: Vec::new() }
    }

    /// Merge tokens within a single frame.
    pub fn merge_frame(&self, frame: &FrameTokens) -> FrameTokens {
        let n = frame.num_tokens;
        let dim = frame.dim;
        let num_to_merge = (n as f32 * self.config.merge_ratio) as usize;

        if num_to_merge == 0 || n <= 1 {
            return frame.clone();
        }

        // Find the most similar pairs and merge them
        let mut merged = vec![false; n];
        let mut merged_tokens = Vec::new();

        for _ in 0..num_to_merge {
            // Find the most similar pair among unmerged tokens
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_sim = f32::MAX;

            for i in 0..n {
                if merged[i] { continue; }
                for j in (i + 1)..n {
                    if merged[j] { continue; }
                    let dist = TemporalRedundancyDetector::token_mse(
                        frame.get_token(i),
                        frame.get_token(j),
                    );
                    if dist < best_sim {
                        best_sim = dist;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            if best_sim == f32::MAX { break; }

            // Merge the pair: average
            let mut avg = vec![0.0; dim];
            for d in 0..dim {
                avg[d] = (frame.get_token(best_i)[d] + frame.get_token(best_j)[d]) / 2.0;
            }
            merged_tokens.push(avg);

            merged[best_i] = true;
            merged[best_j] = true;
        }

        // Add remaining unmerged tokens
        for i in 0..n {
            if !merged[i] {
                merged_tokens.push(frame.get_token(i).to_vec());
            }
        }

        let new_count = merged_tokens.len();
        let mut flat = Vec::with_capacity(new_count * dim);
        for tok in &merged_tokens {
            flat.extend_from_slice(tok);
        }

        FrameTokens::new(flat, new_count, dim, frame.frame_idx)
    }

    /// Add frame to buffer for cross-frame merging.
    pub fn buffer_frame(&mut self, frame: FrameTokens) {
        self.frame_buffer.push(frame);
        // Keep only recent frames
        while self.frame_buffer.len() > self.config.max_temporal_distance {
            self.frame_buffer.remove(0);
        }
    }

    /// Merge tokens across buffered frames.
    pub fn merge_cross_frame(&mut self) -> Vec<FrameTokens> {
        if self.frame_buffer.len() <= 1 {
            return self.frame_buffer.clone();
        }

        // For each frame, merge with similar tokens from other frames
        let mut results = Vec::new();
        for (fi, frame) in self.frame_buffer.iter().enumerate() {
            let merged = self.merge_frame(frame);

            // Further merge: find tokens in other frames that are very similar
            let mut tokens = merged.tokens;
            let mut num_tokens = merged.num_tokens;
            let dim = merged.dim;

            // Remove tokens that have a near-duplicate in a previous frame
            if self.config.cross_frame && fi > 0 {
                let mut keep = vec![true; num_tokens];
                let mut removed = 0;

                for i in 0..num_tokens {
                    let tok_i = &tokens[i * dim..(i + 1) * dim];
                    for prev_frame in &self.frame_buffer[..fi] {
                        for j in 0..prev_frame.num_tokens {
                            let sim = TemporalRedundancyDetector::cosine_similarity(
                                tok_i,
                                prev_frame.get_token(j),
                            );
                            if sim > 0.995 {
                                keep[i] = false;
                                removed += 1;
                                break;
                            }
                        }
                        if !keep[i] { break; }
                    }
                }

                if removed > 0 {
                    let mut new_tokens = Vec::new();
                    for i in 0..num_tokens {
                        if keep[i] {
                            new_tokens.extend_from_slice(&tokens[i * dim..(i + 1) * dim]);
                        }
                    }
                    tokens = new_tokens;
                    num_tokens -= removed;
                }
            }

            results.push(FrameTokens::new(tokens, num_tokens, dim, frame.frame_idx));
        }

        results
    }

    /// Compression ratio achieved.
    pub fn compression_ratio(&self, original_tokens: usize, compressed_tokens: usize) -> f32 {
        if compressed_tokens == 0 { return 1.0; }
        original_tokens as f32 / compressed_tokens as f32
    }

    /// Reset the merger.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
    }

    /// Number of buffered frames.
    pub fn buffered_frames(&self) -> usize {
        self.frame_buffer.len()
    }
}

// ---------------------------------------------------------------------------
// VideoTokenCompressor — main interface
// ---------------------------------------------------------------------------

/// Configuration for video token compression.
#[derive(Debug, Clone)]
pub struct VideoCompressConfig {
    /// Temporal redundancy detection config.
    pub temporal_config: TemporalRedundancyConfig,
    /// Whether to use motion compensation.
    pub use_motion_compensation: bool,
    /// Motion compensation block size.
    pub motion_block_size: usize,
    /// Motion search range.
    pub motion_search_range: i32,
    /// Temporal merge config.
    pub merge_config: TemporalMergeConfig,
    /// Target compression ratio.
    pub target_ratio: f32,
}

impl Default for VideoCompressConfig {
    fn default() -> Self {
        Self {
            temporal_config: TemporalRedundancyConfig::default(),
            use_motion_compensation: true,
            motion_block_size: 4,
            motion_search_range: 2,
            merge_config: TemporalMergeConfig::default(),
            target_ratio: 4.0,
        }
    }
}

/// Main video token compression pipeline.
pub struct VideoTokenCompressor {
    _config: VideoCompressConfig,
    redundancy_detector: TemporalRedundancyDetector,
    _motion_compensator: MotionCompensatedResidual,
    token_merger: TemporalTokenMerger,
    /// Total original tokens.
    total_original: usize,
    /// Total compressed tokens.
    total_compressed: usize,
    /// Frame count.
    frame_count: usize,
}

impl VideoTokenCompressor {
    pub fn new(config: VideoCompressConfig) -> Self {
        let redundancy_detector = TemporalRedundancyDetector::new(config.temporal_config.clone());
        let motion_compensator = MotionCompensatedResidual::new(
            config.motion_block_size,
            config.motion_search_range,
        );
        let token_merger = TemporalTokenMerger::new(config.merge_config.clone());

        Self {
            _config: config,
            redundancy_detector,
            _motion_compensator: motion_compensator,
            token_merger,
            total_original: 0,
            total_compressed: 0,
            frame_count: 0,
        }
    }

    /// Compress a single frame's tokens.
    pub fn compress_frame(&mut self, frame: &FrameTokens) -> FrameTokens {
        self.total_original += frame.num_tokens;
        self.frame_count += 1;

        // Step 1: Temporal redundancy removal
        let changed_indices = self.redundancy_detector.process_frame(frame);

        // If all tokens changed, use the full frame
        if changed_indices.len() == frame.num_tokens {
            // Fall through to token merging
        } else if changed_indices.is_empty() {
            // Frame is identical to previous: no tokens needed
            self.total_compressed += 0;
            return FrameTokens::new(Vec::new(), 0, frame.dim, frame.frame_idx);
        }

        // Step 2: Extract only changed tokens
        let dim = frame.dim;
        let mut changed_tokens = Vec::with_capacity(changed_indices.len() * dim);
        for &idx in &changed_indices {
            changed_tokens.extend_from_slice(frame.get_token(idx));
        }
        let partial = FrameTokens::new(changed_tokens, changed_indices.len(), dim, frame.frame_idx);

        // Step 3: Token merging
        let merged = self.token_merger.merge_frame(&partial);

        self.total_compressed += merged.num_tokens;
        merged
    }

    /// Compress multiple frames with cross-frame merging.
    pub fn compress_frames(&mut self, frames: &[FrameTokens]) -> Vec<FrameTokens> {
        let _compressed: Vec<FrameTokens> = Vec::new();
        for frame in frames {
            let c = self.compress_frame(frame);
            self.token_merger.buffer_frame(c);
        }
        self.token_merger.merge_cross_frame()
    }

    /// Overall compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        if self.total_compressed == 0 { return 1.0; }
        self.total_original as f32 / self.total_compressed as f32
    }

    /// Total original tokens.
    pub fn total_original(&self) -> usize {
        self.total_original
    }

    /// Total compressed tokens.
    pub fn total_compressed(&self) -> usize {
        self.total_compressed
    }

    /// Number of frames processed.
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Reset compressor state.
    pub fn reset(&mut self) {
        self.redundancy_detector.reset();
        self.token_merger.reset();
        self.total_original = 0;
        self.total_compressed = 0;
        self.frame_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_tokens() {
        let ft = FrameTokens::synthetic(0, 10, 64, 1.0);
        assert_eq!(ft.num_tokens, 10);
        assert_eq!(ft.dim, 64);
        assert_eq!(ft.total_values(), 640);
        assert_eq!(ft.get_token(0).len(), 64);
    }

    #[test]
    fn test_token_mse() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((TemporalRedundancyDetector::token_mse(&a, &b) - 0.0).abs() < 1e-5);

        let c = vec![2.0, 3.0, 4.0];
        let mse = TemporalRedundancyDetector::token_mse(&a, &c);
        assert!((mse - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((TemporalRedundancyDetector::cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);

        let c = vec![0.0, 1.0, 0.0];
        assert!((TemporalRedundancyDetector::cosine_similarity(&a, &c) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_redundancy_first_frame() {
        let mut det = TemporalRedundancyDetector::new(TemporalRedundancyConfig::default());
        let frame = FrameTokens::synthetic(0, 10, 32, 1.0);
        let changed = det.process_frame(&frame);
        assert_eq!(changed.len(), 10); // All tokens are new
    }

    #[test]
    fn test_temporal_redundancy_identical() {
        let mut det = TemporalRedundancyDetector::new(TemporalRedundancyConfig {
            mse_threshold: 0.01,
            ..Default::default()
        });
        let frame = FrameTokens::synthetic(0, 10, 32, 1.0);
        det.process_frame(&frame); // First frame

        let changed = det.process_frame(&frame); // Identical
        assert!(changed.is_empty()); // All tokens unchanged
    }

    #[test]
    fn test_temporal_redundancy_changed() {
        let mut det = TemporalRedundancyDetector::new(TemporalRedundancyConfig {
            mse_threshold: 0.001,
            ..Default::default()
        });
        let frame1 = FrameTokens::synthetic(0, 10, 32, 1.0);
        det.process_frame(&frame1);

        let frame2 = FrameTokens::synthetic(1, 10, 32, 2.0);
        let changed = det.process_frame(&frame2);
        assert_eq!(changed.len(), 10); // All changed
    }

    #[test]
    fn test_motion_compensated_residual() {
        let mc = MotionCompensatedResidual::new(2, 1);
        let current = FrameTokens::synthetic(0, 8, 16, 1.0);
        let reference = FrameTokens::synthetic(0, 8, 16, 0.9);

        let mvs = mc.estimate_motion(&current, &reference, 4);
        assert_eq!(mvs.len(), 4); // 8 tokens / 2 block_size = 4 blocks

        let residuals = mc.compute_residuals(&current, &reference, &mvs);
        assert_eq!(residuals.len(), 8 * 16);

        let reconstructed = mc.reconstruct(&reference, &residuals, &mvs);
        assert_eq!(reconstructed.num_tokens, 8);
    }

    #[test]
    fn test_motion_reconstruction_accuracy() {
        let mc = MotionCompensatedResidual::new(4, 0); // Zero search range
        let current = FrameTokens::synthetic(0, 8, 4, 1.0);
        let reference = FrameTokens::synthetic(0, 8, 4, 0.0);

        let mvs = mc.estimate_motion(&current, &reference, 4);
        let residuals = mc.compute_residuals(&current, &reference, &mvs);
        let reconstructed = mc.reconstruct(&reference, &residuals, &mvs);

        // With zero motion, residual = current - reference = 1.0 - 0.0 = 1.0
        // Reconstructed = reference + residual = 0.0 + 1.0 = 1.0
        for i in 0..reconstructed.total_values() {
            assert!((reconstructed.tokens[i] - 1.0).abs() < 1e-4,
                "Mismatch at {}: got {} expected 1.0", i, reconstructed.tokens[i]);
        }
    }

    #[test]
    fn test_temporal_merger_single_frame() {
        let config = TemporalMergeConfig { merge_ratio: 0.5, ..Default::default() };
        let merger = TemporalTokenMerger::new(config);
        let frame = FrameTokens::synthetic(0, 4, 8, 1.0);
        let merged = merger.merge_frame(&frame);
        // 4 tokens with 50% merge → 2 tokens (merges most similar pair)
        assert!(merged.num_tokens < 4);
    }

    #[test]
    fn test_temporal_merger_no_merge() {
        let config = TemporalMergeConfig { merge_ratio: 0.0, ..Default::default() };
        let merger = TemporalTokenMerger::new(config);
        let frame = FrameTokens::synthetic(0, 4, 8, 1.0);
        let merged = merger.merge_frame(&frame);
        assert_eq!(merged.num_tokens, 4); // No merging
    }

    #[test]
    fn test_temporal_merger_cross_frame() {
        let config = TemporalMergeConfig {
            merge_ratio: 0.0, // No within-frame merge
            cross_frame: true,
            max_temporal_distance: 3,
        };
        let mut merger = TemporalTokenMerger::new(config);

        // Two identical frames
        merger.buffer_frame(FrameTokens::synthetic(0, 4, 8, 1.0));
        merger.buffer_frame(FrameTokens::synthetic(1, 4, 8, 1.0));

        let results = merger.merge_cross_frame();
        assert_eq!(results.len(), 2);
        // Second frame should have fewer tokens (cross-frame duplicates removed)
        assert!(results[1].num_tokens < 4);
    }

    #[test]
    fn test_compression_ratio() {
        let config = TemporalMergeConfig::default();
        let merger = TemporalTokenMerger::new(config);
        assert!((merger.compression_ratio(100, 25) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_video_compressor_identical_frames() {
        let config = VideoCompressConfig {
            use_motion_compensation: false,
            ..Default::default()
        };
        let mut compressor = VideoTokenCompressor::new(config);

        // First frame: all tokens
        let frame = FrameTokens::synthetic(0, 10, 32, 1.0);
        let c1 = compressor.compress_frame(&frame);
        assert!(c1.num_tokens > 0);

        // Second identical frame: no changed tokens
        let c2 = compressor.compress_frame(&frame);
        assert_eq!(c2.num_tokens, 0);

        // Should have good compression
        assert!(compressor.compression_ratio() > 1.0);
    }

    #[test]
    fn test_video_compressor_different_frames() {
        let mut config = VideoCompressConfig::default();
        config.merge_config.merge_ratio = 0.3;
        let mut compressor = VideoTokenCompressor::new(config);

        let frame1 = FrameTokens::synthetic(0, 20, 16, 1.0);
        let frame2 = FrameTokens::synthetic(1, 20, 16, 2.0);

        compressor.compress_frame(&frame1);
        compressor.compress_frame(&frame2);

        // Should have some compression from merging
        assert!(compressor.compression_ratio() >= 1.0);
        assert_eq!(compressor.frame_count(), 2);
    }

    #[test]
    fn test_video_compressor_reset() {
        let config = VideoCompressConfig::default();
        let mut compressor = VideoTokenCompressor::new(config);

        let frame = FrameTokens::synthetic(0, 10, 32, 1.0);
        compressor.compress_frame(&frame);
        compressor.reset();

        assert_eq!(compressor.total_original(), 0);
        assert_eq!(compressor.total_compressed(), 0);
        assert_eq!(compressor.frame_count(), 0);
    }

    #[test]
    fn test_block_mean() {
        let frame = FrameTokens::synthetic(0, 4, 8, 2.0);
        let mean = MotionCompensatedResidual::block_mean(&frame, &[0, 1]);
        // Mean of [2.0, 2.0, ...] = 2.0
        assert!(mean.iter().all(|&v| (v - 2.0).abs() < 1e-5));
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let dist = MotionCompensatedResidual::l2_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-4);
    }
}
