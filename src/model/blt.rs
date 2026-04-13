//! Byte Latent Transformer (BLT) tokenizer.
//!
//! Implements byte-level tokenization with dynamic patching based on entropy,
//! following the BLT paper (https://arxiv.org/abs/2402.00841).
//!
//! Key concepts:
//! - Raw UTF-8 bytes as input (no fixed vocabulary)
//! - Dynamic patching: group bytes into variable-length patches based on
//!   next-byte entropy (high entropy = patch boundary)
//! - Entropy-based patch boundary prediction (small MLP)
//! - Patch embedding: mean/sum of byte embeddings within each patch
//! - Cross-patch attention for inter-patch context


// ---------------------------------------------------------------------------
// ByteEmbedding — maps raw bytes to embeddings
// ---------------------------------------------------------------------------

/// Byte-level embedding table: maps each of the 256 byte values to a vector.
#[derive(Debug, Clone)]
pub struct ByteEmbedding {
    /// Embedding table: [256 × dim].
    embeddings: Vec<Vec<f32>>,
    /// Embedding dimension.
    dim: usize,
}

impl ByteEmbedding {
    pub fn new(dim: usize) -> Self {
        let scale = (2.0 / dim as f32).sqrt();
        let embeddings: Vec<Vec<f32>> = (0..256)
            .map(|b| {
                // Deterministic init from byte value
                (0..dim)
                    .map(|i| {
                        let hash = ((b as f32 * 127.0 + i as f32 * 311.0).sin() * 43758.5453).fract() - 0.5;
                        hash * scale
                    })
                    .collect()
            })
            .collect();
        Self { embeddings, dim }
    }

    /// Get the embedding for a byte.
    pub fn embed(&self, byte: u8) -> &[f32] {
        &self.embeddings[byte as usize]
    }

    /// Embed a slice of bytes, returning a flat vector.
    pub fn embed_bytes(&self, bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() * self.dim);
        for &b in bytes {
            out.extend_from_slice(self.embed(b));
        }
        out
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// EntropyModel — predicts next-byte entropy for patch boundaries
// ---------------------------------------------------------------------------

/// Small MLP that predicts the entropy (cross-entropy) of the next byte
/// given a local context window. High predicted entropy → patch boundary.
pub struct EntropyModel {
    /// Context window size.
    window_size: usize,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Layer 1 weights: [hidden × (window_size * 256)].
    w1: Vec<f32>,
    /// Layer 1 bias: [hidden].
    b1: Vec<f32>,
    /// Layer 2 weights: [1 × hidden].
    w2: Vec<f32>,
    /// Layer 2 bias.
    b2: f32,
}

impl EntropyModel {
    pub fn new(window_size: usize, hidden_dim: usize) -> Self {
        let in_dim = window_size * 256; // One-hot byte context
        let scale1 = (2.0 / in_dim as f32).sqrt();
        let scale2 = (2.0 / hidden_dim as f32).sqrt();

        let w1: Vec<f32> = (0..hidden_dim * in_dim)
            .map(|i| (((i as f32 * 0.618).sin() * 43758.5453).fract() - 0.5) * scale1)
            .collect();
        let b1 = vec![0.0; hidden_dim];
        let w2: Vec<f32> = (0..hidden_dim)
            .map(|i| (((i as f32 * 0.314).sin() * 12345.6789).fract() - 0.5) * scale2)
            .collect();

        Self {
            window_size,
            hidden_dim,
            w1,
            b1,
            w2,
            b2: 0.0,
        }
    }

    /// Predict entropy for a byte position given preceding bytes.
    ///
    /// Returns a value in [0, 1] where higher = more entropy = patch boundary.
    pub fn predict_entropy(&self, context: &[u8]) -> f32 {
        let in_dim = self.window_size * 256;
        // Create sparse one-hot input
        let mut hidden = self.b1.clone();

        for (pos, &byte) in context.iter().enumerate() {
            let offset = pos * 256 + byte as usize;
            for h in 0..self.hidden_dim {
                hidden[h] += self.w1[h * in_dim + offset];
            }
        }

        // ReLU activation
        for h in &mut hidden {
            if *h < 0.0 { *h = 0.0; }
        }

        // Output layer
        let mut out = self.b2;
        for h in 0..self.hidden_dim {
            out += self.w2[h] * hidden[h];
        }

        // Sigmoid to get [0, 1]
        1.0 / (1.0 + (-out).exp())
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

// ---------------------------------------------------------------------------
// PatchBoundary — where to split bytes into patches
// ---------------------------------------------------------------------------

/// A patch boundary decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchBoundary {
    /// Byte position after which to insert a boundary.
    pub position: usize,
    /// Predicted entropy at this position.
    pub entropy: f32,
}

// ---------------------------------------------------------------------------
// BltTokenizer — the main tokenizer
// ---------------------------------------------------------------------------

/// Configuration for the BLT tokenizer.
#[derive(Debug, Clone)]
pub struct BltConfig {
    /// Embedding dimension for bytes.
    pub embed_dim: usize,
    /// Context window for entropy model.
    pub entropy_window: usize,
    /// Hidden dimension for entropy MLP.
    pub entropy_hidden: usize,
    /// Entropy threshold above which to create a patch boundary.
    pub entropy_threshold: f32,
    /// Minimum patch length (bytes).
    pub min_patch_len: usize,
    /// Maximum patch length (bytes).
    pub max_patch_len: usize,
    /// How to aggregate byte embeddings within a patch.
    pub patch_aggregation: PatchAggregation,
}

/// How to aggregate byte embeddings within a patch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatchAggregation {
    /// Mean of byte embeddings.
    Mean,
    /// Sum of byte embeddings.
    Sum,
}

impl Default for BltConfig {
    fn default() -> Self {
        Self {
            embed_dim: 256,
            entropy_window: 4,
            entropy_hidden: 64,
            entropy_threshold: 0.5,
            min_patch_len: 1,
            max_patch_len: 16,
            patch_aggregation: PatchAggregation::Mean,
        }
    }
}

/// A patch of bytes with its embedding.
#[derive(Debug, Clone)]
pub struct BytePatch {
    /// The raw bytes in this patch.
    pub bytes: Vec<u8>,
    /// The patch embedding (aggregated byte embeddings).
    pub embedding: Vec<f32>,
    /// Start position in the original byte sequence.
    pub start: usize,
    /// End position (exclusive).
    pub end: usize,
}

impl BytePatch {
    /// Length in bytes.
    pub fn len_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Whether this patch is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// The BLT tokenizer: byte-level tokenization with entropy-based dynamic patching.
pub struct BltTokenizer {
    config: BltConfig,
    byte_embed: ByteEmbedding,
    entropy_model: EntropyModel,
}

impl BltTokenizer {
    pub fn new(config: BltConfig) -> Self {
        let byte_embed = ByteEmbedding::new(config.embed_dim);
        let entropy_model = EntropyModel::new(config.entropy_window, config.entropy_hidden);
        Self { config, byte_embed, entropy_model }
    }

    /// Get the configuration.
    pub fn config(&self) -> &BltConfig {
        &self.config
    }

    /// Get the byte embedding table.
    pub fn byte_embedding(&self) -> &ByteEmbedding {
        &self.byte_embed
    }

    /// Encode text to raw UTF-8 bytes.
    pub fn text_to_bytes(&self, text: &str) -> Vec<u8> {
        text.as_bytes().to_vec()
    }

    /// Decode bytes back to text (lossy).
    pub fn bytes_to_text(&self, bytes: &[u8]) -> String {
        String::from_utf8_lossy(bytes).into_owned()
    }

    /// Predict patch boundaries based on entropy.
    pub fn predict_boundaries(&self, bytes: &[u8]) -> Vec<PatchBoundary> {
        let mut boundaries = Vec::new();
        let window = self.config.entropy_window;

        for i in window..bytes.len() {
            let context = &bytes[i - window..i];
            let entropy = self.entropy_model.predict_entropy(context);

            if entropy >= self.config.entropy_threshold {
                boundaries.push(PatchBoundary {
                    position: i,
                    entropy,
                });
            }
        }

        boundaries
    }

    /// Apply min/max patch length constraints to boundaries.
    pub fn enforce_constraints(&self, boundaries: &[PatchBoundary], total_len: usize) -> Vec<PatchBoundary> {
        let mut filtered = Vec::new();
        let mut last_pos = 0;

        for b in boundaries {
            let patch_len = b.position - last_pos;
            if patch_len >= self.config.min_patch_len {
                filtered.push(*b);
                last_pos = b.position;
            }
        }

        // Ensure the last patch doesn't exceed max_patch_len
        let mut result = Vec::new();
        let mut last = 0;
        for b in &filtered {
            let pos = b.position;
            // Split if too long
            while pos - last > self.config.max_patch_len {
                let split_pos = last + self.config.max_patch_len;
                result.push(PatchBoundary {
                    position: split_pos,
                    entropy: 1.0, // Force boundary
                });
                last = split_pos;
            }
            if pos > last {
                result.push(PatchBoundary { position: pos, entropy: b.entropy });
                last = pos;
            }
        }
        // Handle remaining bytes
        while total_len - last > self.config.max_patch_len {
            let split_pos = last + self.config.max_patch_len;
            result.push(PatchBoundary {
                position: split_pos,
                entropy: 1.0,
            });
            last = split_pos;
        }

        result
    }

    /// Tokenize bytes into patches.
    pub fn tokenize(&self, bytes: &[u8]) -> Vec<BytePatch> {
        if bytes.is_empty() {
            return Vec::new();
        }

        let boundaries = self.predict_boundaries(bytes);
        let boundaries = self.enforce_constraints(&boundaries, bytes.len());

        let mut patches = Vec::new();
        let mut start = 0;

        for b in &boundaries {
            if b.position > start {
                patches.push(self.make_patch(bytes, start, b.position));
            }
            start = b.position;
        }
        // Last patch
        if start < bytes.len() {
            patches.push(self.make_patch(bytes, start, bytes.len()));
        }

        patches
    }

    /// Tokenize text into patches.
    pub fn tokenize_text(&self, text: &str) -> Vec<BytePatch> {
        let bytes = self.text_to_bytes(text);
        self.tokenize(&bytes)
    }

    /// Create a patch from a byte range.
    fn make_patch(&self, bytes: &[u8], start: usize, end: usize) -> BytePatch {
        let patch_bytes = bytes[start..end].to_vec();
        let embedding = self.aggregate_embeddings(&patch_bytes);
        BytePatch {
            bytes: patch_bytes,
            embedding,
            start,
            end,
        }
    }

    /// Aggregate byte embeddings within a patch.
    fn aggregate_embeddings(&self, bytes: &[u8]) -> Vec<f32> {
        let dim = self.config.embed_dim;
        let mut sum = vec![0.0; dim];

        for &b in bytes {
            let emb = self.byte_embed.embed(b);
            for (i, &v) in emb.iter().enumerate() {
                sum[i] += v;
            }
        }

        match self.config.patch_aggregation {
            PatchAggregation::Mean if !bytes.is_empty() => {
                let n = bytes.len() as f32;
                for v in &mut sum {
                    *v /= n;
                }
            }
            PatchAggregation::Sum => {}
            PatchAggregation::Mean => {} // Empty patch, leave as zeros
        }

        sum
    }

    /// Count the number of patches for a given byte sequence.
    pub fn count_patches(&self, bytes: &[u8]) -> usize {
        self.tokenize(bytes).len()
    }

    /// Get patch embeddings as a flat buffer (for GPU upload).
    pub fn patch_embeddings_flat(&self, patches: &[BytePatch]) -> Vec<f32> {
        let mut flat = Vec::with_capacity(patches.len() * self.config.embed_dim);
        for p in patches {
            flat.extend_from_slice(&p.embedding);
        }
        flat
    }

    /// Get patch byte lengths.
    pub fn patch_lengths(&self, patches: &[BytePatch]) -> Vec<usize> {
        patches.iter().map(|p| p.len_bytes()).collect()
    }
}

// ---------------------------------------------------------------------------
// CrossPatchAttention — attention across patches
// ---------------------------------------------------------------------------

/// Cross-patch attention for inter-patch context.
///
/// Operates on patch embeddings and allows information flow between
/// variable-length patches.
pub struct CrossPatchAttention {
    /// Number of attention heads.
    num_heads: usize,
    /// Head dimension.
    head_dim: usize,
    /// Whether to use causal masking.
    causal: bool,
}

impl CrossPatchAttention {
    pub fn new(num_heads: usize, head_dim: usize, causal: bool) -> Self {
        Self { num_heads, head_dim, causal }
    }

    /// Apply cross-patch attention.
    ///
    /// patch_embeds: [num_patches × (num_heads * head_dim)]
    /// Returns: [num_patches × (num_heads * head_dim)]
    pub fn forward(&self, patch_embeds: &[f32], num_patches: usize) -> Vec<f32> {
        let dim = self.num_heads * self.head_dim;
        if num_patches == 0 || patch_embeds.len() < num_patches * dim {
            return Vec::new();
        }

        let mut output = vec![0.0f32; num_patches * dim];
        let scale = (self.head_dim as f32).sqrt();

        for h in 0..self.num_heads {
            let h_offset = h * self.head_dim;

            for q_idx in 0..num_patches {
                let mut max_score = f32::NEG_INFINITY;

                // Compute attention scores
                let mut scores = Vec::with_capacity(num_patches);
                for k_idx in 0..num_patches {
                    if self.causal && k_idx > q_idx {
                        scores.push(f32::NEG_INFINITY);
                        continue;
                    }

                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        let qi = patch_embeds[q_idx * dim + h_offset + d];
                        let ki = patch_embeds[k_idx * dim + h_offset + d];
                        dot += qi * ki;
                    }
                    let score = dot / scale;
                    scores.push(score);
                    if score > max_score {
                        max_score = score;
                    }
                }

                // Softmax
                let mut sum_exp = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                }
                if sum_exp > 0.0 {
                    for s in &mut scores {
                        *s /= sum_exp;
                    }
                }

                // Weighted sum of values
                for k_idx in 0..num_patches {
                    let attn = scores[k_idx];
                    if attn < 1e-8 { continue; }
                    for d in 0..self.head_dim {
                        let vi = patch_embeds[k_idx * dim + h_offset + d];
                        output[q_idx * dim + h_offset + d] += attn * vi;
                    }
                }
            }
        }

        output
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_embedding() {
        let emb = ByteEmbedding::new(64);
        assert_eq!(emb.dim(), 64);
        assert_eq!(emb.embed(b'A').len(), 64);
        assert_eq!(emb.embed(0).len(), 64);

        // Different bytes → different embeddings
        assert_ne!(emb.embed(b'A'), emb.embed(b'B'));
    }

    #[test]
    fn test_byte_embedding_batch() {
        let emb = ByteEmbedding::new(32);
        let bytes = b"hello";
        let embedded = emb.embed_bytes(bytes);
        assert_eq!(embedded.len(), 5 * 32);
    }

    #[test]
    fn test_entropy_model_output_range() {
        let model = EntropyModel::new(4, 32);
        let context = b"hell";
        let entropy = model.predict_entropy(context);
        assert!(entropy >= 0.0 && entropy <= 1.0, "Entropy {} out of range", entropy);
    }

    #[test]
    fn test_entropy_model_deterministic() {
        let model = EntropyModel::new(4, 32);
        let ctx = b"test";
        let e1 = model.predict_entropy(ctx);
        let e2 = model.predict_entropy(ctx);
        assert!((e1 - e2).abs() < 1e-6);
    }

    #[test]
    fn test_blt_tokenize_empty() {
        let tokenizer = BltTokenizer::new(BltConfig::default());
        let patches = tokenizer.tokenize(&[]);
        assert!(patches.is_empty());
    }

    #[test]
    fn test_blt_tokenize_basic() {
        let config = BltConfig {
            entropy_threshold: 0.8, // High threshold → fewer boundaries
            min_patch_len: 1,
            max_patch_len: 8,
            ..BltConfig::default()
        };
        let tokenizer = BltTokenizer::new(config);
        let text = "Hello, world!";
        let patches = tokenizer.tokenize_text(text);
        assert!(!patches.is_empty());
        // All bytes should be covered
        let total_bytes: usize = patches.iter().map(|p| p.len_bytes()).sum();
        assert_eq!(total_bytes, text.len());
    }

    #[test]
    fn test_blt_tokenize_patches_cover_input() {
        let config = BltConfig {
            max_patch_len: 4,
            ..BltConfig::default()
        };
        let tokenizer = BltTokenizer::new(config);
        let bytes = b"abcdefghijklmnopqrstuvwxyz";
        let patches = tokenizer.tokenize(bytes);

        let total: usize = patches.iter().map(|p| p.len_bytes()).sum();
        assert_eq!(total, bytes.len());

        // Check no gaps
        assert_eq!(patches[0].start, 0);
        for i in 1..patches.len() {
            assert_eq!(patches[i].start, patches[i - 1].end);
        }
        assert_eq!(patches.last().unwrap().end, bytes.len());
    }

    #[test]
    fn test_blt_patch_embedding_dimension() {
        let config = BltConfig {
            embed_dim: 128,
            ..BltConfig::default()
        };
        let tokenizer = BltTokenizer::new(config);
        let patches = tokenizer.tokenize(b"test data");
        for p in &patches {
            assert_eq!(p.embedding.len(), 128);
        }
    }

    #[test]
    fn test_blt_patch_aggregation_mean() {
        let config = BltConfig {
            embed_dim: 4,
            patch_aggregation: PatchAggregation::Mean,
            entropy_threshold: 1.1, // No entropy boundaries → single patch
            max_patch_len: 100,
            ..BltConfig::default()
        };
        let tokenizer = BltTokenizer::new(config);
        let patches = tokenizer.tokenize(b"ab");
        assert_eq!(patches.len(), 1);
        // Mean of embed('a') and embed('b')
        let emb_a = tokenizer.byte_embedding().embed(b'a');
        let emb_b = tokenizer.byte_embedding().embed(b'b');
        for i in 0..4 {
            let expected = (emb_a[i] + emb_b[i]) / 2.0;
            assert!((patches[0].embedding[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_blt_patch_aggregation_sum() {
        let config = BltConfig {
            embed_dim: 4,
            patch_aggregation: PatchAggregation::Sum,
            entropy_threshold: 1.1,
            max_patch_len: 100,
            ..BltConfig::default()
        };
        let tokenizer = BltTokenizer::new(config);
        let patches = tokenizer.tokenize(b"ab");
        assert_eq!(patches.len(), 1);
        let emb_a = tokenizer.byte_embedding().embed(b'a');
        let emb_b = tokenizer.byte_embedding().embed(b'b');
        for i in 0..4 {
            let expected = emb_a[i] + emb_b[i];
            assert!((patches[0].embedding[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_blt_enforce_min_patch_len() {
        let config = BltConfig {
            min_patch_len: 3,
            entropy_threshold: 0.0, // All boundaries
            max_patch_len: 100,
            ..BltConfig::default()
        };
        let tokenizer = BltTokenizer::new(config);
        let patches = tokenizer.tokenize(b"abcdefgh");
        for p in &patches {
            assert!(p.len_bytes() >= 3 || p.start + p.len_bytes() == 8,
                "Patch of len {} at pos {} should be >= {} or be the last patch",
                p.len_bytes(), p.start, 3);
        }
    }

    #[test]
    fn test_blt_enforce_max_patch_len() {
        let config = BltConfig {
            max_patch_len: 4,
            entropy_threshold: 1.1, // No entropy boundaries
            ..BltConfig::default()
        };
        let tokenizer = BltTokenizer::new(config);
        let bytes = vec![0u8; 20];
        let patches = tokenizer.tokenize(&bytes);
        for p in &patches {
            assert!(p.len_bytes() <= 4, "Patch of len {} exceeds max 4", p.len_bytes());
        }
    }

    #[test]
    fn test_blt_flat_embeddings() {
        let config = BltConfig {
            embed_dim: 16,
            max_patch_len: 4,
            ..BltConfig::default()
        };
        let tokenizer = BltTokenizer::new(config);
        let patches = tokenizer.tokenize(b"hello world test");
        let flat = tokenizer.patch_embeddings_flat(&patches);
        assert_eq!(flat.len(), patches.len() * 16);
    }

    #[test]
    fn test_blt_patch_lengths() {
        let config = BltConfig {
            max_patch_len: 4,
            ..BltConfig::default()
        };
        let tokenizer = BltTokenizer::new(config);
        let patches = tokenizer.tokenize(b"hello world test");
        let lengths = tokenizer.patch_lengths(&patches);
        assert_eq!(lengths.len(), patches.len());
        let total: usize = lengths.iter().sum();
        assert_eq!(total, b"hello world test".len());
    }

    #[test]
    fn test_cross_patch_attention() {
        let attn = CrossPatchAttention::new(4, 16, false);
        let num_patches = 3;
        let dim = 4 * 16; // 64
        let embeds: Vec<f32> = (0..num_patches * dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let output = attn.forward(&embeds, num_patches);
        assert_eq!(output.len(), num_patches * dim);
    }

    #[test]
    fn test_cross_patch_attention_causal() {
        let attn = CrossPatchAttention::new(2, 8, true);
        let num_patches = 4;
        let dim = 16;
        let embeds = vec![1.0; num_patches * dim];
        let output = attn.forward(&embeds, num_patches);
        assert_eq!(output.len(), num_patches * dim);

        // First patch should attend only to itself
        // (with all-ones input, causal first patch gets self-attention)
        assert!(output[..dim].iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_cross_patch_attention_empty() {
        let attn = CrossPatchAttention::new(2, 8, false);
        let output = attn.forward(&[], 0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_text_to_bytes_roundtrip() {
        let tokenizer = BltTokenizer::new(BltConfig::default());
        let text = "Hello, 🦀 Rust!";
        let bytes = tokenizer.text_to_bytes(text);
        let decoded = tokenizer.bytes_to_text(&bytes);
        assert_eq!(decoded, text);
    }
}
