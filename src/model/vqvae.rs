//! VQ-VAE codebook for multimodal discrete tokenization.
//!
//! Implements Vector Quantization with:
//! - Learnable codebook: K embedding vectors with nearest-neighbor lookup
//! - L2 commitment loss + codebook loss for training
//! - Straight-through estimator (STE) for gradient flow through discretization
//! - Exponential moving average (EMA) codebook updates
//!
//! Can be used standalone or integrated with vision/audio encoders for
//! unified discrete token representations across modalities.
//!
//! Reference: van den Oord et al., "Neural Discrete Representation Learning" (2017)

// ---------------------------------------------------------------------------
// VQCodebook — core vector quantization codebook
// ---------------------------------------------------------------------------

/// A learnable VQ-VAE codebook.
///
/// Maps continuous embedding vectors to discrete codes via nearest-neighbor
/// lookup in a codebook of K vectors. Supports training with commitment loss
/// and EMA updates.
pub struct VQCodebook {
    /// Codebook vectors: [codebook_size × dim].
    embeddings: Vec<f32>,
    /// Number of codebook entries.
    codebook_size: usize,
    /// Embedding dimension.
    dim: usize,
    /// EMA decay rate (0.99 default).
    ema_decay: f32,
    /// EMA usage counts per code: [codebook_size].
    ema_counts: Vec<f32>,
    /// EMA weighted sums per code: [codebook_size × dim].
    ema_sum: Vec<f32>,
    /// Whether EMA tracking is initialized.
    ema_initialized: bool,
    /// Commitment loss coefficient (β).
    commitment_cost: f32,
}

impl VQCodebook {
    /// Create a new codebook with random initialization.
    pub fn new(codebook_size: usize, dim: usize) -> Self {
        // Xavier-uniform-like initialization
        let scale = (2.0 / (dim as f32 + codebook_size as f32)).sqrt();
        let mut embeddings = Vec::with_capacity(codebook_size * dim);
        for i in 0..codebook_size * dim {
            // Deterministic pseudo-random from index
            let x = ((i as f32 * 0.618 + 0.1).sin() * 43758.5453).fract() - 0.5;
            embeddings.push(x * scale);
        }

        Self {
            embeddings,
            codebook_size,
            dim,
            ema_decay: 0.99,
            ema_counts: vec![0.0; codebook_size],
            ema_sum: vec![0.0; codebook_size * dim],
            ema_initialized: false,
            commitment_cost: 0.25,
        }
    }

    /// Create with specific initialization values.
    pub fn with_embeddings(embeddings: Vec<f32>, dim: usize) -> Self {
        let codebook_size = embeddings.len() / dim;
        Self {
            embeddings,
            codebook_size,
            dim,
            ema_decay: 0.99,
            ema_counts: vec![0.0; codebook_size],
            ema_sum: vec![0.0; codebook_size * dim],
            ema_initialized: false,
            commitment_cost: 0.25,
        }
    }

    /// Set the EMA decay rate.
    pub fn with_ema_decay(mut self, decay: f32) -> Self {
        self.ema_decay = decay;
        self
    }

    /// Set the commitment cost (β).
    pub fn with_commitment_cost(mut self, cost: f32) -> Self {
        self.commitment_cost = cost;
        self
    }

    /// Find the nearest codebook entry (returns index).
    pub fn nearest(&self, vector: &[f32]) -> usize {
        debug_assert_eq!(vector.len(), self.dim);
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for k in 0..self.codebook_size {
            let offset = k * self.dim;
            let mut dist = 0.0f32;
            for d in 0..self.dim {
                let diff = vector[d] - self.embeddings[offset + d];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_idx = k;
            }
        }

        best_idx
    }

    /// Quantize a vector: find nearest, return (code_index, quantized_vector, loss).
    pub fn quantize(&self, vector: &[f32]) -> QuantizeResult {
        let code = self.nearest(vector);
        let quantized = self.code(code).to_vec();

        // L2 distances
        let mut encoder_dist = 0.0f32; // ||z_e - sg(e_k)||
        let mut codebook_dist = 0.0f32; // ||sg(z_e) - e_k||

        for d in 0..self.dim {
            let diff_enc = vector[d] - quantized[d];
            encoder_dist += diff_enc * diff_enc;

            let diff_cb = vector[d] - quantized[d]; // sg(z_e) = z_e for distance
            codebook_dist += diff_cb * diff_cb;
        }

        // VQ-VAE loss: codebook_loss + commitment_loss
        // codebook_loss = ||sg(z_e) - e_k||² (push codebook toward encoder output)
        // commitment_loss = β * ||z_e - sg(e_k)||² (push encoder output toward codebook)
        let codebook_loss = codebook_dist;
        let commitment_loss = self.commitment_cost * encoder_dist;
        let total_loss = codebook_loss + commitment_loss;

        QuantizeResult {
            code,
            quantized,
            total_loss,
            codebook_loss,
            commitment_loss,
        }
    }

    /// Straight-through estimator: forward uses quantized, backward passes
    /// gradients through as if the quantization didn't happen.
    ///
    /// Returns (code, z_quantized_ste) where STE output is:
    ///   z_ste = z_e + (z_q - z_e).detach() = z_q
    /// but gradients flow through z_e.
    pub fn quantize_ste(&self, vector: &[f32]) -> (usize, Vec<f32>) {
        let code = self.nearest(vector);
        let quantized = self.code(code).to_vec();
        (code, quantized)
    }

    /// Quantize a batch of vectors.
    pub fn quantize_batch(&self, vectors: &[Vec<f32>]) -> Vec<QuantizeResult> {
        vectors.iter().map(|v| self.quantize(v)).collect()
    }

    /// Update codebook using EMA from a batch of encoder outputs.
    ///
    /// EMA update:
    ///   N_i = γ * N_i + (1-γ) * n_i
    ///   m_i = γ * m_i + (1-γ) * Σ x assigned to i
    ///   e_i = m_i / N_i
    pub fn ema_update(&mut self, encoder_outputs: &[Vec<f32>], codes: &[usize]) {
        if encoder_outputs.len() != codes.len() {
            return;
        }

        // Accumulate per-code sums and counts
        let mut batch_counts = vec![0.0f32; self.codebook_size];
        let mut batch_sums = vec![0.0f32; self.codebook_size * self.dim];

        for (vector, &code) in encoder_outputs.iter().zip(codes.iter()) {
            batch_counts[code] += 1.0;
            let offset = code * self.dim;
            for d in 0..self.dim {
                batch_sums[offset + d] += vector[d];
            }
        }

        if !self.ema_initialized {
            // First update: initialize EMA from batch
            for k in 0..self.codebook_size {
                self.ema_counts[k] = batch_counts[k];
                let offset = k * self.dim;
                for d in 0..self.dim {
                    self.ema_sum[offset + d] = batch_sums[offset + d];
                }
            }
            self.ema_initialized = true;
        } else {
            // EMA update
            let gamma = self.ema_decay;
            for k in 0..self.codebook_size {
                self.ema_counts[k] = gamma * self.ema_counts[k] + (1.0 - gamma) * batch_counts[k];
                let offset = k * self.dim;
                for d in 0..self.dim {
                    self.ema_sum[offset + d] = gamma * self.ema_sum[offset + d] + (1.0 - gamma) * batch_sums[offset + d];
                }
            }
        }

        // Update embeddings from EMA averages
        for k in 0..self.codebook_size {
            let count = self.ema_counts[k];
            if count > 1e-5 {
                let offset = k * self.dim;
                for d in 0..self.dim {
                    self.embeddings[offset + d] = self.ema_sum[offset + d] / count;
                }
            }
        }
    }

    /// Reset dead codes (codes with very low usage).
    /// Reinitializes them from random encoder outputs.
    pub fn reset_dead_codes(&mut self, encoder_outputs: &[Vec<f32>], threshold: f32) -> usize {
        let mut reset_count = 0;

        for k in 0..self.codebook_size {
            if self.ema_counts[k] < threshold && !encoder_outputs.is_empty() {
                // Reinitialize from a random encoder output
                let idx = (k * 7 + 3) % encoder_outputs.len();
                let offset = k * self.dim;
                for d in 0..self.dim {
                    self.embeddings[offset + d] = encoder_outputs[idx][d] + (d as f32 * 0.01).sin() * 0.1;
                }
                self.ema_counts[k] = 0.0;
                reset_count += 1;
            }
        }

        reset_count
    }

    /// Get the code vector for a given index.
    pub fn code(&self, idx: usize) -> &[f32] {
        &self.embeddings[idx * self.dim..(idx + 1) * self.dim]
    }

    /// Decode a sequence of codes back to embedding vectors.
    pub fn decode(&self, codes: &[usize]) -> Vec<Vec<f32>> {
        codes.iter().map(|&c| self.code(c).to_vec()).collect()
    }

    /// Codebook size (number of entries).
    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Access the raw embeddings buffer.
    pub fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }

    /// Get the EMA counts (for monitoring codebook usage).
    pub fn usage_counts(&self) -> &[f32] {
        &self.ema_counts
    }

    /// Compute codebook utilization (fraction of codes with usage > threshold).
    pub fn utilization(&self, threshold: f32) -> f32 {
        let used = self.ema_counts.iter().filter(|&&c| c > threshold).count();
        used as f32 / self.codebook_size as f32
    }
}

// ---------------------------------------------------------------------------
// QuantizeResult
// ---------------------------------------------------------------------------

/// Result of quantizing a single vector.
#[derive(Debug)]
pub struct QuantizeResult {
    /// Discrete code index.
    pub code: usize,
    /// Quantized vector (copy of codebook entry).
    pub quantized: Vec<f32>,
    /// Total loss (codebook + commitment).
    pub total_loss: f32,
    /// Codebook loss: ||sg(z_e) - e_k||²
    pub codebook_loss: f32,
    /// Commitment loss: β * ||z_e - sg(e_k)||
    pub commitment_loss: f32,
}

// ---------------------------------------------------------------------------
// MultiCodebookVQ — hierarchical/multi-head VQ
// ---------------------------------------------------------------------------

/// Multiple codebooks applied in parallel or hierarchically.
///
/// In multi-head mode: each codebook quantizes a different slice of the input.
/// In residual mode: each codebook quantizes the residual from the previous.
pub struct MultiCodebookVQ {
    codebooks: Vec<VQCodebook>,
    /// Number of dimensions per codebook (dim / num_codebooks in multi-head).
    sub_dim: usize,
    mode: MultiCodebookMode,
}

/// How multiple codebooks are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiCodebookMode {
    /// Each codebook quantizes a different slice of the input (parallel).
    MultiHead,
    /// Each codebook quantizes the residual from the previous (sequential).
    Residual,
}

impl MultiCodebookVQ {
    /// Create multi-head VQ: input is split into `num_codebooks` slices.
    pub fn new_multi_head(
        num_codebooks: usize,
        total_dim: usize,
        codebook_size: usize,
    ) -> Self {
        let sub_dim = total_dim / num_codebooks;
        let codebooks = (0..num_codebooks)
            .map(|_| VQCodebook::new(codebook_size, sub_dim))
            .collect();

        Self {
            codebooks,
            sub_dim,
            mode: MultiCodebookMode::MultiHead,
        }
    }

    /// Create residual VQ: each codebook quantizes the residual.
    pub fn new_residual(
        num_codebooks: usize,
        dim: usize,
        codebook_size: usize,
    ) -> Self {
        let codebooks = (0..num_codebooks)
            .map(|_| VQCodebook::new(codebook_size, dim))
            .collect();

        Self {
            codebooks,
            sub_dim: dim,
            mode: MultiCodebookMode::Residual,
        }
    }

    /// Quantize a vector across all codebooks.
    /// Returns (codes, quantized, total_loss).
    pub fn quantize(&self, vector: &[f32]) -> (Vec<usize>, Vec<f32>, f32) {
        match self.mode {
            MultiCodebookMode::MultiHead => {
                let mut codes = Vec::with_capacity(self.codebooks.len());
                let mut quantized = vec![0.0f32; vector.len()];
                let mut total_loss = 0.0f32;

                for (i, codebook) in self.codebooks.iter().enumerate() {
                    let start = i * self.sub_dim;
                    let end = start + self.sub_dim;
                    let slice = &vector[start..end];
                    let result = codebook.quantize(slice);
                    codes.push(result.code);
                    quantized[start..end].copy_from_slice(&result.quantized);
                    total_loss += result.total_loss;
                }

                (codes, quantized, total_loss)
            }
            MultiCodebookMode::Residual => {
                let mut residual = vector.to_vec();
                let mut codes = Vec::with_capacity(self.codebooks.len());
                let mut quantized = vec![0.0f32; vector.len()];
                let mut total_loss = 0.0f32;

                for codebook in &self.codebooks {
                    let result = codebook.quantize(&residual);
                    codes.push(result.code);
                    for d in 0..self.sub_dim {
                        quantized[d] += result.quantized[d];
                        residual[d] -= result.quantized[d];
                    }
                    total_loss += result.total_loss;
                }

                (codes, quantized, total_loss)
            }
        }
    }

    /// Decode codes back to a vector.
    pub fn decode(&self, codes: &[usize]) -> Vec<f32> {
        match self.mode {
            MultiCodebookMode::MultiHead => {
                let mut output = vec![0.0f32; self.codebooks.len() * self.sub_dim];
                for (i, (code, codebook)) in codes.iter().zip(self.codebooks.iter()).enumerate() {
                    let start = i * self.sub_dim;
                    output[start..start + self.sub_dim].copy_from_slice(codebook.code(*code));
                }
                output
            }
            MultiCodebookMode::Residual => {
                let mut output = vec![0.0f32; self.sub_dim];
                for (code, codebook) in codes.iter().zip(self.codebooks.iter()) {
                    let decoded = codebook.code(*code);
                    for d in 0..self.sub_dim {
                        output[d] += decoded[d];
                    }
                }
                output
            }
        }
    }

    /// Update all codebooks with EMA.
    pub fn ema_update(&mut self, encoder_outputs: &[Vec<f32>], all_codes: &[Vec<usize>]) {
        match self.mode {
            MultiCodebookMode::MultiHead => {
                for (i, codebook) in self.codebooks.iter_mut().enumerate() {
                    let start = i * self.sub_dim;
                    let end = start + self.sub_dim;
                    let slices: Vec<Vec<f32>> = encoder_outputs.iter()
                        .map(|v| v[start..end].to_vec())
                        .collect();
                    let codes: Vec<usize> = all_codes.iter().map(|c| c[i]).collect();
                    codebook.ema_update(&slices, &codes);
                }
            }
            MultiCodebookMode::Residual => {
                // For residual, each codebook gets the residual at its stage
                let mut residuals: Vec<Vec<f32>> = encoder_outputs.to_vec();
                for (i, codebook) in self.codebooks.iter_mut().enumerate() {
                    let codes: Vec<usize> = all_codes.iter().map(|c| c[i]).collect();
                    codebook.ema_update(&residuals, &codes);
                    // Subtract quantized from residual for next stage
                    for (res, code) in residuals.iter_mut().zip(codes.iter()) {
                        let quantized = codebook.code(*code);
                        for d in 0..self.sub_dim {
                            res[d] -= quantized[d];
                        }
                    }
                }
            }
        }
    }

    /// Number of codebooks.
    pub fn num_codebooks(&self) -> usize {
        self.codebooks.len()
    }

    /// Access individual codebook.
    pub fn codebook(&self, idx: usize) -> &VQCodebook {
        &self.codebooks[idx]
    }

    /// The mode.
    pub fn mode(&self) -> MultiCodebookMode {
        self.mode
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vq_codebook_nearest() {
        let mut cb = VQCodebook::new(4, 2);
        // Set known embeddings
        cb.embeddings[0..2].copy_from_slice(&[1.0, 0.0]);
        cb.embeddings[2..4].copy_from_slice(&[0.0, 1.0]);
        cb.embeddings[4..6].copy_from_slice(&[-1.0, 0.0]);
        cb.embeddings[6..8].copy_from_slice(&[0.0, -1.0]);

        assert_eq!(cb.nearest(&[0.9, 0.1]), 0);
        assert_eq!(cb.nearest(&[0.1, 0.9]), 1);
        assert_eq!(cb.nearest(&[-0.9, 0.1]), 2);
        assert_eq!(cb.nearest(&[0.1, -0.9]), 3);
    }

    #[test]
    fn test_vq_codebook_quantize() {
        let mut cb = VQCodebook::new(4, 2);
        cb.embeddings[0..2].copy_from_slice(&[1.0, 0.0]);
        cb.embeddings[2..4].copy_from_slice(&[0.0, 1.0]);

        let result = cb.quantize(&[0.8, 0.2]);
        assert_eq!(result.code, 0);
        assert_eq!(result.quantized, vec![1.0, 0.0]);
        assert!(result.total_loss > 0.0);
        assert!(result.commitment_loss > 0.0);
    }

    #[test]
    fn test_vq_codebook_ste() {
        let mut cb = VQCodebook::new(4, 2);
        cb.embeddings[0..2].copy_from_slice(&[1.0, 0.0]);

        let (code, ste) = cb.quantize_ste(&[0.9, 0.1]);
        assert_eq!(code, 0);
        assert_eq!(ste, vec![1.0, 0.0]); // Quantized value
    }

    #[test]
    fn test_vq_codebook_batch() {
        let cb = VQCodebook::new(8, 4);
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let results = cb.quantize_batch(&vectors);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.code < 8);
            assert_eq!(r.quantized.len(), 4);
        }
    }

    #[test]
    fn test_vq_codebook_decode() {
        let cb = VQCodebook::new(4, 2);
        let decoded = cb.decode(&[0, 2, 1]);
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].len(), 2);
    }

    #[test]
    fn test_vq_ema_update() {
        let mut cb = VQCodebook::new(4, 2).with_ema_decay(0.99);
        let outputs = vec![
            vec![1.0, 0.5],
            vec![0.8, 0.3],
            vec![0.9, 0.4],
        ];
        let codes: Vec<usize> = outputs.iter().map(|v| cb.nearest(v)).collect();

        cb.ema_update(&outputs, &codes);
        assert!(cb.ema_initialized);
    }

    #[test]
    fn test_vq_reset_dead_codes() {
        let mut cb = VQCodebook::new(4, 2);
        cb.ema_counts[0] = 10.0;
        cb.ema_counts[1] = 0.001; // Dead code
        cb.ema_counts[2] = 0.001; // Dead code
        cb.ema_counts[3] = 5.0;

        let outputs = vec![vec![1.0, 0.0]];
        let reset = cb.reset_dead_codes(&outputs, 0.01);
        assert_eq!(reset, 2);
    }

    #[test]
    fn test_vq_utilization() {
        let mut cb = VQCodebook::new(8, 4);
        // All codes have zero usage initially
        assert!((cb.utilization(1.0) - 0.0).abs() < 0.01);

        cb.ema_counts[0] = 10.0;
        cb.ema_counts[3] = 5.0;
        assert!((cb.utilization(1.0) - 0.25).abs() < 0.01); // 2/8
    }

    #[test]
    fn test_vq_with_commitment_cost() {
        let cb = VQCodebook::new(4, 2)
            .with_commitment_cost(0.5)
            .with_ema_decay(0.95);
        let result = cb.quantize(&[1.0, 0.0]);
        // Commitment loss = 0.5 * encoder_dist
        assert!(result.commitment_loss > 0.0);
    }

    #[test]
    fn test_multi_head_vq() {
        let mvq = MultiCodebookVQ::new_multi_head(2, 8, 16);
        assert_eq!(mvq.num_codebooks(), 2);
        assert_eq!(mvq.mode(), MultiCodebookMode::MultiHead);

        let vector = vec![1.0, 0.5, -0.5, 0.0, 0.3, -0.2, 0.8, -0.1];
        let (codes, quantized, loss) = mvq.quantize(&vector);
        assert_eq!(codes.len(), 2);
        assert_eq!(quantized.len(), 8);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_multi_head_vq_roundtrip() {
        let mvq = MultiCodebookVQ::new_multi_head(2, 4, 32);
        let vector = vec![1.0, 0.5, -0.5, 0.0];
        let (codes, _, _) = mvq.quantize(&vector);
        let decoded = mvq.decode(&codes);
        assert_eq!(decoded.len(), 4);
        // Quantized, not exact
    }

    #[test]
    fn test_residual_vq() {
        let rvq = MultiCodebookVQ::new_residual(3, 4, 16);
        assert_eq!(rvq.num_codebooks(), 3);
        assert_eq!(rvq.mode(), MultiCodebookMode::Residual);

        let vector = vec![1.0, 0.5, -0.5, 0.0];
        let (codes, quantized, loss) = rvq.quantize(&vector);
        assert_eq!(codes.len(), 3);
        assert_eq!(quantized.len(), 4);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_residual_vq_decode() {
        let rvq = MultiCodebookVQ::new_residual(2, 4, 16);
        let vector = vec![0.8, -0.3, 0.5, 0.1];
        let (codes, _, _) = rvq.quantize(&vector);
        let decoded = rvq.decode(&codes);
        assert_eq!(decoded.len(), 4);
    }

    #[test]
    fn test_multi_codebook_ema() {
        let mut mvq = MultiCodebookVQ::new_multi_head(2, 4, 8);
        let vectors = vec![
            vec![1.0, 0.5, -0.5, 0.0],
            vec![0.3, -0.2, 0.8, -0.1],
        ];
        let codes: Vec<Vec<usize>> = vectors.iter()
            .map(|v| { let (c, _, _) = mvq.quantize(v); c })
            .collect();
        mvq.ema_update(&vectors, &codes);
        // Should not panic
    }
}
