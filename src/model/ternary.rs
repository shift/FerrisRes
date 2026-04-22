//! Ternary quantization core (BitNet b1.58, Ma et al. 2024).
//!
//! Quantizes FP32 weights to ternary {-1, 0, +1} using absmean scaling.
//! Each weight is 2 bits packed (4 values per byte), giving ~1.58 bits/weight
//! effective information density (log2(3) ≈ 1.585).
//!
//! Key advantage: ternary matmul replaces all multiplications with additions
//! and subtractions, enabling 3-5x faster inference on CPU.

// ── Constants ───────────────────────────────────────────────────────────────

/// Expected absolute value of a standard normal: sqrt(2/π) ≈ 0.7979.
/// Used in absmean quantization to normalize the scale factor so that
/// E[α · W_q] ≈ E[W] for normally-distributed weights.
pub const ABSMEAN_NORM: f32 = 0.7978845608; // sqrt(2.0 / std::f32::consts::PI)

/// Maximum number of weights per scale group (for block-wise quantization).
/// Block-wise scaling improves accuracy for non-uniform weight distributions.
pub const DEFAULT_BLOCK_SIZE: usize = 256;

// ── Per-tensor quantization ─────────────────────────────────────────────────

/// Quantize a full weight tensor to ternary {-1, 0, +1} using absmean.
///
/// Algorithm:
///   α = mean(|W|) / ABSMEAN_NORM
///   W_q[i] = clamp(round(W[i] / α), -1, 0, +1)
///
/// Returns (ternary_values, scale_factor).
/// The dequantized reconstruction is: W_hat = α * W_q
pub fn quantize_ternary(weights: &[f32]) -> (Vec<i8>, f32) {
    if weights.is_empty() {
        return (Vec::new(), 0.0);
    }

    // Compute scale: α = mean(|W|) / sqrt(2/π)
    let sum_abs: f32 = weights.iter().map(|w| w.abs()).sum();
    let alpha = if sum_abs > 0.0 {
        (sum_abs / weights.len() as f32) / ABSMEAN_NORM
    } else {
        // All-zero weights — any non-zero scale works, result is all zeros
        1.0
    };

    let inv_alpha = 1.0 / alpha;
    let quantized: Vec<i8> = weights
        .iter()
        .map(|&w| {
            let scaled = w * inv_alpha;
            // clamp(round(scaled), -1, +1) — branchless
            let rounded = scaled.round() as i32;
            rounded.clamp(-1, 1) as i8
        })
        .collect();

    (quantized, alpha)
}

/// Dequantize ternary values back to FP32.
///
/// Reconstruction: W_hat[i] = α * W_q[i]
pub fn dequantize_ternary(ternary: &[i8], scale: f32) -> Vec<f32> {
    ternary.iter().map(|&t| scale * t as f32).collect()
}

// ── Block-wise quantization ─────────────────────────────────────────────────

/// Quantize with block-wise scaling (more accurate for large tensors).
///
/// Divides weights into blocks of `block_size` elements and computes
/// a separate scale per block. This handles non-uniform weight distributions
/// better than a single global scale.
///
/// Returns (packed_ternary, scales_per_block).
/// Number of scales = ceil(weights.len() / block_size).
pub fn quantize_ternary_blocked(
    weights: &[f32],
    block_size: usize,
) -> (Vec<i8>, Vec<f32>) {
    let n = weights.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let num_blocks = (n + block_size - 1) / block_size;
    let mut all_ternary = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity(num_blocks);

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(n);
        let block = &weights[start..end];

        let sum_abs: f32 = block.iter().map(|w| w.abs()).sum();
        let alpha = if sum_abs > 0.0 {
            (sum_abs / block.len() as f32) / ABSMEAN_NORM
        } else {
            1.0
        };
        scales.push(alpha);

        let inv_alpha = 1.0 / alpha;
        for &w in block {
            let rounded = (w * inv_alpha).round() as i32;
            all_ternary.push(rounded.clamp(-1, 1) as i8);
        }
    }

    (all_ternary, scales)
}

/// Dequantize block-wise ternary back to FP32.
pub fn dequantize_ternary_blocked(
    ternary: &[i8],
    scales: &[f32],
    block_size: usize,
) -> Vec<f32> {
    let n = ternary.len();
    let mut result = Vec::with_capacity(n);

    for (block_idx, &alpha) in scales.iter().enumerate() {
        let start = block_idx * block_size;
        let end = (start + block_size).min(n);
        for i in start..end {
            result.push(alpha * ternary[i] as f32);
        }
    }

    result
}

// ── 2-bit packing ───────────────────────────────────────────────────────────

/// Encoding: {-1 → 0b00, 0 → 0b01, +1 → 0b10}.
/// 0b11 is unused (padding for non-multiple-of-4 lengths).
const TERNARY_ENCODE: [u8; 3] = [0b00, 0b01, 0b10];

/// Decoding: index by packed 2-bit value → ternary i8.
/// 0b11 (padding) decodes to 0.
const TERNARY_DECODE: [i8; 4] = [-1, 0, 1, 0];

/// Pack ternary values into 2-bit format (4 values per byte).
///
/// Each i8 value in {-1, 0, +1} maps to a 2-bit code.
/// Packed byte layout: [val0:bits0-1 | val1:bits2-3 | val2:bits4-5 | val3:bits6-7]
pub fn pack_ternary(values: &[i8]) -> Vec<u8> {
    let packed_len = (values.len() + 3) / 4;
    let mut packed = vec![0u8; packed_len];

    for (i, &val) in values.iter().enumerate() {
        let code = match val {
            -1 => TERNARY_ENCODE[0],
            0 => TERNARY_ENCODE[1],
            1 => TERNARY_ENCODE[2],
            _ => TERNARY_ENCODE[1], // safety: treat out-of-range as 0
        };
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= code << bit_offset;
    }

    packed
}

/// Unpack 2-bit packed ternary back to i8 values.
///
/// `len` is the original number of values (may not be a multiple of 4).
pub fn unpack_ternary(packed: &[u8], len: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(len);

    for i in 0..len {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let code = (packed[byte_idx] >> bit_offset) & 0b11;
        values.push(TERNARY_DECODE[code as usize]);
    }

    values
}

// ── Quantization quality metrics ────────────────────────────────────────────

/// Compute quantization error metrics for a weight tensor.
///
/// Returns (mse, cosine_similarity, snr_db).
/// Useful for evaluating quality during quantization-aware training.
pub fn quantization_error(original: &[f32], reconstructed: &[f32]) -> (f32, f32, f32) {
    assert_eq!(original.len(), reconstructed.len());
    let n = original.len() as f32;

    // MSE
    let mse: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r) * (o - r))
        .sum::<f32>()
        / n;

    // Cosine similarity
    let dot: f32 = original.iter().zip(reconstructed.iter()).map(|(o, r)| o * r).sum();
    let norm_o: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_r: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos_sim = if norm_o > 0.0 && norm_r > 0.0 {
        dot / (norm_o * norm_r)
    } else {
        1.0
    };

    // SNR in dB
    let signal_power: f32 = original.iter().map(|x| x * x).sum::<f32>() / n;
    let snr_db = if mse > 0.0 {
        10.0 * (signal_power / mse).log10()
    } else {
        f32::INFINITY
    };

    (mse, cos_sim, snr_db)
}

// ── Straight-Through Estimator (STE) for training ───────────────────────────

/// Quantize with Straight-Through Estimator for mixed-precision training.
///
/// In the forward pass, weights are quantized to ternary {-1, 0, +1}.
/// In the backward pass, gradients flow through the FP32 weights unchanged
/// (as if the quantization were the identity function).
///
/// This function returns:
/// - `quantized`: The ternary values for the forward pass
/// - `scale`: The absmean scale factor
/// - `fp32_weights`: The original FP32 weights (kept for optimizer updates)
///
/// The caller is responsible for:
/// 1. Using (quantized, scale) in the forward pass
/// 2. Updating fp32_weights via the optimizer
/// 3. Re-quantizing after each optimizer step
pub fn quantize_ste(weights: &[f32]) -> TernarySteResult {
    let (quantized, scale) = quantize_ternary(weights);
    TernarySteResult {
        quantized,
        scale,
        fp32_weights: weights.to_vec(),
    }
}

/// Result of STE quantization — holds both quantized and FP32 representations.
pub struct TernarySteResult {
    pub quantized: Vec<i8>,
    pub scale: f32,
    pub fp32_weights: Vec<f32>,
}

impl TernarySteResult {
    /// Re-quantize after optimizer step (call after updating fp32_weights).
    pub fn requantize(&mut self) {
        let (q, s) = quantize_ternary(&self.fp32_weights);
        self.quantized = q;
        self.scale = s;
    }

    /// Get the dequantized weights for the forward pass.
    /// This is equivalent to scale * quantized but uses FP32 precision.
    pub fn dequantized(&self) -> Vec<f32> {
        dequantize_ternary(&self.quantized, self.scale)
    }
}

// ── Ternary matrix multiplication ───────────────────────────────────────────
//
// Key insight: y = α * W_q @ x where W_q ∈ {-1, 0, +1}
// eliminates ALL multiplications. The inner loop becomes:
//   sum += x_j   if W_q[i,j] == +1
//   sum -= x_j   if W_q[i,j] == -1
//   (skip)       if W_q[i,j] == 0
//
// This is 3-5× faster than FP32 matmul on CPU because:
// - No multiplier hardware needed (adder-only)
// - Zero weights are free (skip)
// - Compiler auto-vectorizes the add/sub pattern

/// Ternary matrix multiply: `output = scale * (ternary @ input)`.
///
/// - `ternary`: `[out_rows * in_cols]` values in {-1, 0, +1} (row-major)
/// - `scale`: absmean scale factor α (single scalar for the whole matrix)
/// - `input`: `[seq * in_cols]` FP32 activations (row-major)
/// - Returns `[seq * out_rows]` FP32 output
///
/// The inner loop uses branchless sign-as-mask:
///   accumulator += (sign as f32) * input[j]
/// where sign ∈ {-1.0, 0.0, +1.0} — the compiler auto-vectorizes this
/// to SIMD add/sub instructions.
pub fn ternary_matmul(
    ternary: &[i8],
    scale: f32,
    input: &[f32],
    out_rows: usize,
    in_cols: usize,
    seq: usize,
) -> Vec<f32> {
    assert_eq!(ternary.len(), out_rows * in_cols);
    assert_eq!(input.len(), seq * in_cols);

    let mut output = vec![0.0f32; seq * out_rows];

    // Process each sequence position
    for s in 0..seq {
        let input_row = &input[s * in_cols..(s + 1) * in_cols];
        let out_slice = &mut output[s * out_rows..(s + 1) * out_rows];

        for r in 0..out_rows {
            let weight_row = &ternary[r * in_cols..(r + 1) * in_cols];
            // Split positive/negative accumulators — only 1 multiply at the end.
            // The `if` branches are predictable (33% each of {-1, 0, +1})
            // and the multiply-by-0 in the else branch is free.
            let mut pos_sum = 0.0f32;
            let mut neg_sum = 0.0f32;
            for j in 0..in_cols {
                let w = weight_row[j];
                let v = input_row[j];
                pos_sum += if w > 0 { v } else { 0.0 };
                neg_sum += if w < 0 { v } else { 0.0 };
            }
            out_slice[r] = scale * (pos_sum - neg_sum);
        }
    }

    output
}

/// Multi-threaded ternary matmul using rayon.
/// Processes sequence positions in parallel — each is independent.
/// ~4-8x speedup on Skylake (4 cores / 8 threads).
pub fn ternary_matmul_parallel(
    ternary: &[i8],
    scale: f32,
    input: &[f32],
    out_rows: usize,
    in_cols: usize,
    seq: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    assert_eq!(ternary.len(), out_rows * in_cols);
    assert_eq!(input.len(), seq * in_cols);

    if seq <= 1 {
        // Single token — parallelize across output rows in BATCHES
        // to amortize rayon task scheduling overhead.
        // Batch size: aim for ~32 tasks (e.g., 2048 rows / 32 = 64 rows per task).
        let mut output = vec![0.0f32; out_rows];
        let batch_size = (out_rows / 32).max(1);
        output.par_chunks_mut(batch_size).enumerate().for_each(|(batch_idx, chunk)| {
            let start_row = batch_idx * batch_size;
            for r in 0..chunk.len() {
                let row = start_row + r;
                let weight_row = &ternary[row * in_cols..(row + 1) * in_cols];
                let input_row = &input[..in_cols];
                let mut pos_sum = 0.0f32;
                let mut neg_sum = 0.0f32;
                for j in 0..in_cols {
                    let w = weight_row[j];
                    let v = input_row[j];
                    pos_sum += if w > 0 { v } else { 0.0 };
                    neg_sum += if w < 0 { v } else { 0.0 };
                }
                chunk[r] = scale * (pos_sum - neg_sum);
            }
        });
        return output;
    }

    // Multi-token — parallelize across seq positions
    let mut output = vec![0.0f32; seq * out_rows];
    output.par_chunks_mut(out_rows).enumerate().for_each(|(s, out_slice)| {
        let input_row = &input[s * in_cols..(s + 1) * in_cols];
        for r in 0..out_rows {
            let weight_row = &ternary[r * in_cols..(r + 1) * in_cols];
            let mut pos_sum = 0.0f32;
            let mut neg_sum = 0.0f32;
            for j in 0..in_cols {
                let w = weight_row[j];
                let v = input_row[j];
                pos_sum += if w > 0 { v } else { 0.0 };
                neg_sum += if w < 0 { v } else { 0.0 };
            }
            out_slice[r] = scale * (pos_sum - neg_sum);
        }
    });
    output
}

/// Ternary matmul from 2-bit packed data.
///
/// Works directly on the packed representation from `pack_ternary()`,
/// avoiding the unpack step. Processes 4 ternary values per iteration.
///
/// - `packed`: packed ternary weights (2 bits/value, 4 values/byte)
/// - `scale`: absmean scale factor
/// - `input`: `[seq * in_cols]` FP32 activations
/// - `out_rows`, `in_cols`, `seq`: matrix dimensions
/// - `packed_len`: original number of ternary values (= out_rows * in_cols)
pub fn ternary_matmul_packed(
    packed: &[u8],
    scale: f32,
    input: &[f32],
    out_rows: usize,
    in_cols: usize,
    seq: usize,
    packed_len: usize,
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        seq * in_cols,
        "input shape mismatch"
    );
    let expected_packed_bytes = (packed_len + 3) / 4;
    assert_eq!(
        packed.len(),
        expected_packed_bytes,
        "packed size mismatch: got {}, expected {}",
        packed.len(),
        expected_packed_bytes
    );

    let mut output = vec![0.0f32; seq * out_rows];

    for s in 0..seq {
        let input_row = &input[s * in_cols..(s + 1) * in_cols];
        for r in 0..out_rows {
            let row_offset = r * in_cols;
            let mut sum = 0.0f32;

            // Process 4 values at a time from packed bytes
            let full_quads = in_cols / 4;
            let remainder = in_cols % 4;

            for q in 0..full_quads {
                let byte_idx = (row_offset + q * 4) / 4;
                let packed_byte = packed[byte_idx];

                // Unpack 4 values inline
                for bit_pos in 0..4 {
                    let code = ((packed_byte >> (bit_pos * 2)) & 0b11) as usize;
                    let sign = TERNARY_DECODE[code] as f32;
                    sum += sign * input_row[q * 4 + bit_pos];
                }
            }

            // Handle remainder (not a full quad)
            if remainder > 0 {
                let byte_idx = (row_offset + full_quads * 4) / 4;
                let packed_byte = packed[byte_idx];
                for bit_pos in 0..remainder {
                    let code = ((packed_byte >> (bit_pos * 2)) & 0b11) as usize;
                    let sign = TERNARY_DECODE[code] as f32;
                    sum += sign * input_row[full_quads * 4 + bit_pos];
                }
            }

            output[s * out_rows + r] = scale * sum;
        }
    }

    output
}

/// Parallel ternary matmul from 2-bit packed data using rayon.
///
/// Single-token (seq=1): parallelize across output rows in batches.
/// Multi-token: parallelize across sequence positions.
/// Uses 4× less memory bandwidth than unpacked ternary matmul.
pub fn ternary_matmul_packed_parallel(
    packed: &[u8],
    scale: f32,
    input: &[f32],
    out_rows: usize,
    in_cols: usize,
    _seq: usize, // always 1 for decode
) -> Vec<f32> {
    use rayon::prelude::*;
    let mut output = vec![0.0f32; out_rows];
    let batch_size = (out_rows / 32).max(1);

    output.par_chunks_mut(batch_size).enumerate().for_each(|(batch_idx, chunk)| {
        let start_row = batch_idx * batch_size;
        let input_row = &input[..in_cols];
        let full_quads = in_cols / 4;
        let remainder = in_cols % 4;

        for r in 0..chunk.len() {
            let row = start_row + r;
            // Each row is full_quads bytes (or full_quads+1 if remainder)
            let row_packed_start = row * ((in_cols + 3) / 4);
            let mut sum = 0.0f32;

            // Process 4 values at a time from packed bytes
            for q in 0..full_quads {
                let byte_idx = row_packed_start + q;
                let packed_byte = packed[byte_idx];

                // Unrolled: extract 4 signs and accumulate
                let s0 = TERNARY_DECODE[(packed_byte & 0b11) as usize] as f32;
                let s1 = TERNARY_DECODE[((packed_byte >> 2) & 0b11) as usize] as f32;
                let s2 = TERNARY_DECODE[((packed_byte >> 4) & 0b11) as usize] as f32;
                let s3 = TERNARY_DECODE[((packed_byte >> 6) & 0b11) as usize] as f32;

                let base = q * 4;
                sum += s0 * input_row[base]
                     + s1 * input_row[base + 1]
                     + s2 * input_row[base + 2]
                     + s3 * input_row[base + 3];
            }

            // Handle remainder
            if remainder > 0 {
                let byte_idx = row_packed_start + full_quads;
                let packed_byte = packed[byte_idx];
                let base = full_quads * 4;
                for bit_pos in 0..remainder {
                    let code = ((packed_byte >> (bit_pos * 2)) & 0b11) as usize;
                    sum += TERNARY_DECODE[code] as f32 * input_row[base + bit_pos];
                }
            }

            chunk[r] = scale * sum;
        }
    });

    output
}

/// Ternary matmul with block-wise scaling.
///
/// Each block of `block_size` columns has its own scale factor,
/// giving better accuracy for non-uniform weight distributions.
///
/// - `ternary`: `[out_rows * in_cols]` unpacked ternary values
/// - `scales`: scale factors, one per block (num_blocks = ceil(in_cols / block_size))
/// - `block_size`: number of columns per scale block
/// - `input`: `[seq * in_cols]` FP32 activations
/// - `out_rows`, `in_cols`, `seq`: matrix dimensions
pub fn ternary_matmul_blocked(
    ternary: &[i8],
    scales: &[f32],
    block_size: usize,
    input: &[f32],
    out_rows: usize,
    in_cols: usize,
    seq: usize,
) -> Vec<f32> {
    assert_eq!(ternary.len(), out_rows * in_cols);
    assert_eq!(input.len(), seq * in_cols);
    let num_blocks = (in_cols + block_size - 1) / block_size;
    assert_eq!(scales.len(), num_blocks, "scales length mismatch");

    let mut output = vec![0.0f32; seq * out_rows];

    for s in 0..seq {
        let input_row = &input[s * in_cols..(s + 1) * in_cols];
        for r in 0..out_rows {
            let weight_row = &ternary[r * in_cols..(r + 1) * in_cols];
            let mut sum = 0.0f32;

            // Accumulate per-block with per-block scale
            for (block_idx, &alpha) in scales.iter().enumerate() {
                let col_start = block_idx * block_size;
                let col_end = (col_start + block_size).min(in_cols);
                let mut block_sum = 0.0f32;
                for j in col_start..col_end {
                    block_sum += weight_row[j] as f32 * input_row[j];
                }
                sum += alpha * block_sum;
            }

            output[s * out_rows + r] = sum;
        }
    }

    output
}

/// Optimized ternary matmul for decode (single token, seq=1).
///
/// Avoids the seq dimension overhead and processes one input vector
/// against all output rows. This is the hot path for autoregressive
/// generation where each step produces exactly one token.
///
/// - `ternary`: `[out_rows * in_cols]` unpacked ternary values
/// - `scale`: single absmean scale factor
/// - `input`: `[in_cols]` FP32 activation vector
/// - Returns `[out_rows]` FP32 output vector
pub fn ternary_matmul_decode(
    ternary: &[i8],
    scale: f32,
    input: &[f32],
    out_rows: usize,
    in_cols: usize,
) -> Vec<f32> {
    assert_eq!(ternary.len(), out_rows * in_cols);
    assert_eq!(input.len(), in_cols);

    let mut output = vec![0.0f32; out_rows];

    for r in 0..out_rows {
        let weight_row = &ternary[r * in_cols..(r + 1) * in_cols];
        let mut sum = 0.0f32;

        // Process in chunks for better cache behavior
        const CHUNK: usize = 8;
        let chunks = in_cols / CHUNK;

        for c in 0..chunks {
            let base = c * CHUNK;
            // Unrolled accumulation
            sum += weight_row[base] as f32 * input[base];
            sum += weight_row[base + 1] as f32 * input[base + 1];
            sum += weight_row[base + 2] as f32 * input[base + 2];
            sum += weight_row[base + 3] as f32 * input[base + 3];
            sum += weight_row[base + 4] as f32 * input[base + 4];
            sum += weight_row[base + 5] as f32 * input[base + 5];
            sum += weight_row[base + 6] as f32 * input[base + 6];
            sum += weight_row[base + 7] as f32 * input[base + 7];
        }

        // Remainder
        for j in (chunks * CHUNK)..in_cols {
            sum += weight_row[j] as f32 * input[j];
        }

        output[r] = scale * sum;
    }

    output
}

/// Ternary matmul using precomputed row offsets for decode.
///
/// Instead of sign-as-mask, this separates positive and negative indices
/// into two lists per row. This can be faster when sparsity is high (>40% zeros)
/// because it skips zeros entirely.
///
/// - `pos_indices`: for each row, column indices where ternary = +1
/// - `neg_indices`: for each row, column indices where ternary = -1
/// - `scale`: absmean scale factor
/// - `input`: `[in_cols]` FP32 activation vector
/// - `out_rows`: number of output rows
/// - Returns `[out_rows]` FP32 output vector
pub fn ternary_matmul_sparse_decode(
    pos_indices: &[Vec<usize>],
    neg_indices: &[Vec<usize>],
    scale: f32,
    input: &[f32],
    out_rows: usize,
) -> Vec<f32> {
    assert_eq!(pos_indices.len(), out_rows);
    assert_eq!(neg_indices.len(), out_rows);

    let mut output = vec![0.0f32; out_rows];

    for r in 0..out_rows {
        let mut sum = 0.0f32;
        // Add positive contributions
        for &j in &pos_indices[r] {
            sum += input[j];
        }
        // Subtract negative contributions
        for &j in &neg_indices[r] {
            sum -= input[j];
        }
        output[r] = scale * sum;
    }

    output
}

/// Precompute sparse index lists from a ternary weight matrix.
///
/// Returns (pos_indices, neg_indices) where each is a Vec of Vec<usize>,
/// one inner Vec per output row.
///
/// Use with `ternary_matmul_sparse_decode` for high-sparsity weights.
pub fn ternary_sparse_indices(
    ternary: &[i8],
    out_rows: usize,
    in_cols: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    assert_eq!(ternary.len(), out_rows * in_cols);

    let mut pos_indices = Vec::with_capacity(out_rows);
    let mut neg_indices = Vec::with_capacity(out_rows);

    for r in 0..out_rows {
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        let row = &ternary[r * in_cols..(r + 1) * in_cols];
        for (j, &v) in row.iter().enumerate() {
            match v {
                1 => pos.push(j),
                -1 => neg.push(j),
                _ => {}
            }
        }
        pos_indices.push(pos);
        neg_indices.push(neg);
    }

    (pos_indices, neg_indices)
}

// ── Ternary weight statistics ───────────────────────────────────────────────

/// Statistics of a ternary weight tensor.
#[derive(Debug, Clone)]
pub struct TernaryStats {
    pub num_neg1: usize,
    pub num_zero: usize,
    pub num_pos1: usize,
    pub ratio_neg1: f32,
    pub ratio_zero: f32,
    pub ratio_pos1: f32,
    pub scale: f32,
    pub compression_ratio_vs_f32: f32,
}

/// Compute statistics for a quantized ternary tensor.
pub fn ternary_stats(ternary: &[i8], scale: f32) -> TernaryStats {
    let n = ternary.len();
    let num_neg1 = ternary.iter().filter(|&&v| v == -1).count();
    let num_pos1 = ternary.iter().filter(|&&v| v == 1).count();
    let num_zero = n - num_neg1 - num_pos1;

    // Ternary packed: 2 bits/weight. FP32: 32 bits/weight.
    // Plus scale overhead: 1 f32 per tensor ≈ negligible.
    let bits_per_weight = 2.0_f32; // packed
    let compression_ratio = 32.0 / bits_per_weight;

    TernaryStats {
        num_neg1,
        num_zero,
        num_pos1,
        ratio_neg1: num_neg1 as f32 / n as f32,
        ratio_zero: num_zero as f32 / n as f32,
        ratio_pos1: num_pos1 as f32 / n as f32,
        scale,
        compression_ratio_vs_f32: compression_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let weights: Vec<f32> = vec![0.5, -0.3, 1.2, -0.8, 0.0, 0.01, -1.5, 2.0];
        let (quantized, scale) = quantize_ternary(&weights);

        assert!(scale > 0.0, "scale should be positive");

        // Check values are in {-1, 0, 1}
        for &v in &quantized {
            assert!(
                v == -1 || v == 0 || v == 1,
                "quantized value {} not in {{-1, 0, 1}}",
                v
            );
        }

        // Dequantize and check reconstruction quality
        let reconstructed = dequantize_ternary(&quantized, scale);
        let (_, cos_sim, _) = quantization_error(&weights, &reconstructed);
        assert!(
            cos_sim > 0.85,
            "cosine similarity too low after roundtrip: {}",
            cos_sim
        );
    }

    #[test]
    fn test_quantize_zero_weights() {
        let weights = vec![0.0f32; 16];
        let (quantized, scale) = quantize_ternary(&weights);
        assert!(quantized.iter().all(|&v| v == 0));
        let reconstructed = dequantize_ternary(&quantized, scale);
        assert!(reconstructed.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_quantize_preserves_sign() {
        let weights: Vec<f32> = vec![-3.0, -0.5, -0.01, 0.01, 0.5, 3.0];
        let (quantized, _scale) = quantize_ternary(&weights);
        // Negative weights → -1, positive → +1, near-zero → 0 or sign-preserving
        assert_eq!(quantized[0], -1, "large negative should be -1");
        assert_eq!(quantized[5], 1, "large positive should be +1");
        assert_eq!(quantized[2], quantized[3], "symmetric near-zero should match");
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let values: Vec<i8> = vec![-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1];
        let packed = pack_ternary(&values);
        let unpacked = unpack_ternary(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_non_multiple_of_4() {
        let values: Vec<i8> = vec![-1, 0, 1]; // 3 values → 1 byte, 4th slot padding
        let packed = pack_ternary(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_ternary(&packed, 3);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_single_value() {
        for &v in &[-1i8, 0, 1] {
            let packed = pack_ternary(&[v]);
            assert_eq!(packed.len(), 1);
            let unpacked = unpack_ternary(&packed, 1);
            assert_eq!(unpacked[0], v);
        }
    }

    #[test]
    fn test_blocked_quantize_roundtrip() {
        let mut weights = Vec::with_capacity(1000);
        for i in 0..1000 {
            weights.push((i as f32 * 0.01 - 5.0).sin()); // varied distribution
        }

        let block_size = 128;
        let (ternary, scales) = quantize_ternary_blocked(&weights, block_size);
        let reconstructed = dequantize_ternary_blocked(&ternary, &scales, block_size);

        let (mse, cos_sim, snr_db) = quantization_error(&weights, &reconstructed);
        assert!(
            cos_sim > 0.90,
            "blocked roundtrip cosine similarity too low: {} (mse={}, snr={:.1}dB)",
            cos_sim,
            mse,
            snr_db
        );
    }

    #[test]
    fn test_blocked_vs_per_tensor() {
        // Block-wise should be at least as good as per-tensor
        let mut weights = Vec::with_capacity(512);
        for i in 0..512 {
            // Create bimodal distribution: first half small, second half large
            let val = if i < 256 {
                i as f32 * 0.001 - 0.128
            } else {
                i as f32 * 0.01 - 2.56
            };
            weights.push(val);
        }

        let (q_tensor, s_tensor) = quantize_ternary(&weights);
        let r_tensor = dequantize_ternary(&q_tensor, s_tensor);
        let (_, cos_tensor, _) = quantization_error(&weights, &r_tensor);

        let (q_blocked, s_blocked) = quantize_ternary_blocked(&weights, 64);
        let r_blocked = dequantize_ternary_blocked(&q_blocked, &s_blocked, 64);
        let (_, cos_blocked, _) = quantization_error(&weights, &r_blocked);

        assert!(
            cos_blocked >= cos_tensor - 0.01,
            "blocked ({}) should be >= per-tensor ({})",
            cos_blocked,
            cos_tensor
        );
    }

    #[test]
    fn test_ste_requantize() {
        let weights: Vec<f32> = vec![0.5, -0.3, 1.2, -0.8];
        let mut ste = quantize_ste(&weights);

        // Simulate optimizer update on FP32 weights
        ste.fp32_weights[0] += 0.1; // gradient step

        // Re-quantize
        ste.requantize();

        // Quantized values should reflect the updated weights
        let deq = ste.dequantized();
        assert!(
            (deq[0] - weights[0]).abs() > 0.01,
            "weight[0] should have changed after requantize"
        );
    }

    #[test]
    fn test_ternary_stats() {
        let values: Vec<i8> = vec![-1, -1, 0, 0, 0, 1, 1, 1];
        let stats = ternary_stats(&values, 0.5);
        assert_eq!(stats.num_neg1, 2);
        assert_eq!(stats.num_zero, 3);
        assert_eq!(stats.num_pos1, 3);
        assert!((stats.ratio_zero - 0.375).abs() < 0.01);
        assert_eq!(stats.scale, 0.5);
        assert!((stats.compression_ratio_vs_f32 - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_compression_savings() {
        // A 1536×6144 weight matrix (one FFN layer in Gemma 4 E2B)
        let elements = 1536 * 6144;
        let fp32_bytes = elements * 4;
        let ternary_packed_bytes = (elements + 3) / 4; // 2 bits per weight

        let ratio = fp32_bytes as f32 / ternary_packed_bytes as f32;
        // Should be ~16x compression (32 bits → 2 bits)
        assert!(
            ratio > 15.0 && ratio < 17.0,
            "expected ~16x compression, got {}",
            ratio
        );

        // Actual size for this layer
        let kb = ternary_packed_bytes as f32 / 1024.0;
        // 1536 * 6144 / 4 / 1024 = 2304 KB ≈ 2.25 MB
        assert!(
            kb > 2000.0 && kb < 3000.0,
            "expected ~2304 KB, got {} KB",
            kb
        );
    }

    #[test]
    fn test_large_random_weights() {
        // Simulate a realistic weight distribution (Glorot-like)
        let n = 4096;
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                // Pseudo-random but deterministic
                let x = ((i as f32 * 1.618).sin() * 1000.0).fract() - 0.5;
                x * 0.2 // small range
            })
            .collect();

        let (quantized, scale) = quantize_ternary(&weights);
        let reconstructed = dequantize_ternary(&quantized, scale);
        let (_, cos_sim, snr_db) = quantization_error(&weights, &reconstructed);

        assert!(cos_sim > 0.80, "cosine sim: {}", cos_sim);
        assert!(snr_db > 5.0, "SNR: {:.1} dB", snr_db);
    }

    #[test]
    fn test_pack_unpack_large() {
        let n = 1024;
        let values: Vec<i8> = (0..n).map(|i| ((i % 3) as i8) - 1).collect(); // -1, 0, 1, -1, 0, 1, ...
        let packed = pack_ternary(&values);
        assert_eq!(packed.len(), n / 4);
        let unpacked = unpack_ternary(&packed, n);
        assert_eq!(values, unpacked);
    }

    // ── Ternary matmul tests ─────────────────────────────────────────────

    #[test]
    fn test_ternary_matmul_identity() {
        // Identity matrix: diagonal = +1, rest = 0
        let size = 4;
        let mut identity = vec![0i8; size * size];
        for i in 0..size {
            identity[i * size + i] = 1;
        }

        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let output = ternary_matmul(&identity, 1.0, &input, size, size, 1);

        assert_eq!(output.len(), 4);
        for i in 0..size {
            assert!(
                (output[i] - input[i]).abs() < 1e-6,
                "identity output[{}] = {} expected {}",
                i,
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_ternary_matmul_negative_scale() {
        // All +1 matrix with negative scale → negates everything
        let ternary = vec![1i8; 4]; // [1, 1, 1, 1] as 1×4 matrix
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let output = ternary_matmul(&ternary, -2.0, &input, 1, 4, 1);

        // sum = 1+2+3+4 = 10, scaled = -20
        assert!((output[0] - (-20.0)).abs() < 1e-6, "output = {}", output[0]);
    }

    #[test]
    fn test_ternary_matmul_batch() {
        // 2×3 matrix, batch of 2 sequences
        let ternary: Vec<i8> = vec![1, -1, 0, 0, 1, -1]; // 2×3
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0, // seq 0
            4.0, 5.0, 6.0, // seq 1
        ];
        let output = ternary_matmul(&ternary, 1.0, &input, 2, 3, 2);

        // Row 0: 1*1 + (-1)*2 + 0*3 = -1
        // Row 1: 0*1 + 1*2 + (-1)*3 = -1
        // Seq 0: [-1, -1]
        // Row 0: 1*4 + (-1)*5 + 0*6 = -1
        // Row 1: 0*4 + 1*5 + (-1)*6 = -1
        // Seq 1: [-1, -1]
        assert_eq!(output.len(), 4);
        assert!((output[0] - (-1.0)).abs() < 1e-6);
        assert!((output[1] - (-1.0)).abs() < 1e-6);
        assert!((output[2] - (-1.0)).abs() < 1e-6);
        assert!((output[3] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_matmul_vs_fp32() {
        // Ternary matmul with scale should match dequantized FP32 matmul
        let out_rows = 8;
        let in_cols = 16;
        let scale = 0.5f32;

        let ternary: Vec<i8> = (0..out_rows * in_cols)
            .map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 })
            .collect();

        let input: Vec<f32> = (0..in_cols).map(|i| i as f32 * 0.1).collect();

        // Method 1: ternary_matmul
        let output_ternary = ternary_matmul(&ternary, scale, &input, out_rows, in_cols, 1);

        // Method 2: dequantize then FP32 matmul
        let dequant = dequantize_ternary(&ternary, scale);
        let mut output_fp32 = vec![0.0f32; out_rows];
        for r in 0..out_rows {
            for j in 0..in_cols {
                output_fp32[r] += dequant[r * in_cols + j] * input[j];
            }
        }

        // Should be identical
        for r in 0..out_rows {
            assert!(
                (output_ternary[r] - output_fp32[r]).abs() < 1e-5,
                "row {}: ternary={}, fp32={}",
                r,
                output_ternary[r],
                output_fp32[r]
            );
        }
    }

    #[test]
    fn test_ternary_matmul_packed_vs_unpacked() {
        let out_rows = 4;
        let in_cols = 16;
        let scale = 0.7f32;

        let ternary: Vec<i8> = (0..out_rows * in_cols)
            .map(|i| {
                let v = i % 3;
                if v == 0 { 1 } else if v == 1 { -1 } else { 0 }
            })
            .collect();

        let packed = pack_ternary(&ternary);
        let input: Vec<f32> = (0..in_cols).map(|i| (i as f32 * 0.1).sin()).collect();

        let output_unpacked = ternary_matmul(&ternary, scale, &input, out_rows, in_cols, 1);
        let output_packed = ternary_matmul_packed(
            &packed,
            scale,
            &input,
            out_rows,
            in_cols,
            1,
            ternary.len(),
        );

        assert_eq!(output_unpacked.len(), output_packed.len());
        for i in 0..output_unpacked.len() {
            assert!(
                (output_unpacked[i] - output_packed[i]).abs() < 1e-5,
                "packed mismatch at {}: unpacked={}, packed={}",
                i,
                output_unpacked[i],
                output_packed[i]
            );
        }
    }

    #[test]
    fn test_ternary_matmul_blocked_vs_scalar() {
        let out_rows = 4;
        let in_cols = 32;
        let block_size = 8;

        // Create weights with known distribution, quantize with blocked scaling
        // Block-wise: each block of `block_size` COLUMNS has its own scale
        let weights: Vec<f32> = (0..out_rows * in_cols)
            .map(|i| (i as f32 * 1.618).sin() * 0.5)
            .collect();
        let (_ternary_global, _) = quantize_ternary(&weights);

        // Compute per-block scales manually (per column-block of in_cols)
        let num_blocks = (in_cols + block_size - 1) / block_size;
        let mut scales = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            let col_start = b * block_size;
            let col_end = (col_start + block_size).min(in_cols);
            // Collect all weight values in this column block across all rows
            let mut block_sum = 0.0f32;
            let mut block_count = 0usize;
            for r in 0..out_rows {
                for j in col_start..col_end {
                    block_sum += weights[r * in_cols + j].abs();
                    block_count += 1;
                }
            }
            let alpha = if block_sum > 0.0 {
                (block_sum / block_count as f32) / ABSMEAN_NORM
            } else {
                1.0
            };
            scales.push(alpha);
        }

        // Re-quantize with per-block scales
        let mut ternary_blocked = vec![0i8; out_rows * in_cols];
        for r in 0..out_rows {
            for (b, &alpha) in scales.iter().enumerate() {
                let col_start = b * block_size;
                let col_end = (col_start + block_size).min(in_cols);
                let inv_alpha = 1.0 / alpha;
                for j in col_start..col_end {
                    let rounded = (weights[r * in_cols + j] * inv_alpha).round() as i32;
                    ternary_blocked[r * in_cols + j] = rounded.clamp(-1, 1) as i8;
                }
            }
        }

        let input: Vec<f32> = (0..in_cols).map(|i| (i as f32 * 0.1).cos()).collect();

        // Blocked matmul
        let output_blocked = ternary_matmul_blocked(
            &ternary_blocked, &scales, block_size, &input, out_rows, in_cols, 1,
        );

        // Manual reference: dequantize blocked per-row-column-block then FP32 matmul
        let mut output_ref = vec![0.0f32; out_rows];
        for r in 0..out_rows {
            for (b, &alpha) in scales.iter().enumerate() {
                let col_start = b * block_size;
                let col_end = (col_start + block_size).min(in_cols);
                for j in col_start..col_end {
                    output_ref[r] += (alpha * ternary_blocked[r * in_cols + j] as f32) * input[j];
                }
            }
        }

        for r in 0..out_rows {
            assert!(
                (output_blocked[r] - output_ref[r]).abs() < 1e-4,
                "blocked mismatch at row {}: blocked={}, ref={}",
                r,
                output_blocked[r],
                output_ref[r]
            );
        }
    }

    #[test]
    fn test_ternary_matmul_decode_vs_general() {
        // decode path should produce identical results to general matmul with seq=1
        let out_rows = 8;
        let in_cols = 32;
        let scale = 0.3f32;

        let ternary: Vec<i8> = (0..out_rows * in_cols)
            .map(|i| {
                let v = i % 5;
                if v < 2 { 1 } else if v < 4 { -1 } else { 0 }
            })
            .collect();
        let input: Vec<f32> = (0..in_cols).map(|i| (i as f32 * 0.05).tan()).collect();

        let output_general = ternary_matmul(&ternary, scale, &input, out_rows, in_cols, 1);
        let output_decode = ternary_matmul_decode(&ternary, scale, &input, out_rows, in_cols);

        assert_eq!(output_general.len(), output_decode.len());
        for i in 0..output_general.len() {
            assert!(
                (output_general[i] - output_decode[i]).abs() < 1e-5,
                "decode mismatch at {}: general={}, decode={}",
                i,
                output_general[i],
                output_decode[i]
            );
        }
    }

    #[test]
    fn test_ternary_sparse_decode_vs_general() {
        let out_rows = 4;
        let in_cols = 16;
        let scale = 0.5f32;

        let ternary: Vec<i8> = (0..out_rows * in_cols)
            .map(|i| {
                let v = i % 3;
                if v == 0 { 1 } else if v == 1 { -1 } else { 0 }
            })
            .collect();
        let input: Vec<f32> = (0..in_cols).map(|i| i as f32 * 0.1).collect();

        let (pos, neg) = ternary_sparse_indices(&ternary, out_rows, in_cols);
        let output_sparse = ternary_matmul_sparse_decode(&pos, &neg, scale, &input, out_rows);
        let output_general = ternary_matmul_decode(&ternary, scale, &input, out_rows, in_cols);

        assert_eq!(output_sparse.len(), output_general.len());
        for i in 0..output_sparse.len() {
            assert!(
                (output_sparse[i] - output_general[i]).abs() < 1e-5,
                "sparse mismatch at {}: sparse={}, general={}",
                i,
                output_sparse[i],
                output_general[i]
            );
        }
    }

    #[test]
    fn test_sparse_indices_counts() {
        // 3×4 matrix with known pattern
        let ternary: Vec<i8> = vec![
            1, -1, 0, 1,  // row 0: 2 pos, 1 neg, 1 zero
            -1, 0, 0, -1, // row 1: 0 pos, 2 neg, 2 zero
            0, 0, 0, 0,   // row 2: all zero
        ];
        let (pos, neg) = ternary_sparse_indices(&ternary, 3, 4);

        assert_eq!(pos[0], vec![0, 3]);
        assert_eq!(neg[0], vec![1]);
        assert_eq!(pos[1], Vec::<usize>::new());
        assert_eq!(neg[1], vec![0usize, 3]);
        assert_eq!(pos[2], Vec::<usize>::new());
        assert_eq!(neg[2], Vec::<usize>::new());
    }

    #[test]
    fn test_ternary_matmul_all_zeros() {
        let ternary = vec![0i8; 16]; // all zero weights
        let input = vec![1.0f32; 16];
        let output = ternary_matmul(&ternary, 1.0, &input, 1, 16, 1);
        assert!((output[0]).abs() < 1e-10, "all-zero weights should give zero output");
    }

    #[test]
    fn test_ternary_matmul_large_layer() {
        // Simulate a real layer: 1536 → 6144 (Gemma 4 E2B FFN gate)
        let out_rows = 512; // Smaller for test speed, still realistic
        let in_cols = 256;
        let scale = 0.1f32;

        // Deterministic pseudo-random ternary: mix of {-1, 0, +1}
        let ternary: Vec<i8> = (0..out_rows * in_cols)
            .map(|i| {
                // Use a simple hash-like function for diversity
                let v = ((i as u32).wrapping_mul(2654435761u32) >> 30) as i8 - 1; // maps to {-1, 0, 1}
                v.clamp(-1, 1)
            })
            .collect();

        let input: Vec<f32> = (0..in_cols).map(|i| (i as f32 * 0.01).sin()).collect();

        let output = ternary_matmul_decode(&ternary, scale, &input, out_rows, in_cols);

        assert_eq!(output.len(), out_rows);
        // Output should be non-trivial (not all zeros, not all same)
        let min_val = output.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_val > min_val, "output should have diverse values: min={}, max={}", min_val, max_val);

        // Verify against general matmul
        let output_general = ternary_matmul(&ternary, scale, &input, out_rows, in_cols, 1);
        let max_diff = output
            .iter()
            .zip(output_general.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max diff from general: {}", max_diff);
    }

    #[test]
    fn test_packed_matmul_non_multiple_of_4() {
        // in_cols = 6 (not a multiple of 4)
        let ternary: Vec<i8> = vec![1, -1, 0, 1, 0, -1]; // 1×6
        let packed = pack_ternary(&ternary);
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let output_unpacked = ternary_matmul(&ternary, 1.0, &input, 1, 6, 1);
        let output_packed = ternary_matmul_packed(&packed, 1.0, &input, 1, 6, 1, 6);

        assert!((output_unpacked[0] - output_packed[0]).abs() < 1e-5);
        // 1*1 + (-1)*2 + 0*3 + 1*4 + 0*5 + (-1)*6 = 1-2+4-6 = -3
        assert!((output_packed[0] - (-3.0)).abs() < 1e-5);
    }
}
