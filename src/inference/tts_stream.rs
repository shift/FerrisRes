//! Streaming TTS output — AudioStreamReconstructor with overlap-add.
//!
//! Converts a stream of EnCodec multi-codebook tokens into a continuous
//! PCM waveform with artifact-free cross-fade at chunk boundaries.
//!
//! Pipeline:
//!   1. SpeechHead produces [num_codebooks × T] token IDs per autoregressive step
//!   2. AudioStreamReconstructor buffers tokens until a full chunk is ready
//!   3. Decodes via EnCodecConv (or simplified codebook lookup)
//!   4. Applies overlap-add cross-fade with raised-cosine window
//!   5. Outputs PCM chunks ready for ALSA/PulseAudio playback
//!
//! Latency target: sub-200ms from token generated → sound heard.

// ---------------------------------------------------------------------------
// AudioStreamReconstructor — buffered decode with overlap-add
// ---------------------------------------------------------------------------

/// Configuration for the streaming audio reconstructor.
#[derive(Debug, Clone)]
pub struct AudioStreamReconstructorConfig {
    /// Number of EnCodec codebooks (e.g., 8).
    pub num_codebooks: usize,
    /// Codebook size (e.g., 1024).
    pub codebook_size: usize,
    /// Number of frames to accumulate before decoding.
    /// 4-8 frames gives sub-200ms latency at 75 Hz frame rate.
    pub frames_per_chunk: usize,
    /// Cross-fade duration in samples (at target sample rate).
    /// 5-10ms cross-fade prevents phase-discontinuity clicks.
    pub crossfade_samples: usize,
    /// Target sample rate for output (e.g., 24000).
    pub sample_rate: u32,
    /// Frame rate from the encoder (e.g., 75 Hz for EnCodec).
    pub frame_rate: f32,
}

impl Default for AudioStreamReconstructorConfig {
    fn default() -> Self {
        Self {
            num_codebooks: 8,
            codebook_size: 1024,
            frames_per_chunk: 6,
            crossfade_samples: 240,  // 10ms at 24kHz
            sample_rate: 24000,
            frame_rate: 75.0,
        }
    }
}

impl AudioStreamReconstructorConfig {
    /// Create config for standard EnCodec 24kHz.
    pub fn encodec_24k() -> Self {
        Self::default()
    }

    /// Estimated samples per frame (samples_per_frame = sample_rate / frame_rate).
    pub fn samples_per_frame(&self) -> usize {
        (self.sample_rate as f32 / self.frame_rate) as usize
    }

    /// Estimated samples per chunk.
    pub fn samples_per_chunk(&self) -> usize {
        self.samples_per_frame() * self.frames_per_chunk
    }

    /// Estimated latency in milliseconds.
    pub fn estimated_latency_ms(&self) -> f32 {
        // Time to accumulate a chunk + decode + crossfade
        let chunk_duration_ms = self.frames_per_chunk as f32 / self.frame_rate * 1000.0;
        let crossfade_ms = self.crossfade_samples as f32 / self.sample_rate as f32 * 1000.0;
        chunk_duration_ms + crossfade_ms
    }
}

/// Streaming audio reconstructor with overlap-add cross-fade.
///
/// Buffers incoming multi-codebook tokens, decodes chunks via codebook
/// lookup, and applies raised-cosine windowed cross-fade at boundaries
/// to prevent audible clicks.
pub struct AudioStreamReconstructor {
    config: AudioStreamReconstructorConfig,
    /// Codebook embeddings: [num_codebooks × (codebook_size × embed_dim)].
    /// Each codebook maps token indices to embedding vectors.
    codebooks: Vec<Vec<f32>>,
    /// Embedding dimension per codebook entry.
    embed_dim: usize,
    /// Token buffer: [num_codebooks][frame_idx] → token ID.
    token_buffer: Vec<Vec<u32>>,
    /// Output PCM buffer (accumulated samples for playback).
    pcm_buffer: Vec<f32>,
    /// Tail from previous chunk for overlap-add.
    overlap_tail: Vec<f32>,
    /// Total frames processed.
    frames_processed: usize,
    /// Total chunks decoded.
    chunks_decoded: usize,
}

impl AudioStreamReconstructor {
    /// Create a new reconstructor with random codebook initialization.
    pub fn new(config: AudioStreamReconstructorConfig) -> Self {
        let embed_dim = 128; // Standard EnCodec embedding dimension
        let codebooks: Vec<Vec<f32>> = (0..config.num_codebooks)
            .map(|cb| {
                let scale = (2.0 / (config.codebook_size + embed_dim) as f32).sqrt();
                (0..config.codebook_size * embed_dim)
                    .map(|i| {
                        let seed = i as f32 + cb as f32 * 500.0;
                        let x = ((seed * 0.618 + 0.1).sin() * 43758.5453).fract() - 0.5;
                        x * scale
                    })
                    .collect()
            })
            .collect();

        let token_buffer = vec![Vec::new(); config.num_codebooks];

        Self {
            config,
            codebooks,
            embed_dim,
            token_buffer,
            pcm_buffer: Vec::new(),
            overlap_tail: Vec::new(),
            frames_processed: 0,
            chunks_decoded: 0,
        }
    }

    /// Create with pre-trained codebook weights.
    pub fn with_codebooks(
        config: AudioStreamReconstructorConfig,
        codebooks: Vec<Vec<f32>>,
        embed_dim: usize,
    ) -> Self {
        assert_eq!(codebooks.len(), config.num_codebooks);
        let token_buffer = vec![Vec::new(); config.num_codebooks];
        Self {
            config,
            codebooks,
            embed_dim,
            token_buffer,
            pcm_buffer: Vec::new(),
            overlap_tail: Vec::new(),
            frames_processed: 0,
            chunks_decoded: 0,
        }
    }

    /// Process a single frame of multi-codebook tokens.
    ///
    /// `tokens`: one token per codebook (len = num_codebooks).
    /// Returns PCM samples if a full chunk is ready, None otherwise.
    pub fn process_frame(&mut self, tokens: &[u32]) -> Option<Vec<f32>> {
        assert_eq!(tokens.len(), self.config.num_codebooks);
        for (cb, &token) in self.token_buffer.iter_mut().zip(tokens.iter()) {
            cb.push(token);
        }
        self.frames_processed += 1;

        // Check if we have a full chunk
        if self.token_buffer[0].len() >= self.config.frames_per_chunk {
            self.decode_chunk()
        } else {
            None
        }
    }

    /// Process a batch of frames at once.
    ///
    /// `all_tokens`: [num_codebooks × num_frames] token IDs.
    /// Returns all PCM samples produced.
    pub fn process_batch(&mut self, all_tokens: &[Vec<u32>]) -> Vec<f32> {
        assert_eq!(all_tokens.len(), self.config.num_codebooks);
        let num_frames = all_tokens[0].len();
        let mut output = Vec::new();

        for frame_idx in 0..num_frames {
            let frame_tokens: Vec<u32> = all_tokens.iter()
                .map(|cb| cb.get(frame_idx).copied().unwrap_or(0))
                .collect();
            if let Some(pcm) = self.process_frame(&frame_tokens) {
                output.extend_from_slice(&pcm);
            }
        }
        output
    }

    /// Decode a full chunk from the token buffer using codebook lookup.
    fn decode_chunk(&mut self) -> Option<Vec<f32>> {
        let num_frames = self.config.frames_per_chunk;
        let spf = self.config.samples_per_frame();

        // Decode each frame via codebook lookup
        let mut chunk_pcm = vec![0.0f32; num_frames * spf];

        for frame_idx in 0..num_frames {
            // Sum contributions from all codebooks
            let mut frame_embedding = vec![0.0f32; self.embed_dim];

            for (cb_idx, cb_tokens) in self.token_buffer.iter().enumerate() {
                if frame_idx < cb_tokens.len() {
                    let token = cb_tokens[frame_idx] as usize;
                    if token < self.config.codebook_size {
                        let codebook = &self.codebooks[cb_idx];
                        for d in 0..self.embed_dim {
                            let offset = token * self.embed_dim + d;
                            if offset < codebook.len() {
                                frame_embedding[d] += codebook[offset];
                            }
                        }
                    }
                }
            }

            // Map embedding to waveform samples (simplified: energy → amplitude)
            let energy: f32 = frame_embedding.iter().map(|&v| v * v).sum::<f32>()
                / self.embed_dim as f32;
            let amplitude = energy.sqrt() * 0.1; // Scale down

            let base = frame_idx * spf;
            for s in 0..spf {
                if base + s < chunk_pcm.len() {
                    // Simple sinusoidal approximation at base frequency
                    let t = s as f32 / spf as f32;
                    chunk_pcm[base + s] = amplitude * (2.0 * std::f32::consts::PI * t).sin();
                }
            }
        }

        // Apply overlap-add cross-fade with previous chunk's tail
        let output = self.overlap_add(chunk_pcm);

        // Clear processed tokens from buffer
        for cb in &mut self.token_buffer {
            cb.drain(0..num_frames);
        }

        self.chunks_decoded += 1;
        Some(output)
    }

    /// Apply overlap-add with raised-cosine cross-fade.
    fn overlap_add(&mut self, chunk: Vec<f32>) -> Vec<f32> {
        let crossfade = self.config.crossfade_samples.min(chunk.len());

        if self.overlap_tail.is_empty() {
            // First chunk: no cross-fade needed
            // Save tail for next chunk
            let tail_start = chunk.len().saturating_sub(crossfade);
            self.overlap_tail = chunk[tail_start..].to_vec();
            return chunk;
        }

        // Cross-fade: blend overlap_tail with chunk start
        let mut output = Vec::with_capacity(chunk.len());
        let tail_len = self.overlap_tail.len().min(crossfade);

        // Blend overlapping region with raised-cosine window
        for i in 0..tail_len {
            let alpha = i as f32 / tail_len as f32; // 0→1 raised cosine
            let window = 0.5 * (1.0 - (std::f32::consts::PI * alpha).cos()); // Hann window
            let tail_val = self.overlap_tail.get(i).copied().unwrap_or(0.0);
            let chunk_val = chunk.get(i).copied().unwrap_or(0.0);
            output.push(tail_val * (1.0 - window) + chunk_val * window);
        }

        // Append rest of chunk after cross-fade
        if tail_len < chunk.len() {
            output.extend_from_slice(&chunk[tail_len..]);
        }

        // Save new tail
        let tail_start = chunk.len().saturating_sub(crossfade);
        self.overlap_tail = chunk[tail_start..].to_vec();

        output
    }

    /// Flush any remaining buffered tokens.
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        if self.token_buffer[0].is_empty() {
            return None;
        }
        // Temporarily set frames_per_chunk to remaining count
        let remaining = self.token_buffer[0].len();
        if remaining > 0 {
            // Decode with whatever we have
            let num_frames = remaining;
            let spf = self.config.samples_per_frame();
            let mut chunk_pcm = vec![0.0f32; num_frames * spf];

            for frame_idx in 0..num_frames {
                let mut frame_embedding = vec![0.0f32; self.embed_dim];
                for (cb_idx, cb_tokens) in self.token_buffer.iter().enumerate() {
                    if frame_idx < cb_tokens.len() {
                        let token = cb_tokens[frame_idx] as usize;
                        if token < self.config.codebook_size {
                            let codebook = &self.codebooks[cb_idx];
                            for d in 0..self.embed_dim {
                                let offset = token * self.embed_dim + d;
                                if offset < codebook.len() {
                                    frame_embedding[d] += codebook[offset];
                                }
                            }
                        }
                    }
                }
                let energy: f32 = frame_embedding.iter().map(|&v| v * v).sum::<f32>()
                    / self.embed_dim as f32;
                let amplitude = energy.sqrt() * 0.1;
                let base = frame_idx * spf;
                for s in 0..spf {
                    if base + s < chunk_pcm.len() {
                        let t = s as f32 / spf as f32;
                        chunk_pcm[base + s] = amplitude * (2.0 * std::f32::consts::PI * t).sin();
                    }
                }
            }

            let output = self.overlap_add(chunk_pcm);
            for cb in &mut self.token_buffer {
                cb.drain(0..num_frames);
            }
            self.chunks_decoded += 1;
            Some(output)
        } else {
            None
        }
    }

    /// Get accumulated PCM buffer.
    pub fn pcm_buffer(&self) -> &[f32] {
        &self.pcm_buffer
    }

    /// Drain the PCM buffer (for playback).
    pub fn drain_pcm(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.pcm_buffer)
    }

    /// Number of frames processed.
    pub fn frames_processed(&self) -> usize {
        self.frames_processed
    }

    /// Number of chunks decoded.
    pub fn chunks_decoded(&self) -> usize {
        self.chunks_decoded
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        for cb in &mut self.token_buffer {
            cb.clear();
        }
        self.pcm_buffer.clear();
        self.overlap_tail.clear();
        self.frames_processed = 0;
        self.chunks_decoded = 0;
    }

    /// Config accessor.
    pub fn config(&self) -> &AudioStreamReconstructorConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Raised cosine (Hann) window utility
// ---------------------------------------------------------------------------

/// Generate a Hann window of given length.
pub fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let n = i as f32 / (length - 1).max(1) as f32;
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * n).cos())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = AudioStreamReconstructorConfig::default();
        assert_eq!(config.num_codebooks, 8);
        assert_eq!(config.codebook_size, 1024);
        assert_eq!(config.frames_per_chunk, 6);
        assert_eq!(config.sample_rate, 24000);
        assert!((config.frame_rate - 75.0).abs() < 1e-5);
    }

    #[test]
    fn test_config_latency_estimate() {
        let config = AudioStreamReconstructorConfig::default();
        let latency = config.estimated_latency_ms();
        // 6 frames / 75Hz = 80ms chunk + 10ms crossfade ≈ 90ms
        assert!(latency > 0.0 && latency < 200.0, "Latency {}ms should be <200ms", latency);
    }

    #[test]
    fn test_config_samples_per_frame() {
        let config = AudioStreamReconstructorConfig::default();
        // 24000 / 75 = 320
        assert_eq!(config.samples_per_frame(), 320);
    }

    #[test]
    fn test_config_samples_per_chunk() {
        let config = AudioStreamReconstructorConfig::default();
        // 320 * 6 = 1920
        assert_eq!(config.samples_per_chunk(), 1920);
    }

    #[test]
    fn test_reconstructor_creation() {
        let recon = AudioStreamReconstructor::new(
            AudioStreamReconstructorConfig::default(),
        );
        assert_eq!(recon.frames_processed(), 0);
        assert_eq!(recon.chunks_decoded(), 0);
        assert!(recon.pcm_buffer().is_empty());
    }

    #[test]
    fn test_process_frame_no_output_until_chunk_full() {
        let mut recon = AudioStreamReconstructor::new(
            AudioStreamReconstructorConfig {
                frames_per_chunk: 4,
                ..Default::default()
            },
        );
        // Process 3 frames — not enough for a chunk
        for _ in 0..3 {
            let tokens = vec![0u32; 8];
            assert!(recon.process_frame(&tokens).is_none());
        }
        assert_eq!(recon.frames_processed(), 3);
    }

    #[test]
    fn test_process_frame_output_on_chunk_full() {
        let mut recon = AudioStreamReconstructor::new(
            AudioStreamReconstructorConfig {
                frames_per_chunk: 4,
                ..Default::default()
            },
        );
        // Process 4 frames — should produce output
        for _ in 0..3 {
            let tokens = vec![0u32; 8];
            assert!(recon.process_frame(&tokens).is_none());
        }
        let tokens = vec![0u32; 8];
        let pcm = recon.process_frame(&tokens);
        assert!(pcm.is_some());
        let pcm = pcm.unwrap();
        assert!(!pcm.is_empty());
        assert_eq!(recon.chunks_decoded(), 1);
    }

    #[test]
    fn test_process_batch() {
        let config = AudioStreamReconstructorConfig {
            frames_per_chunk: 4,
            ..Default::default()
        };
        let mut recon = AudioStreamReconstructor::new(config);

        // 8 frames = 2 chunks
        let all_tokens: Vec<Vec<u32>> = (0..8).map(|_| vec![0u32; 8]).collect();
        let pcm = recon.process_batch(&all_tokens);
        assert!(!pcm.is_empty());
        assert_eq!(recon.chunks_decoded(), 2);
    }

    #[test]
    fn test_overlap_add_first_chunk() {
        let config = AudioStreamReconstructorConfig {
            frames_per_chunk: 2,
            crossfade_samples: 100,
            sample_rate: 8000,
            frame_rate: 50.0,
            ..Default::default()
        };
        let mut recon = AudioStreamReconstructor::new(config);

        // First chunk: no cross-fade
        for _ in 0..2 {
            recon.process_frame(&vec![0u32; 8]);
        }
        // Should have overlap tail stored now
        assert!(!recon.overlap_tail.is_empty());
    }

    #[test]
    fn test_overlap_add_second_chunk() {
        let config = AudioStreamReconstructorConfig {
            frames_per_chunk: 2,
            crossfade_samples: 100,
            sample_rate: 8000,
            frame_rate: 50.0,
            ..Default::default()
        };
        let mut recon = AudioStreamReconstructor::new(config);

        // First chunk
        for _ in 0..2 {
            recon.process_frame(&vec![0u32; 8]);
        }
        // Second chunk — should trigger overlap-add
        for _ in 0..2 {
            recon.process_frame(&vec![1u32; 8]);
        }
        assert_eq!(recon.chunks_decoded(), 2);
    }

    #[test]
    fn test_flush_remaining() {
        let config = AudioStreamReconstructorConfig {
            frames_per_chunk: 4,
            ..Default::default()
        };
        let mut recon = AudioStreamReconstructor::new(config);

        // Process 2 frames (not enough for a chunk)
        for _ in 0..2 {
            recon.process_frame(&vec![0u32; 8]);
        }

        // Flush should decode remaining
        let pcm = recon.flush();
        assert!(pcm.is_some());
    }

    #[test]
    fn test_flush_empty() {
        let mut recon = AudioStreamReconstructor::new(
            AudioStreamReconstructorConfig::default(),
        );
        assert!(recon.flush().is_none());
    }

    #[test]
    fn test_reset() {
        let mut recon = AudioStreamReconstructor::new(
            AudioStreamReconstructorConfig {
                frames_per_chunk: 2,
                ..Default::default()
            },
        );
        for _ in 0..2 {
            recon.process_frame(&vec![0u32; 8]);
        }
        assert!(recon.frames_processed() > 0);
        recon.reset();
        assert_eq!(recon.frames_processed(), 0);
        assert_eq!(recon.chunks_decoded(), 0);
        assert!(recon.pcm_buffer().is_empty());
    }

    #[test]
    fn test_hann_window() {
        let window = hann_window(8);
        assert_eq!(window.len(), 8);
        // Hann window starts at 0, peaks at 1.0 in the middle
        assert!(window[0] < 0.1);
        assert!(window[7] < 0.1);
        assert!(window[3] > 0.5 || window[4] > 0.5);
    }

    #[test]
    fn test_with_codebooks() {
        let config = AudioStreamReconstructorConfig {
            num_codebooks: 2,
            codebook_size: 4,
            ..Default::default()
        };
        let codebooks = vec![
            vec![0.1f32; 4 * 16],
            vec![0.2f32; 4 * 16],
        ];
        let recon = AudioStreamReconstructor::with_codebooks(config, codebooks, 16);
        assert_eq!(recon.embed_dim, 16);
    }
}
