//! Streaming audio I/O pipeline for real-time audio processing.
//!
//! Provides chunked audio processing for integration with EnCodec
//! and real-time multimodal conversation:
//! - LiveAudioStream: capture audio in fixed-size windows
//! - AudioChunker: split audio into overlapping windows for streaming
//! - StreamingAudioEncoder: encode chunks and feed tokens as they arrive
//! - Ring buffer for low-latency audio capture

// ---------------------------------------------------------------------------
// AudioFormat — describes the audio sample format
// ---------------------------------------------------------------------------

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioSampleFormat {
    F32,
    I16,
    I32,
    U8,
}

/// Audio format descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: AudioSampleFormat,
}

impl AudioFormat {
    pub fn mono_16k() -> Self {
        Self { sample_rate: 16000, channels: 1, sample_format: AudioSampleFormat::F32 }
    }

    pub fn stereo_44k() -> Self {
        Self { sample_rate: 44100, channels: 2, sample_format: AudioSampleFormat::F32 }
    }

    pub fn mono_24k() -> Self {
        Self { sample_rate: 24000, channels: 1, sample_format: AudioSampleFormat::F32 }
    }

    /// Bytes per sample.
    pub fn bytes_per_sample(&self) -> usize {
        match self.sample_format {
            AudioSampleFormat::F32 => 4,
            AudioSampleFormat::I16 => 2,
            AudioSampleFormat::I32 => 4,
            AudioSampleFormat::U8 => 1,
        }
    }

    /// Bytes per frame (one sample per channel).
    pub fn bytes_per_frame(&self) -> usize {
        self.bytes_per_sample() * self.channels as usize
    }

    /// Duration of `n_samples` in seconds.
    pub fn duration_seconds(&self, n_samples: usize) -> f32 {
        n_samples as f32 / self.sample_rate as f32
    }

    /// Number of samples for a given duration.
    pub fn samples_for_duration(&self, seconds: f32) -> usize {
        (self.sample_rate as f32 * seconds) as usize
    }
}

// ---------------------------------------------------------------------------
// AudioWindow — a chunk of audio samples
// ---------------------------------------------------------------------------

/// A window of audio samples (interleaved multi-channel, f32).
#[derive(Debug, Clone)]
pub struct AudioWindow {
    /// Sample data (interleaved f32).
    pub samples: Vec<f32>,
    /// Audio format.
    pub format: AudioFormat,
    /// Start sample index in the stream.
    pub start_sample: usize,
}

impl AudioWindow {
    pub fn new(samples: Vec<f32>, format: AudioFormat, start_sample: usize) -> Self {
        Self { samples, format, start_sample }
    }

    /// Number of frames (samples per channel).
    pub fn num_frames(&self) -> usize {
        self.samples.len() / self.format.channels as usize
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f32 {
        self.format.duration_seconds(self.num_frames())
    }

    /// Whether this window is silent (all zeros).
    pub fn is_silent(&self) -> bool {
        self.samples.iter().all(|&s| s == 0.0)
    }

    /// RMS energy of the window.
    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() { return 0.0; }
        let sum_sq: f32 = self.samples.iter().map(|&s| s * s).sum();
        (sum_sq / self.samples.len() as f32).sqrt()
    }

    /// Peak amplitude.
    pub fn peak(&self) -> f32 {
        self.samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max)
    }

    /// Get mono by averaging channels.
    pub fn to_mono(&self) -> Vec<f32> {
        let channels = self.format.channels as usize;
        let frames = self.num_frames();
        let mut mono = Vec::with_capacity(frames);
        for i in 0..frames {
            let mut sum = 0.0f32;
            for ch in 0..channels {
                sum += self.samples[i * channels + ch];
            }
            mono.push(sum / channels as f32);
        }
        mono
    }

    /// Apply a fade-in over the first `n` samples.
    pub fn fade_in(&mut self, n: usize) {
        let channels = self.format.channels as usize;
        for i in 0..n.min(self.num_frames()) {
            let gain = i as f32 / n as f32;
            for ch in 0..channels {
                self.samples[i * channels + ch] *= gain;
            }
        }
    }

    /// Apply a fade-out over the last `n` samples.
    pub fn fade_out(&mut self, n: usize) {
        let channels = self.format.channels as usize;
        let frames = self.num_frames();
        for i in 0..n.min(frames) {
            let gain = i as f32 / n as f32;
            let idx = frames - 1 - i;
            for ch in 0..channels {
                self.samples[idx * channels + ch] *= gain;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AudioChunker — splits audio into overlapping windows
// ---------------------------------------------------------------------------

/// Configuration for audio chunking.
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Window size in samples (per channel).
    pub window_samples: usize,
    /// Hop size between windows.
    pub hop_samples: usize,
    /// Whether to apply fade in/out for overlap.
    pub fade: bool,
}

impl ChunkerConfig {
    /// 30-second windows at 16kHz with 50% overlap.
    pub fn seconds_30_16k() -> Self {
        Self {
            window_samples: 16000 * 30,
            hop_samples: 16000 * 15,
            fade: true,
        }
    }

    /// 10-second windows at 24kHz with 5-second hop.
    pub fn seconds_10_24k() -> Self {
        Self {
            window_samples: 24000 * 10,
            hop_samples: 24000 * 5,
            fade: true,
        }
    }

    /// Number of windows for a given total number of samples.
    pub fn num_windows(&self, total_samples: usize) -> usize {
        if total_samples < self.window_samples {
            return 1;
        }
        (total_samples - self.window_samples) / self.hop_samples + 1
    }
}

/// Splits an audio stream into overlapping windows.
pub struct AudioChunker {
    config: ChunkerConfig,
    format: AudioFormat,
    /// Buffered samples.
    buffer: Vec<f32>,
    /// Current position.
    position: usize,
    /// Windows emitted so far.
    windows_emitted: usize,
}

impl AudioChunker {
    pub fn new(format: AudioFormat, config: ChunkerConfig) -> Self {
        Self {
            config,
            format,
            buffer: Vec::new(),
            position: 0,
            windows_emitted: 0,
        }
    }

    /// Push new samples into the chunker.
    pub fn push(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);
    }

    /// Try to get the next window.
    pub fn next_window(&mut self) -> Option<AudioWindow> {
        let channels = self.format.channels as usize;
        let needed = self.config.window_samples * channels;

        let start_sample = self.windows_emitted * self.config.hop_samples;
        let start_idx = start_sample * channels;

        if start_idx + needed > self.buffer.len() {
            return None;
        }

        let samples = self.buffer[start_idx..start_idx + needed].to_vec();
        let mut window = AudioWindow::new(samples, self.format, start_sample);

        if self.config.fade {
            let fade_len = (self.config.hop_samples * channels).min(needed / 2);
            // Only apply fade if not the first window
            if self.windows_emitted > 0 {
                window.fade_in(fade_len);
            }
            window.fade_out(fade_len);
        }

        self.windows_emitted += 1;
        Some(window)
    }

    /// Flush remaining samples as a final (possibly shorter) window.
    pub fn flush(&mut self) -> Option<AudioWindow> {
        let channels = self.format.channels as usize;
        let start_sample = self.windows_emitted * self.config.hop_samples;
        let start_idx = start_sample * channels;

        if start_idx >= self.buffer.len() {
            return None;
        }

        let _remaining = self.buffer.len() - start_idx;
        let samples = self.buffer[start_idx..].to_vec();
        let window = AudioWindow::new(samples, self.format, start_sample);
        self.windows_emitted += 1;
        Some(window)
    }

    /// Number of complete windows available.
    pub fn available_windows(&self) -> usize {
        let channels = self.format.channels as usize;
        let needed = self.config.window_samples * channels;
        let buffered = self.buffer.len().saturating_sub(self.position * channels);

        if buffered < needed { return 0; }
        (buffered - needed) / (self.config.hop_samples * channels) + 1
    }

    /// Reset the chunker.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.position = 0;
        self.windows_emitted = 0;
    }

    /// Number of buffered samples.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len() / self.format.channels as usize
    }
}

// ---------------------------------------------------------------------------
// AudioRingBuffer — lock-free ring buffer for live capture
// ---------------------------------------------------------------------------

/// Ring buffer for audio capture (single-producer, single-consumer).
pub struct AudioRingBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
    full: bool,
}

impl AudioRingBuffer {
    pub fn new(capacity_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity_samples],
            capacity: capacity_samples,
            write_pos: 0,
            read_pos: 0,
            full: false,
        }
    }

    /// Write samples into the ring buffer. Returns number written.
    pub fn write(&mut self, samples: &[f32]) -> usize {
        let mut written = 0;
        for &s in samples {
            if self.full { break; }
            self.buffer[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.capacity;
            if self.write_pos == self.read_pos {
                self.full = true;
            }
            written += 1;
        }
        written
    }

    /// Read samples from the ring buffer. Returns number read.
    pub fn read(&mut self, out: &mut [f32]) -> usize {
        let mut read = 0;
        for o in out.iter_mut() {
            if self.read_pos == self.write_pos && !self.full { break; }
            *o = self.buffer[self.read_pos];
            self.read_pos = (self.read_pos + 1) % self.capacity;
            self.full = false;
            read += 1;
        }
        read
    }

    /// Number of samples available to read.
    pub fn available(&self) -> usize {
        if self.full {
            self.capacity
        } else if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.read_pos == self.write_pos && !self.full
    }

    /// Capacity in samples.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reset the ring buffer.
    pub fn reset(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.full = false;
    }
}

// ---------------------------------------------------------------------------
// StreamingAudioEncoder — encode audio chunks into tokens
// ---------------------------------------------------------------------------

/// Encodes audio windows into discrete tokens for the model.
pub struct StreamingAudioEncoder {
    /// Codebook size (e.g., 1024 for EnCodec).
    codebook_size: usize,
    /// Number of codebooks (RVQ layers).
    num_codebooks: usize,
    /// Frame rate (tokens per second).
    frame_rate: f32,
    /// Accumulated tokens.
    tokens: Vec<Vec<u32>>,
    /// Total frames encoded.
    frames_encoded: usize,
}

impl StreamingAudioEncoder {
    pub fn new(codebook_size: usize, num_codebooks: usize, frame_rate: f32) -> Self {
        Self {
            codebook_size,
            num_codebooks,
            frame_rate,
            tokens: Vec::new(),
            frames_encoded: 0,
        }
    }

    /// Encode a window of audio into tokens.
    ///
    /// In a real implementation, this would run the EnCodec encoder.
    /// Here we simulate it with a simple quantization.
    pub fn encode_window(&mut self, window: &AudioWindow) -> Vec<Vec<u32>> {
        let duration = window.duration();
        let n_frames = (duration * self.frame_rate) as usize;
        let mono = window.to_mono();

        let mut frame_tokens = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let mut cb_tokens = Vec::with_capacity(self.num_codebooks);
            let sample_offset = (frame_idx as f32 / self.frame_rate * window.format.sample_rate as f32) as usize;
            let sample = mono.get(sample_offset).copied().unwrap_or(0.0);

            for cb in 0..self.num_codebooks {
                // Simple quantization: map [-1, 1] → [0, codebook_size)
                let normalized = (sample + 1.0) / 2.0;
                let quantized = (normalized * self.codebook_size as f32).floor() as u32;
                let token = quantized.min(self.codebook_size as u32 - 1);
                // Add codebook-specific offset for variety
                let token = (token + cb as u32 * 7) % self.codebook_size as u32;
                cb_tokens.push(token);
            }

            frame_tokens.push(cb_tokens);
        }

        // Append to accumulated tokens
        self.tokens.extend(frame_tokens.clone());
        self.frames_encoded += n_frames;

        frame_tokens
    }

    /// Get all accumulated tokens.
    pub fn tokens(&self) -> &[Vec<u32>] {
        &self.tokens
    }

    /// Flatten tokens for model input: [codebook × frames].
    pub fn flattened_tokens(&self) -> Vec<u32> {
        let mut flat = Vec::new();
        for cb in 0..self.num_codebooks {
            for frame in &self.tokens {
                flat.push(frame[cb]);
            }
        }
        flat
    }

    /// Number of frames encoded.
    pub fn frames_encoded(&self) -> usize {
        self.frames_encoded
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.frames_encoded = 0;
    }

    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format() {
        let fmt = AudioFormat::mono_16k();
        assert_eq!(fmt.sample_rate, 16000);
        assert_eq!(fmt.channels, 1);
        assert_eq!(fmt.bytes_per_sample(), 4);
        assert_eq!(fmt.bytes_per_frame(), 4);
        assert!((fmt.duration_seconds(16000) - 1.0).abs() < 1e-5);
        assert_eq!(fmt.samples_for_duration(0.5), 8000);
    }

    #[test]
    fn test_audio_format_stereo() {
        let fmt = AudioFormat::stereo_44k();
        assert_eq!(fmt.channels, 2);
        assert_eq!(fmt.bytes_per_frame(), 8);
    }

    #[test]
    fn test_audio_window_basic() {
        let fmt = AudioFormat::mono_16k();
        let samples = vec![0.5, -0.3, 0.8, 0.0];
        let window = AudioWindow::new(samples.clone(), fmt, 0);
        assert_eq!(window.num_frames(), 4);
        assert!((window.duration() - 4.0 / 16000.0).abs() < 1e-8);
    }

    #[test]
    fn test_audio_window_rms() {
        let fmt = AudioFormat::mono_16k();
        let window = AudioWindow::new(vec![1.0, -1.0, 1.0, -1.0], fmt, 0);
        assert!((window.rms() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_audio_window_silent() {
        let fmt = AudioFormat::mono_16k();
        let window = AudioWindow::new(vec![0.0; 100], fmt, 0);
        assert!(window.is_silent());
    }

    #[test]
    fn test_audio_window_peak() {
        let fmt = AudioFormat::mono_16k();
        let window = AudioWindow::new(vec![0.1, -0.5, 0.3], fmt, 0);
        assert!((window.peak() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_audio_window_to_mono() {
        let fmt = AudioFormat { sample_rate: 16000, channels: 2, sample_format: AudioSampleFormat::F32 };
        // Stereo: [L0, R0, L1, R1]
        let window = AudioWindow::new(vec![2.0, 4.0, 6.0, 8.0], fmt, 0);
        let mono = window.to_mono();
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 3.0).abs() < 1e-5);
        assert!((mono[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_audio_window_fade() {
        let fmt = AudioFormat::mono_16k();
        let mut window = AudioWindow::new(vec![1.0; 100], fmt, 0);
        window.fade_in(10);
        assert!(window.samples[0] < 0.1);
        assert!((window.samples[9] - 0.9).abs() < 0.01);
        assert!((window.samples[10] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_chunker_config() {
        let config = ChunkerConfig::seconds_30_16k();
        assert_eq!(config.window_samples, 480000);
        assert_eq!(config.hop_samples, 240000);
        // 60 seconds → 3 windows (0s, 15s, 30s starts)
        assert_eq!(config.num_windows(16000 * 60), 3);
    }

    #[test]
    fn test_audio_chunker_basic() {
        let fmt = AudioFormat::mono_16k();
        let config = ChunkerConfig {
            window_samples: 100,
            hop_samples: 50,
            fade: false,
        };
        let mut chunker = AudioChunker::new(fmt, config);
        chunker.push(&vec![1.0; 200]);
        let w1 = chunker.next_window().unwrap();
        assert_eq!(w1.num_frames(), 100);
        let w2 = chunker.next_window().unwrap();
        assert_eq!(w2.num_frames(), 100);
        assert_eq!(w2.start_sample, 50);
        let w3 = chunker.next_window().unwrap();
        assert_eq!(w3.start_sample, 100);
        assert!(chunker.next_window().is_none());
    }

    #[test]
    fn test_audio_chunker_flush() {
        let fmt = AudioFormat::mono_16k();
        let config = ChunkerConfig {
            window_samples: 100,
            hop_samples: 100,
            fade: false,
        };
        let mut chunker = AudioChunker::new(fmt, config);
        chunker.push(&vec![1.0; 150]);
        let w1 = chunker.next_window().unwrap();
        assert_eq!(w1.num_frames(), 100);
        assert!(chunker.next_window().is_none());
        let w2 = chunker.flush().unwrap();
        assert_eq!(w2.num_frames(), 50);
    }

    #[test]
    fn test_audio_chunker_fade() {
        let fmt = AudioFormat::mono_16k();
        let config = ChunkerConfig {
            window_samples: 100,
            hop_samples: 50,
            fade: true,
        };
        let mut chunker = AudioChunker::new(fmt, config);
        chunker.push(&vec![1.0; 200]);
        let w1 = chunker.next_window().unwrap();
        // First window: no fade_in, yes fade_out
        assert!((w1.samples[0] - 1.0).abs() < 1e-5);
        // Second window: fade_in applied
        let w2 = chunker.next_window().unwrap();
        assert!(w2.samples[0] < 1.0); // Faded in
    }

    #[test]
    fn test_ring_buffer_basic() {
        let mut rb = AudioRingBuffer::new(8);
        assert!(rb.is_empty());
        assert_eq!(rb.available(), 0);

        assert_eq!(rb.write(&[1.0, 2.0, 3.0]), 3);
        assert_eq!(rb.available(), 3);

        let mut out = [0.0f32; 2];
        assert_eq!(rb.read(&mut out), 2);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 2.0).abs() < 1e-5);
        assert_eq!(rb.available(), 1);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut rb = AudioRingBuffer::new(4);
        rb.write(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rb.available(), 4);
        assert!(rb.full);

        let mut out = [0.0f32; 2];
        rb.read(&mut out);
        assert!((out[0] - 1.0).abs() < 1e-5);

        rb.write(&[5.0, 6.0]);
        assert_eq!(rb.available(), 4);
    }

    #[test]
    fn test_ring_buffer_full_write() {
        let mut rb = AudioRingBuffer::new(4);
        rb.write(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rb.write(&[5.0]), 0); // Full, can't write
    }

    #[test]
    fn test_ring_buffer_reset() {
        let mut rb = AudioRingBuffer::new(4);
        rb.write(&[1.0, 2.0]);
        rb.reset();
        assert!(rb.is_empty());
        assert_eq!(rb.available(), 0);
    }

    #[test]
    fn test_streaming_encoder() {
        let fmt = AudioFormat::mono_16k();
        let mut encoder = StreamingAudioEncoder::new(1024, 4, 50.0); // 50 fps
        let window = AudioWindow::new(vec![0.5; 16000], fmt, 0); // 1 second
        let tokens = encoder.encode_window(&window);
        assert_eq!(tokens.len(), 50); // 50 frames
        assert_eq!(tokens[0].len(), 4); // 4 codebooks
        assert!(tokens[0][0] < 1024);
    }

    #[test]
    fn test_streaming_encoder_accumulate() {
        let fmt = AudioFormat::mono_16k();
        let mut encoder = StreamingAudioEncoder::new(256, 2, 25.0);
        let w1 = AudioWindow::new(vec![0.5; 16000], fmt, 0);
        let w2 = AudioWindow::new(vec![0.3; 16000], fmt, 16000);

        encoder.encode_window(&w1);
        encoder.encode_window(&w2);
        assert_eq!(encoder.frames_encoded(), 50); // 2 × 25 fps
        assert_eq!(encoder.tokens().len(), 50);
    }

    #[test]
    fn test_streaming_encoder_flatten() {
        let fmt = AudioFormat::mono_16k();
        let mut encoder = StreamingAudioEncoder::new(256, 2, 10.0);
        let window = AudioWindow::new(vec![0.5; 16000], fmt, 0);
        encoder.encode_window(&window);
        let flat = encoder.flattened_tokens();
        // 2 codebooks × 10 frames = 20 tokens
        assert_eq!(flat.len(), 20);
    }

    #[test]
    fn test_streaming_encoder_reset() {
        let fmt = AudioFormat::mono_16k();
        let mut encoder = StreamingAudioEncoder::new(256, 2, 10.0);
        let window = AudioWindow::new(vec![0.5; 16000], fmt, 0);
        encoder.encode_window(&window);
        encoder.reset();
        assert_eq!(encoder.frames_encoded(), 0);
        assert!(encoder.tokens().is_empty());
    }
}
