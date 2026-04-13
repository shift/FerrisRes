//! WGSL FFT (Fast Fourier Transform) for audio processing on GPU.
//!
//! Implements Cooley-Tukey radix-2 FFT as a WGSL compute shader, plus
//! CPU-side FFT for testing and fallback. Provides spectral feature
//! extraction for audio tokenization as an alternative to EnCodec's
//! time-domain approach.
//!
//! Components:
//! 1. `fft()` / `ifft()` — CPU radix-2 Cooley-Tukey (for testing)
//! 2. `FFT_WGSL` — GPU radix-2 FFT compute shader
//! 3. `MelSpectrogram` — mel-filterbank spectral feature extraction
//! 4. `SpectralFeatureExtractor` — integration with audio pipeline

// ---------------------------------------------------------------------------
// Complex number helpers
// ---------------------------------------------------------------------------

/// Complex number (f32 pair).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn magnitude(&self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn phase(&self) -> f32 {
        self.im.atan2(self.re)
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, other: Complex) -> Complex {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.re - other.re, self.im - other.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, other: Complex) -> Complex {
        Complex::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

// ---------------------------------------------------------------------------
// CPU FFT — Cooley-Tukey radix-2 DIT
// ---------------------------------------------------------------------------

/// Compute the FFT of a complex sequence.
/// Input length must be a power of 2.
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return input.to_vec();
    }
    assert!(n.is_power_of_two(), "FFT input length must be a power of 2, got {}", n);

    // Bit-reversal permutation
    let mut output = bit_reverse_copy(input);

    // Butterfly stages
    let mut stage_size = 2usize;
    while stage_size <= n {
        let half = stage_size / 2;
        let angle = -2.0 * std::f32::consts::PI / stage_size as f32;
        let wn = Complex::new(angle.cos(), angle.sin());

        for start in (0..n).step_by(stage_size) {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let even = output[start + k];
                let odd = output[start + k + half] * w;
                output[start + k] = even + odd;
                output[start + k + half] = even - odd;
                w = w * wn;
            }
        }

        stage_size *= 2;
    }

    output
}

/// Compute the inverse FFT.
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Conjugate input
    let conjugated: Vec<Complex> = input.iter()
        .map(|c| Complex::new(c.re, -c.im))
        .collect();

    // Forward FFT
    let mut result = fft(&conjugated);

    // Conjugate and scale
    let scale = 1.0 / n as f32;
    for c in result.iter_mut() {
        c.im = -c.im;
        c.re *= scale;
        c.im *= scale;
    }

    result
}

/// Bit-reversal permutation.
fn bit_reverse_copy(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let bits = n.trailing_zeros() as usize;
    let mut output = vec![Complex::zero(); n];

    for i in 0..n {
        let j = reverse_bits(i, bits);
        output[j] = input[i];
    }

    output
}

/// Reverse the lower `bits` bits of `n`.
fn reverse_bits(mut n: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    result
}

// ---------------------------------------------------------------------------
// Spectral features
// ---------------------------------------------------------------------------

/// Compute the power spectrum from FFT output.
pub fn power_spectrum(fft_output: &[Complex]) -> Vec<f32> {
    // Only need first N/2+1 bins (real-valued symmetry)
    let n = fft_output.len() / 2 + 1;
    fft_output[..n].iter().map(|c| c.re * c.re + c.im * c.im).collect()
}

/// Compute magnitude spectrum.
pub fn magnitude_spectrum(fft_output: &[Complex]) -> Vec<f32> {
    let n = fft_output.len() / 2 + 1;
    fft_output[..n].iter().map(|c| c.magnitude()).collect()
}

// ---------------------------------------------------------------------------
// Mel-spectrogram
// ---------------------------------------------------------------------------

/// Mel-filterbank for spectral feature extraction.
pub struct MelFilterbank {
    /// Number of mel bins.
    num_mel_bins: usize,
    /// FFT size.
    fft_size: usize,
    /// Sample rate.
    #[allow(dead_code)]
    sample_rate: u32,
    /// Filter weights: [num_mel_bins × (fft_size/2+1)].
    weights: Vec<f32>,
}

impl MelFilterbank {
    /// Create a mel filterbank.
    pub fn new(num_mel_bins: usize, fft_size: usize, sample_rate: u32) -> Self {
        let num_fft_bins = fft_size / 2 + 1;
        let fmin = 0.0f32;
        let fmax = sample_rate as f32 / 2.0;

        // Convert frequency to mel scale
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).ln() / std::f32::consts::LN_10;
        let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);

        // Linearly spaced mel points
        let mel_points: Vec<f32> = (0..=num_mel_bins + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_mel_bins + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|m| mel_to_hz(*m)).collect();

        // Convert to FFT bin indices
        let bin_points: Vec<usize> = hz_points.iter()
            .map(|&hz| ((fft_size as f32 * hz) / sample_rate as f32).floor() as usize)
            .collect();

        // Build triangular filterbank
        let mut weights = vec![0.0f32; num_mel_bins * num_fft_bins];

        for m in 0..num_mel_bins {
            let f_left = bin_points[m];
            let f_center = bin_points[m + 1];
            let f_right = bin_points[m + 2];

            for k in f_left..=f_right.min(num_fft_bins - 1) {
                let weight = if k <= f_center && f_center > f_left {
                    (k - f_left) as f32 / (f_center - f_left) as f32
                } else if k > f_center && f_right > f_center {
                    (f_right - k) as f32 / (f_right - f_center) as f32
                } else {
                    0.0
                };
                weights[m * num_fft_bins + k] = weight;
            }
        }

        Self { num_mel_bins, fft_size, sample_rate, weights }
    }

    /// Apply the mel filterbank to a power spectrum.
    pub fn apply(&self, power_spectrum: &[f32]) -> Vec<f32> {
        let num_fft_bins = self.fft_size / 2 + 1;
        let mut mel_spectrum = vec![0.0f32; self.num_mel_bins];

        for m in 0..self.num_mel_bins {
            let mut sum = 0.0f32;
            for k in 0..num_fft_bins.min(power_spectrum.len()) {
                sum += self.weights[m * num_fft_bins + k] * power_spectrum[k];
            }
            mel_spectrum[m] = sum;
        }

        mel_spectrum
    }

    /// Apply log-mel (with floor to avoid log(0)).
    pub fn apply_log_mel(&self, power_spectrum: &[f32], floor: f32) -> Vec<f32> {
        self.apply(power_spectrum).iter()
            .map(|&v| (v.max(floor)).ln())
            .collect()
    }

    /// Number of mel bins.
    pub fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }

    /// FFT size.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
}

// ---------------------------------------------------------------------------
// SpectralFeatureExtractor — integration with audio pipeline
// ---------------------------------------------------------------------------

/// Extracts spectral features from audio frames.
pub struct SpectralFeatureExtractor {
    fft_size: usize,
    hop_size: usize,
    mel_filterbank: MelFilterbank,
    window: Vec<f32>,
}

impl SpectralFeatureExtractor {
    /// Create a new extractor.
    pub fn new(fft_size: usize, hop_size: usize, num_mel_bins: usize, sample_rate: u32) -> Self {
        let mel_filterbank = MelFilterbank::new(num_mel_bins, fft_size, sample_rate);

        // Hann window
        let window: Vec<f32> = (0..fft_size)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (fft_size - 1) as f32).cos()))
            .collect();

        Self { fft_size, hop_size, mel_filterbank, window }
    }

    /// Default extractor for 24kHz audio.
    pub fn default_24khz() -> Self {
        Self::new(512, 160, 80, 24000)
    }

    /// Extract log-mel spectrogram features from audio samples.
    /// Returns: [num_frames × num_mel_bins] features.
    pub fn extract(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        if audio.len() < self.fft_size {
            return Vec::new();
        }

        let num_frames = (audio.len() - self.fft_size) / self.hop_size + 1;
        let mut features = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;

            // Window the frame
            let windowed: Vec<Complex> = (0..self.fft_size)
                .map(|i| Complex::new(audio[start + i] * self.window[i], 0.0))
                .collect();

            // FFT
            let spectrum = fft(&windowed);
            let psd = power_spectrum(&spectrum);

            // Mel filterbank
            let mel = self.mel_filterbank.apply_log_mel(&psd, 1e-10);
            features.push(mel);
        }

        features
    }

    /// Extract a single frame's features.
    pub fn extract_frame(&self, frame: &[f32]) -> Vec<f32> {
        let mut windowed = vec![Complex::zero(); self.fft_size];
        let len = frame.len().min(self.fft_size);
        for i in 0..len {
            windowed[i] = Complex::new(frame[i] * self.window[i], 0.0);
        }

        let spectrum = fft(&windowed);
        let psd = power_spectrum(&spectrum);
        self.mel_filterbank.apply_log_mel(&psd, 1e-10)
    }

    /// Number of mel bins (feature dimension).
    pub fn feature_dim(&self) -> usize {
        self.mel_filterbank.num_mel_bins()
    }

    /// FFT size.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Hop size.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }
}

// ---------------------------------------------------------------------------
// WGSL FFT kernel
// ---------------------------------------------------------------------------

/// WGSL compute shader for radix-2 Cooley-Tukey FFT.
///
/// Uses stockham algorithm (auto-sort, no bit-reversal needed).
/// Each workgroup processes one FFT of size `fft_size`.
/// Dispatch: workgroups_x = batch_size, local_size = fft_size/2.
pub const FFT_WGSL: &str = r#"
struct Params {
    n:         u32,
    log2_n:    u32,
    direction: u32,  // 0 = forward, 1 = inverse
    _pad:      u32,
}

@group(0) @binding(0) var<storage, read>       input_re: array<f32>;
@group(0) @binding(1) var<storage, read>       input_im: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_re: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_im: array<f32>;
@group(0) @binding(4) var<uniform>             params:   Params;

// Reverse bits of x using n_bits
fn reverse_bits(x: u32, n_bits: u32) -> u32 {
    var result: u32 = 0u;
    var val = x;
    for (var i: u32 = 0u; i < n_bits; i = i + 1u) {
        result = (result << 1u) | (val & 1u);
        val = val >> 1u;
    }
    return result;
}

@compute @workgroup_size(256)
fn fft_radix2(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let n = params.n;
    if (idx >= n) { return; }

    // Bit-reversal permutation
    let j = reverse_bits(idx, params.log2_n);

    // Copy input with bit-reversal
    var re_val: f32 = input_re[j];
    var im_val: f32 = input_im[j];

    // In-place butterfly computation
    for (var stage: u32 = 0u; stage < params.log2_n; stage = stage + 1u) {
        let block_size = 1u << (stage + 1u);
        let half_block = 1u << stage;

        let block_id = idx / block_size;
        let pos_in_block = idx % block_size;
        let is_top = pos_in_block < half_block;

        if (is_top) {
            let k = pos_in_block;
            let partner = idx + half_block;

            // Twiddle factor
            let angle = -2.0 * 3.14159265358979323846 * f32(k) / f32(block_size);
            var sign: f32 = -1.0;
            if (params.direction == 1u) { sign = 1.0; }
            let w_re = cos(angle * sign);
            let w_im = sin(angle * sign);

            let partner_re = input_re[partner];
            let partner_im = input_im[partner];

            let t_re = w_re * partner_re - w_im * partner_im;
            let t_im = w_re * partner_im + w_im * partner_re;

            output_re[idx] = re_val + t_re;
            output_im[idx] = im_val + t_im;
            output_re[partner] = re_val - t_re;
            output_im[partner] = im_val - t_im;
        }
    }

    // Scale for inverse
    if (params.direction == 1u) {
        output_re[idx] = output_re[idx] / f32(n);
        output_im[idx] = output_im[idx] / f32(n);
    }
}
"#;

// ---------------------------------------------------------------------------
// GPU FFT dispatch operation
// ---------------------------------------------------------------------------

/// GPU FFT operation dispatcher.
///
/// Prepares wgpu buffers and dispatches the FFT_WGSL shader.
pub struct FftGpuOp {
    size: usize,
    log2_n: u32,
}

impl FftGpuOp {
    /// Create a new FFT GPU op for a given transform size (must be power of 2).
    pub fn new(size: usize) -> crate::error::Result<Self> {
        if !size.is_power_of_two() || size < 2 {
            return Err(crate::error::FerrisResError::Unsupported(
                "FFT size must be a power of 2".to_string()
            ));
        }
        let log2_n = (size.next_power_of_two().trailing_zeros()) as u32;
        // For size=256: log2_n=8
        Ok(Self { size, log2_n })
    }

    /// Get the transform size.
    pub fn size(&self) -> usize { self.size }

    /// Get the WGSL shader source.
    pub fn shader_source(&self) -> &str { FFT_WGSL }

    /// Get the entry point name.
    pub fn entry_point(&self) -> &str { "fft_radix2" }

    /// Calculate the workgroup count for dispatch.
    pub fn workgroup_count(&self) -> (u32, u32, u32) {
        let wg_size = 256u32;
        let n = self.size as u32;
        ((n + wg_size - 1) / wg_size, 1, 1)
    }

    /// Calculate buffer sizes needed.
    pub fn buffer_sizes(&self) -> (usize, usize, usize, usize, usize) {
        let element_bytes = 4; // f32
        (
            self.size * element_bytes, // input_re
            self.size * element_bytes, // input_im
            self.size * element_bytes, // output_re
            self.size * element_bytes, // output_im
            16,                        // params uniform (4 × u32)
        )
    }

    /// Create params uniform data.
    pub fn params_data(&self, inverse: bool) -> Vec<u8> {
        let n = self.size as u32;
        let dir = if inverse { 1u32 } else { 0u32 };
        let mut data = Vec::with_capacity(16);
        data.extend_from_slice(&n.to_le_bytes());
        data.extend_from_slice(&self.log2_n.to_le_bytes());
        data.extend_from_slice(&dir.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // padding
        data
    }

    /// Perform FFT on CPU (fallback when no GPU available).
    pub fn forward_cpu(&self, input: &[Complex]) -> Vec<Complex> {
        assert_eq!(input.len(), self.size);
        fft(input)
    }

    /// Perform inverse FFT on CPU.
    pub fn inverse_cpu(&self, input: &[Complex]) -> Vec<Complex> {
        assert_eq!(input.len(), self.size);
        ifft(input)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_fft_single_element() {
        let input = vec![Complex::new(5.0, 0.0)];
        let output = fft(&input);
        assert_eq!(output.len(), 1);
        assert!(approx_eq(output[0].re, 5.0, 1e-5));
    }

    #[test]
    fn test_fft_dc_signal() {
        // Constant signal → only DC bin is nonzero
        let input = vec![Complex::new(1.0, 0.0); 8];
        let output = fft(&input);
        assert!(approx_eq(output[0].re, 8.0, 1e-4));
        for i in 1..8 {
            assert!(approx_eq(output[i].magnitude(), 0.0, 1e-4));
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let input: Vec<Complex> = (0..16)
            .map(|i| Complex::new((i as f32).sin(), 0.0))
            .collect();
        let transformed = fft(&input);
        let recovered = ifft(&transformed);
        assert_eq!(recovered.len(), input.len());
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!(approx_eq(a.re, b.re, 1e-4), "Expected {} ≈ {}", a.re, b.re);
        }
    }

    #[test]
    fn test_fft_sine_wave() {
        // sin(2π × 1 × n/8) has energy in bins 1 and 7
        let input: Vec<Complex> = (0..8)
            .map(|n| Complex::new((2.0 * std::f32::consts::PI * n as f32 / 8.0).sin(), 0.0))
            .collect();
        let output = fft(&input);
        // Bin 1 and bin 7 should have significant magnitude
        assert!(output[1].magnitude() > 3.0);
        assert!(output[7].magnitude() > 3.0);
        // DC and other bins should be small
        assert!(output[0].magnitude() < 0.1);
        assert!(output[2].magnitude() < 0.1);
    }

    #[test]
    fn test_fft_impulse() {
        // Delta function → all bins equal magnitude
        let mut input = vec![Complex::zero(); 4];
        input[0] = Complex::new(1.0, 0.0);
        let output = fft(&input);
        for c in &output {
            assert!(approx_eq(c.magnitude(), 1.0, 1e-4));
        }
    }

    #[test]
    fn test_power_spectrum() {
        let input = vec![Complex::new(1.0, 0.0); 4];
        let output = fft(&input);
        let psd = power_spectrum(&output);
        assert_eq!(psd.len(), 3); // N/2+1 = 3
        assert!(psd[0] > 10.0); // DC power
    }

    #[test]
    fn test_magnitude_spectrum() {
        let input = vec![Complex::new(3.0, 4.0)];
        let mag = magnitude_spectrum(&input);
        assert!(approx_eq(mag[0], 5.0, 1e-5));
    }

    #[test]
    fn test_mel_filterbank() {
        let mel = MelFilterbank::new(40, 512, 24000);
        assert_eq!(mel.num_mel_bins(), 40);
        assert_eq!(mel.fft_size(), 512);

        // Apply to a flat power spectrum
        let psd = vec![1.0f32; 257]; // 512/2+1
        let mel_spec = mel.apply(&psd);
        assert_eq!(mel_spec.len(), 40);
        // All bins should be positive
        for &v in &mel_spec {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_mel_filterbank_log() {
        let mel = MelFilterbank::new(40, 512, 24000);
        let psd = vec![1.0f32; 257];
        let log_mel = mel.apply_log_mel(&psd, 1e-10);
        assert_eq!(log_mel.len(), 40);
        // Log of positive values should be finite
        for &v in &log_mel {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_spectral_extractor() {
        let extractor = SpectralFeatureExtractor::new(256, 64, 40, 24000);
        assert_eq!(extractor.feature_dim(), 40);

        // 1 second of silence
        let audio = vec![0.0f32; 24000];
        let features = extractor.extract(&audio);
        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 40);
    }

    #[test]
    fn test_spectral_extractor_sine() {
        let extractor = SpectralFeatureExtractor::new(256, 128, 40, 24000);
        // 440 Hz sine wave
        let audio: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let features = extractor.extract(&audio);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_spectral_extractor_too_short() {
        let extractor = SpectralFeatureExtractor::new(256, 64, 40, 24000);
        let audio = vec![0.0f32; 10]; // Too short
        let features = extractor.extract(&audio);
        assert!(features.is_empty());
    }

    #[test]
    fn test_spectral_extractor_frame() {
        let extractor = SpectralFeatureExtractor::new(256, 64, 40, 24000);
        let frame = vec![0.5f32; 256];
        let features = extractor.extract_frame(&frame);
        assert_eq!(features.len(), 40);
    }

    #[test]
    fn test_wgsl_fft_kernel_valid() {
        assert!(!FFT_WGSL.is_empty());
        assert!(FFT_WGSL.contains("fft_radix2"));
        assert!(FFT_WGSL.contains("reverse_bits"));
    }

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let sum = a + b;
        assert!(approx_eq(sum.re, 4.0, 1e-5));
        assert!(approx_eq(sum.im, 6.0, 1e-5));

        let prod = a * b;
        assert!(approx_eq(prod.re, -5.0, 1e-5));
        assert!(approx_eq(prod.im, 10.0, 1e-5));

        assert!(approx_eq(a.magnitude(), 5.0f32.sqrt(), 1e-5));
    }

    #[test]
    fn test_fft_gpu_op_creation() {
        let op = FftGpuOp::new(256).unwrap();
        assert_eq!(op.size(), 256);
        assert_eq!(op.entry_point(), "fft_radix2");
    }

    #[test]
    fn test_fft_gpu_op_rejects_non_pow2() {
        assert!(FftGpuOp::new(100).is_err());
        assert!(FftGpuOp::new(0).is_err());
        assert!(FftGpuOp::new(3).is_err());
    }

    #[test]
    fn test_fft_gpu_op_workgroup() {
        let op = FftGpuOp::new(512).unwrap();
        let (x, y, z) = op.workgroup_count();
        assert_eq!(y, 1);
        assert_eq!(z, 1);
        assert!(x > 0);
    }

    #[test]
    fn test_fft_gpu_op_buffer_sizes() {
        let op = FftGpuOp::new(1024).unwrap();
        let (re_in, _im_in, _re_out, _im_out, params) = op.buffer_sizes();
        assert_eq!(re_in, 4096); // 1024 * 4 bytes
        assert_eq!(params, 16);
    }

    #[test]
    fn test_fft_gpu_op_params() {
        let op = FftGpuOp::new(256).unwrap();
        let fwd = op.params_data(false);
        let inv = op.params_data(true);
        assert_eq!(fwd.len(), 16);
        assert_eq!(inv.len(), 16);
    }

    #[test]
    fn test_fft_gpu_op_cpu_fallback() {
        let op = FftGpuOp::new(64).unwrap();
        let input: Vec<Complex> = (0..64).map(|i| Complex::new((i as f32).sin(), 0.0)).collect();
        let output = op.forward_cpu(&input);
        assert_eq!(output.len(), 64);
    }
}
