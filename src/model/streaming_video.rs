//! Streaming video I/O pipeline for progressive video processing.
//!
//! Provides frame-by-frame video loading and processing:
//! - FrameSampler: extract N frames per second from video
//! - VideoStreamReader: process frames as they arrive
//! - Frame preprocessing: resize to model input resolution
//! - Integration with VisionEncoder + token merging

// ---------------------------------------------------------------------------
// VideoFormat — describes the video format
// ---------------------------------------------------------------------------

/// Video codec format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VideoCodec {
    RawRGB,
    H264,
    H265,
    VP9,
    AV1,
}

/// Video metadata.
#[derive(Debug, Clone)]
pub struct VideoMetadata {
    pub width: usize,
    pub height: usize,
    pub fps: f32,
    pub duration_seconds: f32,
    pub codec: VideoCodec,
    pub num_frames: usize,
}

impl VideoMetadata {
    pub fn new(width: usize, height: usize, fps: f32, duration_seconds: f32) -> Self {
        let num_frames = (fps * duration_seconds) as usize;
        Self {
            width, height, fps, duration_seconds,
            codec: VideoCodec::RawRGB,
            num_frames,
        }
    }

    /// Total number of pixels per frame.
    pub fn pixels_per_frame(&self) -> usize {
        self.width * self.height
    }

    /// Byte size of a single raw RGB frame.
    pub fn frame_bytes(&self) -> usize {
        self.width * self.height * 3
    }
}

// ---------------------------------------------------------------------------
// VideoFrame — a single decoded frame
// ---------------------------------------------------------------------------

/// A decoded video frame.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// RGB pixel data (row-major).
    pub pixels: Vec<u8>,
    /// Frame width.
    pub width: usize,
    /// Frame height.
    pub height: usize,
    /// Frame index in the video.
    pub frame_idx: usize,
    /// Timestamp in seconds.
    pub timestamp: f32,
}

impl VideoFrame {
    pub fn new(pixels: Vec<u8>, width: usize, height: usize, frame_idx: usize, timestamp: f32) -> Self {
        Self { pixels, width, height, frame_idx, timestamp }
    }

    /// Create a synthetic test frame.
    pub fn synthetic(frame_idx: usize, width: usize, height: usize, value: u8) -> Self {
        let fps = 30.0;
        Self {
            pixels: vec![value; width * height * 3],
            width,
            height,
            frame_idx,
            timestamp: frame_idx as f32 / fps,
        }
    }

    /// Number of pixels.
    pub fn num_pixels(&self) -> usize {
        self.width * self.height
    }

    /// Get pixel at (x, y) as (R, G, B).
    pub fn get_pixel(&self, x: usize, y: usize) -> (u8, u8, u8) {
        let offset = (y * self.width + x) * 3;
        (
            self.pixels[offset],
            self.pixels[offset + 1],
            self.pixels[offset + 2],
        )
    }

    /// Set pixel at (x, y).
    pub fn set_pixel(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        let offset = (y * self.width + x) * 3;
        self.pixels[offset] = r;
        self.pixels[offset + 1] = g;
        self.pixels[offset + 2] = b;
    }

    /// Resize using nearest-neighbor (simple, no deps).
    pub fn resize_nearest(&self, new_width: usize, new_height: usize) -> VideoFrame {
        let mut out = vec![0u8; new_width * new_height * 3];
        let x_ratio = self.width as f32 / new_width as f32;
        let y_ratio = self.height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 * x_ratio) as usize;
                let src_y = (y as f32 * y_ratio) as usize;
                let src_offset = (src_y * self.width + src_x) * 3;
                let dst_offset = (y * new_width + x) * 3;
                out[dst_offset] = self.pixels[src_offset];
                out[dst_offset + 1] = self.pixels[src_offset + 1];
                out[dst_offset + 2] = self.pixels[src_offset + 2];
            }
        }

        VideoFrame::new(out, new_width, new_height, self.frame_idx, self.timestamp)
    }

    /// Convert to f32 patches for model input.
    pub fn to_patches(&self, patch_size: usize) -> Vec<Vec<f32>> {
        let mut patches = Vec::new();
        for y in (0..self.height).step_by(patch_size) {
            for x in (0..self.width).step_by(patch_size) {
                let mut patch = Vec::with_capacity(patch_size * patch_size * 3);
                for py in 0..patch_size {
                    for px in 0..patch_size {
                        let ix = x + px;
                        let iy = y + py;
                        if ix < self.width && iy < self.height {
                            let (r, g, b) = self.get_pixel(ix, iy);
                            patch.push(r as f32 / 255.0);
                            patch.push(g as f32 / 255.0);
                            patch.push(b as f32 / 255.0);
                        } else {
                            patch.push(0.0);
                            patch.push(0.0);
                            patch.push(0.0);
                        }
                    }
                }
                patches.push(patch);
            }
        }
        patches
    }

    /// Normalize with ImageNet mean/std.
    pub fn normalize_imagenet(&self) -> Vec<f32> {
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        let mut out = Vec::with_capacity(self.pixels.len());
        for i in (0..self.pixels.len()).step_by(3) {
            out.push((self.pixels[i] as f32 / 255.0 - mean[0]) / std[0]);
            if i + 1 < self.pixels.len() {
                out.push((self.pixels[i + 1] as f32 / 255.0 - mean[1]) / std[1]);
            }
            if i + 2 < self.pixels.len() {
                out.push((self.pixels[i + 2] as f32 / 255.0 - mean[2]) / std[2]);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// FrameSampler — controls which frames to extract
// ---------------------------------------------------------------------------

/// Configuration for frame sampling.
#[derive(Debug, Clone)]
pub struct FrameSamplerConfig {
    /// Target frames per second to extract.
    pub target_fps: f32,
    /// Maximum frames to extract (0 = unlimited).
    pub max_frames: usize,
    /// Target width for resizing.
    pub target_width: usize,
    /// Target height for resizing.
    pub target_height: usize,
    /// Whether to resize frames.
    pub resize: bool,
}

impl Default for FrameSamplerConfig {
    fn default() -> Self {
        Self {
            target_fps: 1.0,
            max_frames: 0,
            target_width: 224,
            target_height: 224,
            resize: true,
        }
    }
}

/// Samples frames from a video at a target rate.
pub struct FrameSampler {
    config: FrameSamplerConfig,
    /// Frame interval (every N-th frame).
    frame_interval: usize,
    /// Next frame index to extract.
    next_frame: usize,
    /// Frames extracted so far.
    extracted: usize,
}

impl FrameSampler {
    pub fn new(config: FrameSamplerConfig, source_fps: f32) -> Self {
        let frame_interval = if config.target_fps >= source_fps {
            1
        } else {
            (source_fps / config.target_fps).round() as usize
        };
        Self {
            config,
            frame_interval,
            next_frame: 0,
            extracted: 0,
        }
    }

    /// Whether a given frame index should be sampled.
    pub fn should_sample(&self, frame_idx: usize) -> bool {
        if self.config.max_frames > 0 && self.extracted >= self.config.max_frames {
            return false;
        }
        frame_idx == self.next_frame
    }

    /// Process a frame: decide whether to sample and optionally resize.
    pub fn process_frame(&mut self, frame: &VideoFrame) -> Option<VideoFrame> {
        if !self.should_sample(frame.frame_idx) {
            return None;
        }

        self.next_frame += self.frame_interval;
        self.extracted += 1;

        if self.config.resize {
            Some(frame.resize_nearest(self.config.target_width, self.config.target_height))
        } else {
            Some(frame.clone())
        }
    }

    /// Number of frames extracted.
    pub fn extracted_count(&self) -> usize {
        self.extracted
    }

    /// Total frames that would be extracted for a video of given length.
    pub fn total_for_duration(&self, duration_seconds: f32) -> usize {
        let total = (duration_seconds * self.config.target_fps) as usize;
        if self.config.max_frames > 0 {
            total.min(self.config.max_frames)
        } else {
            total
        }
    }

    /// Reset sampler.
    pub fn reset(&mut self) {
        self.next_frame = 0;
        self.extracted = 0;
    }
}

// ---------------------------------------------------------------------------
// VideoStreamReader — streaming frame-by-frame processing
// ---------------------------------------------------------------------------

/// Streaming video reader that processes frames as they arrive.
pub struct VideoStreamReader {
    metadata: VideoMetadata,
    sampler: FrameSampler,
    /// Buffered frames ready for consumption.
    buffer: Vec<VideoFrame>,
    /// Whether the stream is finished.
    finished: bool,
}

impl VideoStreamReader {
    pub fn new(metadata: VideoMetadata, sampler_config: FrameSamplerConfig) -> Self {
        let sampler = FrameSampler::new(sampler_config, metadata.fps);
        Self {
            metadata,
            sampler,
            buffer: Vec::new(),
            finished: false,
        }
    }

    /// Push a raw decoded frame into the reader.
    pub fn push_frame(&mut self, frame: VideoFrame) {
        if let Some(sampled) = self.sampler.process_frame(&frame) {
            self.buffer.push(sampled);
        }
    }

    /// Get the next sampled frame.
    pub fn next_frame(&mut self) -> Option<VideoFrame> {
        self.buffer.pop()
    }

    /// Get all buffered frames.
    pub fn drain_frames(&mut self) -> Vec<VideoFrame> {
        std::mem::take(&mut self.buffer)
    }

    /// Number of buffered frames.
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Mark the stream as finished.
    pub fn finish(&mut self) {
        self.finished = true;
    }

    /// Whether the stream is finished.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get the video metadata.
    pub fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    /// Get the sampler.
    pub fn sampler(&self) -> &FrameSampler {
        &self.sampler
    }

    /// Reset the reader.
    pub fn reset(&mut self) {
        self.sampler.reset();
        self.buffer.clear();
        self.finished = false;
    }
}

// ---------------------------------------------------------------------------
// TemporalFrameBuffer — buffers frames for temporal attention
// ---------------------------------------------------------------------------

/// Buffers consecutive frames for temporal attention processing.
pub struct TemporalFrameBuffer {
    /// Maximum number of frames to buffer.
    max_frames: usize,
    /// Buffered frames.
    frames: Vec<VideoFrame>,
}

impl TemporalFrameBuffer {
    pub fn new(max_frames: usize) -> Self {
        Self { max_frames, frames: Vec::new() }
    }

    /// Add a frame, evicting the oldest if at capacity.
    pub fn push(&mut self, frame: VideoFrame) {
        if self.frames.len() >= self.max_frames {
            self.frames.remove(0);
        }
        self.frames.push(frame);
    }

    /// Get all buffered frames.
    pub fn frames(&self) -> &[VideoFrame] {
        &self.frames
    }

    /// Number of buffered frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Whether the buffer is full.
    pub fn is_full(&self) -> bool {
        self.frames.len() >= self.max_frames
    }

    /// Flatten all frames into a single patch tensor for temporal attention.
    /// Returns (flat_data, num_frames, num_patches_per_frame, patch_dim).
    pub fn flatten_for_temporal(&self, patch_size: usize) -> (Vec<f32>, usize, usize, usize) {
        if self.frames.is_empty() {
            return (Vec::new(), 0, 0, 0);
        }

        let first_patches = self.frames[0].to_patches(patch_size);
        let num_patches = first_patches.len();
        let patch_dim = patch_size * patch_size * 3;

        let mut flat = Vec::with_capacity(self.frames.len() * num_patches * patch_dim);
        for frame in &self.frames {
            let patches = frame.to_patches(patch_size);
            for patch in patches {
                flat.extend_from_slice(&patch);
            }
        }

        (flat, self.frames.len(), num_patches, patch_dim)
    }

    /// Reset the buffer.
    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_metadata() {
        let meta = VideoMetadata::new(1920, 1080, 30.0, 60.0);
        assert_eq!(meta.num_frames, 1800);
        assert_eq!(meta.pixels_per_frame(), 1920 * 1080);
        assert_eq!(meta.frame_bytes(), 1920 * 1080 * 3);
    }

    #[test]
    fn test_video_frame_basic() {
        let frame = VideoFrame::synthetic(0, 4, 4, 128);
        assert_eq!(frame.num_pixels(), 16);
        let (r, g, b) = frame.get_pixel(0, 0);
        assert_eq!(r, 128);
        assert_eq!(g, 128);
        assert_eq!(b, 128);
    }

    #[test]
    fn test_video_frame_set_pixel() {
        let mut frame = VideoFrame::synthetic(0, 4, 4, 0);
        frame.set_pixel(2, 3, 255, 128, 64);
        let (r, g, b) = frame.get_pixel(2, 3);
        assert_eq!(r, 255);
        assert_eq!(g, 128);
        assert_eq!(b, 64);
    }

    #[test]
    fn test_video_frame_resize() {
        let frame = VideoFrame::synthetic(0, 8, 8, 200);
        let resized = frame.resize_nearest(4, 4);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(resized.pixels.len(), 48);
    }

    #[test]
    fn test_video_frame_patches() {
        let frame = VideoFrame::synthetic(0, 4, 4, 128);
        let patches = frame.to_patches(2);
        assert_eq!(patches.len(), 4); // 2×2 grid
        assert_eq!(patches[0].len(), 12); // 2×2×3
    }

    #[test]
    fn test_video_frame_normalize() {
        let frame = VideoFrame::synthetic(0, 2, 2, 255);
        let normed = frame.normalize_imagenet();
        assert_eq!(normed.len(), 12); // 2×2×3
        // 255/255=1.0, normalized = (1.0 - 0.485) / 0.229 ≈ 2.25
        assert!(normed[0] > 1.0);
    }

    #[test]
    fn test_frame_sampler() {
        let config = FrameSamplerConfig {
            target_fps: 1.0,
            ..Default::default()
        };
        let sampler = FrameSampler::new(config, 30.0);
        assert_eq!(sampler.frame_interval, 30); // Every 30th frame
    }

    #[test]
    fn test_frame_sampler_process() {
        let config = FrameSamplerConfig {
            target_fps: 10.0,
            resize: false,
            ..Default::default()
        };
        let mut sampler = FrameSampler::new(config, 30.0);
        assert_eq!(sampler.frame_interval, 3);

        // Frame 0: sampled
        let f0 = VideoFrame::synthetic(0, 4, 4, 0);
        assert!(sampler.process_frame(&f0).is_some());

        // Frame 1, 2: not sampled
        let f1 = VideoFrame::synthetic(1, 4, 4, 0);
        assert!(sampler.process_frame(&f1).is_none());
        let f2 = VideoFrame::synthetic(2, 4, 4, 0);
        assert!(sampler.process_frame(&f2).is_none());

        // Frame 3: sampled
        let f3 = VideoFrame::synthetic(3, 4, 4, 0);
        assert!(sampler.process_frame(&f3).is_some());

        assert_eq!(sampler.extracted_count(), 2);
    }

    #[test]
    fn test_frame_sampler_max_frames() {
        let config = FrameSamplerConfig {
            target_fps: 30.0,
            max_frames: 2,
            resize: false,
            ..Default::default()
        };
        let mut sampler = FrameSampler::new(config, 30.0);

        for i in 0..5 {
            let f = VideoFrame::synthetic(i, 4, 4, 0);
            sampler.process_frame(&f);
        }
        assert_eq!(sampler.extracted_count(), 2);
    }

    #[test]
    fn test_frame_sampler_total_for_duration() {
        let config = FrameSamplerConfig {
            target_fps: 2.0,
            max_frames: 0,
            ..Default::default()
        };
        let sampler = FrameSampler::new(config, 30.0);
        assert_eq!(sampler.total_for_duration(10.0), 20);
    }

    #[test]
    fn test_frame_sampler_with_max() {
        let config = FrameSamplerConfig {
            target_fps: 2.0,
            max_frames: 5,
            ..Default::default()
        };
        let sampler = FrameSampler::new(config, 30.0);
        assert_eq!(sampler.total_for_duration(10.0), 5);
    }

    #[test]
    fn test_video_stream_reader() {
        let meta = VideoMetadata::new(8, 8, 30.0, 1.0);
        let config = FrameSamplerConfig {
            target_fps: 10.0,
            resize: false,
            ..Default::default()
        };
        let mut reader = VideoStreamReader::new(meta, config);

        for i in 0..30 {
            reader.push_frame(VideoFrame::synthetic(i, 8, 8, (i * 8) as u8));
        }
        reader.finish();

        assert!(reader.is_finished());
        assert_eq!(reader.buffered_count(), 10); // 30 frames at 10fps = 10
    }

    #[test]
    fn test_video_stream_reader_drain() {
        let meta = VideoMetadata::new(4, 4, 10.0, 1.0);
        let config = FrameSamplerConfig {
            target_fps: 10.0,
            resize: false,
            ..Default::default()
        };
        let mut reader = VideoStreamReader::new(meta, config);

        for i in 0..10 {
            reader.push_frame(VideoFrame::synthetic(i, 4, 4, 0));
        }

        let frames = reader.drain_frames();
        assert_eq!(frames.len(), 10);
        assert_eq!(reader.buffered_count(), 0);
    }

    #[test]
    fn test_temporal_frame_buffer() {
        let mut buf = TemporalFrameBuffer::new(3);
        assert!(buf.is_empty());

        buf.push(VideoFrame::synthetic(0, 4, 4, 0));
        buf.push(VideoFrame::synthetic(1, 4, 4, 0));
        assert_eq!(buf.len(), 2);
        assert!(!buf.is_full());

        buf.push(VideoFrame::synthetic(2, 4, 4, 0));
        assert!(buf.is_full());

        // Push when full: evicts oldest
        buf.push(VideoFrame::synthetic(3, 4, 4, 0));
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.frames()[0].frame_idx, 1); // Frame 0 was evicted
    }

    #[test]
    fn test_temporal_frame_buffer_flatten() {
        let mut buf = TemporalFrameBuffer::new(5);
        buf.push(VideoFrame::synthetic(0, 4, 4, 128));
        buf.push(VideoFrame::synthetic(1, 4, 4, 128));

        let (flat, num_frames, num_patches, patch_dim) = buf.flatten_for_temporal(2);
        assert_eq!(num_frames, 2);
        assert_eq!(num_patches, 4); // 2×2 grid
        assert_eq!(patch_dim, 12); // 2×2×3
        assert_eq!(flat.len(), 2 * 4 * 12);
    }

    #[test]
    fn test_temporal_frame_buffer_empty_flatten() {
        let buf = TemporalFrameBuffer::new(5);
        let (flat, _nf, _np, _pd) = buf.flatten_for_temporal(2);
        assert!(flat.is_empty());
    }
}
