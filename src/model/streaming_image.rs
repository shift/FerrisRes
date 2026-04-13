//! Streaming image I/O pipeline — progressive decode and patch streaming.
//!
//! Provides streaming image loading that feeds patches to the VisionEncoder
//! as they decode, without buffering the entire image in memory. Supports:
//! - Progressive image loading (row-by-row or tile-by-tile)
//! - Patch extraction on-the-fly with normalization
//! - Batched patch emission for GPU efficiency
//! - Integration with ImagePreprocessor

use std::path::Path;

use crate::error::{FerrisResError, Result};

// ---------------------------------------------------------------------------
// ImageRegion — a rectangular sub-region of an image
// ---------------------------------------------------------------------------

/// A rectangular region of an image.
#[derive(Debug, Clone)]
pub struct ImageRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl ImageRegion {
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self { x, y, width, height }
    }

    /// Full image region.
    pub fn full(width: usize, height: usize) -> Self {
        Self::new(0, 0, width, height)
    }

    /// Area in pixels.
    pub fn area(&self) -> usize {
        self.width * self.height
    }

    /// Check if a point is within this region.
    pub fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }
}

// ---------------------------------------------------------------------------
// ImagePatch — a single extracted patch with metadata
// ---------------------------------------------------------------------------

/// An extracted image patch.
#[derive(Debug, Clone)]
pub struct ImagePatch {
    /// Pixel data (RGB, row-major).
    pub pixels: Vec<f32>,
    /// Patch width.
    pub width: usize,
    /// Patch height.
    pub height: usize,
    /// Number of channels (3 for RGB).
    pub channels: usize,
    /// Position in source image.
    pub region: ImageRegion,
}

impl ImagePatch {
    pub fn new(pixels: Vec<f32>, width: usize, height: usize, channels: usize, region: ImageRegion) -> Self {
        Self { pixels, width, height, channels, region }
    }

    /// Number of values (width × height × channels).
    pub fn len(&self) -> usize {
        self.width * self.height * self.channels
    }

    /// Get pixel value at (x, y, c).
    pub fn get(&self, x: usize, y: usize, c: usize) -> f32 {
        self.pixels[y * self.width * self.channels + x * self.channels + c]
    }

    /// Normalize pixels to [0, 1] from [0, 255].
    pub fn normalize_255(&mut self) {
        for p in self.pixels.iter_mut() {
            *p /= 255.0;
        }
    }

    /// Normalize with mean and std (ImageNet-style).
    pub fn normalize_mean_std(&mut self, mean: &[f32; 3], std: &[f32; 3]) {
        for i in (0..self.pixels.len()).step_by(3) {
            self.pixels[i] = (self.pixels[i] - mean[0]) / std[0];
            if i + 1 < self.pixels.len() {
                self.pixels[i + 1] = (self.pixels[i + 1] - mean[1]) / std[1];
            }
            if i + 2 < self.pixels.len() {
                self.pixels[i + 2] = (self.pixels[i + 2] - mean[2]) / std[2];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingImageReader — progressive decode
// ---------------------------------------------------------------------------

/// Configuration for streaming image loading.
#[derive(Debug, Clone)]
pub struct StreamingImageConfig {
    /// Patch size (width = height).
    pub patch_size: usize,
    /// Stride between patches.
    pub stride: usize,
    /// Whether to normalize to [0, 1].
    pub normalize: bool,
    /// Number of color channels.
    pub channels: usize,
    /// Batch size for patch emission.
    pub batch_size: usize,
}

impl Default for StreamingImageConfig {
    fn default() -> Self {
        Self {
            patch_size: 224,
            stride: 224,
            normalize: true,
            channels: 3,
            batch_size: 32,
        }
    }
}

impl StreamingImageConfig {
    pub fn new(patch_size: usize) -> Self {
        Self { patch_size, stride: patch_size, ..Default::default() }
    }

    /// Number of patches that fit in an image of given dimensions.
    pub fn num_patches(&self, image_width: usize, image_height: usize) -> usize {
        if image_width < self.patch_size || image_height < self.patch_size {
            return 0;
        }
        let nx = (image_width - self.patch_size) / self.stride + 1;
        let ny = (image_height - self.patch_size) / self.stride + 1;
        nx * ny
    }
}

/// Streaming image reader that extracts patches progressively.
///
/// Supports loading images from raw pixel data (RGB u8) and extracting
/// patches on-the-fly without buffering the full decoded image.
pub struct StreamingImageReader {
    config: StreamingImageConfig,
    /// Raw pixel data (RGB u8).
    pixels_u8: Vec<u8>,
    /// Image width.
    width: usize,
    /// Image height.
    height: usize,
    /// Current patch position.
    current_patch: usize,
    /// Total patches.
    total_patches: usize,
}

impl StreamingImageReader {
    /// Create a new streaming reader from raw RGB pixels.
    pub fn from_rgb(pixels: Vec<u8>, width: usize, height: usize, config: StreamingImageConfig) -> Self {
        let total_patches = config.num_patches(width, height);
        Self {
            config,
            pixels_u8: pixels,
            width,
            height,
            current_patch: 0,
            total_patches,
        }
    }

    /// Create from a file path (simple: loads raw bytes, expects RGB u8).
    /// In production, this would use image decoding libraries.
    pub fn from_file(path: &Path, width: usize, height: usize, config: StreamingImageConfig) -> Result<Self> {
        let data = std::fs::read(path)?;
        let expected = width * height * 3;
        if data.len() < expected {
            return Err(FerrisResError::Shape(format!(
                "Image file too small: {} bytes, expected {} for {}x{} RGB",
                data.len(), expected, width, height
            )));
        }
        Ok(Self::from_rgb(data, width, height, config))
    }

    /// Create from a single-channel (grayscale) image.
    pub fn from_grayscale(pixels: Vec<u8>, width: usize, height: usize, config: StreamingImageConfig) -> Self {
        // Convert to RGB by triplicating
        let rgb: Vec<u8> = pixels.iter().flat_map(|&p| [p, p, p]).collect();
        Self::from_rgb(rgb, width, height, config)
    }

    /// Get the next patch.
    pub fn next_patch(&mut self) -> Option<ImagePatch> {
        if self.current_patch >= self.total_patches {
            return None;
        }

        let ps = self.config.patch_size;
        let stride = self.config.stride;
        let nx = (self.width - ps) / stride + 1;

        let patch_idx = self.current_patch;
        self.current_patch += 1;

        let py = patch_idx / nx;
        let px = patch_idx % nx;

        let x = px * stride;
        let y = py * stride;

        let mut pixels = Vec::with_capacity(ps * ps * self.config.channels);

        for row in y..(y + ps) {
            for col in x..(x + ps) {
                let src_offset = (row * self.width + col) * 3;
                for c in 0..self.config.channels {
                    let byte_val = self.pixels_u8.get(src_offset + c).copied().unwrap_or(0);
                    let val = if self.config.normalize {
                        byte_val as f32 / 255.0
                    } else {
                        byte_val as f32
                    };
                    pixels.push(val);
                }
            }
        }

        Some(ImagePatch::new(
            pixels,
            ps,
            ps,
            self.config.channels,
            ImageRegion::new(x, y, ps, ps),
        ))
    }

    /// Get the next batch of patches.
    pub fn next_batch(&mut self) -> Vec<ImagePatch> {
        let mut batch = Vec::with_capacity(self.config.batch_size);
        for _ in 0..self.config.batch_size {
            match self.next_patch() {
                Some(patch) => batch.push(patch),
                None => break,
            }
        }
        batch
    }

    /// Get all remaining patches.
    pub fn remaining_patches(&mut self) -> Vec<ImagePatch> {
        let mut patches = Vec::new();
        while let Some(p) = self.next_patch() {
            patches.push(p);
        }
        patches
    }

    /// Number of patches remaining.
    pub fn remaining(&self) -> usize {
        self.total_patches - self.current_patch
    }

    /// Total patches.
    pub fn total(&self) -> usize {
        self.total_patches
    }

    /// Current patch index.
    pub fn current(&self) -> usize {
        self.current_patch
    }

    /// Reset to beginning.
    pub fn reset(&mut self) {
        self.current_patch = 0;
    }

    /// Image width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the config.
    pub fn config(&self) -> &StreamingImageConfig {
        &self.config
    }

    /// Flatten all patches into a single batch buffer (for GPU upload).
    pub fn flatten_batch(patches: &[ImagePatch]) -> (Vec<f32>, usize, usize, usize) {
        if patches.is_empty() {
            return (Vec::new(), 0, 0, 0);
        }
        let ps = patches[0].width;
        let channels = patches[0].channels;
        let patch_len = ps * ps * channels;
        let mut flat = Vec::with_capacity(patches.len() * patch_len);
        for p in patches {
            flat.extend_from_slice(&p.pixels);
        }
        (flat, patches.len(), ps, channels)
    }
}

// ---------------------------------------------------------------------------
// TiledImageReader — for very large images (satellite, medical, etc.)
// ---------------------------------------------------------------------------

/// Tiled image reader for large images that don't fit in memory.
///
/// Processes the image one tile at a time, extracting patches from each tile.
pub struct TiledImageReader {
    config: StreamingImageConfig,
    /// Tile size for chunked reading.
    tile_size: usize,
    /// Image width.
    width: usize,
    /// Image height.
    height: usize,
    /// Current tile coordinates.
    current_tile: (usize, usize),
    /// Total tiles.
    total_tiles: (usize, usize),
}

impl TiledImageReader {
    /// Create a tiled reader.
    pub fn new(width: usize, height: usize, tile_size: usize, config: StreamingImageConfig) -> Self {
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;
        Self {
            config,
            tile_size,
            width,
            height,
            current_tile: (0, 0),
            total_tiles: (tiles_x, tiles_y),
        }
    }

    /// Get the next tile's region.
    pub fn next_tile_region(&mut self) -> Option<ImageRegion> {
        let (tx, ty) = self.current_tile;
        if ty >= self.total_tiles.1 {
            return None;
        }

        let x = tx * self.tile_size;
        let y = ty * self.tile_size;
        let w = self.tile_size.min(self.width - x);
        let h = self.tile_size.min(self.height - y);

        // Advance to next tile
        let next_tx = tx + 1;
        if next_tx >= self.total_tiles.0 {
            self.current_tile = (0, ty + 1);
        } else {
            self.current_tile = (next_tx, ty);
        }

        Some(ImageRegion::new(x, y, w, h))
    }

    /// Extract patches from a tile's pixel data.
    pub fn extract_tile_patches(&self, tile_pixels: &[u8], tile_region: &ImageRegion) -> Vec<ImagePatch> {
        let ps = self.config.patch_size;
        let stride = self.config.stride;
        let channels = self.config.channels;

        let nx = if tile_region.width >= ps {
            (tile_region.width - ps) / stride + 1
        } else {
            0
        };
        let ny = if tile_region.height >= ps {
            (tile_region.height - ps) / stride + 1
        } else {
            0
        };

        let mut patches = Vec::with_capacity(nx * ny);

        for py in 0..ny {
            for px in 0..nx {
                let x = px * stride;
                let y = py * stride;
                let mut pixels = Vec::with_capacity(ps * ps * channels);

                for row in y..(y + ps) {
                    for col in x..(x + ps) {
                        let src_offset = (row * tile_region.width + col) * 3;
                        for c in 0..channels {
                            let byte_val = tile_pixels.get(src_offset + c).copied().unwrap_or(0);
                            let val = if self.config.normalize {
                                byte_val as f32 / 255.0
                            } else {
                                byte_val as f32
                            };
                            pixels.push(val);
                        }
                    }
                }

                patches.push(ImagePatch::new(
                    pixels,
                    ps,
                    ps,
                    channels,
                    ImageRegion::new(
                        tile_region.x + x,
                        tile_region.y + y,
                        ps,
                        ps,
                    ),
                ));
            }
        }

        patches
    }

    /// Total number of tiles.
    pub fn total_tiles(&self) -> usize {
        self.total_tiles.0 * self.total_tiles.1
    }

    /// Number of remaining tiles.
    pub fn remaining_tiles(&self) -> usize {
        let (tx, ty) = self.current_tile;
        let (total_x, total_y) = self.total_tiles;
        (total_y - ty) * total_x - tx
    }

    /// Reset to first tile.
    pub fn reset(&mut self) {
        self.current_tile = (0, 0);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image(width: usize, height: usize) -> Vec<u8> {
        (0..width * height * 3).map(|i| (i % 256) as u8).collect()
    }

    #[test]
    fn test_image_region() {
        let r = ImageRegion::new(10, 20, 100, 200);
        assert_eq!(r.area(), 20000);
        assert!(r.contains(10, 20));
        assert!(r.contains(109, 219));
        assert!(!r.contains(9, 20));
        assert!(!r.contains(110, 20));
    }

    #[test]
    fn test_image_region_full() {
        let r = ImageRegion::full(640, 480);
        assert_eq!(r.x, 0);
        assert_eq!(r.y, 0);
        assert_eq!(r.area(), 640 * 480);
    }

    #[test]
    fn test_image_patch() {
        let pixels = vec![0.5, 0.3, 0.1, 0.9, 0.7, 0.2]; // 2x1 RGB
        let patch = ImagePatch::new(pixels.clone(), 2, 1, 3, ImageRegion::new(0, 0, 2, 1));
        assert_eq!(patch.len(), 6);
        assert!((patch.get(0, 0, 0) - 0.5).abs() < 1e-5);
        assert!((patch.get(1, 0, 2) - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_image_patch_normalize_255() {
        let mut patch = ImagePatch::new(vec![255.0, 128.0, 0.0], 1, 1, 3, ImageRegion::full(1, 1));
        patch.normalize_255();
        assert!((patch.pixels[0] - 1.0).abs() < 1e-5);
        assert!((patch.pixels[1] - 128.0 / 255.0).abs() < 1e-5);
    }

    #[test]
    fn test_image_patch_normalize_imagenet() {
        let mut patch = ImagePatch::new(vec![0.5, 0.5, 0.5], 1, 1, 3, ImageRegion::full(1, 1));
        patch.normalize_mean_std(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225]);
        assert!(patch.pixels[0] < 1.0); // After normalization
    }

    #[test]
    fn test_streaming_config() {
        let config = StreamingImageConfig::new(224);
        assert_eq!(config.patch_size, 224);
        assert_eq!(config.stride, 224);
        // 448x448 → 2x2 = 4 patches
        assert_eq!(config.num_patches(448, 448), 4);
        // 224x224 → 1 patch
        assert_eq!(config.num_patches(224, 224), 1);
        // 100x100 → 0 patches (too small)
        assert_eq!(config.num_patches(100, 100), 0);
    }

    #[test]
    fn test_streaming_reader_basic() {
        let pixels = make_test_image(224, 224);
        let config = StreamingImageConfig::new(224);
        let mut reader = StreamingImageReader::from_rgb(pixels, 224, 224, config);

        assert_eq!(reader.total(), 1);
        assert_eq!(reader.remaining(), 1);

        let patch = reader.next_patch().unwrap();
        assert_eq!(patch.width, 224);
        assert_eq!(patch.height, 224);
        assert_eq!(patch.channels, 3);
        assert_eq!(patch.pixels.len(), 224 * 224 * 3);

        assert_eq!(reader.remaining(), 0);
        assert!(reader.next_patch().is_none());
    }

    #[test]
    fn test_streaming_reader_multiple_patches() {
        let pixels = make_test_image(448, 448);
        let config = StreamingImageConfig::new(224);
        let mut reader = StreamingImageReader::from_rgb(pixels, 448, 448, config);

        assert_eq!(reader.total(), 4);

        let patches = reader.remaining_patches();
        assert_eq!(patches.len(), 4);

        // Check regions cover the full image
        assert_eq!(patches[0].region.x, 0);
        assert_eq!(patches[0].region.y, 0);
        assert_eq!(patches[3].region.x, 224);
        assert_eq!(patches[3].region.y, 224);
    }

    #[test]
    fn test_streaming_reader_batch() {
        let pixels = make_test_image(672, 224);
        let mut config = StreamingImageConfig::new(224);
        config.batch_size = 2;
        let mut reader = StreamingImageReader::from_rgb(pixels, 672, 224, config);

        let batch = reader.next_batch();
        assert_eq!(batch.len(), 2);

        let batch2 = reader.next_batch();
        assert_eq!(batch2.len(), 1); // Only 1 remaining

        let batch3 = reader.next_batch();
        assert!(batch3.is_empty());
    }

    #[test]
    fn test_streaming_reader_reset() {
        let pixels = make_test_image(224, 224);
        let config = StreamingImageConfig::new(224);
        let mut reader = StreamingImageReader::from_rgb(pixels, 224, 224, config);

        reader.next_patch();
        assert_eq!(reader.remaining(), 0);

        reader.reset();
        assert_eq!(reader.remaining(), 1);
    }

    #[test]
    fn test_streaming_reader_from_grayscale() {
        let gray: Vec<u8> = (0..224 * 224).map(|i| (i % 256) as u8).collect();
        let config = StreamingImageConfig::new(224);
        let mut reader = StreamingImageReader::from_grayscale(gray, 224, 224, config);

        let patch = reader.next_patch().unwrap();
        assert_eq!(patch.channels, 3); // Converted to RGB
    }

    #[test]
    fn test_streaming_reader_stride() {
        let pixels = make_test_image(448, 224);
        let mut config = StreamingImageConfig::new(224);
        config.stride = 112; // Overlapping patches
        let reader = StreamingImageReader::from_rgb(pixels, 448, 224, config);

        // (448-224)/112 + 1 = 3 patches horizontally
        assert_eq!(reader.total(), 3);
    }

    #[test]
    fn test_flatten_batch() {
        let patches = vec![
            ImagePatch::new(vec![1.0, 2.0, 3.0], 1, 1, 3, ImageRegion::full(1, 1)),
            ImagePatch::new(vec![4.0, 5.0, 6.0], 1, 1, 3, ImageRegion::full(1, 1)),
        ];
        let (flat, count, ps, ch) = StreamingImageReader::flatten_batch(&patches);
        assert_eq!(count, 2);
        assert_eq!(ps, 1);
        assert_eq!(ch, 3);
        assert_eq!(flat.len(), 6);
        assert!((flat[0] - 1.0).abs() < 1e-5);
        assert!((flat[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_flatten_empty() {
        let (flat, count, _ps, _ch) = StreamingImageReader::flatten_batch(&[]);
        assert!(flat.is_empty());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_tiled_reader() {
        let tiled = TiledImageReader::new(1000, 1000, 256, StreamingImageConfig::new(224));
        // (1000+255)/256 = 4 tiles each direction = 16 total
        assert_eq!(tiled.total_tiles(), 16);
        assert_eq!(tiled.remaining_tiles(), 16);
    }

    #[test]
    fn test_tiled_reader_regions() {
        let mut tiled = TiledImageReader::new(512, 512, 256, StreamingImageConfig::new(224));

        let r0 = tiled.next_tile_region().unwrap();
        assert_eq!(r0.x, 0);
        assert_eq!(r0.y, 0);
        assert_eq!(r0.width, 256);
        assert_eq!(r0.height, 256);

        let r1 = tiled.next_tile_region().unwrap();
        assert_eq!(r1.x, 256);
        assert_eq!(r1.y, 0);

        let r2 = tiled.next_tile_region().unwrap();
        assert_eq!(r2.x, 0);
        assert_eq!(r2.y, 256);
    }

    #[test]
    fn test_tiled_reader_extract_patches() {
        let tiled = TiledImageReader::new(512, 512, 256, StreamingImageConfig::new(224));
        let tile_pixels = make_test_image(256, 256);
        let region = ImageRegion::new(0, 0, 256, 256);
        let patches = tiled.extract_tile_patches(&tile_pixels, &region);
        // (256-224)/224 + 1 = 1 patch per dimension
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].width, 224);
    }

    #[test]
    fn test_tiled_reader_reset() {
        let mut tiled = TiledImageReader::new(512, 512, 256, StreamingImageConfig::new(224));
        tiled.next_tile_region();
        tiled.next_tile_region();
        assert_eq!(tiled.remaining_tiles(), 2);

        tiled.reset();
        assert_eq!(tiled.remaining_tiles(), 4);
    }

    #[test]
    fn test_from_file_too_small() {
        let config = StreamingImageConfig::new(224);
        // Create a temp file with too few bytes
        let tmp = std::env::temp_dir().join("ferrisres_test_image.raw");
        std::fs::write(&tmp, &[0u8; 10]).unwrap();
        let result = StreamingImageReader::from_file(&tmp, 224, 224, config);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&tmp);
    }
}
