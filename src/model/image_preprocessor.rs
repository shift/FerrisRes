use crate::error::{Result, FerrisResError};

pub struct ImagePreprocessor {
    pub target_height: u32,
    pub target_width: u32,
    pub normalize: bool,
}

impl ImagePreprocessor {
    pub fn new(target_height: u32, target_width: u32, normalize: bool) -> Self {
        Self {
            target_height,
            target_width,
            normalize,
        }
    }

    pub fn preprocess(&self, image_data: &[u8]) -> Result<Vec<f32>> {
        let img = image::load_from_memory(image_data)
            .map_err(|e| FerrisResError::Unsupported(format!("Failed to decode image: {e}")))?;

        let resized = img.resize_exact(
            self.target_width,
            self.target_height,
            image::imageops::FilterType::Lanczos3,
        );

        let rgb = resized.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let mut tensor = Vec::with_capacity(h * w * 3);

        for pixel in rgb.pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            tensor.push(r);
            tensor.push(g);
            tensor.push(b);
        }

        if self.normalize {
            let clip_mean = [0.48145466, 0.4578275, 0.40821073];
            let clip_std = [0.26862954, 0.26130258, 0.27577711];
            for i in 0..tensor.len() {
                let c = i % 3;
                tensor[i] = (tensor[i] - clip_mean[c]) / clip_std[c];
            }
        }

        Ok(tensor)
    }
}
