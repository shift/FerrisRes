//! Safetensors model weight loader.
//!
//! Parses the HuggingFace safetensors format and loads weights into
//! FerrisRes model components. Supports FP32, FP16, BF16 dtypes with
//! automatic conversion to f32 for GPU upload.
//!
//! Format reference: https://github.com/huggingface/safetensors
//!
//! File layout:
//!   [8 bytes: header_length as little-endian u64]
//!   [header_length bytes: JSON metadata]
//!   [remaining bytes: raw tensor data]
//!
//! JSON metadata maps tensor names to {dtype, shape, data_offsets}.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use wgpu::{Device, Queue, BufferDescriptor, BufferUsages};

use crate::compute::GpuBuffer;
use crate::error::{FerrisResError, Result};

// ---------------------------------------------------------------------------
// Safetensors file format types
// ---------------------------------------------------------------------------

/// Tensor metadata entry in the safetensors JSON header.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Parsed safetensors header.
#[derive(Debug, Clone, Deserialize)]
pub struct SafetensorsHeader {
    /// Map from tensor name to metadata.
    /// We parse as HashMap<String, TensorMeta> but the actual JSON may
    /// contain a "__metadata__" key at the top level that we skip.
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorMeta>,
}

/// A single tensor loaded from a safetensors file.
#[derive(Debug)]
pub struct LoadedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: Vec<f32>,
}

/// Result of loading one or more safetensors files.
#[derive(Debug)]
pub struct LoadedWeights {
    pub tensors: HashMap<String, LoadedTensor>,
    pub source_files: Vec<PathBuf>,
}

// ---------------------------------------------------------------------------
// Dtype handling
// ---------------------------------------------------------------------------

/// Number of bytes per element for a given safetensors dtype.
fn dtype_size(dtype: &str) -> Result<usize> {
    match dtype {
        "F32" => Ok(4),
        "F16" | "BF16" => Ok(2),
        "F8_E4M3" | "F8_E5M2" => Ok(1),
        "I8" => Ok(1),
        "I16" => Ok(2),
        "I32" => Ok(4),
        "I64" => Ok(8),
        "U8" => Ok(1),
        "BOOL" => Ok(1),
        other => Err(FerrisResError::Unsupported(format!(
            "Unsupported safetensors dtype: {}", other
        ))),
    }
}

/// Convert raw bytes to Vec<f32> based on dtype.
fn bytes_to_f32(raw: &[u8], dtype: &str) -> Result<Vec<f32>> {
    match dtype {
        "F32" => {
            if raw.len() % 4 != 0 {
                return Err(FerrisResError::Shape(format!(
                    "F32 data length {} not aligned to 4 bytes", raw.len()
                )));
            }
            Ok(bytemuck::cast_slice(raw).to_vec())
        }
        "F16" => {
            if raw.len() % 2 != 0 {
                return Err(FerrisResError::Shape(format!(
                    "F16 data length {} not aligned to 2 bytes", raw.len()
                )));
            }
            // Convert f16 to f32: use half crate if available, otherwise
            // manual conversion
            let halfs: Vec<u16> = raw.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            Ok(halfs.iter().map(|&h| f16_to_f32(h)).collect())
        }
        "BF16" => {
            if raw.len() % 2 != 0 {
                return Err(FerrisResError::Shape(format!(
                    "BF16 data length {} not aligned to 2 bytes", raw.len()
                )));
            }
            let halfs: Vec<u16> = raw.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            Ok(halfs.iter().map(|&h| bf16_to_f32(h)).collect())
        }
        other => Err(FerrisResError::Unsupported(format!(
            "Cannot convert dtype {} to f32 yet", other
        ))),
    }
}

/// Convert IEEE f16 (half precision) to f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;

    if exp == 0 {
        if mant == 0 {
            // ±zero
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal: (-1)^sign × 2^(-14) × (0.mantissa)
        let val = (mant as f32) * 2f32.powi(-24);
        return if sign == 1 { -val } else { val };
    }

    if exp == 31 {
        if mant == 0 {
            return if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
        }
        return f32::NAN;
    }

    // Normalized: (-1)^sign × 2^(exp-15) × (1 + mant/1024)
    let val = (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15);
    if sign == 1 { -val } else { val }
}

/// Convert BF16 (bfloat16) to f32.
/// BF16 has the same exponent range as FP32 but only 7 mantissa bits.
fn bf16_to_f32(h: u16) -> f32 {
    // BF16 is the upper 16 bits of an FP32 number
    let bits = (h as u32) << 16;
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// File parsing
// ---------------------------------------------------------------------------

/// Parse the header of a single safetensors file.
fn parse_header(reader: &mut impl Read) -> Result<(SafetensorsHeader, usize)> {
    // Read 8-byte header length
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    // Read header JSON
    let mut header_buf = vec![0u8; header_len];
    reader.read_exact(&mut header_buf)?;
    let header_str = std::str::from_utf8(&header_buf)
        .map_err(|e| FerrisResError::Shape(format!("Invalid safetensors header UTF-8: {}", e)))?;

    // Parse — skip the __metadata__ key
    let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
        .map_err(|e| FerrisResError::Shape(format!("Invalid safetensors header JSON: {}", e)))?;

    let mut tensors = HashMap::new();
    for (key, value) in raw {
        if key == "__metadata__" {
            continue;
        }
        let meta: TensorMeta = serde_json::from_value(value)
            .map_err(|e| FerrisResError::Shape(format!(
                "Invalid tensor metadata for '{}': {}", key, e
            )))?;
        tensors.insert(key, meta);
    }

    Ok((SafetensorsHeader { tensors }, 8 + header_len))
}

/// Load all tensors from a single safetensors file.
fn load_file(path: &Path) -> Result<Vec<LoadedTensor>> {
    let mut file = BufReader::new(File::open(path)?);
    let (header, data_offset) = parse_header(&mut file)?;

    let file_len = file.get_ref().metadata()?.len() as usize;

    let mut results = Vec::new();
    for (name, meta) in &header.tensors {
        let (start, end) = meta.data_offsets;
        let byte_len = end - start;
        let element_count: usize = meta.shape.iter().product();

        // Validate size
        let expected_bytes = element_count * dtype_size(&meta.dtype)?;
        if byte_len != expected_bytes {
            return Err(FerrisResError::Shape(format!(
                "Tensor '{}': expected {} bytes ({} elements × {} bytes/{}), got {}",
                name, expected_bytes, element_count, meta.dtype, dtype_size(&meta.dtype)?, byte_len
            )));
        }

        // Validate offsets
        if data_offset + end > file_len {
            return Err(FerrisResError::Shape(format!(
                "Tensor '{}': data offsets [{}, {}) exceed file size {}",
                name, start, end, file_len
            )));
        }

        // Read raw bytes
        file.seek(SeekFrom::Start((data_offset + start) as u64))?;
        let mut raw = vec![0u8; byte_len];
        file.read_exact(&mut raw)?;

        // Convert to f32
        let data = bytes_to_f32(&raw, &meta.dtype)?;

        results.push(LoadedTensor {
            name: name.clone(),
            shape: meta.shape.clone(),
            dtype: meta.dtype.clone(),
            data,
        });
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load safetensors weights from a single file or a sharded directory.
///
/// For sharded models, pass a glob pattern like `model-*.safetensors` or
/// the index file path. For single-file models, pass the .safetensors path.
pub fn load_safetensors(path: &Path) -> Result<LoadedWeights> {
    let mut all_tensors = HashMap::new();
    let mut source_files = Vec::new();

    if path.is_dir() {
        // Look for sharded files: model-00001-of-000XX.safetensors
        let pattern = path.join("model-*-of-*.safetensors");
        let _pattern_str = pattern.to_string_lossy();

        let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| FerrisResError::Shape(format!("Cannot read directory: {}", e)))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().unwrap_or_default().to_string_lossy();
                name.starts_with("model-") && name.ends_with(".safetensors")
            })
            .collect();
        shard_paths.sort();

        if shard_paths.is_empty() {
            // Try single file in directory
            let single = path.join("model.safetensors");
            if single.exists() {
                shard_paths.push(single);
            } else {
                return Err(FerrisResError::Shape(format!(
                    "No safetensors files found in {}", path.display()
                )));
            }
        }

        for shard_path in &shard_paths {
            let tensors = load_file(shard_path)?;
            for t in tensors {
                all_tensors.insert(t.name.clone(), t);
            }
            source_files.push(shard_path.clone());
        }
    } else {
        // Single file
        let tensors = load_file(path)?;
        for t in tensors {
            all_tensors.insert(t.name.clone(), t);
        }
        source_files.push(path.to_path_buf());
    }

    tracing::info!(
        "Loaded {} tensors from {} safetensors file(s)",
        all_tensors.len(),
        source_files.len()
    );

    Ok(LoadedWeights {
        tensors: all_tensors,
        source_files,
    })
}

impl LoadedWeights {
    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<&LoadedTensor> {
        self.tensors.get(name)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Upload a tensor to the GPU as a GpuBuffer.
    pub fn upload_to_gpu(
        &self,
        name: &str,
        device: &Arc<Device>,
        queue: &Arc<Queue>,
    ) -> Result<GpuBuffer> {
        let tensor = self.tensors.get(name)
            .ok_or_else(|| FerrisResError::Shape(format!(
                "Tensor '{}' not found in loaded weights", name
            )))?;

        let byte_size = tensor.data.len() * std::mem::size_of::<f32>();
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(name),
            size: byte_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&tensor.data));

        Ok(GpuBuffer::from_existing(buffer, byte_size))
    }

    /// Validate that the loaded weights contain expected tensor names.
    pub fn validate(&self, expected: &[&str]) -> Result<Vec<String>> {
        let mut missing = Vec::new();
        for name in expected {
            if !self.tensors.contains_key(*name) {
                missing.push(name.to_string());
            }
        }
        if missing.is_empty() {
            Ok(missing)
        } else {
            Err(FerrisResError::Shape(format!(
                "Missing {} tensor(s): {}",
                missing.len(),
                missing.join(", ")
            )))
        }
    }

    /// Total number of parameters across all tensors.
    pub fn total_parameters(&self) -> usize {
        self.tensors.values().map(|t| t.data.len()).sum()
    }

    /// Detect the model architecture from tensor names.
    pub fn detect_architecture(&self) -> ModelArchitecture {
        let names = self.tensor_names();

        // Check for BlockAttnRes-specific names
        if names.iter().any(|n| n.contains("q_proj") && n.contains("block_attn")) {
            return ModelArchitecture::BlockAttnRes;
        }

        // Check for standard transformer names (LLaMA, Mistral, etc.)
        let has_attn_q = names.iter().any(|n| n.contains("q_proj") || n.contains("query"));
        let has_attn_k = names.iter().any(|n| n.contains("k_proj") || n.contains("key"));
        let has_attn_v = names.iter().any(|n| n.contains("v_proj") || n.contains("value"));

        if has_attn_q && has_attn_k && has_attn_v {
            // Detect specific family
            if names.iter().any(|n| n.starts_with("model.layers.")) {
                if names.iter().any(|n| n.contains("gate_proj")) {
                    return ModelArchitecture::Llama;
                }
                return ModelArchitecture::Mistral;
            }
            if names.iter().any(|n| n.starts_with("transformer.h.")) {
                return ModelArchitecture::GptNeoX;
            }
            return ModelArchitecture::Standard;
        }

        ModelArchitecture::Unknown
    }

    /// Infer the number of layers from tensor names.
    pub fn infer_num_layers(&self) -> usize {
        let names = self.tensor_names();
        let mut max_layer = 0;
        for name in names {
            // model.layers.N.q_proj.weight
            if let Some(idx) = name.find("layers.") {
                let rest = &name[idx + 7..];
                if let Some(dot) = rest.find('.') {
                    if let Ok(n) = rest[..dot].parse::<usize>() {
                        max_layer = max_layer.max(n + 1);
                    }
                }
            }
            // transformer.h.N
            if let Some(idx) = name.find("h.") {
                let rest = &name[idx + 2..];
                if let Some(dot) = rest.find('.') {
                    if let Ok(n) = rest[..dot].parse::<usize>() {
                        max_layer = max_layer.max(n + 1);
                    }
                }
            }
        }
        max_layer
    }

    /// Infer hidden dimension from an embedding tensor.
    pub fn infer_hidden_dim(&self) -> Option<usize> {
        // Look for embedding tensors
        for name in &["model.embed_tokens.weight", "transformer.wte.weight", "embedding.weight"] {
            if let Some(t) = self.get(name) {
                if t.shape.len() == 2 {
                    return Some(t.shape[1]);
                }
            }
        }
        None
    }

    /// Infer vocabulary size from an embedding tensor.
    pub fn infer_vocab_size(&self) -> Option<usize> {
        for name in &["model.embed_tokens.weight", "transformer.wte.weight", "embedding.weight"] {
            if let Some(t) = self.get(name) {
                if t.shape.len() == 2 {
                    return Some(t.shape[0]);
                }
            }
        }
        None
    }
}

/// Detected model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    BlockAttnRes,
    Llama,
    Mistral,
    GptNeoX,
    Standard,
    Unknown,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BlockAttnRes => write!(f, "BlockAttnRes"),
            Self::Llama => write!(f, "LLaMA"),
            Self::Mistral => write!(f, "Mistral"),
            Self::GptNeoX => write!(f, "GPT-NeoX"),
            Self::Standard => write!(f, "Standard"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Detect architecture from a list of tensor names (shared logic).
pub fn detect_architecture_from_names(names: &[&str]) -> ModelArchitecture {
    // Check for BlockAttnRes-specific names
    if names.iter().any(|n| n.contains("q_proj") && n.contains("block_attn")) {
        return ModelArchitecture::BlockAttnRes;
    }

    let has_attn_q = names.iter().any(|n| n.contains("q_proj") || n.contains("query"));
    let has_attn_k = names.iter().any(|n| n.contains("k_proj") || n.contains("key"));
    let has_attn_v = names.iter().any(|n| n.contains("v_proj") || n.contains("value"));

    if has_attn_q && has_attn_k && has_attn_v {
        if names.iter().any(|n| n.starts_with("model.layers.")) {
            if names.iter().any(|n| n.contains("gate_proj")) {
                return ModelArchitecture::Llama;
            }
            return ModelArchitecture::Mistral;
        }
        if names.iter().any(|n| n.starts_with("transformer.h.")) {
            return ModelArchitecture::GptNeoX;
        }
        return ModelArchitecture::Standard;
    }

    ModelArchitecture::Unknown
}

/// Infer number of layers from tensor names (shared logic).
pub fn infer_layers_from_names(names: &[&str]) -> usize {
    let mut max_layer = 0;
    for name in names {
        if let Some(idx) = name.find("layers.") {
            let rest = &name[idx + 7..];
            if let Some(dot) = rest.find('.') {
                if let Ok(n) = rest[..dot].parse::<usize>() {
                    max_layer = max_layer.max(n + 1);
                }
            }
        }
        if let Some(idx) = name.find("h.") {
            let rest = &name[idx + 2..];
            if let Some(dot) = rest.find('.') {
                if let Ok(n) = rest[..dot].parse::<usize>() {
                    max_layer = max_layer.max(n + 1);
                }
            }
        }
    }
    max_layer
}

// ---------------------------------------------------------------------------
// Safetensors Writer
// ---------------------------------------------------------------------------

/// A tensor to be written to a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorToWrite {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: SafeDtype,
    pub data_f32: Vec<f32>,
}

/// Supported output dtypes for safetensors writing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeDtype {
    F32,
    F16,
    BF16,
}

impl SafeDtype {
    /// Dtype string as used in safetensors JSON header.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }

    /// Bytes per element.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// Convert f32 data to raw bytes for the given dtype.
fn f32_to_bytes(data: &[f32], dtype: SafeDtype) -> Vec<u8> {
    match dtype {
        SafeDtype::F32 => data.iter().flat_map(|f| f.to_le_bytes()).collect(),
        SafeDtype::F16 => data.iter().map(|&f| f32_to_f16_bits(f)).flat_map(|h| h.to_le_bytes()).collect(),
        SafeDtype::BF16 => data.iter().map(|&f| f32_to_bf16_bits(f)).flat_map(|h| h.to_le_bytes()).collect(),
    }
}

/// Convert f32 to f16 bits (IEEE 754 half precision).
fn f32_to_f16_bits(val: f32) -> u16 {
    if val == 0.0 {
        return if val.is_sign_negative() { 0x8000 } else { 0x0000 };
    }
    if val.is_nan() {
        return if val.is_sign_negative() { 0xFFFF } else { 0x7FFF };
    }
    if val.is_infinite() {
        return if val.is_sign_negative() { 0xFC00 } else { 0x7C00 };
    }
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        // Subnormal or zero
        let shift = 1 - new_exp;
        let new_mant = ((0x400000 + mant) >> (shift + 1)) & 0x3FF;
        (sign as u16) << 15 | new_mant as u16
    } else if new_exp >= 31 {
        // Overflow → infinity
        (sign as u16) << 15 | 0x7C00
    } else {
        (sign as u16) << 15 | (new_exp as u16) << 10 | ((mant >> 13) as u16)
    }
}

/// Convert f32 to BF16 bits (bfloat16).
fn f32_to_bf16_bits(val: f32) -> u16 {
    // BF16 is the upper 16 bits of an FP32 number
    (val.to_bits() >> 16) as u16
}

/// Write tensors to a safetensors file.
///
/// Produces standard safetensors format:
///   [8 bytes: header_length as little-endian u64]
///   [header_length bytes: JSON metadata]
///   [remaining bytes: raw tensor data]
pub fn write_safetensors(path: &Path, tensors: &[TensorToWrite]) -> Result<()> {
    let num = |v: u64| -> serde_json::Value {
        serde_json::Value::Number(serde_json::Number::from(v))
    };

    let mut header_json = serde_json::Map::new();
    let mut data_blob = Vec::new();

    // Build header and serialize data in order
    for tensor in tensors {
        let start = data_blob.len();
        let byte_data = f32_to_bytes(&tensor.data_f32, tensor.dtype);
        data_blob.extend_from_slice(&byte_data);
        let end = data_blob.len();

        let mut meta = serde_json::Map::new();
        meta.insert("dtype".to_string(), serde_json::Value::String(tensor.dtype.as_str().to_string()));
        meta.insert(
            "shape".to_string(),
            serde_json::Value::Array(tensor.shape.iter().map(|&d| num(d as u64)).collect()),
        );
        meta.insert(
            "data_offsets".to_string(),
            serde_json::Value::Array(vec![num(start as u64), num(end as u64)]),
        );
        header_json.insert(tensor.name.clone(), serde_json::Value::Object(meta));
    }

    let header_str = serde_json::to_string(&header_json)
        .map_err(|e| FerrisResError::Shape(format!("Failed to serialize safetensors header: {}", e)))?;
    let header_bytes = header_str.as_bytes();
    let header_len = header_bytes.len() as u64;

    use std::io::Write;
    let mut file = File::create(path)
        .map_err(|e| FerrisResError::Shape(format!("Failed to create safetensors file: {}", e)))?;
    file.write_all(&header_len.to_le_bytes())
        .map_err(|e| FerrisResError::Shape(format!("Failed to write header length: {}", e)))?;
    file.write_all(header_bytes)
        .map_err(|e| FerrisResError::Shape(format!("Failed to write header: {}", e)))?;
    file.write_all(&data_blob)
        .map_err(|e| FerrisResError::Shape(format!("Failed to write tensor data: {}", e)))?;

    tracing::info!(
        "Wrote {} tensors ({} bytes) to {}",
        tensors.len(),
        data_blob.len(),
        path.display()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
        assert_eq!(f16_to_f32(0x8000), -0.0f32);
    }

    #[test]
    fn test_f16_to_f32_one() {
        // 1.0 in f16: sign=0, exp=15 (bias), mant=0
        // bits: 0_01111_0000000000 = 0x3C00
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_half() {
        // 0.5 in f16: 0_01110_0000000000 = 0x3800
        let val = f16_to_f32(0x3800);
        assert!((val - 0.5f32).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_negative() {
        // -2.0 in f16: 1_10000_0000000000 = 0xC000
        let val = f16_to_f32(0xC000);
        assert!((val - (-2.0f32)).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_one() {
        // 1.0 in BF16: 0x3F80 (same as FP32 upper 16 bits of 0x3F800000)
        let val = bf16_to_f32(0x3F80);
        assert!((val - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_negative() {
        // -1.0 in BF16: 0xBF80
        let val = bf16_to_f32(0xBF80);
        assert!((val - (-1.0f32)).abs() < 1e-6);
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(dtype_size("F32").unwrap(), 4);
        assert_eq!(dtype_size("F16").unwrap(), 2);
        assert_eq!(dtype_size("BF16").unwrap(), 2);
        assert_eq!(dtype_size("I8").unwrap(), 1);
        assert!(dtype_size("INVALID").is_err());
    }

    #[test]
    fn test_parse_header_minimal() {
        // Build a minimal safetensors header
        let meta = serde_json::json!({
            "test.weight": {
                "dtype": "F32",
                "shape": [2, 3],
                "data_offsets": [0, 24]
            }
        });
        let header_bytes = serde_json::to_vec(&meta).unwrap();
        let header_len = header_bytes.len() as u64;

        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(&header_bytes);
        data.extend_from_slice(&[0u8; 24]); // dummy tensor data

        let mut cursor = std::io::Cursor::new(&data);
        let (header, data_start) = parse_header(&mut cursor).unwrap();

        assert_eq!(header.tensors.len(), 1);
        assert!(header.tensors.contains_key("test.weight"));
        let t = &header.tensors["test.weight"];
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data_offsets, (0, 24));
        assert_eq!(data_start, 8 + header_bytes.len());
    }

    #[test]
    fn test_model_architecture_display() {
        assert_eq!(ModelArchitecture::Llama.to_string(), "LLaMA");
        assert_eq!(ModelArchitecture::BlockAttnRes.to_string(), "BlockAttnRes");
        assert_eq!(ModelArchitecture::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn test_bytes_to_f32() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let raw: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let result = bytes_to_f32(&raw, "F32").unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_write_safetensors_round_trip() {
        let dir = std::env::temp_dir().join("ferrisres_writer_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_write.safetensors");

        let tensors = vec![
            TensorToWrite {
                name: "layer1.weight".to_string(),
                shape: vec![3, 4],
                dtype: SafeDtype::F32,
                data_f32: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            },
            TensorToWrite {
                name: "layer1.bias".to_string(),
                shape: vec![4],
                dtype: SafeDtype::F32,
                data_f32: vec![0.1, 0.2, 0.3, 0.4],
            },
        ];

        write_safetensors(&path, &tensors).unwrap();
        assert!(path.exists());

        // Load back and verify
        let loaded = load_safetensors(&path).unwrap();
        let w = loaded.get("layer1.weight").unwrap();
        assert_eq!(w.shape, vec![3, 4]);
        assert_eq!(w.data.len(), 12);
        assert!((w.data[0] - 1.0).abs() < 1e-6);
        assert!((w.data[11] - 12.0).abs() < 1e-6);

        let b = loaded.get("layer1.bias").unwrap();
        assert_eq!(b.shape, vec![4]);
        assert!((b.data[0] - 0.1).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_safetensors_f16_round_trip() {
        let dir = std::env::temp_dir().join("ferrisres_writer_f16");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_f16.safetensors");

        let original = vec![0.0, 1.0, -1.0, 0.5, 2.0];
        let tensors = vec![
            TensorToWrite {
                name: "test".to_string(),
                shape: vec![5],
                dtype: SafeDtype::F16,
                data_f32: original.clone(),
            },
        ];

        write_safetensors(&path, &tensors).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let t = loaded.get("test").unwrap();
        // F16 has limited precision (~3 decimal digits)
        for (i, (orig, got)) in original.iter().zip(t.data.iter()).enumerate() {
            assert!((orig - got).abs() < 0.01, "Mismatch at {}: {} vs {}", i, orig, got);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_safetensors_bf16_round_trip() {
        let dir = std::env::temp_dir().join("ferrisres_writer_bf16");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_bf16.safetensors");

        let original = vec![0.0, 1.0, -1.0, 0.5, 100.0];
        let tensors = vec![
            TensorToWrite {
                name: "test".to_string(),
                shape: vec![5],
                dtype: SafeDtype::BF16,
                data_f32: original.clone(),
            },
        ];

        write_safetensors(&path, &tensors).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let t = loaded.get("test").unwrap();
        // BF16 has even less mantissa precision
        for (i, (orig, got)) in original.iter().zip(t.data.iter()).enumerate() {
            assert!((orig - got).abs() < 0.1, "Mismatch at {}: {} vs {}", i, orig, got);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_f32_to_f16_round_trip() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 2.0, 0.333, 65504.0];
        for v in &values {
            let bits = f32_to_f16_bits(*v);
            let recovered = f16_to_f32(bits);
            assert!((v - recovered).abs() < 0.01, "F16 round trip failed: {} -> {} -> {}", v, bits, recovered);
        }
    }

    #[test]
    fn test_f32_to_bf16_round_trip() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 2.0, 100.0, -1000.0];
        for v in &values {
            let bits = f32_to_bf16_bits(*v);
            let recovered = bf16_to_f32(bits);
            assert!((v - recovered).abs() < 0.1, "BF16 round trip failed: {} -> {} -> {}", v, bits, recovered);
        }
    }
}

// ---------------------------------------------------------------------------
// Memory-mapped safetensors loader (for large models that don't fit in RAM)
// ---------------------------------------------------------------------------

/// Memory-mapped safetensors file. Keeps the file mmap'd and converts
/// individual tensors to f32 on demand, avoiding the need to hold all
/// tensors in RAM simultaneously.
pub struct MmapedSafetensors {
    /// The mmap'd file data.
    #[allow(dead_code)]
    mmap: memmap2::Mmap,
    /// Parsed header: tensor name → (dtype, shape, start_offset, end_offset)
    tensor_meta: HashMap<String, (String, Vec<usize>, usize, usize)>,
    /// Byte offset where tensor data begins in the file.
    data_offset: usize,
}

impl MmapedSafetensors {
    /// Open a safetensors file using memory mapping.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| FerrisResError::Shape(format!("Cannot open {}: {}", path.display(), e)))?;

        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| FerrisResError::Shape(format!("Cannot mmap {}: {}", path.display(), e)))?
        };

        // Parse header
        if mmap.len() < 8 {
            return Err(FerrisResError::Shape("File too small for safetensors header".into()));
        }

        let header_len = u64::from_le_bytes(
            mmap[0..8].try_into().unwrap()
        ) as usize;

        if 8 + header_len > mmap.len() {
            return Err(FerrisResError::Shape("Header extends beyond file".into()));
        }

        let header_json: HashMap<String, serde_json::Value> = serde_json::from_slice(
            &mmap[8..8 + header_len]
        ).map_err(|e| FerrisResError::Shape(format!("Invalid safetensors JSON: {}", e)))?;

        let data_offset = 8 + header_len;
        let mut tensor_meta = HashMap::new();

        for (name, value) in &header_json {
            // Skip __metadata__ entry
            if name == "__metadata__" { continue; }

            let obj = value.as_object()
                .ok_or_else(|| FerrisResError::Shape(format!("Invalid tensor entry: {}", name)))?;

            let dtype = obj.get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("F32")
                .to_string();

            let shape: Vec<usize> = obj.get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                .unwrap_or_default();

            let offsets = obj.get("data_offsets")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect::<Vec<_>>())
                .unwrap_or_default();

            if offsets.len() >= 2 {
                tensor_meta.insert(name.clone(), (dtype, shape, offsets[0], offsets[1]));
            }
        }

        tracing::info!(
            "Mmap'd {} tensors from {} ({} bytes, data_offset={})",
            tensor_meta.len(), path.display(), mmap.len(), data_offset
        );

        Ok(Self { mmap, tensor_meta, data_offset })
    }

    /// Get a tensor as Vec<f32>, converting from the stored dtype.
    /// Memory is only allocated for this one tensor.
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, _shape, start, end) = self.tensor_meta.get(name)
            .ok_or_else(|| FerrisResError::Shape(format!("Tensor '{}' not found", name)))?;

        let abs_start = self.data_offset + start;
        let abs_end = self.data_offset + end;

        if abs_end > self.mmap.len() {
            return Err(FerrisResError::Shape(format!(
                "Tensor '{}' data [{}, {}) exceeds mmap size {}", name, abs_start, abs_end, self.mmap.len()
            )));
        }

        let raw = &self.mmap[abs_start..abs_end];
        bytes_to_f32(raw, dtype)
    }

    /// Get tensor shape.
    pub fn get_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.tensor_meta.get(name).map(|(_, shape, _, _)| shape.clone())
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_meta.keys().map(|s| s.as_str()).collect()
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensor_meta.len()
    }

    /// Detect architecture from tensor names.
    pub fn detect_architecture(&self) -> ModelArchitecture {
        let names = self.tensor_names();
        detect_architecture_from_names(&names)
    }

    /// Infer number of layers from tensor names.
    pub fn infer_num_layers(&self) -> usize {
        let names = self.tensor_names();
        infer_layers_from_names(&names)
    }

    /// Infer hidden dimension from embed_tokens weight shape.
    pub fn infer_hidden_dim(&self) -> Option<usize> {
        if let Some((_, shape, _, _)) = self.tensor_meta.get("model.embed_tokens.weight") {
            if shape.len() >= 2 { return Some(shape[1]); }
        }
        None
    }

    /// Infer vocab size from embed_tokens weight shape.
    pub fn infer_vocab_size(&self) -> Option<usize> {
        if let Some((_, shape, _, _)) = self.tensor_meta.get("model.embed_tokens.weight") {
            if shape.len() >= 1 { return Some(shape[0]); }
        }
        None
    }

    /// Convert to LoadedWeights by loading all tensors (use only for small models).
    pub fn to_loaded_weights(&self) -> Result<LoadedWeights> {
        let mut tensors = HashMap::new();
        for name in self.tensor_names() {
            let (dtype, shape, _, _) = self.tensor_meta.get(name).cloned().unwrap();
            let data = self.get_tensor_f32(name)?;
            tensors.insert(name.to_string(), LoadedTensor {
                name: name.to_string(),
                shape,
                dtype: dtype,
                data,
            });
        }
        Ok(LoadedWeights {
            tensors,
            source_files: vec![PathBuf::from("mmap")],
        })
    }
}

/// File-backed safetensors reader — NO mmap.
/// Reads individual tensors via seek+read. Peak RAM = 1 tensor (~141MB).
/// Use this on low-RAM machines where mmap page faults would OOM
/// (e.g., Colab T4 with 12.7GB RAM and 10.25GB model).
pub struct FileSafetensors {
    file: File,
    tensor_meta: HashMap<String, (String, Vec<usize>, usize, usize)>,
    data_offset: usize,
}

impl FileSafetensors {
    /// Open a safetensors file for direct I/O (no mmap).
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| FerrisResError::Shape(format!("Cannot open {}: {}", path.display(), e)))?;

        let mut header_len_buf = [0u8; 8];
        use std::io::Read;
        file.read_exact(&mut header_len_buf)
            .map_err(|e| FerrisResError::Shape(format!("Failed to read header length: {}", e)))?;
        let header_len = u64::from_le_bytes(header_len_buf) as usize;

        let mut header_buf = vec![0u8; header_len];
        file.read_exact(&mut header_buf)
            .map_err(|e| FerrisResError::Shape(format!("Failed to read header: {}", e)))?;

        let header_json: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_buf)
            .map_err(|e| FerrisResError::Shape(format!("Invalid safetensors JSON: {}", e)))?;

        let data_offset = 8 + header_len;
        let mut tensor_meta = HashMap::new();

        for (name, value) in &header_json {
            if name == "__metadata__" { continue; }
            let obj = value.as_object()
                .ok_or_else(|| FerrisResError::Shape(format!("Invalid tensor entry: {}", name)))?;
            let dtype = obj.get("dtype").and_then(|v| v.as_str()).unwrap_or("F32").to_string();
            let shape: Vec<usize> = obj.get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                .unwrap_or_default();
            let offsets = obj.get("data_offsets")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect::<Vec<_>>())
                .unwrap_or_default();
            if offsets.len() >= 2 {
                tensor_meta.insert(name.clone(), (dtype, shape, offsets[0], offsets[1]));
            }
        }

        tracing::info!(
            "Opened file-backed safetensors: {} tensors from {} (data_offset={})",
            tensor_meta.len(), path.display(), data_offset
        );

        Ok(Self { file, tensor_meta, data_offset })
    }

    /// Read a single tensor as Vec<f32> via seek+read. No mmap.
    pub fn get_tensor_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let (dtype, _shape, start, end) = self.tensor_meta.get(name)
            .ok_or_else(|| FerrisResError::Shape(format!("Tensor '{}' not found", name)))?;
        let abs_start = self.data_offset + start;
        let byte_len = end - start;
        use std::io::{Seek, SeekFrom};
        self.file.seek(SeekFrom::Start(abs_start as u64))
            .map_err(|e| FerrisResError::Shape(format!("Seek failed for '{}': {}", name, e)))?;
        let mut buf = vec![0u8; byte_len];
        use std::io::Read;
        self.file.read_exact(&mut buf)
            .map_err(|e| FerrisResError::Shape(format!("Read failed for '{}': {}", name, e)))?;
        bytes_to_f32(&buf, dtype)
    }
}
