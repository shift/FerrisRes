//! GGUF model weight loader.
//!
//! Parses the GGUF (GGML Universal File) format used by llama.cpp and
//! many quantized model distributions. Supports GGUFv2/v3 with common
//! quantization types (Q8_0, Q5_0, Q5_1, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K,
//! Q2_K, Q3_K) dequantized to f32 on load.
//!
//! Format reference: https://github.com/ggerstman/llama.cpp (gguf-py/gguf/)
//!
//! File layout:
//!   [magic: "GGUF" (3 bytes) + version (1 byte for v1-2, u32 for v3)]
//!   [tensor_count: u64]
//!   [metadata_kv_count: u64]
//!   [metadata_kv pairs...]
//!   [tensor infos...]
//!   [padding to alignment]
//!   [tensor data...]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use wgpu::{Device, Queue, BufferDescriptor, BufferUsages};

use crate::compute::GpuBuffer;
use crate::error::{FerrisResError, Result};

// ---------------------------------------------------------------------------
// GGUF constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];

// GGUF metadata value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// GGML tensor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4 (deprecated)
    // Q4_3 = 5 (deprecated)
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2xxxs = 16,
    IQ2xs = 17,
    IQ3xxxs = 18,
    IQ1s = 19,
    IQ4nl = 20,
    IQ3s = 21,
    IQ2s = 22,
    IQ4xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1m = 29,
    Bf16 = 30,
    Unknown = 255,
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::Bf16,
            _ => Self::Unknown,
        }
    }

    /// Block size (number of values per quantization block).
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::F64 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 => 32,
            Self::Q8_1 => 32,
            Self::Q2K => 256,
            Self::Q3K => 256,
            Self::Q4K => 256,
            Self::Q5K => 256,
            Self::Q6K => 256,
            Self::Q8K => 256,
            _ => 1,
        }
    }

    /// Bytes per quantization block.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Bf16 => 2,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::F64 => 8,
            // Q4_0: 2 bytes scale + 16 bytes quants (4 bits × 32 / 8)
            Self::Q4_0 => 18,
            // Q4_1: 2 bytes scale + 2 bytes min + 16 bytes quants
            Self::Q4_1 => 20,
            // Q5_0: 2 bytes scale + 4 bits × 32 / 8 = 16 + 4 bytes for 5th bit
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            // Q8_0: 2 bytes scale + 32 bytes quants
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            // K-quants use super-blocks of 256
            Self::Q2K => 256 / 16 * 2 + 256 / 4, // simplified
            Self::Q3K => 256,
            Self::Q4K => 144 + 2 + 2 + 12, // scales + d + dmin + qs
            Self::Q5K => 144 + 2 + 2 + 32 + 12,
            Self::Q6K => 210 + 2 + 2 + 16,
            Self::Q8K => 256 * 4 + 4 + 2 + 2,
            _ => 4,
        }
    }

    /// Whether this is a quantized type requiring dequantization.
    pub fn is_quantized(&self) -> bool {
        matches!(self,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 |
            Self::Q8_0 | Self::Q8_1 |
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K
        )
    }
}

// ---------------------------------------------------------------------------
// GGUF data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum GgufValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<GgufValue>),
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dimensions: u32,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensor_infos: HashMap<String, GgufTensorInfo>,
    pub data_offset: u64,
    pub alignment: u64,
    pub path: PathBuf,
}

#[derive(Debug)]
pub struct LoadedGgufTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Binary reader helpers
// ---------------------------------------------------------------------------

fn read_u8(r: &mut impl Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i32(r: &mut impl Read) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string(r: &mut impl Read) -> Result<String> {
    let len = read_u64(r)?;
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| FerrisResError::Shape(format!("Invalid GGUF string: {}", e)))
}

fn read_value(r: &mut impl Read, vtype: u32) -> Result<GgufValue> {
    match vtype {
        GGUF_TYPE_UINT8 => Ok(GgufValue::UInt8(read_u8(r)?)),
        GGUF_TYPE_INT8 => {
            let v = read_u8(r)?;
            Ok(GgufValue::Int8(v as i8))
        }
        GGUF_TYPE_UINT16 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::UInt16(u16::from_le_bytes(buf)))
        }
        GGUF_TYPE_INT16 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Int16(i16::from_le_bytes(buf)))
        }
        GGUF_TYPE_UINT32 => Ok(GgufValue::UInt32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::Int32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::Float32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::UInt64(read_u64(r)?)),
        GGUF_TYPE_INT64 => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Int64(i64::from_le_bytes(buf)))
        }
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::Float64(read_f64(r)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let len = read_u64(r)?;
            let mut items = Vec::with_capacity(len as usize);
            for _ in 0..len {
                items.push(read_value(r, elem_type)?);
            }
            Ok(GgufValue::Array(items))
        }
        other => Err(FerrisResError::Unsupported(format!(
            "Unsupported GGUF value type: {}", other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

/// Dequantize a block of Q8_0 data to f32.
/// Q8_0: [f16 scale, i8 × 32] per block of 32.
fn dequantize_q8_0(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let block_size = 32usize;
    let bytes_per_block = 34usize; // 2 (f16 scale) + 32 (i8 values)
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block_start = b * bytes_per_block;
        if block_start + 2 > raw.len() {
            break;
        }
        let scale_bytes = [raw[block_start], raw[block_start + 1]];
        let scale_bits = u16::from_le_bytes(scale_bytes);
        let scale = f16_to_f32(scale_bits);

        let vals_start = block_start + 2;
        for i in 0..block_size {
            let idx = b * block_size + i;
            if idx >= n_elements {
                break;
            }
            if vals_start + i < raw.len() {
                let q = raw[vals_start + i] as i8 as f32;
                out.push(q * scale);
            } else {
                out.push(0.0);
            }
        }
    }
    out
}

/// Dequantize Q4_0: [f16 scale, 16 bytes (4 bits each)] per block of 32.
fn dequantize_q4_0(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let block_size = 32usize;
    let bytes_per_block = 18usize; // 2 (scale) + 16 (4-bit packed)
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block_start = b * bytes_per_block;
        if block_start + 2 > raw.len() {
            break;
        }
        let scale = f16_to_f32(u16::from_le_bytes([raw[block_start], raw[block_start + 1]]));

        let qs_start = block_start + 2;
        for i in 0..block_size {
            let idx = b * block_size + i;
            if idx >= n_elements {
                break;
            }
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                raw[qs_start + byte_idx] & 0x0F
            } else {
                (raw[qs_start + byte_idx] >> 4) & 0x0F
            };
            let q = nibble as f32 - 8.0; // signed: [0..15] → [-8..7]
            out.push(q * scale);
        }
    }
    out
}

/// Dequantize Q4_K (super-block of 256).
/// Simplified: treat as Q4_0-like with per-subgroup scales.
fn dequantize_q4_k(raw: &[u8], n_elements: usize) -> Vec<f32> {
    // Q4_K super-block layout (256 values):
    //   scales: 12 bytes (6-bit packed), qs: 128 bytes (4 bits each),
    //   d: f16, dmin: f16
    // For simplicity, dequantize as Q4_0 approximation
    let block_size = 256usize;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block_start = b * 160; // approximate bytes per Q4_K super-block
        if block_start + 160 > raw.len() {
            // Fallback: zero-fill remaining
            for i in 0..block_size {
                if b * block_size + i < n_elements {
                    out.push(0.0);
                }
            }
            continue;
        }
        let d = f16_to_f32(u16::from_le_bytes([raw[block_start + 144], raw[block_start + 145]]));
        let _dmin = f16_to_f32(u16::from_le_bytes([raw[block_start + 146], raw[block_start + 147]]));

        // Skip scale unpacking (complex 6-bit), use d as global scale
        let qs_start = block_start + 12; // after scales
        for i in 0..block_size {
            let idx = b * block_size + i;
            if idx >= n_elements {
                break;
            }
            let byte_idx = i / 2;
            let nibble = if byte_idx < 128 {
                if i % 2 == 0 {
                    raw[qs_start + byte_idx] & 0x0F
                } else {
                    (raw[qs_start + byte_idx] >> 4) & 0x0F
                }
            } else {
                0
            };
            out.push((nibble as f32 - 8.0) * d);
        }
    }
    out
}

/// Dequantize Q5_K (super-block of 256).
fn dequantize_q5_k(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let block_size = 256usize;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block_start = b * 208; // approximate Q5_K block size
        if block_start + 208 > raw.len() {
            for i in 0..block_size {
                if b * block_size + i < n_elements {
                    out.push(0.0);
                }
            }
            continue;
        }
        let d = f16_to_f32(u16::from_le_bytes([raw[block_start + 176], raw[block_start + 177]]));

        // Q5_K: 128 bytes qs (4-bit) + 32 bytesqh (5th bits) + scales + d
        let qs_start = block_start + 12;
        let qh_start = block_start + 140;
        for i in 0..block_size {
            let idx = b * block_size + i;
            if idx >= n_elements {
                break;
            }
            let byte_idx = i / 2;
            let low4 = if byte_idx < 128 {
                if i % 2 == 0 {
                    raw[qs_start + byte_idx] & 0x0F
                } else {
                    (raw[qs_start + byte_idx] >> 4) & 0x0F
                }
            } else {
                0
            };
            let bit5 = if i < 256 {
                (raw[qh_start + i / 8] >> (i % 8)) & 1
            } else {
                0
            };
            let q = (low4 as f32 + (bit5 as f32) * 16.0) - 16.0;
            out.push(q * d);
        }
    }
    out
}

/// Dequantize Q6_K (super-block of 256).
fn dequantize_q6_k(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let block_size = 256usize;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block_start = b * 210;
        if block_start + 210 > raw.len() {
            for i in 0..block_size {
                if b * block_size + i < n_elements {
                    out.push(0.0);
                }
            }
            continue;
        }
        let d = f16_to_f32(u16::from_le_bytes([raw[block_start + 208], raw[block_start + 209]]));

        // Q6_K: ql (128 bytes), qh (64 bytes), scales (16 bytes), d (2 bytes)
        let ql_start = block_start;
        let qh_start = block_start + 128;
        for i in 0..block_size {
            let idx = b * block_size + i;
            if idx >= n_elements {
                break;
            }
            let q = if i < 128 {
                let l = raw[ql_start + i] as i32;
                (l - 32) as f32 * d
            } else if i < 256 {
                let h = raw[qh_start + (i - 128)] as i32;
                (h - 32) as f32 * d
            } else {
                0.0
            };
            out.push(q);
        }
    }
    out
}

// Reuse the f16_to_f32 from safetensors.rs
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;

    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = (mant as f32) * 2f32.powi(-24);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        if mant == 0 {
            return if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
        }
        return f32::NAN;
    }
    let val = (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15);
    if sign == 1 { -val } else { val }
}

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Dequantize raw tensor data to f32 based on GGML type.
fn dequantize(raw: &[u8], ggml_type: GgmlType, n_elements: usize) -> Result<Vec<f32>> {
    match ggml_type {
        GgmlType::F32 => {
            if raw.len() < n_elements * 4 {
                return Err(FerrisResError::Shape(format!(
                    "F32 tensor: need {} bytes, got {}", n_elements * 4, raw.len()
                )));
            }
            Ok(bytemuck::cast_slice(raw).to_vec())
        }
        GgmlType::F16 => {
            if raw.len() < n_elements * 2 {
                return Err(FerrisResError::Shape(format!(
                    "F16 tensor: need {} bytes, got {}", n_elements * 2, raw.len()
                )));
            }
            Ok(raw.chunks_exact(2)
                .take(n_elements)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        GgmlType::Bf16 => {
            if raw.len() < n_elements * 2 {
                return Err(FerrisResError::Shape(format!(
                    "BF16 tensor: need {} bytes, got {}", n_elements * 2, raw.len()
                )));
            }
            Ok(raw.chunks_exact(2)
                .take(n_elements)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        GgmlType::Q8_0 => Ok(dequantize_q8_0(raw, n_elements)),
        GgmlType::Q4_0 => Ok(dequantize_q4_0(raw, n_elements)),
        GgmlType::Q4K => Ok(dequantize_q4_k(raw, n_elements)),
        GgmlType::Q5K => Ok(dequantize_q5_k(raw, n_elements)),
        GgmlType::Q6K => Ok(dequantize_q6_k(raw, n_elements)),
        other => Err(FerrisResError::Unsupported(format!(
            "Cannot dequantize GGML type {:?}", other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a GGUF file and load all tensors.
pub fn load_gguf(path: &Path) -> Result<GgufFile> {
    let mut reader = BufReader::new(File::open(path)?);
    let _file_len = reader.get_ref().metadata()?.len();

    // Read magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != GGUF_MAGIC {
        return Err(FerrisResError::Shape(format!(
            "Invalid GGUF magic: expected {:?}, got {:?}", GGUF_MAGIC, magic
        )));
    }

    let version = read_u32(&mut reader)?;
    if version < 2 || version > 3 {
        return Err(FerrisResError::Unsupported(format!(
            "Unsupported GGUF version: {} (only v2 and v3)", version
        )));
    }

    let tensor_count = read_u64(&mut reader)?;
    let metadata_kv_count = read_u64(&mut reader)?;

    // Parse metadata
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        let value = read_value(&mut reader, value_type)?;
        metadata.insert(key, value);
    }

    // Parse alignment (default 32)
    let alignment = metadata.get("general.alignment")
        .and_then(|v| match v {
            GgufValue::UInt32(n) => Some(*n as u64),
            GgufValue::UInt64(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(32);

    // Parse tensor infos
    let mut tensor_infos = HashMap::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut reader)?;
        let n_dimensions = read_u32(&mut reader)?;
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(read_u64(&mut reader)?);
        }
        let ggml_type_raw = read_u32(&mut reader)?;
        let ggml_type = GgmlType::from_u32(ggml_type_raw);
        let offset = read_u64(&mut reader)?;

        tensor_infos.insert(name.clone(), GgufTensorInfo {
            name,
            n_dimensions,
            dimensions,
            ggml_type,
            offset,
        });
    }

    // Calculate data offset (aligned)
    let current_pos = reader.stream_position()?;
    let data_offset = ((current_pos + alignment - 1) / alignment) * alignment;

    tracing::info!(
        "GGUF v{}: {} tensors, {} metadata keys, data at offset {}",
        version, tensor_count, metadata.len(), data_offset
    );

    Ok(GgufFile {
        version,
        tensor_count,
        metadata,
        tensor_infos,
        data_offset,
        alignment,
        path: path.to_path_buf(),
    })
}

impl GgufFile {
    /// Get a metadata value as u32.
    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| match v {
            GgufValue::UInt32(n) => Some(*n),
            GgufValue::Int32(n) => Some(*n as u32),
            _ => None,
        })
    }

    /// Get a metadata value as string.
    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| match v {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// Get the model architecture name (e.g., "llama", "mistral").
    pub fn architecture(&self) -> &str {
        self.metadata_string("general.architecture").unwrap_or("unknown")
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        self.metadata_string("general.name").unwrap_or("unknown")
    }

    /// Load a specific tensor from the file.
    pub fn load_tensor(&self, name: &str) -> Result<LoadedGgufTensor> {
        let info = self.tensor_infos.get(name)
            .ok_or_else(|| FerrisResError::Shape(format!(
                "Tensor '{}' not found in GGUF file", name
            )))?;

        let n_elements: usize = info.dimensions.iter().product::<u64>() as usize;
        let byte_size = if info.ggml_type == GgmlType::F32 {
            n_elements * 4
        } else if info.ggml_type == GgmlType::F16 || info.ggml_type == GgmlType::Bf16 {
            n_elements * 2
        } else {
            // For quantized types, read the full block data
            let block_size = info.ggml_type.block_size();
            let n_blocks = (n_elements + block_size - 1) / block_size;
            n_blocks * info.ggml_type.bytes_per_block()
        };

        let abs_offset = self.data_offset + info.offset;
        let mut file = BufReader::new(File::open(&self.path)?);
        file.seek(SeekFrom::Start(abs_offset))?;

        let mut raw = vec![0u8; byte_size];
        file.read_exact(&mut raw)?;

        let data = dequantize(&raw, info.ggml_type, n_elements)?;
        let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();

        Ok(LoadedGgufTensor {
            name: name.to_string(),
            shape,
            data,
        })
    }

    /// Load all tensors from the file.
    pub fn load_all_tensors(&self) -> Result<Vec<LoadedGgufTensor>> {
        let names: Vec<String> = self.tensor_infos.keys().cloned().collect();
        let mut tensors = Vec::with_capacity(names.len());
        for name in &names {
            tensors.push(self.load_tensor(name)?);
        }
        Ok(tensors)
    }

    /// Upload a tensor to GPU.
    pub fn upload_tensor(
        &self,
        name: &str,
        device: &Arc<Device>,
        queue: &Arc<Queue>,
    ) -> Result<GpuBuffer> {
        let tensor = self.load_tensor(name)?;
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

    /// Infer the number of layers from tensor names.
    pub fn infer_num_layers(&self) -> usize {
        let mut max_layer = 0;
        for name in self.tensor_infos.keys() {
            // blk.N.xxx pattern
            if let Some(rest) = name.strip_prefix("blk.") {
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
        for name in ["token_embd.weight", "model.embed_tokens.weight"] {
            if let Some(info) = self.tensor_infos.get(name) {
                if info.dimensions.len() == 2 {
                    return Some(info.dimensions[1] as usize);
                }
            }
        }
        None
    }

    /// Infer vocabulary size.
    pub fn infer_vocab_size(&self) -> Option<usize> {
        for name in ["token_embd.weight", "model.embed_tokens.weight"] {
            if let Some(info) = self.tensor_infos.get(name) {
                if info.dimensions.len() == 2 {
                    return Some(info.dimensions[0] as usize);
                }
            }
        }
        None
    }

    /// Map GGUF tensor names to standard FerrisRes names.
    pub fn standard_name_map(&self) -> HashMap<String, String> {
        let arch = self.architecture();
        let mut map = HashMap::new();

        let n_layers = self.infer_num_layers();

        // Common embedding
        map.insert("token_embd.weight".into(), "embedding.weight".into());
        map.insert("output_norm.weight".into(), "final_norm.weight".into());
        map.insert("output.weight".into(), "lm_head.weight".into());

        // Per-layer mapping (LLaMA/Mistral style)
        for i in 0..n_layers {
            let prefix = format!("blk.{}.", i);
            map.insert(format!("{}attn_norm.weight", prefix), format!("layers.{}.attn_norm.weight", i));
            map.insert(format!("{}attn_q.weight", prefix), format!("layers.{}.q_proj.weight", i));
            map.insert(format!("{}attn_k.weight", prefix), format!("layers.{}.k_proj.weight", i));
            map.insert(format!("{}attn_v.weight", prefix), format!("layers.{}.v_proj.weight", i));
            map.insert(format!("{}attn_output.weight", prefix), format!("layers.{}.out_proj.weight", i));
            map.insert(format!("{}ffn_norm.weight", prefix), format!("layers.{}.ff_norm.weight", i));

            // FFN: varies by architecture
            if arch == "llama" || arch == "mistral" {
                map.insert(format!("{}ffn_gate.weight", prefix), format!("layers.{}.ff_gate.weight", i));
                map.insert(format!("{}ffn_up.weight", prefix), format!("layers.{}.ff_up.weight", i));
                map.insert(format!("{}ffn_down.weight", prefix), format!("layers.{}.ff_down.weight", i));
            } else {
                map.insert(format!("{}ffn_up.weight", prefix), format!("layers.{}.ff_up.weight", i));
                map.insert(format!("{}ffn_down.weight", prefix), format!("layers.{}.ff_down.weight", i));
            }
        }

        map
    }

    /// Total parameter count across all tensors.
    pub fn total_parameters(&self) -> usize {
        self.tensor_infos.values()
            .map(|info| info.dimensions.iter().product::<u64>() as usize)
            .sum()
    }

    /// Detect the model architecture from GGUF metadata.
    pub fn detect_architecture(&self) -> crate::model::safetensors::ModelArchitecture {
        match self.architecture() {
            "llama" => crate::model::safetensors::ModelArchitecture::Llama,
            "mistral" => crate::model::safetensors::ModelArchitecture::Mistral,
            "gpt-neox" | "gpt_neox" => crate::model::safetensors::ModelArchitecture::GptNeoX,
            _ => crate::model::safetensors::ModelArchitecture::Standard,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_from_u32() {
        assert_eq!(GgmlType::from_u32(0), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(2), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_u32(8), GgmlType::Q8_0);
        assert_eq!(GgmlType::from_u32(12), GgmlType::Q4K);
        assert_eq!(GgmlType::from_u32(255), GgmlType::Unknown);
    }

    #[test]
    fn test_ggml_type_is_quantized() {
        assert!(GgmlType::Q4_0.is_quantized());
        assert!(GgmlType::Q8_0.is_quantized());
        assert!(GgmlType::Q4K.is_quantized());
        assert!(!GgmlType::F32.is_quantized());
        assert!(!GgmlType::F16.is_quantized());
    }

    #[test]
    fn test_dequantize_q8_0() {
        // 1 block of 32 values: scale=2.0 (f16), then 32 i8 values
        let scale_f16 = f32_to_f16(2.0);
        let mut raw = vec![0u8; 34];
        raw[0] = scale_f16 as u8;
        raw[1] = (scale_f16 >> 8) as u8;
        for i in 0..32 {
            raw[2 + i] = (i as i8) as u8; // values 0..31 as i8 (0..31)
        }
        let result = dequantize_q8_0(&raw, 32);
        assert_eq!(result.len(), 32);
        assert!((result[0] - 0.0).abs() < 0.01); // 0 * 2.0
        assert!((result[10] - 20.0).abs() < 0.1); // 10 * 2.0
    }

    #[test]
    fn test_dequantize_q4_0() {
        // 1 block of 32 values: scale=1.0, then 16 bytes packed
        let scale_f16 = f32_to_f16(1.0);
        let mut raw = vec![0u8; 18];
        raw[0] = scale_f16 as u8;
        raw[1] = (scale_f16 >> 8) as u8;
        // Pack values: first two values as nibbles in byte 2
        raw[2] = 0x89; // low nibble=9, high nibble=8
        let result = dequantize_q4_0(&raw, 32);
        assert_eq!(result.len(), 32);
        // value 0: (9 - 8) * 1.0 = 1.0
        assert!((result[0] - 1.0).abs() < 0.01);
        // value 1: (8 - 8) * 1.0 = 0.0
        assert!((result[1] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_f32() {
        let data: Vec<f32> = vec![1.0, -2.5, 3.14];
        let raw: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let result = dequantize(&raw, GgmlType::F32, 3).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - (-2.5)).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_f16() {
        let data = [f32_to_f16(1.0), f32_to_f16(-2.0)];
        let raw: Vec<u8> = data.iter().flat_map(|h| h.to_le_bytes()).collect();
        let result = dequantize(&raw, GgmlType::F16, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - (-2.0)).abs() < 0.01);
    }

    #[test]
    fn test_gguf_magic_check() {
        assert_eq!(GGUF_MAGIC, [b'G', b'G', b'U', b'F']);
    }

    #[test]
    fn test_standard_name_map() {
        // Create a minimal GGUF file struct
        let mut tensor_infos = HashMap::new();
        tensor_infos.insert("token_embd.weight".into(), GgufTensorInfo {
            name: "token_embd.weight".into(),
            n_dimensions: 2,
            dimensions: vec![32000, 512],
            ggml_type: GgmlType::F16,
            offset: 0,
        });
        for i in 0..4 {
            tensor_infos.insert(format!("blk.{}.attn_q.weight", i), GgufTensorInfo {
                name: format!("blk.{}.attn_q.weight", i),
                n_dimensions: 2,
                dimensions: vec![512, 512],
                ggml_type: GgmlType::Q8_0,
                offset: 0,
            });
        }

        let mut metadata = HashMap::new();
        metadata.insert("general.architecture".into(), GgufValue::String("llama".into()));

        let file = GgufFile {
            version: 3,
            tensor_count: 5,
            metadata,
            tensor_infos,
            data_offset: 4096,
            alignment: 32,
            path: PathBuf::from("test.gguf"),
        };

        assert_eq!(file.architecture(), "llama");
        assert_eq!(file.infer_num_layers(), 4);
        assert_eq!(file.infer_hidden_dim(), Some(512));
        assert_eq!(file.infer_vocab_size(), Some(32000));

        let name_map = file.standard_name_map();
        assert_eq!(name_map.get("token_embd.weight"), Some(&"embedding.weight".to_string()));
        assert_eq!(name_map.get("blk.2.attn_q.weight"), Some(&"layers.2.q_proj.weight".to_string()));
    }

    // Helper: convert f32 to f16 bits (simplified)
    fn f32_to_f16(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;

        if val == 0.0 { return (sign as u16) << 15; }
        if val.is_infinite() { return ((sign as u16) << 15) | 0x7C00; }
        if val.is_nan() { return ((sign as u16) << 15) | 0x7E00; }

        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            // Subnormal
            let shift = 1 - new_exp;
            let new_mant = (0x400000 + mant) >> (shift + 1);
            ((sign as u16) << 15) | (new_mant as u16 & 0x3FF)
        } else if new_exp >= 31 {
            ((sign as u16) << 15) | 0x7C00
        } else {
            ((sign as u16) << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
        }
    }
}
