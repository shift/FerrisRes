# Safetensors Format Research for FerrisRes

## Overview
Safetensors is a secure, efficient tensor storage format by HuggingFace. Zero-copy, safe loading (no pickle).

## File Structure

### Format Layout
1. **8 bytes**: Header size N (little-endian uint64)
2. **N bytes**: JSON header (UTF-8 string)
3. **Rest**: Raw tensor data bytes

### Header JSON Schema
```json
{
  "tensor_name": {
    "dtype": "F16|F32|F64|I8|I16|I32|I64|U8|U16|U32|U64|BOOL",
    "shape": [batch, seq, hidden],
    "data_offsets": [begin, end]
  },
  "__metadata__": {
    "key": "value"
  }
}
```

### Key Advantages
| Feature | Description |
|---------|-----------|
| Secure | No pickle, prevents code execution |
| Zero-copy | mmap for fast loading |
| Lazy loading | Load tensors on demand |
| Bfloat16 | Full bfloat16 support |
| Fp8 | Float8 support |
| No size limit | Efficient for large models |

### Common dtypes
- F16, F32, F64 - Float16/32/64
- I8, I16, I32, I64 - Signed integers
- U8, U16, U32, U64 - Unsigned integers  
- BOOL - Boolean

## Implementation for FerrisRes

### Loading Strategy
1. Parse header (read first 8 bytes for size)
2. Parse JSON to get tensor metadata
3. mmap file for zero-copy access
4. Map dtype to FerrisRes types
5. Handle sharded files (.part-00001-of-xxxxx)

### Shard Loading
HuggingFace sharded models use pattern:
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`

### Tensor Name Mapping
Need to map HuggingFace conventions to FerrisRes:
- `model.embed_tokens.weight` → TokenEmbedding
- `model.layers.*.attention.*` → BlockAttnRes attention
- `model.layers.*.mlp.*` → FFN/MoE

## Status: IN PROGRESS

## Resources
- Main repo: https://github.com/huggingface/safetensors
- Format: https://huggingface.co/docs/safetensors/format