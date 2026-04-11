# GGUF Format Research for FerrisRes

## Overview
GGUF (GPT-Generated Unified Format) is the standard model format for llama.cpp and many compatible inference engines.

## Key Format Characteristics

### File Structure
1. **Header** - Magic number (GGUF), version, tensor count, metadata count
2. **Metadata** - Key-value pairs (string keys, various value types)
3. **Tensor Data** - Weights stored in row-major order

### Metadata Categories
- **General**: architecture, vocab size, context length
- **Model**: hidden size, num layers, num heads, etc.
- **Tokenizer**: tokenizer type, vocab data

### Quantization Types
Standard llama.cpp quantizations:
- Q4_0, Q4_1 - 4-bit (legacy)
- Q4_K_M, Q4_K_S - 4-bit (k-quant)
- Q5_0, Q5_1 - 5-bit
- Q5_K_M, Q5_K_S - 5-bit (k-quant)
- Q6_K - 6-bit (k-quant)
- F16 - Half precision
- F32 - Full precision

### Tensor Naming Convention
Typical tensor names in GGUF:
- `token_embd.weight` - embedding matrix
- `attn_q.weight`, `attn_k.weight`, `attn_v.weight` - attention projections
- `attn_output.weight` - output projection
- `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` - FFN layers

## Implementation for FerrisRes

### Loading Strategy
1. Parse GGUF header (validate magic, version)
2. Read metadata to determine model architecture
3. Map GGUF tensor names to FerrisRes layer structure
4. Handle quantization dequantization for inference
5. Copy weights to GPU tensors

### Key Challenges
- Architecture mapping (GGUF → BlockAttnRes)
- Quantization handling (need dequantization kernels)
- Vocabulary alignment

## Resources
- llama.cpp: https://github.com/ggerganov/llama.cpp
- GGUF spec discussions: https://github.com/ggerganov/llama.cpp/discussions
- k-quant paper: https://arxiv.org/abs/2309.07119

## Status: IN PROGRESS
Need to download actual GGUF spec document.