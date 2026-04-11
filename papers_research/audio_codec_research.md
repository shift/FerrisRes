# Audio Codec Integration Research (EnCodec)

## Overview
Source: https://arxiv.org/abs/2210.13438

## EnCodec Details
- Neural audio codec from Meta
- 32kHz/44.1kHz audio input
- 75x compression ratio
- 75 tokens/second at 32kHz
- Quantized discrete tokens (RVQ)

## Integration for FerrisRes
1. Input: Raw audio → EnCodec encoder
2. Output: Discrete tokens (e.g., 1024 codebook size)
3. Integration: Audio tokens projected to text embedding space

## Alternative: SoundStream
- Google research (2022)
- Similar to EnCodec
- Uses residual vector quantization (RVQ)

## Implementation Priority
- Phase 1: Integrate EnCodec pre-trained encoder
- Phase 2: Fine-tune audio tokenizer
- Phase 3: Cross-modal attention with text

## Status: DONE
