# Video Token Compression Research

## Overview
Research for efficient video tokenization in multimodal models.

## Approaches

### 1. Video Transformer (ViViT)
- Paper: https://arxiv.org/abs/2202.03011
- Tube embedding: spatiotemporal patches
- Encode temporal dimension efficiently

### 2. C-ViViT
- Factorized attention over space and time
- Lower computation than full spatiotemporal

### 3. Token Pooling Methods
- Dynamic temporal pooling
- Semantic compression
- Keep key frames only

## Compression Strategies
| Method | Ratio | Quality |
|--------|-------|---------|
| Key frames | 10-30x | Medium |
| Tube embedding | 4-8x | High |
| Semantic | Variable | High |

## Implementation for FerrisRes
1. Extract frames at configurable FPS (1-30)
2. Apply ViT encoding to frames
3. Temporal attention across frames
4. Output video tokens

## Status: DONE
