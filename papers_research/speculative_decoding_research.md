# Speculative Decoding Research

## Overview
Speculative decoding accelerates LLM inference by using a small draft model to predict multiple tokens, then verifying with the main model.

## Algorithms

### 1. Standard Speculative Decoding (Leviathan et al., 2023)
- Draft model: Smaller, faster model
- Verification: Larger target model
- Accept/Reject each draft token

### 2. DistilBERT Style Distillation
- Train small model to mimic large
- Use as draft

### 3. n-gram Speculative
- Use n-gram as draft
- No additional model needed

## Implementation Requirements
- Draft model (smaller BlockAttnRes)
- Target model (full BlockAttnRes)  
- Verification kernel
- Acceptance algorithm

## FerrisRes Integration
1. Draft: 1-2 layer BlockAttnRes
2. Target: Full model
3. KV cache sharing between models
4. Accept/reject based on probability

## Performance Gains
- 2-3x speedup typical
- Up to 6x with optimal settings

## Status: DONE
