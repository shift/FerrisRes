# FerrisRes

FerrisRes is a Rust-native AI engine that learns on anything — from a $50 Raspberry Pi to a $450K NVIDIA DGX. We've replaced the quadratic bottleneck of standard Transformers with Block AttnRes, a novel linear-time architecture. This isn't just another Python wrapper; it's a zero-dependency binary that runs natively on wgpu. Our DeviceProfile system bridges the gap between edge and enterprise: it automatically triggers gradient checkpointing for 8GB Intel iGPUs, enables FP4/FP16 for Blackwell/Rubin workstations, and scales to distributed multi-GPU training on Azure's NVL72 racks. Whether you're deploying 1M+ token context windows on a Cerebras CS-3 or running decentralized training on a laptop, FerrisRes offers the same codebase with automatic hardware optimization. We're currently in Phase 4, finalizing the end-to-end trainable system that makes decentralized AI a reality.

## Status

Phase 4 — end-to-end trainable system (in progress)
