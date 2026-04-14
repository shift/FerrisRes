//! FerrisRes Armor: 4-layer defensive security proxy.
//!
//! - **L0** (`armor_l0`): Regex pattern engine + Bloom filter for instant PII/blocklist checks
//! - **L1** (`armor_l1`): Distilled neural injection scanner (~5M params)
//! - **L2** (`armor_l2`): RepE safety probe on BlockSummary hidden states
//! - **L3** (`armor_l3`): Parallel PII redaction output sanitizer
//! - **Orchestrator** (`armor`): ArmorLayer tying L0-L3 + self-learning feedback

pub mod armor_l0;
pub mod armor_l1;
pub mod armor_l2;
pub mod armor_l3;
pub mod armor;
