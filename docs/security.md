# FerrisRes Armor: Security Configuration Guide

FerrisRes Armor is a 4-layer defensive security proxy that sits in front of LLM workloads. It uses distilled neural scanners, regex pattern matching, and representation engineering to detect and block malicious inputs, sanitize outputs, and self-improve over time.

## Architecture

```
Input Prompt → L0 (Regex + Bloom) → L1 (Neural Scanner) → Allow/Block
                                                        ↓ (Allow)
Generation → L2 (RepE Probe on Hidden States) → Allow/Block
                                                        ↓ (Allow)
Output   → L3 (PII Redaction) → Clean Output / Redacted Output
```

### L0: Static Defense (Regex + Bloom Filter)

Instant blocking of known malicious strings and PII patterns. 31 regex recognizers covering:

- **PII**: email, SSN, phone numbers (international), credit cards (Visa/MC/Amex), IP (v4/v6), dates of birth, ZIP codes, IBAN, SWIFT/BIC, driver license, passport, CUSIP, medical record numbers
- **Credentials**: API keys (AWS, generic, bearer tokens), private keys, JWT tokens, GitHub/Slack tokens, URLs with embedded credentials
- **Crypto**: Bitcoin and Ethereum wallet addresses
- **Injection heuristics**: "ignore previous instructions", "you are now", "DAN mode", base64 command injection, output exfiltration attempts

Bloom filter provides O(1) lookup for custom blocklist entries. Default 1MB (~8M bits, ~0.01% FPR for 1M entries).

### L1: Neural Scanner (~3.5M params)

ArmorGuardTiny: a 4-layer BERT-style transformer for binary SAFE/INJECTION classification. Distilled from protectai/deberta-v3-base-prompt-injection (183M params).

- Token embedding (vocab=1000, hidden=256) + position embedding
- 4 encoder layers with multi-head self-attention (4 heads) + FFN (256→1024→256)
- [CLS] pooling → linear classifier → sigmoid
- CPU-only with `Vec<f32>` weights, Xavier initialization
- Threshold: 0.7 (configurable)

### L2: RepE Safety Probe

Representation Engineering (Zou et al.) shows safety features live in linear subspaces of transformer hidden states. L2 uses independent logistic regression probes on BlockSummaryLayer outputs.

6 categories: violence, self-harm, sexual, hate, harassment, injection.

Each probe is a single linear layer `[hidden_dim → 1]` + sigmoid. Cost: <1ms per probe.

### L3: Output Sanitizer

PII redaction on generated text using the L0 PatternEngine for detection. Three strategies:

- **Mask**: Replace each character with `*`
- **Replace**: Replace entire match with `[REDACTED]`
- **Truncate**: Remove the match entirely

Injection heuristic recognizers are excluded from output redaction (input-only concern).

## Self-Learning Feedback Loop

When the WASM Sandbox or external sources detect a violation:

1. `feedback_violation()` records the violation with source and reason
2. Violation words are automatically added to the L0 Bloom filter
3. Violation history is maintained (configurable size, default 1000)
4. Future integration: ConceptMap storage → LoRA adapter updates on ArmorGuardTiny

## Configuration

```rust
use ferrisres::security::armor::{ArmorLayer, ArmorConfig};

// Default: all layers enabled
let mut armor = ArmorLayer::new();

// Custom config
let config = ArmorConfig {
    l0_enabled: true,
    l1_enabled: true,
    l2_enabled: false,  // disable L2 probe
    l3_enabled: true,
    l1_threshold: 0.8,  // stricter injection threshold
    l2_threshold: 0.7,
    max_violation_history: 500,
};
let mut armor = ArmorLayer::with_config(config);
```

## Usage

```rust
// Check input before generation
match armor.verify_input("What is the weather?") {
    SecurityVerdict::Allow => { /* proceed */ },
    SecurityVerdict::Block(reason) => { eprintln!("Blocked: {}", reason); },
    SecurityVerdict::Redact(_) => { /* shouldn't happen for input */ },
}

// Check hidden states during generation (L2)
let hidden: Vec<f32> = get_block_summary_output();
match armor.verify_hidden(&hidden) {
    SecurityVerdict::Allow => { /* continue generation */ },
    SecurityVerdict::Block(reason) => { /* stop generation */ },
    _ => {}
}

// Sanitize output after generation
match armor.sanitize_output(&generated_text) {
    SecurityVerdict::Allow => { /* output is clean */ },
    SecurityVerdict::Redact(cleaned) => { /* use cleaned text */ },
    SecurityVerdict::Block(_) => { /* shouldn't happen for output */ },
}

// Custom blocklist
armor.l0.block("malicious_domain.com");
armor.l0.block_many(&["bad_word_1", "bad_word_2"]);

// External violation feedback
armor.feedback_violation(
    "suspicious prompt text",
    ViolationSource::WasmSandbox,
    "Attempted filesystem escape"
);

// Statistics
let stats = armor.stats();
println!("Checks: {}, Blocked: {}, Rate: {:.1}%",
    stats.total_checks, stats.total_blocked, stats.block_rate * 100.0);
```

## Performance

| Layer | Latency | Notes |
|-------|---------|-------|
| L0 Regex | ~10-100μs | Dominated by regex scan, depends on input length |
| L0 Bloom | <1μs | O(1) hash lookup |
| L1 Neural | ~1-5ms | 4-layer BERT forward pass on CPU |
| L2 Probe | <1ms | 6 × dot product of size 256 |
| L3 Redact | ~10-100μs | Same as L0 regex + string replacement |

Total input check (L0+L1): ~1-5ms. Total output check (L3): ~10-100μs.

## API Stability

The security module is **unstable** — breaking changes may occur before v1.0.0.
