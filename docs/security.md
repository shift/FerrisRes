# FerrisRes Armor: Security Configuration Guide

FerrisRes Armor is a 4-layer defensive security proxy that sits in front of LLM workloads. It uses distilled neural scanners, regex pattern matching, and representation engineering to detect and block malicious inputs, sanitize outputs, and self-improve over time.

> **Status: L0 and L3 are implemented and deterministic. L1 has a synthetic
> bootstrap artefact and a versioned loader; L2 is scaffolded. Neither L1 nor
> L2 is enabled by default, and the L1 artefact is not a real scanner — dev
> FNR 25% at 0.33% FPR on synthetic data. Do not enable them until the gate in
> [`tasks/armor-l1-enablement.md`](tasks/armor-l1-enablement.md) passes. The
> CLI `--armor` flag is off by default; `ArmorLayer::new()` now leaves L1 off,
> and `try_with_config` is the production path.

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

### L1: Neural Scanner (2.2M params, synthetic bootstrap only)

ArmorGuardTiny is a bidirectional encoder classifier: 3 layers, hidden 192,
4 heads, FFN 768, max_seq 256, no causal mask, mean pool over non-pad
positions, linear `192→1`, sigmoid.

- **2,214,337 parameters**; artefact `FRAL` v1, 8.86 MB, sha256 trailer.
- Tokenizer id 1: NFKC subset, lowercase, zero-width removal, whitespace
  collapse, one FNV-1a bucket per word into 256..4352, byte tokens for the
  rest.
- CPU training and inference.

**Current state: trained on a synthetic bootstrap, not on a real corpus.**
2,800 train / 600 dev / 600 test, author-written EN/DE templates. Dev is
**25% FNR at 0.33% FPR**; the held-out test reads 0.67% FNR, but it is easier
because its zero-width families normalise onto phrases present in train. The
dev number is the generalisation signal, and it says the fixture is too easy —
not that the scanner works.

Also: only the first 12 positions of `pos_embed` were trained; positions
12..256 keep their initialisation, so scans longer than 12 tokens are not
supported by this artefact despite `max_seq = 256`.

The loader rejects bad magic, version, shape, hash, truncation and tokenizer id
with named errors. `L1Scanner::new_random()` is test-only; `ArmorLayer::new()`
leaves L1 off; `try_with_config` is the production path and is not yet wired
into the CLI.

The corpus, the real FNR/FPR gate, German review and the quarantine policy are
specified in [`tasks/armor-l1-enablement.md`](tasks/armor-l1-enablement.md).

The training corpus, the model changes (3 layers, hidden 192, byte-level plus
hashed word buckets), the FNR/FPR gate, the versioned weight format and the
quarantine policy are specified in
[`tasks/armor-l1-enablement.md`](tasks/armor-l1-enablement.md).

### L2: RepE Safety Probe (not yet trained)

Representation Engineering (Zou et al.) shows safety features live in linear subspaces of transformer hidden states. L2 is intended to use independent logistic regression probes on BlockSummaryLayer outputs.

6 categories: violence, self-harm, sexual, hate, harassment, injection.

Each probe is a single linear layer `[hidden_dim → 1]` + sigmoid. Cost: <1ms per probe.

**Current state: the probes are hash-initialised and there is no hidden-state
hook wired into generation.** L2 is disabled in every supported configuration.
Enabling it needs trained probes per model and a forward hook that measures its
latency on the target device.

### L3: Output Sanitizer

PII redaction on generated text using the L0 PatternEngine for detection. Three strategies:

- **Mask**: Replace each character with `*`
- **Replace**: Replace entire match with `[REDACTED]`
- **Truncate**: Remove the match entirely

Injection heuristic recognizers are excluded from output redaction (input-only concern).

## Self-Learning Feedback Loop

> **Only the Bloom-filter addition is real today.** The ConceptMap → LoRA
> update of ArmorGuardTiny is not implemented, and with hash-initialised
> weights it would have nothing meaningful to update.

When the WASM Sandbox or external sources detect a violation:

1. `feedback_violation()` records the violation with source and reason
2. Violation words are automatically added to the L0 Bloom filter
3. Violation history is maintained (configurable size, default 1000)
4. Future integration: ConceptMap storage → LoRA adapter updates on ArmorGuardTiny

## Configuration

```rust
use ferrisres::security::armor::{ArmorLayer, ArmorConfig};

// Default: all layers enabled (library default). The CLI --armor flag is off
// by default, and enabling L1/L2 without trained weights must be refused once
// the loader lands — see tasks/armor-l1-enablement.md.
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

> The L1 and L2 rows are the **target** for a trained scanner on this hardware.
> They are not measurements. L1 is now 2.2M params (3×192), not the 4×256 shape
> this table was written for; its latency is unmeasured.

| Layer | Latency | Notes |
|-------|---------|-------|
| L0 Regex | ~10-100μs | Dominated by regex scan, depends on input length |
| L0 Bloom | <1μs | O(1) hash lookup |
| L1 Neural | ~1-5ms | 4-layer BERT forward pass on CPU |
| L2 Probe | <1ms | 6 × dot product of size 256 |
| L3 Redact | ~10-100μs | Same as L0 regex + string replacement |

Total input check (L0+L1): ~1-5ms. Total output check (L3): ~10-100μs.

## Security claims and limits

- **Armor is a detector, not the safety boundary.** Tool allowlists, least
  privilege and the control plane decide whether an action happens. A false
  negative in Armor must still be caught there.
- **L0 and L3 are deterministic** and can be enabled today: regex PII
  detection, a custom Bloom blocklist, and output redaction.
- **L1 and L2 are not protection yet.** Their scores are noise until trained
  weights exist. Describing them as scanners in a security review would be
  wrong.
- **No adversarial guarantee.** A 2.5–3.5M classifier will be evaded by
  obfuscation and novel phrasing. German coverage is unmeasured until the
  corpus in the task spec exists.
- **Violation records must not contain raw untrusted text.** The current
  `ViolationRecord.input` does; it must become `content_sha256` plus length
  and source.
- **Quarantine, not silent drop.** When L1 is enabled, withheld content is
  replaced by a typed notice the model can see, and the verdict is recorded.

## API Stability

The security module is **unstable** — breaking changes may occur before v1.0.0.
