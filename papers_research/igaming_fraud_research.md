# iGaming Fraud Engine Research

## Overview
Real-time fraud detection for online gambling platforms requiring:
- WebSocket streaming for continuous player interaction data
- InterventionHead for real-time account actions
- GamingCommissionOracle for RTP compliance

## Data Streams

### Player Interaction Data
| Stream | Description | Anomaly Signal |
|--------|-------------|----------------|
| Mouse trajectories | Heatmap coordinates, hover patterns | Bot detection, assistive software |
| Click latencies | Time between clicks (ms) | Macro usage, auto-clickers |
| Bet sizing patterns | Bet amounts over time | Money laundering, chip dumping |
| Spin/hand timing | Time between game actions | Scripted play, botting |
| Session duration | Total play time | Addiction indicators |

### Technical Stack
- WebSocket for sub-100ms latency
- Redis pub/sub for inter-service communication
- Apache Kafka for event replay/audit

## Regulatory Compliance

### MGA (Malta Gaming Authority)
- Remote Gaming Regulations (S.L. 400.10)
- 5% maximum stake on slots
- Self-exclusion API required
- Anti-money laundering (AML) reporting

### GlüStV (German State Treaty)
- 1€ maximum stake
- 5s spin delay on slots
- Deposit limits enforced
- OASIS blocking list integration

### GLI-19 (Gaming Labs International)
- RNG certification requirements
- RTP verification (95-98% for slots)
- Game outcome audit trails

## Anomaly Detection Patterns

### Betting Syndicates
- Coordinated betting patterns across accounts
- Shared IP addresses / device fingerprints
- Synchronized deposit/withdrawal timing
- Cross-casino colluders

### Latency Arbitrage
- Exploiting delayed game state updates
- Detected via timestamp mismatch >50ms
- Requires edge computing, not central server

### Botting
- Uniform click intervals (std dev <10ms)
- No mouse movement between clicks
- Predictable bet sizing algorithms

## Implementation Components

### WebSocketStreamEncoder
```rust
pub struct WebSocketStreamEncoder {
    player_id: Uuid,
    game_session: String,
    event_types: Vec<StreamEvent>,
}

enum StreamEvent {
    MouseMove { x: f32, y: f32, timestamp: u64 },
    Click { button: u8, latency_ms: f32 },
    Bet { amount: f64, game_state: String },
    Spin { duration_ms: u32 },
}
```

### InterventionHead
```rust
pub trait InterventionHead {
    async fn lock_account(&self, player_id: Uuid, reason: &str) -> Result<()>;
    async fn trigger_kyc(&self, player_id: Uuid, risk_score: f32) -> Result<()>;
    async fn adjust_limits(&self, player_id: Uuid, new_limit: f64) -> Result<()>;
}
```

### GamingCommissionOracle
```rust
pub trait GamingCommissionOracle {
    fn verify_rtp_compliance(&self, game_id: &str, interventions: &[Intervention]) -> bool;
    fn report_aml_suspicion(&self, player_id: &str, evidence: &Evidence) -> Result<()>;
}
```

## References
- MGA Licensing: https://www.mga.org.mt/
- GlüStV 2021: German Interstate Treaty on Gambling
- GLI-19: RNG Certification Standards