//! iGaming Fraud Engine
//! 
//! Real-time fraud detection for online gambling platforms:
//! - WebSocket stream encoder for player interactions
//! - Intervention head for account actions
//! - Gaming commission oracle for RTP compliance

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum IgamingError {
    #[error("WebSocket connection failed: {0}")]
    WebSocket(String),
    
    #[error("Intervention failed: {0}")]
    Intervention(String),
    
    #[error("RTP violation: {0}")]
    RtpViolation(String),
    
    #[error("Anomaly detection: {0}")]
    Anomaly(String),
}

// ============================================================================
// Data Types
// ============================================================================

/// Player interaction event from WebSocket stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlayerEvent {
    MouseMove {
        player_id: Uuid,
        x: f32,
        y: f32,
        timestamp_ms: u64,
    },
    Click {
        player_id: Uuid,
        button: u8,
        latency_ms: f32,
        timestamp_ms: u64,
    },
    Bet {
        player_id: Uuid,
        amount: f64,
        game_id: String,
        timestamp_ms: u64,
    },
    Spin {
        player_id: Uuid,
        duration_ms: u32,
        game_id: String,
        timestamp_ms: u64,
    },
    SessionStart {
        player_id: Uuid,
        session_id: String,
        timestamp_ms: u64,
    },
    SessionEnd {
        player_id: Uuid,
        session_id: String,
        duration_ms: u64,
        timestamp_ms: u64,
    },
}

impl PlayerEvent {
    pub fn player_id(&self) -> Uuid {
        match self {
            PlayerEvent::MouseMove { player_id, .. } => *player_id,
            PlayerEvent::Click { player_id, .. } => *player_id,
            PlayerEvent::Bet { player_id, .. } => *player_id,
            PlayerEvent::Spin { player_id, .. } => *player_id,
            PlayerEvent::SessionStart { player_id, .. } => *player_id,
            PlayerEvent::SessionEnd { player_id, .. } => *player_id,
        }
    }
}

/// Unique player identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Uuid(pub [u8; 16]);

impl Uuid {
    pub fn new_v4() -> Self {
        // Simplified UUID v4 generation
        let mut bytes = [0u8; 16];
        for i in 0..16 {
            bytes[i] = rand_byte();
        }
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        Self(bytes)
    }
}

fn rand_byte() -> u8 {
    // Simplified - in real impl use proper RNG
    123
}

/// Anomaly types detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    BettingSyndicate,
    LatencyArbitrage,
    Botting,
    ChipDumping,
    Collusion,
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    pub anomaly_type: AnomalyType,
    pub confidence: f32,
    pub evidence: Vec<String>,
    pub player_ids: Vec<Uuid>,
    pub timestamp_ms: u64,
}

// ============================================================================
// WebSocket Stream Encoder
// ============================================================================

/// Encodes player interaction streams for Block AttnRes
pub struct WebSocketStreamEncoder {
    event_buffer: HashMap<Uuid, Vec<PlayerEvent>>,
    max_buffer_size: usize,
}

impl WebSocketStreamEncoder {
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            event_buffer: HashMap::new(),
            max_buffer_size,
        }
    }
    
    /// Process incoming player event
    pub fn encode_event(&mut self, event: PlayerEvent) -> Vec<u32> {
        let player_id = event.player_id();
        let events = self.event_buffer
            .entry(player_id)
            .or_insert_with(Vec::new);
        
        // Keep buffer bounded
        if events.len() >= self.max_buffer_size {
            events.remove(0);
        }
        events.push(event.clone());
        
        // Tokenize the event
        self.tokenize_event(&event)
    }
    
    /// Tokenize event for Block AttnRes input
    fn tokenize_event(&self, event: &PlayerEvent) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        match event {
            PlayerEvent::MouseMove { x, y, timestamp_ms, .. } => {
                // Hash-based tokenization
                tokens.push(hash_token("MOUSE_MOVE"));
                tokens.push(hash_coord(*x));
                tokens.push(hash_coord(*y));
                tokens.push(hash_ts(*timestamp_ms));
            }
            PlayerEvent::Click { button, latency_ms, timestamp_ms, .. } => {
                tokens.push(hash_token("CLICK"));
                tokens.push(*button as u32);
                tokens.push(hash_latency(*latency_ms));
                tokens.push(hash_ts(*timestamp_ms));
            }
            PlayerEvent::Bet { amount, game_id, timestamp_ms, .. } => {
                tokens.push(hash_token("BET"));
                tokens.push(hash_amount(*amount));
                tokens.push(hash_string(game_id));
                tokens.push(hash_ts(*timestamp_ms));
            }
            PlayerEvent::Spin { duration_ms, game_id, timestamp_ms, .. } => {
                tokens.push(hash_token("SPIN"));
                tokens.push(hash_duration(*duration_ms));
                tokens.push(hash_string(game_id));
                tokens.push(hash_ts(*timestamp_ms));
            }
            PlayerEvent::SessionStart { session_id, timestamp_ms, .. } => {
                tokens.push(hash_token("SESSION_START"));
                tokens.push(hash_string(session_id));
                tokens.push(hash_ts(*timestamp_ms));
            }
            PlayerEvent::SessionEnd { duration_ms, timestamp_ms, .. } => {
                tokens.push(hash_token("SESSION_END"));
                tokens.push(hash_duration(*duration_ms));
                tokens.push(hash_ts(*timestamp_ms));
            }
        }
        
        tokens
    }
}

/// Tokenize coordinate (quantized to grid)
fn hash_coord(coord: f32) -> u32 {
    let grid = ((coord + 1.0) * 50.0) as u32; // -1 to 1 → 0-100
    grid % 100
}

/// Tokenize latency
fn hash_latency(latency_ms: f32) -> u32 {
    (latency_ms as u32 / 10).min(100) // Quantize to 10ms buckets
}

/// Tokenize amount
fn hash_amount(amount: f64) -> u32 {
    (amount.log10() * 10.0) as u32 % 100 // Log scale
}

/// Tokenize duration
fn hash_duration(duration_ms: u32) -> u32 {
    (duration_ms / 100) as u32 % 1000 // 100ms buckets
}

/// Tokenize timestamp
fn hash_ts(ts: u64) -> u32 {
    ((ts / 1000) % 3600) as u32 // Second within hour
}

/// Hash string to token
fn hash_token(s: &str) -> u32 {
    s.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32)) % 65536
}

/// Hash string
fn hash_string(s: &str) -> u32 {
    hash_token(s)
}

// ============================================================================
// Intervention Head
// ============================================================================

/// Actions the fraud system can take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Intervention {
    LockAccount {
        player_id: Uuid,
        reason: String,
        duration_hours: Option<u32>,
    },
    TriggerKyc {
        player_id: Uuid,
        risk_level: f32,  // 0-1
    },
    AdjustLimits {
        player_id: Uuid,
        new_deposit_limit: f64,
        new_stake_limit: f64,
    },
    FreezeSession {
        player_id: Uuid,
        session_id: String,
        reason: String,
    },
    Alert {
        player_id: Uuid,
        alert_type: String,
        message: String,
    },
}

/// Intervention head for real-time account actions
pub trait InterventionHead {
    async fn execute(&self, intervention: &Intervention) -> Result<(), IgamingError>;
}

/// Mock implementation for testing
pub struct MockInterventionHead {
    actions: Vec<Intervention>,
}

impl MockInterventionHead {
    pub fn new() -> Self {
        Self { actions: Vec::new() }
    }
    
    pub fn get_actions(&self) -> &[Intervention] {
        &self.actions
    }
}

impl InterventionHead for MockInterventionHead {
    async fn execute(&self, intervention: &Intervention) -> Result<(), IgamingError> {
        // In real impl, would call external API
        println!("Intervention: {:?}", intervention);
        Ok(())
    }
}

// ============================================================================
// Gaming Commission Oracle
// ============================================================================

/// RTP (Return to Player) compliance rules
#[derive(Debug, Clone)]
pub struct RtpCompliance {
    pub min_rtp: f64,    // e.g., 0.95 (95%)
    pub max_rtp: f64,    // e.g., 0.98 (98%)
    pub game_id: String,
    pub certified_rtp: f64,
}

/// Intervention that might affect RTP
pub struct RtpAffectingIntervention {
    pub intervention: Intervention,
    pub affected_game_ids: Vec<String>,
    pub potential_rtp_delta: f64,  // Estimated impact
}

/// Gaming commission oracle for regulatory compliance
pub trait GamingCommissionOracle {
    /// Verify interventions don't violate RTP rules
    fn verify_rtp_compliance(
        &self,
        interventions: &[RtpAffectingIntervention],
    ) -> Result<bool, IgamingError>;
    
    /// Check bet limit compliance (e.g., German 1€ max)
    fn verify_bet_limits(&self, player_id: Uuid, amount: f64) -> Result<bool, IgamingError>;
    
    /// Verify cool-off period enforcement
    fn verify_cool_off(&self, player_id: Uuid) -> Result<bool, IgamingError>;
}

/// Mock implementation
pub struct MockGamingCommissionOracle {
    rtp_rules: HashMap<String, RtpCompliance>,
    bet_limits: HashMap<String, f64>,  // jurisdiction -> max bet
    cool_off_periods: HashMap<Uuid, u64>,  // player_id -> last intervention timestamp
}

impl MockGamingCommissionOracle {
    pub fn new() -> Self {
        let mut rtp_rules = HashMap::new();
        rtp_rules.insert(
            "slot_german".to_string(),
            RtpCompliance {
                min_rtp: 0.95,
                max_rtp: 0.98,
                game_id: "slot_german".to_string(),
                certified_rtp: 0.96,
            },
        );
        
        let mut bet_limits = HashMap::new();
        bet_limits.insert("DE".to_string(), 1.0);  // Germany: 1€ max
        bet_limits.insert("MT".to_string(), 10.0); // Malta: 10€ max
        
        Self {
            rtp_rules,
            bet_limits,
            cool_off_periods: HashMap::new(),
        }
    }
}

impl GamingCommissionOracle for MockGamingCommissionOracle {
    fn verify_rtp_compliance(
        &self,
        interventions: &[RtpAffectingIntervention],
    ) -> Result<bool, IgamingError> {
        for intervention in interventions {
            let game_id = &intervention.affected_game_ids.first().cloned().unwrap_or_default();
            if let Some(rule) = self.rtp_rules.get(game_id) {
                let total_delta = intervention.potential_rtp_delta;
                let new_rtp = rule.certified_rtp + total_delta;
                
                if new_rtp < rule.min_rtp || new_rtp > rule.max_rtp {
                    return Err(IgamingError::RtpViolation(format!(
                        "Intervention would alter RTP from {:.2}% to {:.2}% (allowed: {:.1}%-{:.1}%)",
                        rule.certified_rtp * 100.0,
                        new_rtp * 100.0,
                        rule.min_rtp * 100.0,
                        rule.max_rtp * 100.0
                    )));
                }
            }
        }
        Ok(true)
    }
    
    fn verify_bet_limits(&self, player_id: Uuid, amount: f64) -> Result<bool, IgamingError> {
        // Simplified - would check player jurisdiction
        let limit = self.bet_limits.get("MT").unwrap_or(&10.0);
        if amount > *limit {
            return Err(IgamingError::RtpViolation(format!(
                "Bet {} exceeds limit {}",
                amount, limit
            )));
        }
        Ok(true)
    }
    
    fn verify_cool_off(&self, player_id: Uuid) -> Result<bool, IgamingError> {
        if let Some(last_intervention) = self.cool_off_periods.get(&player_id) {
            let hours_since = (*last_intervention as f64) / 3600.0;
            if hours_since < 24.0 {
                return Err(IgamingError::RtpViolation(format!(
                    "Cool-off period active: {} hours since last intervention",
                    hours_since
                )));
            }
        }
        Ok(true)
    }
}

// ============================================================================
// Anomaly Detection
// ============================================================================

/// Anomaly detector for fraud patterns
pub struct AnomalyDetector {
    history_window_ms: u64,
    min_events_for_pattern: usize,
}

impl AnomalyDetector {
    pub fn new(history_window_ms: u64) -> Self {
        Self {
            history_window_ms,
            min_events_for_pattern: 5,
        }
    }
    
    /// Detect botting (superhuman click regularity)
    pub fn detect_bots(&self, events: &[PlayerEvent]) -> Option<AnomalyResult> {
        let clicks: Vec<u64> = events
            .iter()
            .filter_map(|e| match e {
                PlayerEvent::Click { timestamp_ms, .. } => Some(*timestamp_ms),
                _ => None,
            })
            .collect();
        
        if clicks.len() < self.min_events_for_pattern {
            return None;
        }
        
        // Calculate inter-click intervals
        let mut intervals = Vec::new();
        for i in 1..clicks.len() {
            intervals.push(clicks[i] - clicks[i-1]);
        }
        
        // Check standard deviation (bot = very low)
        let mean = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;
        let variance = intervals.iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>() / intervals.len() as f64;
        let std_dev = variance.sqrt();
        
        // Bot threshold: < 50ms std dev
        if std_dev < 50.0 {
            return Some(AnomalyResult {
                anomaly_type: AnomalyType::Botting,
                confidence: 1.0 - (std_dev / 50.0).min(1.0),
                evidence: vec![
                    format!("Inter-click std dev: {:.1}ms (bot threshold: <50ms)", std_dev),
                    format!("Click count: {}", clicks.len()),
                ],
                player_ids: vec![events[0].player_id()],
                timestamp_ms: clicks.last().copied().unwrap_or(0),
            });
        }
        
        None
    }
    
    /// Detect betting syndicate (correlated patterns across accounts)
    pub fn detect_syndicate(
        &self,
        player_events: &HashMap<Uuid, Vec<PlayerEvent>>,
    ) -> Option<AnomalyResult> {
        // Simplified: detect if multiple players bet at same timestamps
        let mut timestamp_counts: HashMap<u64, Vec<Uuid>> = HashMap::new();
        
        for (player_id, events) in player_events {
            for event in events {
                if let PlayerEvent::Bet { timestamp_ms, .. } = event {
                    timestamp_counts
                        .entry(*timestamp_ms)
                        .or_insert_with(Vec::new)
                        .push(*player_id);
                }
            }
        }
        
        // Find timestamps with multiple players
        let mut correlated_players = Vec::new();
        for (ts, players) in &timestamp_counts {
            if players.len() > 1 {
                correlated_players.extend(players.iter());
            }
        }
        
        if correlated_players.len() > 3 {
            return Some(AnomalyResult {
                anomaly_type: AnomalyType::BettingSyndicate,
                confidence: (correlated_players.len() as f32 / 10.0).min(1.0),
                evidence: vec![format!(
                    "{} players with correlated bets",
                    correlated_players.len()
                )],
                player_ids: correlated_players,
                timestamp_ms: 0,
            });
        }
        
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_websocket_encoder() {
        let mut encoder = WebSocketStreamEncoder::new(100);
        let player_id = Uuid::new_v4();
        
        let event = PlayerEvent::Click {
            player_id,
            button: 0,
            latency_ms: 150.0,
            timestamp_ms: 1000,
        };
        
        let tokens = encoder.encode_event(event);
        assert!(!tokens.is_empty());
    }
    
    #[test]
    fn test_rtp_compliance() {
        let oracle = MockGamingCommissionOracle::new();
        
        let intervention = RtpAffectingIntervention {
            intervention: Intervention::LockAccount {
                player_id: Uuid::new_v4(),
                reason: "test".to_string(),
                duration_hours: None,
            },
            affected_game_ids: vec!["slot_german".to_string()],
            potential_rtp_delta: 0.01,  // Would increase RTP
        };
        
        let result = oracle.verify_rtp_compliance(&[intervention]);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_bet_limits() {
        let oracle = MockGamingCommissionOracle::new();
        
        let result = oracle.verify_bet_limits(Uuid::new_v4(), 5.0);
        assert!(result.is_ok());
        
        let result = oracle.verify_bet_limits(Uuid::new_v4(), 100.0);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_bot_detection() {
        let detector = AnomalyDetector::new(60000);
        let player_id = Uuid::new_v4();
        
        // Simulate bot: regular 100ms clicks
        let events = vec![
            PlayerEvent::Click { player_id, button: 0, latency_ms: 0.0, timestamp_ms: 1000 },
            PlayerEvent::Click { player_id, button: 0, latency_ms: 0.0, timestamp_ms: 1100 },
            PlayerEvent::Click { player_id, button: 0, latency_ms: 0.0, timestamp_ms: 1200 },
            PlayerEvent::Click { player_id, button: 0, latency_ms: 0.0, timestamp_ms: 1300 },
            PlayerEvent::Click { player_id, button: 0, latency_ms: 0.0, timestamp_ms: 1400 },
        ];
        
        let result = detector.detect_bots(&events);
        assert!(result.is_some());
        assert_eq!(result.unwrap().anomaly_type, AnomalyType::Botting);
    }
}