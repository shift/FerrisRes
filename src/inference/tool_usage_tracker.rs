//! Tool Usage Tracker — Meta-Learning via Contextual Bandits
//!
//! Tracks (tool, context_embedding, result_quality) tuples over time.
//! Per-tool success rates, per-context-type success rates, time-decayed scores.
//! When selecting tools, weight by historical success in similar contexts.
//!
//! Phase 1: Simple usage tracking with exponential moving averages.
//! Phase 2 (future): LinUCB contextual bandit for exploration/exploitation.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Tool statistics
// ---------------------------------------------------------------------------

/// Statistics for a single tool.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolStats {
    /// Tool name.
    pub tool_name: String,
    /// Total invocations.
    pub total_calls: usize,
    /// Successful invocations.
    pub success_count: usize,
    /// Failed invocations.
    pub failure_count: usize,
    /// Exponential moving average of quality scores.
    pub ema_quality: f32,
    /// EMA decay rate (0.0–1.0).
    pub ema_decay: f32,
    /// Timestamp of last invocation.
    pub last_used: u64,
    /// Total quality across all invocations (for computing mean).
    pub total_quality: f32,
}

impl ToolStats {
    /// Create new stats for a tool.
    pub fn new(tool_name: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            total_calls: 0,
            success_count: 0,
            failure_count: 0,
            ema_quality: 0.0,
            ema_decay: 0.9,
            last_used: 0,
            total_quality: 0.0,
        }
    }

    /// Record a usage event.
    pub fn record(&mut self, quality: f32, timestamp: u64) {
        self.total_calls += 1;
        self.total_quality += quality;

        if quality >= 0.5 {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }

        // Update EMA
        if self.total_calls == 1 {
            self.ema_quality = quality;
        } else {
            self.ema_quality = self.ema_decay * self.ema_quality + (1.0 - self.ema_decay) * quality;
        }

        self.last_used = timestamp;
    }

    /// Success rate (0.0–1.0).
    pub fn success_rate(&self) -> f32 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.success_count as f32 / self.total_calls as f32
        }
    }

    /// Mean quality across all invocations.
    pub fn mean_quality(&self) -> f32 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.total_quality / self.total_calls as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Contextual stats
// ---------------------------------------------------------------------------

/// Statistics for a tool within a specific context bucket.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextToolStats {
    /// EMA quality for this (context, tool) pair.
    pub ema_quality: f32,
    /// Number of samples.
    pub sample_count: usize,
}

// ---------------------------------------------------------------------------
// Usage event
// ---------------------------------------------------------------------------

/// A single tool usage event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UsageEvent {
    /// Tool that was called.
    pub tool_name: String,
    /// Context embedding (quantized hash).
    pub context_hash: u64,
    /// Quality score of the result (0.0–1.0).
    pub quality: f32,
    /// Whether the tool succeeded.
    pub success: bool,
    /// Timestamp.
    pub timestamp: u64,
    /// Arguments hash (for dedup).
    pub args_hash: u64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the tool usage tracker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolUsageTrackerConfig {
    /// EMA decay rate for quality tracking.
    pub ema_decay: f32,
    /// Number of context buckets (quantization granularity).
    pub context_buckets: u64,
    /// Minimum samples before recommending a tool contextually.
    pub min_context_samples: usize,
    /// Maximum events to keep in history.
    pub max_history: usize,
    /// Time decay half-life in seconds (0 = no decay).
    pub time_decay_half_life_secs: f64,
    /// Quality threshold for "success".
    pub success_threshold: f32,
}

impl Default for ToolUsageTrackerConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.9,
            context_buckets: 1024,
            min_context_samples: 3,
            max_history: 10000,
            time_decay_half_life_secs: 3600.0, // 1 hour
            success_threshold: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// ToolUsageTracker
// ---------------------------------------------------------------------------

/// Tracks tool usage over time and recommends tools based on contextual success.
pub struct ToolUsageTracker {
    config: ToolUsageTrackerConfig,
    /// Per-tool global statistics.
    tool_stats: HashMap<String, ToolStats>,
    /// Contextual statistics: (context_hash → tool_name → stats).
    context_stats: HashMap<u64, HashMap<String, ContextToolStats>>,
    /// Usage event history.
    history: Vec<UsageEvent>,
    /// Tool recommendation scores cache.
    recommendation_cache: HashMap<String, f32>,
}

impl ToolUsageTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: ToolUsageTrackerConfig) -> Self {
        Self {
            config,
            tool_stats: HashMap::new(),
            context_stats: HashMap::new(),
            history: Vec::new(),
            recommendation_cache: HashMap::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_tracker() -> Self {
        Self::new(ToolUsageTrackerConfig::default())
    }

    /// Get the tracker configuration.
    pub fn config(&self) -> &ToolUsageTrackerConfig {
        &self.config
    }

    /// Number of tracked tools.
    pub fn tracked_tool_count(&self) -> usize {
        self.tool_stats.len()
    }

    /// Total recorded events.
    pub fn total_events(&self) -> usize {
        self.history.len()
    }

    /// Number of context buckets with data.
    pub fn context_bucket_count(&self) -> usize {
        self.context_stats.len()
    }

    // ---- Main API ----

    /// Record a tool usage event.
    pub fn record(
        &mut self,
        tool_name: &str,
        context_embedding: &[f32],
        quality: f32,
    ) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let context_hash = Self::quantize_embedding(context_embedding, self.config.context_buckets);
        let success = quality >= self.config.success_threshold;

        // Update global tool stats
        self.tool_stats
            .entry(tool_name.to_string())
            .or_insert_with(|| ToolStats::new(tool_name))
            .record(quality, timestamp);

        // Update contextual stats
        let ctx_entry = self.context_stats
            .entry(context_hash)
            .or_default();
        let tool_ctx = ctx_entry
            .entry(tool_name.to_string())
            .or_insert(ContextToolStats {
                ema_quality: 0.0,
                sample_count: 0,
            });
        if tool_ctx.sample_count == 0 {
            tool_ctx.ema_quality = quality;
        } else {
            tool_ctx.ema_quality = self.config.ema_decay * tool_ctx.ema_quality
                + (1.0 - self.config.ema_decay) * quality;
        }
        tool_ctx.sample_count += 1;

        // Record event
        let event = UsageEvent {
            tool_name: tool_name.to_string(),
            context_hash,
            quality,
            success,
            timestamp,
            args_hash: 0,
        };
        self.history.push(event);

        // Trim history if needed
        if self.history.len() > self.config.max_history {
            let excess = self.history.len() - self.config.max_history;
            self.history.drain(..excess);
        }

        // Invalidate cache
        self.recommendation_cache.clear();
    }

    /// Get the best tool for a given context from a list of candidates.
    ///
    /// Uses contextual stats if enough samples exist, falls back to global stats.
    /// Returns `None` if no candidates have been tracked.
    pub fn best_tool_for_context(
        &self,
        context_embedding: &[f32],
        candidates: &[String],
    ) -> Option<String> {
        let context_hash = Self::quantize_embedding(context_embedding, self.config.context_buckets);

        // Try contextual stats first
        if let Some(ctx_tools) = self.context_stats.get(&context_hash) {
            let best = candidates.iter()
                .filter_map(|t| {
                    ctx_tools.get(t).and_then(|stats| {
                        if stats.sample_count >= self.config.min_context_samples {
                            Some((t, stats.ema_quality))
                        } else {
                            None
                        }
                    })
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(t, _)| t.clone());

            if best.is_some() {
                return best;
            }
        }

        // Fall back to global stats
        candidates.iter()
            .filter_map(|t| {
                self.tool_stats.get(t).map(|s| {
                    // Combine success rate with EMA quality
                    let score = 0.5 * s.success_rate() + 0.5 * s.ema_quality;
                    (t, score)
                })
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(t, _)| t.clone())
    }

    /// Get a ranked list of tools for a context.
    pub fn rank_tools_for_context(
        &self,
        context_embedding: &[f32],
        candidates: &[String],
    ) -> Vec<(String, f32)> {
        let context_hash = Self::quantize_embedding(context_embedding, self.config.context_buckets);

        let mut ranked: Vec<(String, f32)> = candidates.iter()
            .map(|t| {
                let ctx_score = self.context_stats
                    .get(&context_hash)
                    .and_then(|m| m.get(t))
                    .filter(|s| s.sample_count >= self.config.min_context_samples)
                    .map(|s| s.ema_quality);

                let global_score = self.tool_stats
                    .get(t)
                    .map(|s| 0.5 * s.success_rate() + 0.5 * s.ema_quality);

                let score = ctx_score
                    .or(global_score)
                    .unwrap_or(0.5); // Default score for unknown tools

                (t.clone(), score)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Get global statistics for a tool.
    pub fn get_tool_stats(&self, tool_name: &str) -> Option<&ToolStats> {
        self.tool_stats.get(tool_name)
    }

    /// Get contextual quality for a (context, tool) pair.
    pub fn get_context_quality(
        &self,
        context_embedding: &[f32],
        tool_name: &str,
    ) -> Option<f32> {
        let hash = Self::quantize_embedding(context_embedding, self.config.context_buckets);
        self.context_stats
            .get(&hash)
            .and_then(|m| m.get(tool_name))
            .map(|s| s.ema_quality)
    }

    /// Get tools that are performing poorly in a context.
    pub fn poorly_performing_tools(
        &self,
        context_embedding: &[f32],
        threshold: f32,
    ) -> Vec<(String, f32)> {
        let context_hash = Self::quantize_embedding(context_embedding, self.config.context_buckets);

        if let Some(ctx_tools) = self.context_stats.get(&context_hash) {
            let mut poor: Vec<_> = ctx_tools.iter()
                .filter(|(_, s)| s.sample_count >= self.config.min_context_samples && s.ema_quality < threshold)
                .map(|(name, s)| (name.clone(), s.ema_quality))
                .collect();
            poor.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            poor
        } else {
            vec![]
        }
    }

    /// Compute time-decayed quality for a tool.
    pub fn time_decayed_quality(&self, tool_name: &str) -> f32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0) as f64;

        if self.config.time_decay_half_life_secs <= 0.0 {
            return self.tool_stats.get(tool_name)
                .map(|s| s.ema_quality)
                .unwrap_or(0.0);
        }

        let events: Vec<&UsageEvent> = self.history.iter()
            .filter(|e| e.tool_name == tool_name)
            .collect();

        if events.is_empty() {
            return 0.0;
        }

        let half_life = self.config.time_decay_half_life_secs;
        let mut weighted_sum = 0.0f64;
        let mut weight_total = 0.0f64;

        for event in &events {
            let age = now - event.timestamp as f64;
            let weight = (-0.693 * age / half_life).exp(); // ln(2) ≈ 0.693
            weighted_sum += weight * event.quality as f64;
            weight_total += weight;
        }

        if weight_total > 0.0 {
            (weighted_sum / weight_total) as f32
        } else {
            0.0
        }
    }

    /// Get usage history for a specific tool.
    pub fn tool_history(&self, tool_name: &str) -> Vec<&UsageEvent> {
        self.history.iter()
            .filter(|e| e.tool_name == tool_name)
            .collect()
    }

    /// Get recent usage events (last N).
    pub fn recent_events(&self, n: usize) -> Vec<&UsageEvent> {
        let start = self.history.len().saturating_sub(n);
        self.history[start..].iter().collect()
    }

    // ---- Quantization ----

    /// Quantize an embedding to a bucket hash for efficient context lookup.
    ///
    /// Uses a simple hash of the embedding to map to one of `num_buckets` bins.
    pub fn quantize_embedding(embedding: &[f32], num_buckets: u64) -> u64 {
        // Simple hash: sum of quantized dimensions
        let mut hash: u64 = 0;
        for (i, &val) in embedding.iter().enumerate() {
            // Quantize each dimension to 8 bits
            let quantized = ((val * 100.0) as i64).wrapping_mul((i + 1) as i64);
            hash = hash.wrapping_add(quantized as u64);
        }
        hash % num_buckets
    }

    // ---- Stats ----

    /// Get global statistics summary.
    pub fn stats(&self) -> ToolUsageTrackerStats {
        let total_calls: usize = self.tool_stats.values().map(|s| s.total_calls).sum();
        let total_successes: usize = self.tool_stats.values().map(|s| s.success_count).sum();
        let avg_quality = if total_calls > 0 {
            self.tool_stats.values()
                .map(|s| s.total_quality)
                .sum::<f32>() / total_calls as f32
        } else {
            0.0
        };

        ToolUsageTrackerStats {
            tracked_tools: self.tool_stats.len(),
            total_events: self.history.len(),
            total_calls,
            total_successes,
            avg_quality,
            context_buckets_used: self.context_stats.len(),
        }
    }

    /// List all tracked tool names.
    pub fn tracked_tools(&self) -> Vec<&str> {
        self.tool_stats.keys().map(|s| s.as_str()).collect()
    }

    // ---- Persistence ----

    /// Save tracker state to JSON.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let data = TrackerState {
            config: self.config.clone(),
            tool_stats: self.tool_stats.clone(),
            context_stats: self.context_stats.clone(),
            history: self.history.clone(),
        };
        let json = serde_json::to_string_pretty(&data).map_err(|e| format!("Serialize error: {}", e))?;

        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let tmp_path = format!("{}.tmp", path.display());
        std::fs::write(&tmp_path, &json).map_err(|e| format!("Write error: {}", e))?;
        std::fs::rename(&tmp_path, path).map_err(|e| format!("Rename error: {}", e))?;

        Ok(())
    }

    /// Load tracker state from JSON.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
        let data: TrackerState = serde_json::from_str(&json).map_err(|e| format!("Deserialize error: {}", e))?;

        Ok(Self {
            config: data.config,
            tool_stats: data.tool_stats,
            context_stats: data.context_stats,
            history: data.history,
            recommendation_cache: HashMap::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Serializable state
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct TrackerState {
    config: ToolUsageTrackerConfig,
    tool_stats: HashMap<String, ToolStats>,
    context_stats: HashMap<u64, HashMap<String, ContextToolStats>>,
    history: Vec<UsageEvent>,
}

// ---------------------------------------------------------------------------
// Stats summary
// ---------------------------------------------------------------------------

/// Summary statistics for the tool usage tracker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolUsageTrackerStats {
    pub tracked_tools: usize,
    pub total_events: usize,
    pub total_calls: usize,
    pub total_successes: usize,
    pub avg_quality: f32,
    pub context_buckets_used: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    #[test]
    fn test_tool_stats_record() {
        let mut stats = ToolStats::new("test_tool");
        stats.record(0.9, 100);
        stats.record(0.8, 200);
        stats.record(0.1, 300); // Failure

        assert_eq!(stats.total_calls, 3);
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.failure_count, 1);
        assert!(stats.ema_quality > 0.0);
        assert!(stats.success_rate() > 0.5);
    }

    #[test]
    fn test_tool_stats_ema() {
        let mut stats = ToolStats::new("test_tool");
        stats.ema_decay = 0.5;
        stats.record(1.0, 100);
        assert!((stats.ema_quality - 1.0).abs() < 0.01);

        stats.record(0.0, 200);
        // EMA = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert!((stats.ema_quality - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_tool_stats_mean_quality() {
        let mut stats = ToolStats::new("test_tool");
        stats.record(0.5, 100);
        stats.record(1.0, 200);
        assert!((stats.mean_quality() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_tool_stats_empty() {
        let stats = ToolStats::new("empty");
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.mean_quality(), 0.0);
    }

    #[test]
    fn test_record_usage() {
        let mut tracker = ToolUsageTracker::default_tracker();
        let ctx = make_embedding(&[1.0, 0.0, 0.0]);

        tracker.record("tool_a", &ctx, 0.9);
        tracker.record("tool_a", &ctx, 0.8);
        tracker.record("tool_b", &ctx, 0.3);

        assert_eq!(tracker.tracked_tool_count(), 2);
        assert_eq!(tracker.total_events(), 3);
    }

    #[test]
    fn test_best_tool_for_context() {
        let mut tracker = ToolUsageTracker::default_tracker();
        tracker.config.min_context_samples = 1;

        let ctx1 = make_embedding(&[1.0, 0.0]);
        let ctx2 = make_embedding(&[0.0, 1.0]);

        // tool_a is great in ctx1, poor in ctx2
        tracker.record("tool_a", &ctx1, 0.95);
        tracker.record("tool_a", &ctx1, 0.9);
        tracker.record("tool_a", &ctx2, 0.2);

        // tool_b is poor in ctx1, great in ctx2
        tracker.record("tool_b", &ctx1, 0.3);
        tracker.record("tool_b", &ctx2, 0.9);
        tracker.record("tool_b", &ctx2, 0.95);

        let candidates = vec!["tool_a".into(), "tool_b".into()];

        let best1 = tracker.best_tool_for_context(&ctx1, &candidates);
        assert_eq!(best1.as_deref(), Some("tool_a"));

        let best2 = tracker.best_tool_for_context(&ctx2, &candidates);
        assert_eq!(best2.as_deref(), Some("tool_b"));
    }

    #[test]
    fn test_best_tool_fallback_to_global() {
        let mut tracker = ToolUsageTracker::default_tracker();
        // High min_context_samples forces fallback to global stats
        tracker.config.min_context_samples = 100;

        tracker.record("good_tool", &[1.0], 0.95);
        tracker.record("good_tool", &[1.0], 0.9);
        tracker.record("bad_tool", &[1.0], 0.2);

        let candidates = vec!["good_tool".into(), "bad_tool".into()];
        let best = tracker.best_tool_for_context(&[1.0], &candidates);
        assert_eq!(best.as_deref(), Some("good_tool"));
    }

    #[test]
    fn test_rank_tools() {
        let mut tracker = ToolUsageTracker::default_tracker();
        tracker.config.min_context_samples = 1;

        tracker.record("alpha", &[1.0], 0.9);
        tracker.record("beta", &[1.0], 0.5);
        tracker.record("gamma", &[1.0], 0.8);

        let ranked = tracker.rank_tools_for_context(
            &[1.0],
            &["alpha".into(), "beta".into(), "gamma".into()],
        );

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].0, "alpha");
        assert!(ranked[0].1 > ranked[1].1);
    }

    #[test]
    fn test_poorly_performing_tools() {
        let mut tracker = ToolUsageTracker::default_tracker();
        tracker.config.min_context_samples = 2;

        tracker.record("good", &[1.0], 0.9);
        tracker.record("good", &[1.0], 0.85);
        tracker.record("bad", &[1.0], 0.2);
        tracker.record("bad", &[1.0], 0.15);

        let poor = tracker.poorly_performing_tools(&[1.0], 0.5);
        assert_eq!(poor.len(), 1);
        assert_eq!(poor[0].0, "bad");
    }

    #[test]
    fn test_time_decayed_quality() {
        let mut tracker = ToolUsageTracker::default_tracker();

        // Record events with explicit timestamps
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Manually create history entries
        tracker.tool_stats.insert("test".into(), {
            let mut s = ToolStats::new("test");
            s.total_calls = 3;
            s.success_count = 3;
            s.ema_quality = 0.8;
            s.total_quality = 2.4;
            s
        });

        tracker.history.push(UsageEvent {
            tool_name: "test".into(),
            context_hash: 0,
            quality: 1.0,
            success: true,
            timestamp: now - 100,      // Recent
            args_hash: 0,
        });
        tracker.history.push(UsageEvent {
            tool_name: "test".into(),
            context_hash: 0,
            quality: 0.6,
            success: true,
            timestamp: now - 10000,    // Older
            args_hash: 0,
        });

        let quality = tracker.time_decayed_quality("test");
        assert!(quality > 0.0);
        // Recent high-quality event should pull it towards 1.0
        assert!(quality > 0.6);
    }

    #[test]
    fn test_get_tool_stats() {
        let mut tracker = ToolUsageTracker::default_tracker();
        tracker.record("my_tool", &[1.0], 0.8);

        let stats = tracker.get_tool_stats("my_tool").unwrap();
        assert_eq!(stats.total_calls, 1);
        assert!(tracker.get_tool_stats("unknown").is_none());
    }

    #[test]
    fn test_get_context_quality() {
        let mut tracker = ToolUsageTracker::default_tracker();
        tracker.record("tool", &[1.0, 0.0], 0.85);

        let quality = tracker.get_context_quality(&[1.0, 0.0], "tool");
        assert!(quality.is_some());
        assert!(quality.unwrap() > 0.8);

        // Different context
        let quality2 = tracker.get_context_quality(&[0.0, 1.0], "tool");
        assert!(quality2.is_none());
    }

    #[test]
    fn test_quantize_embedding() {
        let emb = vec![1.0, 2.0, 3.0];
        let h1 = ToolUsageTracker::quantize_embedding(&emb, 1024);
        let h2 = ToolUsageTracker::quantize_embedding(&emb, 1024);
        assert_eq!(h1, h2);
        assert!(h1 < 1024);

        let emb2 = vec![3.0, 2.0, 1.0];
        let h3 = ToolUsageTracker::quantize_embedding(&emb2, 1024);
        assert_ne!(h1, h3); // Different embeddings → different buckets
    }

    #[test]
    fn test_history_trim() {
        let config = ToolUsageTrackerConfig {
            max_history: 5,
            ..Default::default()
        };
        let mut tracker = ToolUsageTracker::new(config);

        for i in 0..10 {
            tracker.record("tool", &[i as f32], 0.5);
        }

        assert!(tracker.total_events() <= 5);
    }

    #[test]
    fn test_recent_events() {
        let mut tracker = ToolUsageTracker::default_tracker();
        for i in 0..5 {
            tracker.record("tool", &[i as f32], 0.5);
        }

        let recent = tracker.recent_events(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_tool_history() {
        let mut tracker = ToolUsageTracker::default_tracker();
        tracker.record("alpha", &[1.0], 0.9);
        tracker.record("beta", &[1.0], 0.5);
        tracker.record("alpha", &[2.0], 0.8);

        let alpha_history = tracker.tool_history("alpha");
        assert_eq!(alpha_history.len(), 2);

        let beta_history = tracker.tool_history("beta");
        assert_eq!(beta_history.len(), 1);
    }

    #[test]
    fn test_tracked_tools() {
        let mut tracker = ToolUsageTracker::default_tracker();
        tracker.record("a", &[1.0], 0.5);
        tracker.record("b", &[1.0], 0.5);
        tracker.record("c", &[1.0], 0.5);

        let mut tools = tracker.tracked_tools();
        tools.sort();
        assert_eq!(tools, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_stats_summary() {
        let mut tracker = ToolUsageTracker::default_tracker();
        tracker.record("a", &[1.0], 0.9);
        tracker.record("a", &[1.0], 0.8);

        let stats = tracker.stats();
        assert_eq!(stats.tracked_tools, 1);
        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.total_successes, 2);
        assert!(stats.avg_quality > 0.8);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("ferrisres_tracker_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tracker.json");

        let mut tracker = ToolUsageTracker::default_tracker();
        tracker.record("tool_a", &[1.0, 0.0], 0.9);
        tracker.record("tool_b", &[0.0, 1.0], 0.5);

        tracker.save(&path).unwrap();

        let loaded = ToolUsageTracker::load(&path).unwrap();
        assert_eq!(loaded.tracked_tool_count(), 2);
        assert_eq!(loaded.total_events(), 2);
        assert_eq!(loaded.context_bucket_count(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
