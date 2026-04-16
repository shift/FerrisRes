//! Proactive Controller — Bounded Autonomous Behavior
//!
//! Based on proactive assistants, bounded autonomy (human-in-the-loop),
//! and goal maintenance. The system monitors its own state and takes
//! proactive actions when initiative signals fire (concept degradation,
//! tool obsolescence, knowledge gaps).
//!
//! Autonomy levels:
//!   Level 0: Reactive — only respond to user prompts
//!   Level 1: Suggestive — suggest actions, user confirms
//!   Level 2: Semi-autonomous — act, user reviews after
//!   Level 3: Fully autonomous — act independently

// ---------------------------------------------------------------------------
// Autonomy levels
// ---------------------------------------------------------------------------

/// Autonomy level for the proactive controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AutonomyLevel {
    /// Only respond to user prompts.
    Reactive,
    /// Suggest actions, user confirms.
    Suggestive,
    /// Act, user reviews after.
    SemiAutonomous,
    /// Act independently.
    FullyAutonomous,
}

impl AutonomyLevel {
    /// Numeric level.
    pub fn level(&self) -> usize {
        match self {
            AutonomyLevel::Reactive => 0,
            AutonomyLevel::Suggestive => 1,
            AutonomyLevel::SemiAutonomous => 2,
            AutonomyLevel::FullyAutonomous => 3,
        }
    }

    /// Whether this level allows an action requiring the given minimum level.
    pub fn allows(&self, required: AutonomyLevel) -> bool {
        self.level() >= required.level()
    }
}

// ---------------------------------------------------------------------------
// Goals
// ---------------------------------------------------------------------------

/// Source of a proactive goal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GoalSource {
    /// Goal from user prompt.
    UserPrompt(String),
    /// Self-generated (e.g., "concept X quality dropping").
    SelfGenerated(String),
    /// Maintenance task (e.g., "compress concepts").
    Maintenance(String),
}

/// A proactive goal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProactiveGoal {
    /// Unique ID.
    pub id: String,
    /// Goal description.
    pub description: String,
    /// Source.
    pub source: GoalSource,
    /// Priority (0.0–1.0).
    pub priority: f32,
    /// Minimum autonomy level required.
    pub min_autonomy: AutonomyLevel,
    /// Creation timestamp.
    pub created_at: u64,
    /// Deadline (optional).
    pub deadline: Option<u64>,
    /// Whether completed.
    pub completed: bool,
}

// ---------------------------------------------------------------------------
// Action records
// ---------------------------------------------------------------------------

/// Record of a proactive action taken.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ActionRecord {
    /// Action type.
    pub action: String,
    /// Description of what was done.
    pub description: String,
    /// State before action (JSON snapshot).
    pub before_state: String,
    /// State after action (JSON snapshot).
    pub after_state: String,
    /// Associated goal ID.
    pub goal_id: Option<String>,
    /// Timestamp.
    pub timestamp: u64,
    /// Whether the action was auto-approved or needs review.
    pub approved: bool,
    /// Whether the action was rolled back.
    pub rolled_back: bool,
}

// ---------------------------------------------------------------------------
// Initiative signals
// ---------------------------------------------------------------------------

/// A signal that the system should take proactive action.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InitiativeSignal {
    /// Signal type.
    pub signal_type: SignalType,
    /// Description.
    pub description: String,
    /// Priority (0.0–1.0).
    pub priority: f32,
    /// Minimum autonomy level required.
    pub min_autonomy: AutonomyLevel,
    /// Related concept/tool IDs.
    pub related_ids: Vec<String>,
}

/// Types of initiative signals.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SignalType {
    /// Concept quality degradation.
    ConceptDegradation,
    /// Tool success rate dropping.
    ToolObsolescence,
    /// No concepts found for a prompt (knowledge gap).
    KnowledgeGap,
    /// Memory near capacity (need compression).
    MemoryPressure,
    /// Stale concept (not accessed recently).
    StaleConcept,
    /// Post-generation quality below threshold.
    LowQualityOutput,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the proactive controller.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProactiveConfig {
    /// Current autonomy level.
    pub autonomy_level: AutonomyLevel,
    /// Maximum proactive actions per cycle.
    pub max_actions_per_cycle: usize,
    /// Quality threshold below which concept degradation signal fires.
    pub concept_quality_threshold: f32,
    /// Success rate below which tool obsolescence signal fires.
    pub tool_success_threshold: f32,
    /// Number of generations without concept retrieval to trigger knowledge gap.
    pub knowledge_gap_generations: usize,
    /// Memory usage percentage to trigger memory pressure.
    pub memory_pressure_threshold: f32,
    /// Seconds since last access to consider a concept stale.
    pub stale_concept_seconds: u64,
    /// Quality threshold for post-generation review.
    pub output_quality_threshold: f32,
    /// Maximum action log entries.
    pub max_action_log: usize,
}

impl Default for ProactiveConfig {
    fn default() -> Self {
        Self {
            autonomy_level: AutonomyLevel::Suggestive,
            max_actions_per_cycle: 5,
            concept_quality_threshold: 0.5,
            tool_success_threshold: 0.7,
            knowledge_gap_generations: 3,
            memory_pressure_threshold: 0.8,
            stale_concept_seconds: 3600, // 1 hour
            output_quality_threshold: 0.6,
            max_action_log: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// ProactiveController
// ---------------------------------------------------------------------------

/// Monitors system state and takes proactive actions within autonomy bounds.
pub struct ProactiveController {
    config: ProactiveConfig,
    /// Active goals.
    goals: Vec<ProactiveGoal>,
    /// Action log.
    action_log: Vec<ActionRecord>,
    /// Consecutive generations without concept retrieval.
    generations_without_concepts: usize,
    /// Goal counter.
    next_goal_id: u64,
}

impl ProactiveController {
    /// Create a new proactive controller.
    pub fn new(config: ProactiveConfig) -> Self {
        Self {
            config,
            goals: Vec::new(),
            action_log: Vec::new(),
            generations_without_concepts: 0,
            next_goal_id: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_controller() -> Self {
        Self::new(ProactiveConfig::default())
    }

    /// Get the current autonomy level.
    pub fn autonomy_level(&self) -> AutonomyLevel {
        self.config.autonomy_level
    }

    /// Set the autonomy level.
    pub fn set_autonomy_level(&mut self, level: AutonomyLevel) {
        self.config.autonomy_level = level;
    }

    /// Get configuration.
    pub fn config(&self) -> &ProactiveConfig {
        &self.config
    }

    /// Number of active goals.
    pub fn active_goal_count(&self) -> usize {
        self.goals.iter().filter(|g| !g.completed).count()
    }

    /// Number of logged actions.
    pub fn action_count(&self) -> usize {
        self.action_log.len()
    }

    /// Get action log.
    pub fn action_log(&self) -> &[ActionRecord] {
        &self.action_log
    }

    // ---- Signal detection ----

    /// Check for concept quality degradation.
    pub fn check_concept_degradation(
        &self,
        concept_qualities: &[(&str, f32)],
    ) -> Vec<InitiativeSignal> {
        if !self.config.autonomy_level.allows(AutonomyLevel::Suggestive) {
            return vec![];
        }

        concept_qualities.iter()
            .filter(|(_, q)| *q < self.config.concept_quality_threshold)
            .map(|(id, q)| InitiativeSignal {
                signal_type: SignalType::ConceptDegradation,
                description: format!("Concept '{}' quality degraded to {:.2}", id, q),
                priority: 1.0 - q,
                min_autonomy: AutonomyLevel::Suggestive,
                related_ids: vec![id.to_string()],
            })
            .collect()
    }

    /// Check for tool obsolescence.
    pub fn check_tool_obsolescence(
        &self,
        tool_stats: &[(&str, f32)],
    ) -> Vec<InitiativeSignal> {
        if !self.config.autonomy_level.allows(AutonomyLevel::SemiAutonomous) {
            return vec![];
        }

        tool_stats.iter()
            .filter(|(_, rate)| *rate < self.config.tool_success_threshold)
            .map(|(name, rate)| InitiativeSignal {
                signal_type: SignalType::ToolObsolescence,
                description: format!("Tool '{}' success rate dropped to {:.2}", name, rate),
                priority: 1.0 - rate,
                min_autonomy: AutonomyLevel::SemiAutonomous,
                related_ids: vec![name.to_string()],
            })
            .collect()
    }

    /// Record a generation with/without concept retrieval.
    pub fn record_generation(&mut self, concepts_retrieved: usize) {
        if concepts_retrieved == 0 {
            self.generations_without_concepts += 1;
        } else {
            self.generations_without_concepts = 0;
        }
    }

    /// Check for knowledge gaps.
    pub fn check_knowledge_gap(&self) -> Option<InitiativeSignal> {
        if !self.config.autonomy_level.allows(AutonomyLevel::Suggestive) {
            return None;
        }

        if self.generations_without_concepts >= self.config.knowledge_gap_generations {
            Some(InitiativeSignal {
                signal_type: SignalType::KnowledgeGap,
                description: format!(
                    "No concepts retrieved for {} consecutive generations",
                    self.generations_without_concepts
                ),
                priority: 0.7,
                min_autonomy: AutonomyLevel::Suggestive,
                related_ids: vec![],
            })
        } else {
            None
        }
    }

    /// Check for memory pressure.
    pub fn check_memory_pressure(&self, usage_ratio: f32) -> Option<InitiativeSignal> {
        if usage_ratio >= self.config.memory_pressure_threshold {
            Some(InitiativeSignal {
                signal_type: SignalType::MemoryPressure,
                description: format!("Memory usage at {:.0}%", usage_ratio * 100.0),
                priority: usage_ratio,
                min_autonomy: AutonomyLevel::SemiAutonomous,
                related_ids: vec![],
            })
        } else {
            None
        }
    }

    /// Check for post-generation quality.
    pub fn check_output_quality(&self, quality: f32) -> Option<InitiativeSignal> {
        if quality < self.config.output_quality_threshold {
            Some(InitiativeSignal {
                signal_type: SignalType::LowQualityOutput,
                description: format!("Last output quality: {:.2} (below threshold {:.2})", quality, self.config.output_quality_threshold),
                priority: 1.0 - quality,
                min_autonomy: AutonomyLevel::Reactive, // Even reactive level can flag this
                related_ids: vec![],
            })
        } else {
            None
        }
    }

    // ---- Goal management ----

    /// Create a goal from an initiative signal.
    pub fn create_goal_from_signal(&mut self, signal: &InitiativeSignal) -> Option<ProactiveGoal> {
        if !self.config.autonomy_level.allows(signal.min_autonomy) {
            return None;
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let goal = ProactiveGoal {
            id: format!("goal_{}", self.next_goal_id),
            description: signal.description.clone(),
            source: GoalSource::SelfGenerated(format!("{:?}", signal.signal_type)),
            priority: signal.priority,
            min_autonomy: signal.min_autonomy,
            created_at: timestamp,
            deadline: None,
            completed: false,
        };
        self.next_goal_id += 1;
        self.goals.push(goal.clone());
        Some(goal)
    }

    /// Get the highest-priority actionable goal.
    pub fn next_goal(&self) -> Option<&ProactiveGoal> {
        self.goals.iter()
            .filter(|g| !g.completed)
            .filter(|g| self.config.autonomy_level.allows(g.min_autonomy))
            .max_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Complete a goal.
    pub fn complete_goal(&mut self, goal_id: &str) {
        if let Some(goal) = self.goals.iter_mut().find(|g| g.id == goal_id) {
            goal.completed = true;
        }
    }

    // ---- Action logging ----

    /// Log an action.
    pub fn log_action(
        &mut self,
        action: impl Into<String>,
        description: impl Into<String>,
        before_state: impl Into<String>,
        after_state: impl Into<String>,
        goal_id: Option<String>,
    ) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let record = ActionRecord {
            action: action.into(),
            description: description.into(),
            before_state: before_state.into(),
            after_state: after_state.into(),
            goal_id,
            timestamp,
            approved: self.config.autonomy_level.allows(AutonomyLevel::SemiAutonomous),
            rolled_back: false,
        };

        self.action_log.push(record);

        // Trim log if needed
        if self.action_log.len() > self.config.max_action_log {
            let excess = self.action_log.len() - self.config.max_action_log;
            self.action_log.drain(..excess);
        }
    }

    /// Rollback the most recent action.
    pub fn rollback_last(&mut self) -> Option<&ActionRecord> {
        if let Some(record) = self.action_log.last_mut() {
            record.rolled_back = true;
            return Some(self.action_log.last().unwrap());
        }
        None
    }

    /// Get actions pending review (suggestive mode).
    pub fn pending_review(&self) -> Vec<&ActionRecord> {
        self.action_log.iter()
            .filter(|a| !a.approved && !a.rolled_back)
            .collect()
    }

    /// Approve an action.
    pub fn approve_action(&mut self, index: usize) {
        if let Some(record) = self.action_log.get_mut(index) {
            record.approved = true;
        }
    }

    // ---- Stats ----

    /// Get controller statistics.
    pub fn stats(&self) -> ProactiveStats {
        let approved = self.action_log.iter().filter(|a| a.approved).count();
        let rolled_back = self.action_log.iter().filter(|a| a.rolled_back).count();

        ProactiveStats {
            autonomy_level: self.config.autonomy_level,
            active_goals: self.active_goal_count(),
            total_actions: self.action_log.len(),
            approved_actions: approved,
            rolled_back_actions: rolled_back,
        }
    }
}

/// Proactive controller statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProactiveStats {
    pub autonomy_level: AutonomyLevel,
    pub active_goals: usize,
    pub total_actions: usize,
    pub approved_actions: usize,
    pub rolled_back_actions: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autonomy_levels() {
        assert!(AutonomyLevel::FullyAutonomous.allows(AutonomyLevel::Reactive));
        assert!(AutonomyLevel::SemiAutonomous.allows(AutonomyLevel::Suggestive));
        assert!(!AutonomyLevel::Reactive.allows(AutonomyLevel::Suggestive));
        assert!(!AutonomyLevel::Suggestive.allows(AutonomyLevel::SemiAutonomous));
    }

    #[test]
    fn test_autonomy_level_numeric() {
        assert_eq!(AutonomyLevel::Reactive.level(), 0);
        assert_eq!(AutonomyLevel::Suggestive.level(), 1);
        assert_eq!(AutonomyLevel::SemiAutonomous.level(), 2);
        assert_eq!(AutonomyLevel::FullyAutonomous.level(), 3);
    }

    #[test]
    fn test_check_concept_degradation() {
        let controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::Suggestive,
            ..Default::default()
        });

        let qualities = vec![("good_concept", 0.9f32), ("bad_concept", 0.3f32)];
        let signals = controller.check_concept_degradation(&qualities);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].related_ids[0], "bad_concept");
    }

    #[test]
    fn test_check_concept_degradation_reactive() {
        let controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::Reactive,
            ..Default::default()
        });

        let qualities = vec![("bad", 0.1f32)];
        let signals = controller.check_concept_degradation(&qualities);
        assert!(signals.is_empty());
    }

    #[test]
    fn test_check_tool_obsolescence() {
        let controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::SemiAutonomous,
            ..Default::default()
        });

        let tools = vec![("good_tool", 0.9f32), ("bad_tool", 0.5f32)];
        let signals = controller.check_tool_obsolescence(&tools);
        assert_eq!(signals.len(), 1);
    }

    #[test]
    fn test_check_tool_obsolescence_low_autonomy() {
        let controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::Suggestive,
            ..Default::default()
        });

        let tools = vec![("bad_tool", 0.5f32)];
        assert!(controller.check_tool_obsolescence(&tools).is_empty());
    }

    #[test]
    fn test_knowledge_gap() {
        let mut controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::Suggestive,
            knowledge_gap_generations: 2,
            ..Default::default()
        });

        controller.record_generation(0);
        controller.record_generation(0);
        controller.record_generation(0);

        let signal = controller.check_knowledge_gap();
        assert!(signal.is_some());
    }

    #[test]
    fn test_knowledge_gap_reset() {
        let mut controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::Suggestive,
            knowledge_gap_generations: 3,
            ..Default::default()
        });

        controller.record_generation(0);
        controller.record_generation(0);
        controller.record_generation(5); // Reset
        controller.record_generation(0);

        let signal = controller.check_knowledge_gap();
        assert!(signal.is_none());
    }

    #[test]
    fn test_memory_pressure() {
        let controller = ProactiveController::default_controller();
        let signal = controller.check_memory_pressure(0.9);
        assert!(signal.is_some());
        assert!(signal.unwrap().priority > 0.8);
    }

    #[test]
    fn test_no_memory_pressure() {
        let controller = ProactiveController::default_controller();
        assert!(controller.check_memory_pressure(0.5).is_none());
    }

    #[test]
    fn test_output_quality() {
        let controller = ProactiveController::default_controller();
        let signal = controller.check_output_quality(0.3);
        assert!(signal.is_some());
    }

    #[test]
    fn test_good_output_quality() {
        let controller = ProactiveController::default_controller();
        assert!(controller.check_output_quality(0.9).is_none());
    }

    #[test]
    fn test_create_goal_from_signal() {
        let mut controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::Suggestive,
            ..Default::default()
        });

        let signal = InitiativeSignal {
            signal_type: SignalType::ConceptDegradation,
            description: "Test".into(),
            priority: 0.8,
            min_autonomy: AutonomyLevel::Suggestive,
            related_ids: vec!["c1".into()],
        };

        let goal = controller.create_goal_from_signal(&signal);
        assert!(goal.is_some());
        assert_eq!(controller.active_goal_count(), 1);
    }

    #[test]
    fn test_create_goal_insufficient_autonomy() {
        let mut controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::Reactive,
            ..Default::default()
        });

        let signal = InitiativeSignal {
            signal_type: SignalType::ToolObsolescence,
            description: "Test".into(),
            priority: 0.8,
            min_autonomy: AutonomyLevel::SemiAutonomous,
            related_ids: vec![],
        };

        let goal = controller.create_goal_from_signal(&signal);
        assert!(goal.is_none());
    }

    #[test]
    fn test_next_goal_priority() {
        let mut controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::FullyAutonomous,
            ..Default::default()
        });

        // Create two goals
        let s1 = InitiativeSignal {
            signal_type: SignalType::KnowledgeGap,
            description: "Low priority".into(),
            priority: 0.3,
            min_autonomy: AutonomyLevel::Suggestive,
            related_ids: vec![],
        };
        let s2 = InitiativeSignal {
            signal_type: SignalType::ConceptDegradation,
            description: "High priority".into(),
            priority: 0.9,
            min_autonomy: AutonomyLevel::Suggestive,
            related_ids: vec![],
        };

        controller.create_goal_from_signal(&s1);
        controller.create_goal_from_signal(&s2);

        let next = controller.next_goal();
        assert!(next.is_some());
        assert_eq!(next.unwrap().priority, 0.9);
    }

    #[test]
    fn test_complete_goal() {
        let mut controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::Suggestive,
            ..Default::default()
        });

        let signal = InitiativeSignal {
            signal_type: SignalType::KnowledgeGap,
            description: "Test".into(),
            priority: 0.5,
            min_autonomy: AutonomyLevel::Suggestive,
            related_ids: vec![],
        };

        let goal = controller.create_goal_from_signal(&signal).unwrap();
        controller.complete_goal(&goal.id);
        assert_eq!(controller.active_goal_count(), 0);
    }

    #[test]
    fn test_log_action() {
        let mut controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::SemiAutonomous,
            ..Default::default()
        });

        controller.log_action(
            "compress_concepts",
            "Compressed 5 concepts into 1",
            "5 concepts",
            "1 concept",
            None,
        );

        assert_eq!(controller.action_count(), 1);
        assert!(controller.action_log()[0].approved);
    }

    #[test]
    fn test_rollback() {
        let mut controller = ProactiveController::default_controller();
        controller.log_action("test", "test", "before", "after", None);
        controller.rollback_last();
        assert!(controller.action_log()[0].rolled_back);
    }

    #[test]
    fn test_pending_review() {
        let mut controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::Suggestive, // Not auto-approved
            ..Default::default()
        });

        controller.log_action("test", "test", "before", "after", None);
        let pending = controller.pending_review();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_approve_action() {
        let mut controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::Suggestive,
            ..Default::default()
        });

        controller.log_action("test", "test", "before", "after", None);
        controller.approve_action(0);
        assert!(controller.action_log()[0].approved);
    }

    #[test]
    fn test_action_log_trim() {
        let config = ProactiveConfig {
            max_action_log: 3,
            autonomy_level: AutonomyLevel::FullyAutonomous,
            ..Default::default()
        };
        let mut controller = ProactiveController::new(config);

        for i in 0..10 {
            controller.log_action(format!("action_{}", i), "test", "b", "a", None);
        }

        assert!(controller.action_count() <= 3);
    }

    #[test]
    fn test_stats() {
        let mut controller = ProactiveController::new(ProactiveConfig {
            autonomy_level: AutonomyLevel::SemiAutonomous,
            ..Default::default()
        });

        controller.log_action("test", "test", "b", "a", None);
        let stats = controller.stats();
        assert_eq!(stats.autonomy_level, AutonomyLevel::SemiAutonomous);
        assert_eq!(stats.total_actions, 1);
        assert_eq!(stats.approved_actions, 1);
    }

    #[test]
    fn test_set_autonomy_level() {
        let mut controller = ProactiveController::default_controller();
        assert_eq!(controller.autonomy_level(), AutonomyLevel::Suggestive);
        controller.set_autonomy_level(AutonomyLevel::FullyAutonomous);
        assert_eq!(controller.autonomy_level(), AutonomyLevel::FullyAutonomous);
    }
}
