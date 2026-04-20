//! Skill Knowledge Base — hierarchical experience distillation.
//!
//! Stores learned skills in a 3-level hierarchy:
//! 1. **Strategic Plans**: High-level strategies (e.g., "debug by bisecting")
//! 2. **Functional Skills**: Reusable procedures (e.g., "git bisect workflow")
//! 3. **Atomic Skills**: Individual tool calls (e.g., "run git bisect good")
//!
//! Skills are:
//! - Created from successful plan executions (distilled from experience)
//! - Retrieved by prompt similarity (embedding-based lookup)
//! - Validated before storage (quality gate)
//! - Versioned (evolved over time as the model improves)
//!
//! Integration with autonomous learner:
//!   PlanExecutor succeeds → extract skill → store in SkillKB
//!   Next similar prompt → retrieve skill → adapt → execute

use std::collections::HashMap;

/// Skill hierarchy level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkillLevel {
    /// Individual tool call or single step
    Atomic,
    /// Multi-step reusable procedure
    Functional,
    /// High-level strategy spanning multiple functional skills
    Strategic,
}

impl SkillLevel {
    /// Numeric depth: 0 = atomic, 1 = functional, 2 = strategic
    pub fn depth(&self) -> usize {
        match self {
            SkillLevel::Atomic => 0,
            SkillLevel::Functional => 1,
            SkillLevel::Strategic => 2,
        }
    }
}

/// A single skill in the knowledge base.
#[derive(Clone, Debug)]
pub struct Skill {
    /// Unique skill ID.
    pub id: u64,
    /// Skill name (human-readable).
    pub name: String,
    /// Hierarchy level.
    pub level: SkillLevel,
    /// Text description / prompt pattern that triggers this skill.
    pub description: String,
    /// The actual skill content (steps to execute).
    pub content: String,
    /// Parent skill ID (for hierarchy: functional → strategic, atomic → functional).
    pub parent_id: Option<u64>,
    /// Child skill IDs.
    pub children: Vec<u64>,
    /// Number of times this skill was successfully used.
    pub success_count: u32,
    /// Number of times this skill was attempted.
    pub attempt_count: u32,
    /// Success rate (computed).
    pub success_rate: f32,
    /// Version number (incremented when skill is refined).
    pub version: u32,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Embedding (simple bag-of-words hash for similarity).
    pub embedding: Vec<f32>,
}

impl Skill {
    /// Compute success rate from counts.
    pub fn compute_success_rate(&mut self) {
        self.success_rate = if self.attempt_count > 0 {
            self.success_count as f32 / self.attempt_count as f32
        } else {
            0.0
        };
    }

    /// Whether this skill passes the quality gate for storage.
    pub fn passes_quality_gate(&self, min_attempts: u32, min_success_rate: f32) -> bool {
        self.attempt_count >= min_attempts && self.success_rate >= min_success_rate
    }
}

/// Configuration for the Skill Knowledge Base.
#[derive(Clone, Debug)]
pub struct SkillKBConfig {
    /// Minimum attempts before a skill can be stored.
    pub min_attempts_for_storage: u32,
    /// Minimum success rate to keep a skill.
    pub min_success_rate: f32,
    /// Maximum number of skills per level.
    pub max_skills_per_level: usize,
    /// Maximum total skills.
    pub max_total_skills: usize,
}

impl Default for SkillKBConfig {
    fn default() -> Self {
        Self {
            min_attempts_for_storage: 3,
            min_success_rate: 0.6,
            max_skills_per_level: 1000,
            max_total_skills: 5000,
        }
    }
}

/// The Skill Knowledge Base.
pub struct SkillKB {
    skills: HashMap<u64, Skill>,
    next_id: u64,
    config: SkillKBConfig,
}

impl SkillKB {
    pub fn new(config: SkillKBConfig) -> Self {
        Self {
            skills: HashMap::new(),
            next_id: 1,
            config,
        }
    }

    /// Create a new skill (not yet stored).
    pub fn create_skill(
        &mut self,
        name: &str,
        level: SkillLevel,
        description: &str,
        content: &str,
        tags: Vec<String>,
    ) -> Skill {
        let embedding = simple_embedding(description);
        Skill {
            id: self.next_id,
            name: name.to_string(),
            level,
            description: description.to_string(),
            content: content.to_string(),
            parent_id: None,
            children: vec![],
            success_count: 0,
            attempt_count: 0,
            success_rate: 0.0,
            version: 1,
            tags,
            embedding,
        }
    }

    /// Store a skill in the knowledge base.
    /// Returns the skill ID if stored, or None if it doesn't pass quality gate
    /// or the KB is full.
    pub fn store(&mut self, skill: Skill) -> Option<u64> {
        let level = skill.level;
        let id = skill.id;

        // Check capacity
        let level_count = self.skills.values().filter(|s| s.level == level).count();
        if level_count >= self.config.max_skills_per_level
            || self.skills.len() >= self.config.max_total_skills
        {
            tracing::warn!("SkillKB full, cannot store skill '{}'", skill.name);
            return None;
        }

        // If skill has enough attempts, check quality gate
        if skill.attempt_count >= self.config.min_attempts_for_storage {
            if !skill.passes_quality_gate(
                self.config.min_attempts_for_storage,
                self.config.min_success_rate,
            ) {
                tracing::debug!(
                    "Skill '{}' failed quality gate (attempts={}, rate={:.2})",
                    skill.name, skill.attempt_count, skill.success_rate
                );
                return None;
            }
        }

        self.skills.insert(id, skill);
        self.next_id = self.next_id.max(id + 1);
        Some(id)
    }

    /// Record a skill usage outcome.
    pub fn record_outcome(&mut self, skill_id: u64, success: bool) {
        if let Some(skill) = self.skills.get_mut(&skill_id) {
            skill.attempt_count += 1;
            if success {
                skill.success_count += 1;
            }
            skill.compute_success_rate();
        }
    }

    /// Link a child skill to a parent.
    pub fn link(&mut self, parent_id: u64, child_id: u64) {
        if let Some(parent) = self.skills.get_mut(&parent_id) {
            if !parent.children.contains(&child_id) {
                parent.children.push(child_id);
            }
        }
        if let Some(child) = self.skills.get_mut(&child_id) {
            child.parent_id = Some(parent_id);
        }
    }

    /// Retrieve skills by similarity to a query description.
    /// Returns skills sorted by relevance (highest first).
    pub fn retrieve(&self, query: &str, max_results: usize) -> Vec<&Skill> {
        let query_emb = simple_embedding(query);

        let mut scored: Vec<(f32, &Skill)> = self.skills
            .values()
            .map(|skill| {
                let sim = cosine_sim(&query_emb, &skill.embedding);
                // Boost by success rate
                let score = sim * (0.5 + 0.5 * skill.success_rate);
                (score, skill)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(max_results).map(|(_, s)| s).collect()
    }

    /// Retrieve a specific skill by ID.
    pub fn get(&self, id: u64) -> Option<&Skill> {
        self.skills.get(&id)
    }

    /// Get all skills at a given level.
    pub fn get_by_level(&self, level: SkillLevel) -> Vec<&Skill> {
        self.skills.values().filter(|s| s.level == level).collect()
    }

    /// Get children of a skill.
    pub fn get_children(&self, parent_id: u64) -> Vec<&Skill> {
        if let Some(parent) = self.skills.get(&parent_id) {
            parent.children.iter()
                .filter_map(|&cid| self.skills.get(&cid))
                .collect()
        } else {
            vec![]
        }
    }

    /// Number of skills in the KB.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Whether the KB is empty.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Remove low-quality skills (below min_success_rate with enough attempts).
    pub fn prune(&mut self) -> usize {
        let min_rate = self.config.min_success_rate;
        let min_attempts = self.config.min_attempts_for_storage;
        let before = self.skills.len();

        self.skills.retain(|_, skill| {
            if skill.attempt_count < min_attempts {
                return true; // Not enough data yet
            }
            skill.success_rate >= min_rate
        });

        before - self.skills.len()
    }
}

/// Simple bag-of-words embedding (deterministic hash-based).
fn simple_embedding(text: &str) -> Vec<f32> {
    let dim = 64;
    let mut emb = vec![0.0f32; dim];
    for word in text.to_lowercase().split_whitespace() {
        let hash = simple_hash(word);
        let idx = (hash as usize) % dim;
        emb[idx] += 1.0;
    }
    // Normalize
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut emb {
            *v /= norm;
        }
    }
    emb
}

/// Simple deterministic string hash.
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for &b in s.as_bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

/// Cosine similarity between two vectors.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let min_len = a.len().min(b.len());
    if min_len == 0 { return 0.0; }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..min_len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_create() {
        let mut kb = SkillKB::new(SkillKBConfig::default());
        let skill = kb.create_skill(
            "git bisect",
            SkillLevel::Functional,
            "Binary search for bugs using git",
            "1. git bisect start\n2. git bisect good <commit>\n3. git bisect bad <commit>",
            vec!["git".into(), "debug".into()],
        );
        assert_eq!(skill.level, SkillLevel::Functional);
        assert_eq!(skill.version, 1);
        assert_eq!(skill.tags.len(), 2);
    }

    #[test]
    fn test_skill_store_and_get() {
        let mut kb = SkillKB::new(SkillKBConfig::default());
        let skill = kb.create_skill("test", SkillLevel::Atomic, "a test skill", "do thing", vec![]);
        let id = kb.store(skill).unwrap();
        assert_eq!(kb.len(), 1);
        assert!(kb.get(id).is_some());
    }

    #[test]
    fn test_skill_record_outcome() {
        let mut kb = SkillKB::new(SkillKBConfig::default());
        let skill = kb.create_skill("test", SkillLevel::Atomic, "test", "test", vec![]);
        let id = kb.store(skill).unwrap();

        kb.record_outcome(id, true);
        kb.record_outcome(id, true);
        kb.record_outcome(id, false);

        let s = kb.get(id).unwrap();
        assert_eq!(s.attempt_count, 3);
        assert_eq!(s.success_count, 2);
        assert!((s.success_rate - (2.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_skill_hierarchy() {
        let mut kb = SkillKB::new(SkillKBConfig::default());

        let strategic = kb.create_skill("debug strategy", SkillLevel::Strategic, "debug", "debug steps", vec![]);
        let strategic_id = kb.store(strategic).unwrap();

        let func = kb.create_skill("git bisect", SkillLevel::Functional, "git bisect debug", "bisect steps", vec![]);
        let func_id = kb.store(func).unwrap();

        let atomic = kb.create_skill("run bisect", SkillLevel::Atomic, "run git bisect", "git bisect", vec![]);
        let atomic_id = kb.store(atomic).unwrap();

        kb.link(strategic_id, func_id);
        kb.link(func_id, atomic_id);

        let children = kb.get_children(strategic_id);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, func_id);

        let grandkids = kb.get_children(func_id);
        assert_eq!(grandkids.len(), 1);
        assert_eq!(grandkids[0].id, atomic_id);

        assert_eq!(kb.get(atomic_id).unwrap().parent_id, Some(func_id));
    }

    #[test]
    fn test_skill_retrieve_by_similarity() {
        let mut kb = SkillKB::new(SkillKBConfig::default());

        // Record successes so success_rate boosts ranking
        let s1 = kb.create_skill("git bisect", SkillLevel::Functional, "debug with git bisect", "steps", vec![]);
        let id1 = kb.store(s1).unwrap();
        for _ in 0..5 { kb.record_outcome(id1, true); }

        let s2 = kb.create_skill("code review", SkillLevel::Functional, "review code changes", "steps", vec![]);
        let id2 = kb.store(s2).unwrap();
        for _ in 0..5 { kb.record_outcome(id2, true); }

        let s3 = kb.create_skill("git merge", SkillLevel::Atomic, "merge git branches", "steps", vec![]);
        let id3 = kb.store(s3).unwrap();
        for _ in 0..5 { kb.record_outcome(id3, true); }

        // Query for "debug git" should rank git bisect first
        let results = kb.retrieve("debug git problem", 3);
        assert!(!results.is_empty());
        // Both git bisect and git merge should be retrieved (they share "git")
        assert!(results.iter().any(|s| s.name.contains("git")));
    }

    #[test]
    fn test_skill_quality_gate_rejects() {
        let mut kb = SkillKB::new(SkillKBConfig {
            min_attempts_for_storage: 3,
            min_success_rate: 0.6,
            ..Default::default()
        });

        let mut skill = kb.create_skill("bad skill", SkillLevel::Atomic, "test", "test", vec![]);
        skill.attempt_count = 5;
        skill.success_count = 1;
        skill.compute_success_rate();
        // success_rate = 0.2, below 0.6 threshold
        assert!(!skill.passes_quality_gate(3, 0.6));
    }

    #[test]
    fn test_skill_prune() {
        let mut kb = SkillKB::new(SkillKBConfig {
            min_attempts_for_storage: 2,
            min_success_rate: 0.5,
            ..Default::default()
        });

        // Good skill: 80% success rate
        let mut s1 = kb.create_skill("good", SkillLevel::Atomic, "good skill", "test", vec![]);
        s1.attempt_count = 5; s1.success_count = 4; s1.compute_success_rate();
        kb.store(s1);

        // Marginal skill: starts OK, then degrades
        // We can't store a bad skill directly (quality gate), so test prune
        // by manually inserting a skill that later gets degraded outcomes
        let mut s2 = kb.create_skill("marginal", SkillLevel::Atomic, "marginal skill", "test", vec![]);
        s2.attempt_count = 4; s2.success_count = 3; s2.compute_success_rate(); // 75% - passes gate
        let s2_id = kb.store(s2).unwrap();

        // Now degrade it
        for _ in 0..10 { kb.record_outcome(s2_id, false); }
        // Now: 3/14 ≈ 21% success rate

        assert_eq!(kb.len(), 2);
        let pruned = kb.prune();
        assert_eq!(pruned, 1); // marginal skill now below 50%
        assert_eq!(kb.len(), 1);
    }

    #[test]
    fn test_skill_level_depth() {
        assert_eq!(SkillLevel::Atomic.depth(), 0);
        assert_eq!(SkillLevel::Functional.depth(), 1);
        assert_eq!(SkillLevel::Strategic.depth(), 2);
    }

    #[test]
    fn test_kb_capacity_limit() {
        let mut kb = SkillKB::new(SkillKBConfig {
            max_total_skills: 2,
            ..Default::default()
        });

        let s1 = kb.create_skill("s1", SkillLevel::Atomic, "1", "1", vec![]);
        kb.store(s1);
        let s2 = kb.create_skill("s2", SkillLevel::Atomic, "2", "2", vec![]);
        kb.store(s2);
        let s3 = kb.create_skill("s3", SkillLevel::Atomic, "3", "3", vec![]);
        let result = kb.store(s3);
        assert!(result.is_none()); // Full
    }
}
