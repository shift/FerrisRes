//! Domain Detector — Prompt-Based Domain Detection and Specialized Retrieval
//!
//! Detects the domain of a prompt using a hybrid approach:
//!   1. FAST PATH: Keyword scan (deterministic, <1ms)
//!   2. LLM PATH: Model-emitted [domain: X] tags during planning
//!   3. BEHAVIORAL PATH: Infer from tool usage after execution
//!
//! Provides per-domain retrieval bias for tools, concepts, and episodes.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A known domain with its characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainProfile {
    /// Domain name (e.g., "programming", "mathematics").
    pub name: String,
    /// Keywords strongly associated with this domain.
    pub keywords: Vec<String>,
    /// Tools preferred in this domain (tool_name → success_rate).
    pub preferred_tools: HashMap<String, f32>,
    /// Concept categories most relevant.
    pub relevant_concept_categories: Vec<String>,
    /// Episode types most relevant.
    pub relevant_episode_types: Vec<String>,
    /// Cross-domain transfer sources.
    pub transfer_sources: Vec<String>,
    /// Number of interactions in this domain.
    pub interaction_count: u32,
    /// Average quality in this domain.
    pub avg_quality: f32,
}

/// Domain detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainDetection {
    /// Primary detected domain.
    pub domain: String,
    /// Confidence (0.0–1.0).
    pub confidence: f32,
    /// Secondary domain (if ambiguous).
    pub secondary: Option<(String, f32)>,
    /// Detection method used.
    pub method: DetectionMethod,
}

/// How the domain was detected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DetectionMethod {
    /// Matched by keyword scan.
    Keyword { matched_terms: Vec<String> },
    /// Tagged by LLM during planning.
    LlmTagged,
    /// Inferred from tool usage.
    Behavioral { tools_used: Vec<String> },
    /// Default (no domain detected).
    Unknown,
}

/// Retrieval bias for a detected domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRetrievalBias {
    /// Boost factor for preferred tools (tool_name → boost multiplier).
    pub tool_boost: HashMap<String, f32>,
    /// Preferred concept categories.
    pub concept_categories: Vec<String>,
    /// Preferred episode types.
    pub episode_types: Vec<String>,
}

/// A cross-domain transfer candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferCandidate {
    pub source_domain: String,
    pub target_domain: String,
    pub concepts_transferred: u32,
    pub avg_transfer_quality: f32,
    pub validated: bool,
}

/// Configuration for DomainDetector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainDetectorConfig {
    /// Minimum keyword confidence threshold (default: 0.3).
    pub keyword_confidence_threshold: f32,
    /// Boost factor for preferred tools (default: 1.2).
    pub tool_boost_factor: f32,
    /// Maximum domain profiles (default: 50).
    pub max_profiles: usize,
}

impl Default for DomainDetectorConfig {
    fn default() -> Self {
        Self {
            keyword_confidence_threshold: 0.3,
            tool_boost_factor: 1.2,
            max_profiles: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// DomainDetector
// ---------------------------------------------------------------------------

/// Detects domains from prompts and provides retrieval bias.
pub struct DomainDetector {
    config: DomainDetectorConfig,
    /// Domain profiles indexed by name.
    profiles: HashMap<String, DomainProfile>,
    /// Transfer paths: (source, target) → TransferCandidate.
    transfers: HashMap<(String, String), TransferCandidate>,
}

impl DomainDetector {
    pub fn new(config: DomainDetectorConfig) -> Self {
        let mut det = Self {
            config,
            profiles: HashMap::new(),
            transfers: HashMap::new(),
        };
        det.init_default_domains();
        det
    }

    pub fn with_defaults() -> Self {
        Self::new(DomainDetectorConfig::default())
    }

    /// Initialize with built-in domain profiles.
    fn init_default_domains(&mut self) {
        let defaults = vec![
            (
                "programming",
                vec![
                    "function", "class", "compile", "debug", "api", "code", "rust", "python",
                    "javascript", "variable", "loop", "array", "struct", "impl", "module",
                    "import", "export", "async", "thread", "memory", "pointer", "refactor",
                ],
                vec!["code_run", "wasm_parse", "file_read", "file_write"],
                vec!["code", "algorithm", "pattern"],
            ),
            (
                "mathematics",
                vec![
                    "equation", "integral", "derivative", "proof", "theorem", "matrix",
                    "vector", "polynomial", "calculus", "algebra", "geometry", "probability",
                    "statistics", "optimization", "linear", "eigenvalue", "fourier",
                ],
                vec!["math_eval", "compute_stats"],
                vec!["formula", "theorem", "proof"],
            ),
            (
                "science",
                vec![
                    "experiment", "hypothesis", "molecule", "reaction", "force", "energy",
                    "atom", "cell", "dna", "protein", "gravity", "photon", "electron",
                    "entropy", "quantum", "relativity", "evolution",
                ],
                vec!["web_fetch", "compute_stats"],
                vec!["fact", "theory", "experiment"],
            ),
            (
                "medicine",
                vec![
                    "patient", "diagnosis", "symptom", "treatment", "drug", "clinical",
                    "disease", "therapy", "surgery", "prescription", "dosage", "prognosis",
                    "pathology", "radiology", "pharmacology",
                ],
                vec!["web_fetch", "search"],
                vec!["diagnosis", "treatment", "drug"],
            ),
            (
                "finance",
                vec![
                    "stock", "portfolio", "risk", "option", "bond", "trading", "investment",
                    "market", "dividend", "volatility", "hedging", "derivative", "futures",
                    "currency", "exchange", "interest", "capital", "asset",
                ],
                vec!["web_fetch", "math_eval", "compute_stats"],
                vec!["market", "strategy", "risk"],
            ),
            (
                "law",
                vec![
                    "contract", "statute", "liability", "compliance", "regulation",
                    "attorney", "court", "lawsuit", "jurisdiction", "intellectual property",
                    "patent", "copyright", "trademark", "legal", "clause", "verdict",
                ],
                vec!["web_fetch", "search", "file_read"],
                vec!["statute", "case", "contract"],
            ),
        ];

        for (name, keywords, tools, categories) in defaults {
            let mut preferred_tools = HashMap::new();
            for t in tools {
                preferred_tools.insert(t.to_string(), 0.5);
            }

            self.profiles.insert(
                name.to_string(),
                DomainProfile {
                    name: name.to_string(),
                    keywords: keywords.into_iter().map(String::from).collect(),
                    preferred_tools,
                    relevant_concept_categories: categories.into_iter().map(String::from).collect(),
                    relevant_episode_types: vec![name.to_string()],
                    transfer_sources: vec![],
                    interaction_count: 0,
                    avg_quality: 0.5,
                },
            );
        }
    }

    // -----------------------------------------------------------------------
    // Detection
    // -----------------------------------------------------------------------

    /// Detect domain from a prompt using keyword scan.
    pub fn detect(&self, prompt: &str) -> DomainDetection {
        let lower = prompt.to_lowercase();
        let mut best_domain = "unknown".to_string();
        let mut best_score = 0.0f32;
        let mut best_terms = Vec::new();
        let mut second_domain = "unknown".to_string();
        let mut second_score = 0.0f32;

        for (name, profile) in &self.profiles {
            let mut matches = Vec::new();
            for kw in &profile.keywords {
                if lower.contains(&kw.to_lowercase()) {
                    matches.push(kw.clone());
                }
            }

            if matches.is_empty() {
                continue;
            }

            // Score: fraction of keywords matched, weighted by match count
            let score = (matches.len() as f32 / profile.keywords.len().max(1) as f32)
                * (1.0 + (matches.len() as f32).ln());

            if score > best_score {
                second_score = best_score;
                second_domain = best_domain.clone();
                best_score = score;
                best_domain = name.clone();
                best_terms = matches;
            } else if score > second_score {
                second_score = score;
                second_domain = name.clone();
            }
        }

        // Normalize confidence
        let confidence = if best_score > 0.0 {
            (best_score / (best_score + second_score + 0.01)).min(1.0)
        } else {
            0.0
        };

        if confidence < self.config.keyword_confidence_threshold || best_domain == "unknown" {
            return DomainDetection {
                domain: "unknown".to_string(),
                confidence: 0.0,
                secondary: None,
                method: DetectionMethod::Unknown,
            };
        }

        let secondary = if second_score > 0.0 && second_domain != "unknown" {
            Some((second_domain, second_score / (best_score + second_score + 0.01)))
        } else {
            None
        };

        DomainDetection {
            domain: best_domain,
            confidence,
            secondary,
            method: DetectionMethod::Keyword {
                matched_terms: best_terms,
            },
        }
    }

    /// Detect domain from LLM-emitted [domain: X] tag.
    pub fn detect_from_tag(&self, text: &str) -> DomainDetection {
        // Look for [domain: X] pattern
        let re = regex::Regex::new(r"\[domain:\s*(\w+)\]").unwrap();
        if let Some(caps) = re.captures(text) {
            let domain = caps[1].to_lowercase();
            // Verify it's a known domain
            if self.profiles.contains_key(&domain) {
                return DomainDetection {
                    domain: domain.clone(),
                    confidence: 0.9, // High confidence from LLM
                    secondary: None,
                    method: DetectionMethod::LlmTagged,
                };
            }
            // Unknown domain from LLM
            return DomainDetection {
                domain,
                confidence: 0.7,
                secondary: None,
                method: DetectionMethod::LlmTagged,
            };
        }

        DomainDetection {
            domain: "unknown".to_string(),
            confidence: 0.0,
            secondary: None,
            method: DetectionMethod::Unknown,
        }
    }

    /// Detect domain from tool usage (behavioral).
    pub fn detect_from_tools(&self, tools: &[String]) -> DomainDetection {
        let mut best_domain = "unknown".to_string();
        let mut best_score = 0.0f32;
        let mut matched_tools = Vec::new();

        for (name, profile) in &self.profiles {
            let matches: Vec<String> = tools
                .iter()
                .filter(|t| profile.preferred_tools.contains_key(*t))
                .cloned()
                .collect();

            if matches.len() > best_score as usize {
                best_score = matches.len() as f32;
                best_domain = name.clone();
                matched_tools = matches;
            }
        }

        if best_score == 0.0 {
            return DomainDetection {
                domain: "unknown".to_string(),
                confidence: 0.0,
                secondary: None,
                method: DetectionMethod::Unknown,
            };
        }

        let confidence = (best_score / (tools.len() as f32).max(1.0)).min(1.0);

        DomainDetection {
            domain: best_domain,
            confidence,
            secondary: None,
            method: DetectionMethod::Behavioral {
                tools_used: matched_tools,
            },
        }
    }

    // -----------------------------------------------------------------------
    // Profile updates
    // -----------------------------------------------------------------------

    /// Update domain profile with an interaction outcome.
    pub fn update_profile(&mut self, domain: &str, tool: &str, quality: f32) {
        let profile = match self.profiles.get_mut(domain) {
            Some(p) => p,
            None => {
                // Create new domain profile
                if self.profiles.len() < self.config.max_profiles {
                    self.profiles.insert(
                        domain.to_string(),
                        DomainProfile {
                            name: domain.to_string(),
                            keywords: vec![],
                            preferred_tools: HashMap::new(),
                            relevant_concept_categories: vec![],
                            relevant_episode_types: vec![domain.to_string()],
                            transfer_sources: vec![],
                            interaction_count: 0,
                            avg_quality: 0.5,
                        },
                    );
                    self.profiles.get_mut(domain).unwrap()
                } else {
                    return;
                }
            }
        };

        profile.interaction_count += 1;
        let alpha = 0.15;
        profile.avg_quality = alpha * quality + (1.0 - alpha) * profile.avg_quality;

        // Update tool preference
        let tool_quality = profile.preferred_tools.entry(tool.to_string()).or_insert(0.5);
        *tool_quality = alpha * quality + (1.0 - alpha) * *tool_quality;
    }

    // -----------------------------------------------------------------------
    // Retrieval bias
    // -----------------------------------------------------------------------

    /// Get retrieval bias for a detected domain.
    pub fn retrieval_bias(&self, domain: &str) -> DomainRetrievalBias {
        let profile = match self.profiles.get(domain) {
            Some(p) => p,
            None => {
                return DomainRetrievalBias {
                    tool_boost: HashMap::new(),
                    concept_categories: vec![],
                    episode_types: vec![],
                }
            }
        };

        let tool_boost: HashMap<String, f32> = profile
            .preferred_tools
            .iter()
            .filter_map(|(tool, &quality)| {
                if quality > 0.3 {
                    Some((tool.clone(), self.config.tool_boost_factor))
                } else {
                    None
                }
            })
            .collect();

        DomainRetrievalBias {
            tool_boost,
            concept_categories: profile.relevant_concept_categories.clone(),
            episode_types: profile.relevant_episode_types.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Transfer
    // -----------------------------------------------------------------------

    /// Record a cross-domain concept transfer.
    pub fn record_transfer(&mut self, source: &str, target: &str, quality: f32) {
        let key = (source.to_string(), target.to_string());
        let entry = self.transfers.entry(key).or_insert_with(|| TransferCandidate {
            source_domain: source.to_string(),
            target_domain: target.to_string(),
            concepts_transferred: 0,
            avg_transfer_quality: 0.5,
            validated: false,
        });

        entry.concepts_transferred += 1;
        let alpha = 0.2;
        entry.avg_transfer_quality = alpha * quality + (1.0 - alpha) * entry.avg_transfer_quality;

        // Validate after enough transfers
        if entry.concepts_transferred >= 5 && entry.avg_transfer_quality > 0.6 {
            entry.validated = true;
        }
    }

    /// Get transfer candidates for a domain.
    pub fn transfer_candidates(&self, domain: &str) -> Vec<&TransferCandidate> {
        self.transfers
            .values()
            .filter(|t| t.target_domain == domain && t.validated)
            .collect()
    }

    /// Get all validated transfer paths.
    pub fn validated_transfers(&self) -> Vec<&TransferCandidate> {
        self.transfers.values().filter(|t| t.validated).collect()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get a domain profile.
    pub fn get_profile(&self, domain: &str) -> Option<&DomainProfile> {
        self.profiles.get(domain)
    }

    /// Get all known domain names.
    pub fn known_domains(&self) -> Vec<&str> {
        self.profiles.keys().map(|s| s.as_str()).collect()
    }

    /// Number of domain profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether no profiles exist.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_detection_programming() {
        let det = DomainDetector::with_defaults();
        let result = det.detect("How do I implement a red-black tree in Rust?");
        assert_eq!(result.domain, "programming");
        assert!(result.confidence > 0.5);
        assert!(matches!(result.method, DetectionMethod::Keyword { .. }));
    }

    #[test]
    fn test_keyword_detection_math() {
        let det = DomainDetector::with_defaults();
        let result = det.detect("Prove that the eigenvalue of a symmetric matrix is real");
        assert_eq!(result.domain, "mathematics");
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_keyword_confidence() {
        let det = DomainDetector::with_defaults();
        // Many programming terms → high confidence
        let result = det.detect("debug the async function in the rust module and fix the pointer");
        assert_eq!(result.domain, "programming");
        assert!(result.confidence > 0.7);
    }

    #[test]
    fn test_keyword_ambiguity() {
        let det = DomainDetector::with_defaults();
        // Mixed: "statistics" (math) + "experiment" (science)
        let result = det.detect("Run a statistics experiment to test the hypothesis");
        assert!(result.confidence < 0.9 || result.secondary.is_some());
    }

    #[test]
    fn test_llm_tag_parsing() {
        let det = DomainDetector::with_defaults();
        let result = det.detect_from_tag("[domain: programming]\n[plan]\nStep 1: code_run(x)\n[/plan]");
        assert_eq!(result.domain, "programming");
        assert_eq!(result.method, DetectionMethod::LlmTagged);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_llm_tag_unknown_domain() {
        let det = DomainDetector::with_defaults();
        let result = det.detect_from_tag("[domain: culinary]");
        assert_eq!(result.domain, "culinary");
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_llm_tag_no_tag() {
        let det = DomainDetector::with_defaults();
        let result = det.detect_from_tag("no domain tag here");
        assert_eq!(result.domain, "unknown");
        assert_eq!(result.method, DetectionMethod::Unknown);
    }

    #[test]
    fn test_behavioral_detection() {
        let det = DomainDetector::with_defaults();
        let result = det.detect_from_tools(&[
            "code_run".into(),
            "wasm_parse".into(),
        ]);
        assert_eq!(result.domain, "programming");
        assert!(matches!(result.method, DetectionMethod::Behavioral { .. }));
    }

    #[test]
    fn test_behavioral_detection_unknown() {
        let det = DomainDetector::with_defaults();
        let result = det.detect_from_tools(&["mystery_tool".into()]);
        assert_eq!(result.domain, "unknown");
    }

    #[test]
    fn test_domain_profile_creation() {
        let mut det = DomainDetector::with_defaults();
        assert!(det.get_profile("culinary").is_none());

        det.update_profile("culinary", "recipe_search", 0.8);

        let profile = det.get_profile("culinary").unwrap();
        assert_eq!(profile.interaction_count, 1);
    }

    #[test]
    fn test_preferred_tools_update() {
        let mut det = DomainDetector::with_defaults();
        det.update_profile("programming", "code_run", 0.9);
        det.update_profile("programming", "code_run", 0.95);
        det.update_profile("programming", "code_run", 0.85);

        let profile = det.get_profile("programming").unwrap();
        let quality = profile.preferred_tools.get("code_run").unwrap();
        assert!(*quality > 0.5);
    }

    #[test]
    fn test_retrieval_bias_tool_boost() {
        let mut det = DomainDetector::with_defaults();
        det.update_profile("programming", "code_run", 0.9);

        let bias = det.retrieval_bias("programming");
        assert!(bias.tool_boost.contains_key("code_run"));
        assert_eq!(bias.tool_boost["code_run"], 1.2);
    }

    #[test]
    fn test_retrieval_bias_unknown_domain() {
        let det = DomainDetector::with_defaults();
        let bias = det.retrieval_bias("nonexistent");
        assert!(bias.tool_boost.is_empty());
    }

    #[test]
    fn test_retrieval_bias_concept_filter() {
        let det = DomainDetector::with_defaults();
        let bias = det.retrieval_bias("programming");
        assert!(bias.concept_categories.contains(&"code".to_string()));
    }

    #[test]
    fn test_episode_domain_filter() {
        let det = DomainDetector::with_defaults();
        let bias = det.retrieval_bias("medicine");
        assert!(bias.episode_types.contains(&"medicine".to_string()));
    }

    #[test]
    fn test_transfer_path_creation() {
        let mut det = DomainDetector::with_defaults();
        for _ in 0..5 {
            det.record_transfer("programming", "mathematics", 0.8);
        }

        let candidates = det.transfer_candidates("mathematics");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].source_domain, "programming");
        assert!(candidates[0].validated);
    }

    #[test]
    fn test_transfer_quality_tracking() {
        let mut det = DomainDetector::with_defaults();
        det.record_transfer("a", "b", 1.0);
        det.record_transfer("a", "b", 0.0);
        det.record_transfer("a", "b", 1.0);
        det.record_transfer("a", "b", 0.0);
        det.record_transfer("a", "b", 0.3); // avg ~0.46, below 0.6

        let candidates = det.transfer_candidates("b");
        assert!(candidates.is_empty()); // Not validated (quality < 0.6)
    }

    #[test]
    fn test_transfer_validation() {
        let mut det = DomainDetector::with_defaults();
        for _ in 0..6 {
            det.record_transfer("science", "medicine", 0.85);
        }

        let validated = det.validated_transfers();
        assert_eq!(validated.len(), 1);
        assert!(validated[0].validated);
        assert!(validated[0].avg_transfer_quality > 0.7);
    }

    #[test]
    fn test_unknown_domain_handling() {
        let det = DomainDetector::with_defaults();
        let result = det.detect("The quick brown fox jumps over the lazy dog");
        assert_eq!(result.domain, "unknown");
        assert_eq!(result.method, DetectionMethod::Unknown);
    }

    #[test]
    fn test_domain_persistence() {
        let mut det = DomainDetector::with_defaults();
        det.update_profile("programming", "code_run", 0.9);

        // Verify profile is updated
        let profile = det.get_profile("programming").unwrap();
        assert!(profile.avg_quality > 0.5);
        assert!(profile.preferred_tools.contains_key("code_run"));
    }

    #[test]
    fn test_multi_domain_prompt() {
        let det = DomainDetector::with_defaults();
        // "statistics" matches both math and science
        let result = det.detect("Compute statistics for the experiment");
        assert!(result.secondary.is_some() || result.confidence < 0.9);
    }

    #[test]
    fn test_domain_interaction_count() {
        let mut det = DomainDetector::with_defaults();
        det.update_profile("finance", "web_fetch", 0.8);
        det.update_profile("finance", "web_fetch", 0.9);
        det.update_profile("finance", "math_eval", 0.7);

        let profile = det.get_profile("finance").unwrap();
        assert_eq!(profile.interaction_count, 3);
    }

    #[test]
    fn test_known_domains() {
        let det = DomainDetector::with_defaults();
        let domains = det.known_domains();
        assert!(domains.contains(&"programming"));
        assert!(domains.contains(&"mathematics"));
        assert!(domains.contains(&"science"));
    }
}
