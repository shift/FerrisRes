//! Agentic Context Protocol (ACP) integration.
//!
//! ACP defines a standard protocol for AI agents to exchange context,
//! delegate tasks, and coordinate across multiple sessions. This module
//! implements the core ACP types and message handling for FerrisRes.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ACP Message types
// ---------------------------------------------------------------------------

/// An ACP message exchanged between agents.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AcpMessage {
    /// Unique message ID.
    pub id: String,
    /// Sender agent ID.
    pub from: String,
    /// Recipient agent ID (or "broadcast").
    pub to: String,
    /// Message type.
    pub msg_type: AcpMessageType,
    /// Payload (JSON-serializable).
    pub payload: serde_json::Value,
    /// Timestamp (Unix epoch seconds).
    pub timestamp: u64,
    /// Optional correlation ID for request/response.
    pub correlation_id: Option<String>,
}

/// Types of ACP messages.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AcpMessageType {
    /// Request context from another agent.
    ContextRequest,
    /// Provide context to another agent.
    ContextResponse,
    /// Delegate a task to another agent.
    TaskDelegate,
    /// Report task completion.
    TaskResult,
    /// Broadcast status update.
    StatusUpdate,
    /// Register agent capabilities.
    CapabilityAdvertisement,
    /// Error notification.
    Error,
}

// ---------------------------------------------------------------------------
// Agent Profile
// ---------------------------------------------------------------------------

/// An agent's profile and capabilities.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentProfile {
    /// Unique agent ID.
    pub agent_id: String,
    /// Human-readable name.
    pub name: String,
    /// Capabilities this agent provides.
    pub capabilities: Vec<AgentCapability>,
    /// Maximum context window the agent can handle.
    pub max_context_tokens: usize,
    /// Supported modalities.
    pub modalities: Vec<String>,
}

/// A specific capability an agent provides.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentCapability {
    /// Capability name (e.g., "code_generation", "web_search").
    pub name: String,
    /// Description of the capability.
    pub description: String,
    /// Input schema (JSON Schema).
    pub input_schema: serde_json::Value,
    /// Output schema (JSON Schema).
    pub output_schema: serde_json::Value,
}

// ---------------------------------------------------------------------------
// ACP Router
// ---------------------------------------------------------------------------

/// Routes ACP messages between agents.
pub struct AcpRouter {
    /// Registered agents.
    agents: HashMap<String, AgentProfile>,
    /// Message history.
    history: Vec<AcpMessage>,
    /// Maximum history size.
    max_history: usize,
}

impl AcpRouter {
    /// Create a new ACP router.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Register an agent with the router.
    pub fn register(&mut self, profile: AgentProfile) {
        self.agents.insert(profile.agent_id.clone(), profile);
    }

    /// Unregister an agent.
    pub fn unregister(&mut self, agent_id: &str) {
        self.agents.remove(agent_id);
    }

    /// Route a message to the appropriate agent(s).
    pub fn route(&mut self, message: AcpMessage) -> Result<Vec<AcpMessage>, String> {
        // Validate sender
        if !self.agents.contains_key(&message.from) {
            return Err(format!("Unknown sender: {}", message.from));
        }

        // Store in history
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(message.clone());

        let mut responses = Vec::new();

        match message.to.as_str() {
            "broadcast" => {
                // Send to all agents except sender
                for (id, _profile) in &self.agents {
                    if id != &message.from {
                        responses.push(AcpMessage {
                            id: format!("{}-{}", message.id, id),
                            from: message.from.clone(),
                            to: id.clone(),
                            msg_type: message.msg_type.clone(),
                            payload: message.payload.clone(),
                            timestamp: message.timestamp,
                            correlation_id: Some(message.id.clone()),
                        });
                    }
                }
            }
            recipient => {
                if self.agents.contains_key(recipient) {
                    responses.push(message);
                } else {
                    return Err(format!("Unknown recipient: {}", recipient));
                }
            }
        }

        Ok(responses)
    }

    /// Find agents with a specific capability.
    pub fn find_by_capability(&self, capability: &str) -> Vec<&AgentProfile> {
        self.agents.values()
            .filter(|a| a.capabilities.iter().any(|c| c.name == capability))
            .collect()
    }

    /// Get a registered agent.
    pub fn get_agent(&self, agent_id: &str) -> Option<&AgentProfile> {
        self.agents.get(agent_id)
    }

    /// Get message history.
    pub fn history(&self) -> &[AcpMessage] {
        &self.history
    }

    /// Number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

impl Default for AcpRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_profile(id: &str) -> AgentProfile {
        AgentProfile {
            agent_id: id.to_string(),
            name: format!("Agent {}", id),
            capabilities: vec![AgentCapability {
                name: "test".into(),
                description: "Test capability".into(),
                input_schema: serde_json::json!({}),
                output_schema: serde_json::json!({}),
            }],
            max_context_tokens: 4096,
            modalities: vec!["text".into()],
        }
    }

    #[test]
    fn test_router_register() {
        let mut router = AcpRouter::new();
        router.register(test_profile("a1"));
        router.register(test_profile("a2"));
        assert_eq!(router.agent_count(), 2);
    }

    #[test]
    fn test_router_unregister() {
        let mut router = AcpRouter::new();
        router.register(test_profile("a1"));
        router.unregister("a1");
        assert_eq!(router.agent_count(), 0);
    }

    #[test]
    fn test_router_route_direct() {
        let mut router = AcpRouter::new();
        router.register(test_profile("a1"));
        router.register(test_profile("a2"));

        let msg = AcpMessage {
            id: "m1".into(),
            from: "a1".into(),
            to: "a2".into(),
            msg_type: AcpMessageType::ContextRequest,
            payload: serde_json::json!({"query": "test"}),
            timestamp: 0,
            correlation_id: None,
        };

        let results = router.route(msg).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].to, "a2");
    }

    #[test]
    fn test_router_route_broadcast() {
        let mut router = AcpRouter::new();
        router.register(test_profile("a1"));
        router.register(test_profile("a2"));
        router.register(test_profile("a3"));

        let msg = AcpMessage {
            id: "m1".into(),
            from: "a1".into(),
            to: "broadcast".into(),
            msg_type: AcpMessageType::StatusUpdate,
            payload: serde_json::json!({"status": "idle"}),
            timestamp: 0,
            correlation_id: None,
        };

        let results = router.route(msg).unwrap();
        assert_eq!(results.len(), 2); // a2 and a3, not a1
    }

    #[test]
    fn test_router_unknown_sender() {
        let mut router = AcpRouter::new();
        let msg = AcpMessage {
            id: "m1".into(),
            from: "unknown".into(),
            to: "a2".into(),
            msg_type: AcpMessageType::ContextRequest,
            payload: serde_json::json!({}),
            timestamp: 0,
            correlation_id: None,
        };
        assert!(router.route(msg).is_err());
    }

    #[test]
    fn test_router_unknown_recipient() {
        let mut router = AcpRouter::new();
        router.register(test_profile("a1"));

        let msg = AcpMessage {
            id: "m1".into(),
            from: "a1".into(),
            to: "nonexistent".into(),
            msg_type: AcpMessageType::ContextRequest,
            payload: serde_json::json!({}),
            timestamp: 0,
            correlation_id: None,
        };
        assert!(router.route(msg).is_err());
    }

    #[test]
    fn test_find_by_capability() {
        let mut router = AcpRouter::new();
        let mut profile = test_profile("a1");
        profile.capabilities.push(AgentCapability {
            name: "code_gen".into(),
            description: "Code generation".into(),
            input_schema: serde_json::json!({}),
            output_schema: serde_json::json!({}),
        });
        router.register(profile);
        router.register(test_profile("a2"));

        let found = router.find_by_capability("code_gen");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].agent_id, "a1");
    }

    #[test]
    fn test_history() {
        let mut router = AcpRouter::new();
        router.register(test_profile("a1"));
        router.register(test_profile("a2"));

        let msg = AcpMessage {
            id: "m1".into(),
            from: "a1".into(),
            to: "a2".into(),
            msg_type: AcpMessageType::ContextRequest,
            payload: serde_json::json!({}),
            timestamp: 0,
            correlation_id: None,
        };
        router.route(msg).unwrap();
        assert_eq!(router.history().len(), 1);
    }

    #[test]
    fn test_message_serialization() {
        let msg = AcpMessage {
            id: "m1".into(),
            from: "a1".into(),
            to: "a2".into(),
            msg_type: AcpMessageType::TaskDelegate,
            payload: serde_json::json!({"task": "generate"}),
            timestamp: 12345,
            correlation_id: Some("corr1".into()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: AcpMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "m1");
        assert_eq!(parsed.correlation_id, Some("corr1".into()));
    }
}
