//! Kubernetes Audit Stream Encoder + Policy Mutation Head
//! 
//! Continuous audit observability for K8s clusters:
//! - API server audit log ingestion
//! - Cilium Hubble network flows
//! - Policy mutation (OPA Gatekeeper, Kyverno, NetworkPolicy)
//! - BaFin compliance for financial services

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum K8sAuditError {
    #[error("API error: {0}")]
    Api(String),
    
    #[error("Policy generation: {0}")]
    Policy(String),
    
    #[error("Compliance violation: {0}")]
    Compliance(String),
    
    #[error("Network flow: {0}")]
    NetworkFlow(String),
}

// ============================================================================
// Data Types
// ============================================================================

/// K8s API server audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub kind: String,
    pub level: AuditLevel,
    pub audit_id: String,
    pub stage: AuditStage,
    pub request_uri: String,
    pub verb: String,
    pub user: UserInfo,
    pub object_ref: Option<ObjectRef>,
    pub response_status: Option<ResponseStatus>,
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AuditLevel {
    None,
    Metadata,
    Request,
    RequestResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditStage {
    RequestReceived,
    ResponseStarted,
    ResponseComplete,
    Panic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub username: String,
    #[serde(default)]
    pub groups: Vec<String>,
    #[serde(default)]
    pub extra: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectRef {
    pub resource: String,
    #[serde(default]
    pub namespace: Option<String>,
    pub name: String,
    pub api_version: String,
    #[serde(default)]
    pub subresource: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseStatus {
    pub metadata: serde_json::Value,
    #[serde(rename = "code")]
    pub status_code: u16,
    #[serde(default)]
    pub message: Option<String>,
}

/// Cilium Hubble network flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkFlow {
    pub flow_type: FlowType,
    pub source: Endpoint,
    pub destination: Endpoint,
    pub l3_protocol: String,
    pub l4_protocol: Option<L4Protocol>,
    pub verdict: Verdict,
    pub timestamp_ns: u64,
    #[serde(default)]
    pub dropped_reason: Option<String>,
    #[serde(default)]
    pub http: Option<HttpInfo>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FlowType {
    L3,
    L4,
    L7,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    pub identity: u32,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub pod: Option<String>,
    #[serde(default)]
    pub ip: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L4Protocol {
    pub protocol: String,
    pub source_port: u16,
    pub destination_port: u16,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Verdict {
    Forwarded,
    Dropped,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpInfo {
    pub method: Option<String>,
    pub url: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

/// Policy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Policy {
    Gatekeeper(GatekeeperConstraint),
    Kyverno(KyvernoPolicy),
    NetworkPolicy(NetworkPolicySpec),
}

/// OPA Gatekeeper constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatekeeperConstraint {
    pub api_version: String,
    pub kind: String,
    pub metadata: Metadata,
    pub spec: GatekeeperSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub name: String,
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatekeeperSpec {
    #[serde(default)]
    pub enforcement_action: String,
    pub match: GatekeeperMatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatekeeperMatch {
    #[serde(default)]
    pub kinds: Vec<KindSpec>,
    #[serde(default)]
    pub namespaces: Vec<String>,
    #[serde(default)]
    pub label_selector: LabelSelector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KindSpec {
    pub api_groups: Vec<String>,
    pub kinds: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSelector {
    #[serde(default)]
    pub match_labels: HashMap<String, String>,
    #[serde(default)]
    pub match_expressions: Vec<LabelExpression>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelExpression {
    pub key: String,
    pub operator: String,
    pub values: Vec<String>,
}

/// Kyverno policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KyvernoPolicy {
    pub api_version: String,
    pub kind: String,
    pub metadata: Metadata,
    pub spec: KyvernoSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KyvernoSpec {
    #[serde(default)]
    pub background: bool,
    #[serde(default)]
    pub validation_failure_action: String,
    #[serde(default)]
    pub rules: Vec<KyvernoRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KyvernoRule {
    pub name: String,
    #[serde(default)]
    pub match: KyvernoMatch,
    #[serde(default)]
    pub validate: Option<KyvernoValidate>,
    #[serde(default)]
    pub mutate: Option<KyvernoMutate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KyvernoMatch {
    #[serde(default)]
    pub resources: KyvernoResourceMatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KyvernoResourceMatch {
    #[serde(default)]
    pub kinds: Vec<String>,
    #[serde(default)]
    pub namespaces: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KyvernoValidate {
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub pattern: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KyvernoMutate {
    #[serde(default)]
    pub patches: Vec<serde_json::Value>,
}

/// NetworkPolicy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicySpec {
    pub api_version: String,
    pub kind: String,
    pub metadata: Metadata,
    pub spec: NetworkPolicySpecInner,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicySpecInner {
    pub pod_selector: LabelSelector,
    #[serde(default)]
    pub policy_types: Vec<String>,
    #[serde(default)]
    pub ingress: Vec<NetworkPolicyIngressRule>,
    #[serde(default)]
    pub egress: Vec<NetworkPolicyEgressRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyIngressRule {
    #[serde(default)]
    pub ports: Vec<NetworkPolicyPort>,
    #[serde(default)]
    pub from: Vec<NetworkPolicyPeer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyEgressRule {
    #[serde(default)]
    pub ports: Vec<NetworkPolicyPort>,
    #[serde(default)]
    pub to: Vec<NetworkPolicyPeer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyPort {
    #[serde(default)]
    pub protocol: Option<String>,
    #[serde(default)]
    pub port: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicyPeer {
    #[serde(default)]
    pub pod_selector: Option<LabelSelector>,
    #[serde(default)]
    pub namespace_selector: Option<LabelSelector>,
    #[serde(default)]
    pub ip_block: Option<IpBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpBlock {
    pub cidr: String,
    #[serde(default)]
    pub except: Vec<String>,
}

// ============================================================================
// K8s Audit Stream Encoder
// ============================================================================

/// Encodes K8s audit events for Block AttnRes
pub struct K8sAuditStreamEncoder {
    cluster_name: String,
    event_buffer: Vec<AuditEvent>,
    max_buffer_size: usize,
}

impl K8sAuditStreamEncoder {
    pub fn new(cluster_name: String, max_buffer_size: usize) -> Self {
        Self {
            cluster_name,
            event_buffer: Vec::new(),
            max_buffer_size,
        }
    }
    
    /// Encode audit event to tokens
    pub fn encode_audit(&mut self, event: &AuditEvent) -> Vec<u32> {
        if self.event_buffer.len() >= self.max_buffer_size {
            self.event_buffer.remove(0);
        }
        self.event_buffer.push(event.clone());
        
        self.tokenize_audit(event)
    }
    
    /// Encode network flow to tokens
    pub fn encode_flow(&mut self, flow: &NetworkFlow) -> Vec<u32> {
        self.tokenize_flow(flow)
    }
    
    fn tokenize_audit(&self, event: &AuditEvent) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Cluster context
        tokens.push(hash_string(&self.cluster_name));
        
        // Verb
        tokens.push(match event.verb.as_str() {
            "get" => 1,
            "create" => 2,
            "update" => 3,
            "delete" => 4,
            "patch" => 5,
            _ => 0,
        });
        
        // Resource type
        if let Some(obj) = &event.object_ref {
            tokens.push(hash_string(&obj.resource));
            if let Some(ns) = &obj.namespace {
                tokens.push(hash_string(ns));
            }
        }
        
        // User (hashed)
        tokens.push(hash_string(&event.user.username));
        
        // Response status
        if let Some(rs) = &event.response_status {
            tokens.push(rs.status_code as u32);
        }
        
        // Level
        tokens.push(match event.level {
            AuditLevel::None => 0,
            AuditLevel::Metadata => 1,
            AuditLevel::Request => 2,
            AuditLevel::RequestResponse => 3,
        });
        
        tokens
    }
    
    fn tokenize_flow(&self, flow: &NetworkFlow) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Flow type
        tokens.push(match flow.verdict {
            Verdict::Forwarded => 1,
            Verdict::Dropped => 2,
            Verdict::Error => 3,
        });
        
        // Source namespace/pod
        if let Some(ns) = &flow.source.namespace {
            tokens.push(hash_string(ns));
        }
        if let Some(pod) = &flow.source.pod {
            tokens.push(hash_string(pod));
        }
        
        // Dest namespace/pod
        if let Some(ns) = &flow.destination.namespace {
            tokens.push(hash_string(ns));
        }
        if let Some(pod) = &flow.destination.pod {
            tokens.push(hash_string(pod));
        }
        
        // Protocol
        if let Some(l4) = &flow.l4_protocol {
            tokens.push(l4.destination_port as u32);
        }
        
        // Identity
        tokens.push(flow.source.identity);
        tokens.push(flow.destination.identity);
        
        tokens
    }
}

/// Hash string to token
fn hash_string(s: &str) -> u32 {
    s.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32)) % 65536
}

// ============================================================================
// Policy Mutation Head
// ============================================================================

/// Policy mutation head for K8s policy generation
pub trait PolicyMutationHead {
    fn generate_gatekeeper(&self, constraint: &GatekeeperConstraint) -> Result<String, K8sAuditError>;
    fn generate_kyverno(&self, policy: &KyvernoPolicy) -> Result<String, K8sAuditError>;
    fn generate_network_policy(&self, policy: &NetworkPolicySpec) -> Result<String, K8sAuditError>;
}

/// Implementation
pub struct K8sPolicyMutationHead {
    template_dir: String,
}

impl K8sPolicyMutationHead {
    pub fn new(template_dir: String) -> Self {
        Self { template_dir }
    }
}

impl PolicyMutationHead for K8sPolicyMutationHead {
    fn generate_gatekeeper(&self, constraint: &GatekeeperConstraint) -> Result<String, K8sAuditError> {
        serde_yaml::to_string(constraint)
            .map_err(|e| K8sAuditError::Policy(format!("Gatekeeper YAML: {}", e)))
    }
    
    fn generate_kyverno(&self, policy: &KyvernoPolicy) -> Result<String, K8sAuditError> {
        serde_yaml::to_string(policy)
            .map_err(|e| K8sAuditError::Policy(format!("Kyverno YAML: {}", e)))
    }
    
    fn generate_network_policy(&self, policy: &NetworkPolicySpec) -> Result<String, K8sAuditError> {
        serde_yaml::to_string(policy)
            .map_err(|e| K8sAuditError::Policy(format!("NetworkPolicy YAML: {}", e)))
    }
}

// ============================================================================
// BaFin Compliance Oracle
// ============================================================================

/// EU data residency regions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataRegion {
    Germany,
    France,
    Netherlands,
    EU,
    NonEU,
}

/// BaFin compliance rules for financial services
pub struct BaFinComplianceOracle {
    allowed_regions: Vec<DataRegion>,
    protected_namespaces: Vec<String>,
    audit_log_retention_days: u32,
}

impl BaFinComplianceOracle {
    pub fn new() -> Self {
        Self {
            allowed_regions: vec![DataRegion::Germany, DataRegion::EU],
            protected_namespaces: vec![
                "core-banking".to_string(),
                "ledger".to_string(),
                "payments".to_string(),
            ],
            audit_log_retention_days: 2555,  // 7 years (BaFin requirement)
        }
    }
    
    /// Verify data residency compliance
    pub fn verify_residency(&self, source_region: DataRegion, target_region: DataRegion) -> bool {
        self.allowed_regions.contains(&source_region) && 
        self.allowed_regions.contains(&target_region)
    }
    
    /// Verify namespace protection
    pub fn verify_namespace_protection(&self, namespace: &str) -> bool {
        // Protected namespaces only allow NetworkPolicy mutations
        self.protected_namespaces.contains(&namespace.to_string())
    }
    
    /// Verify manifest compliance
    pub fn verify_manifest(&self, manifest: &K8sManifest) -> Result<bool, K8sAuditError> {
        match manifest.kind.as_str() {
            "Pod" | "Deployment" => {
                // Check security context
                if let Some(sec_ctx) = &manifest.security_context {
                    if sec_ctx.run_as_non_root {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            "NetworkPolicy" => {
                // Network policies are allowed in protected namespaces
                Ok(true)
            }
            _ => Ok(true),
        }
    }
}

/// K8s manifest
#[derive(Debug, Clone)]
pub struct K8sManifest {
    pub kind: String,
    pub metadata: Metadata,
    pub namespace: Option<String>,
    pub spec: serde_json::Value,
    pub security_context: Option<SecurityContext>,
}

#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub run_as_non_root: bool,
    pub run_as_user: Option<u32>,
    pub allow_privilege_escalation: bool,
}

impl BaFinComplianceOracle {
    pub fn check_logit_compliance(&self, token_region: u32) -> bool {
        // Map token region to data region
        match token_region % 5 {
            0 => self.allowed_regions.contains(&DataRegion::Germany),
            1 => self.allowed_regions.contains(&DataRegion::France),
            2 => self.allowed_regions.contains(&DataRegion::Netherlands),
            3 => self.allowed_regions.contains(&DataRegion::EU),
            _ => false,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_audit_encoder() {
        let mut encoder = K8sAuditStreamEncoder::new("prod-cluster".to_string(), 100);
        
        let event = AuditEvent {
            kind: "Event".to_string(),
            level: AuditLevel::RequestResponse,
            audit_id: "abc123".to_string(),
            stage: AuditStage::ResponseComplete,
            request_uri: "/api/v1/namespaces/default/pods".to_string(),
            verb: "create".to_string(),
            user: UserInfo {
                username: "admin@example.com".to_string(),
                groups: vec!["system:masters".to_string()],
                extra: HashMap::new(),
            },
            object_ref: Some(ObjectRef {
                resource: "pods".to_string(),
                namespace: Some("default".to_string()),
                name: "my-pod".to_string(),
                api_version: "v1".to_string(),
                subresource: None,
            }),
            response_status: Some(ResponseStatus {
                metadata: serde_json::Value::Null,
                status_code: 201,
                message: None,
            }),
            annotations: HashMap::new(),
        };
        
        let tokens = encoder.encode_audit(&event);
        assert!(!tokens.is_empty());
    }
    
    #[test]
    fn test_gatekeeper_generation() {
        let head = K8sPolicyMutationHead::new("/templates".to_string());
        
        let constraint = GatekeeperConstraint {
            api_version: "constraints.gatekeeper.sh/v1beta1".to_string(),
            kind: "K8sRequiredLabels".to_string(),
            metadata: Metadata {
                name: "require-environment".to_string(),
                namespace: None,
            },
            spec: GatekeeperSpec {
                enforcement_action: "deny".to_string(),
                match: GatekeeperMatch {
                    kinds: vec![KindSpec {
                        api_groups: vec!["".to_string()],
                        kinds: vec!["Pod".to_string()],
                    }],
                    namespaces: vec![],
                    label_selector: LabelSelector {
                        match_labels: HashMap::new(),
                        match_expressions: vec![],
                    },
                },
            },
        };
        
        let yaml = head.generate_gatekeeper(&constraint);
        assert!(yaml.is_ok());
    }
    
    #[test]
    fn test_bafin_residency() {
        let oracle = BaFinComplianceOracle::new();
        
        assert!(oracle.verify_residency(DataRegion::Germany, DataRegion::EU));
        assert!(!oracle.verify_residency(DataRegion::NonEU, DataRegion::Germany));
    }
    
    #[test]
    fn test_namespace_protection() {
        let oracle = BaFinComplianceOracle::new();
        
        assert!(oracle.verify_namespace_protection("core-banking"));
        assert!(!oracle.verify_namespace_protection("default"));
    }
}