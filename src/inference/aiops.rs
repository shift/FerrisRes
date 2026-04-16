//! AIOps — OTLP Stream Encoder + RemediationHead + SREPolicyOracle
//! 
//! Observability data processing for AI operations:
//! - OTLP stream encoding (metrics, logs, traces)
//! - Auto-remediation actions
//! - SRE policy validation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum AIOpsError {
    #[error("OTLP parse: {0}")]
    Parse(String),
    
    #[error("Remediation: {0}")]
    Remediation(String),
    
    #[error("Policy: {0}")]
    Policy(String),
}

// ============================================================================
// OTLP Types
// ============================================================================

/// OTLP metric point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpMetric {
    pub name: String,
    pub value: f64,
    pub timestamp_ms: u64,
    pub resource: String,
}

/// OTLP log record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpLog {
    pub timestamp_ms: u64,
    pub severity: LogSeverity,
    pub body: String,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// OTLP trace span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpTrace {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub start_ms: u64,
    pub duration_ms: u64,
    pub status: SpanStatus,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanStatus {
    Ok,
    Error,
    Unset,
}

// ============================================================================
// OTLP Stream Encoder
// ============================================================================

/// Encodes OTLP metrics, logs, traces for Block AttnRes
pub struct OTLPStreamEncoder {
    metric_names: HashMap<String, u32>,
    log_tokens: HashMap<String, u32>,
}

impl OTLPStreamEncoder {
    pub fn new() -> Self {
        Self {
            metric_names: HashMap::new(),
            log_tokens: HashMap::new(),
        }
    }
    
    /// Encode metric to tokens
    pub fn encode_metric(&mut self, metric: &OtlpMetric) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Metric name token
        let name_token = *self.metric_names.entry(metric.name.clone()).or_insert_with(
            self.metric_names.len() as u32 + 1000
        );
        tokens.push(name_token);
        
        // Quantized value
        tokens.push(quantize_value(metric.value));
        
        // Time bucket (minute of day)
        tokens.push(((metric.timestamp_ms / 60000) % 1440) as u32);
        
        tokens
    }
    
    /// Encode log to tokens
    pub fn encode_log(&mut self, log: &OtlpLog) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Severity token
        tokens.push(match log.severity {
            LogSeverity::Debug => 10000,
            LogSeverity::Info => 10001,
            LogSeverity::Warning => 10002,
            LogSeverity::Error => 10003,
            LogSeverity::Critical => 10004,
        });
        
        // Extract key tokens from log body
        for word in log.body.split_whitespace().take(8) {
            let token = *self.log_tokens.entry(word.to_string()).or_insert_with(
                self.log_tokens.len() as u32 + 2000
            );
            tokens.push(token);
        }
        
        // Time bucket
        tokens.push(((log.timestamp_ms / 60000) % 1440) as u32);
        
        tokens
    }
    
    /// Encode trace to tokens  
    pub fn encode_trace(&self, trace: &OtlpTrace) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Operation type
        let op_code = hash_string(&trace.operation_name) % 1000;
        tokens.push(op_code);
        
        // Duration bucket (log scale)
        if trace.duration_ms > 0 {
            tokens.push((trace.duration_ms.max(1).log2() as u32).min(20));
        }
        
        // Status
        tokens.push(match trace.status {
            SpanStatus::Ok => 1,
            SpanStatus::Error => 2,
            SpanStatus::Unset => 0,
        });
        
        tokens
    }
}

/// Quantize continuous value to bucket
fn quantize_value(v: f64) -> u32 {
    if v <= 0.0 { 0 }
    else if v < 1.0 { (v * 10.0) as u32 + 1 }
    else if v < 100.0 { (v.log10() * 10.0) as u32 + 11 }
    else if v < 10000.0 { (v.log10() * 5.0) as u32 + 21 }
    else { 31 }
}

fn hash_string(s: &str) -> u32 {
    s.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32))
}

// ============================================================================
// Remediation Head
// ============================================================================

/// K8s remediation actions
pub enum RemediationAction {
    ScaleDeployment { namespace: String, name: String, replicas: i32 },
    SetImage { namespace: String, name: String, container: String, image: String },
    Restart { namespace: String, name: String },
    PatchConfig { namespace: String, name: String, key: String, value: String },
    DeletePod { namespace: String, name: String },
}

/// Remediation head - generates K8s actions from predicted embeddings
pub struct RemediationHead {
    action_templates: Vec<RemediationAction>,
}

impl RemediationHead {
    pub fn new() -> Self {
        Self {
            action_templates: vec![
                RemediationAction::ScaleDeployment { 
                    namespace: "default".to_string(), 
                    name: "".to_string(), 
                    replicas: 1 
                },
            ],
        }
    }
    
    /// Predict remediation action from embeddings
    pub fn predict(&self, embeddings: &[f32]) -> RemediationAction {
        // Simplified: classify embedding to action type
        // Real implementation would use trained policy network
        
        if embeddings.is_empty() {
            return self.action_templates[0].clone();
        }
        
        // Simple heuristic: use embedding magnitude
        let mag = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if mag > 10.0 {
            // High anomaly - scale up
            RemediationAction::ScaleDeployment {
                namespace: "default".to_string(),
                name: "app".to_string(),
                replicas: 3,
            }
        } else {
            // Normal - keep Replica 1
            RemediationAction::ScaleDeployment {
                namespace: "default".to_string(),
                name: "app".to_string(),
                replicas: 1,
            }
        }
    }
    
    /// Generate K8s YAML for action
    pub fn to_k8s_yaml(&self, action: &RemediationAction) -> String {
        match action {
            RemediationAction::ScaleDeployment { namespace, name, replicas } => {
                format!(
                    "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  namespace: {}\n  name: {}\nspec:\n  replicas: {}",
                    namespace, name, replicas
                )
            }
            _ => "apiVersion: v1\nkind: Pod".to_string(),
        }
    }
}

// ============================================================================
// SRE Policy Oracle
// ============================================================================

/// SRE policy validation
pub struct SREPolicyOracle {
    protected_namespaces: Vec<String>,
    forbidden_actions: Vec<String>,
    max_replicas: HashMap<String, i32>,
}

impl SREPolicyOracle {
    pub fn new() -> Self {
        Self {
            protected_namespaces: vec![
                "kube-system".to_string(),
                "kube-public".to_string(),
            ],
            forbidden_actions: vec!["delete".to_string(), "force-delete".to_string()],
            max_replicas: HashMap::new(),
        }
    }
    
    /// Validate remediation action
    pub fn validate(&self, action: &RemediationAction) -> Result<bool, AIOpsError> {
        match action {
            RemediationAction::ScaleDeployment { namespace, replicas } => {
                // Check namespace protection
                if self.protected_namespaces.contains(namespace) {
                    return Err(AIOpsError::Policy(format!(
                        "Cannot modify protected namespace: {}", namespace
                    )));
                }
                
                // Check replica limits
                if let Some(&max) = self.max_replicas.get(namespace) {
                    if *replicas > max {
                        return Err(AIOpsError::Policy(format!(
                            "Replica count {} exceeds max {} for namespace {}",
                            replicas, max, namespace
                        )));
                    }
                }
                
                Ok(true)
            }
            RemediationAction::DeletePod { namespace, .. } => {
                if self.protected_namespaces.contains(namespace) {
                    return Err(AIOpsError::Policy(
                        "Cannot delete from protected namespace".to_string()
                    ));
                }
                Ok(true)
            }
            _ => Ok(true),
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
    fn test_metric_encoder() {
        let mut encoder = OTLPStreamEncoder::new();
        
        let metric = OtlpMetric {
            name: "cpu_usage".to_string(),
            value: 75.5,
            timestamp_ms: 1713267600000,
            resource: "container/app".to_string(),
        };
        
        let tokens = encoder.encode_metric(&metric);
        assert!(!tokens.is_empty());
    }
    
    #[test]
    fn test_log_encoder() {
        let encoder = OTLPStreamEncoder::new();
        
        let log = OtlpLog {
            timestamp_ms: 1713267600000,
            severity: LogSeverity::Error,
            body: "Connection refused to database".to_string(),
            attributes: HashMap::new(),
        };
        
        let tokens = encoder.encode_log(&log);
        assert!(!tokens.is_empty());
    }
    
    #[test]
    fn test_trace_encoder() {
        let encoder = OTLPStreamEncoder::new();
        
        let trace = OtlpTrace {
            trace_id: "abc123".to_string(),
            span_id: "span456".to_string(),
            parent_span_id: None,
            operation_name: "http_request".to_string(),
            start_ms: 1713267600000,
            duration_ms: 150,
            status: SpanStatus::Ok,
            attributes: HashMap::new(),
        };
        
        let tokens = encoder.encode_trace(&trace);
        assert!(!tokens.is_empty());
    }
    
    #[test]
    fn test_remediation_head() {
        let head = RemediationHead::new();
        
        let embeddings = vec![1.0, 2.0, 3.0];
        let action = head.predict(&embeddings);
        
        assert!(matches!(action, RemediationAction::ScaleDeployment { .. }));
    }
    
    #[test]
    fn test_policy_oracle() {
        let oracle = SREPolicyOracle::new();
        
        let action = RemediationAction::ScaleDeployment {
            namespace: "default".to_string(),
            name: "app".to_string(),
            replicas: 2,
        };
        
        let result = oracle.validate(&action);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_protected_namespace() {
        let oracle = SREPolicyOracle::new();
        
        let action = RemediationAction::DeletePod {
            namespace: "kube-system".to_string(),
            name: "coredns".to_string(),
        };
        
        let result = oracle.validate(&action);
        assert!(result.is_err());
    }
}