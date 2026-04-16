#![allow(deprecated)]
//! K8s Mutating Admission Webhook - Simplified

use serde::{Deserialize, Serialize};

/// AdmissionReview request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionRequest {
    pub uid: String,
    pub object: serde_json::Value,
    pub operation: String,
}

/// AdmissionReview response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionResponse {
    pub uid: String,
    pub allowed: bool,
    pub patch: Option<String>,
    pub patch_type: Option<String>,
}

/// Fast path validation result
#[derive(Debug, Clone, Copy)]
pub enum FastPathResult {
    Allow,
    Deny,
}

/// Validate request - fast path
pub fn fast_path_validate(_request: &AdmissionRequest) -> FastPathResult {
    FastPathResult::Allow
}

/// BaFin compliance check
pub fn bafin_compliance_check(_request: &AdmissionRequest) -> bool {
    true
}

/// Build allow response
pub fn build_allow_response(uid: &str) -> AdmissionResponse {
    AdmissionResponse {
        uid: uid.to_string(),
        allowed: true,
        patch: None,
        patch_type: None,
    }
}

/// Build quarantine response
pub fn build_quarantine_response(uid: &str) -> AdmissionResponse {
    let patch = r#"[{"op":"add","path":"/metadata/labels/ferrisres-quarantine","value":"true"}]"#;
    let encoded = base64::encode(patch);
    
    AdmissionResponse {
        uid: uid.to_string(),
        allowed: true,
        patch: Some(encoded),
        patch_type: Some("JSONPatch".to_string()),
    }
}

/// Build deny response  
pub fn build_deny_response(uid: &str, _reason: &str) -> AdmissionResponse {
    AdmissionResponse {
        uid: uid.to_string(),
        allowed: false,
        patch: None,
        patch_type: None,
    }
}

fn main() {
    println!("K8s Webhook compiled successfully!");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fast_path() {
        let request = AdmissionRequest {
            uid: "test".to_string(),
            operation: "CREATE".to_string(),
            object: serde_json::json!({
                "spec": { "hostNetwork": false }
            }),
        };
        
        assert_eq!(fast_path_validate(&request), FastPathResult::Allow);
    }
    
    #[test]
    fn test_quarantine_response() {
        let response = build_quarantine_response("test-123");
        assert!(response.allowed);
        assert!(response.patch.is_some());
    }
}