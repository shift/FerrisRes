# K8s Mutating Admission Webhook Research

## Overview
Kubernetes admission control with:
- MutatingWebhookConfiguration
- Fail-closed hybrid with quarantine fallback
- Speculative fast-path using distilled model

## AdmissionReview Schema

### Request
```json
{
  "kind": "AdmissionReview",
  "apiVersion": "admission.k8s.io/v1",
  "request": {
    "uid": "12345678-1234-1234-1234-123456789012",
    "kind": {"group": "", "version": "v1", "kind": "Pod"},
    "resource": {"group": "", "version": "v1", "resource": "pods"},
    "namespace": "default",
    "operation": "CREATE",
    "userInfo": {
      "username": "admin@example.com",
      "groups": ["system:masters"]
    },
    "object": {
      "apiVersion": "v1",
      "kind": "Pod",
      "metadata": {"name": "my-pod", "namespace": "default"},
      "spec": {
        "containers": [{
          "name": "nginx",
          "image": "nginx:latest"
        }]
      }
    }
  }
}
```

### Response
```json
{
  "kind": "AdmissionReview",
  "apiVersion": "admission.k8s.io/v1",
  "response": {
    "uid": "12345678-1234-1234-1234-123456789012",
    "allowed": true,
    "patchType": "JSONPatch",
    "patch": "W3sgb3BZXJhdGlvbiI6eyJhZGQifV1d"
  }
}
```

## Webhook Configuration

### MutatingWebhookConfiguration
```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: ferrisres-policy-webhook
webhooks:
- name: policy.ferrisres.svc
  clientConfig:
    service:
      name: ferrisres-admission
      namespace: ferrisres-system
      path: "/mutate"
    caBundle: LS0tLS1CRUdJTi...
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods", "deployments"]
  namespaceSelector:
    matchLabels:
      ferrisres-enforce: "true"
  timeoutSeconds: 2
  failurePolicy: Fail  # Fail-closed
  sideEffects: None
  admissionReviewVersions: ["v1", "v1beta1"]
  timeoutSeconds: 2
```

## Hybrid Approach

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     K8s API Server                          │
└─────────────────────────────────────────────────────────────┘
                            │
                    Admission Request
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│  Fast Path          │              │  Full Inference     │
│  Quantized Model    │              │  (if budget allows) │
│  (~10ms)            │              │  (~500ms)           │
└─────────────────────┘              └─────────────────────┘
         │                                      │
         ▼                                      ▼
    ALLOW (if clean)                    ALLOW/DENY
                            │
                     Quarantine Fallback
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                      ▼
   Inject NetworkPolicy           Clear after async audit
   (deny all ingress/egress)
```

### Fast-Path Implementation
```rust
pub struct FastPathValidator {
    model: QuantizedModel,  // 8-bit weights
    max_latency: Duration,   // 10ms
}

impl FastPathValidator {
    pub fn validate(&self, pod: &Pod) -> ValidationResult {
        // Quick heuristic checks
        if self.has_untrusted_image(&pod) {
            return ValidationResult::NeedsFullCheck;
        }
        
        // Run quantized inference
        let tokens = self.tokenize_pod(pod);
        let start = Instant::now();
        let prediction = self.model.forward(&tokens);
        
        if start.elapsed() > self.max_latency {
            return ValidationResult::NeedsFullCheck;
        }
        
        if prediction.is_compliant() {
            ValidationResult::Allow
        } else {
            ValidationResult::NeedsFullCheck
        }
    }
}
```

### Quarantine NetworkPolicy
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: quarantine-fallback
spec:
  podSelector:
    matchLabels:
      ferrisres-quarantine: "true"
  policyTypes:
  - Ingress
  - Egress
  ingress: []  # Deny all
  egress:
  # Only allow logging
  - to:
    - namespaceSelector:
        matchLabels:
          kube-system: "true"
    ports:
    - protocol: TCP
      port: 53
```

### Out-of-Band Audit Clearing
```rust
pub async fn clear_quarantine(
    client: &Client,
    pod_name: &str,
    namespace: &str,
) -> Result<()> {
    // Async audit completed successfully
    // Remove quarantine label + NetworkPolicy
    
    let pods = client
        .list_namespaced_pod(namespace)
        .await?
        .into_iter()
        .filter(|p| p.name() == pod_name);
    
    for mut pod in pods {
        pod.labels_mut().remove("ferrisres-quarantine");
        client.replace_pod(&pod).await?;
    }
    
    Ok(())
}
```

## Webhook Server Implementation

### Axum Handler
```rust
pub async fn mutate(
    State(state): State<Arc<AdmissionState>>,
    Json(review): Json<AdmissionReview>,
) -> Json<AdmissionReviewResponse> {
    let request = review.request.unwrap();
    let pod: Pod = serde_json::from_slice(&request.object).unwrap();
    
    // Fast path first
    match state.fast_path.validate(&pod) {
        ValidationResult::Allow => {
            Json(allow_response(request.uid))
        }
        ValidationResult::Deny => {
            Json(deny_response(request.uid, "Policy violation"))
        }
        ValidationResult::NeedsFullCheck => {
            // Run full model inference
            match state.full_inference.validate(&pod).await {
                Ok(true) => Json(allow_response(request.uid)),
                Ok(false) => Json(deny_response(request.uid, "Policy violation")),
                Err(_) => {
                    // Quarantine fallback
                    Json(quarantine_response(request.uid, &pod))
                }
            }
        }
    }
}
```

### JSONPatch Injection
```rust
fn inject_quarantine_label(uid: &str, pod: &Pod) -> AdmissionReviewResponse {
    let patch = json_patch!([
        { "op": "add", "path": "/metadata/labels/ferrisres-quarantine", "value": "true" }
    ]);
    
    AdmissionReviewResponse {
        response: Response {
            uid: uid.to_string(),
            allowed: true,
            patch_type: Some("JSONPatch"),
            patch: Some(base64::encode(serde_json::to_vec(&patch).unwrap())),
            ..Default::default()
        }
    }
}
```

## Node Pool Isolation

### Dedicated Node Pool
```yaml
apiVersion: v1
kind: NodePool
metadata:
  name: admission-webhook
spec:
  taints:
  - key: "ferrisres-webhook"
    effect: NoSchedule
  tolerations:
  - key: "ferrisres-webhook"
    effect: Exist
    operator: Exists
```

## Distroless Container (Nix)

### Nix Flake
```nix
{
  outputs = { self, nixpkgs }: {
    packages.x86_64-linux.admission-webhook = 
      nixpkgs.buildGoModule {
        name = "admission-webhook";
        src = self;
        vendor = ./vendor;
        
        # Distroless base image
        buildImage = {
          from = "gcr.io/distroless/static:nonroot";
          entrypoint = [ "/admission-webhook" ];
        };
      };
  };
}
```

## References
- K8s Admission Webhooks: https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/
- JSONPatch: https://tools.ietf.org/html/rfc6902
- Distroless: https://github.com/GoogleContainerTools/distroless