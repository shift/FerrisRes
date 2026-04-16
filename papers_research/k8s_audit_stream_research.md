# K8s Continuous Audit Stream Research

## Overview
Continuous audit observability for Kubernetes clusters using:
- API Server Audit Logs (policy/v1)
- Cilium Hubble Network Flows (L3-L7)
- Block AttnRes for multi-day context maintenance

## API Server Audit Logs

### Audit Policy Levels
| Level | Events Captured |
|-------|-----------------|
| None | No events |
| Metadata | Request metadata only (user, timestamp, verb) |
| Request | Metadata + request body |
| RequestResponse | Full request + response |

### Event Structure (audit.k8s.io/v1)
```json
{
  "kind": "Event",
  "level": "RequestResponse",
  "auditID": "abc123",
  "stage": "ResponseComplete",
  "requestURI": "/api/v1/namespaces/default/pods",
  "verb": "create",
  "user": {
    "username": "admin@example.com",
    "groups": ["system:masters"]
  },
  "objectRef": {
    "resource": "pods",
    "namespace": "default",
    "name": "malicious-pod",
    "apiVersion": "v1"
  },
  "responseStatus": {
    "metadata": {},
    "code": 201
  },
  "requestObject": {
    "kind": "Pod",
    "spec": { "hostIPC": true }
  }
}
```

## Cilium Hubble Network Flows

### Flow Types
| Type | Description |
|------|-------------|
| L3 | Source/dest IP, protocol, dropped |
| L4 | Ports, TCP flags, bytes/packets |
| L7 | HTTP, DNS, Kafka, gRPC details |

### Hubble Export
```yaml
# Flow record
flow:
  source:
    identity: 1234
    namespace: production
    pod: frontend-abc123
  destination:
    identity: 5678
    pod: backend-xyz789
  l4:
    protocol: TCP
    source_port: 8080
    dest_port: 5432
  verdict: FORWARDED
  timestamp: 2026-04-16T10:00:00Z
```

## Block AttnRes Integration

### Context Maintenance
- Store audit events in Block AttnRes hierarchy
- Query by: time range, namespace, user, resource type
- Track policy changes over weeks/months

### Token Sequence Design
```
[cluster_name] [namespace] [verb] [resource] [user] [result] [policy_decision]
```

## Policy Enforcement

### OPA Gatekeeper
```rego
package k8srequiredlabels

deny[msg] {
  input.review.kind.kind == "Pod"
  not input.review.object.metadata.labels["environment"]
  msg = "Pods must have environment label"
}
```

### Kyverno Policies
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-host-path
spec:
  validationFailureAction: Audit
  rules:
  - name: host-path
    match:
      resources:
        kinds:
        - Pod
    validate:
      pattern:
        spec:
          hostPath: null
```

### NetworkPolicy Quarantine
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: quarantine-malicious
  namespace: default
spec:
  podSelector:
    matchLabels:
      threat-level: critical
  policyTypes:
  - Ingress
  - Egress
```

## Implementation Design

### K8sAuditStreamEncoder
```rust
pub struct K8sAuditStreamEncoder {
    cluster: String,
    source: AuditSource,  // API server or Hubble
}

pub enum AuditSource {
    ApiServer(AuditPolicy),
    Hubble(CiliumEndpoint),
}

impl StreamEncoder for K8sAuditStreamEncoder {
    fn encode(&mut self, event: &AuditEvent) -> Vec<u32> {
        // Tokenize: verb + resource + user + decision
    }
}
```

### PolicyMutationHead
```rust
pub trait PolicyMutationHead {
    async fn apply_gatekeeper(&self, constraint: &str) -> Result<()>;
    async fn apply_kyverno(&self, policy: &KyvernoPolicy) -> Result<()>;
    async fn quarantine_network(&self, pod: &PodSelector) -> Result<()>;
}
```

## References
- K8s Audit: https://kubernetes.io/docs/tasks/debug/debug-cluster/audit/
- Cilium Hubble: https://docs.cilium.io/en/stable/observability/hubble/
- OPA Gatekeeper: https://open-policy-agent.github.io/gatekeeper/website/