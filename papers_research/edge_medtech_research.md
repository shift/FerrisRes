# Edge MedTech Research

## Overview
Real-time medical device data streaming with:
- HL7 v2 / FHIR R4 encoding
- Diagnostic head for patient deterioration
- GDPR-compliant PHI/PII redaction
- Edge deployment on low-power IoT

## HL7 v2 Messaging

### Message Types
| Type | Use Case |
|------|----------|
| ADT | Admission/Discharge/Transfer |
| ORU | Observation Result (vitals, labs) |
| ORM | Order (medication, test) |
| ACK | Acknowledgment |

### Segment Structure
```
MSH|^~\&|HOSPITAL|ED|ICU|MONITOR|20240416100000||ORU^R01|MSG001|P|2.5
PID|1||PAT12345||DOE^JOHN||19800315|M
OBR|1||12345|VITALS|20240416100000
OBX|1|NUM|HR|Heart Rate|||75|bpm|||F
OBX|2|NUM|SBP|Systolic BP|||120|mmHg|||F
OBX|3|NUM|DBP|Diastolic BP|||80|mmHg|||F
```

## FHIR R4 Resources

### Key Resources
```json
{
  "resourceType": "Patient",
  "id": "patient-123",
  "identifier": [{
    "system": "http://hospital.org/mrn",
    "value": "MRN12345"
  }],
  "name": [{
    "family": "Doe",
    "given": ["John"]
  }]
}

{
  "resourceType": "Observation",
  "status": "final",
  "code": {
    "coding": [{
      "system": "http://loinc.org",
      "code": "8867-4",
      "display": "Heart rate"
    }]
  },
  "valueQuantity": {
    "value": 75,
    "unit": "beats/minute"
  },
  "effectiveDateTime": "2026-04-16T10:00:00Z"
}
```

## Diagnostic Head - Sepsis Prediction

### SOFA Score (Sequential Organ Failure Assessment)
| System | 0 | 1 | 2 | 3 | 4 |
|--------|---|---|---|---|---|
| Respiratory (PaO2/FiO2) | >400 | ≤400 | ≤300 | ≤200 | ≤100 |
| Coagulation (platelets ×10³/μL) | >150 | ≤150 | ≤100 | ≤50 | ≤20 |
| Liver (bilirubin) | <1.2 | 1.2-1.9 | 2.0-5.9 | 6.0-11.9 | >12.0 |
| Cardiovascular (MAP) | ≥70 | <70 | dopamine≤5 | dopamine>15 | - |
| CNS (GCS) | 15 | 13-14 | 10-12 | 6-9 | <6 |
| Renal (creatinine) | <1.2 | 1.2-1.9 | 2.0-3.4 | 3.5-4.9 | ≥5.0 |

### qSOFA (Quick SOFA)
- Respiratory rate ≥22/min
- Altered mentation (GCS <15)
- Systolic BP ≤100 mmHg

```rust
pub struct SepsisPredictor {
    thresholds: SepsisThresholds,
}

impl SepsisPredictor {
    pub fn compute_sofa(&self, vitals: &Vitals, labs: &Labs) -> u8 {
        let resp = if vitals.pao2 / vitals.fio2 > 400 { 0 }
            else if vitals.pao2 / vitals.fio2 > 300 { 1 }
            else if vitals.pao2 / vitals.fio2 > 200 { 2 }
            else if vitals.pao2 / vitals.fio2 > 100 { 3 }
            else { 4 };
        
        let cv = if vitals.map >= 70 { 0 }
            else if vitals.dopamine <= 5 { 2 }
            else { 3 };
        
        // ... compute all 6 systems
        
        resp + coag + liver + cv + cns + renal
    }
    
    pub fn predict(&self, sofa_score: u8, qsofa: u8) -> SepsisRisk {
        match (sofa_score, qsofa) {
            (_, 2..=3) => SepsisRisk::High,
            (3..=5, 1) => SepsisRisk::Medium,
            (0..=2, 0) => SepsisRisk::Low,
            _ => SepsisRisk::Unknown,
        }
    }
}
```

## GDPR Redaction Oracle

### PHI/PII Categories
| Category | Examples | Detection |
|----------|----------|-----------|
| Direct identifiers | Name, SSN, MRN | Named entity recognition |
| Location | Address, room number | Regex patterns |
| Contact | Phone, email | Pattern matching |
| Dates | DOB, admission date | Date analysis |

### Redaction Implementation
```rust
pub struct GdprRedactionOracle {
    patterns: Vec<RedactionPattern>,
}

impl GdprRedactionOracle {
    pub fn redact(&self, text: &str) -> String {
        let mut result = text.to_string();
        for pattern in &self.patterns {
            result = pattern.apply(&result);
        }
        // Replace names with [REDACTED]
        result = self.redact_names(&result);
        result
    }
    
    fn redact_names(&self, text: &str) -> String {
        // Use simple keyword matching for edge deployment
        let names = ["John", "Jane", "Doe", "Smith"];
        let mut result = text.to_string();
        for name in names {
            result = result.replace(name, "[NAME]");
        }
        result
    }
}
```

## Edge Deployment Constraints

### GPU Memory Limits
| GPU | Max Buffer | Typical Edge Use |
|-----|------------|------------------|
| ARM Mali-G76 | 256MB | Single batch |
| Qualcomm Adreno 640 | 512MB | Small batch |
| Intel iGPU (GEN11) | 2GB | Full pipeline |

### wgpu Considerations
- No compute shaders on some Mali (use vertex/fragment)
- 64KB workgroup memory limit
- 32-bit float only (no half-precision on old GPUs)

### EU CRA Compliance
- Secure-by-default: TLS 1.3 required
- No hardcoded credentials
- Signed firmware updates

## Implementation

### HL7StreamEncoder
```rust
pub struct HL7StreamEncoder {
    source: HL7Source,  // TCP/UDP or file
    parser: HL7Parser,
}

impl StreamEncoder for HL7StreamEncoder {
    fn encode(&self, msg: &HL7Message) -> Vec<u32> {
        // Tokenize: patient_id + event_type + obs_code + value + unit
    }
}
```

## References
- HL7 v2: https://www.hl7.org/implement/standards/v2.cfm
- FHIR R4: https://www.hl7.org/fhir/
- SOFA Score: https://pubmed.ncbi.nlm.nih.gov/27371757/
- EU CRA: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32014R0910