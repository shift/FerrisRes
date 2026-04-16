# BCI Research — EEG Stream Encoder, Neural Action Bridge

## Overview
Brain-Computer Interface (BCI) for neural control:
- EEG/EMG data formats (EDF/BDF)
- Multi-channel voltage time series
- Prosthetic control via NeuralActionBridge

## EEG Data Formats

### EDF (European Data Format)
```
Header (256 bytes) + Data Records
- Patient ID, Recording Date/Time
- Channels (up to 64)
- Sample rate per channel
- Physical dimensions
```

### BDF (Biosemi Data Format)
- 24-bit resolution
- Up to 256 channels
- Higher dynamic range than EDF

### Sample Structure
```rust
pub struct EegSample {
    pub timestamp_ns: u64,
    pub channels: Vec<f32>,  // 16-256 channels
    pub sample_rate_hz: u32,
}
```

## NeuralStreamEncoder

```rust
pub struct NeuralStreamEncoder {
    num_channels: u32,
    sample_rate_hz: u32,
}

impl NeuralStreamEncoder {
    pub fn encode(&self, sample: &EegSample) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Channel activity tokens (which channels have signal)
        for (i, &v) in sample.channels.iter().enumerate() {
            if v.abs() > 1e-6 {  // noise floor
                tokens.push(i as u32);
                tokens.push(quantize_channel(v));
            }
        }
        
        tokens
    }
}
```

## MNE-Python Reference Pipeline
```python
# Typical preprocessing
import mne

# Bandpass filter 1-40 Hz
raw.filter(1, 40, fir_design='firwin')

# ICA for artifact removal
ica = ICA(n_components=20)
ica.fit(raw)
ica.exclude = [1, 3]  # eye blinks

# Epochs
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5)
```

## NeuralActionBridge

### Control Targets
```rust
pub enum NeuralAction {
    // Prosthetic
    JointAngle { joint: JointId, angle: f32 },
    GripForce { force: f32 },
    // Cursor control
    CursorMove { dx: f32, dy: f32 },
    Click,
}

pub struct NeuralActionBridge {
    model: Arc<StudentModel>,
}
```

### Mapping Neural States to Actions
```rust
impl NeuralActionBridge {
    pub fn predict_action(&self, eeg: &EegSample) -> NeuralAction {
        // Simplified: classify EEG patterns to actions
        // Real implementation would use trained classifier
        
        // Check motor cortex activity
        let motor_channels = &eeg.channels[10..14];  // C3, C4
        let power = motor_channels.iter().map(|x| x * x).sum::<f32>() / 4.0;
        
        if power > 0.5 {
            NeuralAction::Click
        } else {
            // Default: cursor movement
            NeuralAction::CursorMove { dx: 0.0, dy: 0.0 }
        }
    }
}
```

## References
- EDF: https://www.edfplus.info/
- MNE-Python: https://mne.tools/
- BCI competitions: https://bbci.de/