//! BCI — EEG Stream Encoder + Neural Action Bridge
//! 
//! Brain-Computer Interface for neural control:
//! - EEG/EMG data encoding
//! - Prosthetic control
//! - Neural state classification

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum BciError {
    #[error("EEG: {0}")]
    Eeg(String),
    
    #[error("Neural: {0}")]
    Neural(String),
}

// ============================================================================
// EEG Types
// ============================================================================

/// EEG channel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EegChannel {
    Fp1,  // Frontal pole
    Fp2,
    F3,
    F4,
    C3,    // Central
    C4,
    P3,    // Parietal
    P4,
    O1,    // Occipital
    O2,
    T3,    // Temporal
    T4,
    T5,
    T6,
}

/// EEG sample (one timestep)
#[derive(Debug, Clone)]
pub struct EegSample {
    pub channels: Vec<f32>,   // voltage in microvolts
    pub sample_rate_hz: u32,
    pub timestamp_ms: u64,
}

impl EegSample {
    pub fn new(num_channels: usize, sample_rate_hz: u32) -> Self {
        Self {
            channels: vec![0.0; num_channels],
            sample_rate_hz,
            timestamp_ms: 0,
        }
    }
    
    /// Set channel value
    pub fn set_channel(&mut self, idx: usize, value: f32) {
        if idx < self.channels.len() {
            self.channels[idx] = value;
        }
    }
    
    /// Calculate band power in frequency band
    pub fn band_power(&self, _low_hz: f32, _high_hz: f32) -> f32 {
        // Simplified - return RMS of signal
        let sum_sq = self.channels.iter().map(|x| x * x).sum::<f32>();
        (sum_sq / self.channels.len() as f32).sqrt()
    }
}

/// EEG epoch (multiple samples)
#[derive(Debug, Clone)]
pub struct EegEpoch {
    pub samples: Vec<EegSample>,
    pub channel_labels: Vec<String>,
}

impl EegEpoch {
    pub fn new(num_channels: usize, num_samples: usize, sample_rate_hz: u32) -> Self {
        Self {
            samples: (0..num_samples).map(|_| EegSample::new(num_channels, sample_rate_hz)).collect(),
            channel_labels: vec![],
        }
    }
    
    /// Calculate mean channel value
    pub fn mean(&self, channel_idx: usize) -> f32 {
        let sum: f32 = self.samples.iter()
            .filter_map(|s| s.channels.get(channel_idx).copied())
            .sum();
        sum / self.samples.len() as f32
    }
}

// ============================================================================
// EEG Stream Encoder
// ============================================================================

/// Encodes EEG data for Block AttnRes
pub struct EegStreamEncoder {
    pub num_channels: u32,
    pub sample_rate_hz: u32,
    pub reference_channel: Option<String>,
    pub noise_floor_uv: f32,
}

impl EegStreamEncoder {
    pub fn new(num_channels: u32, sample_rate_hz: u32) -> Self {
        Self {
            num_channels,
            sample_rate_hz,
            reference_channel: None,
            noise_floor_uv: 1.0,
        }
    }
    
    /// Encode epoch to tokens
    pub fn encode_epoch(&self, epoch: &EegEpoch) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Channel activity tokens
        for (i, mean) in epoch.channel_labels.iter().enumerate() {
            if epoch.mean(i) > self.noise_floor_uv {
                tokens.push(i as u32 + 100);
            }
        }
        
        // Average power token
        let total_power: f32 = epoch.samples.iter()
            .map(|s| s.band_power(1.0, 40.0))
            .sum::<f32>() / epoch.samples.len() as f32;
        tokens.push(quantize_power(total_power));
        
        tokens
    }
    
    /// Encode sample to tokens
    pub fn encode_sample(&self, sample: &EegSample) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        for (i, &v) in sample.channels.iter().enumerate() {
            if v.abs() > self.noise_floor_uv {
                tokens.push(i as u32);
                tokens.push(quantize_voltage(v));
            }
        }
        
        // Timestamp bucket
        tokens.push(((sample.timestamp_ms / 1000) % 60) as u32);
        
        tokens
    }
}

fn quantize_voltage(uv: f32) -> u32 {
    // Microvolts to quantized bucket
    let abs = uv.abs();
    if abs < 1.0 { 0 }
    else if abs < 5.0 { 1 }
    else if abs < 10.0 { 2 }
    else if abs < 20.0 { 3 }
    else if abs < 50.0 { 4 }
    else { 5 }
}

fn quantize_power(power: f32) -> u32 {
    if power < 1.0 { 0 }
    else if power < 5.0 { 1 }
    else if power < 10.0 { 2 }
    else if power < 20.0 { 3 }
    else { 4 }
}

// ============================================================================
// Neural Action Bridge
// ============================================================================

/// Neural control action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuralAction {
    /// Joint angle (prosthetic control)
    JointAngle { joint: u8, angle_deg: f32 },
    /// Grip force
    GripForce { force_n: f32 },
    /// Cursor movement
    CursorMove { dx: f32, dy: f32 },
    /// Click
    Click,
    /// Rest state
    Rest,
}

/// Neural action bridge - maps neural states to control actions
pub struct NeuralActionBridge {
    pub model_hidden_dim: usize,
    pub num_actions: usize,
}

impl NeuralActionBridge {
    pub fn new(model_hidden_dim: usize) -> Self {
        Self {
            model_hidden_dim,
            num_actions: 6,  // 5 actions + rest
        }
    }
    
    /// Predict action from neural embeddings
    pub fn predict(&self, embeddings: &[f32]) -> NeuralAction {
        if embeddings.is_empty() {
            return NeuralAction::Rest;
        }
        
        // Simplified: classify based on embedding statistics
        let mean: f32 = embeddings.iter().sum::<f32>() / embeddings.len() as f32;
        let variance = embeddings.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / embeddings.len() as f32;
        
        // Motor imagery classification (simplified)
        if variance < 0.1 {
            NeuralAction::Rest
        } else if variance < 0.5 {
            NeuralAction::CursorMove { dx: mean, dy: mean }
        } else if variance < 1.0 {
            NeuralAction::GripForce { force_n: variance * 10.0 }
        } else {
            NeuralAction::Click
        }
    }
    
    /// Map to joint angles for prosthetic
    pub fn to_joint_angles(&self, action: &NeuralAction) -> Vec<f32> {
        match action {
            NeuralAction::JointAngle { joint, angle_deg } => {
                let mut angles = vec![0.0; 5];
                if (*joint as usize) < 5 {
                    angles[*joint as usize] = *angle_deg;
                }
                angles
            }
            NeuralAction::GripForce { force_n } => {
                // Map force to grip angles
                vec![*force_n, *force_n, *force_n, *force_n, 0.0]
            }
            _ => vec![0.0; 5],
        }
    }
}

// ============================================================================
// MNE Reference (for reference processing)
// ============================================================================

/// MNE-style preprocessing pipeline reference
pub struct MnePreprocessor {
    pub low_freq_hz: f32,
    pub high_freq_hz: f32,
    pub notch_freq_hz: Option<f32>,
}

impl MnePreprocessor {
    pub fn new() -> Self {
        Self {
            low_freq_hz: 1.0,
            high_freq_hz: 40.0,
            notch_freq_hz: Some(50.0),
        }
    }
    
    /// Apply bandpass filter (reference)
    pub fn bandpass_filter(&self, _sample: &mut EegSample) {
        // In real impl: apply IIR/FIR filter
        // Simplified: just normalize
        for v in &mut _sample.channels {
            *v = v.max(-500.0).min(500.0);
        }
    }
    
    /// Apply notch filter for line noise
    pub fn notch_filter(&self, _sample: &mut EegSample) {
        if self.notch_freq_hz.is_some() {
            // Simplified: would apply notch filter at 50/60 Hz
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
    fn test_eeg_sample() {
        let sample = EegSample::new(32, 256);
        assert_eq!(sample.channels.len(), 32);
    }
    
    #[test]
    fn test_eeg_encoder() {
        let encoder = EegStreamEncoder::new(32, 256);
        let sample = EegSample::new(32, 256);
        sample.set_channel(0, 10.0);
        
        let tokens = encoder.encode_sample(&sample);
        assert!(!tokens.is_empty());
    }
    
    #[test]
    fn test_neural_action_bridge() {
        let bridge = NeuralActionBridge::new(128);
        
        let embeddings = vec![0.5, 1.0, 1.5];
        let action = bridge.predict(&embeddings);
        
        assert!(matches!(action, NeuralAction::Click | NeuralAction::GripForce { .. }));
    }
    
    #[test]
    fn test_joint_angles() {
        let bridge = NeuralActionBridge::new(128);
        
        let action = NeuralAction::JointAngle { joint: 2, angle_deg: 45.0 };
        let angles = bridge.to_joint_angles(&action);
        
        assert_eq!(angles.len(), 5);
    }
    
    #[test]
    fn test_mne_preprocessor() {
        let mne = MnePreprocessor::new();
        let mut sample = EegSample::new(8, 256);
        sample.set_channel(0, 100.0);
        
        mne.bandpass_filter(&mut sample);
        assert!(sample.channels[0] <= 500.0);
    }
}