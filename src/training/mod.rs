pub mod optimizer;

pub use optimizer::{SgdOptimizer, AdamOptimizer, CrossEntropyLoss};
use std::fmt;

pub struct TrainingState {
    pub epoch: u32,
    pub step: u32,
    pub total_loss: f32,
    pub best_loss: f32,
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            epoch: 0,
            step: 0,
            total_loss: 0.0,
            best_loss: f32::INFINITY,
        }
    }

    pub fn record_loss(&mut self, loss: f32) {
        self.total_loss += loss;
        if loss < self.best_loss {
            self.best_loss = loss;
        }
        tracing::debug!("TrainingState: recorded loss={:.6} best_loss={:.6}", loss, self.best_loss);
    }

    pub fn next_step(&mut self) {
        self.step += 1;
        tracing::debug!("TrainingState: step -> {}", self.step);
    }

    pub fn next_epoch(&mut self) {
        self.epoch += 1;
        self.step = 0;
        self.total_loss = 0.0;
        tracing::info!("TrainingState: epoch -> {}", self.epoch);
    }

    pub fn avg_loss(&self) -> f32 {
        if self.step == 0 {
            return 0.0;
        }
        self.total_loss / self.step as f32
    }

    pub fn summary(&self) -> String {
        format!(
            "epoch={} step={} avg_loss={:.6} best_loss={:.6}",
            self.epoch,
            self.step,
            self.avg_loss(),
            self.best_loss,
        )
    }
}

impl fmt::Display for TrainingState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TrainingConfig {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
    pub gradient_clip_norm: f32,
    pub log_every_n_steps: u32,
    pub checkpoint_every_n_epochs: u32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            learning_rate: 1e-3,
            gradient_clip_norm: 1.0,
            log_every_n_steps: 10,
            checkpoint_every_n_epochs: 1,
        }
    }
}

impl TrainingConfig {
    pub fn new(epochs: u32, batch_size: u32, learning_rate: f32) -> Self {
        Self {
            epochs,
            batch_size,
            learning_rate,
            ..Default::default()
        }
    }
}
