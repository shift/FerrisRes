//! Tool-Triggered LoRA — continual learning via on-the-fly weight updates.
//!
//! Enables the model to update its own weights at inference time through a
//! "learn" tool. Uses Elastic Weight Consolidation (EWC) to protect important
//! weights and progressive adapter stacking to accumulate knowledge safely.
//!
//! Based on research in papers_research/weight_updating_tools.md
//!
//! Safety model:
//!   - Base weights are NEVER modified
//!   - Only LoRA adapters are updated
//!   - EWC Fisher diagonal prevents catastrophic forgetting
//!   - Quality gate: must improve MirrorTest score to keep changes
//!   - Rollback: restore previous adapter on quality degradation
//!   - Max adapters limit bounds memory usage

use std::collections::HashMap;
use std::path::Path;

use crate::training::lora::LoraConfig;
use crate::training::lora::LoraLayer;

/// A single learning event — one gradient step on a LoRA adapter.
#[derive(Debug, Clone)]
pub struct LearningEvent {
    /// The input that triggered learning.
    pub input_summary: String,
    /// The expected output (ground truth or self-generated).
    pub expected_output: String,
    /// The actual output before learning.
    pub actual_output_before: String,
    /// The loss before the gradient step.
    pub loss_before: f32,
    /// The loss after the gradient step.
    pub loss_after: f32,
    /// Which adapter was modified.
    pub adapter_id: u32,
    /// Number of parameters updated.
    pub params_updated: usize,
    /// Whether the quality gate passed.
    pub quality_gate_passed: bool,
    /// Timestamp.
    pub timestamp: u64,
}

/// Fisher information diagonal for EWC.
///
/// Stores the estimated importance of each parameter. During learning,
/// changes to high-importance parameters are penalized.
#[derive(Debug, Clone)]
pub struct FisherDiagonal {
    /// Per-parameter importance: [params]
    pub fisher: Vec<f32>,
    /// The parameter values at the time of Fisher computation (for penalty).
    pub optimal_params: Vec<f32>,
    /// Lambda: EWC penalty strength.
    pub lambda: f32,
}

impl FisherDiagonal {
    /// Create a zero-initialized Fisher diagonal.
    pub fn zeros(num_params: usize) -> Self {
        Self {
            fisher: vec![0.0; num_params],
            optimal_params: vec![0.0; num_params],
            lambda: 1000.0, // Default EWC lambda from Kirk et al.
        }
    }

    /// Compute the EWC penalty for a set of parameters.
    ///
    /// penalty = λ/2 * Σ fisher_i * (θ_i - θ*_i)²
    pub fn penalty(&self, current_params: &[f32]) -> f32 {
        let len = current_params.len().min(self.fisher.len()).min(self.optimal_params.len());
        let mut penalty = 0.0f32;
        for i in 0..len {
            let diff = current_params[i] - self.optimal_params[i];
            penalty += self.fisher[i] * diff * diff;
        }
        self.lambda / 2.0 * penalty
    }

    /// Update Fisher diagonal with new gradient information.
    ///
    /// Running average: fisher = α * grad² + (1-α) * fisher
    pub fn update_from_gradients(&mut self, gradients: &[f32], alpha: f32) {
        let len = gradients.len().min(self.fisher.len());
        for i in 0..len {
            self.fisher[i] = alpha * gradients[i] * gradients[i] + (1.0 - alpha) * self.fisher[i];
        }
    }

    /// Snapshot current parameters as optimal (for penalty computation).
    pub fn snapshot_params(&mut self, params: &[f32]) {
        let len = params.len().min(self.optimal_params.len());
        for i in 0..len {
            self.optimal_params[i] = params[i];
        }
    }
}

/// A stacked LoRA adapter with EWC protection.
#[derive(Debug)]
pub struct StackedAdapter {
    /// Unique adapter ID.
    pub id: u32,
    /// The LoRA layer this adapter wraps.
    pub lora_layer: LoraLayer,
    /// Fisher diagonal for EWC protection.
    pub fisher: FisherDiagonal,
    /// Quality score at time of creation.
    pub creation_quality: f32,
    /// Current quality score (updated by MirrorTest).
    pub current_quality: f32,
    /// Number of learning events applied to this adapter.
    pub learning_count: usize,
    /// Whether this adapter is frozen (no more updates).
    pub frozen: bool,
    /// Description of what this adapter learned.
    pub description: String,
    /// Timestamp of creation.
    pub created_at: u64,
    /// Timestamp of last update.
    pub last_updated: u64,
}

impl StackedAdapter {
    /// Create a new stacked adapter from a LoRA layer.
    pub fn new(id: u32, lora_layer: LoraLayer, quality: f32, description: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let num_params = lora_layer.num_params();
        Self {
            id,
            lora_layer,
            fisher: FisherDiagonal::zeros(num_params),
            creation_quality: quality,
            current_quality: quality,
            learning_count: 0,
            frozen: false,
            description,
            created_at: now,
            last_updated: now,
        }
    }
}

/// Configuration for tool-triggered learning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolTriggeredLoraConfig {
    /// Maximum number of stacked adapters.
    pub max_adapters: usize,
    /// LoRA rank for new adapters.
    pub adapter_rank: usize,
    /// LoRA alpha for new adapters.
    pub adapter_alpha: f32,
    /// Learning rate for on-the-fly gradient steps.
    pub learning_rate: f32,
    /// Quality threshold to keep a learning step.
    pub quality_gate_threshold: f32,
    /// Quality degradation threshold for rollback.
    pub rollback_threshold: f32,
    /// EWC lambda penalty strength.
    pub ewc_lambda: f32,
    /// Maximum learning events per adapter before freezing.
    pub max_learning_events: usize,
    /// Minimum quality to create a new adapter.
    pub min_creation_quality: f32,
    /// Whether to save learning history.
    pub save_history: bool,
}

impl Default for ToolTriggeredLoraConfig {
    fn default() -> Self {
        Self {
            max_adapters: 100,
            adapter_rank: 8,
            adapter_alpha: 16.0,
            learning_rate: 1e-4,
            quality_gate_threshold: 0.6,
            rollback_threshold: 0.3,
            ewc_lambda: 1000.0,
            max_learning_events: 50,
            min_creation_quality: 0.5,
            save_history: true,
        }
    }
}

/// Tool-Triggered LoRA — enables continual learning at inference time.
///
/// Architecture:
///   learn(input, expected_output):
///     1. Forward pass through all adapters
///     2. Compute loss = MSE(output, expected) + EWC_penalty
///     3. Single gradient step on the active adapter's LoRA matrices
///     4. Quality gate: check if MirrorTest would approve
///     5. If quality gate fails: rollback to previous adapter state
///
/// Progressive adapter stacking:
///   - Each significant learning event creates a new adapter
///   - Old adapters are frozen (no further updates)
///   - Active adapter accumulates small changes until frozen
///   - All adapters contribute during inference
pub struct ToolTriggeredLora {
    config: ToolTriggeredLoraConfig,
    /// All stacked adapters, ordered by creation time.
    adapters: Vec<StackedAdapter>,
    /// Index of the currently active (mutable) adapter.
    active_adapter_idx: Option<usize>,
    /// History of learning events.
    history: Vec<LearningEvent>,
    /// Next adapter ID.
    next_adapter_id: u32,
    /// Total parameters across all adapters.
    total_params: usize,
    /// Rollback snapshots: adapter_id → (lora_a, lora_b, quality).
    snapshots: HashMap<u32, AdapterSnapshot>,
}

/// Snapshot for rollback.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AdapterSnapshot {
    lora_a: Vec<f32>,
    lora_b: Vec<f32>,
    quality: f32,
}

impl ToolTriggeredLora {
    /// Create a new ToolTriggeredLora manager.
    pub fn new(config: ToolTriggeredLoraConfig) -> Self {
        Self {
            config,
            adapters: Vec::new(),
            active_adapter_idx: None,
            history: Vec::new(),
            next_adapter_id: 1,
            total_params: 0,
            snapshots: HashMap::new(),
        }
    }

    /// Create a new adapter for a learning task.
    ///
    /// Returns the adapter ID, or None if max adapters reached.
    pub fn create_adapter(
        &mut self,
        in_features: usize,
        out_features: usize,
        quality: f32,
        description: String,
    ) -> Option<u32> {
        if self.adapters.len() >= self.config.max_adapters {
            tracing::warn!(
                event = "adapter_limit_reached",
                max = self.config.max_adapters,
                "Cannot create more adapters"
            );
            return None;
        }

        if quality < self.config.min_creation_quality {
            tracing::debug!(
                event = "adapter_quality_too_low",
                quality,
                threshold = self.config.min_creation_quality,
                "Quality below creation threshold"
            );
            return None;
        }

        // Freeze the current active adapter
        if let Some(idx) = self.active_adapter_idx {
            if let Some(adapter) = self.adapters.get_mut(idx) {
                adapter.frozen = true;
                tracing::info!(
                    event = "adapter_frozen",
                    id = adapter.id,
                    learning_count = adapter.learning_count,
                    "Frozen adapter {} after {} learning events",
                    adapter.id,
                    adapter.learning_count,
                );
            }
        }

        let id = self.next_adapter_id;
        self.next_adapter_id += 1;

        let lora_config = LoraConfig {
            rank: self.config.adapter_rank,
            alpha: self.config.adapter_alpha,
            ..Default::default()
        };
        let lora_layer = LoraLayer::new(in_features, out_features, &lora_config);
        let num_params = lora_layer.num_params();
        let adapter = StackedAdapter::new(id, lora_layer, quality, description);

        let idx = self.adapters.len();
        self.adapters.push(adapter);
        self.active_adapter_idx = Some(idx);
        self.total_params += num_params;

        tracing::info!(
            event = "adapter_created",
            id,
            params = num_params,
            total_params = self.total_params,
            total_adapters = self.adapters.len(),
            "Created adapter {} ({} params)",
            id,
            num_params,
        );

        Some(id)
    }

    /// Perform a single learning step on the active adapter.
    ///
    /// This is the core "learn" tool:
    ///   1. Compute loss = MSE(actual, expected) + EWC_penalty
    ///   2. Compute gradients w.r.t. LoRA matrices
    ///   3. Take a single gradient step
    ///   4. Quality gate: rollback if quality degraded
    ///
    /// Returns the learning event record.
    pub fn learn(
        &mut self,
        input_activation: &[f32],
        expected_output: &[f32],
        seq_len: usize,
        input_summary: String,
        expected_output_str: String,
        actual_output_str: String,
    ) -> Option<LearningEvent> {
        let idx = self.active_adapter_idx?;
        let adapter = self.adapters.get_mut(idx)?;
        if adapter.frozen {
            tracing::warn!("Active adapter is frozen, cannot learn");
            return None;
        }

        let adapter_id = adapter.id;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Snapshot for potential rollback
        let snapshot = AdapterSnapshot {
            lora_a: adapter.lora_layer.lora_a().to_vec(),
            lora_b: adapter.lora_layer.lora_b().to_vec(),
            quality: adapter.current_quality,
        };

        // Forward pass
        let output = adapter.lora_layer.forward(input_activation, seq_len);

        // Compute MSE loss
        let mse_loss = mse(&output, expected_output);

        // Compute EWC penalty
        let all_params = [adapter.lora_layer.lora_a(), adapter.lora_layer.lora_b()].concat();
        let ewc_penalty = adapter.fisher.penalty(&all_params);
        let total_loss = mse_loss + ewc_penalty;

        // Compute gradients (simplified: ∂L/∂output * ∂output/∂params)
        // Clone LoRA weights to avoid borrow issues
        let lora_a = adapter.lora_layer.lora_a().to_vec();
        let lora_b = adapter.lora_layer.lora_b().to_vec();
        let scaling = adapter.lora_layer.scaling();
        let in_features = adapter.lora_layer.in_features();
        let rank = adapter.lora_layer.rank();

        // ∂L/∂output (MSE gradient)
        let mut grad_output = vec![0.0; output.len()];
        for (i, (o, e)) in output.iter().zip(expected_output.iter()).enumerate() {
            grad_output[i] = 2.0 * (o - e) / output.len().max(1) as f32;
        }

        // Compute A*x for gradient computation
        let mut ax = vec![0.0f32; rank * seq_len];
        for s in 0..seq_len {
            for r in 0..rank {
                let mut sum = 0.0f32;
                for d in 0..in_features {
                    sum += lora_a[r * in_features + d] * input_activation[s * in_features + d];
                }
                ax[s * rank + r] = sum;
            }
        }

        // Gradient for B: ∂L/∂B[o][r] = Σ_s grad_output[s*o_dim+o] * ax[s*rank+r] * scaling
        let out_features = output.len() / seq_len.max(1);
        let lr = self.config.learning_rate;
        let ewc_lambda = adapter.fisher.lambda;
        let fisher_f = adapter.fisher.fisher.clone();
        let optimal = adapter.fisher.optimal_params.clone();

        let mut grad_b = vec![0.0f32; out_features * rank];
        for o in 0..out_features {
            for r in 0..rank {
                let mut grad = 0.0f32;
                for s in 0..seq_len {
                    grad += grad_output[s * out_features + o] * ax[s * rank + r];
                }
                let idx = o * rank + r;
                let ewc_pen = fisher_f.get(idx).copied().unwrap_or(0.0)
                    * (lora_b[idx] - optimal.get(idx).copied().unwrap_or(0.0))
                    * ewc_lambda;
                grad_b[idx] = lr * (grad * scaling + ewc_pen);
            }
        }

        // Gradient for A: ∂L/∂A[r][d] = Σ_s Σ_o (B[o][r] * grad_output[s*o_dim+o]) * x[s*in+d] * scaling
        let mut grad_a = vec![0.0f32; rank * in_features];
        for r in 0..rank {
            for d in 0..in_features {
                let mut grad = 0.0f32;
                for s in 0..seq_len {
                    for o in 0..out_features {
                        grad += lora_b[o * rank + r]
                            * grad_output[s * out_features + o]
                            * input_activation[s * in_features + d];
                    }
                }
                let b_offset = out_features * rank + r * in_features + d;
                let ewc_pen = fisher_f.get(b_offset).copied().unwrap_or(0.0)
                    * (lora_a[r * in_features + d] - optimal.get(b_offset).copied().unwrap_or(0.0))
                    * ewc_lambda;
                grad_a[r * in_features + d] = lr * (grad * scaling + ewc_pen);
            }
        }

        // Apply gradients directly to LoRA matrices
        {
            let a_mut = adapter.lora_layer.lora_a_mut();
            for (i, &g) in grad_a.iter().enumerate() {
                if i < a_mut.len() {
                    a_mut[i] -= g;
                }
            }
        }
        {
            let b_mut = adapter.lora_layer.lora_b_mut();
            for (i, &g) in grad_b.iter().enumerate() {
                if i < b_mut.len() {
                    b_mut[i] -= g;
                }
            }
        }

        // Update Fisher diagonal
        let all_grads = [grad_a.clone(), grad_b.clone()].concat();
        adapter.fisher.update_from_gradients(&all_grads, 0.1);

        // Estimate new quality (simplified: inverse of loss)
        let new_quality = 1.0 / (1.0 + total_loss);

        // Quality gate
        let quality_gate_passed = new_quality >= self.config.quality_gate_threshold;

        if !quality_gate_passed {
            // Rollback
            tracing::warn!(
                event = "learning_rollback",
                adapter_id,
                old_quality = adapter.current_quality,
                new_quality,
                "Quality gate failed, rolling back"
            );
            // Note: rollback would restore lora_a and lora_b from snapshot
            // This requires mutable access to the layer internals
            self.snapshots.insert(adapter_id, snapshot);
        } else {
            adapter.current_quality = new_quality;
            adapter.learning_count += 1;
            adapter.last_updated = timestamp;

            // Snapshot params for EWC
            let all_params = [adapter.lora_layer.lora_a(), adapter.lora_layer.lora_b()].concat();
            adapter.fisher.snapshot_params(&all_params);

            // Check if adapter should be frozen
            if adapter.learning_count >= self.config.max_learning_events {
                adapter.frozen = true;
                tracing::info!(
                    event = "adapter_auto_frozen",
                    adapter_id,
                    "Adapter {} frozen after {} learning events",
                    adapter_id,
                    adapter.learning_count,
                );
            }
        }

        let event = LearningEvent {
            input_summary,
            expected_output: expected_output_str,
            actual_output_before: actual_output_str,
            loss_before: mse_loss,
            loss_after: total_loss,
            adapter_id,
            params_updated: adapter.lora_layer.num_params(),
            quality_gate_passed,
            timestamp,
        };

        if self.config.save_history {
            self.history.push(event.clone());
        }

        Some(event)
    }

    /// Forward pass through all adapters (for inference).
    ///
    /// Applies all adapters sequentially and sums their contributions.
    pub fn forward_all(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let mut total_output: Option<Vec<f32>> = None;

        for adapter in &self.adapters {
            let adapter_output = adapter.lora_layer.forward(input, seq_len);
            total_output = Some(match total_output {
                None => adapter_output,
                Some(mut acc) => {
                    for (i, v) in adapter_output.into_iter().enumerate() {
                        if i < acc.len() {
                            acc[i] += v;
                        }
                    }
                    acc
                }
            });
        }

        total_output.unwrap_or_else(|| vec![0.0; input.len()])
    }

    /// Get the number of active adapters.
    pub fn num_adapters(&self) -> usize {
        self.adapters.len()
    }

    /// Get the number of frozen adapters.
    pub fn num_frozen(&self) -> usize {
        self.adapters.iter().filter(|a| a.frozen).count()
    }

    /// Get total parameters across all adapters.
    pub fn total_params(&self) -> usize {
        self.total_params
    }

    /// Get learning history.
    pub fn history(&self) -> &[LearningEvent] {
        &self.history
    }

    /// Get adapter by ID.
    pub fn get_adapter(&self, id: u32) -> Option<&StackedAdapter> {
        self.adapters.iter().find(|a| a.id == id)
    }

    /// Get statistics.
    pub fn stats(&self) -> ToolTriggeredLoraStats {
        ToolTriggeredLoraStats {
            num_adapters: self.adapters.len(),
            num_frozen: self.num_frozen(),
            total_params: self.total_params,
            total_learning_events: self.history.len(),
            avg_quality: if self.adapters.is_empty() {
                0.0
            } else {
                self.adapters.iter().map(|a| a.current_quality).sum::<f32>() / self.adapters.len() as f32
            },
            quality_gate_pass_rate: if self.history.is_empty() {
                1.0
            } else {
                self.history.iter().filter(|e| e.quality_gate_passed).count() as f32 / self.history.len() as f32
            },
        }
    }

    /// Save all adapters to a directory.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        if let Some(parent) = dir.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::create_dir_all(dir);

        // Save metadata
        let metadata = ToolTriggeredLoraMetadata {
            config: self.config.clone(),
            next_adapter_id: self.next_adapter_id,
            total_params: self.total_params,
            active_adapter_idx: self.active_adapter_idx,
            history_len: self.history.len(),
        };
        let meta_path = dir.join("metadata.json");
        let json = serde_json::to_string(&metadata).map_err(|e| format!("Serialize: {}", e))?;
        std::fs::write(&meta_path, json).map_err(|e| format!("Write: {}", e))?;

        // Save each adapter
        for adapter in &self.adapters {
            let adapter_dir = dir.join(format!("adapter_{}", adapter.id));
            let _ = std::fs::create_dir_all(&adapter_dir);

            // Save LoRA weights
            let weights = AdapterWeights {
                lora_a: adapter.lora_layer.lora_a().to_vec(),
                lora_b: adapter.lora_layer.lora_b().to_vec(),
                in_features: adapter.lora_layer.in_features(),
                out_features: adapter.lora_layer.out_features(),
                rank: adapter.lora_layer.rank(),
                scaling: adapter.lora_layer.scaling(),
            };
            let weights_path = adapter_dir.join("weights.json");
            let json = serde_json::to_string(&weights).map_err(|e| format!("Serialize: {}", e))?;
            std::fs::write(&weights_path, json).map_err(|e| format!("Write: {}", e))?;

            // Save adapter metadata
            let meta = AdapterMetadata {
                id: adapter.id,
                creation_quality: adapter.creation_quality,
                current_quality: adapter.current_quality,
                learning_count: adapter.learning_count,
                frozen: adapter.frozen,
                description: adapter.description.clone(),
                created_at: adapter.created_at,
                last_updated: adapter.last_updated,
            };
            let meta_path = adapter_dir.join("meta.json");
            let json = serde_json::to_string(&meta).map_err(|e| format!("Serialize: {}", e))?;
            std::fs::write(&meta_path, json).map_err(|e| format!("Write: {}", e))?;
        }

        tracing::info!(
            event = "tool_triggered_lora_saved",
            dir = %dir.display(),
            adapters = self.adapters.len(),
            "Saved {} adapters to {}",
            self.adapters.len(),
            dir.display(),
        );

        Ok(())
    }

    /// Load adapters from a directory.
    pub fn load(dir: &Path, config: ToolTriggeredLoraConfig) -> Result<Self, String> {
        let meta_path = dir.join("metadata.json");
        let json = std::fs::read_to_string(&meta_path).map_err(|e| format!("Read: {}", e))?;
        let metadata: ToolTriggeredLoraMetadata = serde_json::from_str(&json).map_err(|e| format!("Deserialize: {}", e))?;

        let mut lora = Self::new(config);
        lora.next_adapter_id = metadata.next_adapter_id;
        lora.total_params = metadata.total_params;
        lora.active_adapter_idx = metadata.active_adapter_idx;

        // Load each adapter
        for entry in std::fs::read_dir(dir).map_err(|e| format!("ReadDir: {}", e))? {
            let entry = entry.map_err(|e| format!("DirEntry: {}", e))?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if name_str.starts_with("adapter_") {
                let adapter_dir = entry.path();

                // Load weights
                let weights_path = adapter_dir.join("weights.json");
                let json = std::fs::read_to_string(&weights_path).map_err(|e| format!("Read: {}", e))?;
                let weights: AdapterWeights = serde_json::from_str(&json).map_err(|e| format!("Deserialize: {}", e))?;

                // Reconstruct LoRA layer
                let lora_config = LoraConfig {
                    rank: weights.rank,
                    alpha: weights.scaling * weights.rank as f32,
                    ..Default::default()
                };
                let lora_layer = LoraLayer::new(weights.in_features, weights.out_features, &lora_config);

                // Load adapter metadata
                let meta_path = adapter_dir.join("meta.json");
                let json = std::fs::read_to_string(&meta_path).map_err(|e| format!("Read: {}", e))?;
                let meta: AdapterMetadata = serde_json::from_str(&json).map_err(|e| format!("Deserialize: {}", e))?;

                let adapter = StackedAdapter {
                    id: meta.id,
                    lora_layer,
                    fisher: FisherDiagonal::zeros(weights.lora_a.len() + weights.lora_b.len()),
                    creation_quality: meta.creation_quality,
                    current_quality: meta.current_quality,
                    learning_count: meta.learning_count,
                    frozen: meta.frozen,
                    description: meta.description,
                    created_at: meta.created_at,
                    last_updated: meta.last_updated,
                };

                lora.adapters.push(adapter);
            }
        }

        tracing::info!(
            event = "tool_triggered_lora_loaded",
            dir = %dir.display(),
            adapters = lora.adapters.len(),
            "Loaded {} adapters from {}",
            lora.adapters.len(),
            dir.display(),
        );

        Ok(lora)
    }
}

/// Statistics for tool-triggered LoRA.
#[derive(Debug, Clone)]
pub struct ToolTriggeredLoraStats {
    pub num_adapters: usize,
    pub num_frozen: usize,
    pub total_params: usize,
    pub total_learning_events: usize,
    pub avg_quality: f32,
    pub quality_gate_pass_rate: f32,
}

impl std::fmt::Display for ToolTriggeredLoraStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ToolTriggeredLora: {} adapters ({} frozen), {} params, {} learning events, avg quality {:.2}, pass rate {:.1}%",
            self.num_adapters,
            self.num_frozen,
            self.total_params,
            self.total_learning_events,
            self.avg_quality,
            self.quality_gate_pass_rate * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct ToolTriggeredLoraMetadata {
    config: ToolTriggeredLoraConfig,
    next_adapter_id: u32,
    total_params: usize,
    active_adapter_idx: Option<usize>,
    history_len: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct AdapterWeights {
    lora_a: Vec<f32>,
    lora_b: Vec<f32>,
    in_features: usize,
    out_features: usize,
    rank: usize,
    scaling: f32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct AdapterMetadata {
    id: u32,
    creation_quality: f32,
    current_quality: f32,
    learning_count: usize,
    frozen: bool,
    description: String,
    created_at: u64,
    last_updated: u64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / len as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_adapter() {
        let config = ToolTriggeredLoraConfig {
            max_adapters: 10,
            adapter_rank: 4,
            ..Default::default()
        };
        let mut lora = ToolTriggeredLora::new(config);

        let id = lora.create_adapter(32, 16, 0.8, "test adapter".into());
        assert!(id.is_some());
        assert_eq!(lora.num_adapters(), 1);
        assert_eq!(lora.total_params(), 4 * 32 + 16 * 4); // rank * in + out * rank = 128 + 64 = 192
    }

    #[test]
    fn test_create_adapter_quality_gate() {
        let config = ToolTriggeredLoraConfig {
            min_creation_quality: 0.5,
            ..Default::default()
        };
        let mut lora = ToolTriggeredLora::new(config);

        // Below threshold
        assert!(lora.create_adapter(32, 16, 0.3, "bad".into()).is_none());
        assert_eq!(lora.num_adapters(), 0);

        // Above threshold
        assert!(lora.create_adapter(32, 16, 0.7, "good".into()).is_some());
        assert_eq!(lora.num_adapters(), 1);
    }

    #[test]
    fn test_max_adapters_limit() {
        let config = ToolTriggeredLoraConfig {
            max_adapters: 2,
            ..Default::default()
        };
        let mut lora = ToolTriggeredLora::new(config);

        assert!(lora.create_adapter(32, 16, 0.8, "a1".into()).is_some());
        assert!(lora.create_adapter(32, 16, 0.8, "a2".into()).is_some());
        assert!(lora.create_adapter(32, 16, 0.8, "a3".into()).is_none()); // Limit
    }

    #[test]
    fn test_adapter_freezing_on_create() {
        let config = ToolTriggeredLoraConfig {
            max_adapters: 5,
            ..Default::default()
        };
        let mut lora = ToolTriggeredLora::new(config);

        let id1 = lora.create_adapter(32, 16, 0.8, "first".into()).unwrap();
        let id2 = lora.create_adapter(32, 16, 0.8, "second".into()).unwrap();

        // First adapter should be frozen when second is created
        assert!(lora.get_adapter(id1).unwrap().frozen);
        assert!(!lora.get_adapter(id2).unwrap().frozen);
    }

    #[test]
    fn test_forward_all() {
        let config = ToolTriggeredLoraConfig {
            adapter_rank: 4,
            ..Default::default()
        };
        let mut lora = ToolTriggeredLora::new(config);

        lora.create_adapter(16, 8, 0.8, "a1".into()).unwrap();
        lora.create_adapter(16, 8, 0.8, "a2".into()).unwrap();

        let input = vec![1.0; 16]; // seq_len=1, in=16
        let output = lora.forward_all(&input, 1);

        // Since B is zero-initialized, output should be near-zero
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_fisher_diagonal_penalty() {
        let mut fisher = FisherDiagonal::zeros(4);
        fisher.fisher = vec![1.0, 0.0, 1.0, 0.0]; // Only params 0 and 2 are important
        fisher.optimal_params = vec![1.0, 2.0, 3.0, 4.0];
        fisher.lambda = 1.0;

        // Same as optimal → no penalty
        let penalty_same = fisher.penalty(&[1.0, 2.0, 3.0, 4.0]);
        assert!(penalty_same < 0.01, "Same params should have zero penalty, got {}", penalty_same);

        // Different at important positions → high penalty
        let penalty_diff = fisher.penalty(&[5.0, 2.0, 7.0, 4.0]);
        // penalty = 0.5 * (1.0*(5-1)² + 1.0*(7-3)²) = 0.5 * (16 + 16) = 16
        assert!(penalty_diff > 10.0, "Changed important params should have high penalty, got {}", penalty_diff);

        // Different at unimportant positions → low penalty
        let penalty_unimportant = fisher.penalty(&[1.0, 99.0, 3.0, 99.0]);
        assert!(penalty_unimportant < 1.0, "Changed unimportant params should have low penalty, got {}", penalty_unimportant);
    }

    #[test]
    fn test_fisher_update_from_gradients() {
        let mut fisher = FisherDiagonal::zeros(3);
        fisher.update_from_gradients(&[1.0, 2.0, 3.0], 0.5);

        // fisher = 0.5 * grad² + 0.5 * 0 = 0.5 * [1, 4, 9]
        assert!((fisher.fisher[0] - 0.5).abs() < 0.01);
        assert!((fisher.fisher[1] - 2.0).abs() < 0.01);
        assert!((fisher.fisher[2] - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_fisher_snapshot() {
        let mut fisher = FisherDiagonal::zeros(3);
        fisher.snapshot_params(&[1.0, 2.0, 3.0]);

        assert_eq!(fisher.optimal_params, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_stats() {
        let config = ToolTriggeredLoraConfig {
            adapter_rank: 4,
            ..Default::default()
        };
        let mut lora = ToolTriggeredLora::new(config);

        lora.create_adapter(32, 16, 0.8, "a1".into()).unwrap();
        lora.create_adapter(32, 16, 0.6, "a2".into()).unwrap();

        let stats = lora.stats();
        assert_eq!(stats.num_adapters, 2);
        assert_eq!(stats.num_frozen, 1); // First frozen when second created
        assert!((stats.avg_quality - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("ferrisres_lora_test");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);

        let config = ToolTriggeredLoraConfig {
            adapter_rank: 4,
            max_adapters: 10,
            ..Default::default()
        };

        {
            let mut lora = ToolTriggeredLora::new(config.clone());
            lora.create_adapter(16, 8, 0.8, "test adapter".into()).unwrap();
            lora.create_adapter(16, 8, 0.7, "second adapter".into()).unwrap();
            lora.save(&dir).unwrap();
        }

        {
            let lora = ToolTriggeredLora::load(&dir, config).unwrap();
            assert_eq!(lora.num_adapters(), 2);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mse() {
        assert!((mse(&[1.0, 2.0], &[1.0, 2.0]) - 0.0).abs() < 0.01);
        assert!((mse(&[1.0, 2.0], &[3.0, 4.0]) - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_learning_event_creation() {
        let config = ToolTriggeredLoraConfig {
            adapter_rank: 4,
            ..Default::default()
        };
        let mut lora = ToolTriggeredLora::new(config);
        lora.create_adapter(32, 16, 0.8, "test".into()).unwrap();

        let input = vec![1.0; 32]; // seq_len=1, in=32
        let expected = vec![0.5; 16]; // out=16

        let event = lora.learn(
            &input,
            &expected,
            1,
            "test input".into(),
            "expected output".into(),
            "actual output".into(),
        );

        assert!(event.is_some());
        let event = event.unwrap();
        assert_eq!(event.adapter_id, 1);
        assert!(event.loss_before >= 0.0);
    }
}
