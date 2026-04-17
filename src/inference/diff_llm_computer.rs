//! Differentiable LLM-Computer — Gumbel-Softmax op selection with STE.
//!
//! Makes the CALM VM differentiable so gradients flow through program execution.
//! This enables the model to learn which CALM operations to compose for a given task.
//!
//! Phase 1 (this module):
//!   - Op selection via Gumbel-Softmax with temperature annealing
//!   - Straight-Through Estimator (STE) for hard selections
//!   - Soft attention-based memory read/write (NTM-style DiffMemoryBank)
//!
//! Based on research in papers_research/differentiable_execution.md

use crate::inference::llm_computer::{CalmInstruction, CalmOp};

/// Number of distinct CALM operations (for softmax output size).
const NUM_CALM_OPS: usize = 14;

/// Logits over CALM operations, emitted by the model alongside normal logits.
/// Instead of discrete op selection, the model emits a probability distribution
/// over ops. During forward, Gumbel-Softmax samples a (soft) selection.
/// During backward, STE passes gradients through the hard selection.
#[derive(Debug, Clone)]
pub struct OpLogits {
    /// Logits for op selection: [num_ops]
    pub op_logits: Vec<f32>,
    /// Logits for register A: [max_registers]
    pub reg_a_logits: Vec<f32>,
    /// Logits for register B: [max_registers]
    pub reg_b_logits: Vec<f32>,
    /// Logits for output register: [max_registers]
    pub reg_out_logits: Vec<f32>,
    /// Logits for immediate value (binned): [num_value_bins]
    pub value_logits: Vec<f32>,
    /// Logits for table selection: [max_tables]
    pub table_logits: Vec<f32>,
    /// Logits for instruction type: [num_instruction_types]
    pub instruction_type_logits: Vec<f32>,
}

/// Number of instruction types: Compute, LookUp, Append, Move, LoadConst, BranchIf, Halt
const NUM_INSTRUCTION_TYPES: usize = 7;

impl OpLogits {
    /// Create zero-initialized logits for a given configuration.
    pub fn zeros(config: &DiffLlmComputerConfig) -> Self {
        Self {
            op_logits: vec![0.0; NUM_CALM_OPS],
            reg_a_logits: vec![0.0; config.num_registers],
            reg_b_logits: vec![0.0; config.num_registers],
            reg_out_logits: vec![0.0; config.num_registers],
            value_logits: vec![0.0; config.num_value_bins],
            table_logits: vec![0.0; config.max_tables],
            instruction_type_logits: vec![0.0; NUM_INSTRUCTION_TYPES],
        }
    }

    /// Total number of logits.
    pub fn total_size(&self) -> usize {
        self.op_logits.len()
            + self.reg_a_logits.len()
            + self.reg_b_logits.len()
            + self.reg_out_logits.len()
            + self.value_logits.len()
            + self.table_logits.len()
            + self.instruction_type_logits.len()
    }
}

/// Configuration for the differentiable LLM-Computer.
#[derive(Debug, Clone)]
pub struct DiffLlmComputerConfig {
    /// Number of registers (also determines reg logit size).
    pub num_registers: usize,
    /// Number of lookup tables.
    pub max_tables: usize,
    /// Number of value bins for immediate constants.
    pub num_value_bins: usize,
    /// Gumbel-Softmax temperature. Higher = softer, lower = harder.
    pub temperature: f32,
    /// Minimum temperature (after annealing).
    pub min_temperature: f32,
    /// Temperature annealing rate (per step).
    pub annealing_rate: f32,
    /// Whether to use hard (STE) or soft selections during forward.
    pub use_ste: bool,
    /// Memory size for the differentiable memory bank.
    pub memory_size: usize,
    /// Memory vector dimension.
    pub memory_dim: usize,
    /// Number of read heads.
    pub num_read_heads: usize,
    /// Maximum program length.
    pub max_program_length: usize,
}

impl Default for DiffLlmComputerConfig {
    fn default() -> Self {
        Self {
            num_registers: 16,
            max_tables: 8,
            num_value_bins: 32, // -16 to +15 binned
            temperature: 1.0,
            min_temperature: 0.1,
            annealing_rate: 0.001,
            use_ste: true,
            memory_size: 128,
            memory_dim: 32,
            num_read_heads: 4,
            max_program_length: 64,
        }
    }
}

/// A differentiable step result — contains both the hard (discrete) selection
/// and the soft probabilities for gradient computation.
#[derive(Debug, Clone)]
pub struct DiffStepResult {
    /// The hard (discrete) instruction selected (for execution).
    pub instruction: CalmInstruction,
    /// Soft probabilities over instruction types (for backprop).
    pub instruction_type_probs: Vec<f32>,
    /// Soft probabilities over ops (for backprop).
    pub op_probs: Vec<f32>,
    /// Soft probabilities over registers (for backprop).
    pub reg_probs: Vec<Vec<f32>>,
    /// Gumbel-Softmax temperature used.
    pub temperature: f32,
    /// Whether this step used STE.
    pub used_ste: bool,
}

/// Differentiable memory bank — NTM-style soft attention read/write.
///
/// Replaces the discrete table_lookup/table_set with continuous operations
/// that support gradient flow.
#[derive(Debug, Clone)]
pub struct DiffMemoryBank {
    /// Memory matrix: [memory_size × memory_dim].
    pub memory: Vec<Vec<f32>>,
    /// Usage weights (for addressing): [memory_size].
    pub usage: Vec<f32>,
    /// Read weightings from last read: [num_read_heads × memory_size].
    pub read_weights: Vec<Vec<f32>>,
    /// Write weighting from last write: [memory_size].
    pub write_weights: Vec<f32>,
    /// Memory size.
    pub memory_size: usize,
    /// Memory dimension.
    pub memory_dim: usize,
}

impl DiffMemoryBank {
    /// Create a new differentiable memory bank.
    pub fn new(memory_size: usize, memory_dim: usize) -> Self {
        // Initialize memory with small random values
        let memory = (0..memory_size)
            .map(|_| (0..memory_dim)
                .map(|_| (rand::random::<f32>() - 0.5) * 0.01)
                .collect())
            .collect();

        Self {
            memory,
            usage: vec![0.0; memory_size],
            read_weights: vec![vec![0.0; memory_size]; 4], // 4 read heads
            write_weights: vec![0.0; memory_size],
            memory_size,
            memory_dim,
        }
    }

    /// Soft read: content-based addressing via attention.
    ///
    /// query: [memory_dim] → returns: [memory_dim] (weighted read vector)
    pub fn read(&mut self, query: &[f32], head: usize) -> Vec<f32> {
        let query_norm = vec_norm(query);
        if query_norm < 1e-8 {
            return vec![0.0; self.memory_dim];
        }

        // Compute attention weights based on cosine similarity
        let mut weights = vec![0.0f32; self.memory_size];
        for (i, row) in self.memory.iter().enumerate() {
            let row_norm = vec_norm(row);
            if row_norm < 1e-8 {
                weights[i] = 0.0;
                continue;
            }
            let dot: f32 = query.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
            weights[i] = (dot / (query_norm * row_norm)).exp(); // Softmax with temperature=1
        }

        // Normalize weights
        let sum: f32 = weights.iter().sum();
        if sum > 1e-8 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }

        // Store read weights
        if head < self.read_weights.len() {
            self.read_weights[head] = weights.clone();
        }

        // Weighted sum of memory rows
        let mut result = vec![0.0f32; self.memory_dim];
        for (i, row) in self.memory.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                result[j] += weights[i] * v;
            }
        }

        result
    }

    /// Soft write: update memory locations based on attention weights.
    ///
    /// key: [memory_dim] — addressing key
    /// value: [memory_dim] — value to write
    /// erase: f32 — erase strength (0 = no erase, 1 = full erase)
    pub fn write(&mut self, key: &[f32], value: &[f32], erase: f32) {
        let key_norm = vec_norm(key);
        if key_norm < 1e-8 {
            return;
        }

        // Compute write weights (similar to read)
        let mut weights = vec![0.0f32; self.memory_size];
        for (i, row) in self.memory.iter().enumerate() {
            let row_norm = vec_norm(row);
            if row_norm < 1e-8 {
                continue;
            }
            let dot: f32 = key.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
            weights[i] = (dot / (key_norm * row_norm)).exp();
        }

        let sum: f32 = weights.iter().sum();
        if sum > 1e-8 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }

        self.write_weights = weights.clone();

        // Update memory: m_i = m_i * (1 - erase * w_i) + w_i * value
        for (i, row) in self.memory.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let w = weights[i];
                *cell = *cell * (1.0 - erase * w) + w * value.get(j).copied().unwrap_or(0.0);
            }
            // Update usage
            self.usage[i] = self.usage[i] * 0.99 + weights[i];
        }
    }

    /// Get a memory row (for discrete fallback).
    pub fn get_row(&self, idx: usize) -> &[f32] {
        self.memory.get(idx).map(|r| r.as_slice()).unwrap_or(&[])
    }

    /// Reset memory to initial state.
    pub fn reset(&mut self) {
        for row in self.memory.iter_mut() {
            for cell in row.iter_mut() {
                *cell = (rand::random::<f32>() - 0.5) * 0.01;
            }
        }
        self.usage.fill(0.0);
        for w in self.read_weights.iter_mut() {
            w.fill(0.0);
        }
        self.write_weights.fill(0.0);
    }
}

/// The differentiable LLM-Computer.
///
/// Takes OpLogits from the model and produces differentiable CALM programs.
/// Supports both soft (continuous) execution for training and hard (discrete)
/// execution for inference.
pub struct DiffLlmComputer {
    config: DiffLlmComputerConfig,
    /// Differentiable memory bank.
    memory_bank: DiffMemoryBank,
    /// Current temperature for Gumbel-Softmax.
    current_temperature: f32,
    /// Step counter for temperature annealing.
    step_count: usize,
    /// Soft register file (continuous values for gradient flow).
    soft_registers: Vec<f32>,
}

impl DiffLlmComputer {
    /// Create a new differentiable LLM-Computer.
    pub fn new(config: DiffLlmComputerConfig) -> Self {
        let memory_bank = DiffMemoryBank::new(config.memory_size, config.memory_dim);
        let soft_registers = vec![0.0; config.num_registers];

        Self {
            current_temperature: config.temperature,
            config,
            memory_bank,
            step_count: 0,
            soft_registers,
        }
    }

    /// Decode OpLogits into a differentiable step result.
    ///
    /// Uses Gumbel-Softmax for differentiable sampling:
    ///   1. Add Gumbel noise to logits
    ///   2. Softmax to get probabilities
    ///   3. If STE: hard max for forward, soft probs for backward
    pub fn decode_step(&mut self, logits: &OpLogits) -> DiffStepResult {
        // 1. Decode instruction type
        let instr_probs = gumbel_softmax(&logits.instruction_type_logits, self.current_temperature);
        let instr_type_idx = if self.config.use_ste {
            argmax(&instr_probs) // Hard selection for forward
        } else {
            weighted_sample(&instr_probs) // Soft sampling
        };

        // 2. Decode operation (for Compute type)
        let op_probs = gumbel_softmax(&logits.op_logits, self.current_temperature);
        let op_idx = if self.config.use_ste { argmax(&op_probs) } else { weighted_sample(&op_probs) };

        // 3. Decode registers
        let reg_a_probs = gumbel_softmax(&logits.reg_a_logits, self.current_temperature);
        let reg_b_probs = gumbel_softmax(&logits.reg_b_logits, self.current_temperature);
        let reg_out_probs = gumbel_softmax(&logits.reg_out_logits, self.current_temperature);

        let reg_a = argmax(&reg_a_probs) as u32;
        let reg_b = argmax(&reg_b_probs) as u32;
        let reg_out = argmax(&reg_out_probs) as u32;

        // 4. Decode value (for LoadConst)
        let value_probs = gumbel_softmax(&logits.value_logits, self.current_temperature);
        let value_bin = argmax(&value_probs);
        let value = bin_to_value(value_bin, self.config.num_value_bins);

        // 5. Decode table (for LookUp)
        let table_probs = gumbel_softmax(&logits.table_logits, self.current_temperature);
        let table_id = argmax(&table_probs) as u32;

        // Construct the hard instruction
        let instruction = match instr_type_idx {
            0 => CalmInstruction::Compute {
                op: idx_to_calm_op(op_idx),
                a_reg: reg_a,
                b_reg: reg_b,
                output_reg: reg_out,
            },
            1 => CalmInstruction::LookUp {
                table_id,
                key_reg: reg_a,
                output_reg: reg_out,
            },
            2 => CalmInstruction::Append {
                addr_reg: reg_a,
                value_reg: reg_b,
            },
            3 => CalmInstruction::Move {
                input_reg: reg_a,
                output_reg: reg_out,
            },
            4 => CalmInstruction::LoadConst {
                value,
                output_reg: reg_out,
            },
            5 => CalmInstruction::BranchIf {
                condition_reg: reg_a,
                target: (value as u32).max(0),
            },
            _ => CalmInstruction::Halt,
        };

        // Anneal temperature
        self.step_count += 1;
        self.anneal_temperature();

        DiffStepResult {
            instruction,
            instruction_type_probs: instr_probs,
            op_probs,
            reg_probs: vec![
                reg_a_probs.clone(),
                reg_b_probs.clone(),
                reg_out_probs.clone(),
            ],
            temperature: self.current_temperature,
            used_ste: self.config.use_ste,
        }
    }

    /// Execute a differentiable step using soft registers.
    ///
    /// Updates soft_registers in-place with continuous values.
    /// Returns the output value and the step result.
    pub fn execute_diff_step(&mut self, logits: &OpLogits) -> (f32, DiffStepResult) {
        let step = self.decode_step(logits);

        // Execute using soft registers (continuous)
        let output = match &step.instruction {
            CalmInstruction::Compute { op, a_reg, b_reg, output_reg, .. } => {
                let a = self.soft_read(*a_reg);
                let b = self.soft_read(*b_reg);
                // Differentiable op approximation
                let result = diff_op_execute(op, a, b);
                self.soft_write(*output_reg, result);
                result
            }
            CalmInstruction::Move { input_reg, output_reg } => {
                let val = self.soft_read(*input_reg);
                self.soft_write(*output_reg, val);
                val
            }
            CalmInstruction::LoadConst { value, output_reg } => {
                let v = *value as f32;
                self.soft_write(*output_reg, v);
                v
            }
            CalmInstruction::LookUp { key_reg, output_reg, .. } => {
                let key = self.soft_read(*key_reg);
                // Use soft read from memory bank
                let query = vec![key; self.config.memory_dim];
                let result = self.memory_bank.read(&query, 0);
                let val = result.first().copied().unwrap_or(0.0);
                self.soft_write(*output_reg, val);
                val
            }
            CalmInstruction::Append { addr_reg, value_reg } => {
                let addr = self.soft_read(*addr_reg) as usize;
                let val = self.soft_read(*value_reg);
                if addr < self.config.memory_dim {
                    let key = vec![addr as f32; self.config.memory_dim];
                    let value = vec![val; self.config.memory_dim];
                    self.memory_bank.write(&key, &value, 0.5);
                }
                val
            }
            CalmInstruction::BranchIf { .. } => {
                // Branches are handled by the discrete executor; soft execution is linear
                0.0
            }
            CalmInstruction::Halt => 0.0,
        };

        (output, step)
    }

    /// Decode a full program from a sequence of OpLogits.
    ///
    /// Returns the discrete program (for execution) and all soft probabilities
    /// (for backprop).
    pub fn decode_program(&mut self, all_logits: &[OpLogits]) -> DiffProgramResult {
        let max_len = self.config.max_program_length.min(all_logits.len());
        let mut instructions = Vec::with_capacity(max_len);
        let mut step_results = Vec::with_capacity(max_len);

        for logits in all_logits.iter().take(max_len) {
            let step = self.decode_step(logits);
            instructions.push(step.instruction.clone());

            // Stop on Halt
            let is_halt = matches!(step.instruction, CalmInstruction::Halt);
            step_results.push(step);

            if is_halt {
                break;
            }
        }

        DiffProgramResult {
            instructions,
            step_results,
            temperature: self.current_temperature,
        }
    }

    /// Get the auxiliary loss for the decoded program.
    ///
    /// This loss encourages:
    ///   1. Entropy regularization (don't collapse to one op)
    ///   2. Program length regularization (prefer shorter programs)
    ///   3. Convergence (output should stabilize)
    pub fn auxiliary_loss(step_results: &[DiffStepResult], target_output: f32, actual_output: f32) -> f32 {
        // 1. MSE loss on output
        let output_loss = (actual_output - target_output).powi(2);

        // 2. Entropy regularization: encourage exploration
        let mut entropy_loss = 0.0f32;
        for step in step_results {
            let entropy = -step.instruction_type_probs.iter()
                .filter(|&&p| p > 1e-8)
                .map(|&p| p * p.ln())
                .sum::<f32>();
            // We want to maximize entropy (minimize negative entropy)
            entropy_loss -= entropy;
        }
        let avg_entropy_reg = if step_results.is_empty() { 0.0 } else { entropy_loss / step_results.len() as f32 };

        // 3. Length regularization
        let length_loss = step_results.len() as f32 * 0.01;

        output_loss + 0.1 * avg_entropy_reg + length_loss
    }

    /// Soft read from register file.
    fn soft_read(&self, reg: u32) -> f32 {
        self.soft_registers.get(reg as usize).copied().unwrap_or(0.0)
    }

    /// Soft write to register file.
    fn soft_write(&mut self, reg: u32, value: f32) {
        if let Some(r) = self.soft_registers.get_mut(reg as usize) {
            *r = value;
        }
    }

    /// Anneal temperature towards minimum.
    fn anneal_temperature(&mut self) {
        self.current_temperature = (self.current_temperature - self.config.annealing_rate)
            .max(self.config.min_temperature);
    }

    /// Reset for a new program.
    pub fn reset(&mut self) {
        self.soft_registers.fill(0.0);
        self.memory_bank.reset();
        self.current_temperature = self.config.temperature;
        self.step_count = 0;
    }

    /// Get current temperature.
    pub fn temperature(&self) -> f32 {
        self.current_temperature
    }

    /// Get the memory bank.
    pub fn memory_bank(&self) -> &DiffMemoryBank {
        &self.memory_bank
    }

    /// Get the memory bank mutably.
    pub fn memory_bank_mut(&mut self) -> &mut DiffMemoryBank {
        &mut self.memory_bank
    }

    /// Get soft registers.
    pub fn soft_registers(&self) -> &[f32] {
        &self.soft_registers
    }

    /// Get config.
    pub fn config(&self) -> &DiffLlmComputerConfig {
        &self.config
    }
}

/// Result of decoding a full differentiable program.
#[derive(Debug, Clone)]
pub struct DiffProgramResult {
    /// The discrete instructions (for execution).
    pub instructions: Vec<CalmInstruction>,
    /// Per-step soft probabilities (for backprop).
    pub step_results: Vec<DiffStepResult>,
    /// Temperature used.
    pub temperature: f32,
}

impl DiffProgramResult {
    /// Get program length.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Check if program is empty.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Gumbel-Softmax
// ---------------------------------------------------------------------------

/// Gumbel-Softmax: differentiable sampling from a categorical distribution.
///
/// Gumbel-Max trick: argmax(log(π_i) + g_i) where g_i ~ Gumbel(0,1)
/// Softmax relaxation: softmax((log(π_i) + g_i) / τ)
fn gumbel_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if temperature < 1e-8 || logits.is_empty() {
        // Degenerate: return one-hot at max
        let mut result = vec![0.0; logits.len()];
        if !logits.is_empty() {
            let max_idx = argmax_by_value(logits);
            result[max_idx] = 1.0;
        }
        return result;
    }

    // Add Gumbel noise
    let noisy: Vec<f32> = logits.iter()
        .map(|&l| l + sample_gumbel())
        .collect();

    // Softmax with temperature
    let max_val = noisy.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = noisy.iter().map(|&x| ((x - max_val) / temperature).exp()).collect();
    let sum: f32 = exps.iter().sum();

    if sum < 1e-8 {
        let mut result = vec![0.0; logits.len()];
        result[0] = 1.0;
        return result;
    }

    exps.iter().map(|&e| e / sum).collect()
}

/// Sample from Gumbel(0, 1) distribution.
fn sample_gumbel() -> f32 {
    let u = rand::random::<f32>().max(1e-8);
    (-u.ln()).ln()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn argmax_by_value(v: &[f32]) -> usize {
    argmax(v)
}

fn weighted_sample(probs: &[f32]) -> usize {
    let r = rand::random::<f32>();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Map op index to CalmOp.
fn idx_to_calm_op(idx: usize) -> CalmOp {
    match idx % NUM_CALM_OPS {
        0 => CalmOp::Add,
        1 => CalmOp::Sub,
        2 => CalmOp::Mul,
        3 => CalmOp::Div,
        4 => CalmOp::Mod,
        5 => CalmOp::And,
        6 => CalmOp::Or,
        7 => CalmOp::Xor,
        8 => CalmOp::Shl,
        9 => CalmOp::Shr,
        10 => CalmOp::Eq,
        11 => CalmOp::Ne,
        12 => CalmOp::Lt,
        13 => CalmOp::Gt,
        _ => CalmOp::Add,
    }
}

/// Map value bin to actual integer value.
/// Bins are centered: [-(bins/2), ..., 0, ..., (bins/2 - 1)]
fn bin_to_value(bin: usize, num_bins: usize) -> i32 {
    let half = num_bins as i32 / 2;
    (bin as i32) - half
}

/// Differentiable approximation of CALM ops.
///
/// For arithmetic: exact (add, sub, mul are differentiable).
/// For comparisons: sigmoid relaxation.
/// For division: safe with small epsilon.
fn diff_op_execute(op: &CalmOp, a: f32, b: f32) -> f32 {
    match op {
        CalmOp::Add => a + b,
        CalmOp::Sub => a - b,
        CalmOp::Mul => a * b,
        CalmOp::Div => {
            if b.abs() < 1e-8 { 0.0 } else { a / b }
        }
        CalmOp::Mod => {
            if b.abs() < 1e-8 { 0.0 } else { a - b * (a / b).floor() }
        }
        // Sigmoid relaxation for comparisons
        CalmOp::Eq => sigmoid(10.0 * (1.0 - (a - b).abs())),
        CalmOp::Ne => sigmoid(10.0 * (a - b).abs()),
        CalmOp::Lt => sigmoid(10.0 * (b - a)),
        CalmOp::Gt => sigmoid(10.0 * (a - b)),
        // Bitwise: approximated as soft versions
        CalmOp::And => a * b / (a.abs().max(b.abs()) + 1e-8), // Soft AND
        CalmOp::Or => a + b - a * b / (a.abs().max(b.abs()) + 1e-8), // Soft OR
        CalmOp::Xor => (a - b).abs(), // Soft XOR
        CalmOp::Shl => a * (2.0_f32.powf(b)),
        CalmOp::Shr => a / (2.0_f32.powf(b) + 1e-8),
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_logits_zeros() {
        let config = DiffLlmComputerConfig::default();
        let logits = OpLogits::zeros(&config);
        assert_eq!(logits.op_logits.len(), NUM_CALM_OPS);
        assert_eq!(logits.reg_a_logits.len(), config.num_registers);
        assert!(logits.op_logits.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gumbel_softmax_produces_distribution() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let probs = gumbel_softmax(&logits, 1.0);

        assert_eq!(probs.len(), 4);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Probs should sum to 1, got {}", sum);
        assert!(probs.iter().all(|&p| p >= 0.0), "All probs should be non-negative");
    }

    #[test]
    fn test_gumbel_softmax_low_temperature_is_sharp() {
        let logits = vec![0.0, 0.0, 10.0, 0.0]; // Strong preference for index 2
        let probs = gumbel_softmax(&logits, 0.01); // Very low temperature

        assert!(probs[2] > 0.99, "Low temperature should be nearly one-hot, got {}", probs[2]);
    }

    #[test]
    fn test_gumbel_softmax_high_temperature_is_uniform() {
        let logits = vec![1.0, 1.0, 1.0, 1.0]; // Equal logits
        let probs = gumbel_softmax(&logits, 10.0); // High temperature

        // Should be roughly uniform
        for &p in &probs {
            assert!((p - 0.25).abs() < 0.1, "High temperature should be uniform, got {}", p);
        }
    }

    #[test]
    fn test_decode_step_produces_instruction() {
        let config = DiffLlmComputerConfig {
            use_ste: true,
            ..Default::default()
        };
        let mut computer = DiffLlmComputer::new(config);

        let mut logits = OpLogits::zeros(computer.config());
        // Set instruction type to Compute (index 0)
        logits.instruction_type_logits[0] = 10.0;
        // Set op to Add (index 0)
        logits.op_logits[0] = 10.0;
        // Set registers
        logits.reg_a_logits[0] = 5.0;
        logits.reg_b_logits[1] = 5.0;
        logits.reg_out_logits[2] = 5.0;

        let result = computer.decode_step(&logits);
        assert!(matches!(result.instruction, CalmInstruction::Compute { op: CalmOp::Add, .. }));
    }

    #[test]
    fn test_decode_halt() {
        let config = DiffLlmComputerConfig::default();
        let mut computer = DiffLlmComputer::new(config);

        let mut logits = OpLogits::zeros(computer.config());
        // Set instruction type to Halt (index 6)
        logits.instruction_type_logits[6] = 10.0;

        let result = computer.decode_step(&logits);
        assert!(matches!(result.instruction, CalmInstruction::Halt));
    }

    #[test]
    fn test_temperature_annealing() {
        let config = DiffLlmComputerConfig {
            temperature: 1.0,
            min_temperature: 0.1,
            annealing_rate: 0.1,
            ..Default::default()
        };
        let mut computer = DiffLlmComputer::new(config);

        assert!((computer.temperature() - 1.0).abs() < 0.01);

        let logits = OpLogits::zeros(computer.config());
        for _ in 0..5 {
            computer.decode_step(&logits);
        }

        // Should have annealed: 1.0 - 5 * 0.1 = 0.5
        assert!((computer.temperature() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_diff_memory_bank_read_write() {
        let mut bank = DiffMemoryBank::new(16, 8);

        // Write a value
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let value = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.0];
        bank.write(&key, &value, 1.0);

        // Read back
        let result = bank.read(&key, 0);
        // Should retrieve something close to the written value
        // (may not be exact due to soft addressing)
        assert!(result.last().copied().unwrap_or(0.0) > 0.0, "Read should return non-zero");
    }

    #[test]
    fn test_diff_memory_bank_reset() {
        let mut bank = DiffMemoryBank::new(16, 8);
        bank.write(&vec![1.0; 8], &vec![1.0; 8], 1.0);
        bank.reset();

        // Usage should be zeroed
        assert!(bank.usage.iter().all(|&u| u < 0.01));
    }

    #[test]
    fn test_diff_op_execute_arithmetic() {
        // Exact operations
        assert!((diff_op_execute(&CalmOp::Add, 3.0, 4.0) - 7.0).abs() < 0.01);
        assert!((diff_op_execute(&CalmOp::Sub, 10.0, 3.0) - 7.0).abs() < 0.01);
        assert!((diff_op_execute(&CalmOp::Mul, 3.0, 4.0) - 12.0).abs() < 0.01);
        assert!((diff_op_execute(&CalmOp::Div, 10.0, 2.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_diff_op_execute_comparisons() {
        // Sigmoid relaxation
        let eq_result = diff_op_execute(&CalmOp::Eq, 5.0, 5.0);
        assert!(eq_result > 0.9, "Equal values should give high sigmoid, got {}", eq_result);

        let lt_result = diff_op_execute(&CalmOp::Lt, 3.0, 5.0);
        assert!(lt_result > 0.9, "3 < 5 should give high sigmoid, got {}", lt_result);
    }

    #[test]
    fn test_diff_op_execute_safe_div() {
        let div_zero = diff_op_execute(&CalmOp::Div, 10.0, 0.0);
        assert!((div_zero - 0.0).abs() < 0.01, "Div by zero should return 0");
    }

    #[test]
    fn test_bin_to_value() {
        assert_eq!(bin_to_value(16, 32), 0); // Center bin = 0
        assert_eq!(bin_to_value(0, 32), -16); // First bin
        assert_eq!(bin_to_value(31, 32), 15); // Last bin
    }

    #[test]
    fn test_full_diff_execution() {
        let config = DiffLlmComputerConfig {
            temperature: 0.5,
            use_ste: true,
            ..Default::default()
        };
        let mut computer = DiffLlmComputer::new(config);

        // Step 1: LoadConst 5 → reg0
        let mut logits1 = OpLogits::zeros(computer.config());
        logits1.instruction_type_logits[4] = 10.0; // LoadConst
        logits1.reg_out_logits[0] = 10.0; // reg0
        logits1.value_logits[21] = 10.0; // bin 21 → value 5
        let (val1, _) = computer.execute_diff_step(&logits1);
        assert!((val1 - 5.0).abs() < 0.1);

        // Step 2: LoadConst 3 → reg1
        let mut logits2 = OpLogits::zeros(computer.config());
        logits2.instruction_type_logits[4] = 10.0; // LoadConst
        logits2.reg_out_logits[1] = 10.0; // reg1
        logits2.value_logits[19] = 10.0; // bin 19 → value 3
        let (val2, _) = computer.execute_diff_step(&logits2);
        assert!((val2 - 3.0).abs() < 0.1);

        // Step 3: Add reg0 + reg1 → reg2
        let mut logits3 = OpLogits::zeros(computer.config());
        logits3.instruction_type_logits[0] = 10.0; // Compute
        logits3.op_logits[0] = 10.0; // Add
        logits3.reg_a_logits[0] = 10.0; // reg0
        logits3.reg_b_logits[1] = 10.0; // reg1
        logits3.reg_out_logits[2] = 10.0; // reg2
        let (val3, _) = computer.execute_diff_step(&logits3);
        assert!((val3 - 8.0).abs() < 0.1, "Expected 8, got {}", val3);
    }

    #[test]
    fn test_auxiliary_loss() {
        let step = DiffStepResult {
            instruction: CalmInstruction::Halt,
            instruction_type_probs: vec![0.25; 7],
            op_probs: vec![0.25; NUM_CALM_OPS],
            reg_probs: vec![vec![0.5; 16], vec![0.5; 16], vec![0.5; 16]],
            temperature: 1.0,
            used_ste: true,
        };

        let loss = DiffLlmComputer::auxiliary_loss(&[step], 5.0, 5.0);
        // Loss can be slightly negative due to entropy regularization term
        // The key invariant: output_loss = 0 (exact match), so loss ≈ entropy_reg + length_reg
        assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
    }

    #[test]
    fn test_decode_program() {
        let config = DiffLlmComputerConfig {
            temperature: 0.5,
            use_ste: true,
            ..Default::default()
        };
        let mut computer = DiffLlmComputer::new(config.clone());

        // 3 steps: LoadConst, LoadConst, Add
        let make_load = |reg: usize, bin: usize| -> OpLogits {
            let mut l = OpLogits::zeros(&config);
            l.instruction_type_logits[4] = 10.0;
            l.reg_out_logits[reg] = 10.0;
            l.value_logits[bin] = 10.0;
            l
        };

        let mut add_logits = OpLogits::zeros(&config);
        add_logits.instruction_type_logits[0] = 10.0;
        add_logits.op_logits[0] = 10.0;

        let mut halt_logits = OpLogits::zeros(&config);
        halt_logits.instruction_type_logits[6] = 10.0;

        let result = computer.decode_program(&[
            make_load(0, 21), // LoadConst 5 → reg0
            make_load(1, 19), // LoadConst 3 → reg1
            add_logits,
            halt_logits,
        ]);

        assert_eq!(result.len(), 4);
        assert!(matches!(result.instructions.last(), Some(CalmInstruction::Halt)));
    }

    #[test]
    fn test_reset_clears_state() {
        let config = DiffLlmComputerConfig {
            temperature: 1.0,
            annealing_rate: 0.1,
            ..Default::default()
        };
        let mut computer = DiffLlmComputer::new(config);

        let logits = OpLogits::zeros(computer.config());
        for _ in 0..5 {
            computer.decode_step(&logits);
        }

        computer.reset();
        assert!((computer.temperature() - 1.0).abs() < 0.01);
        assert!(computer.soft_registers().iter().all(|&v| v == 0.0));
    }
}
