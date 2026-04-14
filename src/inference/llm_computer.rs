//! LLM-Computer: Treating the transformer as a programmable virtual machine.
//!
//! Based on Percepta research: CALM (Code for Append-only Lookup Machines) DSL
//! compiles program instructions into attention (LookUp gates) and FFN (ReGLU gates).
//! A WASM interpreter can be embedded in transformer weights for exact computation.
//!
//! Key components:
//! - LookUpGate: exact key→value lookup via attention (backed by HullKVCache)
//! - ReGLUGate: computation gate using FFN ReGLU activation pattern
//! - CALM Instruction: simple bytecode for the transformer VM
//! - Program: sequence of CALM instructions compiled to attention/FFN patterns

/// A single CALM (Code for Append-only Lookup Machines) instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalmInstruction {
    /// Look up a value by key: output = table[key]
    LookUp {
        table_id: u32,
        key_reg: u32,
        output_reg: u32,
    },
    /// Write a value to append-only memory: mem[addr] = value
    Append {
        addr_reg: u32,
        value_reg: u32,
    },
    /// Arithmetic: output = a op b
    Compute {
        op: CalmOp,
        a_reg: u32,
        b_reg: u32,
        output_reg: u32,
    },
    /// Conditional: if condition != 0, jump to target
    BranchIf {
        condition_reg: u32,
        target: u32,
    },
    /// Move: output = input
    Move {
        input_reg: u32,
        output_reg: u32,
    },
    /// Load immediate constant into register
    LoadConst {
        value: i32,
        output_reg: u32,
    },
    /// Halt execution
    Halt,
}

/// Arithmetic operations for CALM compute instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalmOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Gt,
}

impl CalmOp {
    /// Execute the operation on two values.
    pub fn execute(&self, a: i32, b: i32) -> i32 {
        match self {
            CalmOp::Add => a.wrapping_add(b),
            CalmOp::Sub => a.wrapping_sub(b),
            CalmOp::Mul => a.wrapping_mul(b),
            CalmOp::Div => if b != 0 { a / b } else { 0 },
            CalmOp::Mod => if b != 0 { a % b } else { 0 },
            CalmOp::And => a & b,
            CalmOp::Or => a | b,
            CalmOp::Xor => a ^ b,
            CalmOp::Shl => a.wrapping_shl(b as u32),
            CalmOp::Shr => a.wrapping_shr(b as u32),
            CalmOp::Eq => if a == b { 1 } else { 0 },
            CalmOp::Ne => if a != b { 1 } else { 0 },
            CalmOp::Lt => if a < b { 1 } else { 0 },
            CalmOp::Gt => if a > b { 1 } else { 0 },
        }
    }

    /// Get all operations.
    pub fn all() -> &'static [CalmOp] {
        &[
            CalmOp::Add, CalmOp::Sub, CalmOp::Mul, CalmOp::Div, CalmOp::Mod,
            CalmOp::And, CalmOp::Or, CalmOp::Xor, CalmOp::Shl, CalmOp::Shr,
            CalmOp::Eq, CalmOp::Ne, CalmOp::Lt, CalmOp::Gt,
        ]
    }
}

/// Configuration for the LLM-Computer virtual machine.
#[derive(Debug, Clone)]
pub struct LlmComputerConfig {
    /// Number of general-purpose registers.
    pub num_registers: usize,
    /// Memory size (number of i32 slots).
    pub memory_size: usize,
    /// Number of lookup tables.
    pub num_tables: usize,
    /// Maximum program length.
    pub max_program_length: usize,
    /// Whether to use HullKVCache for O(log n) lookups.
    pub use_hull_cache: bool,
}

impl Default for LlmComputerConfig {
    fn default() -> Self {
        Self {
            num_registers: 16,
            memory_size: 1024,
            num_tables: 8,
            max_program_length: 256,
            use_hull_cache: true,
        }
    }
}

/// Execution state of the LLM-Computer.
#[derive(Debug)]
pub struct VmState {
    /// General-purpose registers.
    registers: Vec<i32>,
    /// Append-only memory.
    memory: Vec<i32>,
    /// Lookup tables: table_id → (key → value).
    tables: Vec<Vec<(i32, i32)>>,
    /// Program counter.
    pc: usize,
    /// Whether the VM is halted.
    halted: bool,
    /// Step count.
    steps: usize,
    /// Maximum steps before forced halt.
    max_steps: usize,
}

impl VmState {
    /// Create a new VM state.
    pub fn new(config: &LlmComputerConfig) -> Self {
        Self {
            registers: vec![0; config.num_registers],
            memory: vec![0; config.memory_size],
            tables: vec![Vec::new(); config.num_tables],
            pc: 0,
            halted: false,
            steps: 0,
            max_steps: config.max_program_length * 10,
        }
    }

    /// Read a register.
    pub fn read_reg(&self, reg: u32) -> i32 {
        self.registers.get(reg as usize).copied().unwrap_or(0)
    }

    /// Write a register.
    pub fn write_reg(&mut self, reg: u32, value: i32) {
        if let Some(r) = self.registers.get_mut(reg as usize) {
            *r = value;
        }
    }

    /// Read from memory.
    pub fn read_mem(&self, addr: usize) -> i32 {
        self.memory.get(addr).copied().unwrap_or(0)
    }

    /// Read a range of memory.
    pub fn read_mem_range(&self, start: usize, len: usize) -> Vec<i32> {
        self.memory[start..start.min(self.memory.len())].iter()
            .take(len)
            .copied()
            .collect()
    }

    /// Write to memory.
    pub fn write_mem(&mut self, addr: usize, value: i32) {
        if let Some(m) = self.memory.get_mut(addr) {
            *m = value;
        }
    }

    /// Look up a value in a table.
    pub fn table_lookup(&self, table_id: u32, key: i32) -> i32 {
        self.tables
            .get(table_id as usize)
            .and_then(|table| table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v))
            .unwrap_or(0)
    }

    /// Set a table entry.
    pub fn table_set(&mut self, table_id: u32, key: i32, value: i32) {
        if let Some(table) = self.tables.get_mut(table_id as usize) {
            table.push((key, value));
        }
    }

    /// Get the program counter.
    pub fn pc(&self) -> usize {
        self.pc
    }

    /// Check if halted.
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Get step count.
    pub fn steps(&self) -> usize {
        self.steps
    }
}

/// LLM-Computer: a virtual machine that maps transformer operations to computation.
///
/// LookUp instructions → attention heads (exact key→value retrieval)
/// Compute instructions → FFN ReGLU gates (arithmetic/logic)
/// Branch instructions → conditional attention routing
pub struct LlmComputer {
    config: LlmComputerConfig,
    program: Vec<CalmInstruction>,
}

impl LlmComputer {
    /// Create a new LLM-Computer with the given configuration.
    pub fn new(config: LlmComputerConfig) -> Self {
        Self {
            config,
            program: Vec::new(),
        }
    }

    /// Load a CALM program.
    pub fn load_program(&mut self, program: Vec<CalmInstruction>) {
        assert!(program.len() <= self.config.max_program_length,
            "Program too long: {} > {}", program.len(), self.config.max_program_length);
        self.program = program;
    }

    /// Execute the loaded program on fresh VM state.
    pub fn execute(&self) -> VmState {
        let mut state = VmState::new(&self.config);
        self.execute_on(&mut state);
        state
    }

    /// Execute on existing VM state (for step-by-step execution).
    pub fn execute_on(&self, state: &mut VmState) {
        while !state.halted && state.pc < self.program.len() && state.steps < state.max_steps {
            self.step(state);
        }
    }

    /// Execute a single instruction.
    pub fn step(&self, state: &mut VmState) {
        if state.halted || state.pc >= self.program.len() {
            return;
        }

        let instruction = self.program[state.pc].clone();
        state.pc += 1;
        state.steps += 1;

        match instruction {
            CalmInstruction::LookUp { table_id, key_reg, output_reg } => {
                let key = state.read_reg(key_reg);
                let value = state.table_lookup(table_id, key);
                state.write_reg(output_reg, value);
            }
            CalmInstruction::Append { addr_reg, value_reg } => {
                let addr = state.read_reg(addr_reg) as usize;
                let value = state.read_reg(value_reg);
                state.write_mem(addr, value);
            }
            CalmInstruction::Compute { op, a_reg, b_reg, output_reg } => {
                let a = state.read_reg(a_reg);
                let b = state.read_reg(b_reg);
                let result = op.execute(a, b);
                state.write_reg(output_reg, result);
            }
            CalmInstruction::BranchIf { condition_reg, target } => {
                let cond = state.read_reg(condition_reg);
                if cond != 0 {
                    state.pc = target as usize;
                }
            }
            CalmInstruction::Move { input_reg, output_reg } => {
                let value = state.read_reg(input_reg);
                state.write_reg(output_reg, value);
            }
            CalmInstruction::LoadConst { value, output_reg } => {
                state.write_reg(output_reg, value);
            }
            CalmInstruction::Halt => {
                state.halted = true;
            }
        }
    }

    /// Compile a simple program from a high-level description.
    /// This is a minimal compiler for common patterns.
    pub fn compile_add_program(a: i32, b: i32, result_reg: u32) -> Vec<CalmInstruction> {
        vec![
            CalmInstruction::LoadConst { value: a, output_reg: 0 },
            CalmInstruction::LoadConst { value: b, output_reg: 1 },
            CalmInstruction::Compute {
                op: CalmOp::Add,
                a_reg: 0,
                b_reg: 1,
                output_reg: result_reg,
            },
            CalmInstruction::Halt,
        ]
    }

    /// Compile a lookup-table program.
    pub fn compile_lookup_program(
        table_id: u32,
        key: i32,
        output_reg: u32,
    ) -> Vec<CalmInstruction> {
        vec![
            CalmInstruction::LoadConst { value: key, output_reg: 0 },
            CalmInstruction::LookUp {
                table_id,
                key_reg: 0,
                output_reg,
            },
            CalmInstruction::Halt,
        ]
    }

    /// Compile a loop program (sum 1..n).
    pub fn compile_sum_program(n: i32, result_reg: u32) -> Vec<CalmInstruction> {
        // reg0 = counter, reg1 = accumulator
        // reg10 = constant 0, reg11 = constant 1
        vec![
            CalmInstruction::LoadConst { value: n, output_reg: 0 },          // counter = n
            CalmInstruction::LoadConst { value: 0, output_reg: 1 },          // acc = 0
            CalmInstruction::LoadConst { value: 0, output_reg: 10 },         // zero constant
            CalmInstruction::LoadConst { value: 1, output_reg: 11 },         // one constant
            // Loop start (PC=4):
            CalmInstruction::Compute { op: CalmOp::Eq, a_reg: 0, b_reg: 10, output_reg: 3 }, // reg3 = (counter == 0)
            CalmInstruction::BranchIf { condition_reg: 3, target: 9 },       // if counter==0, goto Move
            CalmInstruction::Compute { op: CalmOp::Add, a_reg: 1, b_reg: 0, output_reg: 1 }, // acc += counter
            CalmInstruction::Compute { op: CalmOp::Sub, a_reg: 0, b_reg: 11, output_reg: 0 }, // counter -= 1
            CalmInstruction::BranchIf { condition_reg: 11, target: 4 },       // unconditional jump (reg11=1)
            // End (PC=9):
            CalmInstruction::Move { input_reg: 1, output_reg: result_reg },
            CalmInstruction::Halt,
        ]
    }

    /// Get the config.
    pub fn config(&self) -> &LlmComputerConfig {
        &self.config
    }

    /// Get the program.
    pub fn program(&self) -> &[CalmInstruction] {
        &self.program
    }
}

impl VmState {
    /// Save VM state to binary bytes.
    /// Format: [num_regs u32] [registers] [memory_size u32] [memory]
    ///         [num_tables u32] [per table: num_entries u32] [entries]
    ///         [pc u64] [halted u8] [steps u64]
    pub fn save_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let write_u32 = |buf: &mut Vec<u8>, v: u32| { buf.extend_from_slice(&v.to_le_bytes()); };
        let write_i32 = |buf: &mut Vec<u8>, v: i32| { buf.extend_from_slice(&v.to_le_bytes()); };
        let write_u64 = |buf: &mut Vec<u8>, v: u64| { buf.extend_from_slice(&v.to_le_bytes()); };

        write_u32(&mut buf, self.registers.len() as u32);
        for &r in &self.registers { write_i32(&mut buf, r); }
        write_u32(&mut buf, self.memory.len() as u32);
        for &m in &self.memory { write_i32(&mut buf, m); }
        write_u32(&mut buf, self.tables.len() as u32);
        for table in &self.tables {
            write_u32(&mut buf, table.len() as u32);
            for &(k, v) in table {
                write_i32(&mut buf, k);
                write_i32(&mut buf, v);
            }
        }
        write_u64(&mut buf, self.pc as u64);
        buf.push(if self.halted { 1 } else { 0 });
        write_u64(&mut buf, self.steps as u64);
        buf
    }

    /// Load VM state from binary bytes.
    pub fn load_bytes(data: &[u8]) -> Option<Self> {
        let mut pos = 0usize;
        let read_u32 = |data: &[u8], pos: &mut usize| -> Option<u32> {
            if *pos + 4 > data.len() { return None; }
            let v = u32::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3]]);
            *pos += 4; Some(v)
        };
        let read_i32 = |data: &[u8], pos: &mut usize| -> Option<i32> {
            if *pos + 4 > data.len() { return None; }
            let v = i32::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3]]);
            *pos += 4; Some(v)
        };
        let read_u64 = |data: &[u8], pos: &mut usize| -> Option<u64> {
            if *pos + 8 > data.len() { return None; }
            let v = u64::from_le_bytes(data[*pos..*pos+8].try_into().ok()?);
            *pos += 8; Some(v)
        };

        let num_regs = read_u32(data, &mut pos)? as usize;
        let mut registers = Vec::with_capacity(num_regs);
        for _ in 0..num_regs { registers.push(read_i32(data, &mut pos)?); }

        let mem_size = read_u32(data, &mut pos)? as usize;
        let mut memory = Vec::with_capacity(mem_size);
        for _ in 0..mem_size { memory.push(read_i32(data, &mut pos)?); }

        let num_tables = read_u32(data, &mut pos)? as usize;
        let mut tables = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            let num_entries = read_u32(data, &mut pos)? as usize;
            let mut table = Vec::with_capacity(num_entries);
            for _ in 0..num_entries {
                let k = read_i32(data, &mut pos)?;
                let v = read_i32(data, &mut pos)?;
                table.push((k, v));
            }
            tables.push(table);
        }

        let pc = read_u64(data, &mut pos)? as usize;
        let halted = data.get(pos).copied()? != 0;
        pos += 1;
        let steps = read_u64(data, &mut pos)? as usize;

        Some(Self { registers, memory, tables, pc, halted, steps, max_steps: steps * 10 + 1000 })
    }

    /// Save VM state to file.
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, &self.save_bytes())
    }

    /// Load VM state from file.
    pub fn load_from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        Self::load_bytes(&data).ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::InvalidData, "Failed to parse VM state"))
    }
}

/// Decode CALM instructions from a sequence of token IDs.
/// Maps token IDs to CALM opcodes using a fixed vocabulary.
/// This enables the model to "write its own tools" by generating
/// token sequences that decode to executable bytecode.
pub struct CalmDecoder {
    /// Token ID → instruction mapping.
    vocab: std::collections::HashMap<u32, CalmInstruction>,
}

impl CalmDecoder {
    /// Create a decoder with standard token-to-instruction mapping.
    /// Uses a fixed range of token IDs (e.g., 256000-256015) for CALM ops.
    pub fn new(base_vocab_size: u32) -> Self {
        let mut vocab = std::collections::HashMap::new();
        let base = base_vocab_size;

        // Core instructions (fixed opcodes)
        vocab.insert(base,      CalmInstruction::Halt);
        vocab.insert(base + 1,  CalmInstruction::LoadConst { value: 0, output_reg: 0 }); // Template
        vocab.insert(base + 2,  CalmInstruction::Move { input_reg: 0, output_reg: 0 });
        vocab.insert(base + 3,  CalmInstruction::Compute { op: CalmOp::Add, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 4,  CalmInstruction::Compute { op: CalmOp::Sub, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 5,  CalmInstruction::Compute { op: CalmOp::Mul, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 6,  CalmInstruction::Compute { op: CalmOp::Div, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 7,  CalmInstruction::Compute { op: CalmOp::Eq, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 8,  CalmInstruction::Compute { op: CalmOp::Lt, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 9,  CalmInstruction::Compute { op: CalmOp::Gt, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 10, CalmInstruction::Compute { op: CalmOp::And, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 11, CalmInstruction::Compute { op: CalmOp::Or, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 12, CalmInstruction::Compute { op: CalmOp::Xor, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 13, CalmInstruction::Compute { op: CalmOp::Ne, a_reg: 0, b_reg: 0, output_reg: 0 });
        vocab.insert(base + 14, CalmInstruction::LookUp { table_id: 0, key_reg: 0, output_reg: 0 });
        vocab.insert(base + 15, CalmInstruction::Append { addr_reg: 0, value_reg: 0 });
        vocab.insert(base + 16, CalmInstruction::BranchIf { condition_reg: 0, target: 0 });

        Self { vocab }
    }

    /// Decode a sequence of token IDs into CALM instructions.
    /// Non-CALM tokens are skipped.
    pub fn decode(&self, token_ids: &[u32]) -> Vec<CalmInstruction> {
        token_ids.iter()
            .filter_map(|&id| self.vocab.get(&id).cloned())
            .collect()
    }

    /// Check if a token ID is a CALM instruction.
    pub fn is_calm_token(&self, token_id: u32) -> bool {
        self.vocab.contains_key(&token_id)
    }

    /// Get the base vocab size where CALM tokens start.
    pub fn base_vocab_size(&self) -> u32 {
        *self.vocab.keys().min().unwrap_or(&0)
    }
}

/// Online distillation trigger.
/// Monitors model confidence and triggers background distillation
/// when the model's self-assessment indicates it needs improvement.
pub struct OnlineDistillationTrigger {
    /// Confidence threshold below which distillation is triggered.
    confidence_threshold: f32,
    /// Number of low-confidence events before triggering.
    trigger_count: usize,
    /// Current low-confidence event counter.
    low_confidence_events: usize,
    /// Whether distillation is currently running.
    distilling: bool,
}

impl OnlineDistillationTrigger {
    pub fn new(confidence_threshold: f32, trigger_count: usize) -> Self {
        Self {
            confidence_threshold,
            trigger_count,
            low_confidence_events: 0,
            distilling: false,
        }
    }

    /// Record a prediction and check if distillation should trigger.
    /// `confidence` is max softmax probability of the model's output.
    /// Returns true if distillation should be triggered.
    pub fn record_prediction(&mut self, confidence: f32) -> bool {
        if self.distilling { return false; }
        if confidence < self.confidence_threshold {
            self.low_confidence_events += 1;
            if self.low_confidence_events >= self.trigger_count {
                self.distilling = true;
                return true;
            }
        } else {
            // Reset on high confidence
            self.low_confidence_events = 0;
        }
        false
    }

    /// Mark distillation as complete.
    pub fn distillation_complete(&mut self) {
        self.low_confidence_events = 0;
        self.distilling = false;
    }

    /// Check if distillation is in progress.
    pub fn is_distilling(&self) -> bool {
        self.distilling
    }
}

// ---------------------------------------------------------------------------
// VM State Persistence (task 1c0402e5)
// ---------------------------------------------------------------------------

/// Serializable VM snapshot for persistence across sessions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VmSnapshot {
    pub registers: Vec<i32>,
    pub memory: Vec<i32>,
    pub tables: Vec<Vec<(i32, i32)>>,
    pub pc: usize,
    pub halted: bool,
    pub steps: usize,
    pub max_steps: usize,
    pub program: Vec<CalmInstructionSer>,
    pub timestamp: u64,
    pub label: String,
}

/// Serializable representation of CALM instructions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CalmInstructionSer {
    LookUp { table_id: u32, key_reg: u32, output_reg: u32 },
    Append { addr_reg: u32, value_reg: u32 },
    Compute { op: String, a_reg: u32, b_reg: u32, output_reg: u32 },
    BranchIf { condition_reg: u32, target: u32 },
    Move { input_reg: u32, output_reg: u32 },
    LoadConst { value: i32, output_reg: u32 },
    Halt,
}

impl VmState {
    /// Serialize current VM state to a snapshot.
    pub fn snapshot(&self, program: &[CalmInstruction], label: &str) -> VmSnapshot {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        VmSnapshot {
            registers: self.registers.clone(),
            memory: self.memory.clone(),
            tables: self.tables.clone(),
            pc: self.pc,
            halted: self.halted,
            steps: self.steps,
            max_steps: self.max_steps,
            program: program.iter().map(CalmInstructionSer::from_instr).collect(),
            timestamp,
            label: label.to_string(),
        }
    }

    /// Restore VM state from a snapshot.
    pub fn from_snapshot(snapshot: &VmSnapshot) -> Self {
        Self {
            registers: snapshot.registers.clone(),
            memory: snapshot.memory.clone(),
            tables: snapshot.tables.clone(),
            pc: snapshot.pc,
            halted: snapshot.halted,
            steps: snapshot.steps,
            max_steps: snapshot.max_steps,
        }
    }
}

impl VmSnapshot {
    /// Save snapshot to a file.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string(self).map_err(|e| format!("Serialize: {}", e))?;
        std::fs::write(path, json).map_err(|e| format!("Write: {}", e))
    }

    /// Load snapshot from a file.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("Read: {}", e))?;
        serde_json::from_str(&json).map_err(|e| format!("Deserialize: {}", e))
    }
}

impl CalmInstructionSer {
    fn from_instr(instr: &CalmInstruction) -> Self {
        match instr {
            CalmInstruction::LookUp { table_id, key_reg, output_reg } =>
                Self::LookUp { table_id: *table_id, key_reg: *key_reg, output_reg: *output_reg },
            CalmInstruction::Append { addr_reg, value_reg } =>
                Self::Append { addr_reg: *addr_reg, value_reg: *value_reg },
            CalmInstruction::Compute { op, a_reg, b_reg, output_reg } =>
                Self::Compute {
                    op: format!("{:?}", op).to_lowercase(),
                    a_reg: *a_reg, b_reg: *b_reg, output_reg: *output_reg,
                },
            CalmInstruction::BranchIf { condition_reg, target } =>
                Self::BranchIf { condition_reg: *condition_reg, target: *target },
            CalmInstruction::Move { input_reg, output_reg } =>
                Self::Move { input_reg: *input_reg, output_reg: *output_reg },
            CalmInstruction::LoadConst { value, output_reg } =>
                Self::LoadConst { value: *value, output_reg: *output_reg },
            CalmInstruction::Halt => Self::Halt,
        }
    }

    /// Deserialize back to a CalmInstruction.
    pub fn to_instr(&self) -> Option<CalmInstruction> {
        match self {
            Self::LookUp { table_id, key_reg, output_reg } =>
                Some(CalmInstruction::LookUp { table_id: *table_id, key_reg: *key_reg, output_reg: *output_reg }),
            Self::Append { addr_reg, value_reg } =>
                Some(CalmInstruction::Append { addr_reg: *addr_reg, value_reg: *value_reg }),
            Self::Compute { op, a_reg, b_reg, output_reg } => {
                let calm_op = match op.as_str() {
                    "add" => CalmOp::Add, "sub" => CalmOp::Sub, "mul" => CalmOp::Mul,
                    "div" => CalmOp::Div, "mod" => CalmOp::Mod,
                    "and" => CalmOp::And, "or" => CalmOp::Or, "xor" => CalmOp::Xor,
                    "shl" => CalmOp::Shl, "shr" => CalmOp::Shr,
                    "eq" => CalmOp::Eq, "ne" => CalmOp::Ne, "lt" => CalmOp::Lt, "gt" => CalmOp::Gt,
                    _ => return None,
                };
                Some(CalmInstruction::Compute { op: calm_op, a_reg: *a_reg, b_reg: *b_reg, output_reg: *output_reg })
            }
            Self::BranchIf { condition_reg, target } =>
                Some(CalmInstruction::BranchIf { condition_reg: *condition_reg, target: *target }),
            Self::Move { input_reg, output_reg } =>
                Some(CalmInstruction::Move { input_reg: *input_reg, output_reg: *output_reg }),
            Self::LoadConst { value, output_reg } =>
                Some(CalmInstruction::LoadConst { value: *value, output_reg: *output_reg }),
            Self::Halt => Some(CalmInstruction::Halt),
        }
    }
}

// ---------------------------------------------------------------------------
// CALM Self-Authoring (task 82d7c212)
// ---------------------------------------------------------------------------

/// A self-authored CALM program generated by the model.
#[derive(Debug, Clone)]
pub struct SelfAuthoredProgram {
    /// The program instructions.
    pub instructions: Vec<CalmInstruction>,
    /// Natural language description of what the program does.
    pub description: String,
    /// The model's confidence in the program's correctness.
    pub confidence: f32,
    /// Token IDs that generated this program.
    pub source_tokens: Vec<u32>,
}

/// Authoring result — either a valid program or compilation errors.
#[derive(Debug, Clone)]
pub enum AuthoringResult {
    Success(SelfAuthoredProgram),
    CompileError { message: String, token_position: usize },
}

/// CALM self-authoring engine.
/// Converts model output (token IDs) into executable CALM programs.
pub struct CalmAuthoringEngine {
    decoder: CalmDecoder,
    max_program_length: usize,
}

impl CalmAuthoringEngine {
    pub fn new(vocab_size: u32) -> Self {
        Self {
            decoder: CalmDecoder::new(vocab_size),
            max_program_length: 256,
        }
    }

    /// Author a CALM program from model-generated token IDs.
    /// The tokens between CALM_START and CALM_END markers are decoded.
    pub fn author_from_tokens(&self, tokens: &[u32]) -> AuthoringResult {
        let program = self.decoder.decode(tokens);

        if program.is_empty() {
            return AuthoringResult::CompileError {
                message: "No valid CALM instructions found in token sequence".into(),
                token_position: 0,
            };
        }

        if program.len() > self.max_program_length {
            return AuthoringResult::CompileError {
                message: format!("Program too long: {} > {}", program.len(), self.max_program_length),
                token_position: tokens.len().saturating_sub(1),
            };
        }

        // Validate: program must end with Halt
        let has_halt = program.last().map_or(false, |i|
            matches!(i, CalmInstruction::Halt)
        );

        let mut instructions = program;
        if !has_halt {
            instructions.push(CalmInstruction::Halt);
        }

        // Validate register bounds
        for (idx, instr) in instructions.iter().enumerate() {
            if let Err(msg) = self.validate_instruction(instr) {
                return AuthoringResult::CompileError {
                    message: msg,
                    token_position: idx,
                };
            }
        }

        AuthoringResult::Success(SelfAuthoredProgram {
            confidence: 0.5, // Will be updated after execution
            description: String::new(), // Filled by model
            source_tokens: tokens.to_vec(),
            instructions,
        })
    }

    /// Validate a single instruction.
    fn validate_instruction(&self, instr: &CalmInstruction) -> Result<(), String> {
        const MAX_REG: u32 = 15; // 16 registers
        match instr {
            CalmInstruction::LookUp { key_reg, output_reg, .. } => {
                if *key_reg > MAX_REG || *output_reg > MAX_REG {
                    return Err(format!("Register out of bounds"));
                }
            }
            CalmInstruction::Compute { a_reg, b_reg, output_reg, .. } => {
                if *a_reg > MAX_REG || *b_reg > MAX_REG || *output_reg > MAX_REG {
                    return Err(format!("Register out of bounds"));
                }
            }
            CalmInstruction::BranchIf { condition_reg, target } => {
                if *condition_reg > MAX_REG {
                    return Err(format!("Register out of bounds"));
                }
                if *target as usize >= self.max_program_length {
                    return Err(format!("Branch target out of bounds: {}", target));
                }
            }
            CalmInstruction::LoadConst { output_reg, .. } => {
                if *output_reg > MAX_REG {
                    return Err(format!("Register out of bounds"));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Execute a self-authored program and return the result.
    pub fn execute_authored(&self, authored: &SelfAuthoredProgram) -> VmExecutionResult {
        let config = LlmComputerConfig::default();
        let mut computer = LlmComputer::new(config);
        computer.load_program(authored.instructions.clone());
        let state = computer.execute();

        VmExecutionResult {
            halted_normally: state.is_halted(),
            final_register_0: state.read_reg(0),
            steps: state.steps(),
            memory_snapshot: state.read_mem_range(0, 64),
        }
    }
}

/// Result of executing a self-authored program.
#[derive(Debug, Clone)]
pub struct VmExecutionResult {
    pub halted_normally: bool,
    pub final_register_0: i32,
    pub steps: usize,
    pub memory_snapshot: Vec<i32>,
}

// ---------------------------------------------------------------------------
// Self-Correction via VM Ground Truth (task 5f28d6ed)
// ---------------------------------------------------------------------------

/// Self-correction using VM execution as ground truth.
/// The model predicts an answer, the VM computes the correct answer,
/// and the difference becomes a loss signal for backpropagation.
pub struct VmSelfCorrection {
    engine: CalmAuthoringEngine,
    /// History of corrections for curriculum learning.
    corrections: Vec<CorrectionRecord>,
}

/// A single correction record.
#[derive(Debug, Clone)]
pub struct CorrectionRecord {
    /// What the model predicted.
    pub model_prediction: i32,
    /// What the VM computed (ground truth).
    pub vm_result: i32,
    /// The loss from this correction.
    pub loss: f32,
    /// Whether the model was correct.
    pub correct: bool,
}

impl VmSelfCorrection {
    pub fn new(vocab_size: u32) -> Self {
        Self {
            engine: CalmAuthoringEngine::new(vocab_size),
            corrections: Vec::new(),
        }
    }

    /// Compute ground truth loss: compare model prediction to VM execution.
    ///
    /// `model_prediction`: the value the model predicted (e.g., register 0).
    /// `program_tokens`: token IDs that encode the CALM program.
    ///
    /// Returns (loss, vm_result, correct).
    pub fn compute_correction(
        &mut self,
        model_prediction: i32,
        program_tokens: &[u32],
    ) -> (f32, i32, bool) {
        match self.engine.author_from_tokens(program_tokens) {
            AuthoringResult::Success(program) => {
                let result = self.engine.execute_authored(&program);
                let vm_result = result.final_register_0;
                let correct = model_prediction == vm_result;

                // MSE-style loss
                let diff = (model_prediction - vm_result) as f32;
                let loss = diff * diff;

                self.corrections.push(CorrectionRecord {
                    model_prediction,
                    vm_result,
                    loss,
                    correct,
                });

                (loss, vm_result, correct)
            }
            AuthoringResult::CompileError { message: _message, .. } => {
                // Compilation failure → high loss
                let loss = 100.0; // Penalty for non-compiling code
                self.corrections.push(CorrectionRecord {
                    model_prediction,
                    vm_result: 0,
                    loss,
                    correct: false,
                });
                (loss, 0, false)
            }
        }
    }

    /// Compute batch correction loss.
    pub fn batch_correction_loss(&self) -> f32 {
        if self.corrections.is_empty() { return 0.0; }
        self.corrections.iter().map(|c| c.loss).sum::<f32>() / self.corrections.len() as f32
    }

    /// Get accuracy over all corrections.
    pub fn accuracy(&self) -> f32 {
        if self.corrections.is_empty() { return 0.0; }
        self.corrections.iter().filter(|c| c.correct).count() as f32 / self.corrections.len() as f32
    }

    /// Get correction history.
    pub fn corrections(&self) -> &[CorrectionRecord] {
        &self.corrections
    }

    /// Clear correction history.
    pub fn reset(&mut self) {
        self.corrections.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calm_op_arithmetic() {
        assert_eq!(CalmOp::Add.execute(3, 4), 7);
        assert_eq!(CalmOp::Sub.execute(10, 3), 7);
        assert_eq!(CalmOp::Mul.execute(3, 4), 12);
        assert_eq!(CalmOp::Div.execute(10, 3), 3);
        assert_eq!(CalmOp::Mod.execute(10, 3), 1);
    }

    #[test]
    fn test_calm_op_bitwise() {
        assert_eq!(CalmOp::And.execute(0b1100, 0b1010), 0b1000);
        assert_eq!(CalmOp::Or.execute(0b1100, 0b1010), 0b1110);
        assert_eq!(CalmOp::Xor.execute(0b1100, 0b1010), 0b0110);
    }

    #[test]
    fn test_calm_op_comparison() {
        assert_eq!(CalmOp::Eq.execute(3, 3), 1);
        assert_eq!(CalmOp::Eq.execute(3, 4), 0);
        assert_eq!(CalmOp::Lt.execute(3, 4), 1);
        assert_eq!(CalmOp::Gt.execute(4, 3), 1);
    }

    #[test]
    fn test_div_by_zero() {
        assert_eq!(CalmOp::Div.execute(10, 0), 0);
        assert_eq!(CalmOp::Mod.execute(10, 0), 0);
    }

    #[test]
    fn test_simple_add_program() {
        let config = LlmComputerConfig::default();
        let mut vm = LlmComputer::new(config);
        let program = LlmComputer::compile_add_program(7, 5, 2);
        vm.load_program(program);

        let state = vm.execute();
        assert!(state.is_halted());
        assert_eq!(state.read_reg(2), 12);
    }

    #[test]
    fn test_lookup_program() {
        let config = LlmComputerConfig::default();
        let mut vm = LlmComputer::new(config);
        let program = LlmComputer::compile_lookup_program(0, 42, 1);
        vm.load_program(program);

        let mut state = VmState::new(vm.config());
        state.table_set(0, 42, 100); // table[0][42] = 100
        vm.execute_on(&mut state);

        assert!(state.is_halted());
        assert_eq!(state.read_reg(1), 100);
    }

    #[test]
    fn test_lookup_miss() {
        let config = LlmComputerConfig::default();
        let mut vm = LlmComputer::new(config);
        let program = LlmComputer::compile_lookup_program(0, 99, 1);
        vm.load_program(program);

        let mut state = VmState::new(vm.config());
        state.table_set(0, 42, 100); // Key 42 exists, but we look up 99
        vm.execute_on(&mut state);

        assert_eq!(state.read_reg(1), 0); // Missing key → 0
    }

    #[test]
    fn test_memory_append() {
        let config = LlmComputerConfig::default();
        let mut vm = LlmComputer::new(config);
        vm.load_program(vec![
            CalmInstruction::LoadConst { value: 5, output_reg: 0 },  // addr
            CalmInstruction::LoadConst { value: 42, output_reg: 1 }, // value
            CalmInstruction::Append { addr_reg: 0, value_reg: 1 },
            CalmInstruction::Halt,
        ]);

        let state = vm.execute();
        assert_eq!(state.read_mem(5), 42);
    }

    #[test]
    fn test_conditional_branch() {
        let config = LlmComputerConfig::default();
        let mut vm = LlmComputer::new(config);
        vm.load_program(vec![
            CalmInstruction::LoadConst { value: 1, output_reg: 0 },
            CalmInstruction::LoadConst { value: 10, output_reg: 1 },
            CalmInstruction::LoadConst { value: 20, output_reg: 2 },
            // if reg0 != 0 (true), skip to PC=5 (load 100)
            CalmInstruction::BranchIf { condition_reg: 0, target: 5 },
            CalmInstruction::LoadConst { value: 0, output_reg: 3 },  // skipped
            CalmInstruction::LoadConst { value: 100, output_reg: 3 },
            CalmInstruction::Halt,
        ]);

        let state = vm.execute();
        assert_eq!(state.read_reg(3), 100); // Took the branch
    }

    #[test]
    fn test_sum_program() {
        let config = LlmComputerConfig {
            num_registers: 16,
            max_program_length: 256,
            ..Default::default()
        };
        let mut vm = LlmComputer::new(config);
        let program = LlmComputer::compile_sum_program(5, 6);
        vm.load_program(program);

        let state = vm.execute();
        assert!(state.is_halted());
        assert_eq!(state.read_reg(1), 15); // acc should be 15
        assert_eq!(state.read_reg(6), 15); // result_reg should be 15
    }

    #[test]
    fn test_step_by_step_execution() {
        let config = LlmComputerConfig::default();
        let mut vm = LlmComputer::new(config);
        vm.load_program(vec![
            CalmInstruction::LoadConst { value: 10, output_reg: 0 },
            CalmInstruction::LoadConst { value: 20, output_reg: 1 },
            CalmInstruction::Compute { op: CalmOp::Add, a_reg: 0, b_reg: 1, output_reg: 2 },
            CalmInstruction::Halt,
        ]);

        let mut state = VmState::new(vm.config());

        vm.step(&mut state);
        assert_eq!(state.pc(), 1);
        assert_eq!(state.read_reg(0), 10);

        vm.step(&mut state);
        assert_eq!(state.read_reg(1), 20);

        vm.step(&mut state);
        assert_eq!(state.read_reg(2), 30);

        vm.step(&mut state);
        assert!(state.is_halted());
    }

    #[test]
    fn test_vm_state_initial() {
        let config = LlmComputerConfig::default();
        let state = VmState::new(&config);
        assert_eq!(state.pc(), 0);
        assert!(!state.is_halted());
        assert_eq!(state.steps(), 0);
        assert_eq!(state.read_reg(0), 0);
    }

    #[test]
    fn test_all_ops_execute() {
        for op in CalmOp::all() {
            let result = op.execute(10, 5);
            // Just ensure no panic — value validation is in specific tests
            let _ = result;
        }
    }

    #[test]
    fn test_vm_state_save_load() {
        let config = LlmComputerConfig::default();
        let mut state = VmState::new(&config);
        state.write_reg(0, 42);
        state.write_reg(1, 99);
        state.write_mem(0, 7);
        state.table_set(0, 10, 20);

        let bytes = state.save_bytes();
        let loaded = VmState::load_bytes(&bytes).unwrap();
        assert_eq!(loaded.read_reg(0), 42);
        assert_eq!(loaded.read_reg(1), 99);
        assert_eq!(loaded.read_mem(0), 7);
        assert_eq!(loaded.table_lookup(0, 10), 20);
    }

    #[test]
    fn test_vm_state_file_roundtrip() {
        let dir = std::env::temp_dir().join("ferrisres_vm_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vm_state.bin");

        let config = LlmComputerConfig::default();
        let mut state = VmState::new(&config);
        state.write_reg(3, 12345);
        state.save_to_file(&path).unwrap();

        let loaded = VmState::load_from_file(&path).unwrap();
        assert_eq!(loaded.read_reg(3), 12345);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_calm_decoder() {
        let decoder = CalmDecoder::new(256000);
        assert!(decoder.is_calm_token(256000)); // Halt
        assert!(decoder.is_calm_token(256016)); // BranchIf
        assert!(!decoder.is_calm_token(255999)); // Not CALM
        assert!(!decoder.is_calm_token(256017)); // Past range

        let program = decoder.decode(&[256000, 255999, 256003]); // Halt, skip, Add
        assert_eq!(program.len(), 2); // Non-CALM token skipped
    }

    #[test]
    fn test_online_distillation_trigger() {
        let mut trigger = OnlineDistillationTrigger::new(0.3, 3);
        assert!(!trigger.is_distilling());

        // High confidence — no trigger
        assert!(!trigger.record_prediction(0.9));
        assert!(!trigger.record_prediction(0.8));

        // Low confidence — accumulates
        assert!(!trigger.record_prediction(0.1));
        assert!(!trigger.record_prediction(0.2));
        assert!(trigger.record_prediction(0.15)); // 3rd low → triggers
        assert!(trigger.is_distilling());

        // Complete
        trigger.distillation_complete();
        assert!(!trigger.is_distilling());
    }
}
