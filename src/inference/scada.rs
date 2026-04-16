//! SCADA — Modbus Encoder + PLCAction Head + Industrial Safety Oracle
//! 
//! Industrial control system integration:
//! - Modbus TCP/RTU protocol
//! - PLC action control
//! - Functional safety (IEC 61508)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum ScadaError {
    #[error("Modbus: {0}")]
    Modbus(String),
    
    #[error("PLC: {0}")]
    Plc(String),
    
    #[error("Safety: {0}")]
    Safety(String),
}

// ============================================================================
// Modbus Types
// ============================================================================

/// Modbus function code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModbusFunction {
    ReadCoils = 0x01,
    ReadDiscreteInputs = 0x02,
    ReadHoldingRegisters = 0x03,
    ReadInputRegisters = 0x04,
    WriteSingleCoil = 0x05,
    WriteSingleRegister = 0x06,
    WriteMultipleCoils = 0x0F,
    WriteMultipleRegisters = 0x10,
}

impl ModbusFunction {
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0x01 => Some(Self::ReadCoils),
            0x02 => Some(Self::ReadDiscreteInputs),
            0x03 => Some(Self::ReadHoldingRegisters),
            0x04 => Some(Self::ReadInputRegisters),
            0x05 => Some(Self::WriteSingleCoil),
            0x06 => Some(Self::WriteSingleRegister),
            0x0F => Some(Self::WriteMultipleCoils),
            0x10 => Some(Self::WriteMultipleRegisters),
            _ => None,
        }
    }
}

/// Modbus register map entry
#[derive(Debug, Clone)]
pub struct Register {
    pub address: u16,
    pub name: String,
    pub value: f32,
    pub unit: String,
    pub min: f32,
    pub max: f32,
}

/// Modbus TCP frame
#[derive(Debug, Clone)]
pub struct ModbusFrame {
    pub transaction_id: u16,
    pub protocol_id: u16,
    pub unit_id: u8,
    pub function: ModbusFunction,
    pub data: Vec<u8>,
}

// ============================================================================
// Modbus Encoder
// ============================================================================

/// Encodes Modbus registers/coils for Block AttnRes
pub struct ModbusEncoder {
    pub register_map: HashMap<u16, Register>,
    pub coil_states: HashMap<u16, bool>,
    pub holding_registers: HashMap<u16, f32>,
    pub input_registers: HashMap<u16, f32>,
}

impl ModbusEncoder {
    pub fn new() -> Self {
        Self {
            register_map: HashMap::new(),
            coil_states: HashMap::new(),
            holding_registers: HashMap::new(),
            input_registers: HashMap::new(),
        }
    }
    
    /// Add a register to the map
    pub fn add_register(&mut self, addr: u16, name: &str, unit: &str, min: f32, max: f32) {
        self.register_map.insert(addr, Register {
            address: addr,
            name: name.to_string(),
            value: 0.0,
            unit: unit.to_string(),
            min,
            max,
        });
    }
    
    /// Encode a Modbus frame to tokens
    pub fn encode_frame(&self, frame: &ModbusFrame) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Function code
        tokens.push(frame.function as u32 + 1000);
        
        // Parse data based on function
        match frame.function {
            ModbusFunction::ReadHoldingRegisters | ModbusFunction::ReadInputRegisters => {
                // Register values
                let mut pos = 0;
                while pos + 2 <= frame.data.len() {
                    let addr = u16::from_le_bytes([frame.data[pos], frame.data[pos + 1]]);
                    pos += 2;
                    let value = self.registers_to_value(addr);
                    tokens.push(quantize_value(value));
                }
            }
            ModbusFunction::ReadCoils | ModbusFunction::ReadDiscreteInputs => {
                // Bit states
                for (i, byte) in frame.data.iter().enumerate() {
                    for bit in 0..8 {
                        let state = (byte >> bit) & 1 == 1;
                        tokens.push(if state { 1 } else { 0 });
                    }
                }
            }
            _ => {}
        }
        
        tokens
    }
    
    /// Get register value by address
    fn registers_to_value(&self, addr: u16) -> f32 {
        self.holding_registers.get(&addr)
            .or_else(|| self.input_registers.get(&addr))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Quantize continuous value
fn quantize_value(v: f32) -> u32 {
    if v <= 0.0 { 0 }
    else if v < 1.0 { (v * 10.0) as u32 + 1 }
    else if v < 100.0 { (v.log10() * 10.0) as u32 + 11 }
    else { 31 }
}

// ============================================================================
// PLC Action Head
// ============================================================================

/// PLC output action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlcAction {
    SetCoil { address: u16, value: bool },
    SetRegister { address: u16, value: u16 },
    WriteMultiple { start: u16, values: Vec<u16> },
}

/// PLC action head - generates control signals from embeddings
pub struct PLCActionHead {
    pub output_coils: HashMap<u16, String>,
    pub output_registers: HashMap<u16, String>,
}

impl PLCActionHead {
    pub fn new() -> Self {
        Self {
            output_coils: HashMap::new(),
            output_registers: HashMap::new(),
        }
    }
    
    /// Map embedding to PLC action
    pub fn predict(&self, embeddings: &[f32]) -> Vec<PlcAction> {
        if embeddings.is_empty() {
            return vec![];
        }
        
        let mag = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut actions = Vec::new();
        
        // Simple threshold logic
        if mag > 5.0 {
            // High activity - enable outputs
            for (addr, name) in &self.output_coils {
                if name.contains("enable") {
                    actions.push(PlcAction::SetCoil {
                        address: *addr,
                        value: true,
                    });
                }
            }
        } else {
            // Low activity - disable
            for (addr, name) in &self.output_coils {
                if name.contains("enable") {
                    actions.push(PlcAction::SetCoil {
                        address: *addr,
                        value: false,
                    });
                }
            }
        }
        
        actions
    }
    
    /// Generate Modbus write frame
    pub fn to_modbus_frame(&self, action: &PlcAction) -> ModbusFrame {
        match action {
            PlcAction::SetCoil { address, value } => {
                let data = if *value {
                    vec![0xFF, 0x00]
                } else {
                    vec![0x00, 0x00]
                };
                ModbusFrame {
                    transaction_id: 1,
                    protocol_id: 0,
                    unit_id: 1,
                    function: ModbusFunction::WriteSingleCoil,
                    data,
                }
            }
            PlcAction::SetRegister { address, value } => {
                let data = value.to_le_bytes().to_vec();
                ModbusFrame {
                    transaction_id: 1,
                    protocol_id: 0,
                    unit_id: 1,
                    function: ModbusFunction::WriteSingleRegister,
                    data,
                }
            }
            PlcAction::WriteMultiple { start, values } => {
                let mut data = vec![];
                for v in values {
                    data.extend_from_slice(&v.to_le_bytes());
                }
                ModbusFrame {
                    transaction_id: 1,
                    protocol_id: 0,
                    unit_id: 1,
                    function: ModbusFunction::WriteMultipleRegisters,
                    data,
                }
            }
        }
    }
}

// ============================================================================
// Industrial Safety Oracle
// ============================================================================

/// IEC 61508 Safety Integrity Level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyLevel {
    Sil1,  // 90% reliability
    Sil2,  // 99%
    Sil3,  // 99.9%
    Sil4,  // 99.99%
}

/// Safety validation result
#[derive(Debug, Clone)]
pub enum SafetyResult {
    Valid,
    Warning(String),
    Violation(String),
    EmergencyStop(String),
}

/// Industrial safety oracle
pub struct IndustrialSafetyOracle {
    pub safety_level: SafetyLevel,
    pub max_pressure: f32,
    pub max_temperature: f32,
    pub protected_valves: Vec<u16>,
    pub interlocks: Vec<Interlock>,
}

#[derive(Debug, Clone)]
pub struct Interlock {
    pub trigger_address: u16,
    pub target_address: u16,
    pub required_state: bool,
}

impl IndustrialSafetyOracle {
    pub fn new() -> Self {
        Self {
            safety_level: SafetyLevel::Sil2,
            max_pressure: 100.0,  // bar
            max_temperature: 150.0, // °C
            protected_valves: vec![],
            interlocks: vec![],
        }
    }
    
    /// Add interlock
    pub fn add_interlock(&mut self, trigger: u16, target: u16, required: bool) {
        self.interlocks.push(Interlock {
            trigger_address: trigger,
            target_address: target,
            required_state: required,
        });
    }
    
    /// Validate action
    pub fn validate(&self, action: &PlcAction, sensors: &HashMap<u16, f32>) -> SafetyResult {
        match action {
            PlcAction::SetRegister { address, value } => {
                // Check pressure threshold
                if let Some(&pressure) = sensors.get(address) {
                    if pressure > self.max_pressure {
                        return SafetyResult::Violation(format!(
                            "Pressure {} exceeds max {} bar",
                            pressure, self.max_pressure
                        ));
                    }
                }
            }
            PlcAction::SetCoil { address, value } => {
                // Check valve protection
                if self.protected_valves.contains(address) && !*value {
                    return SafetyResult::Warning(format!(
                        "Closing protected valve at {}",
                        address
                    ));
                }
            }
            _ => {}
        }
        
        SafetyResult::Valid
    }
    
    /// Validate interlock
    pub fn validate_interlocks(&self, coils: &HashMap<u16, bool>) -> SafetyResult {
        for il in &self.interlocks {
            let trigger = coils.get(&il.trigger_address).copied().unwrap_or(false);
            let target = coils.get(&il.target_address).copied().unwrap_or(false);
            
            if trigger != il.required_state {
                return SafetyResult::EmergencyStop(format!(
                    "Interlock violated: trigger {} at {}, expected {}",
                    il.trigger_address, trigger, il.required_state
                ));
            }
        }
        
        SafetyResult::Valid
    }
    
    /// Generate emergency stop sequence
    pub fn emergency_stop(&self) -> Vec<PlcAction> {
        vec![
            PlcAction::SetCoil { address: 200, value: false },  // Motor off
            PlcAction::SetCoil { address: 201, value: true }, // Brake on
            PlcAction::SetRegister { address: 300, value: 0 },  // Speed = 0
        ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_modbus_encoder() {
        let encoder = ModbusEncoder::new();
        assert!(encoder.register_map.is_empty());
    }
    
    #[test]
    fn test_plc_action_head() {
        let head = PLCActionHead::new();
        let embeddings = vec![1.0, 2.0, 3.0];
        let actions = head.predict(&embeddings);
        assert!(!actions.is_empty());
    }
    
    #[test]
    fn test_safety_oracle() {
        let oracle = IndustrialSafetyOracle::new();
        assert_eq!(oracle.safety_level, SafetyLevel::Sil2);
    }
    
    #[test]
    fn test_safety_validation() {
        let oracle = IndustrialSafetyOracle::new();
        let mut sensors = HashMap::new();
        sensors.insert(100, 50.0);
        
        let action = PlcAction::SetRegister { address: 100, value: 50 };
        let result = oracle.validate(&action, &sensors);
        
        assert!(matches!(result, SafetyResult::Valid));
    }
    
    #[test]
    fn test_pressure_violation() {
        let oracle = IndustrialSafetyOracle::new();
        let mut sensors = HashMap::new();
        sensors.insert(100, 150.0);  // Above max
        
        let action = PlcAction::SetRegister { address: 100, value: 150 };
        let result = oracle.validate(&action, &sensors);
        
        assert!(matches!(result, SafetyResult::Violation(_)));
    }
    
    #[test]
    fn test_emergency_stop() {
        let oracle = IndustrialSafetyOracle::new();
        let actions = oracle.emergency_stop();
        
        assert_eq!(actions.len(), 3);
        assert!(matches!(actions[0], PlcAction::SetCoil { .. }));
    }
}