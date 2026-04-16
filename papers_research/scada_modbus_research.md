# SCADA Research — Modbus Encoder, PLC Action Head, Industrial Safety Oracle

## Overview
Industrial control system (SCADA) integration:
- Modbus TCP/RTU protocol
- OPC-UA binary protocol
- PLC action control
- Functional safety (IEC 61508)

## Modbus Protocol

### Function Codes
| Code | Name | Description |
|------|------|-------------|
| 0x01 | Read Coils | Read 1-2000 coil states |
| 0x02 | Read Discrete Inputs | Read 1-2000 input states |
| 0x03 | Read Holding Registers | Read 1-125 holding registers |
| 0x04 | Read Input Registers | Read 1-125 input registers |
| 0x05 | Write Single Coil | Force single coil on/off |
| 0x06 | Write Single Register | Preset single register |
| 0x0F | Write Multiple Coils | Force multiple coils |
| 0x10 | Write Multiple Registers | Preset multiple registers |

### Register Map Example
```
Address  | Register        | Type   | Description
---------|-----------------|--------|--------------
0-99    | System          | R/W    | Configuration
100-199 | AI (0-10V)      | R      | Analog inputs
200-299 | AO (0-10V)      | R/W    | Analog outputs
300-399 | DI              | R      | Digital inputs
400-499 | DO              | R/W    | Digital outputs
500-599 | Status          | R      | Runtime status
```

### Modbus TCP Frame
```rust
pub struct ModbusTcpFrame {
    pub transaction_id: u16,
    pub protocol_id: u16,
    pub unit_id: u8,
    pub function_code: u8,
    pub data: Vec<u8>,
}
```

## ModbusEncoder

```rust
pub struct ModbusEncoder {
    slave_id: u8,
    register_map: RegisterMap,
}

pub struct RegisterMap {
    pub analog_inputs: HashMap<u16, Register>,
    pub analog_outputs: HashMap<u16, Register>,
    pub digital_inputs: HashMap<u16, Register>,
    pub digital_outputs: HashMap<u16, Register>,
}

impl StreamEncoder for ModbusEncoder {
    fn encode(&self, frame: &ModbusTcpFrame) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        tokens.push(frame.function_code as u32);
        
        // Parse register values
        match frame.function_code {
            0x03 | 0x04 => {
                // Read holding/input registers
                let values = parse_registers(&frame.data);
                for val in values.iter().take(16) {
                    tokens.push(*val);
                }
            }
            0x01 | 0x02 => {
                // Read coils/discrete inputs
                let bits = parse_bits(&frame.data);
                for bit in bits.iter().take(32) {
                    tokens.push(*bit);
                }
            }
            _ => {}
        }
        
        tokens
    }
}
```

## OPC-UA Binary Protocol

### Node/Attribute Model
```rust
pub struct OpcUaNode {
    node_id: String,
    node_class: NodeClass,
    attributes: HashMap<AttributeId, Variant>,
}

pub enum NodeClass {
    Object,
    Variable,
    Method,
    DataType,
}
```

## PLCActionHead

```rust
pub struct PLCActionHead {
    plc_address: String,
}

pub enum PlcAction {
    SetCoil { address: u16, value: bool },
    SetRegister { address: u16, value: u16 },
    WriteMultiple { start: u16, values: Vec<u16> },
}

impl ActionHead for PLCActionHead {
    async fn execute(&self, action: &PlcAction) -> Result<ActionResult> {
        match action {
            PlcAction::SetCoil { address, value } => {
                // Modbus function 0x05
            }
            PlcAction::SetRegister { address, value } => {
                // Modbus function 0x06
            }
            PlcAction::WriteMultiple { start, values } => {
                // Modbus function 0x10
            }
        }
    }
}
```

## Industrial Safety Oracle

### IEC 61508 Safety Levels (SIL 1-4)
| SIL | Required Safety | Target |
|-----|-----------------|--------|
| 1 | 10⁻⁵ - 10⁻⁴ | 90% |
| 2 | 10⁻⁶ - 10⁻⁵ | 99% |
| 3 | 10⁻⁷ - 10⁻⁶ | 99.9% |
| 4 | 10⁻⁸ - 10⁻⁷ | 99.99% |

### Valve + Pressure Cross-Check
```rust
pub struct SafetyValidator {
    valve_positions: HashMap<u16, ValveState>,
    pressure_sensors: HashMap<u16, f32>,
    max_pressure: f32,
}

impl ComplianceOracle for SafetyValidator {
    fn validate_action(&self, action: &PlcAction) -> ValidationResult {
        // Check if action would exceed pressure limits
        match action {
            PlcAction::SetRegister { address, value } => {
                if let Some(sensor) = self.pressure_sensors.get(address) {
                    if *sensor > self.max_pressure {
                        return ValidationResult::Violation(
                            format!("Pressure {} exceeds max {}", sensor, self.max_pressure)
                        );
                    }
                }
            }
            _ => {}
        }
        
        ValidationResult::Valid
    }
}
```

### Interlock Validation
```rust
pub fn validate_interlock(&self, actions: &[PlcAction]) -> bool {
    // Example: Cannot open inlet valve if outlet is closed
    let has_inlet_open = actions.iter().any(|a| 
        matches!(a, PlcAction::SetCoil { address: 100, value: true })
    );
    let has_outlet_open = actions.iter().any(|a|
        matches!(a, PlcAction::SetCoil { address: 101, value: true })
    );
    
    if has_inlet_open && !has_outlet_open {
        return false;  // Interlock violation
    }
    
    true
}
```

### Emergency Stop Logic
```rust
pub fn handle_emergency_stop(&self, estop_id: u16) -> Vec<PlcAction> {
    // Generate shutdown sequence
    vec![
        PlcAction::SetCoil { address: 200, value: false },  // Motor off
        PlcAction::SetCoil { address: 201, value: true },   // Brake on
        PlcAction::SetRegister { address: 300, value: 0 },   // Speed = 0
    ]
}
```

## References
- Modbus: https://modbus.org/
- OPC-UA: https://opcfoundation.org/
- IEC 61508: https://www.iec.ch/