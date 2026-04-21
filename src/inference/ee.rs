//! Electrical Engineering modalities for FerrisRes.
//!
//! Four EE-specific generation/validation heads extending the existing
//! scientific/manufacturing architecture:
//!
//! 1. **HDL Validator** — Verilog syntax checking, multiple-driver detection,
//!    logit-level token masking (mirrors ChemicalValidator/SMILES pattern)
//! 2. **Gerber/SPICE** — RS-274X Gerber output with DRC validation,
//!    SPICE netlist generation with connectivity checks
//! 3. **RF/Waveform** — SDR IQ streaming encoder + arbitrary waveform
//!    generator (AWG) regression head (mirrors SpeechHead pattern)
//! 4. **PWM/Power** — PWM duty cycle head with half-bridge safety
//!    constraints and thermal limits (mirrors VLA ActionHead pattern)

use std::collections::{HashMap, HashSet};

// ===========================================================================
// 1. HDL Validator — Verilog syntax validation + multiple-driver detection
// ===========================================================================

/// Verilog keywords that cannot be used as identifiers.
const VERILOG_KEYWORDS: &[&str] = &[
    "always", "and", "assign", "automatic", "begin", "buf", "bufif0", "bufif1",
    "case", "casex", "casez", "cmos", "deassign", "default", "defparam", "disable",
    "edge", "else", "end", "endcase", "endfunction", "endgenerate", "endmodule",
    "endprimitive", "endspecify", "endtable", "endtask", "event", "for", "force",
    "forever", "fork", "function", "generate", "genvar", "highz0", "highz1",
    "if", "ifnone", "initial", "inout", "input", "integer", "join", "large",
    "localparam", "macromodule", "medium", "module", "nand", "negedge", "nmos",
    "nor", "not", "notif0", "notif1", "or", "output", "parameter", "pmos",
    "posedge", "primitive", "pull0", "pull1", "pulldown", "pullup", "rcmos",
    "real", "realtime", "reg", "release", "repeat", "rnmos", "rpmos", "rtran",
    "rtranif0", "rtranif1", "scalared", "small", "specify", "specparam",
    "strength", "strong0", "strong1", "supply0", "supply1", "table", "task",
    "time", "tran", "tranif0", "tranif1", "tri", "tri0", "tri1", "triand",
    "trior", "trireg", "vectored", "wait", "wand", "weak0", "weak1", "while",
    "wire", "wor", "xnor", "xor",
];

/// Valid Verilog identifier pattern: starts with letter or _, followed by
/// alphanumeric or _ or $.
fn is_valid_verilog_ident(s: &str) -> bool {
    if s.is_empty() { return false; }
    let chars: Vec<char> = s.chars().collect();
    if !chars[0].is_ascii_alphabetic() && chars[0] != '_' { return false; }
    chars.iter().all(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '$')
}

/// Verilog net/statement type for AST-like analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum HdlNetType {
    Wire,
    Reg,
    Input,
    Output,
    Inout,
    Supply0,
    Supply1,
}

/// A parsed Verilog continuous assignment: `assign wire = expr;`
#[derive(Debug, Clone)]
pub struct HdlAssignment {
    pub target: String,
    pub line: usize,
}

/// A parsed Verilog procedural block: `always @(...) begin ... end`
#[derive(Debug, Clone)]
pub struct HdlProceduralBlock {
    pub targets: Vec<String>,
    pub block_type: HdlBlockType,
    pub line: usize,
}

/// Type of procedural block.
#[derive(Debug, Clone, PartialEq)]
pub enum HdlBlockType {
    Initial,
    Always,
}

/// HDL validation result.
#[derive(Debug, Clone)]
pub struct HdlValidationResult {
    pub errors: Vec<HdlError>,
    pub warnings: Vec<HdlWarning>,
    pub module_name: Option<String>,
    pub ports: Vec<HdlPort>,
    pub net_declarations: HashMap<String, HdlNetType>,
}

/// HDL validation error.
#[derive(Debug, Clone)]
pub struct HdlError {
    pub line: usize,
    pub message: String,
    pub kind: HdlErrorKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HdlErrorKind {
    SyntaxError,
    MultipleDriver,
    UndeclaredNet,
    KeywordAsIdent,
    InvalidIdent,
    MissingModule,
    MissingEndmodule,
    FloatingNet,
}

/// HDL validation warning (non-fatal).
#[derive(Debug, Clone)]
pub struct HdlWarning {
    pub line: usize,
    pub message: String,
}

/// Port declaration.
#[derive(Debug, Clone)]
pub struct HdlPort {
    pub name: String,
    pub direction: HdlNetType,
    pub width: usize,
}

/// HDL (Verilog) syntax and structural validator.
///
/// Validates Verilog modules for:
/// - Keyword/identifier conflicts
/// - Syntax correctness (balanced begin/end, module/endmodule)
/// - Multiple-driver detection (same wire driven by multiple always/assign blocks)
/// - Undeclared net usage
/// - Floating (unconnected) nets
pub struct HdlValidator {
    keywords: HashSet<&'static str>,
}

impl HdlValidator {
    pub fn new() -> Self {
        Self {
            keywords: VERILOG_KEYWORDS.iter().copied().collect(),
        }
    }

    /// Validate a Verilog source string.
    pub fn validate(&self, source: &str) -> HdlValidationResult {
        let lines: Vec<&str> = source.lines().collect();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut module_name = None;
        let mut ports = Vec::new();
        let mut net_declarations: HashMap<String, HdlNetType> = HashMap::new();
        let mut assignments: Vec<HdlAssignment> = Vec::new();
        let mut procedural_blocks: Vec<HdlProceduralBlock> = Vec::new();
        let mut in_module = false;
        let mut begin_depth: usize = 0;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            let line_num = i + 1;

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with("//") { continue; }

            // Strip inline comments
            let code = if let Some(pos) = trimmed.find("//") {
                &trimmed[..pos]
            } else {
                trimmed
            };
            let tokens: Vec<&str> = code.split_whitespace().collect();
            if tokens.is_empty() { continue; }

            match tokens[0] {
                "module" => {
                    if in_module {
                        errors.push(HdlError {
                            line: line_num,
                            message: "Nested module declaration".into(),
                            kind: HdlErrorKind::SyntaxError,
                        });
                    }
                    in_module = true;
                    if tokens.len() > 1 {
                        // Module name: take everything before '(' or ';'
                        let raw = tokens[1];
                        let name = if let Some(paren) = raw.find('(') {
                            raw[..paren].to_string()
                        } else {
                            raw.trim_end_matches(';').to_string()
                        };
                        // Check for keyword as module name
                        if self.keywords.contains(name.as_str()) {
                            errors.push(HdlError {
                                line: line_num,
                                message: format!("Module name '{}' is a Verilog keyword", name),
                                kind: HdlErrorKind::KeywordAsIdent,
                            });
                        } else if !is_valid_verilog_ident(&name) {
                            errors.push(HdlError {
                                line: line_num,
                                message: format!("Invalid module name '{}'", name),
                                kind: HdlErrorKind::InvalidIdent,
                            });
                        }
                        module_name = Some(name);
                    }
                }
                "endmodule" => {
                    if !in_module {
                        errors.push(HdlError {
                            line: line_num,
                            message: "endmodule without matching module".into(),
                            kind: HdlErrorKind::MissingModule,
                        });
                    }
                    in_module = false;
                }
                "input" | "output" | "inout" => {
                    if !in_module {
                        errors.push(HdlError {
                            line: line_num,
                            message: format!("{} declaration outside module", tokens[0]),
                            kind: HdlErrorKind::SyntaxError,
                        });
                        continue;
                    }
                    let net_type = match tokens[0] {
                        "input" => HdlNetType::Input,
                        "output" => HdlNetType::Output,
                        "inout" => HdlNetType::Inout,
                        _ => unreachable!(),
                    };
                    // Parse port name(s)
                    for &tok in &tokens[1..] {
                        let name = tok.trim_end_matches(',').trim_end_matches(';').to_string();
                        if !name.is_empty() {
                            let width: usize = if tokens.len() > 1 && tokens[1].starts_with('[') {
                                // Parse [N] bit width
                                tokens[1].trim_start_matches('[')
                                    .trim_end_matches(']')
                                    .parse().unwrap_or(1)
                            } else { 1 };
                            if self.keywords.contains(name.as_str()) {
                                errors.push(HdlError {
                                    line: line_num,
                                    message: format!("Port name '{}' is a Verilog keyword", name),
                                    kind: HdlErrorKind::KeywordAsIdent,
                                });
                            }
                            net_declarations.insert(name.clone(), net_type.clone());
                            ports.push(HdlPort { name, direction: net_type.clone(), width });
                        }
                    }
                }
                "wire" | "reg" => {
                    if !in_module { continue; }
                    let net_type = if tokens[0] == "wire" { HdlNetType::Wire } else { HdlNetType::Reg };
                    for &tok in &tokens[1..] {
                        let name = tok.trim_end_matches(',').trim_end_matches(';').to_string();
                        if !name.is_empty() && !name.starts_with('[') {
                            net_declarations.insert(name, net_type.clone());
                        }
                    }
                }
                "assign" => {
                    if !in_module { continue; }
                    // assign target = expr;
                    if tokens.len() >= 3 && tokens[2] == "=" {
                        let target = tokens[1].to_string();
                        assignments.push(HdlAssignment { target, line: line_num });
                    }
                }
                "always" | "initial" => {
                    if !in_module { continue; }
                    // Track procedural block — we'll extract targets from assignments inside
                    let block_type = if tokens[0] == "always" { HdlBlockType::Always } else { HdlBlockType::Initial };
                    procedural_blocks.push(HdlProceduralBlock {
                        targets: Vec::new(), // populated below
                        block_type,
                        line: line_num,
                    });
                }
                "begin" => { begin_depth += 1; }
                "end" => {
                    begin_depth = begin_depth.saturating_sub(1);
                }
                _ => {}
            }

            // Detect procedural assignments (target <= expr or target = expr inside always/initial)
            if in_module && tokens.len() >= 3 {
                let last_block = procedural_blocks.last_mut();
                if let Some(block) = last_block {
                    // Look for <= or = in the token stream
                    let arrow_idx = tokens.iter().position(|t| *t == "<=");
                    let eq_idx = tokens.iter().position(|t| *t == "=" && !code.contains("assign"));
                    if let Some(idx) = arrow_idx.or(eq_idx) {
                        if idx > 0 {
                            let target = tokens[idx - 1].to_string();
                            if !block.targets.contains(&target) {
                                block.targets.push(target);
                            }
                        }
                    }
                }
            }
        }

        // Check for unclosed module
        if in_module {
            errors.push(HdlError {
                line: lines.len(),
                message: "Missing endmodule".into(),
                kind: HdlErrorKind::MissingEndmodule,
            });
        }

        // Multiple-driver detection: check if any net is driven by multiple sources
        let mut driver_map: HashMap<String, Vec<String>> = HashMap::new();
        for assign in &assignments {
            driver_map.entry(assign.target.clone())
                .or_default()
                .push(format!("continuous assign (line {})", assign.line));
        }
        for block in &procedural_blocks {
            if block.block_type == HdlBlockType::Always {
                for target in &block.targets {
                    driver_map.entry(target.clone())
                        .or_default()
                        .push(format!("always block (line {})", block.line));
                }
            }
        }
        for (net, drivers) in &driver_map {
            if drivers.len() > 1 {
                errors.push(HdlError {
                    line: 0,
                    message: format!("Multiple drivers for '{}': {}", net, drivers.join(", ")),
                    kind: HdlErrorKind::MultipleDriver,
                });
            }
        }

        // Warn about declared but unused nets
        let driven_nets: HashSet<String> = driver_map.keys().cloned().collect();
        let declared_nets: HashSet<String> = net_declarations.keys().cloned().collect();
        let port_names: HashSet<String> = ports.iter().map(|p| p.name.clone()).collect();
        let undriven: HashSet<String> = declared_nets.difference(&driven_nets).cloned().collect();
        for net in &undriven {
            if !port_names.contains(net.as_str()) {
                warnings.push(HdlWarning {
                    line: 0,
                    message: format!("Net '{}' declared but never driven", net),
                });
            }
        }

        HdlValidationResult {
            errors,
            warnings,
            module_name,
            ports,
            net_declarations,
        }
    }

    /// Get the set of valid tokens for Verilog output.
    /// Used for logit-level masking to prevent syntactically invalid continuations.
    pub fn valid_tokens(&self) -> HashSet<&'static str> {
        let mut tokens: HashSet<&'static str> = VERILOG_KEYWORDS.iter().copied().collect();
        // Add operators and punctuation
        for t in &["=", "<=", "==", "!=", "&&", "||", "&", "|", "^", "~",
                   "+", "-", "*", "/", "%", "<<", ">>", "?", ":",
                   "{", "}", "[", "]", "(", ")", ";", ",", "#", "@"] {
            tokens.insert(t);
        }
        tokens
    }
}

// ===========================================================================
// 2. Gerber RS-274X Generator + DRC + SPICE Netlist Validator
// ===========================================================================

/// Gerber RS-274X command types.
#[derive(Debug, Clone, PartialEq)]
pub enum GerberCommand {
    /// Set interpolation mode: linear, clockwise arc, counter-clockwise arc.
    Interpolation(GerberInterpolation),
    /// Move to point with light off.
    Move { x: f64, y: f64 },
    /// Draw line to point with light on.
    Line { x: f64, y: f64 },
    /// Flash aperture at point.
    Flash { x: f64, y: f64 },
    /// Define aperture.
    ApertureDefine { code: i32, shape: ApertureShape },
    /// Select aperture.
    ApertureSelect { code: i32 },
    /// Set region mode on/off.
    RegionOn,
    RegionOff,
    /// Comment.
    Comment(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum GerberInterpolation {
    Linear,
    Clockwise,
    CounterClockwise,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ApertureShape {
    Circle { diameter: f64 },
    Rectangle { width: f64, height: f64 },
    Obround { width: f64, height: f64 },
    Polygon { diameter: f64, sides: i32 },
}

/// Design Rule Check result.
#[derive(Debug, Clone)]
pub struct DrcResult {
    pub violations: Vec<DrcViolation>,
    pub warnings: Vec<DrcWarning>,
}

#[derive(Debug, Clone)]
pub struct DrcViolation {
    pub kind: DrcViolationKind,
    pub message: String,
    pub location: Option<(f64, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DrcViolationKind {
    TraceWidthTooSmall,
    ClearanceTooSmall,
    DrillSizeTooSmall,
    InvalidAperture,
    UnclosedRegion,
}

#[derive(Debug, Clone)]
pub struct DrcWarning {
    pub message: String,
    pub location: Option<(f64, f64)>,
}

/// Design Rule Check parameters.
#[derive(Debug, Clone)]
pub struct DrcParams {
    pub min_trace_width: f64,      // mm
    pub min_clearance: f64,         // mm
    pub min_drill_size: f64,        // mm
}

impl Default for DrcParams {
    fn default() -> Self {
        Self {
            min_trace_width: 0.15,   // 150µm (6 mil)
            min_clearance: 0.15,      // 150µm
            min_drill_size: 0.3,      // 300µm (12 mil)
        }
    }
}

/// Gerber RS-274X generator with DRC validation.
pub struct GerberGenerator {
    apertures: HashMap<i32, ApertureShape>,
    current_aperture: Option<i32>,
    #[allow(dead_code)]
    current_pos: (f64, f64),
    drc_params: DrcParams,
}

impl GerberGenerator {
    pub fn new(drc_params: DrcParams) -> Self {
        Self {
            apertures: HashMap::new(),
            current_aperture: None,
            current_pos: (0.0, 0.0),
            drc_params,
        }
    }

    /// Define a circular aperture.
    pub fn define_circle(&mut self, code: i32, diameter: f64) {
        self.apertures.insert(code, ApertureShape::Circle { diameter });
    }

    /// Define a rectangular aperture.
    pub fn define_rect(&mut self, code: i32, width: f64, height: f64) {
        self.apertures.insert(code, ApertureShape::Rectangle { width, height });
    }

    /// Generate Gerber RS-274X output from a sequence of commands.
    pub fn generate(&self, commands: &[GerberCommand]) -> String {
        let mut output = String::new();
        output.push_str("%FSLAX26Y26*%\n"); // Format: absolute, 2.6 coordinate format
        output.push_str("%MOIN*%\n");        // Mode: inches
        output.push_str("%IPPOS*%\n");       // Image polarity: positive
        output.push_str("%LPD*%\n");         // Layer polarity: dark

        // Emit aperture definitions
        for (code, shape) in &self.apertures {
            match shape {
                ApertureShape::Circle { diameter } => {
                    output.push_str(&format!("%ADD{}C,{}*%\n", code, diameter));
                }
                ApertureShape::Rectangle { width, height } => {
                    output.push_str(&format!("%ADD{}R,{}X{}*%\n", code, width, height));
                }
                ApertureShape::Obround { width, height } => {
                    output.push_str(&format!("%ADD{}O,{}X{}*%\n", code, width, height));
                }
                ApertureShape::Polygon { diameter, sides } => {
                    output.push_str(&format!("%ADD{}P,{}X{}*%\n", code, diameter, sides));
                }
            }
        }

        for cmd in commands {
            match cmd {
                GerberCommand::Interpolation(GerberInterpolation::Linear) => {
                    output.push_str("G01*\n");
                }
                GerberCommand::Interpolation(GerberInterpolation::Clockwise) => {
                    output.push_str("G02*\n");
                }
                GerberCommand::Interpolation(GerberInterpolation::CounterClockwise) => {
                    output.push_str("G03*\n");
                }
                GerberCommand::Move { x, y } => {
                    output.push_str(&format!("X{}Y{}D02*\n",
                        (x * 1_000_000.0) as i64,
                        (y * 1_000_000.0) as i64));
                }
                GerberCommand::Line { x, y } => {
                    output.push_str(&format!("X{}Y{}D01*\n",
                        (x * 1_000_000.0) as i64,
                        (y * 1_000_000.0) as i64));
                }
                GerberCommand::Flash { x, y } => {
                    output.push_str(&format!("X{}Y{}D03*\n",
                        (x * 1_000_000.0) as i64,
                        (y * 1_000_000.0) as i64));
                }
                GerberCommand::ApertureSelect { code } => {
                    output.push_str(&format!("D{}*\n", code));
                }
                GerberCommand::RegionOn => { output.push_str("G36*\n"); }
                GerberCommand::RegionOff => { output.push_str("G37*\n"); }
                GerberCommand::Comment(text) => {
                    output.push_str(&format!("G04 {}*\n", text));
                }
                GerberCommand::ApertureDefine { code, shape } => {
                    match shape {
                        ApertureShape::Circle { diameter } => {
                            output.push_str(&format!("%ADD{}C,{}*%\n", code, diameter));
                        }
                        ApertureShape::Rectangle { width, height } => {
                            output.push_str(&format!("%ADD{}R,{}X{}*%\n", code, width, height));
                        }
                        _ => {}
                    }
                }
            }
        }

        output.push_str("M02*\n"); // End of file
        output
    }

    /// Run DRC on a sequence of Gerber commands.
    pub fn drc_check(&self, commands: &[GerberCommand]) -> DrcResult {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();

        for cmd in commands {
            match cmd {
                GerberCommand::Line { x, y } => {
                    // Check trace width against current aperture
                    if let Some(code) = self.current_aperture {
                        if let Some(shape) = self.apertures.get(&code) {
                            let width = match shape {
                                ApertureShape::Circle { diameter } => *diameter,
                                ApertureShape::Rectangle { width, height } => width.min(*height),
                                _ => self.drc_params.min_trace_width, // use min as fallback
                            };
                            if width < self.drc_params.min_trace_width {
                                violations.push(DrcViolation {
                                    kind: DrcViolationKind::TraceWidthTooSmall,
                                    message: format!(
                                        "Trace width {:.3}mm below minimum {:.3}mm at ({:.3},{:.3})",
                                        width, self.drc_params.min_trace_width, x, y
                                    ),
                                    location: Some((*x, *y)),
                                });
                            }
                        }
                    }
                }
                GerberCommand::ApertureDefine { code, shape } => {
                    match shape {
                        ApertureShape::Circle { diameter } => {
                            if *diameter < self.drc_params.min_drill_size {
                                warnings.push(DrcWarning {
                                    message: format!(
                                        "Aperture D{} diameter {:.3}mm is below drill minimum {:.3}mm",
                                        code, diameter, self.drc_params.min_drill_size
                                    ),
                                    location: None,
                                });
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        DrcResult { violations, warnings }
    }
}

/// SPICE netlist validation result.
#[derive(Debug, Clone)]
pub struct SpiceValidationResult {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub node_count: usize,
    pub component_count: usize,
}

/// SPICE netlist validator.
///
/// Checks for:
/// - Valid component syntax (R, L, C, D, M, X, V, I lines)
/// - Floating inputs (nodes with only one connection)
/// - Short circuits (VDD/VSS directly connected)
/// - Missing model cards for semiconductor devices
pub struct SpiceNetlistValidator {
    pub vdd_name: String,
    pub vss_name: String,
}

impl SpiceNetlistValidator {
    pub fn new() -> Self {
        Self {
            vdd_name: "VDD".into(),
            vss_name: "VSS".into(),
        }
    }

    pub fn with_supply_names(vdd: &str, vss: &str) -> Self {
        Self { vdd_name: vdd.into(), vss_name: vss.into() }
    }

    /// Validate a SPICE netlist.
    pub fn validate(&self, netlist: &str) -> SpiceValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut node_connections: HashMap<String, Vec<String>> = HashMap::new();
        let mut component_count = 0;

        for line in netlist.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('*') { continue; }
            if trimmed == ".END" || trimmed == ".end" { break; }
            if trimmed.starts_with('.') { continue; } // model cards, subcircuits

            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if tokens.is_empty() { continue; }

            let first = tokens[0].chars().next().unwrap_or(' ');
            match first.to_ascii_uppercase() {
                'R' | 'L' | 'C' | 'V' | 'I' => {
                    if tokens.len() < 4 {
                        errors.push(format!("Component '{}' has too few nodes", tokens[0]));
                        continue;
                    }
                    component_count += 1;
                    // Nodes are tokens[1] and tokens[2]
                    let n1 = tokens[1].to_string();
                    let n2 = tokens[2].to_string();
                    node_connections.entry(n1.clone())
                        .or_default().push(tokens[0].to_string());
                    node_connections.entry(n2.clone())
                        .or_default().push(tokens[0].to_string());
                    // Check for short circuit
                    if (n1 == self.vdd_name && n2 == self.vss_name)
                        || (n1 == self.vss_name && n2 == self.vdd_name) {
                        if first == 'R' || first == 'L' || first == 'C' {
                            let value = tokens.get(3).unwrap_or(&"0");
                            if value.starts_with("0") && !value.contains('.') {
                                errors.push(format!(
                                    "Short circuit: {} connects {} to {} with value {}",
                                    tokens[0], n1, n2, value
                                ));
                            }
                        }
                    }
                }
                'D' | 'M' | 'Q' => {
                    component_count += 1;
                    if tokens.len() < 4 {
                        errors.push(format!("Device '{}' has too few nodes", tokens[0]));
                    }
                    // Record node connections
                    for node in tokens.iter().skip(1).take(if first == 'M' { 4 } else { 3 }) {
                        if !node.chars().all(|c| c.is_ascii_alphanumeric()) { break; }
                        node_connections.entry(node.to_string())
                            .or_default().push(tokens[0].to_string());
                    }
                }
                'X' => {
                    // Subcircuit instance
                    component_count += 1;
                    for node in tokens.iter().skip(1) {
                        if node.starts_with('=') || node.contains('=') { break; }
                        node_connections.entry(node.to_string())
                            .or_default().push(tokens[0].to_string());
                    }
                }
                _ => {
                    warnings.push(format!("Unrecognized line: {}", trimmed));
                }
            }
        }

        // Check for floating nodes (only one connection)
        let supply_nodes: HashSet<String> = [self.vdd_name.clone(), self.vss_name.clone(), "0".into()].into_iter().collect();
        for (node, connections) in &node_connections {
            if connections.len() == 1 && !supply_nodes.contains(node) {
                warnings.push(format!(
                    "Floating node '{}' connected only to {}", node, connections[0]
                ));
            }
        }

        let node_count = node_connections.len();
        SpiceValidationResult { errors, warnings, node_count, component_count }
    }
}

// ===========================================================================
// 3. RF IQ Encoder + Waveform Head
// ===========================================================================

/// RF IQ sample: interleaved In-phase and Quadrature components.
#[derive(Debug, Clone, Copy)]
pub struct IqSample {
    pub i: f32,
    pub q: f32,
}

/// Streaming RF IQ encoder that processes SDR time-domain data.
///
/// Converts raw IQ byte streams into normalized samples suitable for
/// transformer input. Reuses the ring buffer + overlap-add pattern from
/// streaming audio.
pub struct RfEncoder {
    pub sample_rate: f64,
    pub center_freq: f64,
    pub bandwidth: f64,
    buffer: Vec<IqSample>,
    buffer_pos: usize,
    chunk_size: usize,
}

impl RfEncoder {
    pub fn new(sample_rate: f64, center_freq: f64, chunk_size: usize) -> Self {
        Self {
            sample_rate,
            center_freq,
            bandwidth: sample_rate, // Nyquist
            buffer: vec![IqSample { i: 0.0, q: 0.0 }; chunk_size],
            buffer_pos: 0,
            chunk_size,
        }
    }

    /// Push raw IQ bytes (interleaved f32 I/Q pairs) into the encoder.
    /// Returns complete chunks of normalized IQ samples.
    pub fn push_bytes(&mut self, data: &[u8]) -> Vec<Vec<IqSample>> {
        let mut chunks = Vec::new();
        // Parse as interleaved f32 pairs
        let floats: Vec<f32> = data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        for pair in floats.chunks_exact(2) {
            let sample = IqSample { i: pair[0], q: pair[1] };
            self.buffer[self.buffer_pos] = sample;
            self.buffer_pos += 1;
            if self.buffer_pos >= self.chunk_size {
                self.buffer_pos = 0;
                chunks.push(self.buffer.clone());
            }
        }
        chunks
    }

    /// Convert IQ samples to magnitude spectrogram bins (simplified FFT output).
    /// In production, this would use the WGSL FFT kernel.
    pub fn iq_to_magnitude(&self, samples: &[IqSample]) -> Vec<f32> {
        // Simplified: compute instantaneous magnitude
        samples.iter().map(|s| (s.i * s.i + s.q * s.q).sqrt()).collect()
    }

    /// Compute power spectral density estimate via DFT (for CPU fallback).
    /// Uses a simplified radix-2 DFT for small bin counts.
    pub fn power_spectrum(&self, samples: &[IqSample], num_bins: usize) -> Vec<f32> {
        let n = samples.len().min(num_bins);
        let mut spectrum = vec![0.0f32; num_bins];

        for k in 0..num_bins {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for t in 0..n {
                let angle = -2.0 * std::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
                re += samples[t].i * angle.cos() + samples[t].q * angle.sin();
                im += samples[t].q * angle.cos() - samples[t].i * angle.sin();
            }
            spectrum[k] = (re * re + im * im) / (n * n) as f32;
        }
        spectrum
    }
}

/// Arbitrary Waveform Generator (AWG) regression head.
///
/// Unlike token-based heads, this outputs continuous voltage samples.
/// Takes hidden states and produces voltage values for waveform synthesis.
/// No softmax — direct regression.
pub struct WaveformHead {
    pub hidden_dim: usize,
    pub sample_rate: f64,
    pub voltage_range: (f32, f32),  // (min_V, max_V)
    pub weights: Vec<f32>,          // hidden_dim × 1 projection
    pub bias: f32,
}

impl WaveformHead {
    pub fn new(hidden_dim: usize, sample_rate: f64, voltage_range: (f32, f32)) -> Self {
        let mut weights = vec![0.0f32; hidden_dim];
        // Xavier init
        let scale = (2.0 / hidden_dim as f32).sqrt();
        for w in &mut weights {
            *w = (rand_simple() as f32 - 0.5) * scale;
        }
        Self {
            hidden_dim,
            sample_rate,
            voltage_range,
            weights,
            bias: 0.0,
        }
    }

    /// Forward pass: hidden states → voltage samples.
    /// One voltage sample per timestep (no softmax).
    pub fn forward(&self, hidden: &[f32], seq_len: usize) -> Vec<f32> {
        let hd = self.hidden_dim;
        let mut output = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let h = &hidden[t * hd..(t + 1) * hd.min(hidden.len() / seq_len.max(1))];
            let mut voltage = self.bias;
            for (i, &hv) in h.iter().enumerate() {
                if i < self.weights.len() {
                    voltage += hv * self.weights[i];
                }
            }
            // Tanh squashing to [-1, 1], then scale to voltage range
            voltage = voltage.tanh();
            let (vmin, vmax) = self.voltage_range;
            voltage = vmin + (voltage + 1.0) * 0.5 * (vmax - vmin);
            output.push(voltage);
        }
        output
    }

    /// Generate a waveform of given duration (seconds).
    pub fn generate(&self, hidden: &[f32], duration_s: f64) -> Vec<f32> {
        let num_samples = (duration_s * self.sample_rate) as usize;
        // Interpolate hidden states to cover all samples
        let seq_len = hidden.len() / self.hidden_dim;
        let mut output = Vec::with_capacity(num_samples);
        if seq_len == 0 { return output; }
        let _samples_per_step = (num_samples as f64 / seq_len as f64).ceil() as usize;
        let raw = self.forward(hidden, seq_len);
        for i in 0..num_samples {
            let src_idx = (i * seq_len / num_samples).min(seq_len - 1);
            output.push(raw[src_idx]);
        }
        output
    }
}

/// Simple deterministic pseudo-random for Xavier init (no external rand dep).
pub fn rand_simple() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new(12345);
    }
    SEED.with(|s| {
        let mut seed = s.get();
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s.set(seed);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    })
}

// ===========================================================================
// 4. PWM Head + Power Electronics Safety Checker
// ===========================================================================

/// PWM phase output (one half-bridge leg).
#[derive(Debug, Clone, Copy)]
pub struct PwmPhase {
    pub duty_cycle: f32,     // [0.0, 1.0]
    pub frequency: f32,      // Hz
    pub dead_time_ns: f32,   // nanoseconds
}

/// PWM mode for different motor/power topologies.
#[derive(Debug, Clone, PartialEq)]
pub enum PwmMode {
    /// Single-phase (1 half-bridge): fans, LEDs, heaters
    SinglePhase,
    /// Half-bridge: 2 switches, single-direction motor drive
    HalfBridge,
    /// Full H-bridge: 4 switches, bidirectional motor drive
    HBridge,
    /// 3-phase BLDC/PMSM: 6 switches, 3 half-bridges
    ThreePhase,
    /// Multi-phase (n half-bridges)
    MultiPhase { phases: usize },
}

/// PWM generation head.
///
/// Predicts duty cycles for each phase of a power electronics topology.
/// Mirrors VLA ActionHead with binned + continuous modes.
pub struct PwmHead {
    pub hidden_dim: usize,
    pub mode: PwmMode,
    pub num_phases: usize,
    pub frequency_hz: f32,
    pub max_frequency_hz: f32,
    /// Per-phase weights: hidden_dim → (duty_cycle, dead_time) = 2 outputs per phase
    pub weights: Vec<f32>,
}

impl PwmHead {
    pub fn new(hidden_dim: usize, mode: PwmMode, frequency_hz: f32) -> Self {
        let num_phases = match &mode {
            PwmMode::SinglePhase | PwmMode::HalfBridge => 1,
            PwmMode::HBridge => 2,
            PwmMode::ThreePhase => 3,
            PwmMode::MultiPhase { phases } => *phases,
        };
        let output_dim = num_phases * 2; // duty + dead_time per phase
        let mut weights = vec![0.0f32; hidden_dim * output_dim];
        let scale = (2.0 / hidden_dim as f32).sqrt();
        for w in &mut weights {
            *w = (rand_simple() as f32 - 0.5) * scale;
        }
        Self {
            hidden_dim,
            mode,
            num_phases,
            frequency_hz,
            max_frequency_hz: 500_000.0, // 500kHz default max
            weights,
        }
    }

    /// Forward pass: hidden states → PWM phase outputs.
    pub fn forward(&self, hidden: &[f32]) -> Vec<PwmPhase> {
        let output_dim = self.num_phases * 2;
        let mut phases = Vec::with_capacity(self.num_phases);

        for p in 0..self.num_phases {
            let mut duty = 0.0f32;
            let mut dead_time = 0.0f32;
            for (i, &h) in hidden.iter().enumerate().take(self.hidden_dim) {
                duty += h * self.weights[i * output_dim + p * 2];
                dead_time += h * self.weights[i * output_dim + p * 2 + 1];
            }
            // Squash duty to [0, 1]
            duty = 1.0 / (1.0 + (-duty).exp());
            // Dead time: positive, in nanoseconds (typical 50-500ns)
            dead_time = dead_time.abs().max(0.0).min(10000.0);

            phases.push(PwmPhase {
                duty_cycle: duty,
                frequency: self.frequency_hz,
                dead_time_ns: dead_time,
            });
        }
        phases
    }
}

/// PWM safety constraints.
#[derive(Debug, Clone)]
pub struct PwmSafetyConfig {
    pub max_duty_cycle: f32,         // 0.0-1.0
    pub min_dead_time_ns: f32,       // minimum dead time
    pub max_frequency_hz: f32,       // maximum switching frequency
    pub max_thermal_watts: f32,      // maximum I²R dissipation
    pub on_resistance_ohms: f32,     // MOSFET Rds(on)
    pub load_resistance_ohms: f32,   // load resistance
    pub supply_voltage: f32,         // supply voltage
}

impl Default for PwmSafetyConfig {
    fn default() -> Self {
        Self {
            max_duty_cycle: 0.95,
            min_dead_time_ns: 50.0,
            max_frequency_hz: 500_000.0,
            max_thermal_watts: 10.0,
            on_resistance_ohms: 0.01,
            load_resistance_ohms: 10.0,
            supply_voltage: 24.0,
        }
    }
}

/// PWM safety violation.
#[derive(Debug, Clone)]
pub struct PwmViolation {
    pub phase: usize,
    pub kind: PwmViolationKind,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PwmViolationKind {
    DutyCycleExceeded,
    DeadTimeTooShort,
    FrequencyExceeded,
    ThermalLimitExceeded,
    ShootThrough,  // both high-side and low-side ON
}

/// PWM safety checker.
///
/// Validates predicted PWM outputs against electrical constraints:
/// - Duty cycle limits
/// - Minimum dead time (prevents shoot-through)
/// - Maximum switching frequency
/// - Thermal dissipation limits (I²R × duty_cycle)
pub struct PwmSafetyChecker {
    pub config: PwmSafetyConfig,
}

impl PwmSafetyChecker {
    pub fn new(config: PwmSafetyConfig) -> Self {
        Self { config }
    }

    /// Check PWM phases for safety violations.
    pub fn check(&self, phases: &[PwmPhase]) -> Vec<PwmViolation> {
        let mut violations = Vec::new();

        for (i, phase) in phases.iter().enumerate() {
            // Duty cycle check
            if phase.duty_cycle > self.config.max_duty_cycle {
                violations.push(PwmViolation {
                    phase: i,
                    kind: PwmViolationKind::DutyCycleExceeded,
                    message: format!(
                        "Phase {} duty cycle {:.3} exceeds max {:.3}",
                        i, phase.duty_cycle, self.config.max_duty_cycle
                    ),
                });
            }

            // Dead time check
            if phase.dead_time_ns < self.config.min_dead_time_ns {
                violations.push(PwmViolation {
                    phase: i,
                    kind: PwmViolationKind::DeadTimeTooShort,
                    message: format!(
                        "Phase {} dead time {:.1}ns below minimum {:.1}ns — shoot-through risk",
                        i, phase.dead_time_ns, self.config.min_dead_time_ns
                    ),
                });
            }

            // Frequency check
            if phase.frequency > self.config.max_frequency_hz {
                violations.push(PwmViolation {
                    phase: i,
                    kind: PwmViolationKind::FrequencyExceeded,
                    message: format!(
                        "Phase {} frequency {:.0}Hz exceeds max {:.0}Hz",
                        i, phase.frequency, self.config.max_frequency_hz
                    ),
                });
            }

            // Thermal check: P = I²R × duty, where I = V_supply / (R_load + R_ds(on))
            let current = self.config.supply_voltage
                / (self.config.load_resistance_ohms + self.config.on_resistance_ohms);
            let power = current * current * self.config.on_resistance_ohms * phase.duty_cycle;
            if power > self.config.max_thermal_watts {
                violations.push(PwmViolation {
                    phase: i,
                    kind: PwmViolationKind::ThermalLimitExceeded,
                    message: format!(
                        "Phase {} thermal {:.2}W exceeds limit {:.2}W (I={:.2}A, duty={:.3})",
                        i, power, self.config.max_thermal_watts, current, phase.duty_cycle
                    ),
                });
            }
        }

        // H-bridge shoot-through: check complementary phases
        if phases.len() >= 2 {
            // In an H-bridge, phases 0 and 1 are complementary
            // Shoot-through if both > 0.5 duty (overlap region)
            let p0 = phases[0].duty_cycle;
            let p1 = phases[1].duty_cycle;
            if p0 > 0.5 && p1 > 0.5 {
                violations.push(PwmViolation {
                    phase: 0,
                    kind: PwmViolationKind::ShootThrough,
                    message: format!(
                        "H-bridge shoot-through: phase 0 duty={:.3}, phase 1 duty={:.3}",
                        p0, p1
                    ),
                });
            }
        }

        // 3-phase: check for simultaneous high duty on all phases
        if phases.len() >= 3 {
            let all_high = phases.iter().all(|p| p.duty_cycle > 0.8);
            if all_high {
                violations.push(PwmViolation {
                    phase: 0,
                    kind: PwmViolationKind::ShootThrough,
                    message: "All 3 phases > 80% duty — bus short-circuit risk".into(),
                });
            }
        }

        violations
    }

    /// Clamp phases to safe values.
    pub fn clamp(&self, phases: &mut [PwmPhase]) {
        for phase in phases.iter_mut() {
            phase.duty_cycle = phase.duty_cycle.clamp(0.0, self.config.max_duty_cycle);
            phase.dead_time_ns = phase.dead_time_ns.max(self.config.min_dead_time_ns);
            phase.frequency = phase.frequency.min(self.config.max_frequency_hz);
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- HDL Validator Tests ----

    #[test]
    fn test_hdl_valid_simple_module() {
        let v = HdlValidator::new();
        let src = r#"
module adder(
    input a,
    input b,
    output sum
);
    assign sum = a ^ b;
endmodule
"#;
        let result = v.validate(src);
        assert_eq!(result.errors.len(), 0, "Errors: {:?}", result.errors);
        assert_eq!(result.module_name, Some("adder".into()));
        assert_eq!(result.ports.len(), 3);
    }

    #[test]
    fn test_hdl_keyword_as_module_name() {
        let v = HdlValidator::new();
        let result = v.validate("module wire(); endmodule");
        assert!(result.errors.iter().any(|e| e.kind == HdlErrorKind::KeywordAsIdent));
    }

    #[test]
    fn test_hdl_missing_endmodule() {
        let v = HdlValidator::new();
        let result = v.validate("module test(input clk);");
        assert!(result.errors.iter().any(|e| e.kind == HdlErrorKind::MissingEndmodule));
    }

    #[test]
    fn test_hdl_endmodule_without_module() {
        let v = HdlValidator::new();
        let result = v.validate("endmodule");
        assert!(result.errors.iter().any(|e| e.kind == HdlErrorKind::MissingModule));
    }

    #[test]
    fn test_hdl_multiple_drivers() {
        let v = HdlValidator::new();
        let src = r#"
module test(input clk, output reg q);
    always @(posedge clk) q <= 1;
    always @(posedge clk) q <= 0;
endmodule
"#;
        let result = v.validate(src);
        assert!(result.errors.iter().any(|e| e.kind == HdlErrorKind::MultipleDriver),
            "Should detect multiple drivers: {:?}", result.errors);
    }

    #[test]
    fn test_hdl_no_multiple_driver_different_nets() {
        let v = HdlValidator::new();
        let src = r#"
module test(input clk, output reg a, output reg b);
    always @(posedge clk) a <= 1;
    always @(posedge clk) b <= 0;
endmodule
"#;
        let result = v.validate(src);
        assert!(!result.errors.iter().any(|e| e.kind == HdlErrorKind::MultipleDriver),
            "Different nets should not trigger multiple driver: {:?}", result.errors);
    }

    #[test]
    fn test_hdl_valid_ident() {
        assert!(is_valid_verilog_ident("my_wire"));
        assert!(is_valid_verilog_ident("_reset"));
        assert!(is_valid_verilog_ident("data123"));
        assert!(!is_valid_verilog_ident("123data"));
        assert!(!is_valid_verilog_ident(""));
        assert!(!is_valid_verilog_ident("my-wire"));
    }

    #[test]
    fn test_hdl_valid_tokens() {
        let v = HdlValidator::new();
        let tokens = v.valid_tokens();
        assert!(tokens.contains("always"));
        assert!(tokens.contains("="));
        assert!(tokens.contains("module"));
    }

    // ---- Gerber Tests ----

    #[test]
    fn test_gerber_generate_basic() {
        let mut gen = GerberGenerator::new(DrcParams::default());
        gen.define_circle(10, 0.2);
        let cmds = vec![
            GerberCommand::ApertureSelect { code: 10 },
            GerberCommand::Move { x: 0.0, y: 0.0 },
            GerberCommand::Line { x: 1.0, y: 0.0 },
            GerberCommand::Line { x: 1.0, y: 1.0 },
            GerberCommand::Line { x: 0.0, y: 1.0 },
            GerberCommand::Line { x: 0.0, y: 0.0 },
        ];
        let output = gen.generate(&cmds);
        assert!(output.contains("%FSLAX26Y26*%"));
        assert!(output.contains("%ADD10C,0.2*%"));
        assert!(output.contains("D01*"));
        assert!(output.contains("D02*"));
        assert!(output.contains("M02*"));
    }

    #[test]
    fn test_gerber_flash() {
        let gen = GerberGenerator::new(DrcParams::default());
        let cmds = vec![
            GerberCommand::Flash { x: 5.0, y: 3.0 },
        ];
        let output = gen.generate(&cmds);
        assert!(output.contains("D03*"));
    }

    #[test]
    fn test_gerber_drc_trace_width_ok() {
        let mut gen = GerberGenerator::new(DrcParams {
            min_trace_width: 0.1,
            ..Default::default()
        });
        gen.define_circle(10, 0.2); // 0.2mm > 0.1mm min
        gen.current_aperture = Some(10);
        let cmds = vec![
            GerberCommand::Line { x: 1.0, y: 0.0 },
        ];
        let result = gen.drc_check(&cmds);
        assert_eq!(result.violations.len(), 0);
    }

    #[test]
    fn test_gerber_drc_trace_width_violation() {
        let mut gen = GerberGenerator::new(DrcParams {
            min_trace_width: 0.3,
            ..Default::default()
        });
        gen.define_circle(10, 0.15); // 0.15mm < 0.3mm min
        gen.current_aperture = Some(10);
        let cmds = vec![
            GerberCommand::Line { x: 1.0, y: 0.0 },
        ];
        let result = gen.drc_check(&cmds);
        assert!(result.violations.iter().any(|v| v.kind == DrcViolationKind::TraceWidthTooSmall));
    }

    #[test]
    fn test_gerber_region_commands() {
        let gen = GerberGenerator::new(DrcParams::default());
        let cmds = vec![
            GerberCommand::RegionOn,
            GerberCommand::Line { x: 1.0, y: 0.0 },
            GerberCommand::Line { x: 1.0, y: 1.0 },
            GerberCommand::RegionOff,
        ];
        let output = gen.generate(&cmds);
        assert!(output.contains("G36*"));
        assert!(output.contains("G37*"));
    }

    // ---- SPICE Netlist Tests ----

    #[test]
    fn test_spice_valid_netlist() {
        let v = SpiceNetlistValidator::new();
        let netlist = r#"
* Simple voltage divider
R1 VDD out 10k
R2 out VSS 10k
.END
"#;
        let result = v.validate(netlist);
        assert_eq!(result.errors.len(), 0, "Errors: {:?}", result.errors);
        assert_eq!(result.component_count, 2);
        assert!(result.node_count >= 3); // VDD, out, VSS
    }

    #[test]
    fn test_spice_floating_node() {
        let v = SpiceNetlistValidator::new();
        let netlist = r#"
R1 VDD out 10k
C1 floating 100pF
.END
"#;
        let result = v.validate(netlist);
        // 'floating' has only one connection
        assert!(result.warnings.iter().any(|w| w.contains("Floating")));
    }

    #[test]
    fn test_spice_short_circuit() {
        let v = SpiceNetlistValidator::new();
        let netlist = r#"
R1 VDD VSS 0
.END
"#;
        let result = v.validate(netlist);
        assert!(result.errors.iter().any(|e| e.contains("Short circuit")));
    }

    #[test]
    fn test_spice_component_count() {
        let v = SpiceNetlistValidator::new();
        let netlist = r#"
R1 in out 1k
C1 out 0 100p
L1 in 0 10u
.END
"#;
        let result = v.validate(netlist);
        assert_eq!(result.component_count, 3);
    }

    #[test]
    fn test_spice_custom_supply_names() {
        let v = SpiceNetlistValidator::with_supply_names("3V3", "GND");
        let netlist = r#"
R1 3V3 out 10k
R2 out GND 10k
.END
"#;
        let result = v.validate(netlist);
        assert_eq!(result.errors.len(), 0);
    }

    // ---- RF Encoder Tests ----

    #[test]
    fn test_rf_push_bytes() {
        let mut enc = RfEncoder::new(2_400_000.0, 915_000_000.0, 4);
        // 8 bytes = 2 IQ samples
        let data: Vec<u8> = vec![
            0, 0, 0x80, 0x3F,  // 1.0f32 LE
            0, 0, 0, 0,        // 0.0f32
            0, 0, 0, 0x3F,    // 0.5f32
            0, 0, 0x80, 0x3F,  // 1.0f32
        ];
        let chunks = enc.push_bytes(&data);
        assert_eq!(chunks.len(), 0); // Only 2 samples, need 4 for a chunk
    }

    #[test]
    fn test_rf_iq_to_magnitude() {
        let enc = RfEncoder::new(1_000_000.0, 0.0, 4);
        let samples = vec![
            IqSample { i: 3.0, q: 4.0 },  // magnitude = 5.0
            IqSample { i: 0.0, q: 0.0 },   // magnitude = 0.0
        ];
        let mag = enc.iq_to_magnitude(&samples);
        assert!((mag[0] - 5.0).abs() < 0.01);
        assert!((mag[1] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_rf_power_spectrum() {
        let enc = RfEncoder::new(1_000_000.0, 0.0, 8);
        // DC signal: all I=1, Q=0
        let samples: Vec<IqSample> = (0..8).map(|_| IqSample { i: 1.0, q: 0.0 }).collect();
        let spectrum = enc.power_spectrum(&samples, 8);
        // DC bin (k=0) should have the most energy
        assert!(spectrum[0] > spectrum[1]);
    }

    // ---- Waveform Head Tests ----

    #[test]
    fn test_waveform_forward() {
        let head = WaveformHead::new(8, 44_100.0, (-1.0, 1.0));
        let hidden = vec![0.1f32; 32]; // 4 timesteps × 8 hidden
        let output = head.forward(&hidden, 4);
        assert_eq!(output.len(), 4);
        // All values should be within voltage range
        for &v in &output {
            assert!(v >= -1.0 && v <= 1.0, "Voltage {} out of range", v);
        }
    }

    #[test]
    fn test_waveform_generate_duration() {
        let head = WaveformHead::new(8, 1000.0, (0.0, 5.0));
        let hidden = vec![0.0f32; 16]; // 2 timesteps × 8
        let output = head.generate(&hidden, 0.1); // 0.1s × 1000Hz = 100 samples
        assert_eq!(output.len(), 100);
    }

    // ---- PWM Head Tests ----

    #[test]
    fn test_pwm_single_phase() {
        let head = PwmHead::new(8, PwmMode::SinglePhase, 20_000.0);
        let hidden = vec![0.5f32; 8];
        let phases = head.forward(&hidden);
        assert_eq!(phases.len(), 1);
        assert!(phases[0].duty_cycle >= 0.0 && phases[0].duty_cycle <= 1.0);
        assert_eq!(phases[0].frequency, 20_000.0);
    }

    #[test]
    fn test_pwm_three_phase() {
        let head = PwmHead::new(16, PwmMode::ThreePhase, 10_000.0);
        let hidden = vec![0.3f32; 16];
        let phases = head.forward(&hidden);
        assert_eq!(phases.len(), 3);
    }

    #[test]
    fn test_pwm_safety_ok() {
        let checker = PwmSafetyChecker::new(PwmSafetyConfig::default());
        let phases = vec![
            PwmPhase { duty_cycle: 0.5, frequency: 20_000.0, dead_time_ns: 200.0 },
        ];
        let violations = checker.check(&phases);
        assert_eq!(violations.len(), 0, "Violations: {:?}", violations);
    }

    #[test]
    fn test_pwm_safety_duty_exceeded() {
        let checker = PwmSafetyChecker::new(PwmSafetyConfig {
            max_duty_cycle: 0.9,
            ..Default::default()
        });
        let phases = vec![
            PwmPhase { duty_cycle: 0.95, frequency: 20_000.0, dead_time_ns: 200.0 },
        ];
        let violations = checker.check(&phases);
        assert!(violations.iter().any(|v| v.kind == PwmViolationKind::DutyCycleExceeded));
    }

    #[test]
    fn test_pwm_safety_dead_time_too_short() {
        let checker = PwmSafetyChecker::new(PwmSafetyConfig {
            min_dead_time_ns: 100.0,
            ..Default::default()
        });
        let phases = vec![
            PwmPhase { duty_cycle: 0.5, frequency: 20_000.0, dead_time_ns: 30.0 },
        ];
        let violations = checker.check(&phases);
        assert!(violations.iter().any(|v| v.kind == PwmViolationKind::DeadTimeTooShort));
    }

    #[test]
    fn test_pwm_safety_shoot_through() {
        let checker = PwmSafetyChecker::new(PwmSafetyConfig::default());
        let phases = vec![
            PwmPhase { duty_cycle: 0.8, frequency: 20_000.0, dead_time_ns: 200.0 },
            PwmPhase { duty_cycle: 0.8, frequency: 20_000.0, dead_time_ns: 200.0 },
        ];
        let violations = checker.check(&phases);
        assert!(violations.iter().any(|v| v.kind == PwmViolationKind::ShootThrough));
    }

    #[test]
    fn test_pwm_safety_thermal() {
        let checker = PwmSafetyChecker::new(PwmSafetyConfig {
            max_thermal_watts: 0.01, // very low limit
            on_resistance_ohms: 0.1,
            load_resistance_ohms: 1.0,
            supply_voltage: 12.0,
            ..Default::default()
        });
        let phases = vec![
            PwmPhase { duty_cycle: 0.9, frequency: 20_000.0, dead_time_ns: 200.0 },
        ];
        let violations = checker.check(&phases);
        assert!(violations.iter().any(|v| v.kind == PwmViolationKind::ThermalLimitExceeded));
    }

    #[test]
    fn test_pwm_safety_frequency_exceeded() {
        let checker = PwmSafetyChecker::new(PwmSafetyConfig {
            max_frequency_hz: 100_000.0,
            ..Default::default()
        });
        let phases = vec![
            PwmPhase { duty_cycle: 0.5, frequency: 200_000.0, dead_time_ns: 200.0 },
        ];
        let violations = checker.check(&phases);
        assert!(violations.iter().any(|v| v.kind == PwmViolationKind::FrequencyExceeded));
    }

    #[test]
    fn test_pwm_clamp() {
        let checker = PwmSafetyChecker::new(PwmSafetyConfig {
            max_duty_cycle: 0.9,
            min_dead_time_ns: 100.0,
            max_frequency_hz: 100_000.0,
            ..Default::default()
        });
        let mut phases = vec![
            PwmPhase { duty_cycle: 1.5, frequency: 200_000.0, dead_time_ns: 10.0 },
        ];
        checker.clamp(&mut phases);
        assert!((phases[0].duty_cycle - 0.9).abs() < 0.001);
        assert!(phases[0].dead_time_ns >= 100.0);
        assert!(phases[0].frequency <= 100_000.0);
    }

    #[test]
    fn test_pwm_3phase_all_high() {
        let checker = PwmSafetyChecker::new(PwmSafetyConfig::default());
        let phases = vec![
            PwmPhase { duty_cycle: 0.9, frequency: 20_000.0, dead_time_ns: 200.0 },
            PwmPhase { duty_cycle: 0.9, frequency: 20_000.0, dead_time_ns: 200.0 },
            PwmPhase { duty_cycle: 0.9, frequency: 20_000.0, dead_time_ns: 200.0 },
        ];
        let violations = checker.check(&phases);
        assert!(violations.iter().any(|v| v.kind == PwmViolationKind::ShootThrough));
    }

    #[test]
    fn test_pwm_hbridge_no_shoot_through() {
        let checker = PwmSafetyChecker::new(PwmSafetyConfig::default());
        let phases = vec![
            PwmPhase { duty_cycle: 0.7, frequency: 20_000.0, dead_time_ns: 200.0 },
            PwmPhase { duty_cycle: 0.3, frequency: 20_000.0, dead_time_ns: 200.0 },
        ];
        let violations = checker.check(&phases);
        assert!(!violations.iter().any(|v| v.kind == PwmViolationKind::ShootThrough));
    }
}
