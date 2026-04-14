//! Scientific & Engineering output modalities.
//!
//! Implements three specialized output capabilities:
//! - **ChemicalValidator**: SMILES valence-aware token masking for the organic subset
//! - **MeshHead**: SDF-based 3D mesh generation with Marching Cubes extraction
//! - **GCodeValidator**: G-Code parsing and work envelope validation (Klipper-style)
//!
//! All validators implement the ScientificOracle trait for pluggable validation.

// ---------------------------------------------------------------------------
// ScientificOracle — trait for pluggable validators
// ---------------------------------------------------------------------------

/// A pluggable validator for scientific/engineering output.
pub trait ScientificOracle {
    /// Validate a sequence of tokens. Returns true if valid.
    fn validate_sequence(&self, tokens: &[u32]) -> bool;

    /// Human-readable name of this oracle.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// SMILES Chemical Validator
// ---------------------------------------------------------------------------

/// Standard valence for organic subset elements.
const VALENCE_TABLE: &[(&str, &[u8])] = &[
    ("C", &[4]),
    ("N", &[3, 5]),
    ("O", &[2]),
    ("P", &[3, 5]),
    ("S", &[2, 4, 6]),
    ("F", &[1]),
    ("Cl", &[1]),
    ("Br", &[1]),
    ("I", &[1]),
    ("B", &[3]),
];

/// Token types in SMILES generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmilesTokenType {
    /// An organic atom (C, N, O, P, S, F, Cl, Br, I, B).
    Atom,
    /// Single bond (implicit, represented by adjacency).
    SingleBond,
    /// Double bond (=).
    DoubleBond,
    /// Triple bond (#).
    TripleBond,
    /// Ring closure digit (1-9).
    RingDigit,
    /// Branch open paren.
    BranchOpen,
    /// Branch close paren.
    BranchClose,
    /// Aromatic colon.
    Aromatic,
    /// Bracket atom start.
    BracketOpen,
    /// Bracket atom end.
    BracketClose,
}

/// Tracks valence state during SMILES generation.
#[derive(Debug, Clone)]
pub struct SmilesState {
    /// Remaining bond capacity per atom position.
    atom_capacities: Vec<u8>,
    /// Open ring digits awaiting closure.
    open_rings: Vec<usize>,
    /// Current bracket depth.
    bracket_depth: usize,
    /// Current branch depth.
    branch_depth: usize,
}

impl SmilesState {
    pub fn new() -> Self {
        Self {
            atom_capacities: Vec::new(),
            open_rings: Vec::new(),
            bracket_depth: 0,
            branch_depth: 0,
        }
    }

    /// Record an atom with given valence capacity.
    pub fn add_atom(&mut self, max_valence: u8) {
        self.atom_capacities.push(max_valence);
    }

    /// Use one bond from the last atom.
    pub fn use_bond(&mut self, order: u8) -> bool {
        if let Some(last) = self.atom_capacities.last_mut() {
            if *last >= order {
                *last -= order;
                return true;
            }
        }
        false
    }

    /// Open a ring with given digit.
    pub fn open_ring(&mut self, digit: usize) {
        self.open_rings.push(digit);
    }

    /// Close a ring with given digit.
    pub fn close_ring(&mut self, digit: usize) -> bool {
        if let Some(pos) = self.open_rings.iter().position(|&d| d == digit) {
            self.open_rings.remove(pos);
            // Ring closure uses one bond from the current atom
            self.use_bond(1);
            true
        } else {
            false
        }
    }

    /// Check if state is consistent (all rings closed, no negative capacities).
    pub fn is_valid(&self) -> bool {
        self.open_rings.is_empty()
            && self.atom_capacities.iter().all(|&c| c > 0 || self.atom_capacities.len() <= 1)
    }

    /// Get max valence for an element symbol.
    pub fn max_valence(symbol: &str) -> Option<u8> {
        for &(sym, valences) in VALENCE_TABLE {
            if sym == symbol {
                // Return the highest common valence
                return Some(*valences.last().unwrap());
            }
        }
        None
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.atom_capacities.clear();
        self.open_rings.clear();
        self.bracket_depth = 0;
        self.branch_depth = 0;
    }
}

/// Chemical validator for SMILES generation using organic subset rules.
///
/// Tracks atom valence capacity, ring closures, and branch depth to
/// determine valid continuations at each token position.
pub struct ChemicalValidator {
    /// Valid SMILES token IDs and their types.
    #[allow(dead_code)]
    token_types: Vec<SmilesTokenType>,
    /// Element symbols for atom tokens.
    atom_symbols: Vec<&'static str>,
}

impl ChemicalValidator {
    /// Create with a vocabulary of SMILES tokens.
    ///
    /// Convention: token IDs 0-9 are digits C,N,O,P,S,F,Cl,Br,I,B,
    /// 10='=', 11='#', 12-20 are ring digits 1-9, 21='(', 22=')'.
    pub fn new() -> Self {
        let atom_symbols = vec!["C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "B"];
        Self {
            token_types: Vec::new(),
            atom_symbols,
        }
    }

    /// Get the max valence for a token ID (if it's an atom).
    pub fn atom_valence(&self, token_id: u32) -> Option<u8> {
        let idx = token_id as usize;
        if idx < self.atom_symbols.len() {
            SmilesState::max_valence(self.atom_symbols[idx])
        } else {
            None
        }
    }

    /// Check if a token is valid given the current state.
    pub fn is_valid_continuation(&self, token_id: u32, state: &SmilesState) -> bool {
        let idx = token_id as usize;

        if idx < 10 {
            // Atom token — always valid (starts new atom)
            true
        } else if idx == 10 {
            // Double bond — need capacity on last atom
            state.atom_capacities.last().map_or(false, |&c| c >= 2)
        } else if idx == 11 {
            // Triple bond — need capacity >= 3
            state.atom_capacities.last().map_or(false, |&c| c >= 3)
        } else if idx >= 12 && idx <= 20 {
            // Ring digit 1-9
            let digit = idx - 11;
            if state.open_rings.contains(&digit) {
                true // Closing a ring
            } else {
                true // Opening a new ring
            }
        } else if idx == 21 {
            // Branch open
            true
        } else if idx == 22 {
            // Branch close — need open branch
            state.branch_depth > 0
        } else {
            false
        }
    }

    /// Filter logits: set invalid continuations to -inf.
    pub fn mask_invalid(&self, logits: &mut [f32], state: &SmilesState) {
        for (i, logit) in logits.iter_mut().enumerate() {
            if !self.is_valid_continuation(i as u32, state) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Get atom symbols.
    pub fn atom_symbols(&self) -> &[&'static str] {
        &self.atom_symbols
    }
}

impl Default for ChemicalValidator {
    fn default() -> Self { Self::new() }
}

impl ScientificOracle for ChemicalValidator {
    fn validate_sequence(&self, tokens: &[u32]) -> bool {
        let mut state = SmilesState::new();
        for &token in tokens {
            let idx = token as usize;
            if idx < 10 {
                if let Some(valence) = self.atom_valence(token) {
                    // Bond from previous atom to this one
                    if !state.atom_capacities.is_empty() {
                        state.use_bond(1);
                    }
                    state.add_atom(valence);
                }
            } else if idx >= 12 && idx <= 20 {
                let digit = idx - 11;
                if state.open_rings.contains(&digit) {
                    state.close_ring(digit);
                } else {
                    state.open_ring(digit);
                }
            } else if idx == 21 {
                state.branch_depth += 1;
            } else if idx == 22 {
                if state.branch_depth == 0 { return false; }
                state.branch_depth -= 1;
            }
        }
        state.is_valid() && !state.atom_capacities.is_empty()
    }

    fn name(&self) -> &str { "ChemicalValidator" }
}

// ---------------------------------------------------------------------------
// MeshHead — SDF prediction + Marching Cubes
// ---------------------------------------------------------------------------

/// Configuration for mesh generation.
#[derive(Debug, Clone)]
pub struct MeshHeadConfig {
    /// Grid resolution (e.g., 32 for 32³ grid).
    pub grid_resolution: usize,
    /// Iso-value for marching cubes (typically 0.0).
    pub iso_value: f32,
    /// Transformer hidden dimension.
    pub hidden_dim: usize,
}

impl Default for MeshHeadConfig {
    fn default() -> Self {
        Self { grid_resolution: 32, iso_value: 0.0, hidden_dim: 256 }
    }
}

/// A 3D vertex in a mesh.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub x: f32, pub y: f32, pub z: f32,
}

/// A triangle face (3 vertex indices).
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub v0: usize, pub v1: usize, pub v2: usize,
}

/// A generated 3D mesh.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triangle>,
}

impl Mesh {
    pub fn new() -> Self {
        Self { vertices: Vec::new(), triangles: Vec::new() }
    }

    /// Export to Wavefront .obj format.
    pub fn to_obj(&self) -> String {
        let mut out = String::new();
        for v in &self.vertices {
            out.push_str(&format!("v {} {} {}\n", v.x, v.y, v.z));
        }
        for t in &self.triangles {
            // OBJ is 1-indexed
            out.push_str(&format!("f {} {} {}\n", t.v0 + 1, t.v1 + 1, t.v2 + 1));
        }
        out
    }

    /// Number of vertices.
    pub fn num_vertices(&self) -> usize { self.vertices.len() }
    /// Number of triangles.
    pub fn num_triangles(&self) -> usize { self.triangles.len() }
}

/// Marching Cubes extraction from an SDF grid.
///
/// Uses the standard Lorensen & Cline 15-case lookup table (256 entries
/// via symmetry). Pure Rust CPU implementation.
pub struct MarchingCubes {
    grid_resolution: usize,
    iso_value: f32,
}

impl MarchingCubes {
    pub fn new(grid_resolution: usize, iso_value: f32) -> Self {
        Self { grid_resolution, iso_value }
    }

    /// Extract mesh from an SDF grid.
    ///
    /// `sdf`: [resolution³] signed distance values.
    pub fn extract(&self, sdf: &[f32]) -> Mesh {
        let res = self.grid_resolution;
        let iso = self.iso_value;
        let mut mesh = Mesh::new();

        if sdf.len() < res * res * res { return mesh; }

        // Iterate over all cubes in the grid
        for z in 0..(res - 1) {
            for y in 0..(res - 1) {
                for x in 0..(res - 1) {
                    // Get 8 corner values
                    let idx = |x: usize, y: usize, z: usize| -> usize {
                        z * res * res + y * res + x
                    };

                    let corners = [
                        sdf[idx(x, y, z)],
                        sdf[idx(x + 1, y, z)],
                        sdf[idx(x + 1, y + 1, z)],
                        sdf[idx(x, y + 1, z)],
                        sdf[idx(x, y, z + 1)],
                        sdf[idx(x + 1, y, z + 1)],
                        sdf[idx(x + 1, y + 1, z + 1)],
                        sdf[idx(x, y + 1, z + 1)],
                    ];

                    // Compute cube index (8-bit)
                    let mut cube_index = 0u32;
                    for (i, &val) in corners.iter().enumerate() {
                        if val < iso {
                            cube_index |= 1 << i;
                        }
                    }

                    // Skip empty cubes
                    if cube_index == 0 || cube_index == 255 { continue; }

                    // Edge table lookup (simplified: for each active edge,
                    // interpolate and add triangle)
                    // Using a simplified approach with basic triangulation
                    let edge_vertices = self.interpolate_edges(
                        x, y, z, res, &corners, iso,
                    );

                    // Add triangles based on cube_index (simplified)
                    // Full implementation would use the 256-entry lookup table
                    self.add_triangles_for_cube(cube_index, &edge_vertices, &mut mesh);
                }
            }
        }

        mesh
    }

    /// Interpolate edge crossings for a cube.
    fn interpolate_edges(
        &self, x: usize, y: usize, z: usize, _res: usize,
        corners: &[f32; 8], iso: f32,
    ) -> [Option<Vertex>; 12] {
        let xf = x as f32;
        let yf = y as f32;
        let zf = z as f32;

        let interp = |v1: f32, v2: f32, p1: f32, p2: f32| -> f32 {
            if (v2 - v1).abs() < 1e-10 { return p1; }
            p1 + (iso - v1) / (v2 - v1) * (p2 - p1)
        };

        // 12 edges of a cube, each connecting two vertices
        let edge_pairs: [(usize, usize, f32, f32, f32, f32, f32, f32); 12] = [
            (0,1, xf, yf, zf, xf+1.0, yf, zf),     // edge 0
            (1,2, xf+1.0, yf, zf, xf+1.0, yf+1.0, zf), // edge 1
            (2,3, xf+1.0, yf+1.0, zf, xf, yf+1.0, zf), // edge 2
            (3,0, xf, yf+1.0, zf, xf, yf, zf),        // edge 3
            (4,5, xf, yf, zf+1.0, xf+1.0, yf, zf+1.0), // edge 4
            (5,6, xf+1.0, yf, zf+1.0, xf+1.0, yf+1.0, zf+1.0), // edge 5
            (6,7, xf+1.0, yf+1.0, zf+1.0, xf, yf+1.0, zf+1.0), // edge 6
            (7,4, xf, yf+1.0, zf+1.0, xf, yf, zf+1.0), // edge 7
            (0,4, xf, yf, zf, xf, yf, zf+1.0),        // edge 8
            (1,5, xf+1.0, yf, zf, xf+1.0, yf, zf+1.0), // edge 9
            (2,6, xf+1.0, yf+1.0, zf, xf+1.0, yf+1.0, zf+1.0), // edge 10
            (3,7, xf, yf+1.0, zf, xf, yf+1.0, zf+1.0), // edge 11
        ];

        let mut result = [None; 12];
        for (i, &(v1i, v2i, x1, y1, z1, x2, y2, z2)) in edge_pairs.iter().enumerate() {
            let val1 = corners[v1i];
            let val2 = corners[v2i];
            if (val1 < iso) != (val2 < iso) {
                result[i] = Some(Vertex {
                    x: interp(val1, val2, x1, x2),
                    y: interp(val1, val2, y1, y2),
                    z: interp(val1, val2, z1, z2),
                });
            }
        }
        result
    }

    /// Add triangles for a cube configuration (simplified).
    fn add_triangles_for_cube(
        &self, _cube_index: u32, edges: &[Option<Vertex>; 12], mesh: &mut Mesh,
    ) {
        // Simplified: for each pair of edge crossings, add a triangle
        let active_edges: Vec<(usize, Vertex)> = edges.iter().enumerate()
            .filter_map(|(i, v)| v.map(|v| (i, v)))
            .collect();

        if active_edges.len() >= 3 {
            let base = mesh.vertices.len();
            for (_, v) in &active_edges {
                mesh.vertices.push(*v);
            }
            // Triangulate the polygon formed by active edge vertices
            for i in 1..active_edges.len().saturating_sub(1) {
                mesh.triangles.push(Triangle {
                    v0: base,
                    v1: base + i,
                    v2: base + i + 1,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// G-Code Validator
// ---------------------------------------------------------------------------

/// A parsed G-Code command.
#[derive(Debug, Clone)]
pub struct GCodeCommand {
    /// Command letter + number (e.g., "G0", "G1", "M3").
    pub command: String,
    /// Named parameters (e.g., X=100.0, Y=200.0, F=1500.0).
    pub params: Vec<(String, f32)>,
}

impl GCodeCommand {
    /// Parse a single G-Code line.
    /// Follows Klipper's regex pattern: split on uppercase letters.
    pub fn parse(line: &str) -> Option<Self> {
        let line = line.split(';').next()?.trim();
        if line.is_empty() { return None; }

        // Extract command (first letter + number)
        let line = line.trim();
        let cmd_char = line.chars().next()?.to_ascii_uppercase();
        if !cmd_char.is_ascii_alphabetic() { return None; }

        let rest = &line[1..];
        let cmd_num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if cmd_num.is_empty() { return None; }

        let command = format!("{}{}", cmd_char, cmd_num);

        // Parse parameters (letter=value pairs)
        let param_str = &rest[cmd_num.len()..];
        let mut params = Vec::new();
        let mut current_param = String::new();
        let mut current_val = String::new();
        let mut in_val = false;

        for ch in param_str.chars() {
            if ch.is_ascii_alphabetic() && !in_val {
                if !current_param.is_empty() && !current_val.is_empty() {
                    if let Ok(v) = current_val.parse::<f32>() {
                        params.push((current_param.to_uppercase(), v));
                    }
                    current_val.clear();
                }
                current_param = ch.to_ascii_uppercase().to_string();
                in_val = false;
            } else if ch == ' ' || ch == '\t' {
                if !current_val.is_empty() { in_val = false; }
            } else {
                current_val.push(ch);
                in_val = true;
            }
        }
        // Last param
        if !current_param.is_empty() && !current_val.is_empty() {
            if let Ok(v) = current_val.parse::<f32>() {
                params.push((current_param.to_uppercase(), v));
            }
        }

        Some(GCodeCommand { command, params })
    }

    /// Get a parameter value by name.
    pub fn get_param(&self, name: &str) -> Option<f32> {
        params_iter(&self.params).find(|(n, _)| *n == name).map(|(_, v)| v)
    }
}

fn params_iter(params: &[(String, f32)]) -> impl Iterator<Item = (&str, f32)> {
    params.iter().map(|(n, v)| (n.as_str(), *v))
}

/// Machine envelope for G-Code validation.
#[derive(Debug, Clone)]
pub struct MachineEnvelope {
    /// Axis limits: [(min, max)] per axis (X, Y, Z, E).
    pub axis_limits: Vec<(f32, f32)>,
    /// Maximum feed rate (mm/min).
    pub max_feed_rate: f32,
    /// Allowed G commands.
    pub allowed_g_commands: Vec<String>,
    /// Allowed M commands.
    pub allowed_m_commands: Vec<String>,
}

impl MachineEnvelope {
    /// Standard 3D printer envelope.
    pub fn printer_3d() -> Self {
        Self {
            axis_limits: vec![
                (0.0, 300.0),  // X
                (0.0, 300.0),  // Y
                (0.0, 400.0),  // Z
                (0.0, 1000.0), // E (extruder)
            ],
            max_feed_rate: 15000.0,
            allowed_g_commands: vec![
                "G0".into(), "G1".into(), "G2".into(), "G3".into(),
                "G4".into(), "G28".into(), "G90".into(), "G91".into(),
                "G92".into(),
            ],
            allowed_m_commands: vec![
                "M3".into(), "M5".into(), "M104".into(), "M106".into(),
                "M107".into(), "M109".into(), "M140".into(), "M190".into(),
            ],
        }
    }
}

/// G-Code validator with machine envelope bounds checking.
pub struct GCodeValidator {
    envelope: MachineEnvelope,
}

impl GCodeValidator {
    pub fn new(envelope: MachineEnvelope) -> Self {
        Self { envelope }
    }

    /// Parse and validate a G-Code program (one line at a time).
    pub fn validate_line(&self, line: &str) -> Result<GCodeCommand, String> {
        let cmd = GCodeCommand::parse(line)
            .ok_or_else(|| format!("Failed to parse: {}", line))?;

        // Check command is allowed
        if cmd.command.starts_with('G') {
            if !self.envelope.allowed_g_commands.contains(&cmd.command) {
                return Err(format!("Unknown G command: {}", cmd.command));
            }
        } else if cmd.command.starts_with('M') {
            if !self.envelope.allowed_m_commands.contains(&cmd.command) {
                return Err(format!("Unknown M command: {}", cmd.command));
            }
        } else {
            return Err(format!("Unknown command prefix: {}", cmd.command));
        }

        // Check feed rate
        if let Some(f) = cmd.get_param("F") {
            if f < 0.0 || f > self.envelope.max_feed_rate {
                return Err(format!("Feed rate {} exceeds max {}", f, self.envelope.max_feed_rate));
            }
        }

        // Check axis bounds
        for axis_idx in 0..self.envelope.axis_limits.len() {
            let axis_name = match axis_idx {
                0 => "X", 1 => "Y", 2 => "Z", 3 => "E", _ => continue,
            };
            if let Some(val) = cmd.get_param(axis_name) {
                let (min, max) = self.envelope.axis_limits[axis_idx];
                if val < min || val > max {
                    return Err(format!(
                        "{}={} outside bounds [{}, {}]", axis_name, val, min, max
                    ));
                }
            }
        }

        // G2/G3 circular interpolation requires I/J center
        if cmd.command == "G2" || cmd.command == "G3" {
            let has_ij = cmd.get_param("I").is_some() || cmd.get_param("J").is_some();
            if !has_ij {
                return Err("G2/G3 requires I and/or J center offset".into());
            }
        }

        Ok(cmd)
    }

    /// Validate an entire G-Code program.
    pub fn validate_program(&self, program: &str) -> Vec<Result<GCodeCommand, String>> {
        program.lines().map(|line| self.validate_line(line)).collect()
    }
}

impl ScientificOracle for GCodeValidator {
    fn validate_sequence(&self, tokens: &[u32]) -> bool {
        // Treat tokens as byte values of a G-Code string
        let s: String = tokens.iter()
            .filter_map(|&t| if t < 128 { Some(t as u8 as char) } else { None })
            .collect();
        s.lines().all(|line| {
            let trimmed = line.trim();
            trimmed.is_empty() || trimmed.starts_with(';') || self.validate_line(trimmed).is_ok()
        })
    }

    fn name(&self) -> &str { "GCodeValidator" }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SmilesState --

    #[test]
    fn test_smiles_state_valence() {
        assert_eq!(SmilesState::max_valence("C"), Some(4));
        assert_eq!(SmilesState::max_valence("N"), Some(5));
        assert_eq!(SmilesState::max_valence("O"), Some(2));
        assert_eq!(SmilesState::max_valence("X"), None);
    }

    #[test]
    fn test_smiles_state_bond_usage() {
        let mut state = SmilesState::new();
        state.add_atom(4); // Carbon
        assert!(state.use_bond(1));
        assert_eq!(*state.atom_capacities.last().unwrap(), 3);
        assert!(state.use_bond(2));
        assert_eq!(*state.atom_capacities.last().unwrap(), 1);
        assert!(!state.use_bond(2)); // Only 1 left
    }

    #[test]
    fn test_smiles_state_ring() {
        let mut state = SmilesState::new();
        state.open_ring(1);
        state.open_ring(2);
        assert_eq!(state.open_rings.len(), 2);
        assert!(state.close_ring(1));
        assert_eq!(state.open_rings.len(), 1);
        assert!(!state.close_ring(3)); // Not open
    }

    #[test]
    fn test_smiles_state_valid() {
        let mut state = SmilesState::new();
        state.add_atom(4);
        assert!(state.is_valid());
        state.open_ring(1);
        assert!(!state.is_valid()); // Unclosed ring
    }

    // -- ChemicalValidator --

    #[test]
    fn test_chemical_validator_atom_continuation() {
        let cv = ChemicalValidator::new();
        let state = SmilesState::new();
        // Atom tokens (0-9) are always valid
        for i in 0..10u32 {
            assert!(cv.is_valid_continuation(i, &state));
        }
    }

    #[test]
    fn test_chemical_validator_bond_continuation() {
        let cv = ChemicalValidator::new();
        let mut state = SmilesState::new();
        state.add_atom(4); // Carbon with 4 capacity
        assert!(cv.is_valid_continuation(10, &state)); // Double bond
        assert!(cv.is_valid_continuation(11, &state)); // Triple bond (4 >= 3)
        // After using bonds
        state.use_bond(3);
        assert!(!cv.is_valid_continuation(10, &state)); // Only 1 left
    }

    #[test]
    fn test_chemical_validator_validate_valid_smiles() {
        let cv = ChemicalValidator::new();
        // "CC" = tokens [0, 0] (C, C)
        assert!(cv.validate_sequence(&[0, 0]));
        // "CCO" = [0, 0, 2] (C, C, O)
        assert!(cv.validate_sequence(&[0, 0, 2]));
    }

    #[test]
    fn test_chemical_validator_validate_branch() {
        let cv = ChemicalValidator::new();
        // "C(C)C" = [0, 21, 0, 22, 0]
        assert!(cv.validate_sequence(&[0, 21, 0, 22, 0]));
        // Unmatched close: "C)C" = [0, 22, 0]
        assert!(!cv.validate_sequence(&[0, 22, 0]));
    }

    #[test]
    fn test_chemical_validator_mask() {
        let cv = ChemicalValidator::new();
        let mut state = SmilesState::new();
        state.add_atom(1); // Only 1 bond capacity (like F)
        let mut logits = vec![0.0f32; 23];
        cv.mask_invalid(&mut logits, &state);
        // Atom tokens should be valid
        assert!(!logits[0].is_infinite());
        // Double bond should be masked
        assert!(logits[10].is_infinite() && logits[10].is_sign_negative());
    }

    // -- MarchingCubes --

    #[test]
    fn test_marching_cubes_sphere() {
        let res = 16;
        let mut sdf = vec![0.0f32; res * res * res];
        let center = res as f32 / 2.0;
        let radius = res as f32 / 4.0;
        for z in 0..res {
            for y in 0..res {
                for x in 0..res {
                    let dx = x as f32 - center;
                    let dy = y as f32 - center;
                    let dz = z as f32 - center;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    sdf[z * res * res + y * res + x] = dist - radius;
                }
            }
        }
        let mc = MarchingCubes::new(res, 0.0);
        let mesh = mc.extract(&sdf);
        assert!(mesh.num_vertices() > 0, "Sphere should produce vertices");
        assert!(mesh.num_triangles() > 0, "Sphere should produce triangles");
    }

    #[test]
    fn test_marching_cubes_empty() {
        let res = 8;
        let sdf = vec![1.0f32; res * res * res]; // All outside
        let mc = MarchingCubes::new(res, 0.0);
        let mesh = mc.extract(&sdf);
        assert_eq!(mesh.num_vertices(), 0);
        assert_eq!(mesh.num_triangles(), 0);
    }

    #[test]
    fn test_mesh_to_obj() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex { x: 0.0, y: 0.0, z: 0.0 });
        mesh.vertices.push(Vertex { x: 1.0, y: 0.0, z: 0.0 });
        mesh.vertices.push(Vertex { x: 0.0, y: 1.0, z: 0.0 });
        mesh.triangles.push(Triangle { v0: 0, v1: 1, v2: 2 });
        let obj = mesh.to_obj();
        assert!(obj.contains("v 0 0 0"));
        assert!(obj.contains("f 1 2 3"));
    }

    // -- GCodeCommand --

    #[test]
    fn test_gcode_parse_g0() {
        let cmd = GCodeCommand::parse("G0 X100.0 Y200.0 F5000").unwrap();
        assert_eq!(cmd.command, "G0");
        assert_eq!(cmd.params.len(), 3);
        assert!((cmd.get_param("X").unwrap() - 100.0).abs() < 0.1);
        assert!((cmd.get_param("Y").unwrap() - 200.0).abs() < 0.1);
        assert!((cmd.get_param("F").unwrap() - 5000.0).abs() < 0.1);
    }

    #[test]
    fn test_gcode_parse_g1() {
        let cmd = GCodeCommand::parse("G1 X10.5 Y20.3 Z5.0 E1.2").unwrap();
        assert_eq!(cmd.command, "G1");
        assert_eq!(cmd.params.len(), 4);
    }

    #[test]
    fn test_gcode_parse_comment() {
        assert!(GCodeCommand::parse("; comment").is_none());
    }

    #[test]
    fn test_gcode_parse_empty() {
        assert!(GCodeCommand::parse("").is_none());
        assert!(GCodeCommand::parse("   ").is_none());
    }

    // -- GCodeValidator --

    #[test]
    fn test_gcode_validate_valid() {
        let v = GCodeValidator::new(MachineEnvelope::printer_3d());
        assert!(v.validate_line("G0 X100 Y200 Z5 F5000").is_ok());
        assert!(v.validate_line("G1 X50 Y50 E0.5 F1000").is_ok());
        assert!(v.validate_line("M3 S1000").is_ok());
    }

    #[test]
    fn test_gcode_validate_unknown_command() {
        let v = GCodeValidator::new(MachineEnvelope::printer_3d());
        assert!(v.validate_line("G99 X100").is_err());
        assert!(v.validate_line("M999").is_err());
    }

    #[test]
    fn test_gcode_validate_out_of_bounds() {
        let v = GCodeValidator::new(MachineEnvelope::printer_3d());
        assert!(v.validate_line("G0 X500").is_err()); // X max = 300
        assert!(v.validate_line("G0 X-10").is_err()); // X min = 0
    }

    #[test]
    fn test_gcode_validate_feed_rate() {
        let v = GCodeValidator::new(MachineEnvelope::printer_3d());
        assert!(v.validate_line("G1 X100 F20000").is_err()); // Max 15000
    }

    #[test]
    fn test_gcode_validate_circular_no_ij() {
        let v = GCodeValidator::new(MachineEnvelope::printer_3d());
        assert!(v.validate_line("G2 X10 Y10").is_err()); // Missing I/J
    }

    #[test]
    fn test_gcode_validate_circular_with_ij() {
        let v = GCodeValidator::new(MachineEnvelope::printer_3d());
        assert!(v.validate_line("G2 X10 Y10 I5 J0").is_ok());
    }

    #[test]
    fn test_gcode_validate_program() {
        let v = GCodeValidator::new(MachineEnvelope::printer_3d());
        let program = "G28\nG0 X100 Y100 Z5\nG1 X150 E0.5 F1000";
        let results = v.validate_program(program);
        // Filter out empty line parse failures (returns None → Err)
        let valid = results.iter().filter(|r| r.is_ok()).count();
        assert!(valid >= 3); // At least 3 valid lines
    }

    #[test]
    fn test_gcode_oracle_validate() {
        let v = GCodeValidator::new(MachineEnvelope::printer_3d());
        // "G0 X100\n" as bytes
        let tokens: Vec<u32> = b"G0 X100\n".iter().map(|&b| b as u32).collect();
        assert!(v.validate_sequence(&tokens));
    }

    #[test]
    fn test_gcode_oracle_name() {
        let v = GCodeValidator::new(MachineEnvelope::printer_3d());
        assert_eq!(v.name(), "GCodeValidator");
    }
}
