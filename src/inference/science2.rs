//! Advanced scientific modalities for FerrisRes.
//!
//! Underserved-domain I/O pipelines leveraging O(n) Block AttnRes scaling:
//!
//! 1. **Bioinformatics** — FASTA nucleotide encoder, protein folding head,
//!    fold validator with Van der Waals collision detection
//! 2. **Cybersecurity** — PCAP stream encoder, eBPF firewall action head
//! 3. **Geospatial** — Multi-spectral GeoTIFF encoder, GeoJSON vector map head
//! 4. **Quantum** — OpenQASM 3.0 generator with topology validation

use std::collections::HashMap;

// ===========================================================================
// 1. Bioinformatics & Genomics
// ===========================================================================

/// DNA/RNA nucleotide bases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nucleotide {
    A, // Adenine
    C, // Cytosine
    G, // Guanine
    T, // Thymine (DNA)
    U, // Uracil (RNA)
    N, // Unknown/any
}

impl Nucleotide {
    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_uppercase() {
            'A' => Some(Nucleotide::A),
            'C' => Some(Nucleotide::C),
            'G' => Some(Nucleotide::G),
            'T' => Some(Nucleotide::T),
            'U' => Some(Nucleotide::U),
            'N' | '?' => Some(Nucleotide::N),
            _ => None,
        }
    }

    /// One-hot encoding for transformer input.
    pub fn to_onehot(&self) -> [f32; 6] {
        match self {
            Nucleotide::A => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            Nucleotide::C => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            Nucleotide::G => [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            Nucleotide::T => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            Nucleotide::U => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            Nucleotide::N => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// Amino acid types (20 standard + special).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AminoAcid {
    Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile,
    Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val,
    Start, Stop, Unknown,
}

impl AminoAcid {
    pub fn from_codon(a: Nucleotide, b: Nucleotide, c: Nucleotide) -> Self {
        // Simplified standard genetic code
        use Nucleotide::*;
        match (a, b, c) {
            (A, T, G) => AminoAcid::Met, // also Start
            (T, A, A) | (T, A, G) | (T, G, A) => AminoAcid::Stop,
            (G, C, T) => AminoAcid::Ala,
            (G, C, C) => AminoAcid::Ala,
            (G, C, A) => AminoAcid::Ala,
            (G, C, G) => AminoAcid::Ala,
            (T, T, T) | (T, T, C) => AminoAcid::Phe,
            (l, _, _) if matches!(l, Nucleotide::A) => AminoAcid::Leu, // placeholder for Leu codons
            _ => AminoAcid::Unknown,
        }
    }
}

/// FASTA sequence encoder.
///
/// Ingests raw FASTA format sequences into one-hot nucleotide embeddings
/// suitable for Block AttnRes. O(n) scaling means entire chromosomes
/// can be processed in a single forward pass.
pub struct NucleotideEncoder {
    pub is_rna: bool,
    pub kmer_size: usize,   // k-mer size for tokenization (default 1)
    pub embedding_dim: usize,
}

impl NucleotideEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        Self { is_rna: false, kmer_size: 1, embedding_dim }
    }

    pub fn with_rna(mut self) -> Self { self.is_rna = true; self }

    /// Parse a FASTA file and return sequences.
    pub fn parse_fasta(&self, fasta: &str) -> Vec<(String, Vec<Nucleotide>)> {
        let mut sequences = Vec::new();
        let mut current_name = String::new();
        let mut current_seq = Vec::new();

        for line in fasta.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('>') {
                if !current_seq.is_empty() {
                    sequences.push((current_name.clone(), current_seq.clone()));
                    current_seq.clear();
                }
                current_name = trimmed[1..].split_whitespace().next().unwrap_or("unknown").to_string();
            } else {
                for c in trimmed.chars() {
                    if let Some(nuc) = Nucleotide::from_char(c) {
                        current_seq.push(nuc);
                    }
                }
            }
        }
        if !current_seq.is_empty() {
            sequences.push((current_name, current_seq));
        }
        sequences
    }

    /// Encode nucleotides into one-hot embedding matrix.
    /// Output shape: [seq_len × embedding_dim].
    /// First 6 dims are one-hot nucleotide, rest are learned positional.
    pub fn encode(&self, nucleotides: &[Nucleotide]) -> Vec<f32> {
        let n = nucleotides.len();
        let dim = self.embedding_dim.max(6);
        let mut output = vec![0.0f32; n * dim];
        for (i, nuc) in nucleotides.iter().enumerate() {
            let onehot = nuc.to_onehot();
            for (j, &v) in onehot.iter().enumerate() {
                output[i * dim + j] = v;
            }
        }
        output
    }

    /// Count nucleotide frequencies.
    pub fn composition(&self, nucleotides: &[Nucleotide]) -> HashMap<Nucleotide, usize> {
        let mut counts = HashMap::new();
        for &nuc in nucleotides {
            *counts.entry(nuc).or_insert(0) += 1;
        }
        counts
    }
}

/// 3D coordinate for protein structure.
#[derive(Debug, Clone, Copy)]
pub struct Atom3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub element: &'static str,  // "C", "N", "O", "S"
    pub van_der_waals_radius: f64, // Angstroms
}

/// Protein torsion angles (radians).
#[derive(Debug, Clone, Copy)]
pub struct TorsionAngles {
    pub phi: f32,   // N-Cα bond rotation
    pub psi: f32,   // Cα-C bond rotation
    pub omega: f32, // peptide bond (usually ~180°)
}

/// Van der Waals radii for common protein atoms (Angstroms).
const VDW_RADII: &[(&str, f64)] = &[
    ("H", 1.20), ("C", 1.70), ("N", 1.55), ("O", 1.52), ("S", 1.80),
];

fn vdw_radius(element: &str) -> f64 {
    VDW_RADII.iter()
        .find(|(e, _)| *e == element)
        .map(|(_, r)| *r)
        .unwrap_or(1.70)
}

/// Protein folding output head.
///
/// Predicts torsion angles (phi, psi, omega) per residue from hidden states.
/// Continuous regression — no softmax.
pub struct ProteinFoldingHead {
    pub hidden_dim: usize,
    pub weights: Vec<f32>, // hidden_dim × 3 (phi, psi, omega)
    pub bias: [f32; 3],
}

impl ProteinFoldingHead {
    pub fn new(hidden_dim: usize) -> Self {
        let mut weights = vec![0.0f32; hidden_dim * 3];
        let scale = (2.0 / hidden_dim as f32).sqrt();
        for w in &mut weights {
            *w = (crate::inference::ee::rand_simple() as f32 - 0.5) * scale;
        }
        Self { hidden_dim, weights, bias: [0.0; 3] }
    }

    /// Forward pass: predict torsion angles for each residue.
    pub fn forward(&self, hidden: &[f32], seq_len: usize) -> Vec<TorsionAngles> {
        let hd = self.hidden_dim;
        let mut output = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let start = t * hd;
            let end = ((t + 1) * hd).min(hidden.len());
            let h = &hidden[start..end];
            let mut angles = [0.0f32; 3];
            for (k, angle) in angles.iter_mut().enumerate() {
                let mut val = self.bias[k];
                for (i, &hv) in h.iter().enumerate().take(hd) {
                    val += hv * self.weights[i * 3 + k];
                }
                // Tanh squashing to [-π, π]
                *angle = val.tanh() * std::f32::consts::PI;
            }
            output.push(TorsionAngles { phi: angles[0], psi: angles[1], omega: angles[2] });
        }
        output
    }

    /// Convert torsion angles to approximate 3D backbone coordinates.
    /// Uses idealized bond lengths: N-Cα=1.46Å, Cα-C=1.53Å, C-N=1.33Å.
    pub fn angles_to_coords(&self, angles: &[TorsionAngles]) -> Vec<Atom3D> {
        let mut atoms = Vec::new();
        let n_ca: f64 = 1.46;
        let ca_c: f64 = 1.53;
        let c_n: f64 = 1.33;

        // Simplified: place atoms along a chain using torsion angles
        let mut x = 0.0f64;
        let mut y = 0.0f64;
        let mut z = 0.0f64;
        for (_i, ang) in angles.iter().enumerate() {
            let d_phi = ang.phi as f64 * 0.1;
            let d_psi = ang.psi as f64 * 0.1;
            atoms.push(Atom3D { x, y, z, element: "N", van_der_waals_radius: vdw_radius("N") });
            x += n_ca * d_phi.cos();
            y += n_ca * d_phi.sin();
            atoms.push(Atom3D { x, y, z, element: "C", van_der_waals_radius: vdw_radius("C") });
            x += ca_c * d_psi.cos();
            z += ca_c * d_psi.sin();
            atoms.push(Atom3D { x, y, z, element: "C", van_der_waals_radius: vdw_radius("C") });
            x += c_n * 0.5;
        }
        atoms
    }
}

/// Fold validator: checks for atom collisions using Van der Waals radii.
pub struct FoldValidator {
    pub clash_distance_factor: f64, // multiplier on sum of VdW radii for clash
}

impl FoldValidator {
    pub fn new() -> Self {
        Self { clash_distance_factor: 0.75 }
    }

    /// Check for steric clashes in a 3D structure.
    /// Returns list of clashing atom pairs.
    pub fn check_clashes(&self, atoms: &[Atom3D]) -> Vec<(usize, usize, f64)> {
        let mut clashes = Vec::new();
        for i in 0..atoms.len() {
            for j in (i + 2)..atoms.len() { // skip bonded neighbors
                let dx = atoms[i].x - atoms[j].x;
                let dy = atoms[i].y - atoms[j].y;
                let dz = atoms[i].z - atoms[j].z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let min_dist = (atoms[i].van_der_waals_radius + atoms[j].van_der_waals_radius)
                    * self.clash_distance_factor;
                if dist < min_dist {
                    clashes.push((i, j, dist));
                }
            }
        }
        clashes
    }

    /// Check if a predicted fold is valid (no steric clashes).
    pub fn is_valid(&self, atoms: &[Atom3D]) -> bool {
        self.check_clashes(atoms).is_empty()
    }
}

// ===========================================================================
// 2. Cybersecurity & Packet-Level Streaming
// ===========================================================================

/// PCAP packet header (simplified).
#[derive(Debug, Clone)]
pub struct PacketHeader {
    pub src_mac: [u8; 6],
    pub dst_mac: [u8; 6],
    pub ethertype: u16,
    pub src_ip: Option<[u8; 4]>,
    pub dst_ip: Option<[u8; 4]>,
    pub protocol: u8,  // 6=TCP, 17=UDP, 1=ICMP
    pub src_port: Option<u16>,
    pub dst_port: Option<u16>,
    pub payload_len: usize,
}

/// PCAP stream encoder.
///
/// Ingests raw packet bytes into embeddings suitable for Block AttnRes.
/// O(n) scaling means sustained line-rate analysis of full packet streams.
pub struct PcapStreamEncoder {
    pub max_payload_bytes: usize,
    pub embedding_dim: usize,
}

impl PcapStreamEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        Self { max_payload_bytes: 1500, embedding_dim }
    }

    /// Parse a simplified Ethernet+IP+TCP/UDP packet from raw bytes.
    pub fn parse_packet(&self, data: &[u8]) -> Option<PacketHeader> {
        if data.len() < 14 { return None; } // Minimum Ethernet header

        let dst_mac = [data[0], data[1], data[2], data[3], data[4], data[5]];
        let src_mac = [data[6], data[7], data[8], data[9], data[10], data[11]];
        let ethertype = u16::from_be_bytes([data[12], data[13]]);

        let mut header = PacketHeader {
            src_mac, dst_mac, ethertype,
            src_ip: None, dst_ip: None,
            protocol: 0,
            src_port: None, dst_port: None,
            payload_len: 0,
        };

        // IPv4
        if ethertype == 0x0800 && data.len() >= 34 {
            let ip_start = 14;
            header.src_ip = Some([data[ip_start+12], data[ip_start+13], data[ip_start+14], data[ip_start+15]]);
            header.dst_ip = Some([data[ip_start+16], data[ip_start+17], data[ip_start+18], data[ip_start+19]]);
            header.protocol = data[ip_start + 9];
            let ihl = (data[ip_start] & 0x0F) as usize * 4;
            let transport_start = ip_start + ihl;
            if data.len() > transport_start + 4 {
                header.src_port = Some(u16::from_be_bytes([data[transport_start], data[transport_start + 1]]));
                header.dst_port = Some(u16::from_be_bytes([data[transport_start + 2], data[transport_start + 3]]));
            }
            let ip_total_len = u16::from_be_bytes([data[16], data[17]]) as usize;
            header.payload_len = ip_total_len.saturating_sub(ihl + 20); // rough
        }

        Some(header)
    }

    /// Encode raw packet bytes into embeddings.
    /// Each byte is embedded as a normalized value [0, 255] / 255.0,
    /// then projected to embedding_dim dimensions.
    pub fn encode_packet(&self, data: &[u8]) -> Vec<f32> {
        let len = data.len().min(self.max_payload_bytes);
        let dim = self.embedding_dim;
        let mut output = vec![0.0f32; len * dim];
        for (i, &byte) in data.iter().take(len).enumerate() {
            let normalized = byte as f32 / 255.0;
            // Simple: repeat normalized value across first dim, zeros for rest
            output[i * dim] = normalized;
            // Add position encoding
            if dim > 1 {
                let pos = (i as f32 * 0.01).sin();
                output[i * dim + 1] = pos;
            }
        }
        output
    }

    /// Format IP address as string.
    pub fn format_ip(ip: &[u8; 4]) -> String {
        format!("{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3])
    }

    /// Format MAC address as string.
    pub fn format_mac(mac: &[u8; 6]) -> String {
        format!("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5])
    }
}

/// eBPF instruction (simplified).
#[derive(Debug, Clone)]
pub struct EbpfInstruction {
    pub opcode: u8,
    pub dst_reg: u8,
    pub src_reg: u8,
    pub offset: i16,
    pub immediate: i32,
}

/// Firewall action: drop, allow, or redirect.
#[derive(Debug, Clone, PartialEq)]
pub enum FirewallAction {
    Allow,
    Drop,
    Redirect { target_ip: [u8; 4], target_port: u16 },
}

/// Firewall action head.
///
/// Predicts anomaly probability and firewall action from packet hidden states.
pub struct FirewallActionHead {
    pub hidden_dim: usize,
    pub weights: Vec<f32>, // hidden_dim × 3 (allow_prob, drop_prob, redirect_prob)
}

impl FirewallActionHead {
    pub fn new(hidden_dim: usize) -> Self {
        let mut weights = vec![0.0f32; hidden_dim * 3];
        let scale = (2.0 / hidden_dim as f32).sqrt();
        for w in &mut weights {
            *w = (crate::inference::ee::rand_simple() as f32 - 0.5) * scale;
        }
        Self { hidden_dim, weights }
    }

    /// Forward pass: predict firewall action.
    pub fn forward(&self, hidden: &[f32]) -> (FirewallAction, f32) {
        let mut logits = [0.0f32; 3];
        for (k, logit) in logits.iter_mut().enumerate() {
            for (i, &h) in hidden.iter().enumerate().take(self.hidden_dim) {
                *logit += h * self.weights[i * 3 + k];
            }
        }
        // Softmax
        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: [f32; 3] = logits.map(|l| (l - max_l).exp());
        let sum: f32 = exps.iter().sum();
        let probs: [f32; 3] = exps.map(|e| e / sum);

        let action = if probs[1] > probs[0] && probs[1] > probs[2] {
            FirewallAction::Drop
        } else if probs[2] > probs[0] && probs[2] > probs[1] {
            FirewallAction::Redirect { target_ip: [0, 0, 0, 0], target_port: 0 }
        } else {
            FirewallAction::Allow
        };
        let anomaly_score = probs[1]; // drop probability
        (action, anomaly_score)
    }
}

// ===========================================================================
// 3. Geospatial — Multi-Spectral Encoder + Vector Map Head
// ===========================================================================

/// GeoTIFF band descriptor.
#[derive(Debug, Clone)]
pub struct GeoBand {
    pub name: String,
    pub wavelength_nm: Option<f64>,   // for optical bands
    pub resolution_m: f64,             // ground sample distance in meters
    pub data: Vec<f32>,                // raster data, row-major
    pub width: usize,
    pub height: usize,
}

/// Multi-spectral satellite encoder.
///
/// Handles N-band satellite imagery (Sentinel-2 has 13 bands, Landsat has 11).
/// Each band is embedded separately then concatenated for cross-band attention.
pub struct GeoSpatialEncoder {
    pub embedding_dim: usize,
    pub band_count: usize,
    pub patch_size: usize, // patch extraction size (default 16×16)
}

impl GeoSpatialEncoder {
    pub fn new(band_count: usize, embedding_dim: usize) -> Self {
        Self { embedding_dim, band_count, patch_size: 16 }
    }

    /// Extract patches from a single band.
    /// Returns patches in row-major order, each of size patch_size × patch_size.
    pub fn extract_patches(&self, band: &GeoBand) -> Vec<Vec<f32>> {
        let ps = self.patch_size;
        let mut patches = Vec::new();
        for y in (0..band.height).step_by(ps) {
            for x in (0..band.width).step_by(ps) {
                let mut patch = Vec::with_capacity(ps * ps);
                for dy in 0..ps {
                    for dx in 0..ps {
                        let px = x + dx;
                        let py = y + dy;
                        let val = if px < band.width && py < band.height {
                            band.data.get(py * band.width + px).copied().unwrap_or(0.0)
                        } else { 0.0 };
                        patch.push(val);
                    }
                }
                patches.push(patch);
            }
        }
        patches
    }

    /// Compute NDVI (Normalized Difference Vegetation Index) from Red and NIR bands.
    /// NDVI = (NIR - Red) / (NIR + Red)
    pub fn ndvi(red: &[f32], nir: &[f32]) -> Vec<f32> {
        red.iter().zip(nir.iter())
            .map(|(&r, &n)| {
                let denom = n + r;
                if denom.abs() < 1e-6 { 0.0 } else { (n - r) / denom }
            })
            .collect()
    }

    /// Embed a multi-band patch into a fixed-size vector.
    pub fn embed_patch(&self, bands: &[Vec<f32>]) -> Vec<f32> {
        let ps = self.patch_size;
        let dim = self.embedding_dim;
        let mut embedding = vec![0.0f32; dim];
        for (b, band_patch) in bands.iter().enumerate() {
            let band_offset = (b * dim / self.band_count).min(dim - 1);
            for (i, &val) in band_patch.iter().enumerate().take(ps * ps) {
                let idx = (band_offset + i % (dim / self.band_count)).min(dim - 1);
                embedding[idx] += val / (self.band_count as f32);
            }
        }
        embedding
    }
}

/// GeoJSON geometry types.
#[derive(Debug, Clone)]
pub enum GeoJsonGeometry {
    Point { lat: f64, lon: f64 },
    LineString { coords: Vec<(f64, f64)> },
    Polygon { outer: Vec<(f64, f64)>, holes: Vec<Vec<(f64, f64)>> },
    BoundingBox { min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64 },
}

/// A detected feature in satellite imagery.
#[derive(Debug, Clone)]
pub struct GeoFeature {
    pub label: String,
    pub confidence: f32,
    pub geometry: GeoJsonGeometry,
}

/// Vector map head.
///
/// Predicts GeoJSON features (bounding boxes, polygons) from geospatial hidden states.
pub struct VectorMapHead {
    pub hidden_dim: usize,
    pub num_classes: usize,  // number of feature types (deforestation, crop, building, etc.)
}

impl VectorMapHead {
    pub fn new(hidden_dim: usize, num_classes: usize) -> Self {
        Self { hidden_dim, num_classes }
    }

    /// Generate GeoJSON from detected features.
    pub fn to_geojson(&self, features: &[GeoFeature]) -> String {
        let mut json = String::from("{\n  \"type\": \"FeatureCollection\",\n  \"features\": [\n");
        for (i, feat) in features.iter().enumerate() {
            if i > 0 { json.push_str(",\n"); }
            json.push_str("    {\n");
            json.push_str("      \"type\": \"Feature\",\n");
            json.push_str(&format!("      \"properties\": {{ \"label\": \"{}\", \"confidence\": {:.3} }},\n",
                feat.label, feat.confidence));
            json.push_str("      \"geometry\": ");
            match &feat.geometry {
                GeoJsonGeometry::Point { lat, lon } => {
                    json.push_str(&format!("{{ \"type\": \"Point\", \"coordinates\": [{}, {}] }}", lon, lat));
                }
                GeoJsonGeometry::BoundingBox { min_lat, min_lon, max_lat, max_lon } => {
                    // GeoJSON polygon: 5 coordinate pairs forming a rectangle
                    json.push_str(&format!(
                        "{{ \"type\": \"Polygon\", \"coordinates\": [[[{:.6},{:.6}],[{:.6},{:.6}],[{:.6},{:.6}],[{:.6},{:.6}],[{:.6},{:.6}]]] }}",
                        min_lon, min_lat,
                        max_lon, min_lat,
                        max_lon, max_lat,
                        min_lon, max_lat,
                        min_lon, min_lat
                    ));
                }
                GeoJsonGeometry::Polygon { outer, .. } => {
                    json.push_str("{ \"type\": \"Polygon\", \"coordinates\": [[");
                    for (j, (lon, lat)) in outer.iter().enumerate() {
                        if j > 0 { json.push_str(", "); }
                        json.push_str(&format!("[{}, {}]", lon, lat));
                    }
                    json.push_str("]] }");
                }
                GeoJsonGeometry::LineString { coords } => {
                    json.push_str("{ \"type\": \"LineString\", \"coordinates\": [");
                    for (j, (lon, lat)) in coords.iter().enumerate() {
                        if j > 0 { json.push_str(", "); }
                        json.push_str(&format!("[{}, {}]", lon, lat));
                    }
                    json.push_str("] }");
                }
            }
            json.push_str("\n    }");
        }
        json.push_str("\n  ]\n}");
        json
    }
}

// ===========================================================================
// 4. Quantum Circuit Synthesis
// ===========================================================================

/// Quantum gate types.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumGate {
    H { qubit: usize },          // Hadamard
    X { qubit: usize },          // Pauli-X (NOT)
    Y { qubit: usize },          // Pauli-Y
    Z { qubit: usize },          // Pauli-Z
    S { qubit: usize },          // S gate (π/2 phase)
    T { qubit: usize },          // T gate (π/4 phase)
    Rz { qubit: usize, angle: f64 }, // Z rotation
    Cx { control: usize, target: usize }, // CNOT
    Cz { control: usize, target: usize }, // CZ
    Swap { qubit_a: usize, qubit_b: usize },
    Measure { qubit: usize, bit: usize },
}

/// Quantum hardware topology.
#[derive(Debug, Clone)]
pub struct QuantumTopology {
    pub num_qubits: usize,
    pub coupling_map: Vec<(usize, usize)>, // which qubits can do 2-qubit gates
}

impl QuantumTopology {
    /// IBM Falcon r5 (27 qubits) topology.
    pub fn ibm_falcon_27() -> Self {
        Self {
            num_qubits: 27,
            coupling_map: vec![
                (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),
                (9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),
                (18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),
                // Cross-resonance connections
                (1,10),(3,12),(5,14),(7,16),(10,19),(12,21),(14,23),(16,25),
            ],
        }
    }

    /// Check if a 2-qubit gate is allowed by the coupling map.
    pub fn allows_coupling(&self, a: usize, b: usize) -> bool {
        self.coupling_map.contains(&(a, b)) || self.coupling_map.contains(&(b, a))
    }
}

/// OpenQASM 3.0 circuit generator.
pub struct OpenQasmGenerator {
    pub topology: QuantumTopology,
    pub gates: Vec<QuantumGate>,
}

impl OpenQasmGenerator {
    pub fn new(topology: QuantumTopology) -> Self {
        Self { topology, gates: Vec::new() }
    }

    /// Add a gate to the circuit.
    pub fn add_gate(&mut self, gate: QuantumGate) -> Result<(), String> {
        // Validate against topology
        match &gate {
            QuantumGate::Cx { control, target } |
            QuantumGate::Cz { control, target } => {
                if !self.topology.allows_coupling(*control, *target) {
                    return Err(format!(
                        "Coupling ({}, {}) not allowed by topology",
                        control, target
                    ));
                }
            }
            _ => {}
        }
        // Check qubit bounds
        match &gate {
            QuantumGate::H { qubit } | QuantumGate::X { qubit } |
            QuantumGate::Y { qubit } | QuantumGate::Z { qubit } |
            QuantumGate::S { qubit } | QuantumGate::T { qubit } |
            QuantumGate::Rz { qubit, .. } => {
                if *qubit >= self.topology.num_qubits {
                    return Err(format!("Qubit {} out of range (max {})", qubit, self.topology.num_qubits - 1));
                }
            }
            QuantumGate::Cx { control, target } |
            QuantumGate::Cz { control, target } => {
                if *control >= self.topology.num_qubits || *target >= self.topology.num_qubits {
                    return Err("Qubit out of range".into());
                }
            }
            _ => {}
        }
        self.gates.push(gate);
        Ok(())
    }

    /// Generate OpenQASM 3.0 code.
    pub fn generate(&self) -> String {
        let mut qasm = String::from("OPENQASM 3.0;\n");
        qasm.push_str("include 'stdgates.inc';\n\n");
        qasm.push_str(&format!("qubit[{}] q;\n", self.topology.num_qubits));
        qasm.push_str(&format!("bit[{}] c;\n\n", self.topology.num_qubits));

        for gate in &self.gates {
            match gate {
                QuantumGate::H { qubit } => qasm.push_str(&format!("h q[{}];\n", qubit)),
                QuantumGate::X { qubit } => qasm.push_str(&format!("x q[{}];\n", qubit)),
                QuantumGate::Y { qubit } => qasm.push_str(&format!("y q[{}];\n", qubit)),
                QuantumGate::Z { qubit } => qasm.push_str(&format!("z q[{}];\n", qubit)),
                QuantumGate::S { qubit } => qasm.push_str(&format!("s q[{}];\n", qubit)),
                QuantumGate::T { qubit } => qasm.push_str(&format!("t q[{}];\n", qubit)),
                QuantumGate::Rz { qubit, angle } => {
                    qasm.push_str(&format!("rz({}) q[{}];\n", angle, qubit));
                }
                QuantumGate::Cx { control, target } => {
                    qasm.push_str(&format!("cx q[{}], q[{}];\n", control, target));
                }
                QuantumGate::Cz { control, target } => {
                    qasm.push_str(&format!("cz q[{}], q[{}];\n", control, target));
                }
                QuantumGate::Swap { qubit_a, qubit_b } => {
                    qasm.push_str(&format!("swap q[{}], q[{}];\n", qubit_a, qubit_b));
                }
                QuantumGate::Measure { qubit, bit } => {
                    qasm.push_str(&format!("c[{}] = measure q[{}];\n", bit, qubit));
                }
            }
        }

        qasm
    }

    /// Validate an OpenQASM string for basic correctness.
    pub fn validate_qasm(qasm: &str) -> Vec<String> {
        let mut errors = Vec::new();
        if !qasm.starts_with("OPENQASM") {
            errors.push("Missing OPENQASM version header".into());
        }
        // Check for balanced brackets
        let mut depth: i32 = 0;
        for c in qasm.chars() {
            match c {
                '{' | '(' | '[' => depth += 1,
                '}' | ')' | ']' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        if depth != 0 {
            errors.push(format!("Unbalanced brackets (depth {})", depth));
        }
        errors
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Bioinformatics Tests ----

    #[test]
    fn test_nucleotide_from_char() {
        assert_eq!(Nucleotide::from_char('A'), Some(Nucleotide::A));
        assert_eq!(Nucleotide::from_char('t'), Some(Nucleotide::T));
        assert_eq!(Nucleotide::from_char('G'), Some(Nucleotide::G));
        assert_eq!(Nucleotide::from_char('x'), None);
    }

    #[test]
    fn test_nucleotide_onehot() {
        let oh = Nucleotide::A.to_onehot();
        assert_eq!(oh[0], 1.0);
        assert_eq!(oh[1], 0.0);
    }

    #[test]
    fn test_fasta_parser() {
        let fasta = ">seq1 test sequence\nACGT\nTGCA\n>seq2\nNNNN";
        let enc = NucleotideEncoder::new(32);
        let seqs = enc.parse_fasta(fasta);
        assert_eq!(seqs.len(), 2);
        assert_eq!(seqs[0].0, "seq1");
        assert_eq!(seqs[0].1.len(), 8); // ACGTTGCA
        assert_eq!(seqs[1].1.len(), 4);
    }

    #[test]
    fn test_nucleotide_encode() {
        let enc = NucleotideEncoder::new(32);
        let nucs = vec![Nucleotide::A, Nucleotide::C, Nucleotide::G, Nucleotide::T];
        let encoded = enc.encode(&nucs);
        assert_eq!(encoded.len(), 4 * 32);
        // First nucleotide (A): one-hot in first 6 dims
        assert_eq!(encoded[0], 1.0); // A
        assert_eq!(encoded[1], 0.0); // not C
    }

    #[test]
    fn test_nucleotide_composition() {
        let enc = NucleotideEncoder::new(32);
        let nucs = vec![Nucleotide::A, Nucleotide::A, Nucleotide::C, Nucleotide::G];
        let comp = enc.composition(&nucs);
        assert_eq!(comp[&Nucleotide::A], 2);
        assert_eq!(comp[&Nucleotide::C], 1);
    }

    #[test]
    fn test_protein_folding_forward() {
        let head = ProteinFoldingHead::new(16);
        let hidden = vec![0.5f32; 32]; // 2 residues × 16 hidden
        let angles = head.forward(&hidden, 2);
        assert_eq!(angles.len(), 2);
        // Angles should be in [-π, π]
        for a in &angles {
            assert!(a.phi >= -std::f32::consts::PI && a.phi <= std::f32::consts::PI);
            assert!(a.psi >= -std::f32::consts::PI && a.psi <= std::f32::consts::PI);
        }
    }

    #[test]
    fn test_fold_validator_no_clash() {
        let v = FoldValidator::new();
        let atoms = vec![
            Atom3D { x: 0.0, y: 0.0, z: 0.0, element: "C", van_der_waals_radius: 1.7 },
            Atom3D { x: 10.0, y: 0.0, z: 0.0, element: "C", van_der_waals_radius: 1.7 },
        ];
        assert!(v.is_valid(&atoms));
    }

    #[test]
    fn test_fold_validator_clash() {
        let v = FoldValidator::new();
        let atoms = vec![
            Atom3D { x: 0.0, y: 0.0, z: 0.0, element: "C", van_der_waals_radius: 1.7 },
            Atom3D { x: 1.0, y: 0.0, z: 0.0, element: "C", van_der_waals_radius: 1.7 },
            // skip index 2 (bonded)
            Atom3D { x: 1.5, y: 0.0, z: 0.0, element: "C", van_der_waals_radius: 1.7 },
        ];
        assert!(!v.is_valid(&atoms));
    }

    // ---- Cybersecurity Tests ----

    #[test]
    fn test_pcap_parse_ethernet() {
        let enc = PcapStreamEncoder::new(64);
        // Minimal Ethernet frame: 14 bytes header
        let mut packet = vec![0u8; 64];
        // Ethernet header
        packet[12] = 0x08; // IPv4
        packet[13] = 0x00;
        // IP header starts at byte 14
        packet[14] = 0x45; // version 4, IHL 5 (20 bytes)
        packet[14 + 9] = 6;   // TCP protocol
        // Source IP at offset 14+12
        packet[14 + 12] = 192;
        packet[14 + 13] = 168;
        packet[14 + 14] = 1;
        packet[14 + 15] = 1;
        // Dest IP at offset 14+16
        packet[14 + 16] = 10;
        packet[14 + 17] = 0;
        packet[14 + 18] = 0;
        packet[14 + 19] = 1;
        // Total length at offset 14+2
        packet[14 + 2] = 0;
        packet[14 + 3] = 50;

        let hdr = enc.parse_packet(&packet).unwrap();
        assert_eq!(hdr.ethertype, 0x0800);
        assert_eq!(hdr.protocol, 6);
        assert!(hdr.src_ip.is_some());
        assert_eq!(hdr.src_ip.unwrap()[0], 192);
    }

    #[test]
    fn test_pcap_encode() {
        let enc = PcapStreamEncoder::new(32);
        let data = vec![0xFF, 0x00, 0x80];
        let encoded = enc.encode_packet(&data);
        assert_eq!(encoded.len(), 3 * 32);
        assert!((encoded[0] - 1.0).abs() < 0.01); // 0xFF / 255 ≈ 1.0
        assert!(encoded[32] < 0.01); // 0x00 / 255 ≈ 0.0
    }

    #[test]
    fn test_format_ip() {
        assert_eq!(PcapStreamEncoder::format_ip(&[192, 168, 1, 1]), "192.168.1.1");
    }

    #[test]
    fn test_format_mac() {
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        assert_eq!(PcapStreamEncoder::format_mac(&mac), "aa:bb:cc:dd:ee:ff");
    }

    #[test]
    fn test_firewall_action_head() {
        let head = FirewallActionHead::new(16);
        let hidden = vec![0.1f32; 16];
        let (action, score) = head.forward(&hidden);
        // Just check it produces valid output
        assert!(score >= 0.0 && score <= 1.0);
        match action {
            FirewallAction::Allow | FirewallAction::Drop | FirewallAction::Redirect { .. } => {}
        }
    }

    // ---- Geospatial Tests ----

    #[test]
    fn test_geospatial_patch_extraction() {
        let enc = GeoSpatialEncoder::new(4, 32);
        let band = GeoBand {
            name: "B04 Red".into(),
            wavelength_nm: Some(665.0),
            resolution_m: 10.0,
            data: vec![1.0f32; 32 * 32],
            width: 32,
            height: 32,
        };
        let patches = enc.extract_patches(&band);
        assert_eq!(patches.len(), 4); // 2×2 grid of 16×16 patches
        assert_eq!(patches[0].len(), 16 * 16);
    }

    #[test]
    fn test_ndvi() {
        let red = vec![0.1, 0.2, 0.3];
        let nir = vec![0.5, 0.3, 0.1];
        let ndvi = GeoSpatialEncoder::ndvi(&red, &nir);
        assert!(ndvi[0] > 0.0); // NIR > Red → positive NDVI
        assert!(ndvi[1] > 0.0);
        // ndvi[2]: Red=0.3, NIR=0.1 → NDVI = (0.1-0.3)/(0.1+0.3) = -0.5
        assert!(ndvi[2] < 0.0);
    }

    #[test]
    fn test_geojson_point() {
        let head = VectorMapHead::new(32, 5);
        let features = vec![
            GeoFeature {
                label: "deforestation".into(),
                confidence: 0.92,
                geometry: GeoJsonGeometry::Point { lat: -3.4653, lon: -62.2159 },
            },
        ];
        let json = head.to_geojson(&features);
        assert!(json.contains("\"type\": \"FeatureCollection\""));
        assert!(json.contains("deforestation"));
        assert!(json.contains("-62.2159"));
    }

    #[test]
    fn test_geojson_bbox() {
        let head = VectorMapHead::new(32, 5);
        let features = vec![
            GeoFeature {
                label: "crop_wheat".into(),
                confidence: 0.85,
                geometry: GeoJsonGeometry::BoundingBox {
                    min_lat: 48.0, min_lon: 2.0, max_lat: 48.5, max_lon: 2.5,
                },
            },
        ];
        let json = head.to_geojson(&features);
        assert!(json.contains("Polygon"));
        assert!(json.contains("crop_wheat"));
    }

    // ---- Quantum Tests ----

    #[test]
    fn test_quantum_gate_basic() {
        let topo = QuantumTopology { num_qubits: 5, coupling_map: vec![(0, 1), (1, 2), (2, 3), (3, 4)] };
        let mut gen = OpenQasmGenerator::new(topo);
        gen.add_gate(QuantumGate::H { qubit: 0 }).unwrap();
        gen.add_gate(QuantumGate::Cx { control: 0, target: 1 }).unwrap();
        gen.add_gate(QuantumGate::Measure { qubit: 0, bit: 0 }).unwrap();
        let qasm = gen.generate();
        assert!(qasm.starts_with("OPENQASM 3.0;"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("cx q[0], q[1];"));
        assert!(qasm.contains("measure q[0]"));
    }

    #[test]
    fn test_quantum_coupling_violation() {
        let topo = QuantumTopology { num_qubits: 5, coupling_map: vec![(0, 1)] };
        let mut gen = OpenQasmGenerator::new(topo);
        assert!(gen.add_gate(QuantumGate::Cx { control: 0, target: 1 }).is_ok());
        assert!(gen.add_gate(QuantumGate::Cx { control: 0, target: 4 }).is_err());
    }

    #[test]
    fn test_quantum_qubit_out_of_range() {
        let topo = QuantumTopology { num_qubits: 3, coupling_map: vec![] };
        let mut gen = OpenQasmGenerator::new(topo);
        assert!(gen.add_gate(QuantumGate::H { qubit: 5 }).is_err());
        assert!(gen.add_gate(QuantumGate::H { qubit: 2 }).is_ok());
    }

    #[test]
    fn test_quantum_validate_qasm() {
        let valid = "OPENQASM 3.0;\ninclude 'stdgates.inc';\nqubit[2] q;\nh q[0];\n";
        assert_eq!(OpenQasmGenerator::validate_qasm(valid).len(), 0);

        let missing_header = "qubit[2] q;\nh q[0];\n";
        assert!(OpenQasmGenerator::validate_qasm(missing_header).len() > 0);
    }

    #[test]
    fn test_quantum_ibm_falcon() {
        let topo = QuantumTopology::ibm_falcon_27();
        assert_eq!(topo.num_qubits, 27);
        assert!(topo.allows_coupling(0, 1));
        assert!(topo.allows_coupling(1, 0)); // bidirectional
    }

    #[test]
    fn test_quantum_rotation_gate() {
        let topo = QuantumTopology { num_qubits: 2, coupling_map: vec![] };
        let mut gen = OpenQasmGenerator::new(topo);
        gen.add_gate(QuantumGate::Rz { qubit: 0, angle: 1.5708 }).unwrap();
        let qasm = gen.generate();
        assert!(qasm.contains("rz(1.5708)"));
    }
}
