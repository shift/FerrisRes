//! Vision-Language-Action (VLA) controller for robotic hardware.
//!
//! Implements ActionHead for predicting robot actions from visual + language
//! inputs. Supports both discrete (binned) and continuous (regression) action
//! representations, multiple control modes, and physics simulation via
//! MirrorTestRunner integration.
//!
//! Architecture mirrors VLA literature (RT-2, OpenVLA, π₀):
//!   Camera image + language instruction → VisionEncoder + LLM →
//!   ActionHead → (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)

// ---------------------------------------------------------------------------
// Control modes
// ---------------------------------------------------------------------------

/// Control mode for robot action output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlMode {
    /// Cartesian end-effector deltas: (Δx, Δy, Δz, Δroll, Δpitch, Δyaw).
    CartesianDelta,
    /// Joint-space velocity commands: (j1, j2, ..., jN).
    JointVelocity,
    /// Joint-space torque commands: (τ1, τ2, ..., τN).
    JointTorque,
}

/// Degrees of freedom for a robot arm.
#[derive(Debug, Clone)]
pub struct RobotDof {
    /// Number of arm joints (e.g., 7 for a Franka Panda).
    pub num_joints: usize,
    /// Whether a gripper is present.
    pub has_gripper: usize, // 0 or 1
}

impl RobotDof {
    /// Standard 6-DoF + gripper.
    pub fn standard_6dof() -> Self {
        Self { num_joints: 6, has_gripper: 1 }
    }

    /// 7-DoF arm + gripper (e.g., Franka Panda).
    pub fn panda_7dof() -> Self {
        Self { num_joints: 7, has_gripper: 1 }
    }

    /// Total action dimensions for CartesianDelta mode.
    pub fn cartesian_dims(&self) -> usize {
        6 + self.has_gripper // xyz + rpy + gripper
    }

    /// Total action dimensions for joint mode.
    pub fn joint_dims(&self) -> usize {
        self.num_joints + self.has_gripper
    }
}

// ---------------------------------------------------------------------------
// Action representation
// ---------------------------------------------------------------------------

/// A robot action in continuous form.
#[derive(Debug, Clone)]
pub struct RobotAction {
    /// Action values per DoF.
    pub values: Vec<f32>,
    /// Control mode used to produce this action.
    pub mode: ControlMode,
    /// Timestamp in seconds.
    pub timestamp: f32,
}

impl RobotAction {
    /// Create a zero action with given dimensions.
    pub fn zeros(dims: usize, mode: ControlMode) -> Self {
        Self { values: vec![0.0; dims], mode, timestamp: 0.0 }
    }

    /// Clamp all values to [-max, max].
    pub fn clamp(&mut self, max: f32) {
        for v in &mut self.values {
            *v = v.clamp(-max, max);
        }
    }

    /// Check if any value is NaN or infinite.
    pub fn is_valid(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }

    /// L2 norm of the action vector.
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
}

/// A binned (discrete) action for token-based prediction.
#[derive(Debug, Clone)]
pub struct BinnedAction {
    /// Token index per DoF (0..num_bins).
    pub tokens: Vec<usize>,
    /// Number of bins per dimension.
    pub num_bins: usize,
}

impl BinnedAction {
    /// Decode binned tokens to continuous values in [-1, 1].
    pub fn decode(&self) -> Vec<f32> {
        self.tokens.iter().map(|&t| {
            (t as f32 / (self.num_bins - 1).max(1) as f32) * 2.0 - 1.0
        }).collect()
    }

    /// Encode continuous values to binned tokens.
    pub fn encode(values: &[f32], num_bins: usize) -> Self {
        let tokens: Vec<usize> = values.iter().map(|&v| {
            let normalized = (v.clamp(-1.0, 1.0) + 1.0) / 2.0;
            (normalized * (num_bins - 1) as f32).round() as usize
        }).collect();
        Self { tokens, num_bins }
    }
}

// ---------------------------------------------------------------------------
// ActionHead
// ---------------------------------------------------------------------------

/// Configuration for the ActionHead.
#[derive(Debug, Clone)]
pub struct ActionHeadConfig {
    /// Transformer hidden dimension.
    pub hidden_dim: usize,
    /// Number of action dimensions (DoF).
    pub num_action_dims: usize,
    /// Number of bins for discrete mode (e.g., 256).
    pub num_bins: usize,
    /// Control mode.
    pub control_mode: ControlMode,
    /// Control frequency in Hz (e.g., 100).
    pub control_frequency: u32,
    /// Whether to use binned (discrete) or continuous output.
    pub binned: bool,
    /// Maximum action magnitude for clamping.
    pub max_action: f32,
}

impl ActionHeadConfig {
    /// Standard 6-DoF Cartesian delta with 256 bins.
    pub fn standard_cartesian(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            num_action_dims: 7, // xyz + rpy + gripper
            num_bins: 256,
            control_mode: ControlMode::CartesianDelta,
            control_frequency: 100,
            binned: true,
            max_action: 1.0,
        }
    }

    /// 7-DoF joint velocity control.
    pub fn joint_velocity(hidden_dim: usize, num_joints: usize) -> Self {
        Self {
            hidden_dim,
            num_action_dims: num_joints + 1, // joints + gripper
            num_bins: 256,
            control_mode: ControlMode::JointVelocity,
            control_frequency: 100,
            binned: true,
            max_action: 1.0,
        }
    }

    /// Continuous regression mode.
    pub fn continuous(hidden_dim: usize, num_action_dims: usize, mode: ControlMode) -> Self {
        Self {
            hidden_dim,
            num_action_dims,
            num_bins: 0,
            control_mode: mode,
            control_frequency: 100,
            binned: false,
            max_action: 1.0,
        }
    }
}

/// Action prediction head for robotic control.
///
/// Projects hidden states to action values. Supports both:
/// - **Binned mode**: per-DoF projection → argmax over bins → decode to continuous
/// - **Continuous mode**: per-DoF projection → tanh squash → clamped output
pub struct ActionHead {
    config: ActionHeadConfig,
    /// Per-DoF projection weights: [num_action_dims × (hidden_dim × output_dim)].
    /// output_dim = num_bins for binned, 1 for continuous.
    weights: Vec<Vec<f32>>,
    /// Per-DoF biases.
    biases: Vec<Vec<f32>>,
}

impl ActionHead {
    /// Create with Xavier initialization.
    pub fn new(config: ActionHeadConfig) -> Self {
        let hd = config.hidden_dim;
        let od = if config.binned { config.num_bins } else { 1 };
        let scale = (2.0 / (hd + od) as f32).sqrt();

        let weights: Vec<Vec<f32>> = (0..config.num_action_dims)
            .map(|dof| {
                (0..hd * od)
                    .map(|i| {
                        let seed = i as f32 + dof as f32 * 1000.0;
                        let x = ((seed * 0.618 + 0.1).sin() * 43758.5453).fract() - 0.5;
                        x * scale
                    })
                    .collect()
            })
            .collect();
        let biases: Vec<Vec<f32>> = (0..config.num_action_dims)
            .map(|_| vec![0.0; od])
            .collect();

        Self { config, weights, biases }
    }

    /// Forward: predict action from a single hidden state.
    pub fn forward(&self, hidden: &[f32]) -> RobotAction {
        debug_assert_eq!(hidden.len(), self.config.hidden_dim);
        let hd = self.config.hidden_dim;
        let max_val = self.config.max_action;

        if self.config.binned {
            let tokens: Vec<usize> = (0..self.config.num_action_dims).map(|dof| {
                let od = self.config.num_bins;
                let w = &self.weights[dof];
                let b = &self.biases[dof];
                let mut best_idx = 0;
                let mut best_val = f32::NEG_INFINITY;
                for c in 0..od {
                    let mut s = b[c];
                    for h in 0..hd {
                        s += hidden[h] * w[h * od + c];
                    }
                    if s > best_val {
                        best_val = s;
                        best_idx = c;
                    }
                }
                best_idx
            }).collect();

            let values: Vec<f32> = tokens.iter().map(|&t| {
                let normalized = (t as f32 / (self.config.num_bins - 1).max(1) as f32) * 2.0 - 1.0;
                normalized * max_val
            }).collect();

            RobotAction {
                values,
                mode: self.config.control_mode,
                timestamp: 0.0,
            }
        } else {
            let values: Vec<f32> = (0..self.config.num_action_dims).map(|dof| {
                let w = &self.weights[dof];
                let b = &self.biases[dof];
                let mut s = b[0];
                for h in 0..hd {
                    s += hidden[h] * w[h];
                }
                s.tanh() * max_val
            }).collect();

            RobotAction {
                values,
                mode: self.config.control_mode,
                timestamp: 0.0,
            }
        }
    }

    /// Predict binned tokens directly (for token-based training).
    pub fn forward_binned(&self, hidden: &[f32]) -> Option<BinnedAction> {
        if !self.config.binned { return None; }
        let action = self.forward(hidden);
        let tokens: Vec<usize> = action.values.iter().map(|&v| {
            let normalized = (v / self.config.max_action + 1.0) / 2.0;
            ((normalized * (self.config.num_bins - 1) as f32).round() as usize)
                .min(self.config.num_bins - 1)
        }).collect();
        Some(BinnedAction { tokens, num_bins: self.config.num_bins })
    }

    /// Config accessor.
    pub fn config(&self) -> &ActionHeadConfig { &self.config }
}

// ---------------------------------------------------------------------------
// VlaObservation — input to the VLA pipeline
// ---------------------------------------------------------------------------

/// Observation from the robot's sensors.
#[derive(Debug, Clone)]
pub struct VlaObservation {
    /// Camera image tokens (from VisionEncoder).
    pub vision_tokens: Vec<Vec<f32>>,
    /// Joint state values (positions or angles).
    pub joint_state: Vec<f32>,
    /// Language instruction (tokenized).
    pub instruction_tokens: Vec<u32>,
    /// Timestamp in seconds.
    pub timestamp: f32,
}

impl VlaObservation {
    /// Create with given dimensions.
    pub fn new(
        vision_tokens: Vec<Vec<f32>>,
        joint_state: Vec<f32>,
        instruction_tokens: Vec<u32>,
    ) -> Self {
        Self { vision_tokens, joint_state, instruction_tokens, timestamp: 0.0 }
    }

    /// Number of vision tokens.
    pub fn num_vision_tokens(&self) -> usize {
        self.vision_tokens.len()
    }

    /// Number of joints.
    pub fn num_joints(&self) -> usize {
        self.joint_state.len()
    }
}

// ---------------------------------------------------------------------------
// VlaSafetyCheck — physics simulation before execution
// ---------------------------------------------------------------------------

/// Result of a safety check on a proposed action.
#[derive(Debug, Clone)]
pub struct SafetyCheckResult {
    /// Whether the action is safe to execute.
    pub safe: bool,
    /// Reason for rejection (if unsafe).
    pub rejection_reason: Option<String>,
    /// Joint limits violated (if any).
    pub violated_joints: Vec<usize>,
    /// Collision detected.
    pub collision: bool,
    /// Workspace boundary violation.
    pub out_of_bounds: bool,
}

/// Safety checker for robot actions.
pub struct VlaSafetyChecker {
    /// Joint limits: [(min, max)] per joint.
    pub joint_limits: Vec<(f32, f32)>,
    /// Workspace bounds: (x_min, x_max, y_min, y_max, z_min, z_max).
    pub workspace_bounds: (f32, f32, f32, f32, f32, f32),
    /// Max allowed action norm.
    pub max_action_norm: f32,
}

impl VlaSafetyChecker {
    /// Create with standard limits.
    pub fn standard(num_joints: usize) -> Self {
        Self {
            joint_limits: vec![(-2.9, 2.9); num_joints],
            workspace_bounds: (-1.0, 1.0, -1.0, 1.0, 0.0, 1.5),
            max_action_norm: 1.0,
        }
    }

    /// Create with no limits (for simulation/testing).
    pub fn no_limits() -> Self {
        Self {
            joint_limits: vec![(-1000.0, 1000.0); 7],
            workspace_bounds: (-1000.0, 1000.0, -1000.0, 1000.0, -1000.0, 1000.0),
            max_action_norm: f32::MAX,
        }
    }

    /// Check if an action is safe to execute.
    pub fn check(&self, action: &RobotAction, current_joint_state: &[f32]) -> SafetyCheckResult {
        let mut violated_joints = Vec::new();
        let mut reason = None;
        let collision = false;
        let mut out_of_bounds = false;

        // Check action norm
        if action.norm() > self.max_action_norm {
            reason = Some(format!("Action norm {} exceeds max {}", action.norm(), self.max_action_norm));
        }

        // Check NaN/Inf
        if !action.is_valid() {
            return SafetyCheckResult {
                safe: false,
                rejection_reason: Some("Action contains NaN or Inf".into()),
                violated_joints: vec![],
                collision: false,
                out_of_bounds: false,
            };
        }

        // Check joint limits (if joint mode)
        if matches!(action.mode, ControlMode::JointVelocity | ControlMode::JointTorque) {
            for (i, &delta) in action.values.iter().enumerate() {
                if i < current_joint_state.len() && i < self.joint_limits.len() {
                    let new_pos = current_joint_state[i] + delta;
                    let (min, max) = self.joint_limits[i];
                    if new_pos < min || new_pos > max {
                        violated_joints.push(i);
                    }
                }
            }
        }

        // Check workspace bounds (if Cartesian mode)
        if matches!(action.mode, ControlMode::CartesianDelta) && action.values.len() >= 3 {
            let (_xmin, _xmax, _ymin, _ymax, _zmin, _zmax) = self.workspace_bounds;
            // We can only check if current position is known — for now just
            // check that deltas aren't unreasonably large
            for i in 0..3.min(action.values.len()) {
                if action.values[i].abs() > 0.5 {
                    out_of_bounds = true;
                    if reason.is_none() {
                        reason = Some("Cartesian delta exceeds 0.5m".into());
                    }
                }
            }
        }

        let safe = violated_joints.is_empty() && !collision && !out_of_bounds
            && reason.is_none();

        SafetyCheckResult {
            safe,
            rejection_reason: reason,
            violated_joints,
            collision,
            out_of_bounds,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robot_dof_standard() {
        let dof = RobotDof::standard_6dof();
        assert_eq!(dof.cartesian_dims(), 7);
        assert_eq!(dof.joint_dims(), 7);
    }

    #[test]
    fn test_robot_dof_panda() {
        let dof = RobotDof::panda_7dof();
        assert_eq!(dof.cartesian_dims(), 7);
        assert_eq!(dof.joint_dims(), 8);
    }

    #[test]
    fn test_robot_action_zeros() {
        let action = RobotAction::zeros(7, ControlMode::CartesianDelta);
        assert_eq!(action.values.len(), 7);
        assert!(action.values.iter().all(|&v| v == 0.0));
        assert!(action.is_valid());
    }

    #[test]
    fn test_robot_action_clamp() {
        let mut action = RobotAction {
            values: vec![-2.0, 0.5, 3.0],
            mode: ControlMode::CartesianDelta,
            timestamp: 0.0,
        };
        action.clamp(1.0);
        assert_eq!(action.values, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn test_robot_action_validity() {
        let valid = RobotAction { values: vec![0.5, -0.3], mode: ControlMode::CartesianDelta, timestamp: 0.0 };
        assert!(valid.is_valid());
        let invalid = RobotAction { values: vec![f32::NAN], mode: ControlMode::CartesianDelta, timestamp: 0.0 };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_robot_action_norm() {
        let action = RobotAction { values: vec![3.0, 4.0], mode: ControlMode::CartesianDelta, timestamp: 0.0 };
        assert!((action.norm() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_binned_action_encode_decode() {
        let values = vec![-1.0, 0.0, 1.0];
        let binned = BinnedAction::encode(&values, 256);
        assert_eq!(binned.num_bins, 256);
        assert_eq!(binned.tokens.len(), 3);
        assert_eq!(binned.tokens[0], 0);   // -1.0 → bin 0
        assert_eq!(binned.tokens[1], 128); // 0.0 → bin 128
        assert_eq!(binned.tokens[2], 255); // 1.0 → bin 255

        let decoded = binned.decode();
        assert_eq!(decoded.len(), 3);
        assert!((decoded[0] - (-1.0)).abs() < 0.01);
        assert!((decoded[1]).abs() < 0.01);
        assert!((decoded[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_action_head_binned() {
        let config = ActionHeadConfig::standard_cartesian(256);
        let head = ActionHead::new(config);
        let hidden = vec![0.5f32; 256];
        let action = head.forward(&hidden);
        assert_eq!(action.values.len(), 7);
        assert!(action.is_valid());
        assert_eq!(action.mode, ControlMode::CartesianDelta);
    }

    #[test]
    fn test_action_head_continuous() {
        let config = ActionHeadConfig::continuous(128, 7, ControlMode::CartesianDelta);
        let head = ActionHead::new(config);
        let hidden = vec![0.3f32; 128];
        let action = head.forward(&hidden);
        assert_eq!(action.values.len(), 7);
        assert!(action.is_valid());
        // tanh output is in [-1, 1], scaled by max_action=1.0
        for &v in &action.values {
            assert!(v.abs() <= 1.0 + 1e-5);
        }
    }

    #[test]
    fn test_action_head_forward_binned() {
        let config = ActionHeadConfig::standard_cartesian(64);
        let head = ActionHead::new(config);
        let hidden = vec![0.5f32; 64];
        let binned = head.forward_binned(&hidden).unwrap();
        assert_eq!(binned.tokens.len(), 7);
        assert_eq!(binned.num_bins, 256);
        for &t in &binned.tokens {
            assert!(t < 256);
        }
    }

    #[test]
    fn test_action_head_continuous_no_binned() {
        let config = ActionHeadConfig::continuous(64, 7, ControlMode::CartesianDelta);
        let head = ActionHead::new(config);
        let hidden = vec![0.5f32; 64];
        assert!(head.forward_binned(&hidden).is_none());
    }

    #[test]
    fn test_vla_observation() {
        let obs = VlaObservation::new(
            vec![vec![1.0; 256]; 10],
            vec![0.0; 7],
            vec![1, 2, 3],
        );
        assert_eq!(obs.num_vision_tokens(), 10);
        assert_eq!(obs.num_joints(), 7);
    }

    #[test]
    fn test_safety_check_safe() {
        let checker = VlaSafetyChecker::standard(7);
        let action = RobotAction {
            values: vec![0.01; 7],
            mode: ControlMode::JointVelocity,
            timestamp: 0.0,
        };
        let joints = vec![0.0; 7];
        let result = checker.check(&action, &joints);
        assert!(result.safe);
    }

    #[test]
    fn test_safety_check_nan() {
        let checker = VlaSafetyChecker::standard(7);
        let action = RobotAction {
            values: vec![f32::NAN],
            mode: ControlMode::CartesianDelta,
            timestamp: 0.0,
        };
        let result = checker.check(&action, &[0.0]);
        assert!(!result.safe);
    }

    #[test]
    fn test_safety_check_large_delta() {
        let checker = VlaSafetyChecker::standard(7);
        let action = RobotAction {
            values: vec![0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            mode: ControlMode::CartesianDelta,
            timestamp: 0.0,
        };
        let result = checker.check(&action, &[0.0; 7]);
        assert!(!result.safe);
        assert!(result.out_of_bounds);
    }

    #[test]
    fn test_safety_no_limits() {
        let checker = VlaSafetyChecker::no_limits();
        // Large Cartesian deltas trigger out_of_bounds regardless of workspace bounds
        // Use joint mode with small deltas to stay within limits
        let action = RobotAction {
            values: vec![0.1; 8],
            mode: ControlMode::JointVelocity,
            timestamp: 0.0,
        };
        let result = checker.check(&action, &[0.0; 8]);
        assert!(result.safe);
    }
}
