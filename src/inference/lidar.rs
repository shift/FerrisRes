//! LiDAR — Point Cloud Encoder + Drive-By-Wire Head + Kinematic Oracle
//! 
//! Autonomous vehicle LiDAR processing:
//! - Native 3D point cloud encoding
//! - Drive-by-wire control
//! - Kinematic safety constraints

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum LidarError {
    #[error("Point cloud: {0}")]
    PointCloud(String),
    
    #[error("Drive-by-wire: {0}")]
    DriveByWire(String),
    
    #[error("Kinematic: {0}")]
    Kinematic(String),
}

// ============================================================================
// Point Cloud Types
// ============================================================================

/// LiDAR point (x, y, z in meters, intensity 0-1)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LidarPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub timestamp_ns: u64,
}

impl LidarPoint {
    pub fn new(x: f32, y: f32, z: f32, intensity: f32) -> Self {
        Self { x, y, z, intensity, timestamp_ns: 0 }
    }
    
    /// Distance from origin
    pub fn distance(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

/// Point cloud frame
#[derive(Debug, Clone)]
pub struct PointCloudFrame {
    pub points: Vec<LidarPoint>,
    pub frame_id: u32,
    pub timestamp_ms: u64,
}

impl PointCloudFrame {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            frame_id: 0,
            timestamp_ms: 0,
        }
    }
    
    /// Number of points
    pub fn num_points(&self) -> usize {
        self.points.len()
    }
    
    /// Bounding box extent
    pub fn extent(&self) -> (f32, f32, f32) {
        if self.points.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let xs = self.points.iter().map(|p| p.x);
        let ys = self.points.iter().map(|p| p.y);
        let zs = self.points.iter().map(|p| p.z);
        (
            xs.clone().fold(f32::MIN, f32::max) - xs.clone().fold(f32::MAX, f32::min),
            ys.clone().fold(f32::MIN, f32::max) - ys.clone().fold(f32::MAX, f32::min),
            zs.clone().fold(f32::MIN, f32::max) - zs.clone().fold(f32::MAX, f32::min),
        )
    }
}

// ============================================================================
// Point Cloud Encoder
// ============================================================================

/// Voxel grid statistics
#[derive(Debug, Clone)]
struct VoxelStats {
    count: u32,
    max_intensity: f32,
    z_min: f32,
    z_max: f32,
}

/// Encodes 3D point clouds without 2D projection
pub struct PointCloudEncoder {
    pub voxel_size: f32,
    pub grid_dims: (u32, u32, u32),
}

impl PointCloudEncoder {
    pub fn new(voxel_size: f32) -> Self {
        Self {
            voxel_size,
            grid_dims: (100, 100, 20),
        }
    }
    
    /// Encode point cloud to tokens (3D voxel encoding)
    pub fn encode(&self, frame: &PointCloudFrame) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Voxelize
        let mut grid: HashMap<(u32, u32, u32), VoxelStats> = HashMap::new();
        
        for point in &frame.points {
            let xb = ((point.x + 50.0) / self.voxel_size) as u32;
            let yb = ((point.y + 50.0) / self.voxel_size) as u32;
            let zb = ((point.z + 5.0) / self.voxel_size) as u32;
            
            if xb < self.grid_dims.0 && yb < self.grid_dims.1 && zb < self.grid_dims.2 {
                let stats = grid.entry((xb, yb, zb)).or_insert(VoxelStats {
                    count: 0,
                    max_intensity: 0.0,
                    z_min: f32::MAX,
                    z_max: f32::MIN,
                });
                stats.count += 1;
                stats.max_intensity = stats.max_intensity.max(point.intensity);
                stats.z_min = stats.z_min.min(point.z);
                stats.z_max = stats.z_max.max(point.z);
            }
        }
        
        // Encode occupied voxels
        for ((xb, yb, zb), stats) in &grid {
            tokens.push(*xb);
            tokens.push(*yb);
            tokens.push(*zb);
            tokens.push(stats.count.min(255) as u32);
            tokens.push((stats.max_intensity * 255.0) as u32);
        }
        
        // Frame metadata
        tokens.push(frame.frame_id);
        tokens.push(frame.points.len() as u32);
        
        tokens
    }
    
    /// Get point cloud statistics
    pub fn stats(&self, frame: &PointCloudFrame) -> PointCloudStats {
        let extent = frame.extent();
        let num_points = frame.num_points();
        
        PointCloudStats {
            num_points,
            extent: (extent.0, extent.1, extent.2),
            max_range: frame.points.iter().map(|p| p.distance()).fold(0.0, f32::max),
        }
    }
}

/// Point cloud statistics
#[derive(Debug, Clone)]
pub struct PointCloudStats {
    pub num_points: usize,
    pub extent: (f32, f32, f32),
    pub max_range: f32,
}

// ============================================================================
// Drive-By-Wire Head
// ============================================================================

/// Vehicle control signal
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DriveByWireControl {
    pub steering_angle: f32,     // radians, -0.5 to 0.5
    pub throttle: f32,            // 0.0 to 1.0
    pub brake_pressure: f32,      // 0.0 to 1.0
    pub gear: GearPosition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GearPosition {
    Park,
    Reverse,
    Neutral,
    Drive,
}

impl Default for DriveByWireControl {
    fn default() -> Self {
        Self {
            steering_angle: 0.0,
            throttle: 0.0,
            brake_pressure: 1.0,  // Brakes engaged by default
            gear: GearPosition::Park,
        }
    }
}

/// Drive-by-wire head - generates vehicle control from embeddings
pub struct DriveByWireHead {
    pub wheelbase: f32,
    pub max_steering_angle: f32,
}

impl DriveByWireHead {
    pub fn new() -> Self {
        Self {
            wheelbase: 3.5,  // meters
            max_steering_angle: 0.5,
        }
    }
    
    /// Predict control from embeddings
    pub fn predict(&self, embeddings: &[f32]) -> DriveByWireControl {
        if embeddings.is_empty() {
            return DriveByWireControl::default();
        }
        
        let mag = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        let mut control = DriveByWireControl::default();
        
        if mag > 5.0 {
            // High activity - move
            control.gear = GearPosition::Drive;
            control.brake_pressure = 0.0;
            control.throttle = (mag / 10.0).min(1.0);
            control.steering_angle = embeddings.get(0).copied().unwrap_or(0.0)
                .clamp(-self.max_steering_angle, self.max_steering_angle);
        } else {
            // Low activity - stop
            control.brake_pressure = 1.0;
            control.gear = GearPosition::Park;
        }
        
        control
    }
    
    /// Calculate curvature from steering
    pub fn curvature(&self, steering_angle: f32) -> f32 {
        if steering_angle.abs() < 0.001 {
            return 0.0;
        }
        // κ = tan(δ) / L
        (steering_angle / self.wheelbase).tan()
    }
}

// ============================================================================
// Kinematic Oracle
// ============================================================================

/// Safety violation
#[derive(Debug, Clone)]
pub enum SafetyViolation {
    FrictionCircle { actual: f32, limit: f32 },
    Rollover { lateral_g: f32, limit: f32 },
    SpeedLimit { actual: f32, limit: f32 },
}

/// Kinematic oracle - validates safety constraints
pub struct KinematicOracle {
    pub max_lateral_accel: f32,       // m/s²
    pub max_longitudinal_accel: f32, // m/s²
    pub max_speed_kmh: f32,
    pub road_friction: f32,
}

impl KinematicOracle {
    pub fn new() -> Self {
        Self {
            max_lateral_accel: 0.5,  // ~0.05g typical
            max_longitudinal_accel: 0.8,
            max_speed_kmh: 120.0,
            road_friction: 0.7,  // dry asphalt
        }
    }
    
    /// Validate control against friction circle
    pub fn validate_friction_circle(&self, control: &DriveByWireControl, velocity_ms: f32) -> Option<SafetyViolation> {
        let curvature = (control.steering_angle / self.wheelbase).tan();
        let lateral_accel = velocity_ms * velocity_ms * curvature.abs();
        
        if lateral_accel > self.max_lateral_accel * self.road_friction {
            return Some(SafetyViolation::FrictionCircle {
                actual: lateral_accel,
                limit: self.max_lateral_accel * self.road_friction,
            });
        }
        None
    }
    
    /// Validate rollover threshold
    pub fn validate_rollover(&self, control: &DriveByWireControl, velocity_ms: f32) -> Option<SafetyViolation> {
        let lateral_g = (velocity_ms * velocity_ms * control.steering_angle.abs()) / (9.81 * self.wheelbase);
        
        if lateral_g > 1.0 {
            return Some(SafetyViolation::Rollover {
                actual: lateral_g,
                limit: 1.0,
            });
        }
        None
    }
    
    /// Validate speed
    pub fn validate_speed(&self, velocity_ms: f32) -> Option<SafetyViolation> {
        let speed_kmh = velocity_ms * 3.6;
        
        if speed_kmh > self.max_speed_kmh {
            return Some(SafetyViolation::SpeedLimit {
                actual: speed_kmh,
                limit: self.max_speed_kmh,
            });
        }
        None
    }
    
    /// Full safety check
    pub fn validate(&self, control: &DriveByWireControl, velocity_ms: f32) -> bool {
        self.validate_friction_circle(control, velocity_ms).is_none()
            && self.validate_rollover(control, velocity_ms).is_none()
            && self.validate_speed(velocity_ms).is_none()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_point_cloud() {
        let frame = PointCloudFrame::new();
        assert_eq!(frame.num_points(), 0);
    }
    
    #[test]
    fn test_lidar_point() {
        let p = LidarPoint::new(1.0, 2.0, 0.0, 0.5);
        assert!((p.distance() - 2.236).abs() < 0.01);
    }
    
    #[test]
    fn test_point_cloud_encoder() {
        let encoder = PointCloudEncoder::new(0.1);
        let mut frame = PointCloudFrame::new();
        frame.points.push(LidarPoint::new(1.0, 2.0, 0.0, 0.5));
        
        let tokens = encoder.encode(&frame);
        assert!(!tokens.is_empty());
    }
    
    #[test]
    fn test_drive_by_wire() {
        let head = DriveByWireHead::new();
        let embeddings = vec![5.0, 1.0, 0.5];
        
        let control = head.predict(&embeddings);
        assert!(control.throttle > 0.0);
    }
    
    #[test]
    fn test_kinematic_oracle() {
        let oracle = KinematicOracle::new();
        
        let control = DriveByWireControl::default();
        let velocity = 10.0;  // 10 m/s = 36 km/h
        
        let result = oracle.validate(&control, velocity);
        assert!(result);
    }
    
    #[test]
    fn test_friction_circle_violation() {
        let oracle = KinematicOracle::new();
        
        let mut control = DriveByWireControl::default();
        control.steering_angle = 0.5;
        
        let result = oracle.validate_friction_circle(&control, 30.0);  // high speed
        assert!(result.is_some());
    }
}