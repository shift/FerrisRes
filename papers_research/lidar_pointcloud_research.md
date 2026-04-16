# LiDAR Research — Point Cloud Encoder, Drive-By-Wire Head, Kinematic Oracle

## Overview
Autonomous vehicle LiDAR processing:
- Velodyne/Ouster point cloud formats
- Native 3D encoding (no 2D projection)
- Drive-by-wire control signals
- Kinematic safety constraints

## LiDAR Point Cloud Formats

### Velodyne HDL-64E
- 64 lasers, dual return
- Data packet: 1206 bytes (12 bytes × 12 data blocks)
- Each block: 2-byte distance + 1-byte intensity

### Ouster OS1-64
- 1024 points per azimuthal sample
- Timestamp, encoder position, near range, far range

### Point Structure
```rust
pub struct LidarPoint {
    x: f32,        // meters
    y: f32,
    z: f32,
    intensity: f32,  // 0-255
    timestamp_ns: u64,
}
```

## PointCloudEncoder

### Approach: Native 3D Voxelization
Instead of projecting to 2D (BEV), encode in 3D voxel grid:

```
For each frame:
1. Voxelize into 100x100x20 grid (0.1m resolution)
2. For each occupied voxel: [x_bin, y_bin, z_bin, density, max_intensity]
3. Flatten to sequence (sorted by spatial locality)
```

```rust
pub struct PointCloudEncoder {
    voxel_size: f32,
    grid_dims: (u32, u32, u32),
}

impl PointCloudEncoder {
    pub fn encode(&self, points: &[LidarPoint]) -> Vec<u32> {
        let mut voxels = Vec::new();
        let mut grid = HashMap<(u32,u32,u32), VoxelStats>::new();
        
        for point in points {
            let bin = (
                (point.x / self.voxel_size) as u32,
                (point.y / self.voxel_size) as u32,
                (point.z / self.voxel_size) as u32,
            );
            
            let stats = grid.entry(bin).or_insert(VoxelStats {
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
        
        // Encode as tokens
        for (bin, stats) in grid {
            tokens.push(bin.0);  // x bin
            tokens.push(bin.1);  // y bin
            tokens.push(bin.2);  // z bin
            tokens.push(stats.count.min(255) as u32);  // density
            tokens.push(stats.max_intensity as u32);   // intensity
        }
        
        tokens
    }
}

struct VoxelStats {
    count: u32,
    max_intensity: f32,
    z_min: f32,
    z_max: f32,
}
```

### nuScenes Format Reference
```
token: str (unique identifier)
sample_token: str (reference to sample)
timestamp: int (nanoseconds)
lidar_token: str (reference to lidar data)
```

## Drive-By-Wire Head

### Control Signals
```rust
pub struct DriveByWireControl {
    steering_angle: f32,    // radians, -0.5 to 0.5
    throttle: f32,          // 0.0 to 1.0
    brake_pressure: f32,    // 0.0 to 1.0
    gear: GearPosition,
}

pub enum GearPosition {
    Park,
    Reverse,
    Neutral,
    Drive,
}
```

### KinematicOracle - Safety Constraints

#### Friction Circle
```rust
pub struct KinematicOracle {
    max_lateral_accel: f32,  // m/s²
    max_longitudinal_accel: f32,
}

impl KinematicOracle {
    pub fn validate(&self, control: &DriveByWireControl, velocity: f32) -> ValidationResult {
        // a = v²/r, r = steering_angle * wheelbase
        let curvature = (control.steering_angle / 3.5).tan();  // wheelbase ~3.5m
        let lateral_accel = velocity * velocity * curvature.abs();
        
        if lateral_accel > self.max_lateral_accel {
            return ValidationResult::Violation(
                format!("Lateral accel {} exceeds limit {}", 
                        lateral_accel, self.max_lateral_accel)
            );
        }
        
        ValidationResult::Valid
    }
}
```

#### Rollover Threshold
```rust
pub fn check_rollover(&self, control: &DriveByWireControl, 
                     velocity: f32, road_friction: f32) -> bool {
    // Simplified: centripetal force < friction * gravity
    let centripetal = velocity.powi(2) * control.steering_angle.abs() / 3.5;
    let max_friction = road_friction * 9.81;
    
    centripetal < max_friction
}
```

## Waymo Dataset Reference
- 1000 scenes, 20s each
- 64-beam LiDAR, 10Hz
- Per-frame: ~180K points
- Taxonomy: vehicle, pedestrian, cyclist, traffic light, sign

## References
- Velodyne: https://velodynelidar.com/products/
- Ouster: https://ouster.com/products/
- nuScenes: https://www.nuscenes.org/
- Waymo: https://waymo.com/open/