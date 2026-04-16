//! Supply Chain — Fleet Stream Encoder + Routing Head + Compliance Oracle
//! 
//! Fleet logistics and routing:
//! - GPS/CAN bus data (OBD-II, NMEA)
//! - Dynamic route planning
//! - Compliance (bridge clearance, Hours of Service, hazmat)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum SupplyChainError {
    #[error("GPS: {0}")]
    Gps(String),
    
    #[error("Routing: {0}")]
    Routing(String),
    
    #[error("Compliance: {0}")]
    Compliance(String),
}

// ============================================================================
// GPS/CAN Types
// ============================================================================

/// GPS coordinate
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GpsCoord {
    pub lat: f64,
    pub lon: f64,
}

impl GpsCoord {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }
    
    /// Distance to another coord in meters
    pub fn distance_to(&self, other: &GpsCoord) -> f64 {
        let dlat = (other.lat - self.lat).to_radians();
        let dlon = (other.lon - self.lon).to_radians();
        let a = (dlat / 2.0).sin().powi(2) 
            + self.lat.to_radians().cos() * other.lat.to_radians().cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();
        6371000.0 * c  // Earth radius in meters
    }
}

/// NMEA sentence type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmeaSentenceType {
    Gga,  // Global positioning
    Vtg,  // Track made good
    Rmc,  // Recommended minimum
}

/// OBD-II PID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObdPid {
    VehicleSpeed = 0x0D,
    ThrottlePosition = 0x11,
    EngineCoolantTemp = 0x05,
    FuelTankLevel = 0x2F,
    Odometer = 0x66,
    Rpm = 0x0C,
}

/// CAN message from vehicle
#[derive(Debug, Clone)]
pub struct CanMessage {
    pub pid: ObdPid,
    pub value: f32,
    pub timestamp_ms: u64,
}

/// Vehicle telemetry
#[derive(Debug, Clone)]
pub struct VehicleTelemetry {
    pub vehicle_id: String,
    pub gps: GpsCoord,
    pub heading: f32,    // degrees
    pub speed: f32,     // km/h
    pub odometer: f64,  // km
    pub can_messages: Vec<CanMessage>,
    pub timestamp_ms: u64,
}

// ============================================================================
// Fleet Stream Encoder
// ============================================================================

/// Encodes vehicle telemetry for Block AttnRes
pub struct FleetStreamEncoder {
    vehicle_token_map: HashMap<String, u32>,
}

impl FleetStreamEncoder {
    pub fn new() -> Self {
        Self {
            vehicle_token_map: HashMap::new(),
        }
    }
    
    /// Encode telemetry to tokens
    pub fn encode(&mut self, telemetry: &VehicleTelemetry) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Vehicle ID token
        let vehicle_token = *self.vehicle_token_map.entry(telemetry.vehicle_id.clone()).or_insert_with(
            self.vehicle_token_map.len() as u32 + 1000
        );
        tokens.push(vehicle_token);
        
        // GPS grid token (quantized)
        let lat_grid = ((telemetry.gps.lat + 90.0) * 100.0) as u32 % 18000;
        let lon_grid = ((telemetry.gps.lon + 180.0) * 100.0) as u32 % 36000;
        tokens.push(lat_grid);
        tokens.push(lon_grid);
        
        // Speed category
        tokens.push(speed_category(telemetry.speed));
        
        // Heading sector (16 sectors)
        tokens.push((telemetry.heading / 22.5) as u32 % 16);
        
        // Time bucket
        tokens.push(((telemetry.timestamp_ms / 3600000) % 24) as u32);
        
        // CAN messages
        for can in &telemetry.can_messages {
            tokens.push(can.pid as u32);
            tokens.push(quantize_can_value(can.value));
        }
        
        tokens
    }
}

fn speed_category(speed_kmh: f32) -> u32 {
    if speed_kmh < 1.0 { 0 }
    else if speed_kmh < 20.0 { 1 }
    else if speed_kmh < 40.0 { 2 }
    else if speed_kmh < 60.0 { 3 }
    else if speed_kmh < 80.0 { 4 }
    else if speed_kmh < 100.0 { 5 }
    else { 6 }
}

fn quantize_can_value(value: f32) -> u32 {
    (value.max(0.0).min(255.0) as u32)
}

// ============================================================================
// Routing Head
// ============================================================================

/// Route constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteConstraints {
    pub vehicle_height_m: f32,
    pub vehicle_width_m: f32,
    pub vehicle_weight_kg: f32,
    pub is_hazmat: bool,
    pub max_driving_hours: f32,
}

impl Default for RouteConstraints {
    fn default() -> Self {
        Self {
            vehicle_height_m: 4.5,
            vehicle_width_m: 2.5,
            vehicle_weight_kg: 40000,
            is_hazmat: false,
            max_driving_hours: 11.0,
        }
    }
}

/// Route segment
#[derive(Debug, Clone)]
pub struct RouteSegment {
    pub start: GpsCoord,
    pub end: GpsCoord,
    pub distance_km: f64,
    pub duration_min: u32,
    pub road_type: RoadType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadType {
    Highway,
    Primary,
    Secondary,
    Local,
}

/// Complete route
#[derive(Debug, Clone)]
pub struct Route {
    pub segments: Vec<RouteSegment>,
    pub total_distance_km: f64,
    pub total_duration_min: u32,
}

/// Routing head - generates routes from embeddings
pub struct RoutingHead {
    pub api_key: Option<String>,
}

impl RoutingHead {
    pub fn new() -> Self {
        Self { api_key: None }
    }
    
    /// Set API key for routing service
    pub fn set_api_key(&mut self, key: String) {
        self.api_key = Some(key);
    }
    
    /// Predict route from embeddings
    pub fn predict(&self, origin: GpsCoord, dest: GpsCoord, constraints: &RouteConstraints) -> Route {
        // Simplified routing - direct path
        let distance = origin.distance_to(&dest) / 1000.0;
        let avg_speed = if constraints.is_hazmat { 60.0 } else { 80.0 };
        let duration_min = ((distance / avg_speed) * 60.0) as u32;
        
        Route {
            segments: vec![RouteSegment {
                start: origin,
                end: dest,
                distance_km: distance,
                duration_min,
                road_type: RoadType::Highway,
            }],
            total_distance_km: distance,
            total_duration_min: duration_min,
        }
    }
    
    /// Estimate fuel cost
    pub fn estimate_fuel(&self, route: &Route, fuel_price_per_liter: f32, fuel_efficiency_l_per_100km: f32) -> f32 {
        (route.total_distance_km as f32 / 100.0) * fuel_efficiency_l_per_100km * fuel_price_per_liter
    }
}

// ============================================================================
// Compliance Oracle
// ============================================================================

/// Bridge clearance data
#[derive(Debug, Clone)]
pub struct BridgeClearance {
    pub id: String,
    pub location: GpsCoord,
    pub clearance_meters: f32,
}

/// Hours of Service rule
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HosRule {
    ElevenHour,  // 11-hour driving
    FourteenHour,  // 14-hour duty window
    ThirtyMinBreak,
}

/// Compliance validation result
#[derive(Debug, Clone)]
pub enum ComplianceResult {
    Valid,
    Warning(String),
    Violation(String),
}

/// Fleet compliance oracle
pub struct FleetComplianceOracle {
    pub bridges: HashMap<String, BridgeClearance>,
    pub restricted_areas: HashMap<String, String>,
    pub hos_rules: Vec<HosRule>,
}

impl FleetComplianceOracle {
    pub fn new() -> Self {
        Self {
            bridges: HashMap::new(),
            restricted_areas: HashMap::new(),
            hos_rules: vec![HosRule::ElevenHour, HosRule::FourteenHour],
        }
    }
    
    /// Add bridge clearance data
    pub fn add_bridge(&mut self, id: &str, lat: f64, lon: f64, clearance_m: f32) {
        self.bridges.insert(id.to_string(), BridgeClearance {
            id: id.to_string(),
            location: GpsCoord::new(lat, lon),
            clearance_meters: clearance_m,
        });
    }
    
    /// Validate route against bridge clearances
    pub fn validate_route(&self, route: &Route, vehicle_height_m: f32) -> ComplianceResult {
        for segment in &route.segments {
            // Check nearby bridges (simplified)
            for bridge in self.bridges.values() {
                let dist = segment.start.distance_to(&bridge.location);
                if dist < 100.0 && vehicle_height_m > bridge.clearance_meters {
                    return ComplianceResult::Violation(format!(
                        "Bridge {} clearance {}m < vehicle {}m",
                        bridge.id, bridge.clearance_meters, vehicle_height_m
                    ));
                }
            }
        }
        
        ComplianceResult::Valid
    }
    
    /// Validate Hours of Service
    pub fn validate_hos(&self, driving_hours: f32, on_duty_hours: f32, breaks_min: u32) -> ComplianceResult {
        for rule in &self.hos_rules {
            match rule {
                HosRule::ElevenHour => {
                    if driving_hours > 11.0 {
                        return ComplianceResult::Violation(
                            format!("Driving {} hours exceeds 11-hour limit", driving_hours)
                        );
                    }
                }
                HosRule::FourteenHour => {
                    if on_duty_hours > 14.0 {
                        return ComplianceResult::Violation(
                            format!("On-duty {} hours exceeds 14-hour limit", on_duty_hours)
                        );
                    }
                }
                HosRule::ThirtyMinBreak => {
                    if breaks_min < 30 && driving_hours > 8.0 {
                        return ComplianceResult::Warning(
                            "Less than 30 min break after 8 hours driving".to_string()
                        );
                    }
                }
            }
        }
        
        ComplianceResult::Valid
    }
    
    /// Validate hazmat route
    pub fn validate_hazmat(&self, route: &Route) -> ComplianceResult {
        for area in self.restricted_areas.values() {
            return ComplianceResult::Violation(format!(
                "Route passes through restricted area: {}", area
            ));
        }
        
        ComplianceResult::Valid
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gps_distance() {
        let a = GpsCoord::new(52.52, 13.405);  // Berlin
        let b = GpsCoord::new(52.51, 13.41);   // Nearby
        
        assert!(a.distance_to(&b) < 1000.0);
    }
    
    #[test]
    fn test_fleet_encoder() {
        let mut encoder = FleetStreamEncoder::new();
        
        let telemetry = VehicleTelemetry {
            vehicle_id: "TRUCK001".to_string(),
            gps: GpsCoord::new(52.52, 13.405),
            heading: 45.0,
            speed: 60.0,
            odometer: 100000.0,
            can_messages: vec![],
            timestamp_ms: 1713267600000,
        };
        
        let tokens = encoder.encode(&telemetry);
        assert!(!tokens.is_empty());
    }
    
    #[test]
    fn test_routing_head() {
        let head = RoutingHead::new();
        
        let origin = GpsCoord::new(52.52, 13.405);
        let dest = GpsCoord::new(52.51, 13.41);
        
        let route = head.predict(origin, dest, &RouteConstraints::default());
        
        assert!(route.total_distance_km > 0.0);
    }
    
    #[test]
    fn test_compliance_oracle() {
        let oracle = FleetComplianceOracle::new();
        
        let route = Route {
            segments: vec![],
            total_distance_km: 100.0,
            total_duration_min: 90,
        };
        
        let result = oracle.validate_route(&route, 4.0);
        assert!(matches!(result, ComplianceResult::Valid));
    }
    
    #[test]
    fn test_hos_violation() {
        let oracle = FleetComplianceOracle::new();
        
        let result = oracle.validate_hos(12.0, 15.0, 20);
        
        assert!(matches!(result, ComplianceResult::Violation(_)));
    }
}