# Supply Chain Research — Fleet Stream Encoder, Routing Head, Compliance Oracle

## Overview
Supply chain logistics involving:
- GPS/CAN bus data (OBD-II, NMEA)
- Weather radar overlay
- Dynamic route planning
- Compliance (bridge clearance, Hours of Service, hazmat)

## GPS/CAN Bus Data Formats

### OBD-II PIDs (On-Board Diagnostics)
| PID | Name | Data |
|-----|------|------|
| 0x0D | Vehicle Speed | km/h |
| 0x11 | Throttle Position | % |
| 0x05 | Engine Coolant Temperature | °C |
| 0x2F | Fuel Tank Level | % |
| 0x66 | Odometer | km |

### NMEA Sentences
```
$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*47
$GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48
$GPRMC,123519,A,4807.038,N,01131.000,E,005.5,054.7,130424,003.1,W*6A
```

## Weather Radar Overlay

### WSR-88D NEXRAD Format
- Level II: Raw I/Q samples
- Level III: Base products ( reflectivity, velocity)
- Tile server: 256x256 grid, 1km resolution

## FleetStreamEncoder

```rust
pub struct FleetStreamEncoder {
    vehicle_id: String,
    gps_buffer: Vec<GpsPoint>,
    can_buffer: Vec<CanMessage>,
}

pub struct GpsPoint {
    lat: f64,
    lon: f64,
    alt: f64,
    heading: f32,
    speed: f32,
    timestamp_ms: u64,
}

pub struct CanMessage {
    pid: u8,
    value: f32,
    timestamp_ms: u64,
}
```

Tokenization:
```
[vehicle_id] [lat_grid] [lon_grid] [speed_cat] [heading_cat] [can_pid] [can_value]
```

## RoutingHead

### HERE/TomTom API Integration
```rust
pub struct RoutingHead {
    api_key: String,
    base_url: String,
}

impl RoutingHead {
    pub async fn get_route(&self, origin: (f64, f64), dest: (f64, f64), 
                          constraints: &RouteConstraints) -> Route {
        // POST /v1/routes
        // Include: vehicle dimensions, hazmat flags, time windows
    }
}
```

## ComplianceOracle

### Bridge Height Clearance
```rust
pub struct BridgeClearanceOracle {
    database: BridgeDatabase,  // GIS data
}

impl ComplianceOracle for BridgeClearanceOracle {
    fn validate_route(&self, route: &Route, vehicle: &Vehicle) -> ValidationResult {
        for segment in &route.segments {
            if let Some(bridge) = self.database.get_bridge(segment.coords) {
                if vehicle.height > bridge.clearance {
                    return ValidationResult::Violation(
                        format!("Bridge {} clearance: {}m < vehicle {}m", 
                                bridge.id, bridge.clearance, vehicle.height)
                    );
                }
            }
        }
        ValidationResult::Valid
    }
}
```

### FMCSA Hours of Service
- 11-hour driving limit
- 14-hour duty window
- 30-minute break required

### Hazmat Routing
- Restricted routes for explosives, flammable, oxidizers
- Tunnel restrictions
- Population density avoidance

## References
- OBD-II: https://en.wikipedia.org/wiki/OBD-II_PIDs
- NMEA: https://www.nmea.org/
- HERE Routing API: https://developer.here.com/documentation/routing-api/
- FMCSA: https://www.fmcsa.dot.gov/