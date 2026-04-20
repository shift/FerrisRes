//! HullKVCache WGSL GPU path — 2D key projection + hull binary search.
//!
//! The HullKVCache projects attention to 2D coordinates, builds a convex hull,
//! and does binary search for O(log n) lookups. This module provides the
//! WGSL GPU kernels for:
//! 1. 2D projection: hidden states → 2D points (learned projection matrix)
//! 2. Convex hull search: binary search on hull edges for nearest match
//! 3. Weight retrieval: retrieve attention weights from hull vertices

#[allow(dead_code)]
/// WGSL kernel for 2D projection: hidden states → 2D points.
/// Input: hidden [seq, hidden_dim], proj_x [hidden_dim], proj_y [hidden_dim]
/// Output: points [seq, 2] (x, y coordinates)
const HULL_PROJECT_2D_WGSL: &str = r#"
    struct Params {
        seq_len: u32,
        hidden_dim: u32,
    }

    @group(0) @binding(0) var<storage, read> hidden: array<f32>;
    @group(0) @binding(1) var<storage, read> proj_x: array<f32>;
    @group(0) @binding(2) var<storage, read> proj_y: array<f32>;
    @group(0) @binding(3) var<storage, read_write> points: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;

    @compute @workgroup_size(64)
    fn project_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
        let t = gid.x;
        if (t >= params.seq_len) {
            return;
        }

        var x: f32 = 0.0;
        var y: f32 = 0.0;
        for (var d = 0u; d < params.hidden_dim; d = d + 1u) {
            let val = hidden[t * params.hidden_dim + d];
            x = x + val * proj_x[d];
            y = y + val * proj_y[d];
        }
        points[t * 2u] = x;
        points[t * 2u + 1u] = y;
    }
"#;

/// WGSL kernel for hull nearest-point binary search.
/// Input: query_point [2], hull_vertices [hull_size, 2]
/// Output: nearest_index [1], nearest_weight [1]
#[allow(dead_code)]
const HULL_SEARCH_WGSL: &str = r#"
    struct Params {
        hull_size: u32,
    }

    @group(0) @binding(0) var<storage, read> query: array<f32>;   // [2]
    @group(0) @binding(1) var<storage, read> hull: array<f32>;    // [hull_size, 2]
    @group(0) @binding(2) var<storage, read_write> result: array<f32>; // [2] (index, distance)
    @group(0) @binding(3) var<uniform> params: Params;

    fn distance_sq(px: f32, py: f32, qx: f32, qy: f32) -> f32 {
        let dx = px - qx;
        let dy = py - qy;
        return dx * dx + dy * dy;
    }

    @compute @workgroup_size(1)
    fn hull_search() {
        let qx = query[0];
        let qy = query[1];
        var best_dist: f32 = 1e30;
        var best_idx: u32 = 0u;

        // Binary search on convex hull: find the edge closest to query,
        // then find the nearest point on that edge.
        // Since hull is convex, we can use ternary search on the arc length.
        var lo: u32 = 0u;
        var hi: u32 = params.hull_size - 1u;

        // For small hulls, just linear scan
        if (params.hull_size <= 32u) {
            for (var i = 0u; i < params.hull_size; i = i + 1u) {
                let d = distance_sq(hull[i * 2u], hull[i * 2u + 1u], qx, qy);
                if (d < best_dist) {
                    best_dist = d;
                    best_idx = i;
                }
            }
        } else {
            // Ternary search on convex polygon
            for (var iter = 0u; iter < 20u; iter = iter + 1u) {
                if (hi - lo < 3u) { break; }
                let mid1 = lo + (hi - lo) / 3u;
                let mid2 = lo + 2u * (hi - lo) / 3u;
                let d1 = distance_sq(hull[mid1 * 2u], hull[mid1 * 2u + 1u], qx, qy);
                let d2 = distance_sq(hull[mid2 * 2u], hull[mid2 * 2u + 1u], qx, qy);
                if (d1 < d2) {
                    hi = mid2;
                } else {
                    lo = mid1;
                }
            }
            // Final scan in [lo, hi]
            for (var i = lo; i <= hi; i = i + 1u) {
                let d = distance_sq(hull[i * 2u], hull[i * 2u + 1u], qx, qy);
                if (d < best_dist) {
                    best_dist = d;
                    best_idx = i;
                }
            }
        }

        result[0] = f32(best_idx);
        result[1] = sqrt(best_dist);
    }
"#;

/// CPU reference for 2D projection.
pub fn hull_project_2d_cpu(
    hidden: &[f32],
    proj_x: &[f32],
    proj_y: &[f32],
    seq_len: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    let mut points = vec![0.0f32; seq_len * 2];
    for t in 0..seq_len {
        let mut x = 0.0f32;
        let mut y = 0.0f32;
        for d in 0..hidden_dim {
            let val = hidden[t * hidden_dim + d];
            x += val * proj_x[d];
            y += val * proj_y[d];
        }
        points[t * 2] = x;
        points[t * 2 + 1] = y;
    }
    points
}

/// CPU reference for hull nearest-point search.
pub fn hull_search_cpu(query: &[f32], hull: &[f32], hull_size: usize) -> (u32, f32) {
    let qx = query[0];
    let qy = query[1];
    let mut best_dist = f32::MAX;
    let mut best_idx = 0u32;

    for i in 0..hull_size {
        let px = hull[i * 2];
        let py = hull[i * 2 + 1];
        let dx = px - qx;
        let dy = py - qy;
        let dist = dx * dx + dy * dy;
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u32;
        }
    }

    (best_idx, best_dist.sqrt())
}

/// CPU reference: build convex hull from 2D points using Andrew's monotone chain.
/// Input: flat [x0, y0, x1, y1, ...] array, 2 values per point.
pub fn convex_hull_cpu(points_xy: &[f32]) -> Vec<usize> {
    let n = points_xy.len() / 2;
    if n < 3 {
        return (0..n).collect();
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let px = points_xy[a * 2];
        let qx = points_xy[b * 2];
        px.partial_cmp(&qx).unwrap_or(std::cmp::Ordering::Equal)
            .then(points_xy[a * 2 + 1].partial_cmp(&points_xy[b * 2 + 1]).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut hull = Vec::new();

    // Lower hull
    for &i in &indices {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let ax = points_xy[a * 2];
            let ay = points_xy[a * 2 + 1];
            let bx = points_xy[b * 2];
            let by = points_xy[b * 2 + 1];
            let ix = points_xy[i * 2];
            let iy = points_xy[i * 2 + 1];
            let cross = (bx - ax) * (iy - ay) - (by - ay) * (ix - ax);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(i);
    }

    // Upper hull
    let lower_len = hull.len() + 1;
    for &i in indices.iter().rev() {
        while hull.len() >= lower_len {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let ax = points_xy[a * 2];
            let ay = points_xy[a * 2 + 1];
            let bx = points_xy[b * 2];
            let by = points_xy[b * 2 + 1];
            let ix = points_xy[i * 2];
            let iy = points_xy[i * 2 + 1];
            let cross: f32 = (bx - ax) * (iy - ay) - (by - ay) * (ix - ax);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(i);
    }

    hull.pop(); // Remove duplicate last point
    hull
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_2d_basic() {
        let hidden = vec![1.0f32, 2.0, 3.0, 4.0]; // 2 tokens, dim 2
        let proj_x = vec![1.0f32, 0.0]; // x = first dim
        let proj_y = vec![0.0f32, 1.0]; // y = second dim

        let points = hull_project_2d_cpu(&hidden, &proj_x, &proj_y, 2, 2);

        assert!((points[0] - 1.0).abs() < 1e-5); // token 0 x
        assert!((points[1] - 2.0).abs() < 1e-5); // token 0 y
        assert!((points[2] - 3.0).abs() < 1e-5); // token 1 x
        assert!((points[3] - 4.0).abs() < 1e-5); // token 1 y
    }

    #[test]
    fn test_project_2d_zero_projection() {
        let hidden = vec![1.0f32, 2.0];
        let proj_x = vec![0.0f32, 0.0];
        let proj_y = vec![0.0f32, 0.0];

        let points = hull_project_2d_cpu(&hidden, &proj_x, &proj_y, 1, 2);
        assert!((points[0]).abs() < 1e-5);
        assert!((points[1]).abs() < 1e-5);
    }

    #[test]
    fn test_hull_search_nearest() {
        let query = vec![3.0f32, 3.0];
        let hull = vec![
            0.0, 0.0,  // vertex 0
            10.0, 0.0, // vertex 1
            10.0, 10.0, // vertex 2
            0.0, 10.0,  // vertex 3
            5.0, 5.0,   // vertex 4
        ];

        let (idx, dist) = hull_search_cpu(&query, &hull, 5);
        assert_eq!(idx, 4, "Should find vertex 4 (5,5) as nearest to (3,3)");
        assert!((dist - 2.0 * 2.0f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_hull_search_exact_match() {
        let query = vec![5.0f32, 5.0];
        let hull = vec![
            0.0, 0.0,
            10.0, 0.0,
            5.0, 5.0,
        ];

        let (idx, dist) = hull_search_cpu(&query, &hull, 3);
        assert_eq!(idx, 2);
        assert!(dist.abs() < 1e-5);
    }

    #[test]
    fn test_convex_hull_square() {
        let points = &[
            0.0f32, 0.0,  // point 0
            1.0, 0.0,     // point 1
            1.0, 1.0,     // point 2
            0.0, 1.0,     // point 3
            0.5, 0.5,     // point 4 (interior)
        ];

        let hull = convex_hull_cpu(points);
        assert_eq!(hull.len(), 4);
        assert!(!hull.contains(&4), "Interior point should not be in hull");
    }

    #[test]
    fn test_convex_hull_triangle() {
        let points = &[
            0.0f32, 0.0,
            2.0, 0.0,
            1.0, 2.0,
        ];

        let hull = convex_hull_cpu(points);
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn test_convex_hull_collinear() {
        let points = &[
            0.0f32, 0.0,
            1.0, 0.0,
            2.0, 0.0,
        ];

        let hull = convex_hull_cpu(points);
        assert!(hull.len() >= 2);
    }

    #[test]
    fn test_convex_hull_single_point() {
        let points = &[1.0f32, 1.0];
        let hull = convex_hull_cpu(points);
        assert_eq!(hull.len(), 1);
    }
}
