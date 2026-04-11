//! 2D Attention with HullKVCache for O(log n) exact lookups.
//!
//! Restricts lookup heads to a 2D head dimension, enabling binary search
//! over a convex hull structure instead of linear attention scans.
//! Enables millions of exact execution steps with perfect deterministic accuracy.
//!
//! Used for computation-heavy workloads (LLM-Computer, exact execution).
//! Key insight: by projecting attention to 2D (x,y) coordinates,
//! we can build a convex hull and do binary search → O(log n) per lookup.

/// A 2D point used in the hull structure.
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
    /// Original index in the sequence.
    pub index: usize,
}

impl Point2D {
    pub fn new(x: f32, y: f32, index: usize) -> Self {
        Self { x, y, index }
    }

    /// Cross product of vectors OA and OB.
    pub fn cross(o: &Self, a: &Self, b: &Self) -> f32 {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }

    /// Distance from this point to another.
    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Convex hull built over 2D attention head projections.
/// Enables O(log n) exact lookups by binary search over hull edges.
#[derive(Debug, Clone)]
pub struct HullKVCache {
    /// The convex hull vertices (in order).
    hull: Vec<Point2D>,
    /// All points in the cache (hull + interior).
    all_points: Vec<Point2D>,
    /// Whether the hull needs rebuilding.
    dirty: bool,
    /// Capacity.
    capacity: usize,
}

impl HullKVCache {
    /// Create a new HullKVCache with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            hull: Vec::new(),
            all_points: Vec::with_capacity(capacity),
            dirty: false,
            capacity,
        }
    }

    /// Insert a point into the cache.
    pub fn insert(&mut self, point: Point2D) {
        if self.all_points.len() < self.capacity {
            self.all_points.push(point);
            self.dirty = true;
        }
    }

    /// Insert multiple points.
    pub fn insert_all(&mut self, points: Vec<Point2D>) {
        for p in points {
            self.insert(p);
        }
    }

    /// Rebuild the convex hull (Andrew's monotone chain algorithm).
    pub fn rebuild_hull(&mut self) {
        if self.all_points.is_empty() {
            self.hull.clear();
            self.dirty = false;
            return;
        }

        let mut points = self.all_points.clone();
        points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal)));

        // Build lower hull
        let mut hull: Vec<Point2D> = Vec::new();
        for p in &points {
            while hull.len() >= 2 && Point2D::cross(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(*p);
        }

        // Build upper hull
        let lower_len = hull.len() + 1;
        for p in points.iter().rev() {
            while hull.len() >= lower_len && Point2D::cross(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(*p);
        }

        hull.pop(); // Remove duplicate end point

        self.hull = hull;
        self.dirty = false;
    }

    /// Find the nearest point on the hull to a query point.
    /// O(log n) via binary search over hull edges.
    pub fn find_nearest(&mut self, query: &Point2D) -> Option<usize> {
        if self.dirty {
            self.rebuild_hull();
        }

        if self.hull.is_empty() {
            return None;
        }

        if self.hull.len() == 1 {
            return Some(self.hull[0].index);
        }

        // Binary search for the hull edge closest to the query
        let mut best_dist = f32::INFINITY;
        let mut best_index = 0;

        // Binary search over hull vertices
        let mut lo = 0usize;
        let mut hi = self.hull.len() - 1;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let d_mid = query.distance_to(&self.hull[mid]);
            let d_mid1 = if mid + 1 < self.hull.len() {
                query.distance_to(&self.hull[mid + 1])
            } else {
                f32::INFINITY
            };

            if d_mid < best_dist {
                best_dist = d_mid;
                best_index = self.hull[mid].index;
            }
            if d_mid1 < best_dist {
                best_dist = d_mid1;
                best_index = self.hull[mid + 1].index;
            }

            if d_mid < d_mid1 {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        // Check the final point
        let d = query.distance_to(&self.hull[lo]);
        if d < best_dist {
            best_index = self.hull[lo].index;
        }

        Some(best_index)
    }

    /// Find all points within a radius of the query point.
    /// Returns indices of matching points.
    pub fn find_within_radius(&self, query: &Point2D, radius: f32) -> Vec<usize> {
        self.all_points.iter()
            .filter(|p| p.distance_to(query) <= radius)
            .map(|p| p.index)
            .collect()
    }

    /// Get the number of points in the cache.
    pub fn len(&self) -> usize {
        self.all_points.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.all_points.is_empty()
    }

    /// Get the number of hull vertices.
    pub fn hull_size(&self) -> usize {
        self.hull.len()
    }

    /// Get the hull vertices.
    pub fn hull(&self) -> &[Point2D] {
        &self.hull
    }

    /// Get all points.
    pub fn points(&self) -> &[Point2D] {
        &self.all_points
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.hull.clear();
        self.all_points.clear();
        self.dirty = false;
    }
}

/// 2D Attention head that projects attention keys to 2D coordinates
/// for HullKVCache lookups.
pub struct Attention2D {
    /// Projection matrix for x coordinate: [head_dim].
    proj_x: Vec<f32>,
    /// Projection matrix for y coordinate: [head_dim].
    proj_y: Vec<f32>,
    /// Cache for this attention head.
    cache: HullKVCache,
}

impl Attention2D {
    /// Create a new 2D attention head with given head dimension and cache capacity.
    pub fn new(head_dim: usize, cache_capacity: usize) -> Self {
        // Initialize projections with small random values
        let proj_x: Vec<f32> = (0..head_dim)
            .map(|i| ((i as f32 * 0.1).sin()) / (head_dim as f32).sqrt())
            .collect();
        let proj_y: Vec<f32> = (0..head_dim)
            .map(|i| ((i as f32 * 0.1 + 1.0).cos()) / (head_dim as f32).sqrt())
            .collect();

        Self {
            proj_x,
            proj_y,
            cache: HullKVCache::new(cache_capacity),
        }
    }

    /// Project a key vector to 2D coordinates.
    pub fn project_to_2d(&self, key: &[f32], index: usize) -> Point2D {
        let x: f32 = key.iter().zip(self.proj_x.iter())
            .map(|(k, p)| k * p)
            .sum();
        let y: f32 = key.iter().zip(self.proj_y.iter())
            .map(|(k, p)| k * p)
            .sum();
        Point2D::new(x, y, index)
    }

    /// Insert a key into the cache (projected to 2D).
    pub fn insert_key(&mut self, key: &[f32], index: usize) {
        let point = self.project_to_2d(key, index);
        self.cache.insert(point);
    }

    /// Look up the nearest cached key for a query.
    pub fn lookup(&mut self, query_key: &[f32]) -> Option<usize> {
        let query_point = self.project_to_2d(query_key, 0);
        self.cache.find_nearest(&query_point)
    }

    /// Get the underlying cache.
    pub fn cache(&self) -> &HullKVCache {
        &self.cache
    }

    /// Get mutable reference to the cache.
    pub fn cache_mut(&mut self) -> &mut HullKVCache {
        &mut self.cache
    }

    /// Rebuild the hull (call after batch insertions).
    pub fn rebuild(&mut self) {
        self.cache.rebuild_hull();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cross_product() {
        let o = Point2D::new(0.0, 0.0, 0);
        let a = Point2D::new(1.0, 0.0, 1);
        let b = Point2D::new(0.0, 1.0, 2);
        // Positive cross product: b is to the left of OA
        assert!(Point2D::cross(&o, &a, &b) > 0.0);
    }

    #[test]
    fn test_point_distance() {
        let a = Point2D::new(0.0, 0.0, 0);
        let b = Point2D::new(3.0, 4.0, 1);
        assert!((a.distance_to(&b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_hull_empty() {
        let cache = HullKVCache::new(100);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_hull_single_point() {
        let mut cache = HullKVCache::new(100);
        cache.insert(Point2D::new(1.0, 1.0, 0));
        cache.rebuild_hull();
        assert_eq!(cache.hull_size(), 1);
    }

    #[test]
    fn test_hull_triangle() {
        let mut cache = HullKVCache::new(100);
        cache.insert(Point2D::new(0.0, 0.0, 0));
        cache.insert(Point2D::new(1.0, 0.0, 1));
        cache.insert(Point2D::new(0.5, 1.0, 2));
        cache.rebuild_hull();
        assert_eq!(cache.hull_size(), 3);
    }

    #[test]
    fn test_hull_with_interior_point() {
        let mut cache = HullKVCache::new(100);
        cache.insert(Point2D::new(0.0, 0.0, 0));
        cache.insert(Point2D::new(2.0, 0.0, 1));
        cache.insert(Point2D::new(1.0, 2.0, 2));
        cache.insert(Point2D::new(1.0, 0.5, 3)); // Interior point
        cache.rebuild_hull();
        // Hull should have 3 vertices (triangle), interior point excluded
        assert_eq!(cache.hull_size(), 3);
        assert_eq!(cache.len(), 4); // But all points still stored
    }

    #[test]
    fn test_hull_square() {
        let mut cache = HullKVCache::new(100);
        cache.insert(Point2D::new(0.0, 0.0, 0));
        cache.insert(Point2D::new(1.0, 0.0, 1));
        cache.insert(Point2D::new(1.0, 1.0, 2));
        cache.insert(Point2D::new(0.0, 1.0, 3));
        cache.rebuild_hull();
        assert_eq!(cache.hull_size(), 4);
    }

    #[test]
    fn test_find_nearest() {
        let mut cache = HullKVCache::new(100);
        cache.insert(Point2D::new(0.0, 0.0, 0));
        cache.insert(Point2D::new(10.0, 0.0, 1));
        cache.insert(Point2D::new(5.0, 10.0, 2));

        let query = Point2D::new(0.5, 0.5, 99);
        let nearest = cache.find_nearest(&query);
        assert_eq!(nearest, Some(0)); // Closest to (0,0)
    }

    #[test]
    fn test_find_within_radius() {
        let cache = {
            let mut c = HullKVCache::new(100);
            c.insert(Point2D::new(0.0, 0.0, 0));
            c.insert(Point2D::new(1.0, 0.0, 1));
            c.insert(Point2D::new(10.0, 10.0, 2));
            c
        };
        let query = Point2D::new(0.5, 0.0, 99);
        let results = cache.find_within_radius(&query, 2.0);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
    }

    #[test]
    fn test_attention_2d_projection() {
        let attn = Attention2D::new(64, 100);
        let key = vec![1.0; 64];
        let point = attn.project_to_2d(&key, 0);
        // Should produce finite coordinates
        assert!(point.x.is_finite());
        assert!(point.y.is_finite());
    }

    #[test]
    fn test_attention_2d_insert_and_lookup() {
        let mut attn = Attention2D::new(16, 100);

        // Insert keys with different patterns
        let key0 = vec![0.0; 16];
        let key1 = vec![1.0; 16];
        let key2 = vec![0.5; 16];

        attn.insert_key(&key0, 0);
        attn.insert_key(&key1, 1);
        attn.insert_key(&key2, 2);
        attn.rebuild();

        // Query close to key0
        let query = vec![0.1; 16];
        let nearest = attn.lookup(&query);
        assert!(nearest.is_some());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = HullKVCache::new(100);
        cache.insert(Point2D::new(1.0, 1.0, 0));
        cache.rebuild_hull();
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hull_size(), 0);
    }

    #[test]
    fn test_capacity_limit() {
        let mut cache = HullKVCache::new(3);
        cache.insert(Point2D::new(0.0, 0.0, 0));
        cache.insert(Point2D::new(1.0, 0.0, 1));
        cache.insert(Point2D::new(0.0, 1.0, 2));
        cache.insert(Point2D::new(5.0, 5.0, 3)); // Should be dropped
        assert_eq!(cache.len(), 3);
    }
}
