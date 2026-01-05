//! Vector distance metrics for similarity search

use crate::error::{CoreError, CoreResult};
use serde::{Deserialize, Serialize};

/// Distance metric types supported by the search engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine)
    #[default]
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Dot product (negative for max similarity)
    DotProduct,
    /// Manhattan (L1) distance
    Manhattan,
}

/// Distance calculation trait for vector operations
pub trait Distance: Send + Sync {
    /// Calculate distance between two vectors
    fn calculate(&self, a: &[f32], b: &[f32]) -> CoreResult<f32>;

    /// Calculate squared distance (optimization for Euclidean)
    fn calculate_squared(&self, a: &[f32], b: &[f32]) -> CoreResult<f32> {
        self.calculate(a, b).map(|d| d * d)
    }

    /// Batch calculate distances from query to multiple vectors
    fn batch_calculate(&self, query: &[f32], vectors: &[&[f32]]) -> CoreResult<Vec<f32>> {
        vectors.iter().map(|v| self.calculate(query, v)).collect()
    }
}

impl Distance for DistanceMetric {
    fn calculate(&self, a: &[f32], b: &[f32]) -> CoreResult<f32> {
        if a.len() != b.len() {
            return Err(CoreError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        match self {
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::DotProduct => dot_product_distance(a, b),
            DistanceMetric::Manhattan => manhattan_distance(a, b),
        }
    }

    fn calculate_squared(&self, a: &[f32], b: &[f32]) -> CoreResult<f32> {
        if a.len() != b.len() {
            return Err(CoreError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        match self {
            DistanceMetric::Euclidean => euclidean_distance_squared(a, b),
            _ => self.calculate(a, b).map(|d| d * d),
        }
    }
}

/// Calculate cosine distance (1 - cosine_similarity)
#[inline]
fn cosine_distance(a: &[f32], b: &[f32]) -> CoreResult<f32> {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let norm = (norm_a * norm_b).sqrt();
    if norm == 0.0 {
        return Ok(1.0); // Maximum distance for zero vectors
    }

    Ok(1.0 - (dot / norm))
}

/// Calculate Euclidean distance
#[inline]
fn euclidean_distance(a: &[f32], b: &[f32]) -> CoreResult<f32> {
    euclidean_distance_squared(a, b).map(|d| d.sqrt())
}

/// Calculate squared Euclidean distance (avoids sqrt for comparisons)
#[inline]
fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> CoreResult<f32> {
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();
    Ok(sum)
}

/// Calculate negative dot product (for maximum similarity)
#[inline]
fn dot_product_distance(a: &[f32], b: &[f32]) -> CoreResult<f32> {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    Ok(-dot) // Negative so lower is better
}

/// Calculate Manhattan (L1) distance
#[inline]
fn manhattan_distance(a: &[f32], b: &[f32]) -> CoreResult<f32> {
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    Ok(sum)
}

/// Normalize a vector to unit length (in-place)
pub fn normalize_vector(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Create a normalized copy of a vector
pub fn normalized(v: &[f32]) -> Vec<f32> {
    let mut result = v.to_vec();
    normalize_vector(&mut result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rstest::rstest;
    use test_case::test_case;

    // Basic distance calculation tests
    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let dist = DistanceMetric::Cosine.calculate(&v, &v).unwrap();
        assert!(
            (dist - 0.0).abs() < 1e-6,
            "Identical vectors should have distance 0"
        );
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = DistanceMetric::Cosine.calculate(&a, &b).unwrap();
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "Orthogonal vectors should have distance 1"
        );
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let dist = DistanceMetric::Cosine.calculate(&a, &b).unwrap();
        assert!(
            (dist - 2.0).abs() < 1e-6,
            "Opposite vectors should have distance 2"
        );
    }

    #[test]
    fn test_euclidean_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let dist = DistanceMetric::Euclidean.calculate(&v, &v).unwrap();
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_unit_vectors() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = DistanceMetric::Euclidean.calculate(&a, &b).unwrap();
        assert!(
            (dist - 5.0).abs() < 1e-6,
            "Distance should be 5 (3-4-5 triangle)"
        );
    }

    #[test_case(vec![1.0, 0.0], vec![0.0, 1.0], DistanceMetric::Euclidean => 1.414213562373095; "euclidean unit orthogonal")]
    #[test_case(vec![1.0, 1.0], vec![1.0, 1.0], DistanceMetric::Manhattan => 0.0; "manhattan identical")]
    #[test_case(vec![0.0, 0.0], vec![3.0, 4.0], DistanceMetric::Manhattan => 7.0; "manhattan 3-4")]
    fn test_distance_cases(a: Vec<f32>, b: Vec<f32>, metric: DistanceMetric) -> f32 {
        metric.calculate(&a, &b).unwrap()
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = DistanceMetric::Cosine.calculate(&a, &b);
        assert!(matches!(result, Err(CoreError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_zero_vector_cosine() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let dist = DistanceMetric::Cosine.calculate(&a, &b).unwrap();
        assert_eq!(dist, 1.0, "Zero vector should have max distance");
    }

    #[test]
    fn test_dot_product_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // dot product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let dist = DistanceMetric::DotProduct.calculate(&a, &b).unwrap();
        assert!((dist - (-32.0)).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_vector() {
        let mut v = vec![3.0, 4.0];
        normalize_vector(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Normalized vector should have unit norm"
        );
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize_vector(&mut v);
        // Should not panic, vector remains zero
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_batch_calculate() {
        let query = vec![1.0, 0.0];
        let vectors: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0], &[-1.0, 0.0]];
        let distances = DistanceMetric::Cosine
            .batch_calculate(&query, &vectors)
            .unwrap();
        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 0.0).abs() < 1e-6); // Identical
        assert!((distances[1] - 1.0).abs() < 1e-6); // Orthogonal
        assert!((distances[2] - 2.0).abs() < 1e-6); // Opposite
    }

    // Property-based tests
    proptest! {
        #[test]
        fn prop_distance_non_negative(
            a in proptest::collection::vec(-100.0f32..100.0, 8..16),
            b in proptest::collection::vec(-100.0f32..100.0, 8..16),
        ) {
            // Only test if dimensions match
            if a.len() == b.len() {
                let dist = DistanceMetric::Euclidean.calculate(&a, &b).unwrap();
                prop_assert!(dist >= 0.0, "Euclidean distance must be non-negative");

                let dist = DistanceMetric::Manhattan.calculate(&a, &b).unwrap();
                prop_assert!(dist >= 0.0, "Manhattan distance must be non-negative");
            }
        }

        #[test]
        fn prop_distance_symmetric(
            a in proptest::collection::vec(-10.0f32..10.0, 8),
            b in proptest::collection::vec(-10.0f32..10.0, 8),
        ) {
            let dist_ab = DistanceMetric::Euclidean.calculate(&a, &b).unwrap();
            let dist_ba = DistanceMetric::Euclidean.calculate(&b, &a).unwrap();
            prop_assert!((dist_ab - dist_ba).abs() < 1e-5, "Distance should be symmetric");
        }

        #[test]
        fn prop_distance_identity(
            v in proptest::collection::vec(-10.0f32..10.0, 8),
        ) {
            let dist = DistanceMetric::Euclidean.calculate(&v, &v).unwrap();
            prop_assert!((dist - 0.0).abs() < 1e-6, "Distance to self should be 0");
        }

        #[test]
        fn prop_triangle_inequality(
            a in proptest::collection::vec(-5.0f32..5.0, 4),
            b in proptest::collection::vec(-5.0f32..5.0, 4),
            c in proptest::collection::vec(-5.0f32..5.0, 4),
        ) {
            let ab = DistanceMetric::Euclidean.calculate(&a, &b).unwrap();
            let bc = DistanceMetric::Euclidean.calculate(&b, &c).unwrap();
            let ac = DistanceMetric::Euclidean.calculate(&a, &c).unwrap();
            // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
            prop_assert!(ac <= ab + bc + 1e-5, "Triangle inequality violated");
        }

        #[test]
        fn prop_cosine_bounded(
            a in proptest::collection::vec(0.1f32..10.0, 8),
            b in proptest::collection::vec(0.1f32..10.0, 8),
        ) {
            let dist = DistanceMetric::Cosine.calculate(&a, &b).unwrap();
            prop_assert!(dist >= 0.0 && dist <= 2.0, "Cosine distance should be in [0, 2]");
        }

        #[test]
        fn prop_normalized_unit_length(
            v in proptest::collection::vec(0.1f32..10.0, 8),
        ) {
            let norm_v = normalized(&v);
            let length: f32 = norm_v.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!((length - 1.0).abs() < 1e-5, "Normalized vector should have unit length");
        }
    }

    // Rstest parameterized tests
    #[rstest]
    #[case(DistanceMetric::Cosine)]
    #[case(DistanceMetric::Euclidean)]
    #[case(DistanceMetric::DotProduct)]
    #[case(DistanceMetric::Manhattan)]
    fn test_all_metrics_handle_empty_equal_vectors(#[case] metric: DistanceMetric) {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let result = metric.calculate(&a, &a);
        assert!(result.is_ok());
    }

    #[rstest]
    #[case(DistanceMetric::Cosine)]
    #[case(DistanceMetric::Euclidean)]
    #[case(DistanceMetric::DotProduct)]
    #[case(DistanceMetric::Manhattan)]
    fn test_all_metrics_reject_dimension_mismatch(#[case] metric: DistanceMetric) {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = metric.calculate(&a, &b);
        assert!(matches!(result, Err(CoreError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_calculate_squared_euclidean() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist_sq = DistanceMetric::Euclidean.calculate_squared(&a, &b).unwrap();
        assert!((dist_sq - 25.0).abs() < 1e-6, "Squared distance should be 25");
    }

    #[test]
    fn test_calculate_squared_cosine() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = DistanceMetric::Cosine.calculate(&a, &b).unwrap();
        let dist_sq = DistanceMetric::Cosine.calculate_squared(&a, &b).unwrap();
        assert!((dist_sq - dist * dist).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_squared_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = DistanceMetric::Euclidean.calculate_squared(&a, &b);
        assert!(matches!(result, Err(CoreError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_normalized_function() {
        let v = vec![3.0, 4.0];
        let norm_v = normalized(&v);
        assert_eq!(norm_v.len(), 2);
        let length: f32 = norm_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_calculate_empty() {
        let query = vec![1.0, 0.0];
        let vectors: Vec<&[f32]> = vec![];
        let distances = DistanceMetric::Cosine.batch_calculate(&query, &vectors).unwrap();
        assert!(distances.is_empty());
    }

    #[test]
    fn test_distance_metric_default() {
        let metric = DistanceMetric::default();
        assert_eq!(metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_distance_metric_serialize() {
        let metric = DistanceMetric::Euclidean;
        let json = serde_json::to_string(&metric).unwrap();
        assert_eq!(json, "\"euclidean\"");
    }

    #[test]
    fn test_distance_metric_deserialize() {
        let metric: DistanceMetric = serde_json::from_str("\"manhattan\"").unwrap();
        assert_eq!(metric, DistanceMetric::Manhattan);
    }

    #[test]
    fn test_distance_metric_clone() {
        let metric = DistanceMetric::DotProduct;
        let cloned = metric.clone();
        assert_eq!(metric, cloned);
    }

    #[test]
    fn test_distance_metric_copy() {
        let metric = DistanceMetric::Cosine;
        let copied: DistanceMetric = metric;
        assert_eq!(copied, DistanceMetric::Cosine);
    }

    #[test]
    fn test_distance_metric_debug() {
        let metric = DistanceMetric::Euclidean;
        let debug = format!("{:?}", metric);
        assert_eq!(debug, "Euclidean");
    }
}
