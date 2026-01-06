//! Embedding types and operations
//!
//! Provides memory-efficient vector representations with SIMD-accelerated
//! distance computations.
//!
//! # Features
//!
//! When the `embeddings` feature is enabled, this module also provides
//! `EmbedderProvider` for computing embeddings using various models via
//! the `embed_anything` crate.

use serde::{Deserialize, Serialize};

#[cfg(feature = "embeddings")]
mod provider;

#[cfg(feature = "embeddings")]
pub use provider::{
    CloudProvider, EmbedderConfig, EmbedderProvider, InferenceBackend, ModelArchitecture,
};

/// A dense embedding vector
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// The raw vector data
    data: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding from a vector
    #[must_use]
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Create a zero embedding of given dimension
    #[must_use]
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: vec![0.0; dim],
        }
    }

    /// Get the dimension of this embedding
    #[must_use]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Get the raw data as a slice
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable access to the raw data
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consume and return the underlying vector
    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Compute the L2 (Euclidean) distance squared to another embedding
    ///
    /// This avoids the sqrt for performance in nearest neighbor search.
    #[must_use]
    pub fn l2_distance_squared(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.dim(), other.dim(), "dimension mismatch");
        l2_distance_squared_simd(&self.data, &other.data)
    }

    /// Compute the cosine similarity to another embedding
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.dim(), other.dim(), "dimension mismatch");
        cosine_similarity_simd(&self.data, &other.data)
    }

    /// Compute the inner product (dot product) with another embedding
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.dim(), other.dim(), "dimension mismatch");
        dot_product_simd(&self.data, &other.data)
    }

    /// Normalize this embedding to unit length (in-place)
    pub fn normalize(&mut self) {
        let norm = self.dot(self).sqrt();
        if norm > f32::EPSILON {
            for x in &mut self.data {
                *x /= norm;
            }
        }
    }

    /// Return a normalized copy of this embedding
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(data: Vec<f32>) -> Self {
        Self::new(data)
    }
}

impl AsRef<[f32]> for Embedding {
    fn as_ref(&self) -> &[f32] {
        &self.data
    }
}

/// Compute L2 distance squared with manual SIMD-friendly loop
///
/// The compiler will auto-vectorize this on supported platforms.
#[inline]
fn l2_distance_squared_simd(a: &[f32], b: &[f32]) -> f32 {
    // Process in chunks of 8 for better vectorization
    const CHUNK_SIZE: usize = 8;
    let mut sum = 0.0f32;

    let chunks = a.len() / CHUNK_SIZE;
    let remainder = a.len() % CHUNK_SIZE;

    // Main loop - compiler will vectorize this
    for i in 0..chunks {
        let base = i * CHUNK_SIZE;
        let mut chunk_sum = 0.0f32;
        for j in 0..CHUNK_SIZE {
            let diff = a[base + j] - b[base + j];
            chunk_sum += diff * diff;
        }
        sum += chunk_sum;
    }

    // Handle remainder
    let base = chunks * CHUNK_SIZE;
    for i in 0..remainder {
        let diff = a[base + i] - b[base + i];
        sum += diff * diff;
    }

    sum
}

/// Compute dot product with SIMD-friendly loop
#[inline]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    const CHUNK_SIZE: usize = 8;
    let mut sum = 0.0f32;

    let chunks = a.len() / CHUNK_SIZE;
    let remainder = a.len() % CHUNK_SIZE;

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;
        let mut chunk_sum = 0.0f32;
        for j in 0..CHUNK_SIZE {
            chunk_sum += a[base + j] * b[base + j];
        }
        sum += chunk_sum;
    }

    let base = chunks * CHUNK_SIZE;
    for i in 0..remainder {
        sum += a[base + i] * b[base + i];
    }

    sum
}

/// Compute cosine similarity with SIMD-friendly loops
#[inline]
fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_a = dot_product_simd(a, a).sqrt();
    let norm_b = dot_product_simd(b, b).sqrt();

    if norm_a > f32::EPSILON && norm_b > f32::EPSILON {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = Embedding::new(vec![1.0, 0.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0, 0.0]);
        let dist = a.l2_distance_squared(&b);
        assert!((dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![1.0, 0.0]);
        let sim = a.cosine_similarity(&b);
        assert!((sim - 1.0).abs() < 1e-6);

        let c = Embedding::new(vec![0.0, 1.0]);
        let sim2 = a.cosine_similarity(&c);
        assert!(sim2.abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut e = Embedding::new(vec![3.0, 4.0]);
        e.normalize();
        let norm = e.dot(&e).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_zeros() {
        let e = Embedding::zeros(10);
        assert_eq!(e.dim(), 10);
        for x in e.as_slice() {
            assert_eq!(*x, 0.0);
        }
    }

    #[test]
    fn test_embedding_dim() {
        let e = Embedding::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(e.dim(), 5);
    }

    #[test]
    fn test_embedding_as_slice() {
        let e = Embedding::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(e.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_embedding_as_mut_slice() {
        let mut e = Embedding::new(vec![1.0, 2.0, 3.0]);
        e.as_mut_slice()[1] = 5.0;
        assert_eq!(e.as_slice(), &[1.0, 5.0, 3.0]);
    }

    #[test]
    fn test_embedding_into_vec() {
        let e = Embedding::new(vec![1.0, 2.0]);
        let v = e.into_vec();
        assert_eq!(v, vec![1.0, 2.0]);
    }

    #[test]
    fn test_embedding_from_vec() {
        let v = vec![1.0, 2.0, 3.0];
        let e: Embedding = v.into();
        assert_eq!(e.dim(), 3);
    }

    #[test]
    fn test_embedding_as_ref() {
        let e = Embedding::new(vec![1.0, 2.0]);
        let slice: &[f32] = e.as_ref();
        assert_eq!(slice, &[1.0, 2.0]);
    }

    #[test]
    fn test_embedding_dot_product() {
        let a = Embedding::new(vec![1.0, 2.0, 3.0]);
        let b = Embedding::new(vec![4.0, 5.0, 6.0]);
        let dot = a.dot(&b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((dot - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_normalized() {
        let e = Embedding::new(vec![3.0, 4.0]);
        let n = e.normalized();

        // Original unchanged
        assert_eq!(e.as_slice(), &[3.0, 4.0]);

        // Normalized has unit length
        let norm = n.dot(&n).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);

        // Check values: [3/5, 4/5] = [0.6, 0.8]
        assert!((n.as_slice()[0] - 0.6).abs() < 1e-6);
        assert!((n.as_slice()[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_normalize_zero_vector() {
        let mut e = Embedding::new(vec![0.0, 0.0, 0.0]);
        e.normalize();
        // Should remain zero (not NaN)
        for x in e.as_slice() {
            assert_eq!(*x, 0.0);
        }
    }

    #[test]
    fn test_embedding_l2_distance_same() {
        let a = Embedding::new(vec![1.0, 2.0, 3.0]);
        let b = Embedding::new(vec![1.0, 2.0, 3.0]);
        let dist = a.l2_distance_squared(&b);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_embedding_cosine_zero_vectors() {
        let a = Embedding::new(vec![0.0, 0.0]);
        let b = Embedding::new(vec![1.0, 0.0]);
        let sim = a.cosine_similarity(&b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_embedding_cosine_opposite() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![-1.0, 0.0]);
        let sim = a.cosine_similarity(&b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_serialization() {
        let e = Embedding::new(vec![1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&e).unwrap();
        let parsed: Embedding = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn test_embedding_clone() {
        let e = Embedding::new(vec![1.0, 2.0]);
        let c = e.clone();
        assert_eq!(e, c);
    }

    #[test]
    fn test_l2_distance_large_vectors() {
        // Test with vectors larger than CHUNK_SIZE (8)
        let a = Embedding::new(vec![1.0; 100]);
        let b = Embedding::new(vec![2.0; 100]);
        let dist = a.l2_distance_squared(&b);
        // Each diff is 1.0, squared is 1.0, 100 of them = 100.0
        assert!((dist - 100.0).abs() < 1e-4);
    }

    #[test]
    fn test_dot_product_large_vectors() {
        // Test with vectors larger than CHUNK_SIZE (8)
        let a = Embedding::new(vec![2.0; 100]);
        let b = Embedding::new(vec![3.0; 100]);
        let dot = a.dot(&b);
        // 2*3 * 100 = 600
        assert!((dot - 600.0).abs() < 1e-4);
    }

    #[test]
    fn test_cosine_large_vectors() {
        let a = Embedding::new(vec![1.0; 100]);
        let b = Embedding::new(vec![1.0; 100]);
        let sim = a.cosine_similarity(&b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_remainder_handling() {
        // Test with size that's not divisible by 8
        let a = Embedding::new(vec![1.0; 13]);
        let b = Embedding::new(vec![1.0; 13]);
        let dist = a.l2_distance_squared(&b);
        assert!(dist.abs() < 1e-6);

        let dot = a.dot(&b);
        assert!((dot - 13.0).abs() < 1e-6);
    }
}
