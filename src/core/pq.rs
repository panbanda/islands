//! Product Quantization for vector compression
//!
//! Implements PQ encoding to reduce vector storage by 4-32x with minimal accuracy loss.

use super::distance::{Distance, DistanceMetric};
use super::error::{CoreError, CoreResult};
use rand::prelude::*;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// Configuration for Product Quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Number of subquantizers (subspaces)
    pub num_subquantizers: usize,
    /// Number of centroids per subquantizer (typically 256 for 8-bit codes)
    pub num_centroids: usize,
    /// Number of k-means iterations for training
    pub training_iterations: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            num_subquantizers: 8,
            num_centroids: 256,
            training_iterations: 25,
            seed: None,
        }
    }
}

impl PQConfig {
    /// Validate configuration
    pub fn validate(&self, dimension: usize) -> CoreResult<()> {
        if self.num_subquantizers == 0 {
            return Err(CoreError::InvalidConfig(
                "num_subquantizers must be > 0".to_string(),
            ));
        }
        if dimension % self.num_subquantizers != 0 {
            return Err(CoreError::InvalidConfig(format!(
                "dimension {} must be divisible by num_subquantizers {}",
                dimension, self.num_subquantizers
            )));
        }
        if self.num_centroids == 0 || self.num_centroids > 65536 {
            return Err(CoreError::InvalidConfig(
                "num_centroids must be in range [1, 65536]".to_string(),
            ));
        }
        Ok(())
    }

    /// Calculate bytes per encoded vector
    pub fn bytes_per_vector(&self) -> usize {
        if self.num_centroids <= 256 {
            self.num_subquantizers // 8-bit codes
        } else {
            self.num_subquantizers * 2 // 16-bit codes
        }
    }
}

/// Codebook containing centroids for one subquantizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCodebook {
    /// Centroid vectors (num_centroids x subvector_dim)
    pub centroids: Vec<Vec<f32>>,
    /// Dimension of each subvector
    pub subvector_dim: usize,
}

impl PQCodebook {
    /// Create empty codebook
    pub fn new(subvector_dim: usize) -> Self {
        Self {
            centroids: Vec::new(),
            subvector_dim,
        }
    }

    /// Find nearest centroid for a subvector
    pub fn find_nearest(&self, subvector: &[f32], metric: &DistanceMetric) -> CoreResult<usize> {
        if subvector.len() != self.subvector_dim {
            return Err(CoreError::DimensionMismatch {
                expected: self.subvector_dim,
                actual: subvector.len(),
            });
        }

        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist = metric.calculate(subvector, centroid)?;
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        Ok(best_idx)
    }

    /// Get centroid by index
    pub fn get_centroid(&self, idx: usize) -> Option<&[f32]> {
        self.centroids.get(idx).map(|v| v.as_slice())
    }
}

/// Product Quantizer for vector compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Configuration
    config: PQConfig,
    /// Codebooks for each subquantizer
    codebooks: Vec<PQCodebook>,
    /// Full vector dimension
    dimension: usize,
    /// Dimension of each subvector
    subvector_dim: usize,
    /// Distance metric
    metric: DistanceMetric,
    /// Whether the quantizer has been trained
    trained: bool,
}

impl ProductQuantizer {
    /// Create a new product quantizer
    pub fn new(dimension: usize, config: PQConfig) -> CoreResult<Self> {
        config.validate(dimension)?;

        let subvector_dim = dimension / config.num_subquantizers;
        let codebooks = (0..config.num_subquantizers)
            .map(|_| PQCodebook::new(subvector_dim))
            .collect();

        Ok(Self {
            config,
            codebooks,
            dimension,
            subvector_dim,
            metric: DistanceMetric::Euclidean,
            trained: false,
        })
    }

    /// Set distance metric
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get number of subquantizers
    pub fn num_subquantizers(&self) -> usize {
        self.config.num_subquantizers
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.dimension * 4; // f32 = 4 bytes
        let compressed_bytes = self.config.bytes_per_vector();
        original_bytes as f32 / compressed_bytes as f32
    }

    /// Train the quantizer on a set of vectors
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> CoreResult<()> {
        if vectors.is_empty() {
            return Err(CoreError::EmptyCollection);
        }

        // Validate dimensions
        for v in vectors {
            if v.len() != self.dimension {
                return Err(CoreError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.len(),
                });
            }
        }

        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Train each subquantizer independently
        for (sq_idx, codebook) in self.codebooks.iter_mut().enumerate() {
            let start = sq_idx * self.subvector_dim;
            let end = start + self.subvector_dim;

            // Extract subvectors
            let subvectors: Vec<Vec<f32>> =
                vectors.iter().map(|v| v[start..end].to_vec()).collect();

            // K-means clustering
            let centroids = kmeans(
                &subvectors,
                self.config.num_centroids,
                self.config.training_iterations,
                &self.metric,
                &mut rng,
            )?;

            codebook.centroids = centroids;
        }

        self.trained = true;
        Ok(())
    }

    /// Encode a vector to PQ codes
    pub fn encode(&self, vector: &[f32]) -> CoreResult<Vec<u16>> {
        if !self.trained {
            return Err(CoreError::PQError("Quantizer not trained".to_string()));
        }
        if vector.len() != self.dimension {
            return Err(CoreError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        let mut codes = Vec::with_capacity(self.config.num_subquantizers);

        for (sq_idx, codebook) in self.codebooks.iter().enumerate() {
            let start = sq_idx * self.subvector_dim;
            let end = start + self.subvector_dim;
            let subvector = &vector[start..end];

            let code = codebook.find_nearest(subvector, &self.metric)?;
            codes.push(code as u16);
        }

        Ok(codes)
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u16]) -> CoreResult<Vec<f32>> {
        if !self.trained {
            return Err(CoreError::PQError("Quantizer not trained".to_string()));
        }
        if codes.len() != self.config.num_subquantizers {
            return Err(CoreError::PQError(format!(
                "Expected {} codes, got {}",
                self.config.num_subquantizers,
                codes.len()
            )));
        }

        let mut vector = Vec::with_capacity(self.dimension);

        for (sq_idx, &code) in codes.iter().enumerate() {
            let centroid = self.codebooks[sq_idx]
                .get_centroid(code as usize)
                .ok_or_else(|| {
                    CoreError::PQError(format!("Invalid code {} at position {}", code, sq_idx))
                })?;
            vector.extend_from_slice(centroid);
        }

        Ok(vector)
    }

    /// Compute asymmetric distance between query and encoded vector
    /// (query is not encoded, for better accuracy)
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u16]) -> CoreResult<f32> {
        if query.len() != self.dimension {
            return Err(CoreError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let mut total_dist = 0.0f32;

        for (sq_idx, &code) in codes.iter().enumerate() {
            let start = sq_idx * self.subvector_dim;
            let end = start + self.subvector_dim;
            let query_sub = &query[start..end];

            let centroid = self.codebooks[sq_idx]
                .get_centroid(code as usize)
                .ok_or_else(|| CoreError::PQError(format!("Invalid code {}", code)))?;

            // For Euclidean, sum squared distances
            let sub_dist: f32 = query_sub
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            total_dist += sub_dist;
        }

        Ok(total_dist.sqrt())
    }

    /// Build distance lookup tables for faster batch search
    pub fn build_distance_tables(&self, query: &[f32]) -> CoreResult<Vec<Vec<f32>>> {
        if query.len() != self.dimension {
            return Err(CoreError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let mut tables = Vec::with_capacity(self.config.num_subquantizers);

        for (sq_idx, codebook) in self.codebooks.iter().enumerate() {
            let start = sq_idx * self.subvector_dim;
            let end = start + self.subvector_dim;
            let query_sub = &query[start..end];

            let table: Vec<f32> = codebook
                .centroids
                .iter()
                .map(|centroid| {
                    query_sub
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum()
                })
                .collect();

            tables.push(table);
        }

        Ok(tables)
    }

    /// Fast distance lookup using precomputed tables
    pub fn table_distance(&self, tables: &[Vec<f32>], codes: &[u16]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(sq_idx, &code)| tables[sq_idx][code as usize])
            .sum::<f32>()
            .sqrt()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> CoreResult<Vec<u8>> {
        bincode::serialize(self).map_err(|e| CoreError::Serialization(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> CoreResult<Self> {
        bincode::deserialize(bytes).map_err(|e| CoreError::Deserialization(e.to_string()))
    }
}

/// Simple k-means clustering
fn kmeans<R: Rng>(
    vectors: &[Vec<f32>],
    k: usize,
    iterations: usize,
    metric: &DistanceMetric,
    rng: &mut R,
) -> CoreResult<Vec<Vec<f32>>> {
    if vectors.is_empty() {
        return Err(CoreError::EmptyCollection);
    }

    let dim = vectors[0].len();
    let k = k.min(vectors.len());

    // Initialize with random vectors (k-means++)
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    // First centroid: random
    let first_idx = rng.r#gen::<usize>() % vectors.len();
    centroids.push(vectors[first_idx].clone());

    // Remaining centroids: weighted by distance
    while centroids.len() < k {
        let mut distances: Vec<f32> = vectors
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .map(|c| metric.calculate(v, c).unwrap_or(f32::MAX))
                    .fold(f32::MAX, f32::min)
            })
            .collect();

        // Normalize to probabilities
        let total: f32 = distances.iter().sum();
        if total > 0.0 {
            for d in &mut distances {
                *d /= total;
            }
        }

        // Weighted random selection
        let threshold: f32 = rng.r#gen();
        let mut cumsum = 0.0;
        let mut selected = 0;
        for (i, &d) in distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                selected = i;
                break;
            }
        }
        centroids.push(vectors[selected].clone());
    }

    // Iterate
    let mut assignments: Vec<usize> = vec![0; vectors.len()];

    for _ in 0..iterations {
        // Assign vectors to nearest centroid
        for (i, v) in vectors.iter().enumerate() {
            let mut best_dist = f32::MAX;
            let mut best_c = 0;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = metric.calculate(v, centroid)?;
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

        // Update centroids
        let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts: Vec<usize> = vec![0; k];

        for (i, v) in vectors.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, &val) in v.iter().enumerate() {
                new_centroids[c][j] += val;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for val in &mut new_centroids[c] {
                    *val /= counts[c] as f32;
                }
            } else {
                // Re-initialize empty clusters
                let random_vec = vectors.choose(rng).unwrap();
                new_centroids[c] = random_vec.clone();
            }
        }

        centroids = new_centroids;
    }

    Ok(centroids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rstest::rstest;

    fn create_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
            .collect()
    }

    #[test]
    fn test_config_default() {
        let config = PQConfig::default();
        assert_eq!(config.num_subquantizers, 8);
        assert_eq!(config.num_centroids, 256);
    }

    #[test]
    fn test_config_validation() {
        let mut config = PQConfig::default();

        // Valid config
        assert!(config.validate(64).is_ok()); // 64 / 8 = 8

        // Dimension not divisible
        assert!(config.validate(65).is_err());

        // Zero subquantizers
        config.num_subquantizers = 0;
        assert!(config.validate(64).is_err());

        // Invalid centroids
        config.num_subquantizers = 8;
        config.num_centroids = 0;
        assert!(config.validate(64).is_err());
    }

    #[test]
    fn test_bytes_per_vector() {
        let config = PQConfig {
            num_subquantizers: 8,
            num_centroids: 256,
            ..Default::default()
        };
        assert_eq!(config.bytes_per_vector(), 8);

        let config = PQConfig {
            num_subquantizers: 8,
            num_centroids: 1024,
            ..Default::default()
        };
        assert_eq!(config.bytes_per_vector(), 16);
    }

    #[test]
    fn test_quantizer_creation() {
        let pq = ProductQuantizer::new(64, PQConfig::default()).unwrap();
        assert_eq!(pq.dimension, 64);
        assert_eq!(pq.subvector_dim, 8);
        assert!(!pq.is_trained());
    }

    #[test]
    fn test_quantizer_invalid_dimension() {
        let result = ProductQuantizer::new(65, PQConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_training() {
        let mut pq = ProductQuantizer::new(
            32,
            PQConfig {
                num_subquantizers: 4,
                num_centroids: 16,
                training_iterations: 10,
                seed: Some(42),
            },
        )
        .unwrap();

        let vectors = create_random_vectors(100, 32, 123);
        pq.train(&vectors).unwrap();

        assert!(pq.is_trained());
        for codebook in &pq.codebooks {
            assert_eq!(codebook.centroids.len(), 16);
        }
    }

    #[test]
    fn test_training_empty() {
        let mut pq = ProductQuantizer::new(32, PQConfig::default()).unwrap();
        let result = pq.train(&[]);
        assert!(matches!(result, Err(CoreError::EmptyCollection)));
    }

    #[test]
    fn test_training_dimension_mismatch() {
        let mut pq = ProductQuantizer::new(32, PQConfig::default()).unwrap();
        let vectors = vec![vec![1.0; 64]]; // Wrong dimension
        let result = pq.train(&vectors);
        assert!(matches!(result, Err(CoreError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_encode_decode() {
        let mut pq = ProductQuantizer::new(
            32,
            PQConfig {
                num_subquantizers: 4,
                num_centroids: 16,
                training_iterations: 10,
                seed: Some(42),
            },
        )
        .unwrap();

        let vectors = create_random_vectors(50, 32, 123);
        pq.train(&vectors).unwrap();

        // Encode and decode
        let original = &vectors[0];
        let codes = pq.encode(original).unwrap();
        let reconstructed = pq.decode(&codes).unwrap();

        assert_eq!(codes.len(), 4);
        assert_eq!(reconstructed.len(), 32);

        // Check reconstruction error is bounded
        let error: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Error should be reasonable (depends on training quality)
        assert!(error < 10.0, "Reconstruction error too high: {}", error);
    }

    #[test]
    fn test_encode_not_trained() {
        let pq = ProductQuantizer::new(32, PQConfig::default()).unwrap();
        let result = pq.encode(&[1.0; 32]);
        assert!(matches!(result, Err(CoreError::PQError(_))));
    }

    #[test]
    fn test_asymmetric_distance() {
        let mut pq = ProductQuantizer::new(
            32,
            PQConfig {
                num_subquantizers: 4,
                num_centroids: 16,
                training_iterations: 10,
                seed: Some(42),
            },
        )
        .unwrap();

        let vectors = create_random_vectors(50, 32, 123);
        pq.train(&vectors).unwrap();

        let query = &vectors[0];
        let codes = pq.encode(&vectors[1]).unwrap();

        let dist = pq.asymmetric_distance(query, &codes).unwrap();
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_distance_tables() {
        let mut pq = ProductQuantizer::new(
            32,
            PQConfig {
                num_subquantizers: 4,
                num_centroids: 16,
                training_iterations: 10,
                seed: Some(42),
            },
        )
        .unwrap();

        let vectors = create_random_vectors(50, 32, 123);
        pq.train(&vectors).unwrap();

        let query = &vectors[0];
        let tables = pq.build_distance_tables(query).unwrap();

        assert_eq!(tables.len(), 4);
        for table in &tables {
            assert_eq!(table.len(), 16);
        }

        // Compare table distance with asymmetric distance
        let codes = pq.encode(&vectors[1]).unwrap();
        let table_dist = pq.table_distance(&tables, &codes);
        let asym_dist = pq.asymmetric_distance(query, &codes).unwrap();

        assert!((table_dist - asym_dist).abs() < 0.001);
    }

    #[test]
    fn test_compression_ratio() {
        let pq = ProductQuantizer::new(128, PQConfig::default()).unwrap();
        let ratio = pq.compression_ratio();
        // 128 * 4 bytes = 512, 8 bytes encoded -> 64x compression
        assert_eq!(ratio, 64.0);
    }

    #[test]
    fn test_serialization() {
        let mut pq = ProductQuantizer::new(
            32,
            PQConfig {
                num_subquantizers: 4,
                num_centroids: 16,
                training_iterations: 5,
                seed: Some(42),
            },
        )
        .unwrap();

        let vectors = create_random_vectors(30, 32, 123);
        pq.train(&vectors).unwrap();

        let bytes = pq.to_bytes().unwrap();
        let restored = ProductQuantizer::from_bytes(&bytes).unwrap();

        assert!(restored.is_trained());
        assert_eq!(restored.dimension, 32);

        // Encode with both and compare
        let codes1 = pq.encode(&vectors[0]).unwrap();
        let codes2 = restored.encode(&vectors[0]).unwrap();
        assert_eq!(codes1, codes2);
    }

    #[rstest]
    #[case(32, 4, 16)]
    #[case(64, 8, 32)]
    #[case(128, 16, 64)]
    fn test_various_configurations(
        #[case] dim: usize,
        #[case] num_sq: usize,
        #[case] num_c: usize,
    ) {
        let mut pq = ProductQuantizer::new(
            dim,
            PQConfig {
                num_subquantizers: num_sq,
                num_centroids: num_c,
                training_iterations: 5,
                seed: Some(42),
            },
        )
        .unwrap();

        let vectors = create_random_vectors(100, dim, 123);
        pq.train(&vectors).unwrap();

        let codes = pq.encode(&vectors[0]).unwrap();
        assert_eq!(codes.len(), num_sq);

        let decoded = pq.decode(&codes).unwrap();
        assert_eq!(decoded.len(), dim);
    }

    proptest! {
        #[test]
        fn prop_encode_decode_dimensions(
            dim in (8usize..64).prop_filter("div by 4", |d| d % 4 == 0),
        ) {
            let mut pq = ProductQuantizer::new(
                dim,
                PQConfig {
                    num_subquantizers: 4,
                    num_centroids: 16,
                    training_iterations: 3,
                    seed: Some(42),
                },
            ).unwrap();

            let vectors = create_random_vectors(20, dim, 123);
            pq.train(&vectors).unwrap();

            let codes = pq.encode(&vectors[0]).unwrap();
            prop_assert_eq!(codes.len(), 4);

            let decoded = pq.decode(&codes).unwrap();
            prop_assert_eq!(decoded.len(), dim);
        }

        #[test]
        fn prop_asymmetric_distance_nonnegative(
            seed in 1u64..1000,
        ) {
            let mut pq = ProductQuantizer::new(
                32,
                PQConfig {
                    num_subquantizers: 4,
                    num_centroids: 16,
                    training_iterations: 3,
                    seed: Some(seed),
                },
            ).unwrap();

            let vectors = create_random_vectors(20, 32, seed);
            pq.train(&vectors).unwrap();

            let query = &vectors[0];
            let codes = pq.encode(&vectors[1]).unwrap();
            let dist = pq.asymmetric_distance(query, &codes).unwrap();

            prop_assert!(dist >= 0.0, "Distance must be non-negative");
        }
    }

    #[test]
    fn test_codebook_find_nearest() {
        let mut codebook = PQCodebook::new(4);
        codebook.centroids = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let metric = DistanceMetric::Euclidean;

        // Query closest to first centroid
        let idx = codebook
            .find_nearest(&[0.9, 0.1, 0.0, 0.0], &metric)
            .unwrap();
        assert_eq!(idx, 0);

        // Query closest to second centroid
        let idx = codebook
            .find_nearest(&[0.1, 0.9, 0.0, 0.0], &metric)
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_kmeans_basic() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![1.1, 0.1],
            vec![0.9, -0.1],
            vec![-1.0, 0.0],
            vec![-1.1, 0.1],
            vec![-0.9, -0.1],
        ];

        let mut rng = StdRng::seed_from_u64(42);
        let centroids = kmeans(&vectors, 2, 10, &DistanceMetric::Euclidean, &mut rng).unwrap();

        assert_eq!(centroids.len(), 2);
        // Centroids should be roughly at (1, 0) and (-1, 0)
    }
}
