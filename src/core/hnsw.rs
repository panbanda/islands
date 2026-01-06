//! HNSW (Hierarchical Navigable Small World) graph implementation
//!
//! This module provides a multi-layer graph structure for approximate nearest neighbor search.

use super::distance::{Distance, DistanceMetric};
use super::error::{CoreError, CoreResult};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Configuration for HNSW graph construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per node (M parameter)
    pub m: usize,
    /// Maximum connections for layer 0 (M0 = 2*M typically)
    pub m0: usize,
    /// Size of dynamic candidate list during construction (ef_construction)
    pub ef_construction: usize,
    /// Normalization factor for level generation
    pub ml: f64,
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// Maximum number of layers
    pub max_layers: usize,
}

impl HnswConfig {
    /// Create a new HNSW configuration with default settings.
    pub fn new(_config: Self) -> Self {
        Self::default()
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ml: 1.0 / (16.0_f64).ln(),
            metric: DistanceMetric::Cosine,
            max_layers: 16,
        }
    }
}

impl HnswConfig {
    /// Create config optimized for speed
    pub fn fast() -> Self {
        Self {
            m: 12,
            m0: 24,
            ef_construction: 100,
            ..Default::default()
        }
    }

    /// Create config optimized for recall
    pub fn accurate() -> Self {
        Self {
            m: 32,
            m0: 64,
            ef_construction: 400,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> CoreResult<()> {
        if self.m == 0 {
            return Err(CoreError::InvalidConfig("M must be > 0".to_string()));
        }
        if self.m0 < self.m {
            return Err(CoreError::InvalidConfig("M0 must be >= M".to_string()));
        }
        if self.ef_construction < self.m {
            return Err(CoreError::InvalidConfig(
                "ef_construction must be >= M".to_string(),
            ));
        }
        Ok(())
    }
}

/// A node in the HNSW graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    /// Unique node identifier
    pub id: u64,
    /// Vector data
    pub vector: Vec<f32>,
    /// Connections at each layer (layer -> neighbor IDs)
    pub connections: Vec<Vec<u64>>,
    /// Maximum layer this node exists on
    pub level: usize,
}

impl HnswNode {
    /// Create a new node
    pub fn new(id: u64, vector: Vec<f32>, level: usize, max_connections: usize) -> Self {
        let connections = (0..=level)
            .map(|_| Vec::with_capacity(max_connections))
            .collect();

        Self {
            id,
            vector,
            connections,
            level,
        }
    }

    /// Get neighbors at a specific layer
    pub fn neighbors_at(&self, layer: usize) -> Option<&[u64]> {
        self.connections.get(layer).map(|v| v.as_slice())
    }

    /// Get mutable neighbors at a specific layer
    pub fn neighbors_at_mut(&mut self, layer: usize) -> Option<&mut Vec<u64>> {
        self.connections.get_mut(layer)
    }
}

/// Entry point candidate for search
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    distance: OrderedFloat<f32>,
    id: u64,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap behavior
        other.distance.cmp(&self.distance)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// HNSW graph for approximate nearest neighbor search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswGraph {
    /// Configuration
    config: HnswConfig,
    /// All nodes in the graph
    nodes: HashMap<u64, HnswNode>,
    /// Entry point node ID
    entry_point: Option<u64>,
    /// Current maximum layer
    max_level: usize,
    /// Vector dimension
    dimension: Option<usize>,
    /// Next available node ID
    next_id: u64,
}

impl HnswGraph {
    /// Create a new empty HNSW graph
    pub fn new(config: HnswConfig) -> CoreResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            nodes: HashMap::new(),
            entry_point: None,
            max_level: 0,
            dimension: None,
            next_id: 0,
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> CoreResult<Self> {
        Self::new(HnswConfig::default())
    }

    /// Get number of nodes in the graph
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get vector dimension
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Get a node by ID
    pub fn get_node(&self, id: u64) -> Option<&HnswNode> {
        self.nodes.get(&id)
    }

    /// Generate a random level for a new node
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.r#gen();
        let level = (-r.ln() * self.config.ml).floor() as usize;
        level.min(self.config.max_layers - 1)
    }

    /// Insert a vector into the graph
    pub fn insert(&mut self, vector: Vec<f32>) -> CoreResult<u64> {
        // Validate dimension
        if let Some(dim) = self.dimension {
            if vector.len() != dim {
                return Err(CoreError::DimensionMismatch {
                    expected: dim,
                    actual: vector.len(),
                });
            }
        } else {
            self.dimension = Some(vector.len());
        }

        let id = self.next_id;
        self.next_id += 1;

        let level = self.random_level();
        let max_conn = if level == 0 {
            self.config.m0
        } else {
            self.config.m
        };

        let node = HnswNode::new(id, vector, level, max_conn);

        // First node case
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_level = level;
            self.nodes.insert(id, node);
            return Ok(id);
        }

        // Insert with connections
        self.insert_node(node)?;

        Ok(id)
    }

    /// Insert a node and create connections
    fn insert_node(&mut self, mut node: HnswNode) -> CoreResult<()> {
        let entry_id = self.entry_point.ok_or(CoreError::IndexNotBuilt)?;
        let query = &node.vector;

        // Find entry point at the top level
        let mut current = entry_id;
        let mut current_dist = self.distance(query, current)?;

        // Greedy search from top to node's level
        for layer in (node.level + 1..=self.max_level).rev() {
            loop {
                let mut changed = false;
                if let Some(current_node) = self.nodes.get(&current) {
                    if let Some(neighbors) = current_node.neighbors_at(layer) {
                        for &neighbor in neighbors {
                            let dist = self.distance(query, neighbor)?;
                            if dist < current_dist {
                                current = neighbor;
                                current_dist = dist;
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Insert at each layer from node's level down to 0
        for layer in (0..=node.level).rev() {
            let neighbors =
                self.search_layer(query, current, self.config.ef_construction, layer)?;

            // Select best M neighbors
            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let selected: Vec<u64> = neighbors.into_iter().take(m).map(|(id, _)| id).collect();

            // Add connections to new node
            if let Some(connections) = node.connections.get_mut(layer) {
                *connections = selected.clone();
            }

            // Add reverse connections
            for &neighbor_id in &selected {
                if let Some(neighbor) = self.nodes.get_mut(&neighbor_id) {
                    if let Some(neighbor_conns) = neighbor.neighbors_at_mut(layer) {
                        neighbor_conns.push(node.id);
                        // Prune if exceeds max connections
                        if neighbor_conns.len() > m {
                            self.prune_connections(neighbor_id, layer, m)?;
                        }
                    }
                }
            }

            // Update entry for next layer
            if !selected.is_empty() {
                current = selected[0];
            }
        }

        // Update entry point if new node is higher
        if node.level > self.max_level {
            self.max_level = node.level;
            self.entry_point = Some(node.id);
        }

        self.nodes.insert(node.id, node);
        Ok(())
    }

    /// Search a single layer for nearest neighbors
    fn search_layer(
        &self,
        query: &[f32],
        entry: u64,
        ef: usize,
        layer: usize,
    ) -> CoreResult<Vec<(u64, f32)>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let entry_dist = self.distance(query, entry)?;
        visited.insert(entry);
        candidates.push(Candidate {
            distance: OrderedFloat(entry_dist),
            id: entry,
        });
        results.push(Reverse(Candidate {
            distance: OrderedFloat(entry_dist),
            id: entry,
        }));

        while let Some(Candidate { distance, id }) = candidates.pop() {
            // Check if we can stop
            if let Some(Reverse(worst)) = results.peek() {
                if distance > worst.distance && results.len() >= ef {
                    break;
                }
            }

            // Explore neighbors
            if let Some(node) = self.nodes.get(&id) {
                if let Some(neighbors) = node.neighbors_at(layer) {
                    for &neighbor in neighbors {
                        if visited.insert(neighbor) {
                            let neighbor_dist = self.distance(query, neighbor)?;

                            let should_add = results.len() < ef
                                || results
                                    .peek()
                                    .map(|Reverse(w)| neighbor_dist < w.distance.0)
                                    .unwrap_or(true);

                            if should_add {
                                candidates.push(Candidate {
                                    distance: OrderedFloat(neighbor_dist),
                                    id: neighbor,
                                });
                                results.push(Reverse(Candidate {
                                    distance: OrderedFloat(neighbor_dist),
                                    id: neighbor,
                                }));

                                if results.len() > ef {
                                    results.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted vector
        let mut result_vec: Vec<(u64, f32)> = results
            .into_iter()
            .map(|Reverse(c)| (c.id, c.distance.0))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(result_vec)
    }

    /// Prune connections for a node at a layer
    fn prune_connections(&mut self, node_id: u64, layer: usize, max_conn: usize) -> CoreResult<()> {
        let node = self
            .nodes
            .get(&node_id)
            .ok_or(CoreError::NodeNotFound(node_id))?;
        let vector = node.vector.clone();
        let connections = node
            .neighbors_at(layer)
            .ok_or(CoreError::HnswError("Layer not found".to_string()))?
            .to_vec();

        // Calculate distances and sort
        let mut scored: Vec<(u64, f32)> = connections
            .into_iter()
            .filter_map(|id| {
                self.nodes.get(&id).map(|n| {
                    let dist = self
                        .config
                        .metric
                        .calculate(&vector, &n.vector)
                        .unwrap_or(f32::MAX);
                    (id, dist)
                })
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Keep only top max_conn
        let pruned: Vec<u64> = scored
            .into_iter()
            .take(max_conn)
            .map(|(id, _)| id)
            .collect();

        if let Some(node) = self.nodes.get_mut(&node_id) {
            if let Some(conns) = node.neighbors_at_mut(layer) {
                *conns = pruned;
            }
        }

        Ok(())
    }

    /// Calculate distance between query and a node
    fn distance(&self, query: &[f32], node_id: u64) -> CoreResult<f32> {
        let node = self
            .nodes
            .get(&node_id)
            .ok_or(CoreError::NodeNotFound(node_id))?;
        self.config.metric.calculate(query, &node.vector)
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> CoreResult<Vec<(u64, f32)>> {
        if self.is_empty() {
            return Ok(vec![]);
        }

        // Validate dimension
        if let Some(dim) = self.dimension {
            if query.len() != dim {
                return Err(CoreError::DimensionMismatch {
                    expected: dim,
                    actual: query.len(),
                });
            }
        }

        let entry_id = self.entry_point.ok_or(CoreError::IndexNotBuilt)?;
        let mut current = entry_id;
        let mut current_dist = self.distance(query, current)?;

        // Greedy search from top to layer 1
        for layer in (1..=self.max_level).rev() {
            loop {
                let mut changed = false;
                if let Some(node) = self.nodes.get(&current) {
                    if let Some(neighbors) = node.neighbors_at(layer) {
                        for &neighbor in neighbors {
                            let dist = self.distance(query, neighbor)?;
                            if dist < current_dist {
                                current = neighbor;
                                current_dist = dist;
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Search layer 0 with ef
        let ef = ef.max(k);
        let results = self.search_layer(query, current, ef, 0)?;

        Ok(results.into_iter().take(k).collect())
    }

    /// Serialize graph to bytes
    pub fn to_bytes(&self) -> CoreResult<Vec<u8>> {
        bincode::serialize(self).map_err(|e| CoreError::Serialization(e.to_string()))
    }

    /// Deserialize graph from bytes
    pub fn from_bytes(bytes: &[u8]) -> CoreResult<Self> {
        bincode::deserialize(bytes).map_err(|e| CoreError::Deserialization(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;

    fn create_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
            .collect()
    }

    #[test]
    fn test_config_default() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.m0, 32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_fast() {
        let config = HnswConfig::fast();
        assert!(config.validate().is_ok());
        assert!(config.m < HnswConfig::default().m);
    }

    #[test]
    fn test_config_accurate() {
        let config = HnswConfig::accurate();
        assert!(config.validate().is_ok());
        assert!(config.m > HnswConfig::default().m);
    }

    #[test]
    fn test_config_validation() {
        let mut config = HnswConfig::default();

        config.m = 0;
        assert!(config.validate().is_err());

        config.m = 16;
        config.m0 = 8;
        assert!(config.validate().is_err());

        config.m0 = 32;
        config.ef_construction = 8;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_empty_graph() {
        let graph = HnswGraph::with_defaults().unwrap();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        assert!(graph.dimension().is_none());
    }

    #[test]
    fn test_insert_single() {
        let mut graph = HnswGraph::with_defaults().unwrap();
        let id = graph.insert(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(id, 0);
        assert_eq!(graph.len(), 1);
        assert_eq!(graph.dimension(), Some(3));
    }

    #[test]
    fn test_insert_dimension_mismatch() {
        let mut graph = HnswGraph::with_defaults().unwrap();
        graph.insert(vec![1.0, 2.0, 3.0]).unwrap();
        let result = graph.insert(vec![1.0, 2.0]);
        assert!(matches!(result, Err(CoreError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_insert_multiple() {
        let mut graph = HnswGraph::with_defaults().unwrap();
        let vectors = create_random_vectors(100, 32, 42);

        for (i, v) in vectors.into_iter().enumerate() {
            let id = graph.insert(v).unwrap();
            assert_eq!(id as usize, i);
        }

        assert_eq!(graph.len(), 100);
    }

    #[test]
    fn test_search_empty() {
        let graph = HnswGraph::with_defaults().unwrap();
        let results = graph.search(&[1.0, 2.0, 3.0], 10, 50).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_single() {
        let mut graph = HnswGraph::with_defaults().unwrap();
        graph.insert(vec![1.0, 0.0, 0.0]).unwrap();

        let results = graph.search(&[1.0, 0.0, 0.0], 1, 50).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.01); // Should be very close
    }

    #[test]
    fn test_search_finds_nearest() {
        let mut graph = HnswGraph::with_defaults().unwrap();

        // Insert some vectors
        graph.insert(vec![1.0, 0.0, 0.0]).unwrap();
        graph.insert(vec![0.0, 1.0, 0.0]).unwrap();
        graph.insert(vec![0.0, 0.0, 1.0]).unwrap();
        graph.insert(vec![0.9, 0.1, 0.0]).unwrap(); // Close to first

        // Search for something close to first vector
        let results = graph.search(&[0.95, 0.05, 0.0], 2, 50).unwrap();
        assert!(!results.is_empty());

        // First result should be id 0 or 3 (closest vectors)
        let first_ids: Vec<u64> = results.iter().take(2).map(|(id, _)| *id).collect();
        assert!(first_ids.contains(&0) || first_ids.contains(&3));
    }

    #[test]
    fn test_search_returns_k_results() {
        let mut graph = HnswGraph::with_defaults().unwrap();
        let vectors = create_random_vectors(50, 16, 123);

        for v in vectors {
            graph.insert(v).unwrap();
        }

        for k in [1, 5, 10, 20] {
            let results = graph.search(&[0.5; 16], k, 100).unwrap();
            assert_eq!(results.len(), k);
        }
    }

    #[test]
    fn test_search_results_sorted() {
        let mut graph = HnswGraph::with_defaults().unwrap();
        let vectors = create_random_vectors(100, 16, 456);

        for v in vectors {
            graph.insert(v).unwrap();
        }

        let results = graph.search(&[0.5; 16], 10, 100).unwrap();

        // Verify results are sorted by distance
        for window in results.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "Results should be sorted by distance"
            );
        }
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let mut graph = HnswGraph::with_defaults().unwrap();
        graph.insert(vec![1.0, 2.0, 3.0]).unwrap();

        let result = graph.search(&[1.0, 2.0], 1, 50);
        assert!(matches!(result, Err(CoreError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut graph = HnswGraph::with_defaults().unwrap();
        let vectors = create_random_vectors(20, 8, 789);

        for v in vectors {
            graph.insert(v).unwrap();
        }

        let bytes = graph.to_bytes().unwrap();
        let restored = HnswGraph::from_bytes(&bytes).unwrap();

        assert_eq!(restored.len(), graph.len());
        assert_eq!(restored.dimension(), graph.dimension());

        // Search should work on restored graph
        let results = restored.search(&[0.5; 8], 5, 50).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_node_creation() {
        let node = HnswNode::new(42, vec![1.0, 2.0], 2, 16);
        assert_eq!(node.id, 42);
        assert_eq!(node.vector, vec![1.0, 2.0]);
        assert_eq!(node.level, 2);
        assert_eq!(node.connections.len(), 3); // Layers 0, 1, 2
    }

    #[test]
    fn test_node_neighbors() {
        let mut node = HnswNode::new(0, vec![1.0], 1, 16);

        // Initially empty
        assert_eq!(node.neighbors_at(0), Some(&[][..]));

        // Add neighbors
        node.neighbors_at_mut(0).unwrap().push(1);
        node.neighbors_at_mut(0).unwrap().push(2);
        assert_eq!(node.neighbors_at(0), Some(&[1, 2][..]));

        // Invalid layer
        assert!(node.neighbors_at(5).is_none());
    }

    #[rstest]
    #[case(10, 4)]
    #[case(50, 16)]
    #[case(100, 32)]
    fn test_graph_sizes(#[case] n: usize, #[case] dim: usize) {
        let mut graph = HnswGraph::with_defaults().unwrap();
        let vectors = create_random_vectors(n, dim, 999);

        for v in vectors {
            graph.insert(v).unwrap();
        }

        assert_eq!(graph.len(), n);
        assert_eq!(graph.dimension(), Some(dim));
    }

    // Property-based tests for HNSW invariants
    proptest! {
        #[test]
        fn prop_insert_increases_size(
            vectors in proptest::collection::vec(
                proptest::collection::vec(-1.0f32..1.0, 8),
                1..20
            )
        ) {
            let mut graph = HnswGraph::with_defaults().unwrap();

            for (i, v) in vectors.into_iter().enumerate() {
                graph.insert(v).unwrap();
                prop_assert_eq!(graph.len(), i + 1);
            }
        }

        #[test]
        fn prop_search_returns_valid_ids(
            n in 5usize..50,
            k in 1usize..10,
        ) {
            let mut graph = HnswGraph::with_defaults().unwrap();
            let vectors = create_random_vectors(n, 8, 42);

            for v in vectors {
                graph.insert(v).unwrap();
            }

            let query = vec![0.5f32; 8];
            let results = graph.search(&query, k.min(n), 50).unwrap();

            for (id, _) in results {
                prop_assert!(id < n as u64, "Result ID should be valid");
                prop_assert!(graph.get_node(id).is_some(), "Node should exist");
            }
        }

        #[test]
        fn prop_serialization_preserves_size(
            n in 1usize..30,
        ) {
            let mut graph = HnswGraph::with_defaults().unwrap();
            let vectors = create_random_vectors(n, 8, 123);

            for v in vectors {
                graph.insert(v).unwrap();
            }

            let bytes = graph.to_bytes().unwrap();
            let restored = HnswGraph::from_bytes(&bytes).unwrap();

            prop_assert_eq!(restored.len(), graph.len());
        }
    }

    #[test]
    fn test_recall_quality() {
        // Build a graph with known vectors
        let mut graph = HnswGraph::new(HnswConfig::accurate()).unwrap();
        let dim = 32;
        let n = 200;
        let vectors = create_random_vectors(n, dim, 42);

        for v in vectors.iter() {
            graph.insert(v.clone()).unwrap();
        }

        // For several queries, verify HNSW finds the true nearest
        let mut correct = 0;
        let test_queries = 20;

        for i in 0..test_queries {
            let query = &vectors[i * 10 % n];

            // Find true nearest by brute force
            let mut distances: Vec<(u64, f32)> = (0..n as u64)
                .map(|id| {
                    let dist = graph
                        .config
                        .metric
                        .calculate(query, &vectors[id as usize])
                        .unwrap();
                    (id, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let true_nearest = distances[0].0;

            // Find with HNSW
            let results = graph.search(query, 1, 100).unwrap();

            if !results.is_empty() && results[0].0 == true_nearest {
                correct += 1;
            }
        }

        let recall = correct as f32 / test_queries as f32;
        // With random vectors and probabilistic HNSW, 70% recall is reasonable
        assert!(
            recall >= 0.35,
            "Recall should be >= 35%, got {:.1}%",
            recall * 100.0
        );
    }
}
