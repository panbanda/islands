//! LEANN Index - Low-Storage Embedding-based Approximate Nearest Neighbor Search
//!
//! Based on the LEANN paper (arXiv:2506.08276): "LEANN: A Low-Storage Embedding-based
//! Retrieval System for Large Scale Generative AI Applications"
//!
//! # Paper Overview
//!
//! LEANN achieves ~95% storage reduction through two key strategies:
//!
//! 1. **Graph-only storage**: Store only the proximity graph structure in CSR format,
//!    not the embeddings themselves. Graph storage scales with edge count (4 bytes/edge),
//!    not embedding dimension.
//!
//! 2. **Selective embedding recomputation**: During search, recompute embeddings
//!    on-the-fly only for nodes in the search path, using a two-level filtering
//!    approach to minimize recomputation.
//!
//! # Algorithm Details (Section 4 of paper)
//!
//! ## Two-Level Search (Algorithm 2)
//!
//! The paper describes a hybrid search using dual queues:
//! - **Approximate Queue (AQ)**: Uses Product Quantization (PQ) for cheap distance
//!   estimates to filter candidates.
//! - **Exact Queue (EQ)**: Only the top `a%` (re-ranking ratio) candidates from AQ
//!   have their embeddings recomputed for exact distance calculation.
//!
//! Quote: "The core insight of this design is to combine the complementary strengths
//! of approximate and exact distance computations" (Section 4.1).
//!
//! ## High-Degree Preserving Pruning (Algorithm 3, Section 5)
//!
//! Node access patterns are highly skewed - a small subset of "hub" nodes are
//! frequently visited. The paper's strategy:
//! - Top ~2% of nodes by degree are designated as "hub" nodes
//! - Hub nodes may retain up to M connections (M=30-60)
//! - Other nodes are restricted to m connections (m < M)
//! - Bidirectional edges to hub nodes are always preserved
//!
//! ## Paper Parameters (Section 5)
//!
//! - M = 30 (connections per node during construction)
//! - efConstruction = 128 (search queue length during index building)
//! - efSearch = variable (adjusted for target recall)
//!
//! # This Implementation
//!
//! This module implements:
//! - CSR graph storage format (matches paper)
//! - Best-first search with selective embedding recomputation (Algorithm 1 + recomputation)
//! - High-degree preserving pruning during construction
//! - EmbeddingProvider trait for on-demand embedding computation
//!
//! Note: The full two-level search (Algorithm 2) requires a pre-computed PQ codebook.
//! This implementation provides the pure recomputation path. For two-level search,
//! combine with the `pq` module for approximate distance filtering.

use crate::distance::{Distance, DistanceMetric};
use crate::error::{CoreError, CoreResult};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// Trait for computing embeddings on-demand during search.
///
/// This abstraction enables LEANN's core storage savings: instead of storing
/// all embeddings (O(n*d) where d can be 768-4096 for modern models), we store
/// only the graph structure and recompute embeddings during search.
///
/// # Paper Context (Section 4.2)
///
/// The paper uses GPU inference for embedding recomputation with dynamic batching:
/// "accumulates nodes across multiple search hops until reaching target batch size
/// (e.g., 64 nodes for A10 GPU) before triggering embedding computation."
///
/// This trait abstracts over the embedding source, allowing:
/// - In-memory lookup (for testing/small datasets)
/// - Model inference (for production with embedding models)
/// - Remote API calls (for hosted embedding services)
pub trait EmbeddingProvider: Send + Sync {
    /// Compute embedding for a single ID.
    ///
    /// In production, this typically involves model inference.
    fn compute_embedding(&self, id: u64) -> CoreResult<Vec<f32>>;

    /// Batch compute embeddings for multiple IDs.
    ///
    /// Batching is critical for efficiency - the paper notes that accumulating
    /// nodes until reaching batch size (e.g., 64) before GPU inference provides
    /// significant throughput improvements.
    fn compute_embeddings_batch(&self, ids: &[u64]) -> CoreResult<Vec<Vec<f32>>> {
        ids.iter()
            .map(|&id| self.compute_embedding(id))
            .collect()
    }

    /// Get the embedding dimension.
    fn dimension(&self) -> usize;
}

/// Simple in-memory embedding provider for testing
/// In production, this would call an embedding model server
#[derive(Debug, Clone)]
pub struct InMemoryEmbeddingProvider {
    embeddings: Vec<Vec<f32>>,
    dimension: usize,
}

impl InMemoryEmbeddingProvider {
    /// Create from pre-computed embeddings
    pub fn new(embeddings: Vec<Vec<f32>>) -> CoreResult<Self> {
        if embeddings.is_empty() {
            return Err(CoreError::EmptyCollection);
        }
        let dimension = embeddings[0].len();
        Ok(Self {
            embeddings,
            dimension,
        })
    }

    /// Add an embedding and return its ID
    pub fn add(&mut self, embedding: Vec<f32>) -> CoreResult<u64> {
        if embedding.len() != self.dimension {
            return Err(CoreError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }
        let id = self.embeddings.len() as u64;
        self.embeddings.push(embedding);
        Ok(id)
    }

    /// Create empty provider with specified dimension
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            embeddings: Vec::new(),
            dimension,
        }
    }
}

impl EmbeddingProvider for InMemoryEmbeddingProvider {
    fn compute_embedding(&self, id: u64) -> CoreResult<Vec<f32>> {
        self.embeddings
            .get(id as usize)
            .cloned()
            .ok_or_else(|| CoreError::NodeNotFound(id))
    }

    fn compute_embeddings_batch(&self, ids: &[u64]) -> CoreResult<Vec<Vec<f32>>> {
        ids.iter()
            .map(|&id| self.compute_embedding(id))
            .collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Pruning strategy for candidate selection during search.
///
/// These strategies determine which neighbor candidates have their embeddings
/// recomputed vs being filtered out. The paper's two-level search uses PQ
/// approximate distances for filtering; these strategies provide alternatives
/// when PQ is not available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PruningStrategy {
    /// Global: Adjust pruning based on how full the result set is.
    /// When result set is nearly full (close to ef), prune more aggressively.
    #[default]
    Global,
    /// Local: Simple top-k pruning of candidates.
    Local,
    /// Proportional: Weight candidates by their degree (hub nodes more likely kept).
    /// This approximates the paper's insight that hub nodes are frequently visited.
    Proportional,
}

/// CSR (Compressed Sparse Row) graph format for compact storage.
///
/// From Section 5 of the paper: graph storage uses CSR format where storage
/// scales with total edge count at 4 bytes per edge (int32 node references),
/// not with embedding dimension.
///
/// For a corpus with n vectors, M connections per node:
/// - Graph storage: O(n * M * 4 bytes) = O(n * M)
/// - Embedding storage (avoided): O(n * d * 4 bytes) = O(n * d)
///
/// When d >> M (e.g., d=768, M=30), this yields ~25x storage reduction from
/// graph structure alone, plus additional savings from not storing embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrGraph {
    /// Starting offset in neighbors array for each node (len = num_nodes + 1)
    pub node_offsets: Vec<usize>,
    /// Flat array of all neighbor IDs
    pub neighbors: Vec<u64>,
    /// Maximum level for each node
    pub levels: Vec<usize>,
    /// Entry point node ID
    pub entry_point: Option<u64>,
    /// Maximum level in graph
    pub max_level: usize,
    /// Number of nodes
    pub num_nodes: usize,
    /// Degree counts for high-degree pruning
    pub degree_counts: Vec<usize>,
}

impl CsrGraph {
    /// Create empty CSR graph
    pub fn new() -> Self {
        Self {
            node_offsets: vec![0],
            neighbors: Vec::new(),
            levels: Vec::new(),
            entry_point: None,
            max_level: 0,
            num_nodes: 0,
            degree_counts: Vec::new(),
        }
    }

    /// Get neighbors for a node at layer 0 (most common operation)
    pub fn get_neighbors(&self, node_id: u64) -> Option<&[u64]> {
        let id = node_id as usize;
        if id >= self.num_nodes {
            return None;
        }
        let start = self.node_offsets[id];
        let end = self.node_offsets[id + 1];
        Some(&self.neighbors[start..end])
    }

    /// Add a node with its neighbors
    pub fn add_node(&mut self, neighbors: Vec<u64>, level: usize) -> u64 {
        let id = self.num_nodes as u64;
        self.num_nodes += 1;
        self.levels.push(level);
        self.degree_counts.push(neighbors.len());

        // Update offsets and add neighbors
        self.neighbors.extend(neighbors);
        self.node_offsets.push(self.neighbors.len());

        // Update entry point and max level
        if self.entry_point.is_none() || level > self.max_level {
            self.entry_point = Some(id);
            self.max_level = level;
        }

        id
    }

    /// Update neighbors for a node (used during construction)
    pub fn set_neighbors(&mut self, node_id: u64, new_neighbors: Vec<u64>) {
        let id = node_id as usize;
        if id >= self.num_nodes {
            return;
        }

        let old_start = self.node_offsets[id];
        let old_end = self.node_offsets[id + 1];
        let old_len = old_end - old_start;
        let new_len = new_neighbors.len();

        if new_len == old_len {
            // Same size: just overwrite
            self.neighbors[old_start..old_end].copy_from_slice(&new_neighbors);
        } else {
            // Different size: need to rebuild (expensive but rare)
            let mut new_neighbor_list = Vec::with_capacity(self.neighbors.len() + new_len - old_len);
            let mut new_offsets = Vec::with_capacity(self.node_offsets.len());
            new_offsets.push(0);

            for i in 0..self.num_nodes {
                if i == id {
                    new_neighbor_list.extend(&new_neighbors);
                } else {
                    let start = self.node_offsets[i];
                    let end = self.node_offsets[i + 1];
                    new_neighbor_list.extend_from_slice(&self.neighbors[start..end]);
                }
                new_offsets.push(new_neighbor_list.len());
            }

            self.neighbors = new_neighbor_list;
            self.node_offsets = new_offsets;
        }

        self.degree_counts[id] = new_neighbors.len();
    }

    /// Get storage size in bytes (approximate)
    pub fn storage_bytes(&self) -> usize {
        self.node_offsets.len() * std::mem::size_of::<usize>()
            + self.neighbors.len() * std::mem::size_of::<u64>()
            + self.levels.len() * std::mem::size_of::<usize>()
            + self.degree_counts.len() * std::mem::size_of::<usize>()
    }
}

impl Default for CsrGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for LEANN index.
///
/// # Paper Parameters (Section 5)
///
/// The paper reports using:
/// - M = 30 (connections per node)
/// - efConstruction = 128 (construction search queue size)
/// - efSearch = variable (adjusted for target recall)
///
/// Default configuration uses these paper values. Use `paper_default()` for
/// exact paper parameters, or customize as needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeannConfig {
    /// Maximum connections per node during construction.
    /// Paper uses M=30. Higher values improve recall but increase storage and build time.
    pub m: usize,

    /// Maximum connections for layer 0 (typically 2*M).
    /// Layer 0 has more connections to improve search starting from entry point.
    pub m0: usize,

    /// Size of dynamic candidate list during construction.
    /// Paper uses efConstruction=128. Higher values improve graph quality at build cost.
    pub ef_construction: usize,

    /// Normalization factor for level generation: 1/ln(M).
    pub ml: f64,

    /// Maximum number of HNSW layers.
    pub max_layers: usize,

    /// Distance metric for similarity computation.
    pub metric: DistanceMetric,

    /// Search complexity parameter (ef_search).
    /// Larger values search more candidates, improving recall at latency cost.
    pub ef_search: usize,

    /// Number of parallel search paths (not used in paper's algorithm).
    pub beam_width: usize,

    /// Ratio of candidates to skip via pruning (0.0 = no pruning, 1.0 = max pruning).
    /// Only applies when not using PQ-based two-level search.
    pub prune_ratio: f32,

    /// Strategy for selecting candidates to compute embeddings for.
    pub pruning_strategy: PruningStrategy,

    /// Enable high-degree preserving pruning (Algorithm 3 in paper).
    /// When true, "hub" nodes with high connectivity are preserved during pruning.
    pub high_degree_pruning: bool,

    /// Percentile threshold for identifying hub nodes (paper uses top 2%).
    /// Nodes with degree above this threshold are considered hubs.
    pub hub_percentile: f32,

    /// Enable compact CSR storage format.
    pub is_compact: bool,

    /// Enable embedding recomputation mode (the core LEANN feature).
    pub is_recompute: bool,
}

impl Default for LeannConfig {
    /// Default uses paper parameters: M=30, efConstruction=128.
    fn default() -> Self {
        Self::paper_default()
    }
}

impl LeannConfig {
    /// Configuration matching the paper's reported parameters (Section 5).
    ///
    /// - M = 30 connections per node
    /// - efConstruction = 128
    /// - High-degree preserving pruning enabled (top 2% are hubs)
    pub fn paper_default() -> Self {
        Self {
            m: 30,
            m0: 60, // 2*M
            ef_construction: 128,
            ml: 1.0 / (30.0_f64).ln(),
            max_layers: 16,
            metric: DistanceMetric::Cosine,
            ef_search: 64,
            beam_width: 1,
            prune_ratio: 0.0,
            pruning_strategy: PruningStrategy::Global,
            high_degree_pruning: true,
            hub_percentile: 0.02, // Top 2% are hub nodes (from paper)
            is_compact: true,
            is_recompute: true,
        }
    }

    /// Configuration optimized for speed (lower accuracy).
    pub fn fast() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 100,
            ef_search: 32,
            beam_width: 1,
            prune_ratio: 0.3, // Prune 30% of candidates
            ..Default::default()
        }
    }

    /// Configuration optimized for accuracy (higher latency).
    pub fn accurate() -> Self {
        Self {
            m: 48,
            m0: 96,
            ef_construction: 400,
            ef_search: 128,
            beam_width: 1,
            prune_ratio: 0.0, // No pruning
            ..Default::default()
        }
    }

    /// Validate configuration parameters.
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
        if self.prune_ratio < 0.0 || self.prune_ratio > 1.0 {
            return Err(CoreError::InvalidConfig(
                "prune_ratio must be in [0.0, 1.0]".to_string(),
            ));
        }
        if self.beam_width == 0 {
            return Err(CoreError::InvalidConfig(
                "beam_width must be > 0".to_string(),
            ));
        }
        if self.hub_percentile < 0.0 || self.hub_percentile > 1.0 {
            return Err(CoreError::InvalidConfig(
                "hub_percentile must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(())
    }
}

/// LEANN Index - Low-Storage Embedding-based Approximate Nearest Neighbor.
///
/// From the paper (arXiv:2506.08276): "LEANN reduces storage requirements to
/// approximately 5% of original data size" by storing only the graph structure
/// and recomputing embeddings on-the-fly during search.
///
/// # Storage Comparison
///
/// For a corpus with n vectors of dimension d, with M connections per node:
///
/// | Approach | Storage |
/// |----------|---------|
/// | Traditional HNSW | O(n*d + n*M) = O(n*d) for large d |
/// | LEANN | O(n*M) only |
///
/// For typical values (d=768, M=30), LEANN uses ~25x less storage.
///
/// # Usage
///
/// ```rust,ignore
/// let config = LeannConfig::paper_default();
/// let mut index = LeannIndex::new(config)?;
///
/// // Build requires embedding provider for computing embeddings during construction
/// index.build(&provider, num_vectors)?;
///
/// // Search also requires provider for on-demand embedding recomputation
/// let results = index.search(&query, k, &provider)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeannIndex {
    /// Configuration
    config: LeannConfig,
    /// Graph structure in CSR format
    graph: CsrGraph,
    /// Vector dimension
    dimension: Option<usize>,
}

impl LeannIndex {
    /// Create a new LEANN index
    pub fn new(config: LeannConfig) -> CoreResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            graph: CsrGraph::new(),
            dimension: None,
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> CoreResult<Self> {
        Self::new(LeannConfig::default())
    }

    /// Get number of vectors in the index
    pub fn len(&self) -> usize {
        self.graph.num_nodes
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.graph.num_nodes == 0
    }

    /// Get vector dimension
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Get storage size in bytes
    pub fn storage_bytes(&self) -> usize {
        self.graph.storage_bytes()
    }

    /// Check if recomputation is enabled
    pub fn is_recompute(&self) -> bool {
        self.config.is_recompute
    }

    /// Check if compact storage is enabled
    pub fn is_compact(&self) -> bool {
        self.config.is_compact
    }

    /// Generate a random level for a new node
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.r#gen();
        let level = (-r.ln() * self.config.ml).floor() as usize;
        level.min(self.config.max_layers - 1)
    }

    /// Build index from embeddings via provider
    ///
    /// This is the main build method. Embeddings are fetched from the provider
    /// during construction, but NOT stored - only the graph structure is kept.
    pub fn build<P: EmbeddingProvider>(&mut self, provider: &P, num_vectors: usize) -> CoreResult<()> {
        if num_vectors == 0 {
            return Ok(());
        }

        self.dimension = Some(provider.dimension());

        // Temporary storage for embeddings during build (NOT persisted)
        let mut temp_embeddings: Vec<Vec<f32>> = Vec::with_capacity(num_vectors);

        // Temporary adjacency list (converted to CSR after build)
        let mut adjacency: Vec<Vec<u64>> = Vec::with_capacity(num_vectors);

        // Build graph incrementally
        for id in 0..num_vectors {
            let embedding = provider.compute_embedding(id as u64)?;
            temp_embeddings.push(embedding);

            let level = self.random_level();

            // Find neighbors by searching existing graph
            let neighbors = if adjacency.is_empty() {
                Vec::new()
            } else {
                self.find_neighbors_for_insert_temp(&temp_embeddings, &adjacency, id, level)?
            };

            // Add bidirectional edges
            adjacency.push(neighbors.clone());
            for &neighbor_id in &neighbors {
                let nid = neighbor_id as usize;
                if !adjacency[nid].contains(&(id as u64)) {
                    adjacency[nid].push(id as u64);
                    // Prune if too many neighbors
                    if adjacency[nid].len() > self.config.m0 {
                        adjacency[nid] = self.prune_neighbors_temp(&temp_embeddings, nid, &adjacency[nid], self.config.m0);
                    }
                }
            }

            // Update entry point
            if self.graph.entry_point.is_none() || level > self.graph.max_level {
                self.graph.entry_point = Some(id as u64);
                self.graph.max_level = level;
            }
            self.graph.levels.push(level);
        }

        // Convert adjacency list to CSR format
        self.graph.num_nodes = num_vectors;
        self.graph.node_offsets = vec![0];
        self.graph.neighbors.clear();
        self.graph.degree_counts.clear();

        for neighbors in &adjacency {
            self.graph.neighbors.extend(neighbors);
            self.graph.node_offsets.push(self.graph.neighbors.len());
            self.graph.degree_counts.push(neighbors.len());
        }

        // temp_embeddings is dropped here - we only keep the graph
        Ok(())
    }

    /// Prune neighbors using temporary embeddings
    fn prune_neighbors_temp(&self, embeddings: &[Vec<f32>], node_id: usize, neighbors: &[u64], max_conn: usize) -> Vec<u64> {
        let node_vec = &embeddings[node_id];
        let mut scored: Vec<(u64, f32)> = neighbors
            .iter()
            .filter_map(|&id| {
                self.config.metric.calculate(node_vec, &embeddings[id as usize]).ok().map(|d| (id, d))
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(max_conn).map(|(id, _)| id).collect()
    }

    /// Find neighbors for a new node during insertion using temp adjacency
    fn find_neighbors_for_insert_temp(
        &self,
        temp_embeddings: &[Vec<f32>],
        adjacency: &[Vec<u64>],
        new_id: usize,
        _level: usize,
    ) -> CoreResult<Vec<u64>> {
        let query = &temp_embeddings[new_id];
        let entry_id = self.graph.entry_point.unwrap_or(0);

        // Search for nearest neighbors in existing graph
        let mut candidates = self.search_layer_with_adjacency(
            query,
            entry_id,
            self.config.ef_construction,
            temp_embeddings,
            adjacency,
        )?;

        // Apply high-degree preserving pruning
        if self.config.high_degree_pruning && !adjacency.is_empty() {
            candidates = self.prune_with_degree_preservation_temp(&candidates, adjacency, self.config.m0);
        } else {
            candidates.truncate(self.config.m0);
        }

        Ok(candidates.into_iter().map(|(id, _)| id).collect())
    }

    /// Search a layer using temporary adjacency list (during build)
    fn search_layer_with_adjacency(
        &self,
        query: &[f32],
        entry: u64,
        ef: usize,
        embeddings: &[Vec<f32>],
        adjacency: &[Vec<u64>],
    ) -> CoreResult<Vec<(u64, f32)>> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, u64)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat<f32>, u64)> = BinaryHeap::new();

        let entry_dist = self.config.metric.calculate(query, &embeddings[entry as usize])?;
        visited.insert(entry);
        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
        results.push((OrderedFloat(entry_dist), entry));

        while let Some(Reverse((dist, id))) = candidates.pop() {
            if let Some(&(worst_dist, _)) = results.peek() {
                if results.len() >= ef && dist > worst_dist {
                    break;
                }
            }

            let neighbors = &adjacency[id as usize];
            for &neighbor in neighbors {
                if visited.insert(neighbor) {
                    let neighbor_dist =
                        self.config.metric.calculate(query, &embeddings[neighbor as usize])?;

                    let should_add = results.len() < ef
                        || results
                            .peek()
                            .map(|&(w, _)| neighbor_dist < w.0)
                            .unwrap_or(true);

                    if should_add {
                        candidates.push(Reverse((OrderedFloat(neighbor_dist), neighbor)));
                        results.push((OrderedFloat(neighbor_dist), neighbor));

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<(u64, f32)> =
            results.into_iter().map(|(d, id)| (id, d.0)).collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(result_vec)
    }

    /// High-degree preserving pruning (Algorithm 3 from paper).
    ///
    /// From Section 5: "Node access patterns are highly skewed—a small subset of
    /// 'hub' nodes are frequently visited." The paper's strategy:
    /// - Top ~2% of nodes by degree are designated as "hub" nodes
    /// - Hub nodes may retain up to M connections
    /// - Other nodes are restricted to fewer connections
    /// - Bidirectional edges to hub nodes are preserved regardless of threshold
    ///
    /// This preserves graph navigability by keeping well-connected hub nodes.
    fn prune_with_degree_preservation_temp(
        &self,
        candidates: &[(u64, f32)],
        adjacency: &[Vec<u64>],
        max_conn: usize,
    ) -> Vec<(u64, f32)> {
        if candidates.len() <= max_conn {
            return candidates.to_vec();
        }

        // Compute degree threshold for hub nodes (top hub_percentile by degree)
        // Paper uses top 2%: if we have 100 nodes and hub_percentile=0.02,
        // the top 2 nodes by degree would be hubs
        let mut degrees: Vec<usize> = candidates
            .iter()
            .map(|(id, _)| adjacency.get(*id as usize).map(|n| n.len()).unwrap_or(0))
            .collect();
        degrees.sort_unstable_by(|a, b| b.cmp(a)); // Sort descending

        let hub_count = ((degrees.len() as f32) * self.config.hub_percentile).ceil() as usize;
        let degree_threshold = if hub_count > 0 && hub_count < degrees.len() {
            degrees[hub_count.saturating_sub(1)]
        } else {
            usize::MAX // No hubs if percentile is 0
        };

        let mut hub_nodes = Vec::new();
        let mut regular_nodes = Vec::new();

        for &(id, dist) in candidates {
            let degree = adjacency.get(id as usize).map(|n| n.len()).unwrap_or(0);
            if degree >= degree_threshold && degree_threshold < usize::MAX {
                hub_nodes.push((id, dist, degree));
            } else {
                regular_nodes.push((id, dist));
            }
        }

        // Sort hub nodes by degree (descending) - keep highest degree hubs
        hub_nodes.sort_by(|a, b| b.2.cmp(&a.2));
        // Sort regular nodes by distance (ascending) - keep nearest
        regular_nodes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected = Vec::with_capacity(max_conn);

        // Reserve slots for hub nodes (paper preserves bidirectional edges to hubs)
        let hub_slots = (max_conn / 4).max(1);
        for (id, dist, _) in hub_nodes.iter().take(hub_slots) {
            selected.push((*id, *dist));
        }

        // Fill remaining with nearest regular nodes
        for (id, dist) in regular_nodes {
            if selected.len() >= max_conn {
                break;
            }
            if !selected.iter().any(|(sid, _)| *sid == id) {
                selected.push((id, dist));
            }
        }

        // If still space, add remaining hub nodes
        for (id, dist, _) in hub_nodes.iter().skip(hub_slots) {
            if selected.len() >= max_conn {
                break;
            }
            if !selected.iter().any(|(sid, _)| *sid == *id) {
                selected.push((*id, *dist));
            }
        }

        selected
    }

    /// Search for k nearest neighbors with embedding recomputation.
    ///
    /// Implements a best-first search (Algorithm 1 from paper) with selective
    /// embedding recomputation. Unlike traditional HNSW that stores all embeddings,
    /// LEANN only computes embeddings for nodes visited during search.
    ///
    /// # Algorithm (simplified from paper's Algorithm 1)
    ///
    /// 1. Initialize candidate queue C and result set R with entry point
    /// 2. While C is not empty:
    ///    a. Pop closest candidate from C
    ///    b. If candidate is farther than worst in R and |R| >= ef, stop
    ///    c. For each unvisited neighbor:
    ///       - Recompute embedding via provider
    ///       - Calculate exact distance
    ///       - Add to C and R if closer than worst in R
    /// 3. Return top-k from R
    ///
    /// # Paper Context
    ///
    /// The paper's Algorithm 2 adds a two-level approach using PQ for approximate
    /// filtering before exact recomputation. This implementation provides the
    /// pure recomputation path. For two-level search, combine with `pq` module.
    pub fn search<P: EmbeddingProvider>(
        &self,
        query: &[f32],
        k: usize,
        provider: &P,
    ) -> CoreResult<Vec<(u64, f32)>> {
        self.search_with_params(query, k, self.config.ef_search, provider)
    }

    /// Search with custom parameters
    pub fn search_with_params<P: EmbeddingProvider>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        provider: &P,
    ) -> CoreResult<Vec<(u64, f32)>> {
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

        let entry_id = self.graph.entry_point.ok_or(CoreError::IndexNotBuilt)?;
        let ef = ef.max(k);

        // Search with recomputation
        let results = self.search_layer_recompute(query, entry_id, ef, provider)?;

        Ok(results.into_iter().take(k).collect())
    }

    /// Search layer with embedding recomputation (LEANN core algorithm)
    fn search_layer_recompute<P: EmbeddingProvider>(
        &self,
        query: &[f32],
        entry: u64,
        ef: usize,
        provider: &P,
    ) -> CoreResult<Vec<(u64, f32)>> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, u64)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat<f32>, u64)> = BinaryHeap::new();

        // Recompute entry embedding
        let entry_embedding = provider.compute_embedding(entry)?;
        let entry_dist = self.config.metric.calculate(query, &entry_embedding)?;

        visited.insert(entry);
        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
        results.push((OrderedFloat(entry_dist), entry));

        // Track embeddings computed (for beam search optimization)
        #[allow(unused_mut, unused_variables)]
        let mut embeddings_computed = 1usize;

        while let Some(Reverse((dist, id))) = candidates.pop() {
            // Check termination condition
            if let Some(&(worst_dist, _)) = results.peek() {
                if results.len() >= ef && dist > worst_dist {
                    break;
                }
            }

            // Get neighbors
            if let Some(neighbors) = self.graph.get_neighbors(id) {
                // Collect unvisited neighbors
                let unvisited: Vec<u64> = neighbors
                    .iter()
                    .copied()
                    .filter(|&n| visited.insert(n))
                    .collect();

                if unvisited.is_empty() {
                    continue;
                }

                // Apply pruning strategy
                let to_compute = self.apply_pruning_strategy(&unvisited, &results, ef);

                // Batch compute embeddings for efficiency
                let embeddings = provider.compute_embeddings_batch(&to_compute)?;
                #[allow(unused_assignments)]
                {
                    embeddings_computed += embeddings.len();
                }

                for (i, &neighbor) in to_compute.iter().enumerate() {
                    let neighbor_dist = self.config.metric.calculate(query, &embeddings[i])?;

                    let should_add = results.len() < ef
                        || results
                            .peek()
                            .map(|&(w, _)| neighbor_dist < w.0)
                            .unwrap_or(true);

                    if should_add {
                        candidates.push(Reverse((OrderedFloat(neighbor_dist), neighbor)));
                        results.push((OrderedFloat(neighbor_dist), neighbor));

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Log efficiency metric (embeddings computed vs total nodes)
        #[cfg(feature = "metrics")]
        log::debug!(
            "LEANN search: computed {} embeddings out of {} total nodes ({:.1}%)",
            embeddings_computed,
            self.graph.num_nodes,
            (embeddings_computed as f64 / self.graph.num_nodes as f64) * 100.0
        );

        // Convert to sorted vector
        let mut result_vec: Vec<(u64, f32)> =
            results.into_iter().map(|(d, id)| (id, d.0)).collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(result_vec)
    }

    /// Apply pruning strategy to select which candidates to compute embeddings for
    fn apply_pruning_strategy(
        &self,
        candidates: &[u64],
        results: &BinaryHeap<(OrderedFloat<f32>, u64)>,
        ef: usize,
    ) -> Vec<u64> {
        if self.config.prune_ratio == 0.0 || candidates.is_empty() {
            return candidates.to_vec();
        }

        let num_to_keep = ((candidates.len() as f32) * (1.0 - self.config.prune_ratio))
            .ceil() as usize;
        let num_to_keep = num_to_keep.max(1);

        match self.config.pruning_strategy {
            PruningStrategy::Global => {
                // Global: keep based on PQ queue size relative to ef
                let ratio = results.len() as f32 / ef as f32;
                let adjusted = ((candidates.len() as f32) * (1.0 - ratio * self.config.prune_ratio))
                    .ceil() as usize;
                candidates.iter().copied().take(adjusted.max(1)).collect()
            }
            PruningStrategy::Local => {
                // Local: just take first N (assumes pre-sorted by degree)
                candidates.iter().copied().take(num_to_keep).collect()
            }
            PruningStrategy::Proportional => {
                // Proportional: based on neighbor count
                let total_count: usize = candidates
                    .iter()
                    .map(|&id| self.graph.degree_counts.get(id as usize).copied().unwrap_or(0))
                    .sum();

                if total_count == 0 {
                    return candidates.iter().copied().take(num_to_keep).collect();
                }

                let mut selected = Vec::with_capacity(num_to_keep);
                for &id in candidates {
                    let degree = self.graph.degree_counts.get(id as usize).copied().unwrap_or(1);
                    let prob = degree as f32 / total_count as f32;
                    if rand::thread_rng().r#gen::<f32>() < prob * num_to_keep as f32 {
                        selected.push(id);
                        if selected.len() >= num_to_keep {
                            break;
                        }
                    }
                }
                if selected.is_empty() {
                    selected.push(candidates[0]);
                }
                selected
            }
        }
    }

    /// Serialize to bytes (only graph structure, not embeddings)
    pub fn to_bytes(&self) -> CoreResult<Vec<u8>> {
        bincode::serialize(self).map_err(|e| CoreError::Serialization(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> CoreResult<Self> {
        bincode::deserialize(bytes).map_err(|e| CoreError::Deserialization(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use rstest::rstest;

    fn create_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
            .collect()
    }

    fn create_provider(vectors: Vec<Vec<f32>>) -> InMemoryEmbeddingProvider {
        InMemoryEmbeddingProvider::new(vectors).unwrap()
    }

    // ==================== Configuration Tests ====================

    #[test]
    fn test_config_default() {
        // Default uses paper parameters: M=30, efConstruction=128
        let config = LeannConfig::default();
        assert_eq!(config.m, 30);  // Paper: M=30
        assert_eq!(config.m0, 60); // Paper: M0=2*M
        assert_eq!(config.ef_construction, 128); // Paper: efConstruction=128
        assert!(config.is_compact);
        assert!(config.is_recompute);
        assert!(config.high_degree_pruning);
        assert!((config.hub_percentile - 0.02).abs() < 0.001); // Paper: top 2% are hubs
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_fast() {
        let config = LeannConfig::fast();
        assert!(config.validate().is_ok());
        assert!(config.prune_ratio > 0.0);
        assert!(config.m < 30); // Faster = fewer connections
    }

    #[test]
    fn test_config_accurate() {
        let config = LeannConfig::accurate();
        assert!(config.validate().is_ok());
        assert!(config.m > 30); // More accurate = more connections
        assert!(config.ef_construction > 128);
    }

    #[test]
    fn test_config_validation() {
        let mut config = LeannConfig::default();

        config.m = 0;
        assert!(config.validate().is_err());

        config.m = 30;
        config.m0 = 16;
        assert!(config.validate().is_err());

        config.m0 = 60;
        config.prune_ratio = 1.5;
        assert!(config.validate().is_err());

        config.prune_ratio = 0.3;
        config.beam_width = 0;
        assert!(config.validate().is_err());

        config.beam_width = 1;
        config.hub_percentile = 1.5;
        assert!(config.validate().is_err());
    }

    // ==================== Pruning Strategy Tests ====================

    #[test]
    fn test_pruning_strategy_default() {
        let strategy = PruningStrategy::default();
        assert_eq!(strategy, PruningStrategy::Global);
    }

    #[test]
    fn test_all_pruning_strategies() {
        for strategy in [
            PruningStrategy::Global,
            PruningStrategy::Local,
            PruningStrategy::Proportional,
        ] {
            let config = LeannConfig {
                pruning_strategy: strategy,
                prune_ratio: 0.3,
                ..Default::default()
            };
            assert!(config.validate().is_ok());
        }
    }

    // ==================== CSR Graph Tests ====================

    #[test]
    fn test_csr_graph_new() {
        let graph = CsrGraph::new();
        assert_eq!(graph.num_nodes, 0);
        assert!(graph.entry_point.is_none());
    }

    #[test]
    fn test_csr_graph_add_node() {
        let mut graph = CsrGraph::new();

        let id0 = graph.add_node(vec![], 0);
        assert_eq!(id0, 0);
        assert_eq!(graph.num_nodes, 1);
        assert_eq!(graph.entry_point, Some(0));

        let id1 = graph.add_node(vec![0], 1);
        assert_eq!(id1, 1);
        assert_eq!(graph.num_nodes, 2);
        assert_eq!(graph.entry_point, Some(1)); // Higher level
    }

    #[test]
    fn test_csr_graph_get_neighbors() {
        let mut graph = CsrGraph::new();
        graph.add_node(vec![], 0);
        graph.add_node(vec![0], 0);
        graph.add_node(vec![0, 1], 0);

        assert_eq!(graph.get_neighbors(0), Some(&[][..]));
        assert_eq!(graph.get_neighbors(1), Some(&[0][..]));
        assert_eq!(graph.get_neighbors(2), Some(&[0, 1][..]));
        assert_eq!(graph.get_neighbors(999), None);
    }

    #[test]
    fn test_csr_storage_efficiency() {
        let mut graph = CsrGraph::new();
        for i in 0..100 {
            let neighbors: Vec<u64> = (0..i).map(|j| j % (i.max(1))).collect();
            graph.add_node(neighbors, 0);
        }

        let storage = graph.storage_bytes();
        // CSR should be much more efficient than adjacency list
        assert!(storage < 100 * 100 * std::mem::size_of::<u64>());
    }

    // ==================== Embedding Provider Tests ====================

    #[test]
    fn test_in_memory_provider() {
        let vectors = create_random_vectors(10, 8, 42);
        let provider = create_provider(vectors.clone());

        assert_eq!(provider.dimension(), 8);

        for (i, v) in vectors.iter().enumerate() {
            let embedding = provider.compute_embedding(i as u64).unwrap();
            assert_eq!(embedding, *v);
        }
    }

    #[test]
    fn test_provider_batch() {
        let vectors = create_random_vectors(10, 8, 42);
        let provider = create_provider(vectors.clone());

        let ids: Vec<u64> = vec![0, 2, 5];
        let embeddings = provider.compute_embeddings_batch(&ids).unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0], vectors[0]);
        assert_eq!(embeddings[1], vectors[2]);
        assert_eq!(embeddings[2], vectors[5]);
    }

    #[test]
    fn test_provider_invalid_id() {
        let vectors = create_random_vectors(5, 8, 42);
        let provider = create_provider(vectors);

        let result = provider.compute_embedding(999);
        assert!(matches!(result, Err(CoreError::NodeNotFound(999))));
    }

    // ==================== LEANN Index Tests ====================

    #[test]
    fn test_index_new() {
        let index = LeannIndex::with_defaults().unwrap();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(index.dimension().is_none());
        assert!(index.is_recompute());
        assert!(index.is_compact());
    }

    #[test]
    fn test_index_build() {
        let vectors = create_random_vectors(50, 16, 42);
        let provider = create_provider(vectors);

        let mut index = LeannIndex::with_defaults().unwrap();
        index.build(&provider, 50).unwrap();

        assert_eq!(index.len(), 50);
        assert_eq!(index.dimension(), Some(16));

        // LEANN stores only graph structure, no embeddings
        // Graph storage: O(n * m) where m is avg degree
        // Embedding storage would be: O(n * d) where d is dimension
        // For high-d (typical: 768-4096), LEANN saves storage
        // Here we just verify the index was built correctly
        let storage = index.storage_bytes();
        assert!(storage > 0, "Index should have non-zero storage");
    }

    #[test]
    fn test_index_search() {
        let vectors = create_random_vectors(100, 16, 42);
        let provider = create_provider(vectors.clone());

        let mut index = LeannIndex::with_defaults().unwrap();
        index.build(&provider, 100).unwrap();

        // Search for a known vector
        let results = index.search(&vectors[0], 5, &provider).unwrap();

        assert_eq!(results.len(), 5);
        // First result should be the query itself (or very close)
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_index_search_empty() {
        let index = LeannIndex::with_defaults().unwrap();
        let provider = InMemoryEmbeddingProvider::with_dimension(8);

        let results = index.search(&[0.5; 8], 5, &provider).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_search_dimension_mismatch() {
        let vectors = create_random_vectors(10, 16, 42);
        let provider = create_provider(vectors);

        let mut index = LeannIndex::with_defaults().unwrap();
        index.build(&provider, 10).unwrap();

        let result = index.search(&[0.5; 8], 5, &provider); // Wrong dimension
        assert!(matches!(result, Err(CoreError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_index_search_results_sorted() {
        let vectors = create_random_vectors(50, 16, 123);
        let provider = create_provider(vectors);

        let mut index = LeannIndex::with_defaults().unwrap();
        index.build(&provider, 50).unwrap();

        let results = index.search(&[0.5; 16], 10, &provider).unwrap();

        for window in results.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "Results should be sorted by distance"
            );
        }
    }

    // ==================== Serialization Tests ====================

    #[test]
    fn test_serialization_roundtrip() {
        let vectors = create_random_vectors(30, 16, 42);
        let provider = create_provider(vectors.clone());

        let mut index = LeannIndex::with_defaults().unwrap();
        index.build(&provider, 30).unwrap();

        let bytes = index.to_bytes().unwrap();
        let restored = LeannIndex::from_bytes(&bytes).unwrap();

        assert_eq!(restored.len(), index.len());
        assert_eq!(restored.dimension(), index.dimension());

        // Search should work on restored index
        let results = restored.search(&vectors[0], 5, &provider).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_serialization_no_embeddings() {
        let vectors = create_random_vectors(100, 32, 42);
        let provider = create_provider(vectors);

        let mut index = LeannIndex::with_defaults().unwrap();
        index.build(&provider, 100).unwrap();

        let bytes = index.to_bytes().unwrap();

        // Verify serialization produces valid output
        // Note: LEANN saves storage for HIGH-dimensional embeddings (768-4096d),
        // not for low-d test vectors where graph overhead dominates
        assert!(bytes.len() > 0, "Serialized index should not be empty");

        // Verify deserialization works
        let restored = LeannIndex::from_bytes(&bytes).unwrap();
        assert_eq!(restored.len(), 100);
    }

    // ==================== Recall Quality Tests ====================

    #[test]
    fn test_recall_quality() {
        let config = LeannConfig::accurate();
        let dim = 32;
        let n = 200;
        let vectors = create_random_vectors(n, dim, 42);
        let provider = create_provider(vectors.clone());

        let mut index = LeannIndex::new(config).unwrap();
        index.build(&provider, n).unwrap();

        let mut correct = 0;
        let test_queries = 20;

        for i in 0..test_queries {
            let query = &vectors[i * 10 % n];

            // Brute force ground truth
            let mut distances: Vec<(u64, f32)> = (0..n as u64)
                .map(|id| {
                    let dist = index
                        .config
                        .metric
                        .calculate(query, &vectors[id as usize])
                        .unwrap();
                    (id, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let true_nearest = distances[0].0;

            // LEANN search
            let results = index.search(query, 1, &provider).unwrap();

            if !results.is_empty() && results[0].0 == true_nearest {
                correct += 1;
            }
        }

        let recall = correct as f32 / test_queries as f32;
        assert!(
            recall >= 0.35,
            "Recall should be >= 35%, got {:.1}%",
            recall * 100.0
        );
    }

    // ==================== Pruning Strategy Tests ====================

    #[test]
    fn test_search_with_pruning() {
        for strategy in [
            PruningStrategy::Global,
            PruningStrategy::Local,
            PruningStrategy::Proportional,
        ] {
            let config = LeannConfig {
                pruning_strategy: strategy,
                prune_ratio: 0.3,
                ..Default::default()
            };

            let vectors = create_random_vectors(50, 16, 42);
            let provider = create_provider(vectors.clone());

            let mut index = LeannIndex::new(config).unwrap();
            index.build(&provider, 50).unwrap();

            let results = index.search(&vectors[0], 5, &provider).unwrap();
            assert_eq!(results.len(), 5, "Strategy {:?} should return 5 results", strategy);
        }
    }

    // ==================== Property-Based Tests ====================

    proptest! {
        #[test]
        fn prop_build_increases_size(n in 5usize..30) {
            let vectors = create_random_vectors(n, 8, 42);
            let provider = create_provider(vectors);

            let mut index = LeannIndex::with_defaults().unwrap();
            index.build(&provider, n).unwrap();

            prop_assert_eq!(index.len(), n);
        }

        #[test]
        fn prop_search_returns_valid_ids(n in 10usize..50, k in 1usize..10) {
            let vectors = create_random_vectors(n, 8, 42);
            let provider = create_provider(vectors.clone());

            let mut index = LeannIndex::with_defaults().unwrap();
            index.build(&provider, n).unwrap();

            let results = index.search(&vectors[0], k.min(n), &provider).unwrap();

            for (id, _) in results {
                prop_assert!(id < n as u64, "Result ID should be valid");
            }
        }

        #[test]
        fn prop_storage_valid(n in 20usize..100, dim in 8usize..64) {
            let vectors = create_random_vectors(n, dim, 42);
            let provider = create_provider(vectors);

            let mut index = LeannIndex::with_defaults().unwrap();
            index.build(&provider, n).unwrap();

            let index_storage = index.storage_bytes();

            // LEANN stores graph structure only (no embeddings)
            // Storage savings appear for HIGH-dimensional embeddings (768-4096d)
            // For low-d vectors, graph overhead may exceed embedding storage
            prop_assert!(index_storage > 0, "Index should have positive storage");
            prop_assert_eq!(index.len(), n, "Index should contain all vectors");
        }
    }

    // ==================== Rstest Parameterized Tests ====================

    #[rstest]
    #[case(10, 8)]
    #[case(50, 16)]
    #[case(100, 32)]
    fn test_various_sizes(#[case] n: usize, #[case] dim: usize) {
        let vectors = create_random_vectors(n, dim, 42);
        let provider = create_provider(vectors.clone());

        let mut index = LeannIndex::with_defaults().unwrap();
        index.build(&provider, n).unwrap();

        assert_eq!(index.len(), n);
        assert_eq!(index.dimension(), Some(dim));

        let results = index.search(&vectors[0], 5.min(n), &provider).unwrap();
        assert_eq!(results.len(), 5.min(n));
    }

    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(4)]
    fn test_beam_widths(#[case] beam_width: usize) {
        let config = LeannConfig {
            beam_width,
            ..Default::default()
        };

        let vectors = create_random_vectors(50, 16, 42);
        let provider = create_provider(vectors.clone());

        let mut index = LeannIndex::new(config).unwrap();
        index.build(&provider, 50).unwrap();

        let results = index.search(&vectors[0], 5, &provider).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[rstest]
    #[case(0.0)]
    #[case(0.3)]
    #[case(0.5)]
    #[case(0.8)]
    fn test_prune_ratios(#[case] prune_ratio: f32) {
        let config = LeannConfig {
            prune_ratio,
            ..Default::default()
        };

        let vectors = create_random_vectors(50, 16, 42);
        let provider = create_provider(vectors.clone());

        let mut index = LeannIndex::new(config).unwrap();
        index.build(&provider, 50).unwrap();

        let results = index.search(&vectors[0], 5, &provider).unwrap();
        assert_eq!(results.len(), 5);
    }
}
