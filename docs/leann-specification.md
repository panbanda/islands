# LEANN Technical Specification for Rust Implementation

## Overview

LEANN (Learned Embedding Approximate Nearest Neighbor) is a storage-efficient graph-based approximate nearest neighbor search system. It reduces index storage to under 5% of original data size by selectively pruning the HNSW graph and recomputing embeddings on-demand during search.

**Key Innovation**: Rather than storing all embeddings, LEANN maintains only a pruned graph topology and recomputes embeddings during query traversal. This trades compute latency for dramatic storage reduction (up to 50x smaller than standard indexes).

**Performance Targets**:
- Storage: < 5% of original embedding data
- Recall@3: 90%+ maintained
- Latency: < 2 seconds for edge devices
- Speedup: 1.76x over baseline with all optimizations

---

## Core Data Structures

### 1. Graph Node

```rust
/// A node in the HNSW graph
pub struct GraphNode {
    /// Unique identifier for this node
    pub id: NodeId,

    /// Layer this node exists in (0 = bottom layer)
    pub layer: u8,

    /// Neighbor connections (indices into node array)
    pub neighbors: Vec<NodeId>,

    /// Whether this is a hub node (high-degree, preserved during pruning)
    pub is_hub: bool,

    /// Metadata for the source document (not the embedding)
    pub metadata: NodeMetadata,
}

/// Node identifier - 32-bit should suffice for most use cases
pub type NodeId = u32;

/// Metadata associated with a node (stored, not computed)
pub struct NodeMetadata {
    /// Reference to source document for embedding recomputation
    pub source_ref: SourceReference,

    /// Chunk boundaries in source document
    pub chunk_start: u32,
    pub chunk_end: u32,

    /// Optional cached PQ code for approximate distance
    pub pq_code: Option<PQCode>,
}
```

### 2. HNSW Graph Structure

```rust
/// Hierarchical Navigable Small World graph
pub struct HnswGraph {
    /// All nodes in the graph, indexed by NodeId
    nodes: Vec<GraphNode>,

    /// Entry point for search (highest layer node)
    entry_point: NodeId,

    /// Maximum layer in the graph
    max_layer: u8,

    /// Construction parameters
    config: HnswConfig,

    /// Set of hub node IDs for quick lookup
    hub_nodes: HashSet<NodeId>,
}

/// HNSW configuration parameters
pub struct HnswConfig {
    /// Maximum connections per node in non-bottom layers
    pub m: usize,  // Default: 16

    /// Maximum connections in bottom layer (layer 0)
    pub m0: usize,  // Default: 2 * m = 32

    /// Size of dynamic candidate list during construction
    pub ef_construction: usize,  // Default: 128

    /// Probability decay factor for layer assignment
    /// P(layer = l) = (1/ml)^l * (1 - 1/ml)
    pub ml: f64,  // Default: 1.0 / ln(m)
}
```

### 3. Product Quantization

```rust
/// Product Quantization codebook for approximate distance computation
pub struct ProductQuantizer {
    /// Number of subspaces (subvectors)
    pub num_subspaces: usize,  // Typically 8-32

    /// Dimension of each subspace
    pub subspace_dim: usize,

    /// Number of centroids per subspace
    pub num_centroids: usize,  // Typically 256 for 8-bit codes

    /// Codebook: [num_subspaces][num_centroids][subspace_dim]
    pub codebook: Array3<f32>,
}

/// Compressed PQ code for a single vector
pub struct PQCode {
    /// Centroid indices for each subspace
    pub codes: Vec<u8>,  // Length = num_subspaces
}

/// Precomputed distance table for fast PQ distance lookup
pub struct PQDistanceTable {
    /// [num_subspaces][num_centroids] distances to query
    pub distances: Array2<f32>,
}
```

### 4. Search State

```rust
/// Priority queue entry for search
#[derive(Clone, PartialEq)]
pub struct SearchCandidate {
    pub node_id: NodeId,
    pub distance: OrderedFloat<f32>,
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance = higher priority
        other.distance.cmp(&self.distance)
    }
}

/// Two-level search state
pub struct TwoLevelSearchState {
    /// Approximate queue - stores nodes with PQ distances
    pub approximate_queue: BinaryHeap<SearchCandidate>,

    /// Exact queue - stores nodes with recomputed exact distances
    pub exact_queue: BinaryHeap<SearchCandidate>,

    /// Result set - top-k candidates
    pub results: BinaryHeap<Reverse<SearchCandidate>>,  // Max-heap for furthest

    /// Visited nodes
    pub visited: HashSet<NodeId>,

    /// Batch accumulator for embedding recomputation
    pub recompute_batch: Vec<NodeId>,
}
```

### 5. Dynamic Batching State

```rust
/// Dynamic batching configuration and state
pub struct DynamicBatcher {
    /// Target batch size (e.g., 64 for A10 GPU)
    pub target_batch_size: usize,

    /// Accumulated nodes awaiting recomputation
    pub pending: Vec<NodeId>,

    /// Whether to block until batch is full
    pub blocking: bool,
}
```

---

## Core Algorithms

### Algorithm 1: Standard Best-First Search (Baseline)

This is the baseline HNSW search algorithm for reference.

```
FUNCTION BestFirstSearch(graph G, query q, entry_point p, k, ef):
    INPUT:
        G: HNSW graph with entry node p
        q: query embedding
        k: number of results requested
        ef: search queue size (ef >= k)
    OUTPUT:
        R: top-k approximate nearest neighbors

    1. C <- {p}                    // Candidate set (min-heap by distance)
    2. R <- {p}                    // Result set
    3. V <- {p}                    // Visited set

    4. WHILE C is not empty:
    5.     c <- extract_min(C)     // Closest candidate
    6.     f <- max(R)             // Furthest in results
    7.     IF distance(c, q) > distance(f, q):
    8.         BREAK               // No better candidates possible
    9.
    10.    FOR each neighbor n of c in G:
    11.        IF n not in V:
    12.            V <- V + {n}
    13.            x_n <- retrieve_embedding(n)  // EXPENSIVE: Full embedding lookup
    14.            d <- distance(q, x_n)
    15.            insert(C, (n, d))
    16.            insert(R, (n, d))
    17.            IF |R| > ef:
    18.                remove_max(R)

    19. RETURN top_k(R, k)
```

**Complexity**:
- Time: O(ef * M * d) where M = average degree, d = embedding dimension
- Space: O(n * d) for storing all embeddings

### Algorithm 2: Two-Level Search with Hybrid Distance

The core LEANN search innovation using approximate and exact queues.

```
FUNCTION TwoLevelSearch(graph G, query q, entry_point p, k, ef, rerank_ratio a):
    INPUT:
        G: Pruned HNSW graph
        q: query embedding (precomputed)
        p: entry point node
        k: number of results
        ef: search queue length
        a: re-ranking ratio (e.g., 0.1 = top 10% promoted)
    OUTPUT:
        R: k closest neighbors with exact distances

    1. visited <- {p}
    2. AQ <- empty                  // Approximate Queue (PQ distances)
    3. EQ <- {(p, exact_dist(p,q))} // Exact Queue
    4. R <- {(p, exact_dist(p,q))}  // Results

    5. WHILE EQ is not empty:
    6.     v <- extract_min(EQ)     // Closest by exact distance
    7.     f <- max(R)              // Furthest in results
    8.     IF exact_dist(v, q) > exact_dist(f, q):
    9.         BREAK
    10.
    11.    // Phase 1: Expand with approximate distances
    12.    FOR each neighbor n of v:
    13.        IF n not in visited:
    14.            visited <- visited + {n}
    15.            d_approx <- pq_distance(n, q)  // CHEAP: PQ lookup
    16.            insert(AQ, (n, d_approx))
    17.
    18.    // Phase 2: Promote top candidates for exact recomputation
    19.    M <- extract_top_ratio(AQ, a)  // Top a% by approximate distance
    20.
    21.    FOR each m in M:
    22.        IF m not in EQ:
    23.            d_exact <- recompute_embedding_distance(m, q)  // On-demand
    24.            insert(EQ, (m, d_exact))
    25.            insert(R, (m, d_exact))
    26.            IF |R| > ef:
    27.                remove_max(R)

    28. RETURN top_k(R, k)
```

**Key Insight**: Most nodes are evaluated with cheap PQ distances. Only promising candidates (top a%) undergo expensive embedding recomputation.

**Parameters**:
- `a` (re-ranking ratio): 0.05-0.15 typical. Lower = faster but less accurate.
- `ef` (search queue): 32-128 typical. Higher = more accurate but slower.

### Algorithm 3: High-Degree Preserving Pruning

```
FUNCTION HighDegreePreservingPrune(graph G, ef, M, m, hub_percentage a):
    INPUT:
        G: Original HNSW graph with vertices V
        ef: Candidate list size for neighbor search
        M: Maximum degree for hub nodes
        m: Maximum degree for regular nodes (m < M)
        a: Percentage of nodes to identify as hubs (e.g., 2%)
    OUTPUT:
        G1: Pruned graph

    1. // Calculate original degrees
    2. D <- empty map
    3. FOR each v in V:
    4.     D[v] <- out_degree(v, G)

    5. // Identify hub nodes (top a% by degree)
    6. sorted_nodes <- sort_descending(V, by=D)
    7. hub_count <- ceil(|V| * a / 100)
    8. V_hub <- sorted_nodes[0..hub_count]

    9. // Build pruned graph
    10. G1 <- empty graph with vertices V

    11. FOR each v in V:
    12.     // Find candidate neighbors
    13.     W <- search(G, v, ef)  // Best-first search for neighbors
    14.
    15.     // Determine degree limit
    16.     IF v in V_hub:
    17.         M_limit <- M
    18.     ELSE:
    19.         M_limit <- m
    20.
    21.     // Select neighbors using HNSW heuristic
    22.     selected <- select_neighbors_heuristic(v, W, M_limit)
    23.
    24.     // Add bidirectional edges
    25.     FOR each s in selected:
    26.         add_edge(G1, v, s)
    27.         add_edge(G1, s, v)
    28.
    29.         // Enforce degree limits on existing nodes
    30.         IF out_degree(s, G1) > M:
    31.             shrink_connections(G1, s, M)

    32. RETURN G1
```

**HNSW Neighbor Selection Heuristic** (used in step 22):

```
FUNCTION select_neighbors_heuristic(v, candidates W, M):
    INPUT:
        v: target node
        W: candidate neighbors
        M: maximum number to select
    OUTPUT:
        R: selected neighbors

    1. R <- empty
    2. W_sorted <- sort_ascending(W, by=distance_to(v))

    3. FOR each e in W_sorted:
    4.     IF |R| >= M:
    5.         BREAK
    6.
    7.     // Check if e is closer to v than to any already-selected neighbor
    8.     good <- true
    9.     FOR each r in R:
    10.        IF distance(e, r) < distance(e, v):
    11.            good <- false
    12.            BREAK
    13.
    14.    IF good:
    15.        R <- R + {e}

    16. RETURN R
```

**Pruning Parameters**:
- `M` (hub degree): 30-64 typical
- `m` (regular degree): 8-16 typical
- `a` (hub percentage): 2-5% typical

### Algorithm 4: Dynamic Batching

```
FUNCTION DynamicBatchSearch(graph G, query q, batch_size B, ...):
    // Same as TwoLevelSearch but with batched recomputation

    1. batch <- empty list
    2. batch_results <- empty map

    // In the recomputation phase (lines 21-27 of Algorithm 2):
    FOR each m in M:
        batch <- batch + {m}

        IF |batch| >= B:
            // Batch recompute all at once
            embeddings <- batch_recompute(batch)
            FOR each (node, emb) in zip(batch, embeddings):
                d_exact <- distance(emb, q)
                batch_results[node] <- d_exact
            batch <- empty

    // Flush remaining batch
    IF |batch| > 0:
        embeddings <- batch_recompute(batch)
        ...
```

**Batch Size Selection**:
- GPU (A10): 64 nodes
- CPU: 8-16 nodes
- Determined by offline profiling for target hardware

---

## Mathematical Formulas

### Distance Metrics

**Euclidean (L2) Distance**:
$$d_{L2}(x, y) = \sqrt{\sum_{i=1}^{d} (x_i - y_i)^2}$$

**Cosine Similarity** (converted to distance):
$$d_{cos}(x, y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$$

**Inner Product** (for normalized vectors):
$$d_{ip}(x, y) = -x \cdot y$$

### Product Quantization Distance

For a query $q$ and database vector $x$ with PQ codes $c = (c_1, ..., c_m)$:

$$d_{PQ}(q, x) = \sum_{j=1}^{m} d(q_j, C_j[c_j])$$

Where:
- $m$ = number of subspaces
- $q_j$ = query subvector for subspace $j$
- $C_j[c_j]$ = centroid $c_j$ from codebook for subspace $j$

**Asymmetric Distance Computation (ADC)**:
Precompute distance table $T$ where $T[j][k] = d(q_j, C_j[k])$
Then: $d_{PQ}(q, x) = \sum_{j=1}^{m} T[j][c_j]$

### Layer Assignment Probability

For HNSW layer assignment:
$$l = \lfloor -\ln(\text{uniform}(0,1)) \cdot m_L \rfloor$$

Where $m_L = 1 / \ln(M)$ and $M$ is the target degree.

### Storage Optimization Objective

Minimize latency subject to storage constraint:

$$\min T = \frac{\sum_{i=1}^{ef} |D_i|}{\text{Throughput}}$$

Subject to:
$$\sum_{i=1}^{n} |D_i| \times \text{sizeof}(\text{NodeId}) < C$$

Where:
- $D_i$ = degree of node $i$
- $ef$ = search queue size
- $C$ = storage budget
- Throughput = embedding server capacity

---

## Rust Implementation Patterns

### Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LeannError {
    #[error("Graph construction failed: {0}")]
    ConstructionError(String),

    #[error("Node {0} not found in graph")]
    NodeNotFound(NodeId),

    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("PQ codebook not trained")]
    PQNotInitialized,

    #[error("Embedding provider error: {0}")]
    EmbeddingError(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, LeannError>;
```

### Trait Definitions

```rust
/// Embedding provider trait for on-demand recomputation
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Batch embed multiple texts (more efficient)
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Embedding dimension
    fn dimension(&self) -> usize;
}

/// Distance metric trait
pub trait DistanceMetric: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Name for serialization
    fn name(&self) -> &'static str;
}

/// Index trait for different backends (HNSW, DiskANN)
pub trait VectorIndex: Send + Sync {
    fn insert(&mut self, id: NodeId, embedding: &[f32]) -> Result<()>;

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>>;

    fn remove(&mut self, id: NodeId) -> Result<()>;

    fn len(&self) -> usize;
}
```

### SIMD-Optimized Distance Computation

```rust
/// L2 distance with manual SIMD (stable Rust)
#[cfg(target_arch = "x86_64")]
pub fn l2_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len());
    let n = a.len();

    unsafe {
        let mut sum = _mm256_setzero_ps();
        let chunks = n / 8;

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum
        let sum128 = _mm_add_ps(
            _mm256_extractf128_ps(sum, 0),
            _mm256_extractf128_ps(sum, 1)
        );
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder
        for i in (chunks * 8)..n {
            let diff = a[i] - b[i];
            result += diff * diff;
        }

        result.sqrt()
    }
}

/// Fallback for non-x86
#[cfg(not(target_arch = "x86_64"))]
pub fn l2_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
```

### Memory-Efficient Graph Storage

```rust
/// Compressed Sparse Row format for graph edges
pub struct CSRGraph {
    /// Cumulative edge counts: edges for node i are in
    /// edge_indices[row_ptr[i]..row_ptr[i+1]]
    row_ptr: Vec<u32>,

    /// Target node IDs for each edge
    edge_indices: Vec<NodeId>,

    /// Number of nodes
    num_nodes: usize,
}

impl CSRGraph {
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        let start = self.row_ptr[node as usize] as usize;
        let end = self.row_ptr[node as usize + 1] as usize;
        &self.edge_indices[start..end]
    }

    pub fn degree(&self, node: NodeId) -> usize {
        let start = self.row_ptr[node as usize];
        let end = self.row_ptr[node as usize + 1];
        (end - start) as usize
    }
}
```

### Thread-Safe Search State

```rust
use parking_lot::RwLock;
use std::sync::Arc;

pub struct ConcurrentHnswIndex {
    graph: Arc<RwLock<HnswGraph>>,
    pq: Arc<ProductQuantizer>,
    config: SearchConfig,
}

impl ConcurrentHnswIndex {
    /// Concurrent search - multiple queries can run simultaneously
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let graph = self.graph.read();
        // Search is read-only, no write lock needed
        self.two_level_search(&graph, query, k)
    }

    /// Insert requires write lock
    pub fn insert(&self, id: NodeId, embedding: &[f32]) -> Result<()> {
        let mut graph = self.graph.write();
        // ... insertion logic
        Ok(())
    }
}
```

---

## Performance Optimization Strategies

### 1. Cache PQ Distance Tables

Precompute query-to-centroid distances once per query:

```rust
impl ProductQuantizer {
    pub fn compute_distance_table(&self, query: &[f32]) -> PQDistanceTable {
        let mut distances = Array2::zeros((self.num_subspaces, self.num_centroids));

        for sub in 0..self.num_subspaces {
            let q_sub = &query[sub * self.subspace_dim..(sub + 1) * self.subspace_dim];
            for cent in 0..self.num_centroids {
                let c = self.codebook.slice(s![sub, cent, ..]);
                distances[[sub, cent]] = l2_distance_squared(q_sub, c.as_slice().unwrap());
            }
        }

        PQDistanceTable { distances }
    }

    pub fn asymmetric_distance(&self, table: &PQDistanceTable, code: &PQCode) -> f32 {
        code.codes.iter().enumerate()
            .map(|(sub, &cent)| table.distances[[sub, cent as usize]])
            .sum::<f32>()
            .sqrt()
    }
}
```

### 2. Hub Node Embedding Caching

```rust
pub struct HubCache {
    /// LRU cache for hub node embeddings
    cache: lru::LruCache<NodeId, Vec<f32>>,

    /// Set of hub node IDs
    hub_nodes: HashSet<NodeId>,

    /// Maximum cache size in bytes
    max_bytes: usize,
}

impl HubCache {
    pub fn get_or_compute<F>(&mut self, node: NodeId, compute: F) -> Vec<f32>
    where
        F: FnOnce() -> Vec<f32>
    {
        if let Some(emb) = self.cache.get(&node) {
            return emb.clone();
        }

        let emb = compute();
        if self.hub_nodes.contains(&node) {
            self.cache.put(node, emb.clone());
        }
        emb
    }
}
```

### 3. Batch Recomputation Pipeline

```rust
pub struct BatchRecomputer<P: EmbeddingProvider> {
    provider: Arc<P>,
    batch_size: usize,
    pending: Vec<(NodeId, String)>,  // (node_id, source_text)
}

impl<P: EmbeddingProvider> BatchRecomputer<P> {
    pub async fn queue(&mut self, node: NodeId, text: String) {
        self.pending.push((node, text));
    }

    pub async fn flush(&mut self) -> Result<Vec<(NodeId, Vec<f32>)>> {
        if self.pending.is_empty() {
            return Ok(vec![]);
        }

        let texts: Vec<&str> = self.pending.iter().map(|(_, t)| t.as_str()).collect();
        let embeddings = self.provider.embed_batch(&texts).await?;

        let results: Vec<_> = self.pending.drain(..)
            .zip(embeddings)
            .map(|((id, _), emb)| (id, emb))
            .collect();

        Ok(results)
    }

    pub fn should_flush(&self) -> bool {
        self.pending.len() >= self.batch_size
    }
}
```

### 4. Memory-Mapped Graph Files

```rust
use memmap2::Mmap;
use std::fs::File;

pub struct MmapGraph {
    mmap: Mmap,
    header: GraphHeader,
}

#[repr(C)]
struct GraphHeader {
    magic: [u8; 4],          // "HNSW"
    version: u32,
    num_nodes: u32,
    max_layer: u8,
    _padding: [u8; 3],
    row_ptr_offset: u64,
    edges_offset: u64,
    metadata_offset: u64,
}

impl MmapGraph {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let header: GraphHeader = unsafe {
            std::ptr::read(mmap.as_ptr() as *const GraphHeader)
        };

        Ok(Self { mmap, header })
    }

    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        // Zero-copy access to neighbors via mmap
        let row_ptr = self.row_ptr_slice();
        let edges = self.edges_slice();

        let start = row_ptr[node as usize] as usize;
        let end = row_ptr[node as usize + 1] as usize;
        &edges[start..end]
    }
}
```

---

## Test Case Specifications

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_distance_accuracy() {
        let dim = 128;
        let num_subspaces = 8;
        let num_centroids = 256;

        let pq = ProductQuantizer::train(training_data, num_subspaces, num_centroids);

        // Test that PQ distance approximates true distance
        for (a, b) in test_pairs {
            let true_dist = l2_distance(&a, &b);
            let pq_dist = pq.distance(&pq.encode(&a), &pq.encode(&b));

            // PQ should be within 20% of true distance for reasonable codebooks
            assert!((pq_dist - true_dist).abs() / true_dist < 0.2);
        }
    }

    #[test]
    fn test_hub_identification() {
        let graph = build_test_graph(1000);
        let hub_pct = 0.02;

        let hubs = identify_hubs(&graph, hub_pct);

        assert_eq!(hubs.len(), 20);  // 2% of 1000

        // Hubs should have higher degree than non-hubs
        let hub_avg_degree: f32 = hubs.iter()
            .map(|&h| graph.degree(h) as f32)
            .sum::<f32>() / hubs.len() as f32;
        let non_hub_avg_degree: f32 = (0..1000)
            .filter(|n| !hubs.contains(&(*n as NodeId)))
            .map(|n| graph.degree(n as NodeId) as f32)
            .sum::<f32>() / (1000 - hubs.len()) as f32;

        assert!(hub_avg_degree > non_hub_avg_degree * 2.0);
    }

    #[test]
    fn test_two_level_search_correctness() {
        let (graph, embeddings) = build_indexed_test_set(10000, 128);
        let query = random_embedding(128);

        // Brute force ground truth
        let mut true_neighbors: Vec<_> = embeddings.iter()
            .enumerate()
            .map(|(i, e)| (i as NodeId, l2_distance(&query, e)))
            .collect();
        true_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_top10: Vec<_> = true_neighbors[..10].iter().map(|(id, _)| *id).collect();

        // Two-level search
        let results = two_level_search(&graph, &query, 10, 64, 0.1);
        let search_top10: Vec<_> = results.iter().map(|r| r.node_id).collect();

        // Should have at least 90% recall
        let recall = true_top10.iter()
            .filter(|id| search_top10.contains(id))
            .count() as f32 / 10.0;

        assert!(recall >= 0.9, "Recall {recall} below 0.9 threshold");
    }

    #[test]
    fn test_pruning_preserves_connectivity() {
        let original = build_test_graph(1000);
        let pruned = prune_graph(&original, 30, 8, 0.02);

        // All nodes should still be reachable from entry point
        let reachable = bfs_reachable(&pruned, pruned.entry_point);
        assert_eq!(reachable.len(), 1000);

        // Pruned graph should be smaller
        let original_edges: usize = (0..1000).map(|n| original.degree(n as NodeId)).sum();
        let pruned_edges: usize = (0..1000).map(|n| pruned.degree(n as NodeId)).sum();
        assert!(pruned_edges < original_edges);
    }
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_pq_distance_symmetric(
        a in prop::collection::vec(-1.0f32..1.0, 128),
        b in prop::collection::vec(-1.0f32..1.0, 128)
    ) {
        let pq = ProductQuantizer::default();
        let code_a = pq.encode(&a);
        let code_b = pq.encode(&b);

        let d1 = pq.distance(&code_a, &code_b);
        let d2 = pq.distance(&code_b, &code_a);

        prop_assert!((d1 - d2).abs() < 1e-6);
    }

    #[test]
    fn prop_search_returns_k_results(
        query in prop::collection::vec(-1.0f32..1.0, 128),
        k in 1usize..100
    ) {
        let index = build_test_index(1000, 128);
        let results = index.search(&query, k);

        prop_assert!(results.len() <= k);
        prop_assert!(results.len() == k.min(1000));
    }
}
```

### Benchmark Suite

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");

    for size in [1000, 10000, 100000, 1000000] {
        let index = build_index(size, 128);
        let query = random_embedding(128);

        group.bench_with_input(
            BenchmarkId::new("two_level", size),
            &size,
            |b, _| {
                b.iter(|| index.two_level_search(&query, 10, 64, 0.1))
            }
        );

        group.bench_with_input(
            BenchmarkId::new("baseline", size),
            &size,
            |b, _| {
                b.iter(|| index.best_first_search(&query, 10, 64))
            }
        );
    }

    group.finish();
}

fn bench_pq_distance(c: &mut Criterion) {
    let pq = ProductQuantizer::new(128, 8, 256);
    let query = random_embedding(128);
    let table = pq.compute_distance_table(&query);
    let codes: Vec<_> = (0..1000).map(|_| random_pq_code(8)).collect();

    c.bench_function("pq_distance_1000", |b| {
        b.iter(|| {
            for code in &codes {
                pq.asymmetric_distance(&table, code);
            }
        })
    });
}

fn bench_simd_distance(c: &mut Criterion) {
    let a = random_embedding(768);
    let b = random_embedding(768);

    c.bench_function("l2_simd_768d", |b| {
        b.iter(|| l2_distance_simd(&a, &b))
    });

    c.bench_function("l2_scalar_768d", |b| {
        b.iter(|| l2_distance_scalar(&a, &b))
    });
}

criterion_group!(benches, bench_search, bench_pq_distance, bench_simd_distance);
criterion_main!(benches);
```

---

## File Format Specifications

### Graph File Format (.hnsw)

```
Header (64 bytes):
  - magic: [u8; 4] = "HNSW"
  - version: u32 = 1
  - num_nodes: u32
  - num_layers: u8
  - entry_point: u32
  - metric: u8 (0=L2, 1=Cosine, 2=InnerProduct)
  - dimension: u32
  - m: u16
  - ef_construction: u16
  - _reserved: [u8; 40]

Node Sections (per layer):
  - layer_header: { layer_id: u8, num_nodes: u32, edges_start: u64 }
  - row_ptr: [u32; num_nodes + 1]
  - edges: [u32; total_edges]

Metadata Section:
  - num_hubs: u32
  - hub_ids: [u32; num_hubs]
  - source_refs: [SourceRef; num_nodes]  // For embedding recomputation
```

### PQ Codebook File Format (.pq)

```
Header (32 bytes):
  - magic: [u8; 4] = "PQCB"
  - version: u32 = 1
  - num_subspaces: u16
  - num_centroids: u16
  - subspace_dim: u16
  - _reserved: [u8; 18]

Codebook Data:
  - centroids: [f32; num_subspaces * num_centroids * subspace_dim]

Precomputed Norms (optional):
  - centroid_norms: [f32; num_subspaces * num_centroids]
```

### PQ Codes File Format (.codes)

```
Header (16 bytes):
  - magic: [u8; 4] = "PQCD"
  - version: u32 = 1
  - num_vectors: u32
  - num_subspaces: u8
  - _reserved: [u8; 3]

Codes:
  - codes: [u8; num_vectors * num_subspaces]  // Packed, row-major
```

---

## Configuration Defaults

```rust
pub struct LeannConfig {
    // HNSW construction
    pub m: usize,                  // 16
    pub m0: usize,                 // 32
    pub ef_construction: usize,   // 128

    // Pruning
    pub hub_degree_m: usize,       // 30
    pub regular_degree_m: usize,   // 8
    pub hub_percentage: f32,       // 0.02 (2%)

    // Search
    pub ef_search: usize,          // 64
    pub rerank_ratio: f32,         // 0.1 (10%)

    // Product Quantization
    pub pq_num_subspaces: usize,   // 8
    pub pq_num_centroids: usize,   // 256
    pub pq_training_samples: usize, // 100_000

    // Dynamic Batching
    pub batch_size: usize,         // 64 for GPU, 16 for CPU
    pub batch_timeout_ms: u64,     // 10

    // Caching
    pub hub_cache_size_mb: usize,  // 256

    // Distance metric
    pub metric: DistanceMetric,    // L2
}
```

---

## References

1. LEANN Paper: https://arxiv.org/abs/2506.08276
2. LEANN GitHub: https://github.com/yichuan-w/LEANN
3. HNSW Original Paper: https://arxiv.org/abs/1603.09320
4. Product Quantization: https://hal.inria.fr/inria-00514462
5. PAIML Aprender (Rust ML patterns): https://github.com/paiml/aprender
