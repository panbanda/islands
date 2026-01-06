//! Search interface and result types

use crate::error::CoreResult;
use crate::hnsw::HnswGraph;
use serde::{Deserialize, Serialize};

/// Configuration for search operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Number of results to return
    pub top_k: usize,
    /// Exploration factor (higher = more accurate, slower)
    pub ef: usize,
    /// Whether to include vector data in results
    pub include_vectors: bool,
    /// Whether to include metadata in results
    pub include_metadata: bool,
    /// Minimum similarity threshold (results below this are filtered)
    pub min_similarity: Option<f32>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            ef: 100,
            include_vectors: false,
            include_metadata: true,
            min_similarity: None,
        }
    }
}

impl SearchConfig {
    /// Create config for fast search
    pub fn fast(k: usize) -> Self {
        Self {
            top_k: k,
            ef: k * 2,
            ..Default::default()
        }
    }

    /// Create config for accurate search
    pub fn accurate(k: usize) -> Self {
        Self {
            top_k: k,
            ef: k * 10,
            ..Default::default()
        }
    }
}

/// A single search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Node/vector ID
    pub id: u64,
    /// Distance or similarity score
    pub score: f32,
    /// Optional vector data
    pub vector: Option<Vec<f32>>,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
    /// Optional associated text
    pub text: Option<String>,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(id: u64, score: f32) -> Self {
        Self {
            id,
            score,
            vector: None,
            metadata: None,
            text: None,
        }
    }

    /// Add vector to result
    pub fn with_vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Add metadata to result
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Add text to result
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Convert distance to similarity (1 / (1 + distance))
    pub fn to_similarity(&self) -> f32 {
        1.0 / (1.0 + self.score)
    }
}

/// Searcher interface for HNSW graph
pub struct Searcher<'a> {
    graph: &'a HnswGraph,
    config: SearchConfig,
}

impl<'a> Searcher<'a> {
    /// Create a new searcher
    pub fn new(graph: &'a HnswGraph) -> Self {
        Self {
            graph,
            config: SearchConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(graph: &'a HnswGraph, config: SearchConfig) -> Self {
        Self { graph, config }
    }

    /// Set top_k
    pub fn top_k(mut self, k: usize) -> Self {
        self.config.top_k = k;
        self
    }

    /// Set ef (exploration factor)
    pub fn ef(mut self, ef: usize) -> Self {
        self.config.ef = ef;
        self
    }

    /// Include vectors in results
    pub fn include_vectors(mut self) -> Self {
        self.config.include_vectors = true;
        self
    }

    /// Set minimum similarity threshold
    pub fn min_similarity(mut self, threshold: f32) -> Self {
        self.config.min_similarity = Some(threshold);
        self
    }

    /// Execute search
    pub fn search(&self, query: &[f32]) -> CoreResult<Vec<SearchResult>> {
        let raw_results = self
            .graph
            .search(query, self.config.top_k, self.config.ef)?;

        let mut results: Vec<SearchResult> = raw_results
            .into_iter()
            .map(|(id, distance)| {
                let mut result = SearchResult::new(id, distance);

                if self.config.include_vectors {
                    if let Some(node) = self.graph.get_node(id) {
                        result = result.with_vector(node.vector.clone());
                    }
                }

                result
            })
            .collect();

        // Apply similarity threshold
        if let Some(min_sim) = self.config.min_similarity {
            results.retain(|r| r.to_similarity() >= min_sim);
        }

        Ok(results)
    }

    /// Execute batch search
    pub fn search_batch(&self, queries: &[Vec<f32>]) -> CoreResult<Vec<Vec<SearchResult>>> {
        queries.iter().map(|q| self.search(q)).collect()
    }
}

/// Multi-index searcher for searching across multiple graphs
pub struct MultiIndexSearcher {
    graphs: Vec<(String, HnswGraph)>,
    config: SearchConfig,
}

impl MultiIndexSearcher {
    /// Create a new multi-index searcher
    pub fn new() -> Self {
        Self {
            graphs: Vec::new(),
            config: SearchConfig::default(),
        }
    }

    /// Add an index
    pub fn add_index(&mut self, name: impl Into<String>, graph: HnswGraph) {
        self.graphs.push((name.into(), graph));
    }

    /// Set search config
    pub fn with_config(mut self, config: SearchConfig) -> Self {
        self.config = config;
        self
    }

    /// Search all indexes and merge results
    pub fn search(&self, query: &[f32]) -> CoreResult<Vec<(String, SearchResult)>> {
        let mut all_results = Vec::new();

        for (name, graph) in &self.graphs {
            let results = graph.search(query, self.config.top_k, self.config.ef)?;

            for (id, score) in results {
                let mut result = SearchResult::new(id, score);

                if self.config.include_vectors {
                    if let Some(node) = graph.get_node(id) {
                        result = result.with_vector(node.vector.clone());
                    }
                }

                all_results.push((name.clone(), result));
            }
        }

        // Sort by score
        all_results.sort_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap());

        // Take top_k
        all_results.truncate(self.config.top_k);

        Ok(all_results)
    }

    /// Get number of indexes
    pub fn num_indexes(&self) -> usize {
        self.graphs.len()
    }

    /// Get total number of vectors across all indexes
    pub fn total_vectors(&self) -> usize {
        self.graphs.iter().map(|(_, g)| g.len()).sum()
    }
}

impl Default for MultiIndexSearcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswConfig;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    fn create_test_graph(n: usize, dim: usize, seed: u64) -> HnswGraph {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut graph = HnswGraph::new(HnswConfig::fast()).unwrap();

        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
            graph.insert(v).unwrap();
        }

        graph
    }

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.top_k, 10);
        assert_eq!(config.ef, 100);
        assert!(!config.include_vectors);
        assert!(config.include_metadata);
    }

    #[test]
    fn test_search_config_fast() {
        let config = SearchConfig::fast(5);
        assert_eq!(config.top_k, 5);
        assert_eq!(config.ef, 10);
    }

    #[test]
    fn test_search_config_accurate() {
        let config = SearchConfig::accurate(5);
        assert_eq!(config.top_k, 5);
        assert_eq!(config.ef, 50);
    }

    #[test]
    fn test_search_result() {
        let result = SearchResult::new(42, 0.5)
            .with_vector(vec![1.0, 2.0])
            .with_text("hello");

        assert_eq!(result.id, 42);
        assert_eq!(result.score, 0.5);
        assert_eq!(result.vector, Some(vec![1.0, 2.0]));
        assert_eq!(result.text, Some("hello".to_string()));
    }

    #[test]
    fn test_search_result_to_similarity() {
        // distance 0 -> similarity 1.0
        let r1 = SearchResult::new(0, 0.0);
        assert_eq!(r1.to_similarity(), 1.0);

        // distance 1 -> similarity 0.5
        let r2 = SearchResult::new(0, 1.0);
        assert_eq!(r2.to_similarity(), 0.5);

        // larger distance -> smaller similarity
        let r3 = SearchResult::new(0, 9.0);
        assert_eq!(r3.to_similarity(), 0.1);
    }

    #[test]
    fn test_searcher_basic() {
        let graph = create_test_graph(50, 16, 42);
        let searcher = Searcher::new(&graph);

        let query = vec![0.5f32; 16];
        let results = searcher.search(&query).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 10); // Default top_k
    }

    #[test]
    fn test_searcher_top_k() {
        let graph = create_test_graph(50, 16, 42);
        let searcher = Searcher::new(&graph).top_k(5);

        let query = vec![0.5f32; 16];
        let results = searcher.search(&query).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_searcher_include_vectors() {
        let graph = create_test_graph(20, 8, 42);
        let searcher = Searcher::new(&graph).include_vectors().top_k(3);

        let query = vec![0.5f32; 8];
        let results = searcher.search(&query).unwrap();

        for result in &results {
            assert!(result.vector.is_some());
            assert_eq!(result.vector.as_ref().unwrap().len(), 8);
        }
    }

    #[test]
    fn test_searcher_min_similarity() {
        let graph = create_test_graph(50, 16, 42);
        let searcher = Searcher::new(&graph).min_similarity(0.5);

        let query = vec![0.5f32; 16];
        let results = searcher.search(&query).unwrap();

        for result in &results {
            assert!(result.to_similarity() >= 0.5);
        }
    }

    #[test]
    fn test_searcher_results_sorted() {
        let graph = create_test_graph(50, 16, 42);
        let searcher = Searcher::new(&graph).top_k(10);

        let query = vec![0.5f32; 16];
        let results = searcher.search(&query).unwrap();

        for window in results.windows(2) {
            assert!(window[0].score <= window[1].score);
        }
    }

    #[test]
    fn test_batch_search() {
        let graph = create_test_graph(50, 8, 42);
        let searcher = Searcher::new(&graph).top_k(5);

        let queries: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32 * 0.1; 8]).collect();
        let batch_results = searcher.search_batch(&queries).unwrap();

        assert_eq!(batch_results.len(), 3);
        for results in batch_results {
            assert_eq!(results.len(), 5);
        }
    }

    #[test]
    fn test_multi_index_searcher() {
        let graph1 = create_test_graph(20, 8, 42);
        let graph2 = create_test_graph(20, 8, 43);

        let mut searcher = MultiIndexSearcher::new();
        searcher.add_index("index1", graph1);
        searcher.add_index("index2", graph2);

        assert_eq!(searcher.num_indexes(), 2);
        assert_eq!(searcher.total_vectors(), 40);

        let query = vec![0.5f32; 8];
        let results = searcher.search(&query).unwrap();

        assert!(!results.is_empty());
        // Results should come from both indexes
        let _sources: std::collections::HashSet<_> =
            results.iter().map(|(name, _)| name.as_str()).collect();
        // May get results from one or both indexes depending on top_k
    }

    #[test]
    fn test_multi_index_searcher_empty() {
        let searcher = MultiIndexSearcher::new();
        assert_eq!(searcher.num_indexes(), 0);
        assert_eq!(searcher.total_vectors(), 0);
    }

    #[rstest]
    #[case(1)]
    #[case(5)]
    #[case(10)]
    #[case(20)]
    fn test_various_k_values(#[case] k: usize) {
        let graph = create_test_graph(50, 8, 42);
        let searcher = Searcher::new(&graph).top_k(k);

        let query = vec![0.5f32; 8];
        let results = searcher.search(&query).unwrap();

        assert_eq!(results.len(), k.min(50));
    }

    #[test]
    fn test_search_result_serialization() {
        let result = SearchResult::new(42, 0.5)
            .with_metadata(serde_json::json!({"file": "test.rs"}))
            .with_text("sample text");

        let json = serde_json::to_string(&result).unwrap();
        let parsed: SearchResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, result.id);
        assert_eq!(parsed.score, result.score);
        assert_eq!(parsed.text, result.text);
    }

    #[test]
    fn test_search_config_serialization() {
        let config = SearchConfig {
            top_k: 20,
            ef: 200,
            include_vectors: true,
            include_metadata: false,
            min_similarity: Some(0.7),
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: SearchConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.top_k, 20);
        assert_eq!(parsed.ef, 200);
        assert!(parsed.include_vectors);
        assert!(!parsed.include_metadata);
        assert_eq!(parsed.min_similarity, Some(0.7));
    }
}
