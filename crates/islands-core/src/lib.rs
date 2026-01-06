#![allow(
    clippy::manual_is_multiple_of,
    clippy::collapsible_if,
    clippy::field_reassign_with_default,
    clippy::approx_constant,
    clippy::manual_range_contains,
    clippy::clone_on_copy,
    clippy::excessive_precision,
    clippy::unnecessary_lazy_evaluations,
    clippy::explicit_auto_deref,
    unexpected_cfgs
)]
//! Islands Core - LEANN-based vector indexing and search
//!
//! This crate provides the core data structures and algorithms for:
//! - HNSW (Hierarchical Navigable Small World) graph-based vector search
//! - Product Quantization (PQ) for vector compression
//! - Vector distance calculations (cosine, euclidean, dot product)
//! - Embedding management
//!
//! # LEANN Implementation
//!
//! Based on the LEANN paper (arXiv:2506.08276), this implementation includes:
//! - High-degree preserving pruning for maintaining graph quality
//! - Two-level search with approximate and exact distance queues
//! - Selective recomputation for memory efficiency
//!
//! # Example
//!
//! ```rust,no_run
//! use islands_core::prelude::*;
//!
//! # fn main() -> islands_core::CoreResult<()> {
//! // Create an HNSW graph
//! let mut graph = HnswGraph::with_defaults()?;
//!
//! // Insert vectors
//! graph.insert(vec![1.0, 0.0, 0.0])?;
//! graph.insert(vec![0.0, 1.0, 0.0])?;
//! graph.insert(vec![0.0, 0.0, 1.0])?;
//!
//! // Search for nearest neighbors
//! let results = graph.search(&[0.9, 0.1, 0.0], 2, 50)?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod distance;
pub mod embedding;
pub mod error;
pub mod hnsw;
pub mod leann;
pub mod pq;
pub mod search;
pub mod storage;

// Re-export main types
pub use distance::{Distance, DistanceMetric};
pub use embedding::Embedding;
pub use error::{CoreError, CoreResult};
pub use hnsw::{HnswConfig, HnswGraph, HnswNode};
pub use leann::{
    CsrGraph, EmbeddingProvider, InMemoryEmbeddingProvider, LeannConfig, LeannIndex,
    PruningStrategy,
};
pub use pq::{PQCodebook, PQConfig, ProductQuantizer};
pub use search::{SearchConfig, SearchResult, Searcher};

// Embedding provider (requires "embeddings" feature)
#[cfg(feature = "embeddings")]
pub use embedding::{
    CloudProvider, EmbedderConfig, EmbedderProvider, InferenceBackend, ModelArchitecture,
};

/// Type alias for HNSW graph index (compatibility)
pub type Index = HnswGraph;
/// Type alias for index configuration (compatibility)
pub type IndexConfig = HnswConfig;
/// Type alias for index builder (compatibility)
pub type IndexBuilder = HnswConfig;
/// Type alias for core error type (compatibility)
pub type Error = CoreError;

/// Prelude for commonly used types
pub mod prelude {
    pub use crate::distance::{Distance, DistanceMetric};
    pub use crate::embedding::Embedding;
    pub use crate::error::{CoreError, CoreResult};
    pub use crate::hnsw::{HnswConfig, HnswGraph};
    pub use crate::leann::{
        CsrGraph, EmbeddingProvider, InMemoryEmbeddingProvider, LeannConfig, LeannIndex,
        PruningStrategy,
    };
    pub use crate::pq::ProductQuantizer;
    pub use crate::search::{SearchConfig, SearchResult, Searcher};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_exports() {
        // Verify all public types are accessible
        let _: DistanceMetric = DistanceMetric::Cosine;
    }

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        let _: DistanceMetric = DistanceMetric::Euclidean;
        let embedding = Embedding::zeros(8);
        assert_eq!(embedding.dim(), 8);
    }
}
