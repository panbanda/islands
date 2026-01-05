//! Error types for pythia-core
//!
//! Comprehensive error handling following PAIML patterns.

use thiserror::Error;

/// Core error types for Pythia vector operations
#[derive(Error, Debug)]
pub enum CoreError {
    /// Vector dimension mismatch
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Empty vector collection
    #[error("Empty vector collection")]
    EmptyCollection,

    /// Invalid configuration parameter
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Index not built yet
    #[error("Index not built")]
    IndexNotBuilt,

    /// Node not found in graph
    #[error("Node not found: {0}")]
    NodeNotFound(u64),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// I/O error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// HNSW graph error
    #[error("HNSW graph error: {0}")]
    HnswError(String),

    /// Product quantization error
    #[error("Product quantization error: {0}")]
    PQError(String),

    /// Search error
    #[error("Search error: {0}")]
    SearchError(String),

    /// Embedding error
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
}

/// Result type alias for Core operations
pub type CoreResult<T> = Result<T, CoreError>;

impl CoreError {
    /// Create a dimension mismatch error
    #[must_use]
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create an invalid config error
    #[must_use]
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CoreError::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        assert!(err.to_string().contains("128"));
        assert!(err.to_string().contains("64"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let core_err: CoreError = io_err.into();
        assert!(matches!(core_err, CoreError::Io(_)));
    }

    #[test]
    fn test_error_helpers() {
        let err = CoreError::dimension_mismatch(100, 50);
        assert!(matches!(
            err,
            CoreError::DimensionMismatch {
                expected: 100,
                actual: 50
            }
        ));

        let err = CoreError::invalid_config("bad value");
        assert!(matches!(err, CoreError::InvalidConfig(_)));
    }
}
