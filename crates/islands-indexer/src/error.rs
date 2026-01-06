//! Error types for islands-indexer

use thiserror::Error;

/// Result type alias for indexer operations
pub type Result<T> = std::result::Result<T, Error>;

/// Comprehensive error type for indexer operations
#[derive(Error, Debug)]
pub enum Error {
    /// Git operation failed
    #[error("git error: {0}")]
    Git(#[from] git2::Error),

    /// Repository not found
    #[error("repository not found: {0}")]
    RepositoryNotFound(String),

    /// Repository already exists
    #[error("repository already exists: {0}")]
    RepositoryExists(String),

    /// Clone failed
    #[error("clone failed: {0}")]
    CloneFailed(String),

    /// Index not found
    #[error("index not found: {0}")]
    IndexNotFound(String),

    /// Indexing failed
    #[error("indexing failed: {0}")]
    IndexingFailed(String),

    /// Provider error
    #[error("provider error: {0}")]
    Provider(#[from] islands_providers::Error),

    /// Core error
    #[error("core error: {0}")]
    Core(#[from] islands_core::Error),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Configuration error
    #[error("configuration error: {0}")]
    Config(String),

    /// Sync error
    #[error("sync error: {0}")]
    Sync(String),
}

impl Error {
    /// Create a repository not found error
    #[must_use]
    pub fn repo_not_found(name: impl Into<String>) -> Self {
        Self::RepositoryNotFound(name.into())
    }

    /// Create an index not found error
    #[must_use]
    pub fn index_not_found(name: impl Into<String>) -> Self {
        Self::IndexNotFound(name.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_repo_not_found() {
        let err = Error::repo_not_found("test/repo");
        assert!(matches!(err, Error::RepositoryNotFound(_)));
        assert_eq!(err.to_string(), "repository not found: test/repo");
    }

    #[test]
    fn test_error_index_not_found() {
        let err = Error::index_not_found("test-index");
        assert!(matches!(err, Error::IndexNotFound(_)));
        assert_eq!(err.to_string(), "index not found: test-index");
    }

    #[test]
    fn test_error_display() {
        let err = Error::CloneFailed("network error".to_string());
        assert_eq!(err.to_string(), "clone failed: network error");

        let err = Error::IndexingFailed("out of memory".to_string());
        assert_eq!(err.to_string(), "indexing failed: out of memory");

        let err = Error::Config("invalid setting".to_string());
        assert_eq!(err.to_string(), "configuration error: invalid setting");

        let err = Error::Sync("connection lost".to_string());
        assert_eq!(err.to_string(), "sync error: connection lost");

        let err = Error::RepositoryExists("owner/repo".to_string());
        assert_eq!(err.to_string(), "repository already exists: owner/repo");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_error_from_json() {
        let json_str = "invalid json {";
        let json_err = serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
    }

    #[test]
    fn test_error_from_core() {
        let core_err = islands_core::Error::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        let err: Error = core_err.into();
        assert!(matches!(err, Error::Core(_)));
    }

    #[test]
    fn test_error_debug() {
        let err = Error::repo_not_found("test/repo");
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("RepositoryNotFound"));
    }
}
