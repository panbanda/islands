//! Error types for pythia-cli

use thiserror::Error;

/// Result type alias for CLI operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for CLI operations
#[derive(Error, Debug)]
pub enum Error {
    /// Indexer error
    #[error("indexer error: {0}")]
    Indexer(#[from] pythia_indexer::Error),

    /// Provider error
    #[error("provider error: {0}")]
    Provider(#[from] pythia_providers::Error),

    /// Configuration error
    #[error("configuration error: {0}")]
    Config(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid argument
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_config() {
        let err = Error::Config("missing field".to_string());
        assert_eq!(err.to_string(), "configuration error: missing field");
    }

    #[test]
    fn test_error_invalid_argument() {
        let err = Error::InvalidArgument("unknown flag".to_string());
        assert_eq!(err.to_string(), "invalid argument: unknown flag");
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
        let json_str = "{ invalid }";
        let json_err = serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
    }

    #[test]
    fn test_error_debug() {
        let err = Error::Config("test error".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Config"));
        assert!(debug_str.contains("test error"));
    }

    #[test]
    fn test_error_indexer_display() {
        let indexer_err = pythia_indexer::Error::IndexNotFound("test-index".to_string());
        let err: Error = indexer_err.into();
        assert!(err.to_string().contains("indexer error"));
    }

    #[test]
    fn test_error_provider_display() {
        let provider_err = pythia_providers::Error::NotFound("bad url".to_string());
        let err: Error = provider_err.into();
        assert!(err.to_string().contains("provider error"));
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_ok() -> Result<i32> {
            Ok(42)
        }

        fn returns_err() -> Result<i32> {
            Err(Error::InvalidArgument("test".into()))
        }

        assert_eq!(returns_ok().unwrap(), 42);
        assert!(returns_err().is_err());
    }

    #[test]
    fn test_error_io_kinds() {
        let err1 = Error::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "access denied",
        ));
        assert!(err1.to_string().contains("access denied"));

        let err2 = Error::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "already exists",
        ));
        assert!(err2.to_string().contains("already exists"));
    }

    #[test]
    fn test_error_config_empty() {
        let err = Error::Config(String::new());
        assert_eq!(err.to_string(), "configuration error: ");
    }

    #[test]
    fn test_error_invalid_argument_special_chars() {
        let err = Error::InvalidArgument("--flag=value with spaces & special!".to_string());
        assert!(err.to_string().contains("--flag=value with spaces & special!"));
    }
}
