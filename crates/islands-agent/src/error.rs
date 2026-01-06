//! Error types for islands-agent

use thiserror::Error;

/// Result type alias for agent operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for agent operations
#[derive(Error, Debug)]
pub enum Error {
    /// LLM API error
    #[error("LLM error: {0}")]
    Llm(String),

    /// Rate limit exceeded
    #[error("rate limit exceeded, retry after {0} seconds")]
    RateLimited(u64),

    /// Context too long
    #[error("context too long: {0} tokens exceeds limit of {1}")]
    ContextTooLong(usize, usize),

    /// Invalid configuration
    #[error("invalid configuration: {0}")]
    Config(String),

    /// Indexer error
    #[error("indexer error: {0}")]
    Indexer(#[from] islands_indexer::Error),

    /// HTTP error
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_llm() {
        let err = Error::Llm("connection timeout".to_string());
        assert_eq!(err.to_string(), "LLM error: connection timeout");
    }

    #[test]
    fn test_error_rate_limited() {
        let err = Error::RateLimited(60);
        assert_eq!(err.to_string(), "rate limit exceeded, retry after 60 seconds");
    }

    #[test]
    fn test_error_rate_limited_zero() {
        let err = Error::RateLimited(0);
        assert_eq!(err.to_string(), "rate limit exceeded, retry after 0 seconds");
    }

    #[test]
    fn test_error_context_too_long() {
        let err = Error::ContextTooLong(10000, 8192);
        assert_eq!(
            err.to_string(),
            "context too long: 10000 tokens exceeds limit of 8192"
        );
    }

    #[test]
    fn test_error_config() {
        let err = Error::Config("missing API key".to_string());
        assert_eq!(err.to_string(), "invalid configuration: missing API key");
    }

    #[test]
    fn test_error_from_indexer() {
        let indexer_err = islands_indexer::Error::IndexNotFound("test-index".to_string());
        let err: Error = indexer_err.into();
        assert!(matches!(err, Error::Indexer(_)));
        assert!(err.to_string().contains("indexer error"));
    }

    #[test]
    fn test_error_from_json() {
        let json_str = "{ invalid }";
        let json_err = serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
        assert!(err.to_string().contains("JSON error"));
    }

    #[test]
    fn test_error_debug() {
        let err = Error::Llm("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Llm"));
    }

    #[test]
    fn test_error_llm_empty() {
        let err = Error::Llm(String::new());
        assert_eq!(err.to_string(), "LLM error: ");
    }

    #[test]
    fn test_error_config_empty() {
        let err = Error::Config(String::new());
        assert_eq!(err.to_string(), "invalid configuration: ");
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_ok() -> Result<i32> {
            Ok(42)
        }

        fn returns_err() -> Result<i32> {
            Err(Error::Llm("failure".into()))
        }

        assert_eq!(returns_ok().unwrap(), 42);
        assert!(returns_err().is_err());
    }

    #[test]
    fn test_error_context_too_long_boundary() {
        let err = Error::ContextTooLong(8193, 8192);
        assert!(err.to_string().contains("8193"));
        assert!(err.to_string().contains("8192"));
    }

    #[test]
    fn test_error_rate_limited_large() {
        let err = Error::RateLimited(3600);
        assert!(err.to_string().contains("3600"));
    }
}
