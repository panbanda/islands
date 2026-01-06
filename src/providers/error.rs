//! Error types for islands-providers.

use thiserror::Error;

/// Result type alias for provider operations.
pub type Result<T> = std::result::Result<T, ProviderError>;

/// Comprehensive error type for provider operations.
#[derive(Error, Debug)]
pub enum ProviderError {
    /// HTTP request failed.
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    /// JSON parsing failed.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Authentication failed or credentials missing.
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// Rate limit exceeded.
    #[error("Rate limit exceeded, retry after {retry_after_secs} seconds")]
    RateLimitExceeded {
        /// Seconds until rate limit resets.
        retry_after_secs: u64,
    },

    /// Resource not found (repository, branch, commit, etc.).
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Repository not found with owner/name details.
    #[error("Repository not found: {owner}/{name}")]
    RepositoryNotFound {
        /// Repository owner.
        owner: String,
        /// Repository name.
        name: String,
    },

    /// Invalid webhook signature.
    #[error("Invalid webhook signature")]
    InvalidWebhookSignature,

    /// Webhook parsing failed.
    #[error("Webhook parsing failed: {0}")]
    WebhookParseError(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    ConfigurationError(String),

    /// Unsupported provider type.
    #[error("Unsupported provider type: {0}")]
    UnsupportedProvider(String),

    /// API returned an error response.
    #[error("API error: {status} - {message}")]
    ApiError {
        /// HTTP status code.
        status: u16,
        /// Error message from API.
        message: String,
    },

    /// URL parsing failed.
    #[error("URL parsing failed: {0}")]
    UrlParseError(#[from] url::ParseError),

    /// Generic operation failed.
    #[error("{0}")]
    OperationFailed(String),
}

impl ProviderError {
    /// Create a repository not found error.
    #[must_use]
    pub fn repo_not_found(owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self::RepositoryNotFound {
            owner: owner.into(),
            name: name.into(),
        }
    }

    /// Create a rate limited error.
    #[must_use]
    pub fn rate_limited(retry_after_secs: u64) -> Self {
        Self::RateLimitExceeded { retry_after_secs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ProviderError::NotFound("repository foo/bar".to_string());
        assert_eq!(err.to_string(), "Resource not found: repository foo/bar");
    }

    #[test]
    fn test_rate_limit_error_display() {
        let err = ProviderError::RateLimitExceeded {
            retry_after_secs: 60,
        };
        assert_eq!(
            err.to_string(),
            "Rate limit exceeded, retry after 60 seconds"
        );
    }

    #[test]
    fn test_api_error_display() {
        let err = ProviderError::ApiError {
            status: 404,
            message: "Not Found".to_string(),
        };
        assert_eq!(err.to_string(), "API error: 404 - Not Found");
    }

    #[test]
    fn test_repo_not_found_helper() {
        let err = ProviderError::repo_not_found("owner", "repo");
        assert_eq!(err.to_string(), "Repository not found: owner/repo");
    }

    #[test]
    fn test_rate_limited_helper() {
        let err = ProviderError::rate_limited(120);
        assert_eq!(
            err.to_string(),
            "Rate limit exceeded, retry after 120 seconds"
        );
    }

    #[test]
    fn test_error_from_json() {
        let json_err: serde_json::Error = serde_json::from_str::<String>("invalid").unwrap_err();
        let provider_err: ProviderError = json_err.into();
        assert!(matches!(provider_err, ProviderError::JsonError(_)));
    }

    #[test]
    fn test_webhook_signature_error() {
        let err = ProviderError::InvalidWebhookSignature;
        assert_eq!(err.to_string(), "Invalid webhook signature");
    }

    #[test]
    fn test_authentication_error() {
        let err = ProviderError::AuthenticationError("token expired".to_string());
        assert_eq!(err.to_string(), "Authentication failed: token expired");
    }

    #[test]
    fn test_configuration_error() {
        let err = ProviderError::ConfigurationError("missing base_url".to_string());
        assert_eq!(err.to_string(), "Invalid configuration: missing base_url");
    }

    #[test]
    fn test_unsupported_provider() {
        let err = ProviderError::UnsupportedProvider("codeberg".to_string());
        assert_eq!(err.to_string(), "Unsupported provider type: codeberg");
    }
}
