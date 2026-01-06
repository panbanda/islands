#![allow(clippy::collapsible_if)]
//! Git provider implementations for Islands.
//!
//! This crate provides unified access to multiple git hosting platforms:
//!
//! - **GitHub** - github.com and GitHub Enterprise
//! - **GitLab** - gitlab.com and self-hosted instances
//! - **Bitbucket** - Bitbucket Cloud
//! - **Gitea** - Self-hosted Gitea instances
//!
//! # Usage
//!
//! ```rust,no_run
//! use islands::providers::{GitHubProvider, GitProvider};
//!
//! #[tokio::main]
//! async fn main() -> islands::providers::Result<()> {
//!     // Create a GitHub provider with a token
//!     let provider = GitHubProvider::with_token("your-token")?;
//!
//!     // Get a repository
//!     let repo = provider.get_repository("rust-lang", "rust").await?;
//!     println!("Repository: {}", repo.full_name);
//!
//!     Ok(())
//! }
//! ```
//!
//! # Factory Usage
//!
//! Use the factory to create providers dynamically:
//!
//! ```rust,no_run
//! use islands::providers::{ProviderFactory, create_provider};
//!
//! # fn main() -> islands::providers::Result<()> {
//! // Using the factory
//! let factory = ProviderFactory::new();
//! let provider = factory.create("github", None)?;
//!
//! // Or use the convenience function
//! let provider = create_provider("github", None, Some("token"), None)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Webhook Handling
//!
//! All providers support webhook parsing and signature verification:
//!
//! ```rust,no_run
//! use std::collections::HashMap;
//! use islands::providers::{GitHubProvider, GitProvider};
//!
//! #[tokio::main]
//! async fn main() -> islands::providers::Result<()> {
//!     let provider = GitHubProvider::with_token("token")?
//!         .with_webhook_secret("webhook-secret");
//!
//!     let mut headers = HashMap::new();
//!     headers.insert("x-github-event".to_string(), "push".to_string());
//!     headers.insert("x-hub-signature-256".to_string(), "sha256=...".to_string());
//!
//!     let body = br#"{"repository": {...}}"#;
//!
//!     if let Some(event) = provider.parse_webhook(&headers, body).await? {
//!         println!("Event type: {}", event.event_type);
//!     }
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod base;
pub mod bitbucket;
pub mod error;
pub mod factory;
pub mod gitea;
pub mod github;
pub mod gitlab;

// Re-export main types
pub use base::{
    AuthType, BaseProvider, GitProvider, ProviderAuth, ProviderConfig, RateLimiter, Repository,
    WebhookEvent,
};
pub use bitbucket::BitbucketProvider;
pub use error::{ProviderError, Result};
/// Type alias for provider error (compatibility)
pub type Error = ProviderError;
pub use factory::{ProviderFactory, ProviderType, create_provider};
pub use gitea::GiteaProvider;
pub use github::GitHubProvider;
pub use gitlab::GitLabProvider;

/// Prelude for commonly used types.
pub mod prelude {
    pub use super::BitbucketProvider;
    pub use super::GitHubProvider;
    pub use super::GitLabProvider;
    pub use super::GiteaProvider;
    pub use super::base::{AuthType, GitProvider, ProviderAuth, ProviderConfig, Repository};
    pub use super::error::{ProviderError, Result};
    pub use super::factory::{ProviderFactory, ProviderType, create_provider};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_all_providers() {
        // GitHub
        let github = GitHubProvider::with_token("token");
        assert!(github.is_ok());
        assert_eq!(github.unwrap().provider_name(), "github");

        // GitLab
        let gitlab = GitLabProvider::with_token("token");
        assert!(gitlab.is_ok());
        assert_eq!(gitlab.unwrap().provider_name(), "gitlab");

        // Bitbucket
        let bitbucket = BitbucketProvider::with_token("token");
        assert!(bitbucket.is_ok());
        assert_eq!(bitbucket.unwrap().provider_name(), "bitbucket");

        // Gitea
        let gitea = GiteaProvider::with_base_url("https://gitea.example.com", None);
        assert!(gitea.is_ok());
        assert_eq!(gitea.unwrap().provider_name(), "gitea");
    }

    #[test]
    fn test_factory_creates_all_providers() {
        let factory = ProviderFactory::new();

        let github = factory.create("github", None);
        assert!(github.is_ok());

        let gitlab = factory.create("gitlab", None);
        assert!(gitlab.is_ok());

        let bitbucket = factory.create("bitbucket", None);
        assert!(bitbucket.is_ok());
    }

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        // These should all be available via prelude
        let _: Option<AuthType> = None;
        let _: Option<ProviderType> = None;

        fn _accept_provider<P: GitProvider>(_: P) {}
    }
}
