//! Git provider trait definition
//!
//! This module defines the core `GitProvider` trait that all provider
//! implementations must satisfy.

use std::future::Future;
use std::pin::Pin;

use crate::auth::{AuthType, ProviderAuth};
use crate::error::Result;
use crate::repository::{Repository, Visibility};
use crate::webhook::WebhookEvent;

/// A boxed future for async trait methods
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Stream of repositories
pub type RepositoryStream<'a> = Pin<Box<dyn futures::Stream<Item = Result<Repository>> + Send + 'a>>;

/// Abstract trait for git hosting providers
///
/// This trait defines the interface that all git provider implementations
/// (GitHub, GitLab, Bitbucket, Gitea) must implement.
pub trait GitProvider: Send + Sync {
    /// Get the provider name (e.g., "github", "gitlab")
    fn name(&self) -> &'static str;

    /// Get the base API URL
    fn base_url(&self) -> &str;

    /// List repositories for the authenticated user or organization
    fn list_repositories(
        &self,
        owner: Option<&str>,
        visibility: Visibility,
        per_page: usize,
    ) -> RepositoryStream<'_>;

    /// Get a specific repository by owner and name
    fn get_repository(&self, owner: &str, name: &str) -> BoxFuture<'_, Result<Repository>>;

    /// Get the default branch for a repository
    fn get_default_branch(&self, owner: &str, name: &str) -> BoxFuture<'_, Result<String>>;

    /// Get the latest commit SHA for a branch or ref
    fn get_latest_commit(
        &self,
        owner: &str,
        name: &str,
        ref_name: Option<&str>,
    ) -> BoxFuture<'_, Result<String>>;

    /// Parse a webhook payload and return an event
    fn parse_webhook(
        &self,
        headers: &http::HeaderMap,
        body: &[u8],
    ) -> Result<Option<WebhookEvent>>;

    /// Get the clone URL for a repository based on auth type
    fn get_clone_url(&self, repo: &Repository, auth: Option<&ProviderAuth>) -> String {
        if let Some(auth) = auth {
            if auth.auth_type == AuthType::Ssh {
                if let Some(ssh_url) = &repo.ssh_url {
                    return ssh_url.clone();
                }
            }
            if let Some(token) = &auth.token {
                // Insert token into clone URL
                return self.inject_token_into_url(&repo.clone_url, token);
            }
        }
        repo.clone_url.clone()
    }

    /// Inject an auth token into a clone URL
    fn inject_token_into_url(&self, url: &str, token: &str) -> String {
        if url.starts_with("https://") {
            url.replacen("https://", &format!("https://{token}@"), 1)
        } else {
            url.to_string()
        }
    }
}

/// Configuration for rate limiting
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per window
    pub max_requests: u32,
    /// Window duration in seconds
    pub window_seconds: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 5000,
            window_seconds: 3600,
        }
    }
}

/// Base configuration shared by all providers
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Base API URL
    pub base_url: String,
    /// Authentication credentials
    pub auth: Option<ProviderAuth>,
    /// Rate limit configuration
    pub rate_limit: RateLimitConfig,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Webhook secret for signature verification
    pub webhook_secret: Option<String>,
}

impl ProviderConfig {
    /// Create a new provider config with the given base URL
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            auth: None,
            rate_limit: RateLimitConfig::default(),
            timeout_seconds: 30,
            webhook_secret: None,
        }
    }

    /// Set authentication
    #[must_use]
    pub fn with_auth(mut self, auth: ProviderAuth) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Set webhook secret
    #[must_use]
    pub fn with_webhook_secret(mut self, secret: impl Into<String>) -> Self {
        self.webhook_secret = Some(secret.into());
        self
    }
}
