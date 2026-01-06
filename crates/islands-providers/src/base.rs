//! Base traits and types for git providers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::error::{ProviderError, Result};

/// Authentication type for git providers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthType {
    /// Personal access token or API token.
    Token,
    /// SSH key authentication.
    Ssh,
    /// OAuth authentication.
    OAuth,
    /// Basic username/password authentication.
    Basic,
}

/// Authentication credentials for a git provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderAuth {
    /// The type of authentication being used.
    pub auth_type: AuthType,
    /// API token (for Token or OAuth auth types).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Username (for Basic auth).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    /// Password (for Basic auth).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
    /// Path to SSH private key file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssh_key_path: Option<PathBuf>,
    /// OAuth client ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth_client_id: Option<String>,
    /// OAuth client secret.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth_client_secret: Option<String>,
}

impl ProviderAuth {
    /// Create a token-based authentication.
    pub fn token(token: impl Into<String>) -> Self {
        Self {
            auth_type: AuthType::Token,
            token: Some(token.into()),
            username: None,
            password: None,
            ssh_key_path: None,
            oauth_client_id: None,
            oauth_client_secret: None,
        }
    }

    /// Create a basic username/password authentication.
    pub fn basic(username: impl Into<String>, password: impl Into<String>) -> Self {
        Self {
            auth_type: AuthType::Basic,
            token: None,
            username: Some(username.into()),
            password: Some(password.into()),
            ssh_key_path: None,
            oauth_client_id: None,
            oauth_client_secret: None,
        }
    }

    /// Create an SSH key authentication.
    pub fn ssh(key_path: impl Into<PathBuf>) -> Self {
        Self {
            auth_type: AuthType::Ssh,
            token: None,
            username: None,
            password: None,
            ssh_key_path: Some(key_path.into()),
            oauth_client_id: None,
            oauth_client_secret: None,
        }
    }
}

/// Represents a git repository.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Repository {
    /// The provider name (github, gitlab, bitbucket, gitea).
    pub provider: String,
    /// Repository owner (user or organization).
    pub owner: String,
    /// Repository name.
    pub name: String,
    /// Full name in owner/name format.
    pub full_name: String,
    /// HTTPS clone URL.
    pub clone_url: String,
    /// SSH clone URL (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssh_url: Option<String>,
    /// Default branch name.
    #[serde(default = "default_branch")]
    pub default_branch: String,
    /// Repository description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Primary programming language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Repository size in kilobytes.
    #[serde(default)]
    pub size_kb: u64,
    /// Last updated timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_updated: Option<DateTime<Utc>>,
    /// Whether the repository is private.
    #[serde(default)]
    pub is_private: bool,
    /// Repository topics/tags.
    #[serde(default)]
    pub topics: Vec<String>,
}

fn default_branch() -> String {
    "main".to_string()
}

impl Repository {
    /// Create a new repository with minimal required information.
    pub fn new(provider: &str, owner: &str, name: &str, clone_url: &str) -> Self {
        Self {
            provider: provider.to_string(),
            owner: owner.to_string(),
            name: name.to_string(),
            full_name: format!("{}/{}", owner, name),
            clone_url: clone_url.to_string(),
            ssh_url: None,
            default_branch: "main".to_string(),
            description: None,
            language: None,
            size_kb: 0,
            last_updated: None,
            is_private: false,
            topics: Vec::new(),
        }
    }

    /// Create a repository from a URL or shorthand.
    ///
    /// Supported formats:
    /// - `https://github.com/owner/repo`
    /// - `https://github.com/owner/repo.git`
    /// - `git@github.com:owner/repo.git`
    /// - `github:owner/repo`
    /// - `owner/repo` (defaults to GitHub)
    ///
    /// Supported providers: github, gitlab, bitbucket
    pub fn from_url(url: &str) -> Result<Self> {
        let url = url.trim();
        if url.is_empty() {
            return Err(ProviderError::ConfigurationError("empty URL".to_string()));
        }

        // Handle shorthand formats
        if let Some(rest) = url.strip_prefix("github:") {
            return Self::parse_owner_repo("github", rest);
        }
        if let Some(rest) = url.strip_prefix("gitlab:") {
            return Self::parse_owner_repo("gitlab", rest);
        }
        if let Some(rest) = url.strip_prefix("bitbucket:") {
            return Self::parse_owner_repo("bitbucket", rest);
        }

        // Handle bare owner/repo format (defaults to GitHub)
        if !url.contains("://") && !url.starts_with("git@") && url.contains('/') {
            return Self::parse_owner_repo("github", url);
        }

        // Handle SSH URLs: git@github.com:owner/repo.git
        if url.starts_with("git@") {
            return Self::parse_ssh_url(url);
        }

        // Handle HTTPS URLs
        Self::parse_https_url(url)
    }

    fn parse_owner_repo(provider: &str, path: &str) -> Result<Self> {
        let path = path.trim_end_matches('/').trim_end_matches(".git");
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() < 2 || parts[0].is_empty() || parts[1].is_empty() {
            return Err(ProviderError::ConfigurationError(format!(
                "invalid repository path: {path}"
            )));
        }
        let owner = parts[0];
        let name = parts[1];
        let clone_url = match provider {
            "github" => format!("https://github.com/{owner}/{name}.git"),
            "gitlab" => format!("https://gitlab.com/{owner}/{name}.git"),
            "bitbucket" => format!("https://bitbucket.org/{owner}/{name}.git"),
            _ => format!("https://{provider}.com/{owner}/{name}.git"),
        };
        Ok(Self::new(provider, owner, name, &clone_url))
    }

    fn parse_ssh_url(url: &str) -> Result<Self> {
        // git@github.com:owner/repo.git
        let url = url.strip_prefix("git@").unwrap_or(url);
        let (host, path) = url
            .split_once(':')
            .ok_or_else(|| ProviderError::ConfigurationError(format!("invalid SSH URL: {url}")))?;

        let provider = Self::host_to_provider(host)?;
        Self::parse_owner_repo(provider, path)
    }

    fn parse_https_url(url: &str) -> Result<Self> {
        // Extract host and path from URL
        let url = url
            .strip_prefix("https://")
            .or_else(|| url.strip_prefix("http://"))
            .ok_or_else(|| {
                ProviderError::ConfigurationError(format!("invalid URL scheme: {url}"))
            })?;

        let (host, path) = url
            .split_once('/')
            .ok_or_else(|| ProviderError::ConfigurationError(format!("invalid URL: {url}")))?;

        let provider = Self::host_to_provider(host)?;
        Self::parse_owner_repo(provider, path)
    }

    fn host_to_provider(host: &str) -> Result<&'static str> {
        match host {
            "github.com" | "www.github.com" => Ok("github"),
            "gitlab.com" | "www.gitlab.com" => Ok("gitlab"),
            "bitbucket.org" | "www.bitbucket.org" => Ok("bitbucket"),
            _ => Err(ProviderError::ConfigurationError(format!(
                "unknown provider for host: {host}"
            ))),
        }
    }

    /// Get a unique identifier for this repository.
    pub fn id(&self) -> String {
        self.full_name.clone()
    }

    /// Get the local path where this repo would be cloned.
    pub fn local_path(&self) -> PathBuf {
        PathBuf::from(&self.provider)
            .join(&self.owner)
            .join(&self.name)
    }
}

/// Represents a webhook event from a git provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    /// Type of event (push, pull_request, etc.).
    pub event_type: String,
    /// The repository this event is for.
    pub repository: Repository,
    /// Git ref (branch or tag name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_name: Option<String>,
    /// Commit SHA before the push.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<String>,
    /// Commit SHA after the push.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    /// Full webhook payload.
    #[serde(default)]
    pub payload: HashMap<String, serde_json::Value>,
}

impl WebhookEvent {
    /// Check if this is a push event.
    pub fn is_push(&self) -> bool {
        self.event_type == "push"
    }
}

/// Rate limiter for API requests.
#[derive(Debug)]
pub struct RateLimiter {
    max_requests: u32,
    window_duration: Duration,
    request_count: RwLock<u32>,
    window_start: RwLock<Instant>,
}

impl RateLimiter {
    /// Create a new rate limiter.
    pub fn new(max_requests: u32, window_seconds: u64) -> Self {
        Self {
            max_requests,
            window_duration: Duration::from_secs(window_seconds),
            request_count: RwLock::new(0),
            window_start: RwLock::new(Instant::now()),
        }
    }

    /// Check rate limit and wait if necessary.
    pub async fn check_and_wait(&self) -> Result<()> {
        loop {
            let now = Instant::now();

            // Check if we need to reset the window
            {
                let window_start = *self.window_start.read().await;
                if now.duration_since(window_start) >= self.window_duration {
                    let mut count = self.request_count.write().await;
                    let mut start = self.window_start.write().await;
                    *count = 0;
                    *start = now;
                }
            }

            // Check current count
            let count = *self.request_count.read().await;
            if count < self.max_requests {
                let mut count = self.request_count.write().await;
                *count += 1;
                return Ok(());
            }

            // Calculate wait time
            let window_start = *self.window_start.read().await;
            let elapsed = now.duration_since(window_start);
            if elapsed < self.window_duration {
                let wait_time = self.window_duration - elapsed;
                tokio::time::sleep(wait_time).await;
            }
        }
    }

    /// Reset the rate limiter.
    pub async fn reset(&self) {
        let mut count = self.request_count.write().await;
        let mut start = self.window_start.write().await;
        *count = 0;
        *start = Instant::now();
    }
}

impl Clone for RateLimiter {
    fn clone(&self) -> Self {
        Self::new(self.max_requests, self.window_duration.as_secs())
    }
}

/// Configuration for a git provider.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Base URL for the API.
    pub base_url: String,
    /// Authentication credentials.
    pub auth: Option<ProviderAuth>,
    /// Webhook secret for signature verification.
    pub webhook_secret: Option<String>,
    /// Maximum requests per rate limit window.
    pub rate_limit_requests: u32,
    /// Rate limit window in seconds.
    pub rate_limit_window: u64,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            auth: None,
            webhook_secret: None,
            rate_limit_requests: 5000,
            rate_limit_window: 3600,
        }
    }
}

/// Abstract interface for git providers.
#[async_trait]
pub trait GitProvider: Send + Sync {
    /// Get the provider name.
    fn provider_name(&self) -> &'static str;

    /// Get the HTTP client.
    fn client(&self) -> &Client;

    /// Get the base URL.
    fn base_url(&self) -> &str;

    /// Get the rate limiter.
    fn rate_limiter(&self) -> &RateLimiter;

    /// Get the authentication credentials.
    fn auth(&self) -> Option<&ProviderAuth>;

    /// Build authentication headers for API requests.
    fn build_auth_headers(&self) -> reqwest::header::HeaderMap;

    /// List repositories for the authenticated user or organization.
    fn list_repositories(
        &self,
        owner: Option<&str>,
        visibility: Option<&str>,
        per_page: u32,
    ) -> BoxStream<'_, Result<Repository>>;

    /// Get a specific repository by owner and name.
    async fn get_repository(&self, owner: &str, name: &str) -> Result<Repository>;

    /// Get the default branch for a repository.
    async fn get_default_branch(&self, owner: &str, name: &str) -> Result<String>;

    /// Get the latest commit SHA for a branch or ref.
    async fn get_latest_commit(
        &self,
        owner: &str,
        name: &str,
        ref_name: Option<&str>,
    ) -> Result<String>;

    /// Parse a webhook payload and return an event.
    async fn parse_webhook(
        &self,
        headers: &HashMap<String, String>,
        body: &[u8],
    ) -> Result<Option<WebhookEvent>>;

    /// Get the appropriate clone URL based on auth type.
    fn get_clone_url(&self, repo: &Repository) -> String {
        if let Some(auth) = self.auth() {
            if auth.auth_type == AuthType::Ssh {
                if let Some(ssh_url) = &repo.ssh_url {
                    return ssh_url.clone();
                }
            }
            if let Some(token) = &auth.token {
                if repo.clone_url.contains("github") {
                    return repo
                        .clone_url
                        .replace("https://", &format!("https://{}@", token));
                }
                if repo.clone_url.contains("gitlab") {
                    return repo
                        .clone_url
                        .replace("https://", &format!("https://oauth2:{}@", token));
                }
            }
        }
        repo.clone_url.clone()
    }
}

/// Base implementation helper for git providers.
#[derive(Debug)]
pub struct BaseProvider {
    /// HTTP client with connection pooling.
    pub client: Client,
    /// Base URL for the API.
    pub base_url: String,
    /// Authentication credentials.
    pub auth: Option<ProviderAuth>,
    /// Webhook secret.
    pub webhook_secret: Option<String>,
    /// Rate limiter.
    pub rate_limiter: Arc<RateLimiter>,
}

impl BaseProvider {
    /// Create a new base provider.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .build()
            .map_err(ProviderError::HttpError)?;

        Ok(Self {
            client,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            auth: config.auth,
            webhook_secret: config.webhook_secret,
            rate_limiter: Arc::new(RateLimiter::new(
                config.rate_limit_requests,
                config.rate_limit_window,
            )),
        })
    }

    /// Make an API request with rate limiting.
    pub async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        headers: reqwest::header::HeaderMap,
    ) -> Result<reqwest::RequestBuilder> {
        self.rate_limiter.check_and_wait().await?;

        let url = format!("{}{}", self.base_url, path);
        Ok(self.client.request(method, &url).headers(headers))
    }

    /// Make a GET request with query parameters.
    pub async fn get(
        &self,
        path: &str,
        headers: reqwest::header::HeaderMap,
        params: &[(&str, &str)],
    ) -> Result<reqwest::Response> {
        let request = self.request(reqwest::Method::GET, path, headers).await?;
        let response = request.query(params).send().await?;
        Self::check_response(response).await
    }

    /// Check response status and return error if not successful.
    pub async fn check_response(response: reqwest::Response) -> Result<reqwest::Response> {
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(ProviderError::NotFound("Resource not found".to_string()));
        }

        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(ProviderError::AuthenticationError(
                "Authentication failed".to_string(),
            ));
        }

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse().ok())
                .unwrap_or(60);
            return Err(ProviderError::RateLimitExceeded {
                retry_after_secs: retry_after,
            });
        }

        let message = response.text().await.unwrap_or_default();
        Err(ProviderError::ApiError {
            status: status.as_u16(),
            message,
        })
    }
}

impl Clone for BaseProvider {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            auth: self.auth.clone(),
            webhook_secret: self.webhook_secret.clone(),
            rate_limiter: Arc::new((*self.rate_limiter).clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_type_serde() {
        let token = AuthType::Token;
        let json = serde_json::to_string(&token).unwrap();
        assert_eq!(json, "\"token\"");

        let parsed: AuthType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, AuthType::Token);
    }

    #[test]
    fn test_provider_auth_token() {
        let auth = ProviderAuth::token("my-token");
        assert_eq!(auth.auth_type, AuthType::Token);
        assert_eq!(auth.token, Some("my-token".to_string()));
        assert!(auth.username.is_none());
    }

    #[test]
    fn test_provider_auth_basic() {
        let auth = ProviderAuth::basic("user", "pass");
        assert_eq!(auth.auth_type, AuthType::Basic);
        assert_eq!(auth.username, Some("user".to_string()));
        assert_eq!(auth.password, Some("pass".to_string()));
    }

    #[test]
    fn test_provider_auth_ssh() {
        let auth = ProviderAuth::ssh("/path/to/key");
        assert_eq!(auth.auth_type, AuthType::Ssh);
        assert_eq!(auth.ssh_key_path, Some(PathBuf::from("/path/to/key")));
    }

    #[test]
    fn test_repository_local_path() {
        let repo = Repository {
            provider: "github".to_string(),
            owner: "rust-lang".to_string(),
            name: "rust".to_string(),
            full_name: "rust-lang/rust".to_string(),
            clone_url: "https://github.com/rust-lang/rust.git".to_string(),
            ssh_url: None,
            default_branch: "master".to_string(),
            description: None,
            language: Some("Rust".to_string()),
            size_kb: 1000,
            last_updated: None,
            is_private: false,
            topics: vec![],
        };

        assert_eq!(repo.local_path(), PathBuf::from("github/rust-lang/rust"));
    }

    #[test]
    fn test_repository_serde() {
        let repo = Repository {
            provider: "github".to_string(),
            owner: "test".to_string(),
            name: "repo".to_string(),
            full_name: "test/repo".to_string(),
            clone_url: "https://github.com/test/repo.git".to_string(),
            ssh_url: Some("git@github.com:test/repo.git".to_string()),
            default_branch: "main".to_string(),
            description: Some("A test repo".to_string()),
            language: None,
            size_kb: 100,
            last_updated: None,
            is_private: true,
            topics: vec!["rust".to_string(), "test".to_string()],
        };

        let json = serde_json::to_string(&repo).unwrap();
        let parsed: Repository = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, repo);
    }

    #[test]
    fn test_webhook_event_creation() {
        let repo = Repository {
            provider: "github".to_string(),
            owner: "test".to_string(),
            name: "repo".to_string(),
            full_name: "test/repo".to_string(),
            clone_url: "https://github.com/test/repo.git".to_string(),
            ssh_url: None,
            default_branch: "main".to_string(),
            description: None,
            language: None,
            size_kb: 0,
            last_updated: None,
            is_private: false,
            topics: vec![],
        };

        let event = WebhookEvent {
            event_type: "push".to_string(),
            repository: repo,
            ref_name: Some("refs/heads/main".to_string()),
            before: Some("abc123".to_string()),
            after: Some("def456".to_string()),
            payload: HashMap::new(),
        };

        assert_eq!(event.event_type, "push");
        assert_eq!(event.ref_name, Some("refs/heads/main".to_string()));
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(5, 1);

        // Should allow 5 requests immediately
        for _ in 0..5 {
            limiter.check_and_wait().await.unwrap();
        }

        // The next request should wait (in a real test we'd verify the timing)
        // For now, just reset and verify it works again
        limiter.reset().await;
        limiter.check_and_wait().await.unwrap();
    }

    #[test]
    fn test_provider_config_default() {
        let config = ProviderConfig::default();
        assert_eq!(config.rate_limit_requests, 5000);
        assert_eq!(config.rate_limit_window, 3600);
        assert!(config.auth.is_none());
    }

    #[test]
    fn test_base_provider_creation() {
        let config = ProviderConfig {
            base_url: "https://api.github.com/".to_string(),
            ..Default::default()
        };

        let provider = BaseProvider::new(config).unwrap();
        assert_eq!(provider.base_url, "https://api.github.com");
    }

    #[test]
    fn test_repository_new() {
        let repo = Repository::new(
            "github",
            "owner",
            "repo",
            "https://github.com/owner/repo.git",
        );

        assert_eq!(repo.provider, "github");
        assert_eq!(repo.owner, "owner");
        assert_eq!(repo.name, "repo");
        assert_eq!(repo.full_name, "owner/repo");
        assert_eq!(repo.clone_url, "https://github.com/owner/repo.git");
        assert_eq!(repo.default_branch, "main");
        assert!(!repo.is_private);
        assert!(repo.topics.is_empty());
        assert!(repo.ssh_url.is_none());
        assert!(repo.description.is_none());
    }

    #[test]
    fn test_repository_id() {
        let repo = Repository::new(
            "gitlab",
            "myorg",
            "myrepo",
            "https://gitlab.com/myorg/myrepo.git",
        );

        assert_eq!(repo.id(), "myorg/myrepo");
    }

    #[test]
    fn test_webhook_event_is_push() {
        let repo = Repository::new("github", "test", "repo", "https://github.com/test/repo.git");

        let push_event = WebhookEvent {
            event_type: "push".to_string(),
            repository: repo.clone(),
            ref_name: None,
            before: None,
            after: None,
            payload: HashMap::new(),
        };

        let pr_event = WebhookEvent {
            event_type: "pull_request".to_string(),
            repository: repo,
            ref_name: None,
            before: None,
            after: None,
            payload: HashMap::new(),
        };

        assert!(push_event.is_push());
        assert!(!pr_event.is_push());
    }

    #[test]
    fn test_rate_limiter_clone() {
        let limiter = RateLimiter::new(100, 60);
        let cloned = limiter.clone();

        assert_eq!(cloned.max_requests, 100);
        assert_eq!(cloned.window_duration, Duration::from_secs(60));
    }

    #[test]
    fn test_default_branch_function() {
        assert_eq!(default_branch(), "main");
    }

    #[test]
    fn test_provider_auth_token_with_string() {
        let auth = ProviderAuth::token(String::from("my-string-token"));
        assert_eq!(auth.token, Some("my-string-token".to_string()));
    }

    #[test]
    fn test_provider_auth_basic_with_strings() {
        let auth = ProviderAuth::basic(String::from("user"), String::from("pass"));
        assert_eq!(auth.username, Some("user".to_string()));
        assert_eq!(auth.password, Some("pass".to_string()));
    }

    #[test]
    fn test_provider_auth_ssh_with_path_buf() {
        let auth = ProviderAuth::ssh(PathBuf::from("/home/user/.ssh/id_rsa"));
        assert_eq!(
            auth.ssh_key_path,
            Some(PathBuf::from("/home/user/.ssh/id_rsa"))
        );
    }

    #[test]
    fn test_base_provider_clone() {
        let config = ProviderConfig {
            base_url: "https://api.test.com".to_string(),
            auth: Some(ProviderAuth::token("test-token")),
            webhook_secret: Some("secret".to_string()),
            ..Default::default()
        };

        let provider = BaseProvider::new(config).unwrap();
        let cloned = provider.clone();

        assert_eq!(cloned.base_url, provider.base_url);
        assert_eq!(
            cloned.auth.as_ref().unwrap().token,
            provider.auth.as_ref().unwrap().token
        );
        assert_eq!(cloned.webhook_secret, provider.webhook_secret);
    }

    #[test]
    fn test_provider_config_with_auth() {
        let config = ProviderConfig {
            base_url: "https://api.github.com".to_string(),
            auth: Some(ProviderAuth::token("gh_token")),
            webhook_secret: Some("webhook_secret".to_string()),
            rate_limit_requests: 1000,
            rate_limit_window: 1800,
        };

        assert_eq!(config.rate_limit_requests, 1000);
        assert_eq!(config.rate_limit_window, 1800);
        assert!(config.auth.is_some());
        assert!(config.webhook_secret.is_some());
    }

    #[test]
    fn test_repository_serialization_without_optional_fields() {
        let repo = Repository::new(
            "bitbucket",
            "team",
            "project",
            "https://bitbucket.org/team/project.git",
        );

        let json = serde_json::to_string(&repo).unwrap();
        assert!(!json.contains("ssh_url")); // skipped when None
        assert!(!json.contains("description")); // skipped when None
        assert!(!json.contains("language")); // skipped when None
        assert!(json.contains("clone_url"));
        assert!(json.contains("full_name"));
    }

    #[test]
    fn test_auth_type_all_variants() {
        let variants = vec![
            (AuthType::Token, "\"token\""),
            (AuthType::Ssh, "\"ssh\""),
            (AuthType::OAuth, "\"o_auth\""),
            (AuthType::Basic, "\"basic\""),
        ];

        for (auth_type, expected) in variants {
            let json = serde_json::to_string(&auth_type).unwrap();
            assert_eq!(json, expected);

            let parsed: AuthType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, auth_type);
        }
    }

    #[test]
    fn test_repository_with_all_fields() {
        let repo = Repository {
            provider: "github".to_string(),
            owner: "owner".to_string(),
            name: "name".to_string(),
            full_name: "owner/name".to_string(),
            clone_url: "https://github.com/owner/name.git".to_string(),
            ssh_url: Some("git@github.com:owner/name.git".to_string()),
            default_branch: "develop".to_string(),
            description: Some("A great project".to_string()),
            language: Some("Rust".to_string()),
            size_kb: 5000,
            last_updated: Some(Utc::now()),
            is_private: true,
            topics: vec!["rust".to_string(), "cli".to_string()],
        };

        let json = serde_json::to_string(&repo).unwrap();
        let parsed: Repository = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.ssh_url, repo.ssh_url);
        assert_eq!(parsed.description, repo.description);
        assert_eq!(parsed.language, repo.language);
        assert_eq!(parsed.is_private, repo.is_private);
        assert_eq!(parsed.topics, repo.topics);
    }

    #[test]
    fn test_webhook_event_with_payload() {
        let repo = Repository::new("github", "test", "repo", "https://github.com/test/repo.git");

        let mut payload = HashMap::new();
        payload.insert("action".to_string(), serde_json::json!("opened"));
        payload.insert("number".to_string(), serde_json::json!(42));

        let event = WebhookEvent {
            event_type: "pull_request".to_string(),
            repository: repo,
            ref_name: Some("refs/heads/feature".to_string()),
            before: Some("aaa".to_string()),
            after: Some("bbb".to_string()),
            payload,
        };

        assert!(!event.is_push());
        assert_eq!(event.payload.len(), 2);
        assert_eq!(event.payload.get("action").unwrap(), "opened");
    }

    #[test]
    fn test_from_url_github_https() {
        let repo = Repository::from_url("https://github.com/tokio-rs/tokio").unwrap();
        assert_eq!(repo.provider, "github");
        assert_eq!(repo.owner, "tokio-rs");
        assert_eq!(repo.name, "tokio");
        assert_eq!(repo.clone_url, "https://github.com/tokio-rs/tokio.git");
    }

    #[test]
    fn test_from_url_github_https_with_git_suffix() {
        let repo = Repository::from_url("https://github.com/tokio-rs/axum.git").unwrap();
        assert_eq!(repo.provider, "github");
        assert_eq!(repo.owner, "tokio-rs");
        assert_eq!(repo.name, "axum");
    }

    #[test]
    fn test_from_url_github_ssh() {
        let repo = Repository::from_url("git@github.com:rust-lang/rust.git").unwrap();
        assert_eq!(repo.provider, "github");
        assert_eq!(repo.owner, "rust-lang");
        assert_eq!(repo.name, "rust");
    }

    #[test]
    fn test_from_url_gitlab_https() {
        let repo = Repository::from_url("https://gitlab.com/inkscape/inkscape").unwrap();
        assert_eq!(repo.provider, "gitlab");
        assert_eq!(repo.owner, "inkscape");
        assert_eq!(repo.name, "inkscape");
    }

    #[test]
    fn test_from_url_bitbucket_https() {
        let repo =
            Repository::from_url("https://bitbucket.org/atlassian/python-bitbucket").unwrap();
        assert_eq!(repo.provider, "bitbucket");
        assert_eq!(repo.owner, "atlassian");
        assert_eq!(repo.name, "python-bitbucket");
    }

    #[test]
    fn test_from_url_with_trailing_slash() {
        let repo = Repository::from_url("https://github.com/tokio-rs/tokio/").unwrap();
        assert_eq!(repo.owner, "tokio-rs");
        assert_eq!(repo.name, "tokio");
    }

    #[test]
    fn test_from_url_invalid_no_path() {
        let result = Repository::from_url("https://github.com/");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_url_invalid_only_owner() {
        let result = Repository::from_url("https://github.com/tokio-rs");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_url_invalid_empty() {
        let result = Repository::from_url("");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_url_shorthand_github() {
        let repo = Repository::from_url("github:tokio-rs/tokio").unwrap();
        assert_eq!(repo.provider, "github");
        assert_eq!(repo.owner, "tokio-rs");
        assert_eq!(repo.name, "tokio");
    }

    #[test]
    fn test_from_url_shorthand_bare() {
        let repo = Repository::from_url("tokio-rs/tokio").unwrap();
        assert_eq!(repo.provider, "github");
        assert_eq!(repo.owner, "tokio-rs");
        assert_eq!(repo.name, "tokio");
    }
}
