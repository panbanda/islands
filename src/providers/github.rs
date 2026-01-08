//! GitHub API provider implementation.

use std::collections::HashMap;

use async_trait::async_trait;
use base64::Engine;
use chrono::{DateTime, Utc};
use futures::stream::{self, BoxStream, StreamExt};
use hmac::{Hmac, Mac};
use reqwest::Client;
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderValue};
use serde::Deserialize;
use sha2::Sha256;

use super::base::{
    AuthType, BaseProvider, GitProvider, ProviderAuth, ProviderConfig, RateLimiter, Repository,
    WebhookEvent,
};
use super::error::{ProviderError, Result};

type HmacSha256 = Hmac<Sha256>;

/// GitHub API response for a repository.
#[derive(Debug, Deserialize)]
struct GitHubRepo {
    name: String,
    full_name: String,
    clone_url: String,
    ssh_url: Option<String>,
    default_branch: Option<String>,
    description: Option<String>,
    language: Option<String>,
    size: Option<u64>,
    updated_at: Option<String>,
    private: Option<bool>,
    topics: Option<Vec<String>>,
    owner: GitHubOwner,
}

#[derive(Debug, Deserialize)]
struct GitHubOwner {
    login: String,
}

#[derive(Debug, Deserialize)]
struct GitHubCommit {
    sha: String,
}

/// GitHub API provider implementation.
#[derive(Debug, Clone)]
pub struct GitHubProvider {
    base: BaseProvider,
    webhook_secret: Option<String>,
}

impl GitHubProvider {
    /// Default GitHub API URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.github.com";

    /// Create a new GitHub provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let webhook_secret = config.webhook_secret.clone();
        let base = BaseProvider::new(config)?;
        Ok(Self {
            base,
            webhook_secret,
        })
    }

    /// Create a new GitHub provider with default settings and a token.
    pub fn with_token(token: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig {
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            auth: Some(ProviderAuth::token(token)),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a new GitHub provider with a custom base URL (for GitHub Enterprise).
    pub fn with_base_url(base_url: impl Into<String>, token: Option<String>) -> Result<Self> {
        let config = ProviderConfig {
            base_url: base_url.into(),
            auth: token.map(ProviderAuth::token),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Set the webhook secret for signature verification.
    pub fn with_webhook_secret(mut self, secret: impl Into<String>) -> Self {
        self.webhook_secret = Some(secret.into());
        self
    }

    fn parse_repository(&self, data: GitHubRepo) -> Repository {
        let updated_at = data.updated_at.as_ref().and_then(|s| {
            DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        });

        Repository {
            provider: self.provider_name().to_string(),
            owner: data.owner.login,
            name: data.name,
            full_name: data.full_name,
            clone_url: data.clone_url,
            ssh_url: data.ssh_url,
            default_branch: data.default_branch.unwrap_or_else(|| "main".to_string()),
            description: data.description,
            language: data.language,
            size_kb: data.size.unwrap_or(0),
            last_updated: updated_at,
            is_private: data.private.unwrap_or(false),
            topics: data.topics.unwrap_or_default(),
        }
    }

    fn verify_webhook_signature(&self, body: &[u8], signature: &str) -> Result<()> {
        let secret = self.webhook_secret.as_ref().ok_or_else(|| {
            ProviderError::ConfigurationError("Webhook secret not set".to_string())
        })?;

        let expected_prefix = "sha256=";
        if !signature.starts_with(expected_prefix) {
            return Err(ProviderError::InvalidWebhookSignature);
        }

        let mut mac =
            HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can accept any key length");
        mac.update(body);
        let expected = hex::encode(mac.finalize().into_bytes());
        let actual = &signature[expected_prefix.len()..];

        if !constant_time_compare(expected.as_bytes(), actual.as_bytes()) {
            return Err(ProviderError::InvalidWebhookSignature);
        }

        Ok(())
    }
}

/// Constant-time comparison to prevent timing attacks.
fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

#[async_trait]
impl GitProvider for GitHubProvider {
    fn provider_name(&self) -> &'static str {
        "github"
    }

    fn client(&self) -> &Client {
        &self.base.client
    }

    fn base_url(&self) -> &str {
        &self.base.base_url
    }

    fn rate_limiter(&self) -> &RateLimiter {
        &self.base.rate_limiter
    }

    fn auth(&self) -> Option<&ProviderAuth> {
        self.base.auth.as_ref()
    }

    fn build_auth_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/vnd.github+json"),
        );
        headers.insert(
            "X-GitHub-Api-Version",
            HeaderValue::from_static("2022-11-28"),
        );

        if let Some(auth) = &self.base.auth {
            match auth.auth_type {
                AuthType::Token => {
                    if let Some(token) = &auth.token {
                        if let Ok(value) = HeaderValue::from_str(&format!("Bearer {}", token)) {
                            headers.insert(AUTHORIZATION, value);
                        }
                    }
                }
                AuthType::Basic => {
                    if let (Some(username), Some(password)) = (&auth.username, &auth.password) {
                        let credentials = format!("{}:{}", username, password);
                        let encoded = base64::engine::general_purpose::STANDARD
                            .encode(credentials.as_bytes());
                        if let Ok(value) = HeaderValue::from_str(&format!("Basic {}", encoded)) {
                            headers.insert(AUTHORIZATION, value);
                        }
                    }
                }
                _ => {}
            }
        }

        headers
    }

    fn list_repositories(
        &self,
        owner: Option<&str>,
        visibility: Option<&str>,
        per_page: u32,
    ) -> BoxStream<'_, Result<Repository>> {
        let owner = owner.map(|s| s.to_string());
        let visibility = visibility.map(|s| s.to_string());

        stream::unfold((1u32, false), move |(page, done)| {
            let owner = owner.clone();
            let visibility = visibility.clone();

            async move {
                if done {
                    return None;
                }

                let path = match &owner {
                    Some(o) => format!("/orgs/{}/repos", o),
                    None => "/user/repos".to_string(),
                };

                let mut params = vec![
                    ("per_page", per_page.to_string()),
                    ("page", page.to_string()),
                ];

                if let Some(vis) = &visibility {
                    params.push(("visibility", vis.clone()));
                }

                let params: Vec<(&str, &str)> =
                    params.iter().map(|(k, v)| (*k, v.as_str())).collect();

                let headers = self.build_auth_headers();

                match self.base.get(&path, headers, &params).await {
                    Ok(response) => match response.json::<Vec<GitHubRepo>>().await {
                        Ok(repos) => {
                            let is_empty = repos.is_empty();
                            let is_last = repos.len() < per_page as usize;

                            let repositories: Vec<Result<Repository>> = repos
                                .into_iter()
                                .map(|r| Ok(self.parse_repository(r)))
                                .collect();

                            if is_empty {
                                None
                            } else {
                                Some((stream::iter(repositories), (page + 1, is_last)))
                            }
                        }
                        Err(e) => Some((
                            stream::iter(vec![Err(ProviderError::HttpError(e))]),
                            (page, true),
                        )),
                    },
                    Err(e) => Some((stream::iter(vec![Err(e)]), (page, true))),
                }
            }
        })
        .flatten()
        .boxed()
    }

    async fn get_repository(&self, owner: &str, name: &str) -> Result<Repository> {
        let path = format!("/repos/{}/{}", owner, name);
        let headers = self.build_auth_headers();

        let response = self.base.get(&path, headers, &[]).await?;
        let repo: GitHubRepo = response.json().await?;
        Ok(self.parse_repository(repo))
    }

    async fn get_default_branch(&self, owner: &str, name: &str) -> Result<String> {
        let repo = self.get_repository(owner, name).await?;
        Ok(repo.default_branch)
    }

    async fn get_latest_commit(
        &self,
        owner: &str,
        name: &str,
        ref_name: Option<&str>,
    ) -> Result<String> {
        let ref_name = match ref_name {
            Some(r) => r.to_string(),
            None => self.get_default_branch(owner, name).await?,
        };

        let path = format!("/repos/{}/{}/commits/{}", owner, name, ref_name);
        let headers = self.build_auth_headers();

        let response = self.base.get(&path, headers, &[]).await?;
        let commit: GitHubCommit = response.json().await?;
        Ok(commit.sha)
    }

    async fn parse_webhook(
        &self,
        headers: &HashMap<String, String>,
        body: &[u8],
    ) -> Result<Option<WebhookEvent>> {
        let event_type = match headers.get("x-github-event") {
            Some(t) => t.clone(),
            None => return Ok(None),
        };

        // Verify signature if webhook secret is set
        if self.webhook_secret.is_some() {
            let signature = headers
                .get("x-hub-signature-256")
                .ok_or(ProviderError::InvalidWebhookSignature)?;
            self.verify_webhook_signature(body, signature)?;
        }

        let payload: serde_json::Value = serde_json::from_slice(body)?;

        let repo_data = match payload.get("repository") {
            Some(r) => r,
            None => return Ok(None),
        };

        let owner = repo_data
            .get("owner")
            .and_then(|o| o.get("login"))
            .and_then(|l| l.as_str())
            .unwrap_or("")
            .to_string();

        let repo = Repository {
            provider: self.provider_name().to_string(),
            owner,
            name: repo_data
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            full_name: repo_data
                .get("full_name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            clone_url: repo_data
                .get("clone_url")
                .and_then(|u| u.as_str())
                .unwrap_or("")
                .to_string(),
            ssh_url: repo_data
                .get("ssh_url")
                .and_then(|u| u.as_str())
                .map(|s| s.to_string()),
            default_branch: repo_data
                .get("default_branch")
                .and_then(|b| b.as_str())
                .unwrap_or("main")
                .to_string(),
            description: None,
            language: None,
            size_kb: 0,
            last_updated: None,
            is_private: false,
            topics: vec![],
        };

        let payload_map: HashMap<String, serde_json::Value> =
            serde_json::from_value(payload.clone()).unwrap_or_default();

        Ok(Some(WebhookEvent {
            event_type,
            repository: repo,
            ref_name: payload
                .get("ref")
                .and_then(|r| r.as_str())
                .map(|s| s.to_string()),
            before: payload
                .get("before")
                .and_then(|b| b.as_str())
                .map(|s| s.to_string()),
            after: payload
                .get("after")
                .and_then(|a| a.as_str())
                .map(|s| s.to_string()),
            payload: payload_map,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let provider = GitHubProvider::with_token("test-token").unwrap();
        assert_eq!(provider.provider_name(), "github");
    }

    #[test]
    fn test_base_url() {
        let provider = GitHubProvider::with_token("test-token").unwrap();
        assert_eq!(provider.base_url(), "https://api.github.com");
    }

    #[test]
    fn test_custom_base_url() {
        let provider = GitHubProvider::with_base_url(
            "https://github.example.com/api/v3",
            Some("token".to_string()),
        )
        .unwrap();
        assert_eq!(provider.base_url(), "https://github.example.com/api/v3");
    }

    #[test]
    fn test_auth_headers_with_token() {
        let provider = GitHubProvider::with_token("my-token").unwrap();
        let headers = provider.build_auth_headers();

        assert!(headers.contains_key(AUTHORIZATION));
        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer my-token");
        assert_eq!(headers.get(ACCEPT).unwrap(), "application/vnd.github+json");
    }

    #[test]
    fn test_auth_headers_with_basic() {
        let config = ProviderConfig {
            base_url: GitHubProvider::DEFAULT_BASE_URL.to_string(),
            auth: Some(ProviderAuth::basic("user", "pass")),
            ..Default::default()
        };
        let provider = GitHubProvider::new(config).unwrap();
        let headers = provider.build_auth_headers();

        assert!(headers.contains_key(AUTHORIZATION));
        let auth_value = headers.get(AUTHORIZATION).unwrap().to_str().unwrap();
        assert!(auth_value.starts_with("Basic "));
    }

    #[test]
    fn test_auth_headers_without_auth() {
        let config = ProviderConfig {
            base_url: GitHubProvider::DEFAULT_BASE_URL.to_string(),
            auth: None,
            ..Default::default()
        };
        let provider = GitHubProvider::new(config).unwrap();
        let headers = provider.build_auth_headers();

        assert!(!headers.contains_key(AUTHORIZATION));
        assert!(headers.contains_key(ACCEPT));
    }

    #[test]
    fn test_parse_repository() {
        let provider = GitHubProvider::with_token("test").unwrap();

        let github_repo = GitHubRepo {
            name: "test-repo".to_string(),
            full_name: "owner/test-repo".to_string(),
            clone_url: "https://github.com/owner/test-repo.git".to_string(),
            ssh_url: Some("git@github.com:owner/test-repo.git".to_string()),
            default_branch: Some("main".to_string()),
            description: Some("A test repository".to_string()),
            language: Some("Rust".to_string()),
            size: Some(1024),
            updated_at: Some("2024-01-15T10:30:00Z".to_string()),
            private: Some(true),
            topics: Some(vec!["rust".to_string(), "testing".to_string()]),
            owner: GitHubOwner {
                login: "owner".to_string(),
            },
        };

        let repo = provider.parse_repository(github_repo);

        assert_eq!(repo.provider, "github");
        assert_eq!(repo.owner, "owner");
        assert_eq!(repo.name, "test-repo");
        assert_eq!(repo.full_name, "owner/test-repo");
        assert_eq!(repo.clone_url, "https://github.com/owner/test-repo.git");
        assert_eq!(
            repo.ssh_url,
            Some("git@github.com:owner/test-repo.git".to_string())
        );
        assert_eq!(repo.default_branch, "main");
        assert_eq!(repo.description, Some("A test repository".to_string()));
        assert_eq!(repo.language, Some("Rust".to_string()));
        assert_eq!(repo.size_kb, 1024);
        assert!(repo.last_updated.is_some());
        assert!(repo.is_private);
        assert_eq!(repo.topics, vec!["rust".to_string(), "testing".to_string()]);
    }

    #[test]
    fn test_webhook_signature_verification() {
        let provider = GitHubProvider::with_token("test")
            .unwrap()
            .with_webhook_secret("mysecret");

        let body = b"test payload";

        // Generate a valid signature
        let mut mac = HmacSha256::new_from_slice(b"mysecret").unwrap();
        mac.update(body);
        let signature = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));

        assert!(provider.verify_webhook_signature(body, &signature).is_ok());
    }

    #[test]
    fn test_webhook_signature_verification_invalid() {
        let provider = GitHubProvider::with_token("test")
            .unwrap()
            .with_webhook_secret("mysecret");

        let body = b"test payload";
        let signature = "sha256=invalid_signature";

        assert!(matches!(
            provider.verify_webhook_signature(body, signature),
            Err(ProviderError::InvalidWebhookSignature)
        ));
    }

    #[test]
    fn test_webhook_signature_without_prefix() {
        let provider = GitHubProvider::with_token("test")
            .unwrap()
            .with_webhook_secret("mysecret");

        let body = b"test payload";
        let signature = "no_prefix_signature";

        assert!(matches!(
            provider.verify_webhook_signature(body, signature),
            Err(ProviderError::InvalidWebhookSignature)
        ));
    }

    #[test]
    fn test_constant_time_compare() {
        assert!(constant_time_compare(b"hello", b"hello"));
        assert!(!constant_time_compare(b"hello", b"world"));
        assert!(!constant_time_compare(b"hello", b"helloworld"));
        assert!(!constant_time_compare(b"", b"a"));
        assert!(constant_time_compare(b"", b""));
    }

    #[tokio::test]
    async fn test_parse_webhook_push_event() {
        let provider = GitHubProvider::with_token("test").unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-github-event".to_string(), "push".to_string());

        let payload = serde_json::json!({
            "ref": "refs/heads/main",
            "before": "abc123",
            "after": "def456",
            "repository": {
                "name": "test-repo",
                "full_name": "owner/test-repo",
                "clone_url": "https://github.com/owner/test-repo.git",
                "ssh_url": "git@github.com:owner/test-repo.git",
                "default_branch": "main",
                "owner": {
                    "login": "owner"
                }
            }
        });

        let body = serde_json::to_vec(&payload).unwrap();
        let result = provider.parse_webhook(&headers, &body).await.unwrap();

        assert!(result.is_some());
        let event = result.unwrap();
        assert_eq!(event.event_type, "push");
        assert_eq!(event.repository.name, "test-repo");
        assert_eq!(event.repository.owner, "owner");
        assert_eq!(event.ref_name, Some("refs/heads/main".to_string()));
        assert_eq!(event.before, Some("abc123".to_string()));
        assert_eq!(event.after, Some("def456".to_string()));
    }

    #[tokio::test]
    async fn test_parse_webhook_no_event_type() {
        let provider = GitHubProvider::with_token("test").unwrap();
        let headers = HashMap::new();
        let body = b"{}";

        let result = provider.parse_webhook(&headers, body).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_parse_webhook_no_repository() {
        let provider = GitHubProvider::with_token("test").unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-github-event".to_string(), "ping".to_string());

        let body = b"{}";
        let result = provider.parse_webhook(&headers, body).await.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_clone_url_with_token() {
        let provider = GitHubProvider::with_token("my-token").unwrap();

        let repo = Repository {
            provider: "github".to_string(),
            owner: "test".to_string(),
            name: "repo".to_string(),
            full_name: "test/repo".to_string(),
            clone_url: "https://github.com/test/repo.git".to_string(),
            ssh_url: Some("git@github.com:test/repo.git".to_string()),
            default_branch: "main".to_string(),
            description: None,
            language: None,
            size_kb: 0,
            last_updated: None,
            is_private: false,
            topics: vec![],
        };

        let url = provider.get_clone_url(&repo);
        assert_eq!(url, "https://my-token@github.com/test/repo.git");
    }

    #[test]
    fn test_get_clone_url_with_ssh() {
        let config = ProviderConfig {
            base_url: GitHubProvider::DEFAULT_BASE_URL.to_string(),
            auth: Some(ProviderAuth::ssh("/path/to/key")),
            ..Default::default()
        };
        let provider = GitHubProvider::new(config).unwrap();

        let repo = Repository {
            provider: "github".to_string(),
            owner: "test".to_string(),
            name: "repo".to_string(),
            full_name: "test/repo".to_string(),
            clone_url: "https://github.com/test/repo.git".to_string(),
            ssh_url: Some("git@github.com:test/repo.git".to_string()),
            default_branch: "main".to_string(),
            description: None,
            language: None,
            size_kb: 0,
            last_updated: None,
            is_private: false,
            topics: vec![],
        };

        let url = provider.get_clone_url(&repo);
        assert_eq!(url, "git@github.com:test/repo.git");
    }

    #[test]
    fn test_get_clone_url_without_auth() {
        let config = ProviderConfig {
            base_url: GitHubProvider::DEFAULT_BASE_URL.to_string(),
            auth: None,
            ..Default::default()
        };
        let provider = GitHubProvider::new(config).unwrap();

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

        let url = provider.get_clone_url(&repo);
        assert_eq!(url, "https://github.com/test/repo.git");
    }

    #[test]
    fn test_parse_repository_minimal() {
        let provider = GitHubProvider::with_token("test").unwrap();

        let github_repo = GitHubRepo {
            name: "repo".to_string(),
            full_name: "owner/repo".to_string(),
            clone_url: "https://github.com/owner/repo.git".to_string(),
            ssh_url: None,
            default_branch: None,
            description: None,
            language: None,
            size: None,
            updated_at: None,
            private: None,
            topics: None,
            owner: GitHubOwner {
                login: "owner".to_string(),
            },
        };

        let repo = provider.parse_repository(github_repo);

        assert_eq!(repo.name, "repo");
        assert_eq!(repo.default_branch, "main");
        assert!(!repo.is_private);
        assert!(repo.topics.is_empty());
    }

    mod http_mock_tests {
        use super::*;
        use futures::StreamExt;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        async fn create_provider_with_mock(mock_server: &MockServer) -> GitHubProvider {
            GitHubProvider::with_base_url(mock_server.uri(), Some("test-token".to_string()))
                .unwrap()
        }

        #[tokio::test]
        async fn test_get_repository_success() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/repos/owner/repo"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "name": "repo",
                    "full_name": "owner/repo",
                    "clone_url": "https://github.com/owner/repo.git",
                    "ssh_url": "git@github.com:owner/repo.git",
                    "default_branch": "main",
                    "description": "Test repository",
                    "language": "Rust",
                    "size": 1024,
                    "updated_at": "2024-01-15T10:30:00Z",
                    "private": false,
                    "topics": ["rust", "testing"],
                    "owner": {
                        "login": "owner"
                    }
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let result = provider.get_repository("owner", "repo").await;

            assert!(result.is_ok());
            let repo = result.unwrap();
            assert_eq!(repo.name, "repo");
            assert_eq!(repo.owner, "owner");
            assert_eq!(repo.default_branch, "main");
            assert_eq!(repo.language, Some("Rust".to_string()));
        }

        #[tokio::test]
        async fn test_get_repository_not_found() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/repos/owner/nonexistent"))
                .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                    "message": "Not Found"
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let result = provider.get_repository("owner", "nonexistent").await;

            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), ProviderError::NotFound(_)));
        }

        #[tokio::test]
        async fn test_get_repository_unauthorized() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/repos/owner/private-repo"))
                .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                    "message": "Bad credentials"
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let result = provider.get_repository("owner", "private-repo").await;

            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                ProviderError::AuthenticationError(_)
            ));
        }

        #[tokio::test]
        async fn test_get_repository_rate_limited() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/repos/owner/repo"))
                .respond_with(
                    ResponseTemplate::new(429)
                        .insert_header("retry-after", "60")
                        .set_body_json(serde_json::json!({
                            "message": "API rate limit exceeded"
                        })),
                )
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let result = provider.get_repository("owner", "repo").await;

            assert!(result.is_err());
            match result.unwrap_err() {
                ProviderError::RateLimitExceeded { retry_after_secs } => {
                    assert_eq!(retry_after_secs, 60);
                }
                e => panic!("Expected RateLimitExceeded, got {:?}", e),
            }
        }

        #[tokio::test]
        async fn test_list_repositories_success() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/user/repos"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {
                        "name": "repo1",
                        "full_name": "owner/repo1",
                        "clone_url": "https://github.com/owner/repo1.git",
                        "default_branch": "main",
                        "owner": { "login": "owner" }
                    },
                    {
                        "name": "repo2",
                        "full_name": "owner/repo2",
                        "clone_url": "https://github.com/owner/repo2.git",
                        "default_branch": "main",
                        "owner": { "login": "owner" }
                    }
                ])))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let repos: Vec<_> = provider.list_repositories(None, None, 10).collect().await;

            assert_eq!(repos.len(), 2);
            assert!(repos[0].is_ok());
            assert!(repos[1].is_ok());
            assert_eq!(repos[0].as_ref().unwrap().name, "repo1");
            assert_eq!(repos[1].as_ref().unwrap().name, "repo2");
        }

        #[tokio::test]
        async fn test_list_repositories_empty() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/user/repos"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let repos: Vec<_> = provider.list_repositories(None, None, 10).collect().await;

            assert!(repos.is_empty());
        }

        #[tokio::test]
        async fn test_list_repositories_for_org() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/orgs/myorg/repos"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {
                        "name": "org-repo",
                        "full_name": "myorg/org-repo",
                        "clone_url": "https://github.com/myorg/org-repo.git",
                        "default_branch": "main",
                        "owner": { "login": "myorg" }
                    }
                ])))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let repos: Vec<_> = provider
                .list_repositories(Some("myorg"), None, 10)
                .collect()
                .await;

            assert_eq!(repos.len(), 1);
            assert_eq!(repos[0].as_ref().unwrap().name, "org-repo");
        }

        #[tokio::test]
        async fn test_get_latest_commit_success() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/repos/owner/repo/commits/main"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "sha": "abc123def456"
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let result = provider
                .get_latest_commit("owner", "repo", Some("main"))
                .await;

            assert!(result.is_ok());
            assert_eq!(result.unwrap(), "abc123def456");
        }

        #[tokio::test]
        async fn test_get_latest_commit_uses_default_branch() {
            let mock_server = MockServer::start().await;

            // First call to get repository for default branch
            Mock::given(method("GET"))
                .and(path("/repos/owner/repo"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "name": "repo",
                    "full_name": "owner/repo",
                    "clone_url": "https://github.com/owner/repo.git",
                    "default_branch": "develop",
                    "owner": { "login": "owner" }
                })))
                .mount(&mock_server)
                .await;

            // Then call to get commit for that branch
            Mock::given(method("GET"))
                .and(path("/repos/owner/repo/commits/develop"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "sha": "def789"
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let result = provider.get_latest_commit("owner", "repo", None).await;

            assert!(result.is_ok());
            assert_eq!(result.unwrap(), "def789");
        }

        #[tokio::test]
        async fn test_get_default_branch_success() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/repos/owner/repo"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "name": "repo",
                    "full_name": "owner/repo",
                    "clone_url": "https://github.com/owner/repo.git",
                    "default_branch": "develop",
                    "owner": { "login": "owner" }
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let result = provider.get_default_branch("owner", "repo").await;

            assert!(result.is_ok());
            assert_eq!(result.unwrap(), "develop");
        }

        #[tokio::test]
        async fn test_list_repositories_unauthorized() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/user/repos"))
                .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                    "message": "Bad credentials"
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let repos: Vec<_> = provider.list_repositories(None, None, 10).collect().await;

            assert_eq!(repos.len(), 1);
            assert!(repos[0].is_err());
            assert!(matches!(
                repos[0].as_ref().unwrap_err(),
                ProviderError::AuthenticationError(_)
            ));
        }
    }
}
