//! Gitea API provider implementation.

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

/// Gitea API response for a repository.
#[derive(Debug, Deserialize)]
struct GiteaRepo {
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
    owner: Option<GiteaOwner>,
}

#[derive(Debug, Deserialize)]
struct GiteaOwner {
    login: Option<String>,
    username: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GiteaCommit {
    sha: String,
}

/// Gitea API provider implementation.
#[derive(Debug, Clone)]
pub struct GiteaProvider {
    base: BaseProvider,
    webhook_secret: Option<String>,
}

impl GiteaProvider {
    /// Create a new Gitea provider with the given configuration.
    ///
    /// Note: Gitea requires a base URL since it's self-hosted.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let webhook_secret = config.webhook_secret.clone();

        // Ensure the URL ends with /api/v1
        let mut config = config;
        if !config.base_url.ends_with("/api/v1") {
            config.base_url = format!("{}/api/v1", config.base_url.trim_end_matches('/'));
        }

        let base = BaseProvider::new(config)?;
        Ok(Self {
            base,
            webhook_secret,
        })
    }

    /// Create a new Gitea provider with the given base URL and token.
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

    fn parse_repository(&self, data: GiteaRepo) -> Repository {
        let owner = data
            .owner
            .as_ref()
            .and_then(|o| o.login.clone().or_else(|| o.username.clone()))
            .unwrap_or_default();

        let updated_at = data.updated_at.as_ref().and_then(|s| {
            DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        });

        Repository {
            provider: self.provider_name().to_string(),
            owner,
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

        let mut mac =
            HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can accept any key length");
        mac.update(body);
        let expected = hex::encode(mac.finalize().into_bytes());

        if !constant_time_compare(expected.as_bytes(), signature.as_bytes()) {
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
impl GitProvider for GiteaProvider {
    fn provider_name(&self) -> &'static str {
        "gitea"
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
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));

        if let Some(auth) = &self.base.auth {
            match auth.auth_type {
                AuthType::Token => {
                    if let Some(token) = &auth.token {
                        if let Ok(value) = HeaderValue::from_str(&format!("token {}", token)) {
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

                let params = [("limit", per_page.to_string()), ("page", page.to_string())];

                let params: Vec<(&str, &str)> =
                    params.iter().map(|(k, v)| (*k, v.as_str())).collect();

                let headers = self.build_auth_headers();

                match self.base.get(&path, headers, &params).await {
                    Ok(response) => match response.json::<Vec<GiteaRepo>>().await {
                        Ok(repos) => {
                            let is_empty = repos.is_empty();
                            let is_last = repos.len() < per_page as usize;

                            let repositories: Vec<Result<Repository>> = repos
                                .into_iter()
                                .filter_map(|r| {
                                    let repo = self.parse_repository(r);
                                    // Apply visibility filter
                                    if let Some(ref vis) = visibility {
                                        if vis == "private" && !repo.is_private {
                                            return None;
                                        }
                                        if vis == "public" && repo.is_private {
                                            return None;
                                        }
                                    }
                                    Some(Ok(repo))
                                })
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
        let repo: GiteaRepo = response.json().await?;
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

        let path = format!("/repos/{}/{}/git/commits/{}", owner, name, ref_name);
        let headers = self.build_auth_headers();

        let response = self.base.get(&path, headers, &[]).await?;
        let commit: GiteaCommit = response.json().await?;
        Ok(commit.sha)
    }

    async fn parse_webhook(
        &self,
        headers: &HashMap<String, String>,
        body: &[u8],
    ) -> Result<Option<WebhookEvent>> {
        // Gitea uses x-gitea-event, but also supports x-gogs-event for compatibility
        let event_type = headers
            .get("x-gitea-event")
            .or_else(|| headers.get("x-gogs-event"))
            .cloned();

        let event_type = match event_type {
            Some(t) => t,
            None => return Ok(None),
        };

        // Verify signature if webhook secret is set
        if self.webhook_secret.is_some() {
            let signature = headers
                .get("x-gitea-signature")
                .or_else(|| headers.get("x-gogs-signature"))
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
            .and_then(|o| o.get("login").or_else(|| o.get("username")))
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
        let provider = GiteaProvider::with_base_url("https://gitea.example.com", None).unwrap();
        assert_eq!(provider.provider_name(), "gitea");
    }

    #[test]
    fn test_base_url_with_api() {
        let provider =
            GiteaProvider::with_base_url("https://gitea.example.com/api/v1", None).unwrap();
        assert_eq!(provider.base_url(), "https://gitea.example.com/api/v1");
    }

    #[test]
    fn test_base_url_without_api() {
        let provider = GiteaProvider::with_base_url("https://gitea.example.com", None).unwrap();
        assert_eq!(provider.base_url(), "https://gitea.example.com/api/v1");
    }

    #[test]
    fn test_auth_headers_with_token() {
        let provider =
            GiteaProvider::with_base_url("https://gitea.example.com", Some("my-token".to_string()))
                .unwrap();
        let headers = provider.build_auth_headers();

        assert!(headers.contains_key(AUTHORIZATION));
        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "token my-token");
        assert_eq!(headers.get(ACCEPT).unwrap(), "application/json");
    }

    #[test]
    fn test_auth_headers_with_basic() {
        let config = ProviderConfig {
            base_url: "https://gitea.example.com".to_string(),
            auth: Some(ProviderAuth::basic("user", "pass")),
            ..Default::default()
        };
        let provider = GiteaProvider::new(config).unwrap();
        let headers = provider.build_auth_headers();

        assert!(headers.contains_key(AUTHORIZATION));
        let auth_value = headers.get(AUTHORIZATION).unwrap().to_str().unwrap();
        assert!(auth_value.starts_with("Basic "));
    }

    #[test]
    fn test_auth_headers_without_auth() {
        let config = ProviderConfig {
            base_url: "https://gitea.example.com".to_string(),
            auth: None,
            ..Default::default()
        };
        let provider = GiteaProvider::new(config).unwrap();
        let headers = provider.build_auth_headers();

        assert!(!headers.contains_key(AUTHORIZATION));
        assert!(headers.contains_key(ACCEPT));
    }

    #[test]
    fn test_parse_repository() {
        let provider = GiteaProvider::with_base_url("https://gitea.example.com", None).unwrap();

        let gitea_repo = GiteaRepo {
            name: "test-repo".to_string(),
            full_name: "owner/test-repo".to_string(),
            clone_url: "https://gitea.example.com/owner/test-repo.git".to_string(),
            ssh_url: Some("git@gitea.example.com:owner/test-repo.git".to_string()),
            default_branch: Some("main".to_string()),
            description: Some("A test repository".to_string()),
            language: Some("Rust".to_string()),
            size: Some(1024),
            updated_at: Some("2024-01-15T10:30:00Z".to_string()),
            private: Some(true),
            topics: Some(vec!["rust".to_string(), "testing".to_string()]),
            owner: Some(GiteaOwner {
                login: Some("owner".to_string()),
                username: None,
            }),
        };

        let repo = provider.parse_repository(gitea_repo);

        assert_eq!(repo.provider, "gitea");
        assert_eq!(repo.owner, "owner");
        assert_eq!(repo.name, "test-repo");
        assert_eq!(repo.full_name, "owner/test-repo");
        assert_eq!(
            repo.clone_url,
            "https://gitea.example.com/owner/test-repo.git"
        );
        assert_eq!(
            repo.ssh_url,
            Some("git@gitea.example.com:owner/test-repo.git".to_string())
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
    fn test_parse_repository_with_username() {
        let provider = GiteaProvider::with_base_url("https://gitea.example.com", None).unwrap();

        let gitea_repo = GiteaRepo {
            name: "repo".to_string(),
            full_name: "user/repo".to_string(),
            clone_url: "https://gitea.example.com/user/repo.git".to_string(),
            ssh_url: None,
            default_branch: None,
            description: None,
            language: None,
            size: None,
            updated_at: None,
            private: None,
            topics: None,
            owner: Some(GiteaOwner {
                login: None,
                username: Some("user".to_string()),
            }),
        };

        let repo = provider.parse_repository(gitea_repo);

        assert_eq!(repo.owner, "user");
        assert_eq!(repo.name, "repo");
        assert_eq!(repo.default_branch, "main");
        assert!(!repo.is_private);
    }

    #[test]
    fn test_webhook_signature_verification() {
        let provider = GiteaProvider::with_base_url("https://gitea.example.com", None)
            .unwrap()
            .with_webhook_secret("mysecret");

        let body = b"test payload";

        // Generate a valid signature (Gitea doesn't use prefix)
        let mut mac = HmacSha256::new_from_slice(b"mysecret").unwrap();
        mac.update(body);
        let signature = hex::encode(mac.finalize().into_bytes());

        assert!(provider.verify_webhook_signature(body, &signature).is_ok());
    }

    #[test]
    fn test_webhook_signature_verification_invalid() {
        let provider = GiteaProvider::with_base_url("https://gitea.example.com", None)
            .unwrap()
            .with_webhook_secret("mysecret");

        let body = b"test payload";
        let signature = "invalid_signature";

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
    }

    #[tokio::test]
    async fn test_parse_webhook_push_event() {
        let provider = GiteaProvider::with_base_url("https://gitea.example.com", None).unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-gitea-event".to_string(), "push".to_string());

        let payload = serde_json::json!({
            "ref": "refs/heads/main",
            "before": "abc123",
            "after": "def456",
            "repository": {
                "name": "test-repo",
                "full_name": "owner/test-repo",
                "clone_url": "https://gitea.example.com/owner/test-repo.git",
                "ssh_url": "git@gitea.example.com:owner/test-repo.git",
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
    async fn test_parse_webhook_gogs_compatibility() {
        let provider = GiteaProvider::with_base_url("https://gitea.example.com", None).unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-gogs-event".to_string(), "push".to_string());

        let payload = serde_json::json!({
            "repository": {
                "name": "repo",
                "full_name": "owner/repo",
                "clone_url": "https://gitea.example.com/owner/repo.git",
                "owner": {
                    "username": "owner"
                }
            }
        });

        let body = serde_json::to_vec(&payload).unwrap();
        let result = provider.parse_webhook(&headers, &body).await.unwrap();

        assert!(result.is_some());
        let event = result.unwrap();
        assert_eq!(event.event_type, "push");
    }

    #[tokio::test]
    async fn test_parse_webhook_no_event_type() {
        let provider = GiteaProvider::with_base_url("https://gitea.example.com", None).unwrap();
        let headers = HashMap::new();
        let body = b"{}";

        let result = provider.parse_webhook(&headers, body).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_parse_webhook_no_repository() {
        let provider = GiteaProvider::with_base_url("https://gitea.example.com", None).unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-gitea-event".to_string(), "create".to_string());

        let body = b"{}";
        let result = provider.parse_webhook(&headers, body).await.unwrap();
        assert!(result.is_none());
    }
}
