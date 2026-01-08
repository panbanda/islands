//! Bitbucket Cloud API provider implementation.

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

/// Bitbucket API response for a repository.
#[derive(Debug, Deserialize)]
struct BitbucketRepo {
    slug: String,
    full_name: String,
    description: Option<String>,
    language: Option<String>,
    size: Option<u64>,
    updated_on: Option<String>,
    is_private: Option<bool>,
    mainbranch: Option<BitbucketBranch>,
    owner: Option<BitbucketOwner>,
    workspace: Option<BitbucketWorkspace>,
    links: Option<BitbucketLinks>,
}

#[derive(Debug, Deserialize)]
struct BitbucketBranch {
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BitbucketOwner {
    username: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BitbucketWorkspace {
    slug: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BitbucketLinks {
    clone: Option<Vec<BitbucketCloneLink>>,
}

#[derive(Debug, Deserialize)]
struct BitbucketCloneLink {
    name: Option<String>,
    href: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BitbucketCommit {
    hash: String,
}

#[derive(Debug, Deserialize)]
struct BitbucketPaginated<T> {
    values: Vec<T>,
    next: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BitbucketPermissionRepo {
    repository: BitbucketRepo,
}

/// Bitbucket Cloud API provider implementation.
#[derive(Debug, Clone)]
pub struct BitbucketProvider {
    base: BaseProvider,
    webhook_secret: Option<String>,
}

impl BitbucketProvider {
    /// Default Bitbucket API URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.bitbucket.org/2.0";

    /// Create a new Bitbucket provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let webhook_secret = config.webhook_secret.clone();
        let base = BaseProvider::new(config)?;
        Ok(Self {
            base,
            webhook_secret,
        })
    }

    /// Create a new Bitbucket provider with default settings and a token.
    pub fn with_token(token: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig {
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            auth: Some(ProviderAuth::token(token)),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a new Bitbucket provider with basic authentication.
    pub fn with_basic_auth(
        username: impl Into<String>,
        app_password: impl Into<String>,
    ) -> Result<Self> {
        let config = ProviderConfig {
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            auth: Some(ProviderAuth::basic(username, app_password)),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Set the webhook secret for signature verification.
    pub fn with_webhook_secret(mut self, secret: impl Into<String>) -> Self {
        self.webhook_secret = Some(secret.into());
        self
    }

    fn parse_repository(&self, data: BitbucketRepo) -> Repository {
        let owner = data
            .owner
            .as_ref()
            .and_then(|o| o.username.clone())
            .or_else(|| data.workspace.as_ref().and_then(|w| w.slug.clone()))
            .unwrap_or_default();

        let (clone_url, ssh_url) = data
            .links
            .as_ref()
            .and_then(|l| l.clone.as_ref())
            .map(|links| {
                let https = links
                    .iter()
                    .find(|l| l.name.as_deref() == Some("https"))
                    .and_then(|l| l.href.clone())
                    .unwrap_or_default();
                let ssh = links
                    .iter()
                    .find(|l| l.name.as_deref() == Some("ssh"))
                    .and_then(|l| l.href.clone());
                (https, ssh)
            })
            .unwrap_or_default();

        let updated_at = data.updated_on.as_ref().and_then(|s| {
            DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        });

        Repository {
            provider: self.provider_name().to_string(),
            owner,
            name: data.slug,
            full_name: data.full_name,
            clone_url,
            ssh_url,
            default_branch: data
                .mainbranch
                .and_then(|b| b.name)
                .unwrap_or_else(|| "main".to_string()),
            description: data.description,
            language: data.language,
            size_kb: data.size.map(|s| s / 1024).unwrap_or(0),
            last_updated: updated_at,
            is_private: data.is_private.unwrap_or(false),
            topics: vec![], // Bitbucket doesn't have topics
        }
    }

    fn verify_webhook_signature(&self, body: &[u8], signature: &str) -> Result<()> {
        let secret = self.webhook_secret.as_ref().ok_or_else(|| {
            ProviderError::ConfigurationError("Webhook secret not set".to_string())
        })?;

        let mut mac =
            HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can accept any key length");
        mac.update(body);
        let expected = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));

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
impl GitProvider for BitbucketProvider {
    fn provider_name(&self) -> &'static str {
        "bitbucket"
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
        let per_page = per_page.min(100); // Bitbucket max is 100

        stream::unfold(
            (None::<String>, false, true),
            move |(next_url, done, is_first)| {
                let owner = owner.clone();
                let visibility = visibility.clone();

                async move {
                    if done {
                        return None;
                    }

                    let (path, use_permission_endpoint) = if let Some(url) = next_url {
                        (url, owner.is_none())
                    } else if is_first {
                        match &owner {
                            Some(o) => (format!("/repositories/{}", o), false),
                            None => ("/user/permissions/repositories".to_string(), true),
                        }
                    } else {
                        return None;
                    };

                    let params = if is_first {
                        vec![("pagelen", per_page.to_string())]
                    } else {
                        vec![]
                    };

                    let params: Vec<(&str, &str)> =
                        params.iter().map(|(k, v)| (*k, v.as_str())).collect();

                    let headers = self.build_auth_headers();

                    // Handle absolute URLs from pagination
                    let response = if path.starts_with("http") {
                        if self.base.rate_limiter.check_and_wait().await.is_err() {
                            return None;
                        }
                        match self.base.client.get(&path).headers(headers).send().await {
                            Ok(r) => match BaseProvider::check_response(r).await {
                                Ok(resp) => resp,
                                Err(e) => {
                                    return Some((stream::iter(vec![Err(e)]), (None, true, false)));
                                }
                            },
                            Err(e) => {
                                return Some((
                                    stream::iter(vec![Err(e.into())]),
                                    (None, true, false),
                                ));
                            }
                        }
                    } else {
                        match self.base.get(&path, headers, &params).await {
                            Ok(r) => r,
                            Err(e) => {
                                return Some((stream::iter(vec![Err(e)]), (None, true, false)));
                            }
                        }
                    };

                    if use_permission_endpoint {
                        match response
                            .json::<BitbucketPaginated<BitbucketPermissionRepo>>()
                            .await
                        {
                            Ok(paginated) => {
                                let next = paginated.next;
                                let is_last = next.is_none();
                                let repositories: Vec<Result<Repository>> = paginated
                                    .values
                                    .into_iter()
                                    .filter_map(|item| {
                                        let repo = self.parse_repository(item.repository);
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

                                if repositories.is_empty() && is_last {
                                    None
                                } else {
                                    Some((stream::iter(repositories), (next, is_last, false)))
                                }
                            }
                            Err(e) => Some((
                                stream::iter(vec![Err(ProviderError::HttpError(e))]),
                                (None, true, false),
                            )),
                        }
                    } else {
                        match response.json::<BitbucketPaginated<BitbucketRepo>>().await {
                            Ok(paginated) => {
                                let next = paginated.next;
                                let is_last = next.is_none();
                                let repositories: Vec<Result<Repository>> = paginated
                                    .values
                                    .into_iter()
                                    .filter_map(|repo_data| {
                                        let repo = self.parse_repository(repo_data);
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

                                if repositories.is_empty() && is_last {
                                    None
                                } else {
                                    Some((stream::iter(repositories), (next, is_last, false)))
                                }
                            }
                            Err(e) => Some((
                                stream::iter(vec![Err(ProviderError::HttpError(e))]),
                                (None, true, false),
                            )),
                        }
                    }
                }
            },
        )
        .flatten()
        .boxed()
    }

    async fn get_repository(&self, owner: &str, name: &str) -> Result<Repository> {
        let path = format!("/repositories/{}/{}", owner, name);
        let headers = self.build_auth_headers();

        let response = self.base.get(&path, headers, &[]).await?;
        let repo: BitbucketRepo = response.json().await?;
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

        let path = format!("/repositories/{}/{}/commits/{}", owner, name, ref_name);
        let headers = self.build_auth_headers();
        let params = [("pagelen", "1")];

        let response = self.base.get(&path, headers, &params).await?;
        let paginated: BitbucketPaginated<BitbucketCommit> = response.json().await?;

        paginated
            .values
            .into_iter()
            .next()
            .map(|c| c.hash)
            .ok_or_else(|| ProviderError::NotFound(format!("No commits found for {}", ref_name)))
    }

    async fn parse_webhook(
        &self,
        headers: &HashMap<String, String>,
        body: &[u8],
    ) -> Result<Option<WebhookEvent>> {
        let event_key = match headers.get("x-event-key") {
            Some(k) => k.clone(),
            None => return Ok(None),
        };

        // Verify signature if webhook secret is set
        if self.webhook_secret.is_some() {
            if let Some(signature) = headers.get("x-hub-signature") {
                self.verify_webhook_signature(body, signature)?;
            }
        }

        let payload: serde_json::Value = serde_json::from_slice(body)?;

        let repo_data = match payload.get("repository") {
            Some(r) => r,
            None => return Ok(None),
        };

        // Parse repository from webhook payload
        let repo_json: BitbucketRepo = serde_json::from_value(repo_data.clone())?;
        let repo = self.parse_repository(repo_json);

        // Extract push changes if present
        let (ref_name, before, after) = payload
            .get("push")
            .and_then(|p| p.get("changes"))
            .and_then(|c| c.as_array())
            .and_then(|changes| changes.first())
            .map(|change| {
                let new_state = change.get("new");
                let old_state = change.get("old");

                let ref_name = new_state
                    .and_then(|n| n.get("name"))
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string());

                let after = new_state
                    .and_then(|n| n.get("target"))
                    .and_then(|t| t.get("hash"))
                    .and_then(|h| h.as_str())
                    .map(|s| s.to_string());

                let before = old_state
                    .and_then(|o| o.get("target"))
                    .and_then(|t| t.get("hash"))
                    .and_then(|h| h.as_str())
                    .map(|s| s.to_string());

                (ref_name, before, after)
            })
            .unwrap_or((None, None, None));

        let payload_map: HashMap<String, serde_json::Value> =
            serde_json::from_value(payload.clone()).unwrap_or_default();

        Ok(Some(WebhookEvent {
            event_type: event_key,
            repository: repo,
            ref_name,
            before,
            after,
            payload: payload_map,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let provider = BitbucketProvider::with_token("test-token").unwrap();
        assert_eq!(provider.provider_name(), "bitbucket");
    }

    #[test]
    fn test_base_url() {
        let provider = BitbucketProvider::with_token("test-token").unwrap();
        assert_eq!(provider.base_url(), "https://api.bitbucket.org/2.0");
    }

    #[test]
    fn test_auth_headers_with_token() {
        let provider = BitbucketProvider::with_token("my-token").unwrap();
        let headers = provider.build_auth_headers();

        assert!(headers.contains_key(AUTHORIZATION));
        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer my-token");
        assert_eq!(headers.get(ACCEPT).unwrap(), "application/json");
    }

    #[test]
    fn test_auth_headers_with_basic() {
        let provider = BitbucketProvider::with_basic_auth("user", "app_password").unwrap();
        let headers = provider.build_auth_headers();

        assert!(headers.contains_key(AUTHORIZATION));
        let auth_value = headers.get(AUTHORIZATION).unwrap().to_str().unwrap();
        assert!(auth_value.starts_with("Basic "));
    }

    #[test]
    fn test_auth_headers_without_auth() {
        let config = ProviderConfig {
            base_url: BitbucketProvider::DEFAULT_BASE_URL.to_string(),
            auth: None,
            ..Default::default()
        };
        let provider = BitbucketProvider::new(config).unwrap();
        let headers = provider.build_auth_headers();

        assert!(!headers.contains_key(AUTHORIZATION));
        assert!(headers.contains_key(ACCEPT));
    }

    #[test]
    fn test_parse_repository() {
        let provider = BitbucketProvider::with_token("test").unwrap();

        let bitbucket_repo = BitbucketRepo {
            slug: "test-repo".to_string(),
            full_name: "owner/test-repo".to_string(),
            description: Some("A test repository".to_string()),
            language: Some("Rust".to_string()),
            size: Some(1024 * 1024), // 1 MB
            updated_on: Some("2024-01-15T10:30:00Z".to_string()),
            is_private: Some(true),
            mainbranch: Some(BitbucketBranch {
                name: Some("main".to_string()),
            }),
            owner: Some(BitbucketOwner {
                username: Some("owner".to_string()),
            }),
            workspace: None,
            links: Some(BitbucketLinks {
                clone: Some(vec![
                    BitbucketCloneLink {
                        name: Some("https".to_string()),
                        href: Some("https://bitbucket.org/owner/test-repo.git".to_string()),
                    },
                    BitbucketCloneLink {
                        name: Some("ssh".to_string()),
                        href: Some("git@bitbucket.org:owner/test-repo.git".to_string()),
                    },
                ]),
            }),
        };

        let repo = provider.parse_repository(bitbucket_repo);

        assert_eq!(repo.provider, "bitbucket");
        assert_eq!(repo.owner, "owner");
        assert_eq!(repo.name, "test-repo");
        assert_eq!(repo.full_name, "owner/test-repo");
        assert_eq!(repo.clone_url, "https://bitbucket.org/owner/test-repo.git");
        assert_eq!(
            repo.ssh_url,
            Some("git@bitbucket.org:owner/test-repo.git".to_string())
        );
        assert_eq!(repo.default_branch, "main");
        assert_eq!(repo.description, Some("A test repository".to_string()));
        assert_eq!(repo.language, Some("Rust".to_string()));
        assert_eq!(repo.size_kb, 1024);
        assert!(repo.last_updated.is_some());
        assert!(repo.is_private);
        assert!(repo.topics.is_empty()); // Bitbucket doesn't have topics
    }

    #[test]
    fn test_parse_repository_with_workspace() {
        let provider = BitbucketProvider::with_token("test").unwrap();

        let bitbucket_repo = BitbucketRepo {
            slug: "repo".to_string(),
            full_name: "workspace/repo".to_string(),
            description: None,
            language: None,
            size: None,
            updated_on: None,
            is_private: None,
            mainbranch: None,
            owner: None,
            workspace: Some(BitbucketWorkspace {
                slug: Some("workspace".to_string()),
            }),
            links: None,
        };

        let repo = provider.parse_repository(bitbucket_repo);

        assert_eq!(repo.owner, "workspace");
        assert_eq!(repo.name, "repo");
        assert_eq!(repo.default_branch, "main");
        assert!(!repo.is_private);
    }

    #[test]
    fn test_webhook_signature_verification() {
        let provider = BitbucketProvider::with_token("test")
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
        let provider = BitbucketProvider::with_token("test")
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
    fn test_constant_time_compare() {
        assert!(constant_time_compare(b"hello", b"hello"));
        assert!(!constant_time_compare(b"hello", b"world"));
        assert!(!constant_time_compare(b"hello", b"helloworld"));
    }

    #[tokio::test]
    async fn test_parse_webhook_push_event() {
        let provider = BitbucketProvider::with_token("test").unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-event-key".to_string(), "repo:push".to_string());

        let payload = serde_json::json!({
            "push": {
                "changes": [{
                    "new": {
                        "name": "main",
                        "target": {
                            "hash": "def456"
                        }
                    },
                    "old": {
                        "target": {
                            "hash": "abc123"
                        }
                    }
                }]
            },
            "repository": {
                "slug": "test-repo",
                "full_name": "owner/test-repo",
                "owner": {
                    "username": "owner"
                },
                "mainbranch": {
                    "name": "main"
                },
                "links": {
                    "clone": [{
                        "name": "https",
                        "href": "https://bitbucket.org/owner/test-repo.git"
                    }]
                }
            }
        });

        let body = serde_json::to_vec(&payload).unwrap();
        let result = provider.parse_webhook(&headers, &body).await.unwrap();

        assert!(result.is_some());
        let event = result.unwrap();
        assert_eq!(event.event_type, "repo:push");
        assert_eq!(event.repository.name, "test-repo");
        assert_eq!(event.repository.owner, "owner");
        assert_eq!(event.ref_name, Some("main".to_string()));
        assert_eq!(event.before, Some("abc123".to_string()));
        assert_eq!(event.after, Some("def456".to_string()));
    }

    #[tokio::test]
    async fn test_parse_webhook_no_event_key() {
        let provider = BitbucketProvider::with_token("test").unwrap();
        let headers = HashMap::new();
        let body = b"{}";

        let result = provider.parse_webhook(&headers, body).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_parse_webhook_no_repository() {
        let provider = BitbucketProvider::with_token("test").unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-event-key".to_string(), "repo:fork".to_string());

        let body = b"{}";
        let result = provider.parse_webhook(&headers, body).await.unwrap();
        assert!(result.is_none());
    }

    mod http_mock_tests {
        use super::*;
        use futures::StreamExt;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        async fn create_provider_with_mock(mock_server: &MockServer) -> BitbucketProvider {
            let config = ProviderConfig {
                base_url: mock_server.uri(),
                auth: Some(ProviderAuth::token("test-token")),
                ..Default::default()
            };
            BitbucketProvider::new(config).unwrap()
        }

        #[tokio::test]
        async fn test_get_repository_success() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/repositories/owner/repo"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "slug": "repo",
                    "full_name": "owner/repo",
                    "description": "Test repository",
                    "language": "Rust",
                    "size": 1048576,
                    "updated_on": "2024-01-15T10:30:00Z",
                    "is_private": false,
                    "mainbranch": { "name": "main" },
                    "owner": { "username": "owner" },
                    "links": {
                        "clone": [
                            { "name": "https", "href": "https://bitbucket.org/owner/repo.git" },
                            { "name": "ssh", "href": "git@bitbucket.org:owner/repo.git" }
                        ]
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
                .and(path("/repositories/owner/nonexistent"))
                .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                    "error": { "message": "Repository owner/nonexistent not found" }
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
                .and(path("/repositories/owner/private-repo"))
                .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                    "error": { "message": "Authentication required" }
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
                .and(path("/repositories/owner/repo"))
                .respond_with(
                    ResponseTemplate::new(429)
                        .insert_header("retry-after", "90")
                        .set_body_json(serde_json::json!({
                            "error": { "message": "Rate limit exceeded" }
                        })),
                )
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let result = provider.get_repository("owner", "repo").await;

            assert!(result.is_err());
            match result.unwrap_err() {
                ProviderError::RateLimitExceeded { retry_after_secs } => {
                    assert_eq!(retry_after_secs, 90);
                }
                e => panic!("Expected RateLimitExceeded, got {:?}", e),
            }
        }

        #[tokio::test]
        async fn test_list_repositories_for_workspace() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/repositories/myworkspace"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "values": [
                        {
                            "slug": "repo1",
                            "full_name": "myworkspace/repo1",
                            "mainbranch": { "name": "main" },
                            "workspace": { "slug": "myworkspace" },
                            "links": {
                                "clone": [
                                    { "name": "https", "href": "https://bitbucket.org/myworkspace/repo1.git" }
                                ]
                            }
                        },
                        {
                            "slug": "repo2",
                            "full_name": "myworkspace/repo2",
                            "mainbranch": { "name": "main" },
                            "workspace": { "slug": "myworkspace" },
                            "links": {
                                "clone": [
                                    { "name": "https", "href": "https://bitbucket.org/myworkspace/repo2.git" }
                                ]
                            }
                        }
                    ],
                    "next": null
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let repos: Vec<_> = provider
                .list_repositories(Some("myworkspace"), None, 10)
                .collect()
                .await;

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
                .and(path("/repositories/emptyworkspace"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "values": [],
                    "next": null
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let repos: Vec<_> = provider
                .list_repositories(Some("emptyworkspace"), None, 10)
                .collect()
                .await;

            assert!(repos.is_empty());
        }

        #[tokio::test]
        async fn test_get_latest_commit_success() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/repositories/owner/repo/commits/main"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "values": [
                        { "hash": "abc123def456" }
                    ]
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
                .and(path("/repositories/owner/repo"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "slug": "repo",
                    "full_name": "owner/repo",
                    "mainbranch": { "name": "develop" },
                    "owner": { "username": "owner" },
                    "links": {
                        "clone": [
                            { "name": "https", "href": "https://bitbucket.org/owner/repo.git" }
                        ]
                    }
                })))
                .mount(&mock_server)
                .await;

            // Then call to get commit for that branch
            Mock::given(method("GET"))
                .and(path("/repositories/owner/repo/commits/develop"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "values": [
                        { "hash": "def789" }
                    ]
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
                .and(path("/repositories/owner/repo"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "slug": "repo",
                    "full_name": "owner/repo",
                    "mainbranch": { "name": "develop" },
                    "owner": { "username": "owner" },
                    "links": {
                        "clone": [
                            { "name": "https", "href": "https://bitbucket.org/owner/repo.git" }
                        ]
                    }
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
                .and(path("/repositories/owner"))
                .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                    "error": { "message": "Authentication required" }
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let repos: Vec<_> = provider
                .list_repositories(Some("owner"), None, 10)
                .collect()
                .await;

            assert_eq!(repos.len(), 1);
            assert!(repos[0].is_err());
            assert!(matches!(
                repos[0].as_ref().unwrap_err(),
                ProviderError::AuthenticationError(_)
            ));
        }

        #[tokio::test]
        async fn test_get_latest_commit_no_commits() {
            let mock_server = MockServer::start().await;

            Mock::given(method("GET"))
                .and(path("/repositories/owner/repo/commits/main"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "values": []
                })))
                .mount(&mock_server)
                .await;

            let provider = create_provider_with_mock(&mock_server).await;
            let result = provider
                .get_latest_commit("owner", "repo", Some("main"))
                .await;

            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), ProviderError::NotFound(_)));
        }
    }
}
