//! GitLab API provider implementation.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::{self, BoxStream, StreamExt};
use reqwest::Client;
use reqwest::header::{ACCEPT, HeaderMap, HeaderValue};
use serde::Deserialize;

use super::base::{
    AuthType, BaseProvider, GitProvider, ProviderAuth, ProviderConfig, RateLimiter, Repository,
    WebhookEvent,
};
use super::error::{ProviderError, Result};

/// GitLab API response for a project.
#[derive(Debug, Deserialize)]
struct GitLabProject {
    path: String,
    path_with_namespace: String,
    http_url_to_repo: String,
    ssh_url_to_repo: Option<String>,
    default_branch: Option<String>,
    description: Option<String>,
    last_activity_at: Option<String>,
    visibility: Option<String>,
    topics: Option<Vec<String>>,
    namespace: Option<GitLabNamespace>,
    statistics: Option<GitLabStatistics>,
}

#[derive(Debug, Deserialize)]
struct GitLabNamespace {
    path: Option<String>,
    full_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GitLabStatistics {
    repository_size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct GitLabCommit {
    id: String,
}

/// GitLab API provider implementation.
#[derive(Debug, Clone)]
pub struct GitLabProvider {
    base: BaseProvider,
    webhook_secret: Option<String>,
}

impl GitLabProvider {
    /// Default GitLab API URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://gitlab.com/api/v4";

    /// Create a new GitLab provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let webhook_secret = config.webhook_secret.clone();
        let base = BaseProvider::new(config)?;
        Ok(Self {
            base,
            webhook_secret,
        })
    }

    /// Create a new GitLab provider with default settings and a token.
    pub fn with_token(token: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig {
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            auth: Some(ProviderAuth::token(token)),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a new GitLab provider with a custom base URL (for self-hosted GitLab).
    pub fn with_base_url(base_url: impl Into<String>, token: Option<String>) -> Result<Self> {
        let mut url = base_url.into();
        if !url.ends_with("/api/v4") {
            url = format!("{}/api/v4", url.trim_end_matches('/'));
        }
        let config = ProviderConfig {
            base_url: url,
            auth: token.map(ProviderAuth::token),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Set the webhook secret for token verification.
    pub fn with_webhook_secret(mut self, secret: impl Into<String>) -> Self {
        self.webhook_secret = Some(secret.into());
        self
    }

    fn parse_repository(&self, data: GitLabProject) -> Repository {
        let owner = data
            .namespace
            .as_ref()
            .and_then(|ns| {
                ns.path.clone().or_else(|| {
                    ns.full_path
                        .as_ref()
                        .and_then(|fp| fp.split('/').next().map(|s| s.to_string()))
                })
            })
            .unwrap_or_default();

        let updated_at = data.last_activity_at.as_ref().and_then(|s| {
            DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
        });

        let size_kb = data
            .statistics
            .as_ref()
            .and_then(|s| s.repository_size)
            .map_or(0, |s| s / 1024);

        Repository {
            provider: self.provider_name().to_string(),
            owner,
            name: data.path,
            full_name: data.path_with_namespace,
            clone_url: data.http_url_to_repo,
            ssh_url: data.ssh_url_to_repo,
            default_branch: data.default_branch.unwrap_or_else(|| "main".to_string()),
            description: data.description,
            language: None, // GitLab doesn't return primary language in project response
            size_kb,
            last_updated: updated_at,
            is_private: data.visibility.as_deref() == Some("private"),
            topics: data.topics.unwrap_or_default(),
        }
    }

    fn verify_webhook_token(&self, token: &str) -> Result<()> {
        let secret = self.webhook_secret.as_ref().ok_or_else(|| {
            ProviderError::ConfigurationError("Webhook secret not set".to_string())
        })?;

        if token != secret {
            return Err(ProviderError::InvalidWebhookSignature);
        }

        Ok(())
    }

    fn encode_project_path(owner: &str, name: &str) -> String {
        urlencoding::encode(&format!("{}/{}", owner, name)).into_owned()
    }
}

#[async_trait]
impl GitProvider for GitLabProvider {
    fn provider_name(&self) -> &'static str {
        "gitlab"
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
            if matches!(auth.auth_type, AuthType::Token | AuthType::OAuth) {
                if let Some(token) = &auth.token {
                    if let Ok(value) = HeaderValue::from_str(token) {
                        headers.insert("PRIVATE-TOKEN", value);
                    }
                }
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
                    Some(o) => format!("/groups/{}/projects", urlencoding::encode(o)),
                    None => "/projects".to_string(),
                };

                let mut params = vec![
                    ("per_page", per_page.to_string()),
                    ("page", page.to_string()),
                ];

                if let Some(vis) = &visibility {
                    params.push(("visibility", vis.clone()));
                }

                if owner.is_none() {
                    params.push(("membership", "true".to_string()));
                }

                let params: Vec<(&str, &str)> =
                    params.iter().map(|(k, v)| (*k, v.as_str())).collect();

                let headers = self.build_auth_headers();

                match self.base.get(&path, headers, &params).await {
                    Ok(response) => match response.json::<Vec<GitLabProject>>().await {
                        Ok(projects) => {
                            let is_empty = projects.is_empty();
                            let is_last = projects.len() < per_page as usize;

                            let repositories: Vec<Result<Repository>> = projects
                                .into_iter()
                                .map(|p| Ok(self.parse_repository(p)))
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
        let project_path = Self::encode_project_path(owner, name);
        let path = format!("/projects/{}", project_path);
        let headers = self.build_auth_headers();

        let response = self.base.get(&path, headers, &[]).await?;
        let project: GitLabProject = response.json().await?;
        Ok(self.parse_repository(project))
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

        let project_path = Self::encode_project_path(owner, name);
        let path = format!("/projects/{}/repository/commits/{}", project_path, ref_name);
        let headers = self.build_auth_headers();

        let response = self.base.get(&path, headers, &[]).await?;
        let commit: GitLabCommit = response.json().await?;
        Ok(commit.id)
    }

    async fn parse_webhook(
        &self,
        headers: &HashMap<String, String>,
        body: &[u8],
    ) -> Result<Option<WebhookEvent>> {
        let event_type = match headers.get("x-gitlab-event") {
            Some(t) => t.to_lowercase().replace(' ', "_"),
            None => return Ok(None),
        };

        // Verify token if webhook secret is set
        if self.webhook_secret.is_some() {
            let token = headers
                .get("x-gitlab-token")
                .ok_or(ProviderError::InvalidWebhookSignature)?;
            self.verify_webhook_token(token)?;
        }

        let payload: serde_json::Value = serde_json::from_slice(body)?;

        let project = match payload.get("project") {
            Some(p) => p,
            None => return Ok(None),
        };

        let repo = Repository {
            provider: self.provider_name().to_string(),
            owner: project
                .get("namespace")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            name: project
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            full_name: project
                .get("path_with_namespace")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            clone_url: project
                .get("http_url")
                .and_then(|u| u.as_str())
                .unwrap_or("")
                .to_string(),
            ssh_url: project
                .get("ssh_url")
                .and_then(|u| u.as_str())
                .map(|s| s.to_string()),
            default_branch: project
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
    use reqwest::header::ACCEPT;

    #[test]
    fn test_provider_name() {
        let provider = GitLabProvider::with_token("test-token").unwrap();
        assert_eq!(provider.provider_name(), "gitlab");
    }

    #[test]
    fn test_base_url() {
        let provider = GitLabProvider::with_token("test-token").unwrap();
        assert_eq!(provider.base_url(), "https://gitlab.com/api/v4");
    }

    #[test]
    fn test_custom_base_url_with_api() {
        let provider = GitLabProvider::with_base_url(
            "https://gitlab.example.com/api/v4",
            Some("token".to_string()),
        )
        .unwrap();
        assert_eq!(provider.base_url(), "https://gitlab.example.com/api/v4");
    }

    #[test]
    fn test_custom_base_url_without_api() {
        let provider =
            GitLabProvider::with_base_url("https://gitlab.example.com", Some("token".to_string()))
                .unwrap();
        assert_eq!(provider.base_url(), "https://gitlab.example.com/api/v4");
    }

    #[test]
    fn test_auth_headers_with_token() {
        let provider = GitLabProvider::with_token("my-token").unwrap();
        let headers = provider.build_auth_headers();

        assert!(headers.contains_key("PRIVATE-TOKEN"));
        assert_eq!(headers.get("PRIVATE-TOKEN").unwrap(), "my-token");
        assert_eq!(headers.get(ACCEPT).unwrap(), "application/json");
    }

    #[test]
    fn test_auth_headers_without_auth() {
        let config = ProviderConfig {
            base_url: GitLabProvider::DEFAULT_BASE_URL.to_string(),
            auth: None,
            ..Default::default()
        };
        let provider = GitLabProvider::new(config).unwrap();
        let headers = provider.build_auth_headers();

        assert!(!headers.contains_key("PRIVATE-TOKEN"));
        assert!(headers.contains_key(ACCEPT));
    }

    #[test]
    fn test_encode_project_path() {
        assert_eq!(
            GitLabProvider::encode_project_path("owner", "repo"),
            "owner%2Frepo"
        );
        assert_eq!(
            GitLabProvider::encode_project_path("my-org", "my-project"),
            "my-org%2Fmy-project"
        );
    }

    #[test]
    fn test_parse_repository() {
        let provider = GitLabProvider::with_token("test").unwrap();

        let gitlab_project = GitLabProject {
            path: "test-repo".to_string(),
            path_with_namespace: "owner/test-repo".to_string(),
            http_url_to_repo: "https://gitlab.com/owner/test-repo.git".to_string(),
            ssh_url_to_repo: Some("git@gitlab.com:owner/test-repo.git".to_string()),
            default_branch: Some("main".to_string()),
            description: Some("A test repository".to_string()),
            last_activity_at: Some("2024-01-15T10:30:00Z".to_string()),
            visibility: Some("private".to_string()),
            topics: Some(vec!["rust".to_string(), "testing".to_string()]),
            namespace: Some(GitLabNamespace {
                path: Some("owner".to_string()),
                full_path: Some("owner".to_string()),
            }),
            statistics: Some(GitLabStatistics {
                repository_size: Some(1024 * 1024), // 1 MB
            }),
        };

        let repo = provider.parse_repository(gitlab_project);

        assert_eq!(repo.provider, "gitlab");
        assert_eq!(repo.owner, "owner");
        assert_eq!(repo.name, "test-repo");
        assert_eq!(repo.full_name, "owner/test-repo");
        assert_eq!(repo.clone_url, "https://gitlab.com/owner/test-repo.git");
        assert_eq!(
            repo.ssh_url,
            Some("git@gitlab.com:owner/test-repo.git".to_string())
        );
        assert_eq!(repo.default_branch, "main");
        assert_eq!(repo.description, Some("A test repository".to_string()));
        assert!(repo.language.is_none()); // GitLab doesn't provide this
        assert_eq!(repo.size_kb, 1024);
        assert!(repo.last_updated.is_some());
        assert!(repo.is_private);
        assert_eq!(repo.topics, vec!["rust".to_string(), "testing".to_string()]);
    }

    #[test]
    fn test_parse_repository_minimal() {
        let provider = GitLabProvider::with_token("test").unwrap();

        let gitlab_project = GitLabProject {
            path: "repo".to_string(),
            path_with_namespace: "user/repo".to_string(),
            http_url_to_repo: "https://gitlab.com/user/repo.git".to_string(),
            ssh_url_to_repo: None,
            default_branch: None,
            description: None,
            last_activity_at: None,
            visibility: None,
            topics: None,
            namespace: None,
            statistics: None,
        };

        let repo = provider.parse_repository(gitlab_project);

        assert_eq!(repo.owner, "");
        assert_eq!(repo.name, "repo");
        assert_eq!(repo.default_branch, "main");
        assert!(!repo.is_private);
        assert!(repo.topics.is_empty());
    }

    #[test]
    fn test_webhook_token_verification() {
        let provider = GitLabProvider::with_token("test")
            .unwrap()
            .with_webhook_secret("mysecret");

        assert!(provider.verify_webhook_token("mysecret").is_ok());
        assert!(matches!(
            provider.verify_webhook_token("wrongsecret"),
            Err(ProviderError::InvalidWebhookSignature)
        ));
    }

    #[tokio::test]
    async fn test_parse_webhook_push_event() {
        let provider = GitLabProvider::with_token("test").unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-gitlab-event".to_string(), "Push Hook".to_string());

        let payload = serde_json::json!({
            "ref": "refs/heads/main",
            "before": "abc123",
            "after": "def456",
            "project": {
                "name": "test-repo",
                "path_with_namespace": "owner/test-repo",
                "http_url": "https://gitlab.com/owner/test-repo.git",
                "ssh_url": "git@gitlab.com:owner/test-repo.git",
                "default_branch": "main",
                "namespace": "owner"
            }
        });

        let body = serde_json::to_vec(&payload).unwrap();
        let result = provider.parse_webhook(&headers, &body).await.unwrap();

        assert!(result.is_some());
        let event = result.unwrap();
        assert_eq!(event.event_type, "push_hook");
        assert_eq!(event.repository.name, "test-repo");
        assert_eq!(event.repository.owner, "owner");
        assert_eq!(event.ref_name, Some("refs/heads/main".to_string()));
    }

    #[tokio::test]
    async fn test_parse_webhook_no_event_type() {
        let provider = GitLabProvider::with_token("test").unwrap();
        let headers = HashMap::new();
        let body = b"{}";

        let result = provider.parse_webhook(&headers, body).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_parse_webhook_no_project() {
        let provider = GitLabProvider::with_token("test").unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-gitlab-event".to_string(), "System Hook".to_string());

        let body = b"{}";
        let result = provider.parse_webhook(&headers, body).await.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_clone_url_with_token() {
        let provider = GitLabProvider::with_token("my-token").unwrap();

        let repo = Repository {
            provider: "gitlab".to_string(),
            owner: "test".to_string(),
            name: "repo".to_string(),
            full_name: "test/repo".to_string(),
            clone_url: "https://gitlab.com/test/repo.git".to_string(),
            ssh_url: Some("git@gitlab.com:test/repo.git".to_string()),
            default_branch: "main".to_string(),
            description: None,
            language: None,
            size_kb: 0,
            last_updated: None,
            is_private: false,
            topics: vec![],
        };

        let url = provider.get_clone_url(&repo);
        assert_eq!(url, "https://oauth2:my-token@gitlab.com/test/repo.git");
    }
}
