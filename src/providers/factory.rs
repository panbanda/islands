//! Provider factory for creating git provider instances.

use std::sync::Arc;

use super::base::{GitProvider, ProviderAuth, ProviderConfig};
use super::bitbucket::BitbucketProvider;
use super::error::{ProviderError, Result};
use super::gitea::GiteaProvider;
use super::github::GitHubProvider;
use super::gitlab::GitLabProvider;

/// Supported provider types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderType {
    /// GitHub (github.com or GitHub Enterprise)
    GitHub,
    /// GitLab (gitlab.com or self-hosted)
    GitLab,
    /// Bitbucket Cloud
    Bitbucket,
    /// Gitea (self-hosted)
    Gitea,
}

impl ProviderType {
    /// Parse provider type from a string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "github" | "github.com" => Some(Self::GitHub),
            "gitlab" | "gitlab.com" => Some(Self::GitLab),
            "bitbucket" | "bitbucket.org" => Some(Self::Bitbucket),
            "gitea" => Some(Self::Gitea),
            _ => None,
        }
    }

    /// Get the default base URL for this provider type.
    pub fn default_base_url(&self) -> Option<&'static str> {
        match self {
            Self::GitHub => Some(GitHubProvider::DEFAULT_BASE_URL),
            Self::GitLab => Some(GitLabProvider::DEFAULT_BASE_URL),
            Self::Bitbucket => Some(BitbucketProvider::DEFAULT_BASE_URL),
            Self::Gitea => None, // Gitea requires a custom URL
        }
    }
}

/// Factory for creating git provider instances.
#[derive(Debug, Clone, Default)]
pub struct ProviderFactory {
    /// Default authentication to use when creating providers.
    default_auth: Option<ProviderAuth>,
}

impl ProviderFactory {
    /// Create a new factory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set default authentication for all providers.
    pub fn with_default_auth(mut self, auth: ProviderAuth) -> Self {
        self.default_auth = Some(auth);
        self
    }

    /// Create a provider from a repository URL.
    pub fn from_url(&self, url: &str) -> Result<Arc<dyn GitProvider>> {
        let provider_type = Self::detect_provider(url)?;
        self.create(&provider_type, None)
    }

    /// Create a provider by name.
    pub fn create(
        &self,
        provider_type: &str,
        config: Option<ProviderConfig>,
    ) -> Result<Arc<dyn GitProvider>> {
        let provider_type_enum = ProviderType::parse(provider_type)
            .ok_or_else(|| ProviderError::UnsupportedProvider(provider_type.to_string()))?;

        let config = config.unwrap_or_else(|| self.default_config(provider_type_enum));

        match provider_type_enum {
            ProviderType::GitHub => {
                let provider = GitHubProvider::new(config)?;
                Ok(Arc::new(provider))
            }
            ProviderType::GitLab => {
                let provider = GitLabProvider::new(config)?;
                Ok(Arc::new(provider))
            }
            ProviderType::Bitbucket => {
                let provider = BitbucketProvider::new(config)?;
                Ok(Arc::new(provider))
            }
            ProviderType::Gitea => {
                if config.base_url.is_empty() {
                    return Err(ProviderError::ConfigurationError(
                        "Gitea provider requires a base_url".to_string(),
                    ));
                }
                let provider = GiteaProvider::new(config)?;
                Ok(Arc::new(provider))
            }
        }
    }

    /// Create a provider with a specific type enum.
    pub fn create_typed(
        &self,
        provider_type: ProviderType,
        config: Option<ProviderConfig>,
    ) -> Result<Arc<dyn GitProvider>> {
        let config = config.unwrap_or_else(|| self.default_config(provider_type));

        match provider_type {
            ProviderType::GitHub => {
                let provider = GitHubProvider::new(config)?;
                Ok(Arc::new(provider))
            }
            ProviderType::GitLab => {
                let provider = GitLabProvider::new(config)?;
                Ok(Arc::new(provider))
            }
            ProviderType::Bitbucket => {
                let provider = BitbucketProvider::new(config)?;
                Ok(Arc::new(provider))
            }
            ProviderType::Gitea => {
                if config.base_url.is_empty() {
                    return Err(ProviderError::ConfigurationError(
                        "Gitea provider requires a base_url".to_string(),
                    ));
                }
                let provider = GiteaProvider::new(config)?;
                Ok(Arc::new(provider))
            }
        }
    }

    /// Detect provider type from URL.
    pub fn detect_provider(url: &str) -> Result<String> {
        let url_lower = url.to_lowercase();

        if url_lower.contains("github.com") || url_lower.contains("github.") {
            return Ok("github".to_string());
        }
        if url_lower.contains("gitlab.com") || url_lower.contains("gitlab.") {
            return Ok("gitlab".to_string());
        }
        if url_lower.contains("bitbucket.org") || url_lower.contains("bitbucket.") {
            return Ok("bitbucket".to_string());
        }
        if url_lower.contains("gitea.") || url_lower.contains("/gitea/") {
            return Ok("gitea".to_string());
        }

        Err(ProviderError::UnsupportedProvider(format!(
            "Cannot detect provider from URL: {}",
            url
        )))
    }

    /// Parse owner/repo from a git URL.
    pub fn parse_repo_path(url: &str) -> Result<(String, String)> {
        let url = url.trim_end_matches('/').trim_end_matches(".git");

        // Handle SSH URLs (git@github.com:owner/repo)
        if let Some(pos) = url.find(':') {
            if url.starts_with("git@") {
                let path = &url[pos + 1..];
                return Self::split_owner_repo(path);
            }
        }

        // Handle HTTPS URLs
        if let Some(pos) = url.rfind('/') {
            let path = &url[..pos];
            if let Some(owner_pos) = path.rfind('/') {
                let owner = &path[owner_pos + 1..];
                let repo = &url[pos + 1..];
                return Ok((owner.to_string(), repo.to_string()));
            }
        }

        Err(ProviderError::ConfigurationError(format!(
            "Cannot parse owner/repo from URL: {}",
            url
        )))
    }

    fn split_owner_repo(path: &str) -> Result<(String, String)> {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() >= 2 {
            Ok((parts[0].to_string(), parts[1].to_string()))
        } else {
            Err(ProviderError::ConfigurationError(format!(
                "Invalid repository path: {}",
                path
            )))
        }
    }

    fn default_config(&self, provider_type: ProviderType) -> ProviderConfig {
        let base_url = provider_type.default_base_url().unwrap_or("").to_string();

        ProviderConfig {
            base_url,
            auth: self.default_auth.clone(),
            ..Default::default()
        }
    }
}

/// Create a provider from configuration.
///
/// This is a convenience function that matches the Python API.
pub fn create_provider(
    provider_type: &str,
    base_url: Option<&str>,
    token: Option<&str>,
    webhook_secret: Option<&str>,
) -> Result<Arc<dyn GitProvider>> {
    let provider_type_enum = ProviderType::parse(provider_type)
        .ok_or_else(|| ProviderError::UnsupportedProvider(provider_type.to_string()))?;

    let base_url = base_url
        .map(|s| s.to_string())
        .or_else(|| provider_type_enum.default_base_url().map(|s| s.to_string()))
        .unwrap_or_default();

    let auth = token.map(ProviderAuth::token);

    let config = ProviderConfig {
        base_url,
        auth,
        webhook_secret: webhook_secret.map(|s| s.to_string()),
        ..Default::default()
    };

    let factory = ProviderFactory::new();
    factory.create_typed(provider_type_enum, Some(config))
}

/// Parse a repository URL to extract provider type, owner, and name.
///
/// Supports all git URL formats via gix-url:
/// - `https://github.com/owner/repo`
/// - `git@github.com:owner/repo.git`
/// - `ssh://git@github.com/owner/repo.git`
/// - `git://github.com/owner/repo`
pub fn parse_repo_url(url: &str) -> Result<(ProviderType, String, String, Option<String>)> {
    let parsed = gix_url::Url::try_from(url)
        .map_err(|e| ProviderError::ConfigurationError(format!("Invalid git URL: {}", e)))?;

    let host = parsed
        .host()
        .ok_or_else(|| ProviderError::ConfigurationError("Missing host in URL".to_string()))?
        .to_string();

    let provider_type = match host.as_str() {
        "github.com" | "www.github.com" => ProviderType::GitHub,
        "gitlab.com" | "www.gitlab.com" => ProviderType::GitLab,
        "bitbucket.org" | "www.bitbucket.org" => ProviderType::Bitbucket,
        _ => ProviderType::Gitea, // Assume self-hosted Gitea for unknown hosts
    };

    let path = parsed
        .path
        .to_string()
        .trim_start_matches('/')
        .trim_end_matches(".git")
        .to_string();

    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 2 {
        return Err(ProviderError::ConfigurationError(
            "URL must contain owner/repo".to_string(),
        ));
    }

    let owner = parts[0].to_string();
    let name = parts[1].to_string();

    let base_url = if provider_type == ProviderType::Gitea {
        Some(format!("https://{}", host))
    } else {
        None
    };

    Ok((provider_type, owner, name, base_url))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_github() {
        let result = ProviderFactory::detect_provider("https://github.com/rust-lang/rust");
        assert_eq!(result.unwrap(), "github");
    }

    #[test]
    fn test_detect_gitlab() {
        let result = ProviderFactory::detect_provider("https://gitlab.com/group/project");
        assert_eq!(result.unwrap(), "gitlab");
    }

    #[test]
    fn test_detect_bitbucket() {
        let result = ProviderFactory::detect_provider("https://bitbucket.org/team/repo");
        assert_eq!(result.unwrap(), "bitbucket");
    }

    #[test]
    fn test_detect_gitea() {
        let result = ProviderFactory::detect_provider("https://gitea.example.com/user/repo");
        assert_eq!(result.unwrap(), "gitea");
    }

    #[test]
    fn test_detect_unknown() {
        let result = ProviderFactory::detect_provider("https://unknown.com/repo");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_https_url() {
        let (owner, repo) =
            ProviderFactory::parse_repo_path("https://github.com/rust-lang/rust.git").unwrap();
        assert_eq!(owner, "rust-lang");
        assert_eq!(repo, "rust");
    }

    #[test]
    fn test_parse_ssh_url() {
        let (owner, repo) =
            ProviderFactory::parse_repo_path("git@github.com:rust-lang/rust.git").unwrap();
        assert_eq!(owner, "rust-lang");
        assert_eq!(repo, "rust");
    }

    #[test]
    fn test_parse_url_without_git_suffix() {
        let (owner, repo) =
            ProviderFactory::parse_repo_path("https://github.com/owner/repo").unwrap();
        assert_eq!(owner, "owner");
        assert_eq!(repo, "repo");
    }

    #[test]
    fn test_create_github_provider() {
        let factory = ProviderFactory::new();
        let provider = factory.create("github", None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "github");
    }

    #[test]
    fn test_create_gitlab_provider() {
        let factory = ProviderFactory::new();
        let provider = factory.create("gitlab", None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "gitlab");
    }

    #[test]
    fn test_create_bitbucket_provider() {
        let factory = ProviderFactory::new();
        let provider = factory.create("bitbucket", None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "bitbucket");
    }

    #[test]
    fn test_create_gitea_provider_without_url() {
        let factory = ProviderFactory::new();
        let provider = factory.create("gitea", None);
        assert!(provider.is_err());
    }

    #[test]
    fn test_create_gitea_provider_with_url() {
        let factory = ProviderFactory::new();
        let config = ProviderConfig {
            base_url: "https://gitea.example.com".to_string(),
            ..Default::default()
        };
        let provider = factory.create("gitea", Some(config));
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "gitea");
    }

    #[test]
    fn test_create_unknown_provider() {
        let factory = ProviderFactory::new();
        let provider = factory.create("unknown", None);
        assert!(provider.is_err());
    }

    #[test]
    fn test_from_url() {
        let factory = ProviderFactory::new();
        let provider = factory.from_url("https://github.com/rust-lang/rust");
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "github");
    }

    #[test]
    fn test_from_url_gitlab() {
        let factory = ProviderFactory::new();
        let provider = factory.from_url("https://gitlab.com/group/project");
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "gitlab");
    }

    #[test]
    fn test_provider_type_from_str() {
        assert_eq!(ProviderType::parse("github"), Some(ProviderType::GitHub));
        assert_eq!(
            ProviderType::parse("github.com"),
            Some(ProviderType::GitHub)
        );
        assert_eq!(ProviderType::parse("gitlab"), Some(ProviderType::GitLab));
        assert_eq!(
            ProviderType::parse("bitbucket"),
            Some(ProviderType::Bitbucket)
        );
        assert_eq!(ProviderType::parse("gitea"), Some(ProviderType::Gitea));
        assert_eq!(ProviderType::parse("unknown"), None);
    }

    #[test]
    fn test_provider_type_default_url() {
        assert_eq!(
            ProviderType::GitHub.default_base_url(),
            Some("https://api.github.com")
        );
        assert_eq!(
            ProviderType::GitLab.default_base_url(),
            Some("https://gitlab.com/api/v4")
        );
        assert_eq!(
            ProviderType::Bitbucket.default_base_url(),
            Some("https://api.bitbucket.org/2.0")
        );
        assert_eq!(ProviderType::Gitea.default_base_url(), None);
    }

    #[test]
    fn test_factory_with_default_auth() {
        let factory = ProviderFactory::new().with_default_auth(ProviderAuth::token("my-token"));
        let provider = factory.create("github", None).unwrap();
        assert!(provider.auth().is_some());
    }

    #[test]
    fn test_create_provider_function() {
        let provider = create_provider("github", None, Some("token"), None);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "github");
    }

    #[test]
    fn test_create_provider_with_custom_url() {
        let provider = create_provider(
            "github",
            Some("https://github.example.com/api/v3"),
            Some("token"),
            None,
        );
        assert!(provider.is_ok());
        let p = provider.unwrap();
        assert_eq!(p.provider_name(), "github");
        assert_eq!(p.base_url(), "https://github.example.com/api/v3");
    }

    #[test]
    fn test_parse_repo_url_ssh_github() {
        let result = parse_repo_url("git@github.com:rust-lang/rust.git");
        assert!(result.is_ok());

        let (provider, owner, name, base_url) = result.unwrap();
        assert_eq!(provider, ProviderType::GitHub);
        assert_eq!(owner, "rust-lang");
        assert_eq!(name, "rust");
        assert!(base_url.is_none());
    }

    #[test]
    fn test_parse_repo_url_ssh_gitlab() {
        let result = parse_repo_url("git@gitlab.com:group/project.git");
        assert!(result.is_ok());

        let (provider, owner, name, _) = result.unwrap();
        assert_eq!(provider, ProviderType::GitLab);
        assert_eq!(owner, "group");
        assert_eq!(name, "project");
    }

    #[test]
    fn test_parse_repo_url_ssh_bitbucket() {
        let result = parse_repo_url("git@bitbucket.org:team/repo.git");
        assert!(result.is_ok());

        let (provider, owner, name, _) = result.unwrap();
        assert_eq!(provider, ProviderType::Bitbucket);
        assert_eq!(owner, "team");
        assert_eq!(name, "repo");
    }

    #[test]
    fn test_parse_repo_url_ssh_self_hosted() {
        let result = parse_repo_url("git@git.example.com:user/project.git");
        assert!(result.is_ok());

        let (provider, owner, name, base_url) = result.unwrap();
        assert_eq!(provider, ProviderType::Gitea);
        assert_eq!(owner, "user");
        assert_eq!(name, "project");
        assert_eq!(base_url, Some("https://git.example.com".to_string()));
    }

    #[test]
    fn test_parse_repo_url_ssh_without_git_suffix() {
        let result = parse_repo_url("git@github.com:owner/repo");
        assert!(result.is_ok());

        let (_, owner, name, _) = result.unwrap();
        assert_eq!(owner, "owner");
        assert_eq!(name, "repo");
    }

    #[test]
    fn test_gix_url_exploration() {
        use gix_url::Url;

        // Test various URL formats that gix should support
        let test_cases = [
            (
                "https://github.com/owner/repo.git",
                "github.com",
                "owner/repo",
            ),
            ("git@github.com:owner/repo.git", "github.com", "owner/repo"),
            (
                "ssh://git@github.com/owner/repo.git",
                "github.com",
                "owner/repo",
            ),
        ];

        for (url_str, expected_host, expected_path) in test_cases {
            let url =
                Url::try_from(url_str).unwrap_or_else(|_| panic!("Failed to parse: {}", url_str));
            assert_eq!(
                url.host().map(|h| h.to_string()),
                Some(expected_host.to_string()),
                "Host mismatch for {}",
                url_str
            );
            let path = url
                .path
                .to_string()
                .trim_start_matches('/')
                .trim_end_matches(".git")
                .to_string();
            assert_eq!(path, expected_path, "Path mismatch for {}", url_str);
        }
    }
}
