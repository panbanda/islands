//! Authentication types for git providers

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Authentication type for git providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuthType {
    /// Personal access token
    Token,
    /// SSH key authentication
    Ssh,
    /// OAuth authentication
    OAuth,
    /// Basic username/password
    Basic,
}

impl Default for AuthType {
    fn default() -> Self {
        Self::Token
    }
}

/// Authentication credentials for a git provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderAuth {
    /// Type of authentication
    pub auth_type: AuthType,
    /// Personal access token (for Token auth)
    #[serde(skip_serializing)]
    pub token: Option<String>,
    /// Username (for Basic auth)
    pub username: Option<String>,
    /// Password (for Basic auth)
    #[serde(skip_serializing)]
    pub password: Option<String>,
    /// Path to SSH private key
    pub ssh_key_path: Option<PathBuf>,
    /// OAuth client ID
    pub oauth_client_id: Option<String>,
    /// OAuth client secret
    #[serde(skip_serializing)]
    pub oauth_client_secret: Option<String>,
}

impl ProviderAuth {
    /// Create token-based authentication
    #[must_use]
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

    /// Create SSH key authentication
    #[must_use]
    pub fn ssh(key_path: PathBuf) -> Self {
        Self {
            auth_type: AuthType::Ssh,
            token: None,
            username: None,
            password: None,
            ssh_key_path: Some(key_path),
            oauth_client_id: None,
            oauth_client_secret: None,
        }
    }

    /// Create basic authentication
    #[must_use]
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
}
