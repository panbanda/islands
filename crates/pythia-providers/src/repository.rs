//! Repository data structures

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Represents a git repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Repository {
    /// Provider name (github, gitlab, etc.)
    pub provider: String,
    /// Repository owner (user or organization)
    pub owner: String,
    /// Repository name
    pub name: String,
    /// Full name (owner/name)
    pub full_name: String,
    /// HTTPS clone URL
    pub clone_url: String,
    /// SSH clone URL
    pub ssh_url: Option<String>,
    /// Default branch name
    pub default_branch: String,
    /// Repository description
    pub description: Option<String>,
    /// Primary language
    pub language: Option<String>,
    /// Size in kilobytes
    pub size_kb: u64,
    /// Last update timestamp
    pub last_updated: Option<DateTime<Utc>>,
    /// Whether the repository is private
    pub is_private: bool,
    /// Repository topics/tags
    pub topics: Vec<String>,
}

impl Repository {
    /// Create a new repository with minimal required fields
    #[must_use]
    pub fn new(provider: &str, owner: &str, name: &str, clone_url: &str) -> Self {
        Self {
            provider: provider.to_string(),
            owner: owner.to_string(),
            name: name.to_string(),
            full_name: format!("{owner}/{name}"),
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

    /// Get the local path where this repo would be cloned
    #[must_use]
    pub fn local_path(&self, base: &std::path::Path) -> PathBuf {
        base.join(&self.provider).join(&self.owner).join(&self.name)
    }

    /// Get the unique identifier for this repository
    #[must_use]
    pub fn id(&self) -> String {
        format!("{}/{}", self.provider, self.full_name)
    }
}

/// Visibility filter for listing repositories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    /// All repositories
    All,
    /// Public repositories only
    Public,
    /// Private repositories only
    Private,
}

impl Default for Visibility {
    fn default() -> Self {
        Self::All
    }
}
