//! Repository state tracking

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::providers::Repository;

/// Tracks the state of a managed repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryState {
    /// The repository metadata
    pub repository: Repository,
    /// Local filesystem path
    pub local_path: PathBuf,
    /// Last known commit SHA
    pub last_commit: Option<String>,
    /// Timestamp of last sync
    pub last_synced: Option<DateTime<Utc>>,
    /// Whether the repository has been indexed
    pub indexed: bool,
    /// Timestamp of last indexing
    pub indexed_at: Option<DateTime<Utc>>,
    /// Error message if any
    pub error: Option<String>,
}

impl RepositoryState {
    /// Create a new repository state
    #[must_use]
    pub fn new(repository: Repository, local_path: PathBuf) -> Self {
        Self {
            repository,
            local_path,
            last_commit: None,
            last_synced: None,
            indexed: false,
            indexed_at: None,
            error: None,
        }
    }

    /// Get the unique key for this repository
    #[must_use]
    pub fn key(&self) -> String {
        self.repository.id()
    }

    /// Check if the repository needs re-indexing
    #[must_use]
    pub fn needs_reindex(&self) -> bool {
        !self.indexed || self.error.is_some()
    }

    /// Mark as synced with the given commit
    pub fn mark_synced(&mut self, commit: String) {
        self.last_commit = Some(commit);
        self.last_synced = Some(Utc::now());
        self.error = None;
    }

    /// Mark as indexed
    pub fn mark_indexed(&mut self) {
        self.indexed = true;
        self.indexed_at = Some(Utc::now());
        self.error = None;
    }

    /// Set error state
    pub fn set_error(&mut self, error: impl Into<String>) {
        self.error = Some(error.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_repo() -> Repository {
        Repository::new("github", "test", "repo", "https://github.com/test/repo.git")
    }

    #[test]
    fn test_repository_state_new() {
        let repo = test_repo();
        let state = RepositoryState::new(repo.clone(), PathBuf::from("/tmp/repo"));

        assert_eq!(state.repository.id(), "test/repo");
        assert_eq!(state.local_path, PathBuf::from("/tmp/repo"));
        assert!(state.last_commit.is_none());
        assert!(state.last_synced.is_none());
        assert!(!state.indexed);
        assert!(state.indexed_at.is_none());
        assert!(state.error.is_none());
    }

    #[test]
    fn test_repository_state_key() {
        let repo = test_repo();
        let state = RepositoryState::new(repo, PathBuf::from("/tmp/repo"));

        assert_eq!(state.key(), "test/repo");
    }

    #[test]
    fn test_repository_state_needs_reindex_new() {
        let repo = test_repo();
        let state = RepositoryState::new(repo, PathBuf::from("/tmp/repo"));

        // New state should need re-indexing
        assert!(state.needs_reindex());
    }

    #[test]
    fn test_repository_state_needs_reindex_indexed() {
        let repo = test_repo();
        let mut state = RepositoryState::new(repo, PathBuf::from("/tmp/repo"));
        state.mark_indexed();

        // Indexed state should not need re-indexing
        assert!(!state.needs_reindex());
    }

    #[test]
    fn test_repository_state_needs_reindex_with_error() {
        let repo = test_repo();
        let mut state = RepositoryState::new(repo, PathBuf::from("/tmp/repo"));
        state.mark_indexed();
        state.set_error("some error");

        // State with error should need re-indexing even if indexed
        assert!(state.needs_reindex());
    }

    #[test]
    fn test_repository_state_mark_synced() {
        let repo = test_repo();
        let mut state = RepositoryState::new(repo, PathBuf::from("/tmp/repo"));
        state.set_error("previous error");

        let commit = "abc123def456".to_string();
        state.mark_synced(commit.clone());

        assert_eq!(state.last_commit, Some(commit));
        assert!(state.last_synced.is_some());
        assert!(state.error.is_none()); // Error cleared
    }

    #[test]
    fn test_repository_state_mark_indexed() {
        let repo = test_repo();
        let mut state = RepositoryState::new(repo, PathBuf::from("/tmp/repo"));
        state.set_error("previous error");

        state.mark_indexed();

        assert!(state.indexed);
        assert!(state.indexed_at.is_some());
        assert!(state.error.is_none()); // Error cleared
    }

    #[test]
    fn test_repository_state_set_error() {
        let repo = test_repo();
        let mut state = RepositoryState::new(repo, PathBuf::from("/tmp/repo"));

        state.set_error("clone failed");
        assert_eq!(state.error, Some("clone failed".to_string()));

        state.set_error(String::from("network error"));
        assert_eq!(state.error, Some("network error".to_string()));
    }

    #[test]
    fn test_repository_state_serialization() {
        let repo = test_repo();
        let mut state = RepositoryState::new(repo, PathBuf::from("/tmp/repo"));
        state.mark_synced("abc123".to_string());
        state.mark_indexed();

        let json = serde_json::to_string(&state).unwrap();
        let parsed: RepositoryState = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.key(), state.key());
        assert_eq!(parsed.last_commit, state.last_commit);
        assert_eq!(parsed.indexed, state.indexed);
    }
}
