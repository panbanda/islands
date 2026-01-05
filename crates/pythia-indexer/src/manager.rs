//! Repository manager for cloning and updating repositories

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use git2::{FetchOptions, RemoteCallbacks, Repository as GitRepo};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, warn};

use pythia_providers::{GitProvider, Repository};

use crate::error::{Error, Result};
use crate::state::RepositoryState;

/// Manages repository cloning, updating, and state tracking
pub struct RepositoryManager {
    /// Base path for repository storage
    storage_path: PathBuf,
    /// Configured git providers
    providers: HashMap<String, Arc<dyn GitProvider>>,
    /// Semaphore for limiting concurrent git operations
    semaphore: Semaphore,
    /// Repository states
    states: RwLock<HashMap<String, RepositoryState>>,
}

impl RepositoryManager {
    /// Create a new repository manager
    #[must_use]
    pub fn new(
        storage_path: PathBuf,
        providers: HashMap<String, Arc<dyn GitProvider>>,
        max_concurrent: usize,
    ) -> Self {
        Self {
            storage_path,
            providers,
            semaphore: Semaphore::new(max_concurrent),
            states: RwLock::new(HashMap::new()),
        }
    }

    /// Get the local path for a repository
    #[must_use]
    pub fn local_path(&self, repo: &Repository) -> PathBuf {
        self.storage_path
            .join(&repo.provider)
            .join(&repo.owner)
            .join(&repo.name)
    }

    /// Get the current state of a repository
    pub async fn get_state(&self, repo: &Repository) -> Option<RepositoryState> {
        let states = self.states.read().await;
        states.get(&repo.id()).cloned()
    }

    /// List all tracked repository states
    pub async fn list_states(&self) -> Vec<RepositoryState> {
        let states = self.states.read().await;
        states.values().cloned().collect()
    }

    /// Clone a repository to local storage
    pub async fn clone_repository(&self, repo: &Repository) -> Result<RepositoryState> {
        let _permit = self.semaphore.acquire().await.unwrap();

        let local_path = self.local_path(repo);
        let key = repo.id();

        info!("Cloning {} to {:?}", key, local_path);

        // Remove existing directory if present
        if local_path.exists() {
            std::fs::remove_dir_all(&local_path)?;
        }

        // Create parent directories
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Get clone URL from provider
        let provider = self
            .providers
            .get(&repo.provider)
            .ok_or_else(|| Error::Config(format!("unknown provider: {}", repo.provider)))?;

        let clone_url = provider.get_clone_url(repo);

        // Clone in a blocking task
        let local_path_clone = local_path.clone();
        let branch = repo.default_branch.clone();

        let commit =
            tokio::task::spawn_blocking(move || clone_repo(&clone_url, &local_path_clone, &branch))
                .await
                .map_err(|e| Error::CloneFailed(e.to_string()))??;

        let mut state = RepositoryState::new(repo.clone(), local_path);
        state.mark_synced(commit);

        // Store state
        let mut states = self.states.write().await;
        states.insert(key, state.clone());

        info!("Successfully cloned {}", repo.id());
        Ok(state)
    }

    /// Update a repository with latest changes
    pub async fn update_repository(&self, repo: &Repository) -> Result<RepositoryState> {
        let _permit = self.semaphore.acquire().await.unwrap();

        let local_path = self.local_path(repo);

        if !local_path.exists() {
            // Repository doesn't exist locally, clone it instead
            drop(_permit);
            return self.clone_repository(repo).await;
        }

        info!("Updating {}", repo.id());

        let local_path_clone = local_path.clone();
        let result = tokio::task::spawn_blocking(move || pull_repo(&local_path_clone))
            .await
            .map_err(|e| Error::Sync(e.to_string()))?;

        match result {
            Ok(commit) => {
                let mut states = self.states.write().await;
                let key = repo.id();

                if let Some(state) = states.get_mut(&key) {
                    let needs_reindex = state.last_commit.as_ref() != Some(&commit);
                    state.mark_synced(commit);
                    if needs_reindex {
                        state.indexed = false;
                    }
                    info!("Updated {}, needs_reindex={}", key, needs_reindex);
                    Ok(state.clone())
                } else {
                    let mut state = RepositoryState::new(repo.clone(), local_path);
                    state.mark_synced(commit);
                    states.insert(key, state.clone());
                    Ok(state)
                }
            }
            Err(e) => {
                error!("Failed to update {}: {}", repo.id(), e);
                let mut states = self.states.write().await;
                if let Some(state) = states.get_mut(&repo.id()) {
                    state.set_error(e.to_string());
                }
                Err(e)
            }
        }
    }

    /// Mark a repository as indexed
    pub async fn mark_indexed(&self, repo: &Repository) {
        let mut states = self.states.write().await;
        if let Some(state) = states.get_mut(&repo.id()) {
            state.mark_indexed();
        }
    }

    /// Check if a repository needs re-indexing
    pub async fn needs_reindex(&self, repo: &Repository) -> bool {
        let states = self.states.read().await;
        states
            .get(&repo.id())
            .map(RepositoryState::needs_reindex)
            .unwrap_or(true)
    }

    /// Remove a repository from local storage
    pub async fn remove_repository(&self, repo: &Repository) -> Result<()> {
        let local_path = self.local_path(repo);

        if local_path.exists() {
            std::fs::remove_dir_all(&local_path)?;
        }

        let mut states = self.states.write().await;
        states.remove(&repo.id());

        info!("Removed {}", repo.id());
        Ok(())
    }
}

/// Clone a repository (blocking)
fn clone_repo(url: &str, path: &Path, branch: &str) -> Result<String> {
    let mut builder = git2::build::RepoBuilder::new();
    builder.branch(branch);

    // Shallow clone
    let mut fetch_opts = FetchOptions::new();
    fetch_opts.depth(1);
    builder.fetch_options(fetch_opts);

    let repo = builder.clone(url, path)?;
    let head = repo.head()?;
    let commit = head.peel_to_commit()?;

    Ok(commit.id().to_string())
}

/// Pull latest changes from remote (blocking)
fn pull_repo(path: &Path) -> Result<String> {
    let repo = GitRepo::open(path)?;

    // Fetch from origin
    let mut remote = repo.find_remote("origin")?;

    let mut fetch_opts = FetchOptions::new();
    remote.fetch(&["HEAD"], Some(&mut fetch_opts), None)?;

    // Get the fetch head
    let fetch_head = repo.find_reference("FETCH_HEAD")?;
    let commit = fetch_head.peel_to_commit()?;

    // Fast-forward if possible
    let head_ref = repo.head()?;
    if head_ref.is_branch() {
        let mut reference = head_ref;
        reference.set_target(commit.id(), "fast-forward")?;
    }

    Ok(commit.id().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_repo() -> Repository {
        Repository::new(
            "github",
            "test-owner",
            "test-repo",
            "https://github.com/test-owner/test-repo.git",
        )
    }

    fn create_manager() -> RepositoryManager {
        RepositoryManager::new(PathBuf::from("/tmp/test-repos"), HashMap::new(), 4)
    }

    #[test]
    fn test_manager_new() {
        let manager = create_manager();

        assert_eq!(manager.storage_path, PathBuf::from("/tmp/test-repos"));
        assert!(manager.providers.is_empty());
    }

    #[test]
    fn test_manager_new_with_providers() {
        use pythia_providers::factory::create_provider;

        let mut providers: HashMap<String, Arc<dyn GitProvider>> = HashMap::new();
        let github = create_provider("github", None, None, None).unwrap();
        providers.insert("github".to_string(), github);

        let manager = RepositoryManager::new(
            PathBuf::from("/var/repos"),
            providers,
            8,
        );

        assert_eq!(manager.providers.len(), 1);
        assert!(manager.providers.contains_key("github"));
    }

    #[test]
    fn test_local_path() {
        let manager = create_manager();
        let repo = test_repo();

        let path = manager.local_path(&repo);

        assert_eq!(
            path,
            PathBuf::from("/tmp/test-repos/github/test-owner/test-repo")
        );
    }

    #[test]
    fn test_local_path_different_providers() {
        let manager = create_manager();

        let github_repo = Repository::new(
            "github",
            "owner",
            "repo",
            "https://github.com/owner/repo.git",
        );
        let gitlab_repo = Repository::new(
            "gitlab",
            "owner",
            "repo",
            "https://gitlab.com/owner/repo.git",
        );

        assert_eq!(
            manager.local_path(&github_repo),
            PathBuf::from("/tmp/test-repos/github/owner/repo")
        );
        assert_eq!(
            manager.local_path(&gitlab_repo),
            PathBuf::from("/tmp/test-repos/gitlab/owner/repo")
        );
    }

    #[test]
    fn test_local_path_special_characters() {
        let manager = create_manager();
        let repo = Repository::new(
            "github",
            "org-name",
            "repo.name-test",
            "https://github.com/org-name/repo.name-test.git",
        );

        let path = manager.local_path(&repo);

        assert_eq!(
            path,
            PathBuf::from("/tmp/test-repos/github/org-name/repo.name-test")
        );
    }

    #[tokio::test]
    async fn test_get_state_not_found() {
        let manager = create_manager();
        let repo = test_repo();

        let state = manager.get_state(&repo).await;

        assert!(state.is_none());
    }

    #[tokio::test]
    async fn test_list_states_empty() {
        let manager = create_manager();

        let states = manager.list_states().await;

        assert!(states.is_empty());
    }

    #[tokio::test]
    async fn test_needs_reindex_no_state() {
        let manager = create_manager();
        let repo = test_repo();

        // Repository with no state should need reindex
        assert!(manager.needs_reindex(&repo).await);
    }

    #[tokio::test]
    async fn test_mark_indexed_no_state() {
        let manager = create_manager();
        let repo = test_repo();

        // Should not panic even with no state
        manager.mark_indexed(&repo).await;

        // Still no state since mark_indexed doesn't create one
        assert!(manager.get_state(&repo).await.is_none());
    }

    #[tokio::test]
    async fn test_remove_repository_nonexistent() {
        let manager = create_manager();
        let repo = test_repo();

        // Should succeed even if repo doesn't exist
        let result = manager.remove_repository(&repo).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_clone_repository_no_provider() {
        let manager = create_manager();
        let repo = test_repo();

        let result = manager.clone_repository(&repo).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unknown provider"));
    }

    #[tokio::test]
    async fn test_state_management_lifecycle() {
        let manager = create_manager();

        // Insert a state manually for testing state management
        {
            let mut states = manager.states.write().await;
            let repo = test_repo();
            let mut state = RepositoryState::new(repo.clone(), PathBuf::from("/tmp/test"));
            state.mark_synced("abc123".to_string());
            states.insert(repo.id(), state);
        }

        let repo = test_repo();

        // Get state should now return the state
        let state = manager.get_state(&repo).await;
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.last_commit, Some("abc123".to_string()));
        assert!(!state.indexed);

        // List states should include it
        let states = manager.list_states().await;
        assert_eq!(states.len(), 1);

        // needs_reindex should return true (not indexed)
        assert!(manager.needs_reindex(&repo).await);

        // Mark as indexed
        manager.mark_indexed(&repo).await;

        // Now needs_reindex should return false
        assert!(!manager.needs_reindex(&repo).await);

        // Verify state was updated
        let state = manager.get_state(&repo).await.unwrap();
        assert!(state.indexed);
        assert!(state.indexed_at.is_some());
    }

    #[tokio::test]
    async fn test_concurrent_state_access() {
        let manager = Arc::new(create_manager());

        // Simulate concurrent access
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let manager = Arc::clone(&manager);
                tokio::spawn(async move {
                    let repo = Repository::new(
                        "github",
                        "owner",
                        &format!("repo-{}", i),
                        &format!("https://github.com/owner/repo-{}.git", i),
                    );
                    manager.needs_reindex(&repo).await
                })
            })
            .collect();

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result); // All should need reindex
        }
    }

    #[test]
    fn test_semaphore_permits() {
        let manager = RepositoryManager::new(
            PathBuf::from("/tmp"),
            HashMap::new(),
            3,
        );

        // Check that semaphore has correct number of permits
        assert_eq!(manager.semaphore.available_permits(), 3);
    }
}
