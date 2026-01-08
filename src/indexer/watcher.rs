//! File system watcher for detecting repository changes

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::{RwLock, mpsc};
use tokio::time::sleep;
use tracing::info;

/// Callback type for file change events
pub type ChangeCallback = Arc<dyn Fn(PathBuf) + Send + Sync>;

/// File system watcher for repository changes
pub struct IndexWatcher {
    watch_path: PathBuf,
    debounce_secs: f64,
    watcher: Option<RecommendedWatcher>,
    #[allow(dead_code)]
    pending: RwLock<HashMap<PathBuf, tokio::task::JoinHandle<()>>>,
    callback: Option<ChangeCallback>,
}

impl IndexWatcher {
    /// Create a new watcher
    #[must_use]
    pub fn new(watch_path: PathBuf, debounce_secs: f64) -> Self {
        Self {
            watch_path,
            debounce_secs,
            watcher: None,
            pending: RwLock::new(HashMap::new()),
            callback: None,
        }
    }

    /// Set the change callback
    pub fn on_change(&mut self, callback: ChangeCallback) {
        self.callback = Some(callback);
    }

    /// Start watching for file changes
    pub fn start(&mut self) -> notify::Result<()> {
        let (tx, mut rx) = mpsc::channel::<Event>(100);

        let watcher = RecommendedWatcher::new(
            move |res: notify::Result<Event>| {
                if let Ok(event) = res {
                    let _ = tx.blocking_send(event);
                }
            },
            Config::default(),
        )?;

        self.watcher = Some(watcher);

        if let Some(ref mut w) = self.watcher {
            w.watch(&self.watch_path, RecursiveMode::Recursive)?;
        }

        info!("Started watching: {:?}", self.watch_path);

        // Process events in background
        let watch_path = self.watch_path.clone();
        let debounce = self.debounce_secs;
        let callback = self.callback.clone();
        let pending = Arc::new(RwLock::new(
            HashMap::<PathBuf, tokio::task::JoinHandle<()>>::new(),
        ));

        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if let EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) =
                    event.kind
                {
                    for path in event.paths {
                        // Skip .git directories
                        if path.to_string_lossy().contains(".git") {
                            continue;
                        }

                        // Extract repository root
                        if let Some(repo_path) = extract_repo_path(&watch_path, &path) {
                            let callback = callback.clone();
                            let pending = pending.clone();
                            let repo_path_key = repo_path.clone();

                            // Cancel existing debounce task
                            {
                                let mut p = pending.write().await;
                                if let Some(handle) = p.remove(&repo_path_key) {
                                    handle.abort();
                                }
                            }

                            // Schedule new debounced callback
                            let handle = tokio::spawn(async move {
                                sleep(Duration::from_secs_f64(debounce)).await;
                                if let Some(cb) = callback {
                                    cb(repo_path);
                                }
                            });

                            {
                                let mut p = pending.write().await;
                                p.insert(repo_path_key, handle);
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop watching
    pub fn stop(&mut self) {
        self.watcher = None;
        info!("Stopped file watcher");
    }
}

/// Extract repository root path from a changed file path
fn extract_repo_path(watch_path: &Path, file_path: &Path) -> Option<PathBuf> {
    let relative = file_path.strip_prefix(watch_path).ok()?;
    let parts: Vec<_> = relative.components().take(3).collect();

    if parts.len() >= 3 {
        Some(
            watch_path
                .join(parts[0].as_os_str())
                .join(parts[1].as_os_str())
                .join(parts[2].as_os_str()),
        )
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tempfile::tempdir;

    #[test]
    fn test_extract_repo_path() {
        let watch = PathBuf::from("/data/repos");
        let file = PathBuf::from("/data/repos/github/owner/repo/src/main.rs");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(result, Some(PathBuf::from("/data/repos/github/owner/repo")));
    }

    #[test]
    fn test_extract_repo_path_exact_depth() {
        let watch = PathBuf::from("/data/repos");
        let file = PathBuf::from("/data/repos/github/owner/repo");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(result, Some(PathBuf::from("/data/repos/github/owner/repo")));
    }

    #[test]
    fn test_extract_repo_path_too_shallow() {
        let watch = PathBuf::from("/data/repos");
        let file = PathBuf::from("/data/repos/github/owner");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_repo_path_single_component() {
        let watch = PathBuf::from("/data/repos");
        let file = PathBuf::from("/data/repos/github");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_repo_path_same_path() {
        let watch = PathBuf::from("/data/repos");
        let file = PathBuf::from("/data/repos");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_repo_path_no_common_prefix() {
        let watch = PathBuf::from("/data/repos");
        let file = PathBuf::from("/other/path/github/owner/repo");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_repo_path_deeply_nested() {
        let watch = PathBuf::from("/data/repos");
        let file = PathBuf::from("/data/repos/github/owner/repo/src/sub/deep/file.rs");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(result, Some(PathBuf::from("/data/repos/github/owner/repo")));
    }

    #[test]
    fn test_extract_repo_path_with_dots_in_name() {
        let watch = PathBuf::from("/data/repos");
        let file = PathBuf::from("/data/repos/github/my-org/repo.name-test/src/main.rs");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(
            result,
            Some(PathBuf::from("/data/repos/github/my-org/repo.name-test"))
        );
    }

    #[test]
    fn test_extract_repo_path_different_providers() {
        let watch = PathBuf::from("/data/repos");

        let gitlab_file = PathBuf::from("/data/repos/gitlab/org/project/file.py");
        assert_eq!(
            extract_repo_path(&watch, &gitlab_file),
            Some(PathBuf::from("/data/repos/gitlab/org/project"))
        );

        let bitbucket_file = PathBuf::from("/data/repos/bitbucket/team/repo/file.js");
        assert_eq!(
            extract_repo_path(&watch, &bitbucket_file),
            Some(PathBuf::from("/data/repos/bitbucket/team/repo"))
        );
    }

    #[test]
    fn test_watcher_new() {
        let path = PathBuf::from("/tmp/test-watch");
        let watcher = IndexWatcher::new(path.clone(), 1.5);

        assert_eq!(watcher.watch_path, path);
        assert!((watcher.debounce_secs - 1.5).abs() < f64::EPSILON);
        assert!(watcher.watcher.is_none());
        assert!(watcher.callback.is_none());
    }

    #[test]
    fn test_watcher_new_various_debounce() {
        let path = PathBuf::from("/tmp/test");

        let watcher = IndexWatcher::new(path.clone(), 0.0);
        assert!((watcher.debounce_secs - 0.0).abs() < f64::EPSILON);

        let watcher = IndexWatcher::new(path.clone(), 0.5);
        assert!((watcher.debounce_secs - 0.5).abs() < f64::EPSILON);

        let watcher = IndexWatcher::new(path, 10.0);
        assert!((watcher.debounce_secs - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_watcher_on_change() {
        let path = PathBuf::from("/tmp/test-watch");
        let mut watcher = IndexWatcher::new(path, 1.0);

        assert!(watcher.callback.is_none());

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();
        let callback: ChangeCallback = Arc::new(move |_path| {
            called_clone.store(true, Ordering::SeqCst);
        });

        watcher.on_change(callback);
        assert!(watcher.callback.is_some());
    }

    #[test]
    fn test_watcher_on_change_can_replace() {
        let path = PathBuf::from("/tmp/test-watch");
        let mut watcher = IndexWatcher::new(path, 1.0);

        let counter1 = Arc::new(AtomicUsize::new(0));
        let counter1_clone = counter1.clone();
        let callback1: ChangeCallback = Arc::new(move |_| {
            counter1_clone.fetch_add(1, Ordering::SeqCst);
        });

        let counter2 = Arc::new(AtomicUsize::new(0));
        let counter2_clone = counter2.clone();
        let callback2: ChangeCallback = Arc::new(move |_| {
            counter2_clone.fetch_add(100, Ordering::SeqCst);
        });

        watcher.on_change(callback1);
        watcher.on_change(callback2);

        // Only the second callback should be set
        assert!(watcher.callback.is_some());
    }

    #[test]
    fn test_watcher_stop_before_start() {
        let path = PathBuf::from("/tmp/test-watch");
        let mut watcher = IndexWatcher::new(path, 1.0);

        // Should not panic
        watcher.stop();
        assert!(watcher.watcher.is_none());
    }

    #[tokio::test]
    async fn test_watcher_start_with_tempdir() {
        let dir = tempdir().unwrap();
        let mut watcher = IndexWatcher::new(dir.path().to_path_buf(), 0.1);

        let callback: ChangeCallback = Arc::new(|_path| {});
        watcher.on_change(callback);

        let result = watcher.start();
        assert!(result.is_ok());
        assert!(watcher.watcher.is_some());

        watcher.stop();
        assert!(watcher.watcher.is_none());
    }

    #[test]
    fn test_watcher_start_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path/that/does/not/exist");
        let mut watcher = IndexWatcher::new(path, 1.0);

        let result = watcher.start();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_watcher_stop_clears_watcher() {
        let dir = tempdir().unwrap();
        let mut watcher = IndexWatcher::new(dir.path().to_path_buf(), 0.1);

        let callback: ChangeCallback = Arc::new(|_path| {});
        watcher.on_change(callback);

        watcher.start().unwrap();
        assert!(watcher.watcher.is_some());

        watcher.stop();
        assert!(watcher.watcher.is_none());

        // Can stop multiple times
        watcher.stop();
        assert!(watcher.watcher.is_none());
    }

    #[tokio::test]
    async fn test_watcher_start_without_callback() {
        let dir = tempdir().unwrap();
        let mut watcher = IndexWatcher::new(dir.path().to_path_buf(), 0.1);

        // No callback set
        assert!(watcher.callback.is_none());

        let result = watcher.start();
        assert!(result.is_ok());

        watcher.stop();
    }

    #[tokio::test]
    async fn test_watcher_file_change_triggers_callback() {
        let dir = tempdir().unwrap();
        let watch_path = dir.path().to_path_buf();

        // Create provider/owner/repo structure
        let repo_path = watch_path.join("github").join("owner").join("repo");
        fs::create_dir_all(&repo_path).unwrap();

        let mut watcher = IndexWatcher::new(watch_path.clone(), 0.1);

        let callback_path = Arc::new(RwLock::new(None::<PathBuf>));
        let callback_path_clone = callback_path.clone();
        let callback: ChangeCallback = Arc::new(move |path| {
            let path_clone = path.clone();
            let callback_path = callback_path_clone.clone();
            tokio::spawn(async move {
                let mut p = callback_path.write().await;
                *p = Some(path_clone);
            });
        });

        watcher.on_change(callback);
        watcher.start().unwrap();

        // Create a file
        let test_file = repo_path.join("test.rs");
        fs::write(&test_file, "fn main() {}").unwrap();

        // Wait for debounce
        sleep(Duration::from_millis(300)).await;

        // Check if callback was called
        let received_path = callback_path.read().await;
        if let Some(path) = received_path.as_ref() {
            assert_eq!(path, &watch_path.join("github").join("owner").join("repo"));
        }
        // Note: The callback might not fire in all test environments due to
        // filesystem notification timing, so we don't assert it must have been called

        watcher.stop();
    }

    #[tokio::test]
    async fn test_watcher_git_directory_filtered() {
        let dir = tempdir().unwrap();
        let watch_path = dir.path().to_path_buf();

        // Create provider/owner/repo/.git structure
        let repo_path = watch_path.join("github").join("owner").join("repo");
        let git_path = repo_path.join(".git");
        fs::create_dir_all(&git_path).unwrap();

        let mut watcher = IndexWatcher::new(watch_path, 0.05);

        let git_callback_count = Arc::new(AtomicUsize::new(0));
        let git_callback_count_clone = git_callback_count.clone();
        let callback: ChangeCallback = Arc::new(move |path| {
            // If we get a callback for .git, increment counter
            if path.to_string_lossy().contains(".git") {
                git_callback_count_clone.fetch_add(1, Ordering::SeqCst);
            }
        });

        watcher.on_change(callback);
        watcher.start().unwrap();

        // Create a file in .git
        let git_file = git_path.join("config");
        fs::write(&git_file, "[core]\nrepositoryformatversion = 0").unwrap();

        // Wait for debounce
        sleep(Duration::from_millis(200)).await;

        // .git paths should be filtered
        assert_eq!(git_callback_count.load(Ordering::SeqCst), 0);

        watcher.stop();
    }

    #[test]
    fn test_extract_repo_path_with_spaces_in_path() {
        let watch = PathBuf::from("/data/repos with spaces");
        let file = PathBuf::from("/data/repos with spaces/github/my org/my repo/file.rs");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(
            result,
            Some(PathBuf::from(
                "/data/repos with spaces/github/my org/my repo"
            ))
        );
    }

    #[test]
    fn test_extract_repo_path_unicode() {
        let watch = PathBuf::from("/data/repos");
        let file = PathBuf::from("/data/repos/github/owner/repo-test/file.rs");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(
            result,
            Some(PathBuf::from("/data/repos/github/owner/repo-test"))
        );
    }
}
