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

    #[test]
    fn test_extract_repo_path() {
        let watch = PathBuf::from("/data/repos");
        let file = PathBuf::from("/data/repos/github/owner/repo/src/main.rs");

        let result = extract_repo_path(&watch, &file);
        assert_eq!(result, Some(PathBuf::from("/data/repos/github/owner/repo")));
    }
}
