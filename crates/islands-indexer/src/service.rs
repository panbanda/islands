//! Main indexer service

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::time::{Duration, interval};
use tracing::{error, info};

use islands_providers::{GitProvider, Repository, WebhookEvent};

#[cfg(feature = "embeddings")]
use islands_core::embedding::{CloudProvider, EmbedderConfig, EmbedderProvider, InferenceBackend};

use crate::error::{Error, Result};
use crate::manager::RepositoryManager;
use crate::state::RepositoryState;

/// Information about a LEANN index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    /// Index name (provider/owner/repo)
    pub name: String,
    /// Path to the index file
    pub path: PathBuf,
    /// Source repository
    pub repository: Repository,
    /// When the index was created
    pub created_at: DateTime<Utc>,
    /// When the index was last updated
    pub updated_at: DateTime<Utc>,
    /// Number of indexed files
    pub file_count: usize,
    /// Index size in bytes
    pub size_bytes: u64,
}

/// Configuration for the indexer service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerConfig {
    /// Base path for repository storage
    pub repos_path: PathBuf,
    /// Base path for index storage
    pub indexes_path: PathBuf,
    /// Maximum concurrent git operations
    pub max_concurrent_syncs: usize,
    /// Sync interval in seconds
    pub sync_interval_secs: u64,
    /// File extensions to index
    pub index_extensions: Vec<String>,
    /// Embedding configuration
    #[cfg(feature = "embeddings")]
    #[serde(default)]
    pub embedding: EmbeddingConfig,
}

/// Configuration for embedding model
#[cfg(feature = "embeddings")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum EmbeddingConfig {
    /// Use a preset local model (bge-small, bge-base, bge-large, minilm, jina-small, nomic)
    Local {
        /// Preset name or HuggingFace model ID
        #[serde(default = "default_local_model")]
        model: String,
        /// Batch size for processing
        #[serde(default = "default_batch_size")]
        batch_size: usize,
    },
    /// Use OpenAI embeddings API
    OpenAI {
        /// Model name (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
        #[serde(default = "default_openai_model")]
        model: String,
        /// API key (defaults to OPENAI_API_KEY env var)
        api_key: Option<String>,
        /// Batch size for API calls
        #[serde(default = "default_batch_size")]
        batch_size: usize,
    },
    /// Use Cohere embeddings API
    Cohere {
        /// Model name (embed-english-v3.0, embed-multilingual-v3.0)
        #[serde(default = "default_cohere_model")]
        model: String,
        /// API key (defaults to COHERE_API_KEY env var)
        api_key: Option<String>,
        /// Batch size for API calls
        #[serde(default = "default_batch_size")]
        batch_size: usize,
    },
}

#[cfg(feature = "embeddings")]
impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self::Local {
            model: default_local_model(),
            batch_size: default_batch_size(),
        }
    }
}

#[cfg(feature = "embeddings")]
fn default_local_model() -> String {
    "bge-small".to_string()
}

#[cfg(feature = "embeddings")]
fn default_openai_model() -> String {
    "text-embedding-3-small".to_string()
}

#[cfg(feature = "embeddings")]
fn default_cohere_model() -> String {
    "embed-english-v3.0".to_string()
}

#[cfg(feature = "embeddings")]
fn default_batch_size() -> usize {
    32
}

#[cfg(feature = "embeddings")]
impl EmbeddingConfig {
    /// Get the batch size from any variant
    pub fn batch_size(&self) -> usize {
        match self {
            Self::Local { batch_size, .. }
            | Self::OpenAI { batch_size, .. }
            | Self::Cohere { batch_size, .. } => *batch_size,
        }
    }
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            repos_path: PathBuf::from("/data/islands/repos"),
            indexes_path: PathBuf::from("/data/islands/indexes"),
            max_concurrent_syncs: 4,
            sync_interval_secs: 300,
            index_extensions: vec![
                "py", "js", "ts", "jsx", "tsx", "java", "go", "rs", "c", "cpp", "h", "hpp", "cs",
                "rb", "php", "swift", "kt", "scala", "sql", "sh", "bash", "yaml", "yml", "json",
                "toml", "md", "rst", "txt",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            #[cfg(feature = "embeddings")]
            embedding: EmbeddingConfig::default(),
        }
    }
}

/// Stored index with HNSW graph and file metadata
pub struct StoredIndex {
    /// HNSW graph for vector search
    pub graph: islands_core::HnswGraph,
    /// File paths and content indexed (id -> (path, content))
    pub files: Vec<(String, String)>,
    /// Index metadata
    pub info: IndexInfo,
}

/// Main service for indexing repositories with LEANN
pub struct IndexerService {
    config: IndexerConfig,
    repo_manager: RepositoryManager,
    indexes: RwLock<HashMap<String, IndexInfo>>,
    /// Stored HNSW graphs with file metadata
    graphs: RwLock<HashMap<String, StoredIndex>>,
    running: RwLock<bool>,
    #[cfg(feature = "embeddings")]
    embedder: Option<Arc<EmbedderProvider>>,
}

impl IndexerService {
    /// Create a new indexer service
    pub fn new(config: IndexerConfig, providers: HashMap<String, Arc<dyn GitProvider>>) -> Self {
        // Ensure directories exist
        std::fs::create_dir_all(&config.repos_path).ok();
        std::fs::create_dir_all(&config.indexes_path).ok();

        let repo_manager = RepositoryManager::new(
            config.repos_path.clone(),
            providers,
            config.max_concurrent_syncs,
        );

        Self {
            config,
            repo_manager,
            indexes: RwLock::new(HashMap::new()),
            graphs: RwLock::new(HashMap::new()),
            running: RwLock::new(false),
            #[cfg(feature = "embeddings")]
            embedder: None,
        }
    }

    /// Initialize the embedding model asynchronously.
    ///
    /// This must be called before indexing if the `embeddings` feature is enabled.
    /// For local models, the model will be downloaded from HuggingFace on first use.
    #[cfg(feature = "embeddings")]
    pub async fn init_embedder(&mut self) -> Result<()> {
        // Clone embedding config to avoid borrow issues with preset initialization
        let embedding_config = self.config.embedding.clone();

        let provider = match &embedding_config {
            EmbeddingConfig::Local { model, batch_size } => {
                info!("Initializing local embedder with model: {}", model);

                // Try preset first, then fall back to custom HuggingFace model
                match model.as_str() {
                    "bge-small" | "bge-base" | "bge-large" | "minilm" | "jina-small" | "nomic" => {
                        return self.init_embedder_from_preset(model).await;
                    }
                    _ => {}
                }

                let config = EmbedderConfig::bert(model).with_batch_size(*batch_size);
                EmbedderProvider::from_config(config)
                    .await
                    .map_err(|e| Error::IndexingFailed(format!("Failed to load model: {e}")))?
            }
            EmbeddingConfig::OpenAI {
                model,
                api_key,
                batch_size,
            } => {
                info!("Initializing OpenAI embedder with model: {}", model);

                let mut config = EmbedderConfig::openai(model).with_batch_size(*batch_size);
                if let Some(key) = api_key {
                    config.cloud_provider = Some(CloudProvider::OpenAI {
                        api_key: Some(key.clone()),
                    });
                }

                EmbedderProvider::from_config(config)
                    .await
                    .map_err(|e| Error::IndexingFailed(format!("Failed to init OpenAI: {e}")))?
            }
            EmbeddingConfig::Cohere {
                model,
                api_key,
                batch_size,
            } => {
                info!("Initializing Cohere embedder with model: {}", model);

                let config = EmbedderConfig {
                    model_id: model.clone(),
                    backend: InferenceBackend::Cloud,
                    batch_size: *batch_size,
                    cloud_provider: Some(CloudProvider::Cohere {
                        api_key: api_key.clone(),
                    }),
                    ..Default::default()
                };

                EmbedderProvider::from_config(config)
                    .await
                    .map_err(|e| Error::IndexingFailed(format!("Failed to init Cohere: {e}")))?
            }
        };

        info!(
            "Embedder initialized with dimension: {}",
            provider.dimension()
        );
        self.embedder = Some(Arc::new(provider));
        Ok(())
    }

    #[cfg(feature = "embeddings")]
    async fn init_embedder_from_preset(&mut self, preset: &str) -> Result<()> {
        let provider = EmbedderProvider::from_preset(preset)
            .await
            .map_err(|e| Error::IndexingFailed(format!("Failed to load preset {preset}: {e}")))?;

        info!(
            "Embedder initialized with dimension: {}",
            provider.dimension()
        );
        self.embedder = Some(Arc::new(provider));
        Ok(())
    }

    /// Get the embedder if initialized.
    #[cfg(feature = "embeddings")]
    pub fn embedder(&self) -> Option<&Arc<EmbedderProvider>> {
        self.embedder.as_ref()
    }

    /// Get the repository manager
    #[must_use]
    pub fn repository_manager(&self) -> &RepositoryManager {
        &self.repo_manager
    }

    /// Get the index path for a repository
    fn index_path(&self, repo: &Repository) -> PathBuf {
        self.config
            .indexes_path
            .join(&repo.provider)
            .join(&repo.owner)
            .join(format!("{}.leann", repo.name))
    }

    /// Add and index a repository
    pub async fn add_repository(&self, repo: &Repository) -> Result<RepositoryState> {
        let state = self.repo_manager.clone_repository(repo).await?;
        self.index_repository(repo).await?;
        Ok(state)
    }

    /// Sync a repository and re-index if needed
    pub async fn sync_repository(&self, repo: &Repository) -> Result<RepositoryState> {
        let state = self.repo_manager.update_repository(repo).await?;

        if self.repo_manager.needs_reindex(repo).await {
            self.index_repository(repo).await?;
        }

        Ok(state)
    }

    /// Build a LEANN index for a repository
    async fn index_repository(&self, repo: &Repository) -> Result<IndexInfo> {
        let state = self
            .repo_manager
            .get_state(repo)
            .await
            .ok_or_else(|| Error::repo_not_found(repo.id()))?;

        if state.error.is_some() {
            return Err(Error::IndexingFailed(state.error.unwrap_or_default()));
        }

        let index_path = self.index_path(repo);
        if let Some(parent) = index_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        info!("Indexing {}", repo.id());

        // Collect files to index
        let local_path = state.local_path.clone();
        let extensions = self.config.index_extensions.clone();

        let files = tokio::task::spawn_blocking(move || collect_files(&local_path, &extensions))
            .await
            .map_err(|e| Error::IndexingFailed(e.to_string()))?;

        let file_count = files.len();
        info!("Found {} files to index", file_count);

        // Build embeddings and index
        #[cfg(feature = "embeddings")]
        let graph = self.build_index_with_embeddings(&files).await?;

        #[cfg(not(feature = "embeddings"))]
        let graph = self.build_index_placeholder(&files)?;

        // Mark as indexed
        self.repo_manager.mark_indexed(repo).await;

        // Calculate size (estimated from graph)
        let size_bytes = graph.len() as u64 * 4 * 384; // rough estimate: vectors * sizeof(f32) * dim

        let info = IndexInfo {
            name: repo.id(),
            path: index_path,
            repository: repo.clone(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            file_count,
            size_bytes,
        };

        // Store graph with file metadata
        let stored = StoredIndex {
            graph,
            files,
            info: info.clone(),
        };

        {
            let mut graphs = self.graphs.write().await;
            graphs.insert(info.name.clone(), stored);
        }

        // Store index info
        let mut indexes = self.indexes.write().await;
        indexes.insert(info.name.clone(), info.clone());

        info!("Indexed {}: {} files", repo.id(), file_count);
        Ok(info)
    }

    /// Build index with actual embeddings from embed_anything
    #[cfg(feature = "embeddings")]
    async fn build_index_with_embeddings(
        &self,
        files: &[(String, String)],
    ) -> Result<islands_core::HnswGraph> {
        let embedder = self.embedder.as_ref().ok_or_else(|| {
            Error::IndexingFailed(
                "Embedder not initialized. Call init_embedder() first.".to_string(),
            )
        })?;

        let dimension = embedder.dimension();
        let mut graph = islands_core::HnswGraph::new(islands_core::HnswConfig::default())
            .map_err(|e| Error::IndexingFailed(e.to_string()))?;

        // Process files in batches
        let batch_size = self.config.embedding.batch_size();
        let total_batches = (files.len() + batch_size - 1) / batch_size;

        for (batch_idx, chunk) in files.chunks(batch_size).enumerate() {
            debug!(
                "Processing batch {}/{} ({} files)",
                batch_idx + 1,
                total_batches,
                chunk.len()
            );

            // Extract content for embedding
            let texts: Vec<&str> = chunk.iter().map(|(_, content)| content.as_str()).collect();

            // Generate embeddings
            let embeddings = embedder
                .embed_texts_raw(&texts)
                .await
                .map_err(|e| Error::IndexingFailed(format!("Embedding failed: {e}")))?;

            // Insert into graph
            for embedding in embeddings {
                graph
                    .insert(embedding)
                    .map_err(|e| Error::IndexingFailed(e.to_string()))?;
            }
        }

        info!(
            "Built index with {} vectors (dimension: {})",
            graph.len(),
            dimension
        );
        Ok(graph)
    }

    /// Build index with placeholder zero embeddings (no embeddings feature)
    #[cfg(not(feature = "embeddings"))]
    fn build_index_placeholder(
        &self,
        files: &[(String, String)],
    ) -> Result<islands_core::HnswGraph> {
        let mut graph = islands_core::HnswGraph::new(islands_core::HnswConfig::fast())
            .map_err(|e| Error::IndexingFailed(e.to_string()))?;

        for (_path, _content) in files {
            let embedding = islands_core::embedding::Embedding::zeros(384);
            graph
                .insert(embedding.into_vec())
                .map_err(|e| Error::IndexingFailed(e.to_string()))?;
        }

        Ok(graph)
    }

    /// Search across indexed repositories
    ///
    /// When the `embeddings` feature is enabled, this uses the embedding model
    /// to convert the query to a vector and search the HNSW index.
    pub async fn search(
        &self,
        query: &str,
        index_names: Option<&[String]>,
        top_k: usize,
    ) -> Result<Vec<serde_json::Value>> {
        info!("Searching for: {}", query);

        #[cfg(feature = "embeddings")]
        return self.search_with_embeddings(query, index_names, top_k).await;

        #[cfg(not(feature = "embeddings"))]
        {
            tracing::warn!("Search without embeddings feature returns empty results");
            let _ = (query, index_names, top_k);
            Ok(Vec::new())
        }
    }

    /// Search using actual embeddings
    #[cfg(feature = "embeddings")]
    async fn search_with_embeddings(
        &self,
        query: &str,
        index_names: Option<&[String]>,
        top_k: usize,
    ) -> Result<Vec<serde_json::Value>> {
        let embedder = match self.embedder.as_ref() {
            Some(e) => e,
            None => {
                // If embedder not initialized and no indexes exist, return empty results
                let graphs = self.graphs.read().await;
                if graphs.is_empty() {
                    return Ok(Vec::new());
                }
                return Err(Error::IndexingFailed(
                    "Embedder not initialized. Call init_embedder() first.".to_string(),
                ));
            }
        };

        // Embed the query
        let query_embedding = embedder
            .embed_texts_raw(&[query])
            .await
            .map_err(|e| Error::IndexingFailed(format!("Query embedding failed: {e}")))?
            .into_iter()
            .next()
            .ok_or_else(|| Error::IndexingFailed("No query embedding returned".to_string()))?;

        let graphs = self.graphs.read().await;

        // Determine which indexes to search
        let target_names: Vec<&String> = match index_names {
            Some(names) => names.iter().collect(),
            None => graphs.keys().collect(),
        };

        let mut all_results: Vec<(f32, String, String, String)> = Vec::new(); // (score, index_name, path, snippet)

        for name in target_names {
            if let Some(stored) = graphs.get(name) {
                // Search the HNSW graph
                // ef parameter controls accuracy vs speed tradeoff (higher = more accurate)
                let ef = top_k.max(100);
                let search_results = stored
                    .graph
                    .search(&query_embedding, top_k, ef)
                    .map_err(|e| Error::IndexingFailed(format!("Search failed: {e}")))?;

                // Map results back to file metadata
                for (id, distance) in search_results {
                    if let Some((path, content)) = stored.files.get(id as usize) {
                        // Convert distance to similarity score (for cosine, lower is better)
                        let score = 1.0 - distance;
                        let snippet = content.chars().take(200).collect::<String>();
                        all_results.push((score, name.clone(), path.clone(), snippet));
                    }
                }
            }
        }

        // Sort by score descending and take top_k
        all_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(top_k);

        // Convert to JSON
        let results: Vec<serde_json::Value> = all_results
            .into_iter()
            .map(|(score, index, path, snippet)| {
                serde_json::json!({
                    "score": score,
                    "index": index,
                    "path": path,
                    "snippet": snippet
                })
            })
            .collect();

        info!("Found {} results", results.len());
        Ok(results)
    }

    /// List all indexes
    pub async fn list_indexes(&self) -> Vec<IndexInfo> {
        let indexes = self.indexes.read().await;
        indexes.values().cloned().collect()
    }

    /// Get a specific index
    pub async fn get_index(&self, name: &str) -> Option<IndexInfo> {
        let indexes = self.indexes.read().await;
        indexes.get(name).cloned()
    }

    /// Handle a webhook event
    pub async fn handle_webhook(&self, event: &WebhookEvent) -> Result<()> {
        if event.is_push() {
            info!("Webhook push event for {}", event.repository.id());
            self.sync_repository(&event.repository).await?;
        }
        Ok(())
    }

    /// Start the background sync loop
    pub async fn start_sync_loop(&self) {
        {
            let mut running = self.running.write().await;
            if *running {
                return;
            }
            *running = true;
        }

        info!(
            "Starting sync loop (interval: {}s)",
            self.config.sync_interval_secs
        );

        let mut ticker = interval(Duration::from_secs(self.config.sync_interval_secs));

        loop {
            ticker.tick().await;

            if !*self.running.read().await {
                break;
            }

            let states = self.repo_manager.list_states().await;
            for state in states {
                if !*self.running.read().await {
                    break;
                }

                if let Err(e) = self.sync_repository(&state.repository).await {
                    error!("Sync failed for {}: {}", state.repository.id(), e);
                }
            }
        }

        info!("Sync loop stopped");
    }

    /// Stop the background sync loop
    pub async fn stop_sync_loop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }
}

/// Collect files to index from a directory
fn collect_files(root: &std::path::Path, extensions: &[String]) -> Vec<(String, String)> {
    let mut files = Vec::new();

    let walker = walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !name.starts_with('.') && name != "node_modules" && name != "target"
        });

    for entry in walker.filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();

        // Check extension
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if !extensions.iter().any(|e| e == ext) {
                continue;
            }
        } else {
            continue;
        }

        // Read content
        if let Ok(content) = std::fs::read_to_string(path) {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            files.push((relative, content));
        }
    }

    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_indexer_config_default() {
        let config = IndexerConfig::default();

        assert_eq!(config.repos_path, PathBuf::from("/data/islands/repos"));
        assert_eq!(config.indexes_path, PathBuf::from("/data/islands/indexes"));
        assert_eq!(config.max_concurrent_syncs, 4);
        assert_eq!(config.sync_interval_secs, 300);
        assert!(config.index_extensions.contains(&"py".to_string()));
        assert!(config.index_extensions.contains(&"rs".to_string()));
        assert!(config.index_extensions.contains(&"js".to_string()));
        assert!(config.index_extensions.contains(&"ts".to_string()));
    }

    #[test]
    fn test_indexer_config_serialization() {
        let config = IndexerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: IndexerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.max_concurrent_syncs, config.max_concurrent_syncs);
        assert_eq!(parsed.sync_interval_secs, config.sync_interval_secs);
    }

    #[test]
    fn test_index_info_serialization() {
        let repo = islands_providers::Repository::new(
            "github",
            "test",
            "repo",
            "https://github.com/test/repo.git",
        );

        let info = IndexInfo {
            name: "github/test/repo".to_string(),
            path: PathBuf::from("/data/indexes/github/test/repo.leann"),
            repository: repo,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            file_count: 100,
            size_bytes: 1024,
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: IndexInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, info.name);
        assert_eq!(parsed.file_count, 100);
        assert_eq!(parsed.size_bytes, 1024);
    }

    #[test]
    fn test_collect_files_empty_dir() {
        let dir = tempdir().unwrap();
        let extensions = vec!["rs".to_string(), "py".to_string()];

        let files = collect_files(dir.path(), &extensions);
        assert!(files.is_empty());
    }

    #[test]
    fn test_collect_files_with_files() {
        let dir = tempdir().unwrap();
        // Create a non-hidden root to avoid tempdir's .tmpXXX name being filtered
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        // Create some test files
        std::fs::write(root.join("test.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("test.py"), "print('hello')").unwrap();
        std::fs::write(root.join("test.txt"), "ignored").unwrap();

        let extensions = vec!["rs".to_string(), "py".to_string()];
        let files = collect_files(&root, &extensions);

        assert_eq!(files.len(), 2);

        let names: Vec<_> = files.iter().map(|(name, _)| name.as_str()).collect();
        assert!(names.contains(&"test.rs"));
        assert!(names.contains(&"test.py"));
    }

    #[test]
    fn test_collect_files_ignores_hidden() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        // Create a hidden file
        std::fs::write(root.join(".hidden.rs"), "hidden").unwrap();
        std::fs::write(root.join("visible.rs"), "visible").unwrap();

        let extensions = vec!["rs".to_string()];
        let files = collect_files(&root, &extensions);

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "visible.rs");
    }

    #[test]
    fn test_collect_files_ignores_node_modules() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        let node_modules = root.join("node_modules");
        std::fs::create_dir(&node_modules).unwrap();

        std::fs::write(node_modules.join("dep.js"), "module.exports = {}").unwrap();
        std::fs::write(root.join("main.js"), "console.log()").unwrap();

        let extensions = vec!["js".to_string()];
        let files = collect_files(&root, &extensions);

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "main.js");
    }

    #[test]
    fn test_collect_files_ignores_target() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        let target = root.join("target");
        std::fs::create_dir(&target).unwrap();

        std::fs::write(target.join("build.rs"), "compiled").unwrap();
        std::fs::write(root.join("src.rs"), "source").unwrap();

        let extensions = vec!["rs".to_string()];
        let files = collect_files(&root, &extensions);

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "src.rs");
    }

    #[test]
    fn test_collect_files_nested_directories() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        let subdir = root.join("src").join("nested");
        std::fs::create_dir_all(&subdir).unwrap();

        std::fs::write(root.join("root.rs"), "root").unwrap();
        std::fs::write(subdir.join("nested.rs"), "nested").unwrap();

        let extensions = vec!["rs".to_string()];
        let files = collect_files(&root, &extensions);

        assert_eq!(files.len(), 2);
    }

    #[tokio::test]
    async fn test_indexer_service_new() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let providers = HashMap::new();
        let service = IndexerService::new(config, providers);

        // Directories should be created
        assert!(dir.path().join("repos").exists());
        assert!(dir.path().join("indexes").exists());

        // Should start with no indexes
        let indexes = service.list_indexes().await;
        assert!(indexes.is_empty());
    }

    #[tokio::test]
    async fn test_indexer_service_get_nonexistent_index() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());
        let index = service.get_index("nonexistent").await;

        assert!(index.is_none());
    }

    #[tokio::test]
    async fn test_indexer_service_search_empty() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());
        let results = service.search("test query", None, 10).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_indexer_service_stop_sync_loop() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());
        service.stop_sync_loop().await;
        // Should not panic
    }

    #[tokio::test]
    async fn test_indexer_service_search_with_specific_indexes() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());
        let index_names = vec!["repo1".to_string(), "repo2".to_string()];
        let results = service
            .search("test query", Some(&index_names), 10)
            .await
            .unwrap();

        // No indexes exist, so should be empty
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_indexer_service_repository_manager_accessor() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());
        let _manager = service.repository_manager();
        // Just verify we can access it
    }

    #[test]
    fn test_index_info_debug() {
        let repo = islands_providers::Repository::new(
            "github",
            "test",
            "repo",
            "https://github.com/test/repo.git",
        );

        let info = IndexInfo {
            name: "github/test/repo".to_string(),
            path: PathBuf::from("/data/indexes/github/test/repo.leann"),
            repository: repo,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            file_count: 100,
            size_bytes: 1024,
        };

        let debug = format!("{:?}", info);
        assert!(debug.contains("github/test/repo"));
    }

    #[test]
    fn test_indexer_config_custom_extensions() {
        let config = IndexerConfig {
            repos_path: PathBuf::from("/custom/repos"),
            indexes_path: PathBuf::from("/custom/indexes"),
            max_concurrent_syncs: 8,
            sync_interval_secs: 600,
            index_extensions: vec!["custom".to_string()],
            #[cfg(feature = "embeddings")]
            embedding: EmbeddingConfig::default(),
        };

        assert_eq!(config.repos_path, PathBuf::from("/custom/repos"));
        assert_eq!(config.max_concurrent_syncs, 8);
        assert_eq!(config.sync_interval_secs, 600);
        assert_eq!(config.index_extensions, vec!["custom".to_string()]);
    }

    #[test]
    fn test_collect_files_no_extension() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        // Create a file without extension
        std::fs::write(root.join("Makefile"), "all: build").unwrap();
        std::fs::write(root.join("test.rs"), "fn main() {}").unwrap();

        let extensions = vec!["rs".to_string()];
        let files = collect_files(&root, &extensions);

        // Only .rs file should be included
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "test.rs");
    }

    #[test]
    fn test_collect_files_unreadable_file_skipped() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        // Create a binary file that can't be read as UTF-8
        std::fs::write(root.join("binary.rs"), vec![0xFF, 0xFE, 0x00, 0x01]).unwrap();
        std::fs::write(root.join("valid.rs"), "fn main() {}").unwrap();

        let extensions = vec!["rs".to_string()];
        let files = collect_files(&root, &extensions);

        // Binary file might fail read_to_string, valid file should be included
        // At least the valid file should be present
        assert!(files.iter().any(|(name, _)| name == "valid.rs"));
    }

    #[test]
    fn test_collect_files_deeply_nested() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        let deep = root.join("a").join("b").join("c").join("d");
        std::fs::create_dir_all(&deep).unwrap();

        std::fs::write(deep.join("deep.rs"), "fn deep() {}").unwrap();

        let extensions = vec!["rs".to_string()];
        let files = collect_files(&root, &extensions);

        assert_eq!(files.len(), 1);
        assert!(files[0].0.contains("deep.rs"));
    }

    #[test]
    fn test_index_info_clone() {
        let repo = islands_providers::Repository::new(
            "github",
            "test",
            "repo",
            "https://github.com/test/repo.git",
        );

        let info = IndexInfo {
            name: "test".to_string(),
            path: PathBuf::from("/test"),
            repository: repo,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            file_count: 50,
            size_bytes: 512,
        };

        let cloned = info.clone();
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.file_count, info.file_count);
    }

    #[tokio::test]
    async fn test_indexer_service_list_indexes_empty() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());
        let indexes = service.list_indexes().await;

        assert!(indexes.is_empty());
    }

    #[test]
    fn test_indexer_config_debug() {
        let config = IndexerConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("repos_path"));
        assert!(debug.contains("indexes_path"));
    }
}
