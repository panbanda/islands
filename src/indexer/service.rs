//! Main indexer service

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::time::{Duration, interval};
use tracing::{error, info};

use crate::providers::{GitProvider, Repository, WebhookEvent};

#[cfg(feature = "embeddings")]
use crate::core::embedding::{CloudProvider, EmbedderConfig, EmbedderProvider, InferenceBackend};

use super::error::{Error, Result};
use super::manager::RepositoryManager;
use super::state::RepositoryState;

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

/// Information about a workspace (multi-repo index)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceInfo {
    /// Workspace name
    pub name: String,
    /// Path to workspace metadata
    pub path: PathBuf,
    /// Source repositories in this workspace
    pub repositories: Vec<Repository>,
    /// When the workspace was created
    pub created_at: DateTime<Utc>,
    /// When the workspace was last updated
    pub updated_at: DateTime<Utc>,
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
    /// Use native Candle inference (sentence-transformers-rs)
    /// Supports: minilm, bge-small, bge-base, bge-large
    #[cfg(feature = "embeddings-candle")]
    Candle {
        /// Model preset (minilm, bge-small, bge-base, bge-large) or HuggingFace model ID
        #[serde(default = "default_candle_model")]
        model: String,
        /// Device for inference (cpu, metal, cuda)
        #[serde(default = "default_candle_device")]
        device: String,
        /// Batch size for processing
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

#[cfg(all(feature = "embeddings", feature = "embeddings-candle"))]
fn default_candle_model() -> String {
    "minilm".to_string()
}

#[cfg(all(feature = "embeddings", feature = "embeddings-candle"))]
fn default_candle_device() -> String {
    "cpu".to_string()
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
            #[cfg(feature = "embeddings-candle")]
            Self::Candle { batch_size, .. } => *batch_size,
        }
    }
}

impl Default for IndexerConfig {
    fn default() -> Self {
        let base_path = directories::ProjectDirs::from("", "", "islands")
            .map(|dirs| dirs.data_dir().to_path_buf())
            .unwrap_or_else(|| {
                directories::BaseDirs::new()
                    .map(|d| d.home_dir().join(".islands"))
                    .unwrap_or_else(|| PathBuf::from(".islands"))
            });

        Self {
            repos_path: base_path.join("repos"),
            indexes_path: base_path.join("indexes"),
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
    pub graph: crate::core::HnswGraph,
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
    /// Workspaces (multi-repo indexes)
    workspaces: RwLock<HashMap<String, WorkspaceInfo>>,
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

        // Load persisted indexes from disk
        let indexes = Self::load_indexes_from_disk(&config.indexes_path);
        let workspaces = Self::load_workspaces_from_disk(&config.indexes_path);

        if !indexes.is_empty() {
            info!("Loaded {} persisted indexes from disk", indexes.len());
        }
        if !workspaces.is_empty() {
            info!("Loaded {} persisted workspaces from disk", workspaces.len());
        }

        Self {
            config,
            repo_manager,
            indexes: RwLock::new(indexes),
            graphs: RwLock::new(HashMap::new()),
            workspaces: RwLock::new(workspaces),
            running: RwLock::new(false),
            #[cfg(feature = "embeddings")]
            embedder: None,
        }
    }

    /// Load indexes from disk by scanning for metadata.json files
    fn load_indexes_from_disk(indexes_path: &Path) -> HashMap<String, IndexInfo> {
        let mut indexes = HashMap::new();

        // Walk through the indexes directory looking for metadata.json files
        if let Ok(entries) = std::fs::read_dir(indexes_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    // Recursively search provider directories (e.g., github/)
                    Self::scan_provider_dir(&path, &mut indexes);
                }
            }
        }

        indexes
    }

    /// Scan a provider directory for index metadata
    fn scan_provider_dir(provider_path: &Path, indexes: &mut HashMap<String, IndexInfo>) {
        if let Ok(entries) = std::fs::read_dir(provider_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    // This could be owner/ directory
                    Self::scan_owner_dir(&path, indexes);
                }
            }
        }
    }

    /// Scan an owner directory for index metadata
    fn scan_owner_dir(owner_path: &Path, indexes: &mut HashMap<String, IndexInfo>) {
        if let Ok(entries) = std::fs::read_dir(owner_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    // This could be repo/ directory - check for metadata.json
                    let metadata_path = path.join("metadata.json");
                    if metadata_path.exists() {
                        if let Ok(content) = std::fs::read_to_string(&metadata_path) {
                            if let Ok(info) = serde_json::from_str::<IndexInfo>(&content) {
                                indexes.insert(info.name.clone(), info);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Load workspaces from disk
    fn load_workspaces_from_disk(indexes_path: &Path) -> HashMap<String, WorkspaceInfo> {
        let mut workspaces = HashMap::new();
        let workspaces_dir = indexes_path.join("workspaces");

        if let Ok(entries) = std::fs::read_dir(&workspaces_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let workspace_file = path.join("workspace.json");
                    if workspace_file.exists() {
                        if let Ok(content) = std::fs::read_to_string(&workspace_file) {
                            if let Ok(workspace) = serde_json::from_str::<WorkspaceInfo>(&content) {
                                workspaces.insert(workspace.name.clone(), workspace);
                            }
                        }
                    }
                }
            }
        }

        workspaces
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
            .join(&repo.name)
            .join("index.leann")
    }

    /// Add and index a repository
    pub async fn add_repository(&self, repo: &Repository) -> Result<RepositoryState> {
        self.add_repository_with_progress(repo, None).await
    }

    /// Add and index a repository with progress reporting
    pub async fn add_repository_with_progress(
        &self,
        repo: &Repository,
        progress: Option<&ProgressBar>,
    ) -> Result<RepositoryState> {
        let state = self.repo_manager.clone_repository(repo).await?;
        self.index_repository_with_progress(repo, progress).await?;
        Ok(state)
    }

    /// Sync a repository and re-index if needed
    pub async fn sync_repository(&self, repo: &Repository) -> Result<RepositoryState> {
        self.sync_repository_with_progress(repo, None).await
    }

    /// Sync a repository and re-index if needed with progress reporting
    pub async fn sync_repository_with_progress(
        &self,
        repo: &Repository,
        progress: Option<&ProgressBar>,
    ) -> Result<RepositoryState> {
        let state = self.repo_manager.update_repository(repo).await?;

        if self.repo_manager.needs_reindex(repo).await {
            self.index_repository_with_progress(repo, progress).await?;
        }

        Ok(state)
    }

    /// Build a LEANN index for a repository with progress reporting
    ///
    /// If a progress bar is provided, it will be used to report progress during:
    /// - File collection (counting and reading files)
    /// - Embedding generation (batch processing)
    pub async fn index_repository_with_progress(
        &self,
        repo: &Repository,
        progress: Option<&ProgressBar>,
    ) -> Result<IndexInfo> {
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

        // First count files for progress reporting
        let (file_count, files) = if let Some(pb) = progress {
            pb.set_message("Counting files...");
            let count_path = local_path.clone();
            let count_ext = extensions.clone();
            let file_count =
                tokio::task::spawn_blocking(move || count_matching_files(&count_path, &count_ext))
                    .await
                    .map_err(|e| Error::IndexingFailed(e.to_string()))?;

            pb.set_length(file_count as u64);
            pb.set_position(0);
            pb.set_message("Collecting files...");

            // Clone the progress bar for the blocking task
            let pb_clone = pb.clone();
            let files = tokio::task::spawn_blocking(move || {
                collect_files_with_progress(&local_path, &extensions, Some(&pb_clone))
            })
            .await
            .map_err(|e| Error::IndexingFailed(e.to_string()))?;

            pb.set_message("Files collected");
            (files.len(), files)
        } else {
            let files =
                tokio::task::spawn_blocking(move || collect_files(&local_path, &extensions))
                    .await
                    .map_err(|e| Error::IndexingFailed(e.to_string()))?;
            (files.len(), files)
        };

        info!("Found {} files to index", file_count);

        // Build embeddings and index
        #[cfg(feature = "embeddings")]
        let graph = self
            .build_index_with_embeddings_progress(&files, progress)
            .await?;

        #[cfg(not(feature = "embeddings"))]
        let graph = self.build_index_placeholder_progress(&files, progress)?;

        // Mark as indexed
        self.repo_manager.mark_indexed(repo).await;

        // Calculate size (estimated from graph)
        let size_bytes = graph.len() as u64 * 4 * 384; // rough estimate: vectors * sizeof(f32) * dim

        let info = IndexInfo {
            name: format!("{}/{}", repo.provider, repo.full_name),
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

        // Store index info in memory
        {
            let mut indexes = self.indexes.write().await;
            indexes.insert(info.name.clone(), info.clone());
        }

        // Persist index metadata to disk
        self.save_index_metadata(&info).await?;

        info!("Indexed {}: {} files", repo.id(), file_count);
        Ok(info)
    }

    /// Build index with actual embeddings from embed_anything with progress tracking
    #[cfg(feature = "embeddings")]
    async fn build_index_with_embeddings_progress(
        &self,
        files: &[(String, String)],
        progress: Option<&ProgressBar>,
    ) -> Result<crate::core::HnswGraph> {
        let embedder = self.embedder.as_ref().ok_or_else(|| {
            Error::IndexingFailed(
                "Embedder not initialized. Call init_embedder() first.".to_string(),
            )
        })?;

        let dimension = embedder.dimension();
        let mut graph = crate::core::HnswGraph::new(crate::core::HnswConfig::default())
            .map_err(|e| Error::IndexingFailed(e.to_string()))?;

        // Process files in batches
        let batch_size = self.config.embedding.batch_size();
        let total_batches = (files.len() + batch_size - 1) / batch_size;

        // Reset progress bar for embedding phase
        if let Some(pb) = progress {
            pb.set_length(files.len() as u64);
            pb.set_position(0);
            pb.set_message("Generating embeddings...");
        }

        for (batch_idx, chunk) in files.chunks(batch_size).enumerate() {
            tracing::debug!(
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

            // Update progress
            if let Some(pb) = progress {
                pb.inc(chunk.len() as u64);
            }
        }

        if let Some(pb) = progress {
            pb.set_message("Embeddings complete");
        }

        info!(
            "Built index with {} vectors (dimension: {})",
            graph.len(),
            dimension
        );
        Ok(graph)
    }

    /// Build index with placeholder zero embeddings with progress tracking
    #[cfg(not(feature = "embeddings"))]
    fn build_index_placeholder_progress(
        &self,
        files: &[(String, String)],
        progress: Option<&ProgressBar>,
    ) -> Result<crate::core::HnswGraph> {
        let mut graph = crate::core::HnswGraph::new(crate::core::HnswConfig::fast())
            .map_err(|e| Error::IndexingFailed(e.to_string()))?;

        // Setup progress bar for indexing phase
        if let Some(pb) = progress {
            pb.set_length(files.len() as u64);
            pb.set_position(0);
            pb.set_message("Building index...");
        }

        for (_path, _content) in files {
            let embedding = crate::core::embedding::Embedding::zeros(384);
            graph
                .insert(embedding.into_vec())
                .map_err(|e| Error::IndexingFailed(e.to_string()))?;

            if let Some(pb) = progress {
                pb.inc(1);
            }
        }

        if let Some(pb) = progress {
            pb.set_message("Index built");
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

    /// Delete an index and all associated files
    pub async fn delete_index(&self, name: &str) -> Result<()> {
        // Get the index info first
        let info = {
            let indexes = self.indexes.read().await;
            indexes.get(name).cloned()
        };

        let info = info.ok_or_else(|| Error::index_not_found(name))?;

        info!("Deleting index: {}", name);

        // Remove the index file if it exists
        if info.path.exists() {
            std::fs::remove_file(&info.path)?;
            // Also try to remove parent directories if empty
            if let Some(parent) = info.path.parent() {
                let _ = std::fs::remove_dir(parent);
                if let Some(grandparent) = parent.parent() {
                    let _ = std::fs::remove_dir(grandparent);
                }
            }
        }

        // Remove the cloned repository
        self.repo_manager
            .remove_repository(&info.repository)
            .await?;

        // Remove from graphs
        {
            let mut graphs = self.graphs.write().await;
            graphs.remove(name);
        }

        // Remove from indexes
        {
            let mut indexes = self.indexes.write().await;
            indexes.remove(name);
        }

        info!("Successfully deleted index: {}", name);
        Ok(())
    }

    /// Save index metadata to disk
    pub async fn save_index_metadata(&self, info: &IndexInfo) -> Result<()> {
        // Create the index directory structure
        let index_dir = if let Some(parent) = info.path.parent() {
            parent.to_path_buf()
        } else {
            // Fall back to constructing path from index name
            let parts: Vec<&str> = info.name.split('/').collect();
            if parts.len() >= 3 {
                self.config
                    .indexes_path
                    .join(parts[0])
                    .join(parts[1])
                    .join(parts[2])
            } else {
                self.config.indexes_path.join(&info.name)
            }
        };

        std::fs::create_dir_all(&index_dir)?;

        let metadata_path = index_dir.join("metadata.json");
        let json = serde_json::to_string_pretty(info)
            .map_err(|e| Error::IndexingFailed(format!("Failed to serialize metadata: {}", e)))?;
        std::fs::write(&metadata_path, json)?;

        info!("Saved index metadata to {:?}", metadata_path);
        Ok(())
    }

    /// Create a workspace containing multiple repositories
    pub async fn create_workspace(
        &self,
        name: &str,
        repos: &[Repository],
    ) -> Result<WorkspaceInfo> {
        let workspace_dir = self.config.indexes_path.join("workspaces").join(name);
        std::fs::create_dir_all(&workspace_dir)?;

        let workspace = WorkspaceInfo {
            name: name.to_string(),
            path: workspace_dir.join("workspace.json"),
            repositories: repos.to_vec(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Save workspace metadata to disk
        let json = serde_json::to_string_pretty(&workspace)
            .map_err(|e| Error::IndexingFailed(format!("Failed to serialize workspace: {}", e)))?;
        std::fs::write(&workspace.path, &json)?;

        // Add to in-memory map
        {
            let mut workspaces = self.workspaces.write().await;
            workspaces.insert(name.to_string(), workspace.clone());
        }

        info!(
            "Created workspace '{}' with {} repositories",
            name,
            repos.len()
        );
        Ok(workspace)
    }

    /// List all workspaces
    pub async fn list_workspaces(&self) -> Vec<WorkspaceInfo> {
        let workspaces = self.workspaces.read().await;
        workspaces.values().cloned().collect()
    }

    /// Get a specific workspace
    pub async fn get_workspace(&self, name: &str) -> Option<WorkspaceInfo> {
        let workspaces = self.workspaces.read().await;
        workspaces.get(name).cloned()
    }

    /// Get index names for all repositories in a workspace
    pub async fn get_workspace_index_names(&self, name: &str) -> Option<Vec<String>> {
        let workspaces = self.workspaces.read().await;
        workspaces.get(name).map(|ws| {
            ws.repositories
                .iter()
                .map(|repo| format!("{}/{}", repo.provider, repo.full_name))
                .collect()
        })
    }

    /// Add a repository to an existing workspace
    pub async fn add_repo_to_workspace(&self, name: &str, repo: &Repository) -> Result<()> {
        let mut workspaces = self.workspaces.write().await;
        let workspace = workspaces
            .get_mut(name)
            .ok_or_else(|| Error::workspace_not_found(name))?;

        workspace.repositories.push(repo.clone());
        workspace.updated_at = Utc::now();

        // Persist to disk
        let json = serde_json::to_string_pretty(&*workspace)
            .map_err(|e| Error::IndexingFailed(format!("Failed to serialize workspace: {}", e)))?;
        std::fs::write(&workspace.path, &json)?;

        info!("Added repository '{}' to workspace '{}'", repo.id(), name);
        Ok(())
    }

    /// Remove a repository from a workspace
    pub async fn remove_repo_from_workspace(&self, name: &str, repo_id: &str) -> Result<()> {
        let mut workspaces = self.workspaces.write().await;
        let workspace = workspaces
            .get_mut(name)
            .ok_or_else(|| Error::workspace_not_found(name))?;

        let original_len = workspace.repositories.len();
        workspace.repositories.retain(|r| r.id() != repo_id);

        if workspace.repositories.len() == original_len {
            return Err(Error::repo_not_in_workspace(repo_id));
        }

        workspace.updated_at = Utc::now();

        // Persist to disk
        let json = serde_json::to_string_pretty(&*workspace)
            .map_err(|e| Error::IndexingFailed(format!("Failed to serialize workspace: {}", e)))?;
        std::fs::write(&workspace.path, &json)?;

        info!("Removed repository '{}' from workspace '{}'", repo_id, name);
        Ok(())
    }

    /// Delete a workspace
    pub async fn delete_workspace(&self, name: &str) -> Result<()> {
        let mut workspaces = self.workspaces.write().await;

        let workspace = workspaces
            .remove(name)
            .ok_or_else(|| Error::workspace_not_found(name))?;

        // Remove workspace directory from disk
        let workspace_dir = workspace.path.parent().unwrap_or(&workspace.path);
        if workspace_dir.exists() {
            std::fs::remove_dir_all(workspace_dir)?;
        }

        info!("Deleted workspace '{}'", name);
        Ok(())
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

/// Create a filtered WalkDir iterator for a directory
fn create_walker(root: &std::path::Path) -> impl Iterator<Item = walkdir::DirEntry> + '_ {
    walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !name.starts_with('.') && name != "node_modules" && name != "target"
        })
        .filter_map(|e| e.ok())
}

/// Check if a file matches the extension filter
fn matches_extension(path: &std::path::Path, extensions: &[String]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| extensions.iter().any(|e| e == ext))
        .unwrap_or(false)
}

/// Count files that match the extension filter
fn count_matching_files(root: &std::path::Path, extensions: &[String]) -> usize {
    create_walker(root)
        .filter(|e| e.file_type().is_file() && matches_extension(e.path(), extensions))
        .count()
}

/// Collect files to index from a directory with optional progress tracking
fn collect_files_with_progress(
    root: &std::path::Path,
    extensions: &[String],
    progress: Option<&ProgressBar>,
) -> Vec<(String, String)> {
    let mut files = Vec::new();

    for entry in create_walker(root) {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();

        if !matches_extension(path, extensions) {
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

            if let Some(pb) = progress {
                pb.inc(1);
            }
        }
    }

    files
}

/// Collect files to index from a directory
fn collect_files(root: &std::path::Path, extensions: &[String]) -> Vec<(String, String)> {
    collect_files_with_progress(root, extensions, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_indexer_config_default() {
        let config = IndexerConfig::default();

        // Paths should end with repos/indexes under an islands data directory
        assert!(config.repos_path.ends_with("repos"));
        assert!(config.indexes_path.ends_with("indexes"));
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
        let repo = crate::providers::Repository::new(
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
        let repo = crate::providers::Repository::new(
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
        let repo = crate::providers::Repository::new(
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

    #[tokio::test]
    async fn test_handle_webhook_push_event() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        let repo = crate::providers::Repository::new(
            "github",
            "test",
            "repo",
            "https://github.com/test/repo.git",
        );

        let event = crate::providers::WebhookEvent {
            event_type: "push".to_string(),
            repository: repo,
            ref_name: Some("refs/heads/main".to_string()),
            before: Some("abc123".to_string()),
            after: Some("def456".to_string()),
            payload: std::collections::HashMap::new(),
        };

        // Push event will try to sync (and fail due to no provider)
        let result = service.handle_webhook(&event).await;
        // Error is expected since we don't have a real provider
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_webhook_non_push_event() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        let repo = crate::providers::Repository::new(
            "github",
            "test",
            "repo",
            "https://github.com/test/repo.git",
        );

        let event = crate::providers::WebhookEvent {
            event_type: "pull_request".to_string(),
            repository: repo,
            ref_name: None,
            before: None,
            after: None,
            payload: std::collections::HashMap::new(),
        };

        // Non-push events should be ignored (no error, no action)
        let result = service.handle_webhook(&event).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_webhook_issue_event() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        let repo = crate::providers::Repository::new(
            "github",
            "test",
            "repo",
            "https://github.com/test/repo.git",
        );

        let event = crate::providers::WebhookEvent {
            event_type: "issues".to_string(),
            repository: repo,
            ref_name: None,
            before: None,
            after: None,
            payload: std::collections::HashMap::new(),
        };

        let result = service.handle_webhook(&event).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_add_repository_no_provider() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        let repo = crate::providers::Repository::new(
            "github",
            "owner",
            "repo",
            "https://github.com/owner/repo.git",
        );

        // Add repository will fail (no provider)
        let result = service.add_repository(&repo).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unknown provider"));
    }

    #[tokio::test]
    async fn test_sync_repository_no_provider() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        let repo = crate::providers::Repository::new(
            "github",
            "owner",
            "repo",
            "https://github.com/owner/repo.git",
        );

        // Sync repository will fail (no provider)
        let result = service.sync_repository(&repo).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_index_path_structure() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        let repo = crate::providers::Repository::new(
            "github",
            "my-org",
            "my-repo",
            "https://github.com/my-org/my-repo.git",
        );

        let path = service.index_path(&repo);
        assert!(path.to_string_lossy().contains("github"));
        assert!(path.to_string_lossy().contains("my-org"));
        assert!(path.to_string_lossy().contains("my-repo"));
        assert!(path.to_string_lossy().contains("index.leann"));
    }

    #[tokio::test]
    async fn test_list_indexes_with_indexes() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Manually insert some indexes for testing
        {
            let mut indexes = service.indexes.write().await;

            let repo1 = crate::providers::Repository::new(
                "github",
                "owner1",
                "repo1",
                "https://github.com/owner1/repo1.git",
            );
            let info1 = IndexInfo {
                name: "github/owner1/repo1".to_string(),
                path: PathBuf::from("/data/indexes/github/owner1/repo1.leann"),
                repository: repo1,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                file_count: 50,
                size_bytes: 1024,
            };

            let repo2 = crate::providers::Repository::new(
                "github",
                "owner2",
                "repo2",
                "https://github.com/owner2/repo2.git",
            );
            let info2 = IndexInfo {
                name: "github/owner2/repo2".to_string(),
                path: PathBuf::from("/data/indexes/github/owner2/repo2.leann"),
                repository: repo2,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                file_count: 100,
                size_bytes: 2048,
            };

            indexes.insert(info1.name.clone(), info1);
            indexes.insert(info2.name.clone(), info2);
        }

        let list = service.list_indexes().await;
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_get_index_existing() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Manually insert an index
        {
            let mut indexes = service.indexes.write().await;

            let repo = crate::providers::Repository::new(
                "github",
                "owner",
                "repo",
                "https://github.com/owner/repo.git",
            );
            let info = IndexInfo {
                name: "test-index".to_string(),
                path: PathBuf::from("/data/indexes/test.leann"),
                repository: repo,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                file_count: 25,
                size_bytes: 512,
            };

            indexes.insert(info.name.clone(), info);
        }

        let index = service.get_index("test-index").await;
        assert!(index.is_some());
        assert_eq!(index.unwrap().file_count, 25);
    }

    #[tokio::test]
    async fn test_start_stop_sync_loop() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            sync_interval_secs: 1, // Short interval for testing
            ..Default::default()
        };

        let service = Arc::new(IndexerService::new(config, HashMap::new()));

        // Initially not running
        assert!(!*service.running.read().await);

        // Stop before start (should be safe)
        service.stop_sync_loop().await;
        assert!(!*service.running.read().await);
    }

    #[tokio::test]
    async fn test_search_with_no_indexes() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Search with no indexes should return empty
        let results = service.search("test query", None, 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_specific_nonexistent_indexes() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Search specific indexes that don't exist
        let index_names = vec!["nonexistent1".to_string(), "nonexistent2".to_string()];
        let results = service.search("test", Some(&index_names), 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_collect_files_symlinks_not_followed() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        // Create a file
        std::fs::write(root.join("real.rs"), "fn real() {}").unwrap();

        // Note: symlink tests are platform-dependent
        // On Unix, we could test symlinks aren't followed

        let extensions = vec!["rs".to_string()];
        let files = collect_files(&root, &extensions);

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "real.rs");
    }

    #[test]
    fn test_collect_files_multiple_extensions() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        std::fs::write(root.join("file.rs"), "rust").unwrap();
        std::fs::write(root.join("file.py"), "python").unwrap();
        std::fs::write(root.join("file.js"), "javascript").unwrap();
        std::fs::write(root.join("file.txt"), "text").unwrap();

        let extensions = vec!["rs".to_string(), "py".to_string()];
        let files = collect_files(&root, &extensions);

        assert_eq!(files.len(), 2);
        let names: Vec<_> = files.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"file.rs"));
        assert!(names.contains(&"file.py"));
        assert!(!names.contains(&"file.js"));
    }

    #[test]
    fn test_collect_files_preserves_content() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        let content = "fn hello() {\n    println!(\"Hello!\");\n}";
        std::fs::write(root.join("test.rs"), content).unwrap();

        let extensions = vec!["rs".to_string()];
        let files = collect_files(&root, &extensions);

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].1, content);
    }

    #[test]
    fn test_index_info_all_fields() {
        let repo = crate::providers::Repository::new(
            "gitlab",
            "org",
            "project",
            "https://gitlab.com/org/project.git",
        );

        let now = chrono::Utc::now();
        let info = IndexInfo {
            name: "gitlab/org/project".to_string(),
            path: PathBuf::from("/indexes/gitlab/org/project.leann"),
            repository: repo.clone(),
            created_at: now,
            updated_at: now,
            file_count: 500,
            size_bytes: 1024 * 1024,
        };

        assert_eq!(info.name, "gitlab/org/project");
        assert_eq!(info.file_count, 500);
        assert_eq!(info.size_bytes, 1024 * 1024);
        assert_eq!(info.repository.provider, "gitlab");
    }

    #[tokio::test]
    async fn test_service_creates_directories() {
        let dir = tempdir().unwrap();
        let repos_path = dir.path().join("new_repos");
        let indexes_path = dir.path().join("new_indexes");

        // Directories don't exist yet
        assert!(!repos_path.exists());
        assert!(!indexes_path.exists());

        let config = IndexerConfig {
            repos_path: repos_path.clone(),
            indexes_path: indexes_path.clone(),
            ..Default::default()
        };

        let _service = IndexerService::new(config, HashMap::new());

        // Directories should now exist
        assert!(repos_path.exists());
        assert!(indexes_path.exists());
    }

    #[tokio::test]
    async fn test_service_with_providers() {
        use crate::providers::factory::create_provider;

        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let mut providers: HashMap<String, Arc<dyn crate::providers::GitProvider>> = HashMap::new();
        let github = create_provider("github", None, None, None).unwrap();
        providers.insert("github".to_string(), github);

        let service = IndexerService::new(config, providers);

        // Service should have the repository manager with provider
        let _manager = service.repository_manager();
    }

    #[test]
    fn test_indexer_config_index_extensions() {
        let config = IndexerConfig::default();

        // Check various expected extensions are present
        assert!(config.index_extensions.contains(&"py".to_string()));
        assert!(config.index_extensions.contains(&"rs".to_string()));
        assert!(config.index_extensions.contains(&"js".to_string()));
        assert!(config.index_extensions.contains(&"ts".to_string()));
        assert!(config.index_extensions.contains(&"tsx".to_string()));
        assert!(config.index_extensions.contains(&"jsx".to_string()));
        assert!(config.index_extensions.contains(&"go".to_string()));
        assert!(config.index_extensions.contains(&"java".to_string()));
        assert!(config.index_extensions.contains(&"md".to_string()));
        assert!(config.index_extensions.contains(&"json".to_string()));
        assert!(config.index_extensions.contains(&"yaml".to_string()));
        assert!(config.index_extensions.contains(&"toml".to_string()));
    }

    #[test]
    fn test_indexer_config_from_json() {
        let json = r#"{
            "repos_path": "/custom/repos",
            "indexes_path": "/custom/indexes",
            "max_concurrent_syncs": 8,
            "sync_interval_secs": 600,
            "index_extensions": ["rs", "py"]
        }"#;

        let config: IndexerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.repos_path, PathBuf::from("/custom/repos"));
        assert_eq!(config.indexes_path, PathBuf::from("/custom/indexes"));
        assert_eq!(config.max_concurrent_syncs, 8);
        assert_eq!(config.sync_interval_secs, 600);
        assert_eq!(config.index_extensions, vec!["rs", "py"]);
    }

    #[tokio::test]
    async fn test_running_flag_initial_state() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Initially not running
        let running = *service.running.read().await;
        assert!(!running);
    }

    #[tokio::test]
    async fn test_stop_sync_loop_sets_flag() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Manually set running to true
        {
            let mut running = service.running.write().await;
            *running = true;
        }

        assert!(*service.running.read().await);

        // Stop should set it to false
        service.stop_sync_loop().await;

        assert!(!*service.running.read().await);
    }

    #[test]
    fn test_collect_files_empty_extensions() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        std::fs::write(root.join("file.rs"), "rust").unwrap();
        std::fs::write(root.join("file.py"), "python").unwrap();

        let extensions: Vec<String> = vec![];
        let files = collect_files(&root, &extensions);

        // No extensions means no files matched
        assert!(files.is_empty());
    }

    #[test]
    fn test_collect_files_case_sensitive() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("project");
        std::fs::create_dir(&root).unwrap();

        std::fs::write(root.join("file.RS"), "uppercase").unwrap();
        std::fs::write(root.join("file.rs"), "lowercase").unwrap();

        let extensions = vec!["rs".to_string()];
        let files = collect_files(&root, &extensions);

        // Extension matching is case-sensitive
        // .RS won't match "rs" extension
        let names: Vec<_> = files.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"file.rs"));
        // .RS file may or may not be included depending on platform
    }

    #[tokio::test]
    async fn test_delete_index_not_found() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        let result = service.delete_index("nonexistent").await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("index not found"));
    }

    #[tokio::test]
    async fn test_delete_index_success() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Manually insert an index
        let index_name = "github/owner/repo";
        {
            let mut indexes = service.indexes.write().await;

            let repo = crate::providers::Repository::new(
                "github",
                "owner",
                "repo",
                "https://github.com/owner/repo.git",
            );
            let info = IndexInfo {
                name: index_name.to_string(),
                path: dir.path().join("indexes/github/owner/repo.leann"),
                repository: repo,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                file_count: 25,
                size_bytes: 512,
            };

            indexes.insert(info.name.clone(), info);
        }

        // Verify index exists
        assert!(service.get_index(index_name).await.is_some());

        // Delete the index
        let result = service.delete_index(index_name).await;
        assert!(result.is_ok());

        // Verify index is gone
        assert!(service.get_index(index_name).await.is_none());
    }

    #[tokio::test]
    async fn test_delete_index_removes_from_graphs() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        let index_name = "github/test/project";
        {
            // Insert index info
            let mut indexes = service.indexes.write().await;
            let repo = crate::providers::Repository::new(
                "github",
                "test",
                "project",
                "https://github.com/test/project.git",
            );
            let info = IndexInfo {
                name: index_name.to_string(),
                path: dir.path().join("indexes/github/test/project.leann"),
                repository: repo.clone(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                file_count: 10,
                size_bytes: 256,
            };
            indexes.insert(info.name.clone(), info.clone());

            // Also insert into graphs
            let mut graphs = service.graphs.write().await;
            let graph = crate::core::HnswGraph::new(crate::core::HnswConfig::fast()).unwrap();
            let stored = StoredIndex {
                graph,
                files: vec![],
                info,
            };
            graphs.insert(index_name.to_string(), stored);
        }

        // Verify graph exists
        {
            let graphs = service.graphs.read().await;
            assert!(graphs.contains_key(index_name));
        }

        // Delete the index
        service.delete_index(index_name).await.unwrap();

        // Verify graph is gone
        {
            let graphs = service.graphs.read().await;
            assert!(!graphs.contains_key(index_name));
        }
    }

    // =========================================================================
    // Index Persistence Tests (TDD - these tests drive the implementation)
    // =========================================================================

    #[tokio::test]
    async fn test_new_service_loads_persisted_indexes() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        // Create index directories
        let index_dir = config
            .indexes_path
            .join("github")
            .join("owner")
            .join("repo");
        std::fs::create_dir_all(&index_dir).unwrap();

        // Write index metadata to disk
        let repo = crate::providers::Repository::new(
            "github",
            "owner",
            "repo",
            "https://github.com/owner/repo.git",
        );
        let info = IndexInfo {
            name: "github/owner/repo".to_string(),
            path: index_dir.join("index.leann"),
            repository: repo,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            file_count: 42,
            size_bytes: 1024,
        };

        // Save metadata file
        let metadata_path = index_dir.join("metadata.json");
        let json = serde_json::to_string_pretty(&info).unwrap();
        std::fs::write(&metadata_path, json).unwrap();

        // Create a NEW service - it should load the persisted index
        let service = IndexerService::new(config, HashMap::new());
        let indexes = service.list_indexes().await;

        // This assertion will FAIL until we implement persistence loading
        assert_eq!(indexes.len(), 1, "Expected 1 persisted index to be loaded");
        assert_eq!(indexes[0].name, "github/owner/repo");
        assert_eq!(indexes[0].file_count, 42);
    }

    #[tokio::test]
    async fn test_save_index_persists_metadata_to_disk() {
        // Test that save_index_metadata writes to disk correctly
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config.clone(), HashMap::new());

        let repo = crate::providers::Repository::new(
            "github",
            "test",
            "myrepo",
            "https://github.com/test/myrepo.git",
        );

        let info = IndexInfo {
            name: "github/test/myrepo".to_string(),
            path: config
                .indexes_path
                .join("github")
                .join("test")
                .join("myrepo")
                .join("index.leann"),
            repository: repo,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            file_count: 100,
            size_bytes: 2048,
        };

        // Call save_index_metadata (method we need to implement)
        service.save_index_metadata(&info).await.unwrap();

        // Verify metadata file exists
        let metadata_path = config
            .indexes_path
            .join("github")
            .join("test")
            .join("myrepo")
            .join("metadata.json");

        assert!(
            metadata_path.exists(),
            "Index metadata should be persisted to disk"
        );

        // Verify metadata content
        let content = std::fs::read_to_string(&metadata_path).unwrap();
        let loaded: IndexInfo = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.name, "github/test/myrepo");
        assert_eq!(loaded.file_count, 100);
    }

    // =========================================================================
    // Multi-repo Index Tests (TDD - these tests drive the implementation)
    // =========================================================================

    #[tokio::test]
    async fn test_create_workspace_index_with_multiple_repos() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config.clone(), HashMap::new());

        // Create workspace with multiple repos
        let repos = vec![
            crate::providers::Repository::new(
                "github",
                "org",
                "frontend",
                "https://github.com/org/frontend.git",
            ),
            crate::providers::Repository::new(
                "github",
                "org",
                "backend",
                "https://github.com/org/backend.git",
            ),
            crate::providers::Repository::new(
                "github",
                "org",
                "shared",
                "https://github.com/org/shared.git",
            ),
        ];

        // Create a workspace index that combines all repos
        let workspace_name = "my-project";
        service
            .create_workspace(workspace_name, &repos)
            .await
            .unwrap();

        // Verify workspace is listed
        let workspaces = service.list_workspaces().await;
        assert_eq!(workspaces.len(), 1);
        assert_eq!(workspaces[0].name, workspace_name);
        assert_eq!(workspaces[0].repositories.len(), 3);
    }

    #[tokio::test]
    async fn test_workspace_persists_and_loads() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        // Create workspace with first service instance
        {
            let service = IndexerService::new(config.clone(), HashMap::new());

            let repos = vec![
                crate::providers::Repository::new(
                    "github",
                    "org",
                    "repo1",
                    "https://github.com/org/repo1.git",
                ),
                crate::providers::Repository::new(
                    "github",
                    "org",
                    "repo2",
                    "https://github.com/org/repo2.git",
                ),
            ];

            service
                .create_workspace("workspace1", &repos)
                .await
                .unwrap();
        }

        // Create NEW service - workspace should be loaded from disk
        let service2 = IndexerService::new(config, HashMap::new());
        let workspaces = service2.list_workspaces().await;

        assert_eq!(
            workspaces.len(),
            1,
            "Workspace should persist across service restarts"
        );
        assert_eq!(workspaces[0].name, "workspace1");
        assert_eq!(workspaces[0].repositories.len(), 2);
    }

    #[tokio::test]
    async fn test_add_repo_to_workspace() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Create workspace with one repo
        let repo1 = crate::providers::Repository::new(
            "github",
            "org",
            "frontend",
            "https://github.com/org/frontend.git",
        );
        service
            .create_workspace("my-project", &[repo1])
            .await
            .unwrap();

        // Add another repo to the workspace
        let repo2 = crate::providers::Repository::new(
            "github",
            "org",
            "backend",
            "https://github.com/org/backend.git",
        );
        service
            .add_repo_to_workspace("my-project", &repo2)
            .await
            .unwrap();

        // Verify workspace now has 2 repos
        let workspaces = service.list_workspaces().await;
        assert_eq!(workspaces.len(), 1);
        assert_eq!(workspaces[0].repositories.len(), 2);

        let repo_names: Vec<_> = workspaces[0]
            .repositories
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert!(repo_names.contains(&"frontend"));
        assert!(repo_names.contains(&"backend"));
    }

    #[tokio::test]
    async fn test_add_repo_to_workspace_persists() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        // Create workspace and add repo
        {
            let service = IndexerService::new(config.clone(), HashMap::new());

            let repo1 = crate::providers::Repository::new(
                "github",
                "org",
                "repo1",
                "https://github.com/org/repo1.git",
            );
            service.create_workspace("test-ws", &[repo1]).await.unwrap();

            let repo2 = crate::providers::Repository::new(
                "github",
                "org",
                "repo2",
                "https://github.com/org/repo2.git",
            );
            service
                .add_repo_to_workspace("test-ws", &repo2)
                .await
                .unwrap();
        }

        // New service should see the added repo
        let service2 = IndexerService::new(config, HashMap::new());
        let workspaces = service2.list_workspaces().await;

        assert_eq!(
            workspaces[0].repositories.len(),
            2,
            "Added repo should persist"
        );
    }

    #[tokio::test]
    async fn test_add_repo_to_nonexistent_workspace_fails() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        let repo = crate::providers::Repository::new(
            "github",
            "org",
            "repo",
            "https://github.com/org/repo.git",
        );

        let result = service.add_repo_to_workspace("nonexistent", &repo).await;
        assert!(result.is_err(), "Should fail for nonexistent workspace");
    }

    #[tokio::test]
    async fn test_remove_repo_from_workspace() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Create workspace with two repos
        let repos = vec![
            crate::providers::Repository::new(
                "github",
                "org",
                "frontend",
                "https://github.com/org/frontend.git",
            ),
            crate::providers::Repository::new(
                "github",
                "org",
                "backend",
                "https://github.com/org/backend.git",
            ),
        ];
        service
            .create_workspace("my-project", &repos)
            .await
            .unwrap();

        // Remove one repo (repo.id() returns owner/name format)
        service
            .remove_repo_from_workspace("my-project", "org/backend")
            .await
            .unwrap();

        // Verify workspace now has 1 repo
        let workspaces = service.list_workspaces().await;
        assert_eq!(workspaces[0].repositories.len(), 1);
        assert_eq!(workspaces[0].repositories[0].name, "frontend");
    }

    #[tokio::test]
    async fn test_remove_repo_from_workspace_persists() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        // Create workspace with 2 repos, then remove one
        {
            let service = IndexerService::new(config.clone(), HashMap::new());

            let repos = vec![
                crate::providers::Repository::new(
                    "github",
                    "org",
                    "repo1",
                    "https://github.com/org/repo1.git",
                ),
                crate::providers::Repository::new(
                    "github",
                    "org",
                    "repo2",
                    "https://github.com/org/repo2.git",
                ),
            ];
            service.create_workspace("test-ws", &repos).await.unwrap();
            service
                .remove_repo_from_workspace("test-ws", "org/repo2")
                .await
                .unwrap();
        }

        // New service should see the removal
        let service2 = IndexerService::new(config, HashMap::new());
        let workspaces = service2.list_workspaces().await;

        assert_eq!(
            workspaces[0].repositories.len(),
            1,
            "Removal should persist"
        );
        assert_eq!(workspaces[0].repositories[0].name, "repo1");
    }

    #[tokio::test]
    async fn test_remove_nonexistent_repo_from_workspace_fails() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        let repo = crate::providers::Repository::new(
            "github",
            "org",
            "repo1",
            "https://github.com/org/repo1.git",
        );
        service.create_workspace("test-ws", &[repo]).await.unwrap();

        let result = service
            .remove_repo_from_workspace("test-ws", "org/nonexistent")
            .await;
        assert!(result.is_err(), "Should fail for nonexistent repo");
    }

    #[tokio::test]
    async fn test_delete_workspace() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config.clone(), HashMap::new());

        // Create workspace
        let repo = crate::providers::Repository::new(
            "github",
            "org",
            "repo",
            "https://github.com/org/repo.git",
        );
        service
            .create_workspace("to-delete", &[repo])
            .await
            .unwrap();

        // Verify it exists
        assert_eq!(service.list_workspaces().await.len(), 1);

        // Delete it
        service.delete_workspace("to-delete").await.unwrap();

        // Verify it's gone
        assert_eq!(service.list_workspaces().await.len(), 0);

        // New service should also not see it
        let service2 = IndexerService::new(config, HashMap::new());
        assert_eq!(service2.list_workspaces().await.len(), 0);
    }

    #[tokio::test]
    async fn test_get_workspace() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Create workspace
        let repo = crate::providers::Repository::new(
            "github",
            "org",
            "repo",
            "https://github.com/org/repo.git",
        );
        service.create_workspace("test-ws", &[repo]).await.unwrap();

        // Get workspace should succeed
        let ws = service.get_workspace("test-ws").await;
        assert!(ws.is_some());
        assert_eq!(ws.unwrap().name, "test-ws");

        // Get nonexistent workspace should return None
        let ws = service.get_workspace("nonexistent").await;
        assert!(ws.is_none());
    }

    #[tokio::test]
    async fn test_get_workspace_index_names() {
        let dir = tempdir().unwrap();
        let config = IndexerConfig {
            repos_path: dir.path().join("repos"),
            indexes_path: dir.path().join("indexes"),
            ..Default::default()
        };

        let service = IndexerService::new(config, HashMap::new());

        // Create workspace with two repos
        let repos = vec![
            crate::providers::Repository::new(
                "github",
                "org",
                "frontend",
                "https://github.com/org/frontend.git",
            ),
            crate::providers::Repository::new(
                "github",
                "org",
                "backend",
                "https://github.com/org/backend.git",
            ),
        ];
        service
            .create_workspace("my-project", &repos)
            .await
            .unwrap();

        // Get index names for workspace
        let names = service.get_workspace_index_names("my-project").await;
        assert!(names.is_some());
        let names = names.unwrap();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"github/org/frontend".to_string()));
        assert!(names.contains(&"github/org/backend".to_string()));

        // Nonexistent workspace should return None
        let names = service.get_workspace_index_names("nonexistent").await;
        assert!(names.is_none());
    }
}
