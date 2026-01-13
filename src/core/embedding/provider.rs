//! Embedding providers for generating vector embeddings from text.
//!
//! This module provides the [`EmbedderProvider`] enum and associated types for
//! computing embeddings using various backends supported by `embed_anything`.
//!
//! # Supported Backends
//!
//! - **Candle**: Run any HuggingFace model locally via the Candle framework
//! - **ONNX**: Optimized ONNX runtime inference for supported models
//! - **Cloud**: OpenAI, Cohere, and other cloud providers
//!
//! # Example
//!
//! ```rust,ignore
//! use islands::core::embedding::{EmbedderProvider, EmbedderConfig, ModelArchitecture};
//!
//! async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a provider with a HuggingFace model
//!     let provider = EmbedderProvider::from_config(EmbedderConfig {
//!         architecture: ModelArchitecture::Bert,
//!         model_id: "BAAI/bge-small-en-v1.5".to_string(),
//!         ..Default::default()
//!     }).await?;
//!
//!     // Embed text
//!     let embeddings = provider.embed_texts(&["Hello world", "Rust is great"]).await?;
//!     Ok(())
//! }
//! ```

use crate::Embedding;
use crate::core::error::{CoreError, CoreResult};
use embed_anything::config::TextEmbedConfig;
use embed_anything::embeddings::embed::{EmbedData, Embedder, EmbedderBuilder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Model architecture supported by embed_anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    /// BERT-based models (bge, minilm, etc.)
    #[default]
    Bert,
    /// Jina embedding models
    Jina,
    /// Sentence-transformers CLIP
    Clip,
    /// ColBERT late-interaction models
    ColBert,
    /// ColPali vision-language models
    ColPali,
    /// SPLADE sparse models
    Splade,
    /// ModernBERT models
    ModernBert,
}

impl ModelArchitecture {
    /// Convert to the string format expected by embed_anything.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bert => "bert",
            Self::Jina => "jina",
            Self::Clip => "clip",
            Self::ColBert => "colbert",
            Self::ColPali => "colpali",
            Self::Splade => "splade",
            Self::ModernBert => "modernbert",
        }
    }
}

/// Backend for running inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InferenceBackend {
    /// Candle framework (default, supports any HuggingFace model)
    #[default]
    Candle,
    /// ONNX runtime (faster, but requires ONNX-format models)
    Onnx,
    /// Cloud provider (OpenAI, Cohere, etc.)
    Cloud,
}

/// Cloud provider for API-based embeddings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CloudProvider {
    /// OpenAI API
    OpenAI {
        /// API key (defaults to OPENAI_API_KEY env var)
        api_key: Option<String>,
    },
    /// Cohere API
    Cohere {
        /// API key (defaults to COHERE_API_KEY env var)
        api_key: Option<String>,
    },
}

/// Configuration for creating an embedder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    /// Model architecture (bert, jina, etc.)
    pub architecture: ModelArchitecture,

    /// HuggingFace model ID or path
    pub model_id: String,

    /// Inference backend (candle, onnx, or cloud)
    pub backend: InferenceBackend,

    /// Batch size for processing multiple texts
    pub batch_size: usize,

    /// Chunk size for splitting long texts (in tokens)
    pub chunk_size: Option<usize>,

    /// Overlap between chunks (in tokens)
    pub chunk_overlap: Option<usize>,

    /// Optional cloud provider (sets backend to Cloud)
    pub cloud_provider: Option<CloudProvider>,

    /// Revision/branch for HuggingFace model
    pub revision: Option<String>,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Bert,
            model_id: "BAAI/bge-small-en-v1.5".to_string(),
            backend: InferenceBackend::Candle,
            batch_size: 32,
            chunk_size: None,
            chunk_overlap: None,
            cloud_provider: None,
            revision: None,
        }
    }
}

impl EmbedderConfig {
    /// Create a config for a BERT-based model.
    #[must_use]
    pub fn bert(model_id: impl Into<String>) -> Self {
        Self {
            architecture: ModelArchitecture::Bert,
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    /// Create a config for a Jina model.
    #[must_use]
    pub fn jina(model_id: impl Into<String>) -> Self {
        Self {
            architecture: ModelArchitecture::Jina,
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    /// Create a config for OpenAI embeddings.
    #[must_use]
    pub fn openai(model: impl Into<String>) -> Self {
        Self {
            architecture: ModelArchitecture::Bert, // Not used for cloud
            model_id: model.into(),
            backend: InferenceBackend::Cloud,
            cloud_provider: Some(CloudProvider::OpenAI { api_key: None }),
            ..Default::default()
        }
    }

    /// Set the batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set chunk size for text splitting.
    #[must_use]
    pub fn with_chunking(mut self, chunk_size: usize, overlap: usize) -> Self {
        self.chunk_size = Some(chunk_size);
        self.chunk_overlap = Some(overlap);
        self
    }

    /// Use ONNX backend instead of Candle.
    #[must_use]
    pub fn with_onnx(mut self) -> Self {
        self.backend = InferenceBackend::Onnx;
        self
    }
}

/// Embedding provider that wraps embed_anything.
///
/// This is the main interface for computing embeddings. It supports both local
/// models (via Candle or ONNX) and cloud APIs (OpenAI, Cohere).
pub struct EmbedderProvider {
    embedder: Arc<Embedder>,
    config: EmbedderConfig,
    dimension: usize,
}

impl std::fmt::Debug for EmbedderProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedderProvider")
            .field("config", &self.config)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl EmbedderProvider {
    /// Create a new embedder provider from configuration.
    ///
    /// This will download the model from HuggingFace if not already cached.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded or is not supported.
    pub async fn from_config(config: EmbedderConfig) -> CoreResult<Self> {
        let embedder = match config.backend {
            InferenceBackend::Cloud => Self::create_cloud_embedder(&config).await?,
            InferenceBackend::Onnx => Self::create_onnx_embedder(&config).await?,
            InferenceBackend::Candle => Self::create_hf_embedder(&config).await?,
        };

        // Probe dimension by embedding a test string
        let dimension = Self::probe_dimension(&embedder).await?;

        Ok(Self {
            embedder: Arc::new(embedder),
            config,
            dimension,
        })
    }

    /// Create a provider with a pre-configured popular model.
    ///
    /// # Supported presets:
    /// - `"bge-small"` - BAAI/bge-small-en-v1.5 (384 dim, fast)
    /// - `"bge-base"` - BAAI/bge-base-en-v1.5 (768 dim, balanced)
    /// - `"bge-large"` - BAAI/bge-large-en-v1.5 (1024 dim, accurate)
    /// - `"minilm"` - sentence-transformers/all-MiniLM-L6-v2 (384 dim, fast)
    /// - `"jina-small"` - jinaai/jina-embeddings-v2-small-en (512 dim)
    /// - `"nomic"` - nomic-ai/nomic-embed-text-v1.5 (768 dim)
    ///
    /// # Errors
    ///
    /// Returns an error if the model name is not recognized or cannot be loaded.
    pub async fn from_preset(name: &str) -> CoreResult<Self> {
        let config = match name {
            "bge-small" => EmbedderConfig::bert("BAAI/bge-small-en-v1.5"),
            "bge-base" => EmbedderConfig::bert("BAAI/bge-base-en-v1.5"),
            "bge-large" => EmbedderConfig::bert("BAAI/bge-large-en-v1.5"),
            "minilm" => EmbedderConfig::bert("sentence-transformers/all-MiniLM-L6-v2"),
            "jina-small" => EmbedderConfig::jina("jinaai/jina-embeddings-v2-small-en"),
            "nomic" => EmbedderConfig::bert("nomic-ai/nomic-embed-text-v1.5"),
            _ => {
                return Err(CoreError::InvalidConfig(format!(
                    "Unknown preset model: {name}. Use 'bge-small', 'bge-base', 'bge-large', \
                     'minilm', 'jina-small', or 'nomic'."
                )));
            }
        };

        Self::from_config(config).await
    }

    /// Get the embedding dimension for this model.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the configuration used to create this provider.
    #[must_use]
    pub fn config(&self) -> &EmbedderConfig {
        &self.config
    }

    /// Embed a single text string.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding computation fails.
    pub async fn embed_text(&self, text: &str) -> CoreResult<Embedding> {
        let results = self.embed_texts(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| CoreError::EmbeddingError("No embedding returned".to_string()))
    }

    /// Embed multiple text strings.
    ///
    /// This is more efficient than calling `embed_text` multiple times as it
    /// batches the computation.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding computation fails.
    pub async fn embed_texts(&self, texts: &[&str]) -> CoreResult<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let embed_config = TextEmbedConfig::default();
        let embedder = Arc::clone(&self.embedder);
        let texts_owned: Vec<&str> = texts.to_vec();

        let results: Vec<EmbedData> =
            embed_anything::embed_query(&texts_owned, &embedder, Some(&embed_config))
                .await
                .map_err(|e| CoreError::EmbeddingError(format!("Embedding failed: {e}")))?;

        // Convert EmbedData to our Embedding type
        let embeddings: Vec<Embedding> = results
            .into_iter()
            .filter_map(|data| data.embedding.to_dense().ok().map(Embedding::new))
            .collect();

        Ok(embeddings)
    }

    /// Embed texts and return raw f32 vectors (more efficient for LEANN).
    ///
    /// # Errors
    ///
    /// Returns an error if embedding computation fails.
    pub async fn embed_texts_raw(&self, texts: &[&str]) -> CoreResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let embed_config = TextEmbedConfig::default();
        let embedder = Arc::clone(&self.embedder);

        let results: Vec<EmbedData> =
            embed_anything::embed_query(texts, &embedder, Some(&embed_config))
                .await
                .map_err(|e| CoreError::EmbeddingError(format!("Embedding failed: {e}")))?;

        let vectors: Vec<Vec<f32>> = results
            .into_iter()
            .filter_map(|data| data.embedding.to_dense().ok())
            .collect();

        Ok(vectors)
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    async fn create_hf_embedder(config: &EmbedderConfig) -> CoreResult<Embedder> {
        let arch = config.architecture.as_str();
        let model_id = config.model_id.clone();
        let revision = config.revision.clone();

        tokio::task::spawn_blocking(move || {
            let mut builder = EmbedderBuilder::new()
                .model_architecture(arch)
                .model_id(Some(&model_id));

            if let Some(ref rev) = revision {
                builder = builder.revision(Some(rev));
            }

            builder
                .from_pretrained_hf()
                .map_err(|e| CoreError::EmbeddingError(format!("Failed to load HF model: {e}")))
        })
        .await
        .map_err(|e| CoreError::EmbeddingError(format!("Task join error: {e}")))?
    }

    async fn create_onnx_embedder(config: &EmbedderConfig) -> CoreResult<Embedder> {
        let model_id = config.model_id.clone();

        tokio::task::spawn_blocking(move || {
            EmbedderBuilder::new()
                .model_id(Some(&model_id))
                .from_pretrained_onnx()
                .map_err(|e| CoreError::EmbeddingError(format!("Failed to load ONNX model: {e}")))
        })
        .await
        .map_err(|e| CoreError::EmbeddingError(format!("Task join error: {e}")))?
    }

    async fn create_cloud_embedder(config: &EmbedderConfig) -> CoreResult<Embedder> {
        let model_id = config.model_id.clone();
        let api_key = config.cloud_provider.as_ref().and_then(|p| match p {
            CloudProvider::OpenAI { api_key } => api_key.clone(),
            CloudProvider::Cohere { api_key } => api_key.clone(),
        });

        tokio::task::spawn_blocking(move || {
            let mut builder = EmbedderBuilder::new().model_id(Some(&model_id));

            if let Some(key) = api_key {
                builder = builder.api_key(Some(&key));
            }

            builder.from_pretrained_cloud().map_err(|e| {
                CoreError::EmbeddingError(format!("Failed to create cloud embedder: {e}"))
            })
        })
        .await
        .map_err(|e| CoreError::EmbeddingError(format!("Task join error: {e}")))?
    }

    async fn probe_dimension(embedder: &Embedder) -> CoreResult<usize> {
        let config = TextEmbedConfig::default();
        let test_texts: &[&str] = &["test"];

        let results = embed_anything::embed_query(test_texts, embedder, Some(&config))
            .await
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to probe dimension: {e}")))?;

        let dim = results
            .first()
            .and_then(|d| d.embedding.to_dense().ok())
            .map(|v: Vec<f32>| v.len())
            .unwrap_or(0);

        if dim == 0 {
            return Err(CoreError::EmbeddingError(
                "Could not determine embedding dimension".to_string(),
            ));
        }

        Ok(dim)
    }
}

/// Implementation of the LEANN EmbeddingProvider trait.
///
/// This allows `EmbedderProvider` to be used directly with `LeannIndex` for
/// on-demand embedding recomputation during search.
impl crate::core::leann::EmbeddingProvider for EmbedderProvider {
    fn compute_embedding(&self, id: u64) -> CoreResult<Vec<f32>> {
        // For LEANN, we need the original text associated with this ID.
        // This is typically handled by the indexer, not the embedder.
        // Return an error indicating this method requires text lookup.
        Err(CoreError::EmbeddingError(format!(
            "EmbedderProvider::compute_embedding requires text lookup for id {id}. \
             Use EmbeddingProviderWithStorage instead."
        )))
    }

    fn compute_embeddings_batch(&self, ids: &[u64]) -> CoreResult<Vec<Vec<f32>>> {
        Err(CoreError::EmbeddingError(format!(
            "EmbedderProvider::compute_embeddings_batch requires text lookup for {} ids. \
             Use EmbeddingProviderWithStorage instead.",
            ids.len()
        )))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = EmbedderConfig::default();
        assert_eq!(config.architecture, ModelArchitecture::Bert);
        assert_eq!(config.model_id, "BAAI/bge-small-en-v1.5");
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_config_builder() {
        let config = EmbedderConfig::bert("custom/model")
            .with_batch_size(64)
            .with_chunking(512, 50)
            .with_onnx();

        assert_eq!(config.model_id, "custom/model");
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.chunk_size, Some(512));
        assert_eq!(config.chunk_overlap, Some(50));
        assert_eq!(config.backend, InferenceBackend::Onnx);
    }

    #[test]
    fn test_openai_config() {
        let config = EmbedderConfig::openai("text-embedding-3-small");
        assert!(config.cloud_provider.is_some());
        assert_eq!(config.model_id, "text-embedding-3-small");
        assert_eq!(config.backend, InferenceBackend::Cloud);
    }

    #[test]
    fn test_architecture_as_str() {
        assert_eq!(ModelArchitecture::Bert.as_str(), "bert");
        assert_eq!(ModelArchitecture::Jina.as_str(), "jina");
        assert_eq!(ModelArchitecture::Splade.as_str(), "splade");
    }

    #[test]
    fn test_config_serialization() {
        let config = EmbedderConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: EmbedderConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_id, config.model_id);
    }
}
