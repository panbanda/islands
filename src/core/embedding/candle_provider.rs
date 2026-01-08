//! Native Candle embedding provider for sentence-transformers models.
//!
//! This module provides a self-contained embedding solution using the Candle ML
//! framework for inference. It supports HuggingFace sentence-transformer models
//! with automatic model download and caching.
//!
//! # Features
//!
//! - Local inference with no API keys required
//! - Native Candle implementation
//! - Automatic model download from HuggingFace Hub
//! - Support for GPU acceleration (Metal, CUDA)
//! - Compatible with sentence-transformers models (BERT-based)
//!
//! # Example
//!
//! ```rust,no_run
//! use islands::core::embedding::{CandleEmbedder, CandleEmbedderConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Use the default all-MiniLM-L6-v2 model
//! let embedder = CandleEmbedder::new(CandleEmbedderConfig::default()).await?;
//!
//! // Embed text
//! let embeddings = embedder.embed_texts(&["Hello world", "Rust is great"]).await?;
//! # Ok(())
//! # }
//! ```

use crate::core::error::{CoreError, CoreResult};
use crate::Embedding;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Supported sentence-transformer models for Candle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CandleModel {
    /// all-MiniLM-L6-v2: 384 dimensions, fast and efficient
    AllMiniLmL6V2,
    /// all-MiniLM-L12-v2: 384 dimensions, slightly more accurate
    AllMiniLmL12V2,
    /// paraphrase-MiniLM-L6-v2: 384 dimensions, optimized for paraphrase
    ParaphraseMiniLmL6V2,
    /// BAAI/bge-small-en-v1.5: 384 dimensions, high quality
    BgeSmallEnV15,
    /// BAAI/bge-base-en-v1.5: 768 dimensions, balanced
    BgeBaseEnV15,
    /// BAAI/bge-large-en-v1.5: 1024 dimensions, highest quality
    BgeLargeEnV15,
    /// Custom HuggingFace model ID
    Custom {
        /// HuggingFace model repository ID
        model_id: String,
        /// Expected embedding dimension
        dimension: usize,
    },
}

impl Default for CandleModel {
    fn default() -> Self {
        Self::AllMiniLmL6V2
    }
}

impl CandleModel {
    /// Get the HuggingFace model ID for this model.
    #[must_use]
    pub fn model_id(&self) -> &str {
        match self {
            Self::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::AllMiniLmL12V2 => "sentence-transformers/all-MiniLM-L12-v2",
            Self::ParaphraseMiniLmL6V2 => "sentence-transformers/paraphrase-MiniLM-L6-v2",
            Self::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            Self::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            Self::BgeLargeEnV15 => "BAAI/bge-large-en-v1.5",
            Self::Custom { model_id, .. } => model_id,
        }
    }

    /// Get the embedding dimension for this model.
    #[must_use]
    pub fn dimension(&self) -> usize {
        match self {
            Self::AllMiniLmL6V2 => 384,
            Self::AllMiniLmL12V2 => 384,
            Self::ParaphraseMiniLmL6V2 => 384,
            Self::BgeSmallEnV15 => 384,
            Self::BgeBaseEnV15 => 768,
            Self::BgeLargeEnV15 => 1024,
            Self::Custom { dimension, .. } => *dimension,
        }
    }

    /// Create a model configuration from a preset name.
    ///
    /// Supported presets: "minilm", "minilm-l12", "paraphrase", "bge-small", "bge-base", "bge-large"
    pub fn from_preset(name: &str) -> CoreResult<Self> {
        match name.to_lowercase().as_str() {
            "minilm" | "all-minilm-l6-v2" => Ok(Self::AllMiniLmL6V2),
            "minilm-l12" | "all-minilm-l12-v2" => Ok(Self::AllMiniLmL12V2),
            "paraphrase" | "paraphrase-minilm" => Ok(Self::ParaphraseMiniLmL6V2),
            "bge-small" => Ok(Self::BgeSmallEnV15),
            "bge-base" => Ok(Self::BgeBaseEnV15),
            "bge-large" => Ok(Self::BgeLargeEnV15),
            _ => Err(CoreError::InvalidConfig(format!(
                "Unknown Candle model preset: {name}. Use 'minilm', 'minilm-l12', 'paraphrase', \
                 'bge-small', 'bge-base', or 'bge-large'."
            ))),
        }
    }
}

/// Device configuration for Candle inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CandleDevice {
    /// CPU inference (default)
    #[default]
    Cpu,
    /// Metal GPU acceleration (macOS)
    Metal,
    /// CUDA GPU acceleration (NVIDIA)
    Cuda,
}

impl CandleDevice {
    /// Convert to Candle Device.
    fn to_candle_device(self) -> CoreResult<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "candle-metal")]
            Self::Metal => Device::new_metal(0)
                .map_err(|e| CoreError::EmbeddingError(format!("Failed to create Metal device: {e}"))),
            #[cfg(not(feature = "candle-metal"))]
            Self::Metal => {
                tracing::warn!("Metal not available, falling back to CPU");
                Ok(Device::Cpu)
            }
            #[cfg(feature = "candle-cuda")]
            Self::Cuda => Device::new_cuda(0)
                .map_err(|e| CoreError::EmbeddingError(format!("Failed to create CUDA device: {e}"))),
            #[cfg(not(feature = "candle-cuda"))]
            Self::Cuda => {
                tracing::warn!("CUDA not available, falling back to CPU");
                Ok(Device::Cpu)
            }
        }
    }
}

/// Configuration for the Candle embedder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleEmbedderConfig {
    /// Model to use for embeddings
    pub model: CandleModel,
    /// Device for inference
    pub device: CandleDevice,
    /// Whether to normalize embeddings to unit length
    pub normalize: bool,
}

impl Default for CandleEmbedderConfig {
    fn default() -> Self {
        Self {
            model: CandleModel::default(),
            device: CandleDevice::default(),
            normalize: true,
        }
    }
}

impl CandleEmbedderConfig {
    /// Create config with a specific model.
    #[must_use]
    pub fn with_model(mut self, model: CandleModel) -> Self {
        self.model = model;
        self
    }

    /// Create config from a preset name.
    pub fn from_preset(name: &str) -> CoreResult<Self> {
        Ok(Self {
            model: CandleModel::from_preset(name)?,
            ..Default::default()
        })
    }

    /// Set the inference device.
    #[must_use]
    pub fn with_device(mut self, device: CandleDevice) -> Self {
        self.device = device;
        self
    }

    /// Disable embedding normalization.
    #[must_use]
    pub fn without_normalization(mut self) -> Self {
        self.normalize = false;
        self
    }
}

/// Internal model state
struct ModelState {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

/// Native Candle embedding provider.
///
/// Provides high-performance sentence embeddings using the Candle ML framework.
/// Models are automatically downloaded from HuggingFace Hub on first use.
pub struct CandleEmbedder {
    state: Mutex<ModelState>,
    config: CandleEmbedderConfig,
}

impl std::fmt::Debug for CandleEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleEmbedder")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl CandleEmbedder {
    /// Create a new Candle embedder with the given configuration.
    ///
    /// This will download the model from HuggingFace if not already cached.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub async fn new(config: CandleEmbedderConfig) -> CoreResult<Self> {
        let device = config.device.to_candle_device()?;
        let model_id = config.model.model_id().to_string();

        tracing::info!("Loading Candle model: {} on {:?}", model_id, config.device);

        // Download model files from HuggingFace Hub
        let api = Api::new()
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to create HF API: {e}")))?;
        let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model));

        // Download required files
        let config_path = repo
            .get("config.json")
            .await
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to download config.json: {e}")))?;

        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to download tokenizer.json: {e}")))?;

        let weights_path = repo
            .get("model.safetensors")
            .await
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to download model.safetensors: {e}")))?;

        // Load configuration
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to read config: {e}")))?;
        let bert_config: BertConfig = serde_json::from_str(&config_content)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to parse config: {e}")))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to load tokenizer: {e}")))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| CoreError::EmbeddingError(format!("Failed to load weights: {e}")))?
        };

        // Create BERT model
        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to create BERT model: {e}")))?;

        tracing::info!(
            "Candle model loaded: {} (dimension: {})",
            model_id,
            config.model.dimension()
        );

        Ok(Self {
            state: Mutex::new(ModelState {
                model,
                tokenizer,
                device,
            }),
            config,
        })
    }

    /// Create an embedder with the default all-MiniLM-L6-v2 model.
    pub async fn default_model() -> CoreResult<Self> {
        Self::new(CandleEmbedderConfig::default()).await
    }

    /// Create an embedder from a preset name.
    pub async fn from_preset(name: &str) -> CoreResult<Self> {
        Self::new(CandleEmbedderConfig::from_preset(name)?).await
    }

    /// Get the embedding dimension for this model.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.config.model.dimension()
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &CandleEmbedderConfig {
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
    /// # Errors
    ///
    /// Returns an error if embedding computation fails.
    pub async fn embed_texts(&self, texts: &[&str]) -> CoreResult<Vec<Embedding>> {
        let raw = self.embed_texts_raw(texts).await?;
        Ok(raw.into_iter().map(Embedding::new).collect())
    }

    /// Embed texts and return raw f32 vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding computation fails.
    pub async fn embed_texts_raw(&self, texts: &[&str]) -> CoreResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let state = self
            .state
            .lock()
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to lock model state: {e}")))?;

        let texts_owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();

        // Tokenize all texts
        let encodings = state
            .tokenizer
            .encode_batch(texts_owned, true)
            .map_err(|e| CoreError::EmbeddingError(format!("Tokenization failed: {e}")))?;

        let batch_size = encodings.len();

        // Find max sequence length in this batch
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        // Prepare input tensors
        let mut all_input_ids = Vec::with_capacity(batch_size * max_len);
        let mut all_token_type_ids = Vec::with_capacity(batch_size * max_len);
        let mut all_attention_mask = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let type_ids = encoding.get_type_ids();
            let mask = encoding.get_attention_mask();

            // Pad to max_len
            for i in 0..max_len {
                if i < ids.len() {
                    all_input_ids.push(ids[i] as i64);
                    all_token_type_ids.push(type_ids[i] as i64);
                    all_attention_mask.push(mask[i] as i64);
                } else {
                    all_input_ids.push(0i64);
                    all_token_type_ids.push(0i64);
                    all_attention_mask.push(0i64);
                }
            }
        }

        // Create tensors
        let input_ids = Tensor::from_vec(all_input_ids, (batch_size, max_len), &state.device)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to create input_ids tensor: {e}")))?;

        let token_type_ids = Tensor::from_vec(all_token_type_ids, (batch_size, max_len), &state.device)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to create token_type_ids tensor: {e}")))?;

        let attention_mask = Tensor::from_vec(all_attention_mask.clone(), (batch_size, max_len), &state.device)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to create attention_mask tensor: {e}")))?;

        // Run model forward pass
        let output = state
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| CoreError::EmbeddingError(format!("Model forward pass failed: {e}")))?;

        // Mean pooling: average over sequence dimension, weighted by attention mask
        let attention_mask_f32 = Tensor::from_vec(
            all_attention_mask.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
            (batch_size, max_len),
            &state.device,
        )
        .map_err(|e| CoreError::EmbeddingError(format!("Failed to create attention mask f32: {e}")))?;

        // Expand attention mask to match hidden size
        let mask_expanded = attention_mask_f32
            .unsqueeze(2)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to expand mask: {e}")))?
            .broadcast_as(output.dims())
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to broadcast mask: {e}")))?;

        // Sum embeddings weighted by mask
        let sum_embeddings = (output.clone() * mask_expanded.clone())
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to multiply: {e}")))?
            .sum(1)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to sum: {e}")))?;

        // Sum of mask for normalization
        let sum_mask = mask_expanded
            .sum(1)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to sum mask: {e}")))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to clamp: {e}")))?;

        // Mean pooling
        let mean_embeddings = (sum_embeddings / sum_mask)
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to divide: {e}")))?;

        // Optional L2 normalization
        let final_embeddings = if self.config.normalize {
            let norm = mean_embeddings
                .sqr()
                .map_err(|e| CoreError::EmbeddingError(format!("Failed to square: {e}")))?
                .sum_keepdim(1)
                .map_err(|e| CoreError::EmbeddingError(format!("Failed to sum for norm: {e}")))?
                .sqrt()
                .map_err(|e| CoreError::EmbeddingError(format!("Failed to sqrt: {e}")))?
                .clamp(1e-12, f64::MAX)
                .map_err(|e| CoreError::EmbeddingError(format!("Failed to clamp norm: {e}")))?;

            mean_embeddings
                .broadcast_div(&norm)
                .map_err(|e| CoreError::EmbeddingError(format!("Failed to normalize: {e}")))?
        } else {
            mean_embeddings
        };

        // Convert to Vec<Vec<f32>>
        let shape = final_embeddings.dims();
        let flat: Vec<f32> = final_embeddings
            .flatten_all()
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to flatten: {e}")))?
            .to_vec1()
            .map_err(|e| CoreError::EmbeddingError(format!("Failed to convert to vec: {e}")))?;

        let dim = shape[1];
        let mut result = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * dim;
            let end = start + dim;
            result.push(flat[start..end].to_vec());
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CandleEmbedderConfig::default();
        assert_eq!(config.model, CandleModel::AllMiniLmL6V2);
        assert_eq!(config.device, CandleDevice::Cpu);
        assert!(config.normalize);
    }

    #[test]
    fn test_model_presets() {
        assert_eq!(
            CandleModel::from_preset("minilm").unwrap(),
            CandleModel::AllMiniLmL6V2
        );
        assert_eq!(
            CandleModel::from_preset("bge-small").unwrap(),
            CandleModel::BgeSmallEnV15
        );
        assert_eq!(
            CandleModel::from_preset("bge-large").unwrap(),
            CandleModel::BgeLargeEnV15
        );
    }

    #[test]
    fn test_model_dimensions() {
        assert_eq!(CandleModel::AllMiniLmL6V2.dimension(), 384);
        assert_eq!(CandleModel::BgeBaseEnV15.dimension(), 768);
        assert_eq!(CandleModel::BgeLargeEnV15.dimension(), 1024);
    }

    #[test]
    fn test_model_ids() {
        assert_eq!(
            CandleModel::AllMiniLmL6V2.model_id(),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        assert_eq!(CandleModel::BgeSmallEnV15.model_id(), "BAAI/bge-small-en-v1.5");
    }

    #[test]
    fn test_custom_model() {
        let model = CandleModel::Custom {
            model_id: "custom/model".to_string(),
            dimension: 512,
        };
        assert_eq!(model.model_id(), "custom/model");
        assert_eq!(model.dimension(), 512);
    }

    #[test]
    fn test_config_builder() {
        let config = CandleEmbedderConfig::default()
            .with_model(CandleModel::BgeBaseEnV15)
            .with_device(CandleDevice::Cpu)
            .without_normalization();

        assert_eq!(config.model, CandleModel::BgeBaseEnV15);
        assert_eq!(config.device, CandleDevice::Cpu);
        assert!(!config.normalize);
    }

    #[test]
    fn test_unknown_preset() {
        let result = CandleModel::from_preset("unknown-model");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = CandleEmbedderConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: CandleEmbedderConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, config.model);
        assert_eq!(parsed.device, config.device);
    }
}
