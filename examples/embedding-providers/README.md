# Embedding Provider Configuration

Configure different embedding providers for semantic search.

## Available Providers

Islands supports multiple embedding providers through the `embed_anything` crate:

| Provider | Type | Models | Cost |
|----------|------|--------|------|
| **Local** | On-device | bge-small, bge-base, bge-large, minilm, jina-small, nomic | Free |
| **OpenAI** | Cloud API | text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 | Per token |
| **Cohere** | Cloud API | embed-english-v3.0, embed-multilingual-v3.0 | Per token |

## Local Embeddings

Best for development and privacy-sensitive use cases.

```rust
use islands_indexer::EmbeddingConfig;

// Use a preset model (downloaded automatically)
let config = EmbeddingConfig::Local {
    model: "bge-small".to_string(),  // ~130MB, 384 dimensions
    batch_size: 32,
};

// Or use any HuggingFace model
let config = EmbeddingConfig::Local {
    model: "BAAI/bge-m3".to_string(),  // Multilingual
    batch_size: 16,
};
```

### Available Presets

| Preset | Dimensions | Size | Speed | Quality |
|--------|------------|------|-------|---------|
| `bge-small` | 384 | ~130MB | Fast | Good |
| `bge-base` | 768 | ~440MB | Medium | Better |
| `bge-large` | 1024 | ~1.3GB | Slow | Best |
| `minilm` | 384 | ~90MB | Fastest | Fair |
| `jina-small` | 512 | ~150MB | Fast | Good |
| `nomic` | 768 | ~550MB | Medium | Good |

## OpenAI Embeddings

Best for production with high-quality embeddings.

```rust
use islands_indexer::EmbeddingConfig;

let config = EmbeddingConfig::OpenAI {
    model: "text-embedding-3-small".to_string(),
    api_key: Some(std::env::var("OPENAI_API_KEY").ok()),
    batch_size: 100,
};
```

### Models

| Model | Dimensions | Quality | Cost |
|-------|------------|---------|------|
| `text-embedding-3-small` | 1536 | Good | $0.02/1M tokens |
| `text-embedding-3-large` | 3072 | Best | $0.13/1M tokens |
| `text-embedding-ada-002` | 1536 | Good | $0.10/1M tokens |

## Cohere Embeddings

Excellent multilingual support.

```rust
use islands_indexer::EmbeddingConfig;

let config = EmbeddingConfig::Cohere {
    model: "embed-english-v3.0".to_string(),
    api_key: Some(std::env::var("COHERE_API_KEY").ok()),
    batch_size: 96,
};
```

### Models

| Model | Dimensions | Languages |
|-------|------------|-----------|
| `embed-english-v3.0` | 1024 | English |
| `embed-multilingual-v3.0` | 1024 | 100+ languages |

## Configuration File

```toml
# config.toml

# Option 1: Local embeddings (default)
[indexer.embedding]
provider = "local"
model = "bge-small"
batch_size = 32

# Option 2: OpenAI
# [indexer.embedding]
# provider = "openai"
# model = "text-embedding-3-small"
# batch_size = 100

# Option 3: Cohere
# [indexer.embedding]
# provider = "cohere"
# model = "embed-english-v3.0"
# batch_size = 96
```

## Environment Variables

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Cohere
export COHERE_API_KEY="..."
```

## Switching Providers

You can switch providers by updating the configuration. Note that switching providers requires re-indexing since embedding dimensions differ.

```rust
// Check current provider
println!("Using: {:?}", indexer_config.embedding);

// Indexes are provider-specific
// data/indexes/openai/...
// data/indexes/local/...
```

## Performance Comparison

| Provider | Indexing Speed | Search Speed | Quality |
|----------|---------------|--------------|---------|
| Local (bge-small) | ~1000 files/min | <10ms | Good |
| Local (bge-large) | ~200 files/min | <10ms | Excellent |
| OpenAI | ~500 files/min* | <10ms | Excellent |
| Cohere | ~500 files/min* | <10ms | Excellent |

*Depends on network latency and rate limits

## Cost Estimation

For a codebase with 10,000 files averaging 200 tokens each:

| Provider | Indexing Cost | Re-indexing |
|----------|--------------|-------------|
| Local | $0 | $0 |
| OpenAI (3-small) | ~$0.04 | ~$0.04 |
| OpenAI (3-large) | ~$0.26 | ~$0.26 |
| Cohere | ~$0.02 | ~$0.02 |
