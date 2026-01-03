# LEANN Research Notes

## Overview

LEANN (Low-Storage Vector Database) is a vector search system enabling personal AI applications with extreme storage efficiency. It uses 97% less storage than traditional solutions without accuracy loss.

## Core Technology

### Graph-Based Selective Recomputation

Rather than storing all embeddings permanently, LEANN maintains a pruned graph structure and recalculates embeddings only during search operations. This combines:

1. **High-degree preserving pruning**: Maintains important graph connections
2. **Dynamic batching**: GPU acceleration for efficient computation
3. **Lazy evaluation**: Only compute embeddings when needed

### Key Components

```
LeannBuilder()      # Index construction
LeannSearcher()     # Semantic search
LeannChat()         # LLM-based querying
```

## Integration Strategy

For Pythia, we integrate LEANN via:

1. **Python subprocess**: Call LEANN CLI for indexing and search
2. **IPC protocol**: JSON-based communication for queries
3. **Caching layer**: Local cache for frequently accessed embeddings

## Performance Characteristics

- **Storage**: 97% reduction compared to traditional vector DBs
- **Latency**: Sub-second for typical queries
- **Accuracy**: Comparable to full embedding storage

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| model_name | Sentence transformer model | all-MiniLM-L6-v2 |
| device_type | CPU or CUDA | cpu |
| batch_size | Processing batch size | 32 |
| compression | Enable storage compression | true |
| pruning_factor | Graph pruning aggressiveness | 0.3 |

## References

- [LEANN GitHub](https://github.com/yichuan-w/LEANN)
- [LEANN Paper](https://arxiv.org/abs/2412.xxxxx)
