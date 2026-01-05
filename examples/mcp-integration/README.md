# MCP Server Integration

Use Pythia as a Model Context Protocol (MCP) server for AI assistants.

## Overview

The MCP server exposes Pythia's search capabilities as tools that AI assistants (like Claude) can use to search and understand codebases.

## Available Tools

| Tool | Description |
|------|-------------|
| `search` | Semantic search across indexed repositories |
| `get_file` | Retrieve full file contents |
| `list_indexes` | List all indexed repositories |
| `index_status` | Get indexing status and statistics |

## Starting the Server

### Standalone

```bash
# Start MCP server on stdio
pythia mcp

# Or with explicit transport
pythia mcp --transport stdio
```

### With Claude Desktop

Add to your Claude Desktop config (`~/.config/claude/mcp.json`):

```json
{
  "mcpServers": {
    "pythia": {
      "command": "pythia",
      "args": ["mcp"],
      "env": {
        "PYTHIA_CONFIG": "/path/to/config.toml"
      }
    }
  }
}
```

### With Claude Code

```bash
# Add Pythia as an MCP server
claude mcp add pythia -- pythia mcp

# Or with custom config
claude mcp add pythia -- pythia mcp --config /path/to/config.toml
```

## Tool Schemas

### search

```json
{
  "name": "search",
  "description": "Search code semantically across indexed repositories",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query"
      },
      "limit": {
        "type": "integer",
        "description": "Maximum results (default: 10)",
        "default": 10
      },
      "repositories": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Filter to specific repositories"
      }
    },
    "required": ["query"]
  }
}
```

### get_file

```json
{
  "name": "get_file",
  "description": "Get the full contents of a file",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repository": {
        "type": "string",
        "description": "Repository name (owner/repo)"
      },
      "path": {
        "type": "string",
        "description": "File path within repository"
      }
    },
    "required": ["repository", "path"]
  }
}
```

## Example Interaction

```
User: How does the authentication middleware work in the API?

Claude: [Uses pythia.search tool]
Query: "authentication middleware implementation"

[Results show relevant files]

Claude: [Uses pythia.get_file tool]
Repository: "myorg/api-server"
Path: "src/middleware/auth.rs"

Claude: Based on the code, the authentication middleware works by...
```

## Programmatic Usage

```rust
use pythia_mcp::{McpServer, McpConfig};
use pythia_indexer::IndexerService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create indexer with your repositories
    let indexer = IndexerService::new(Default::default());

    // Create MCP server
    let config = McpConfig {
        name: "pythia".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let server = McpServer::new(config, indexer);

    // Run on stdio
    server.run_stdio().await?;

    Ok(())
}
```

## Configuration

```toml
# pythia-mcp.toml

[mcp]
name = "pythia"
version = "0.1.0"

[mcp.capabilities]
tools = true
resources = false
prompts = false

[indexer]
repos_path = "./data/repos"
indexes_path = "./data/indexes"

[indexer.embedding]
provider = "local"
model = "bge-small"
```

## Security Considerations

1. **File Access**: The MCP server only exposes indexed files
2. **Rate Limiting**: Built-in rate limiting for search queries
3. **Authentication**: Configure API keys for cloud embeddings securely
4. **Sandboxing**: Consider running in a container for isolation
