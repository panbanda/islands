---
name: islands-mcp
description: Start Islands MCP server for LLM integration
invocation: islands serve
triggers:
  - start mcp server
  - islands server
  - mcp integration
  - llm server
---

# Islands MCP Server Skill

Start Islands as an MCP (Model Context Protocol) server for seamless LLM integration.

## Usage

```bash
# Start stdio server (default, for Claude Code)
islands serve

# Start HTTP server for remote access
islands serve --transport http --port 8080

# Specify host for network access
islands serve --transport http --host 0.0.0.0 --port 8080
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| --transport | Transport mode (stdio, http) | stdio |
| --port | HTTP port | 8080 |
| --host | HTTP host | 127.0.0.1 |

## MCP Tools Available

### search
Search indexed codebases using semantic search.

```json
{
  "query": "authentication middleware",
  "index": "myproject",
  "limit": 10
}
```

### ask
Ask questions about the codebase.

```json
{
  "question": "How is user authentication implemented?",
  "index": "myproject"
}
```

### list_indexes
List all indexed codebases.

### get_index
Get information about a specific index.

```json
{
  "name": "myproject"
}
```

## Claude Code Integration

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "islands": {
      "command": "islands",
      "args": ["serve"]
    }
  }
}
```

Or with specific data directory:

```json
{
  "mcpServers": {
    "islands": {
      "command": "islands",
      "args": ["serve", "--config", "/path/to/islands.toml"]
    }
  }
}
```

## HTTP Server

For remote or containerized deployments:

```bash
islands serve --transport http --host 0.0.0.0 --port 8080
```

Access via HTTP SSE at `http://localhost:8080`.

## Docker Usage

```bash
docker run -v /path/to/data:/data ghcr.io/jon/islands:latest serve
```

## Kubernetes

Deploy using the Helm chart:

```bash
helm install islands ./deployments/helm/islands
```

## Tips

- Use stdio transport for local Claude Code integration
- Use HTTP transport for remote or shared access
- Index codebases before starting the server
- Check server health with the version command
