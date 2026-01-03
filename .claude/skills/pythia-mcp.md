---
name: pythia-mcp
description: Start Pythia MCP server for LLM integration
invocation: pythia serve
triggers:
  - start mcp server
  - pythia server
  - mcp integration
  - llm server
---

# Pythia MCP Server Skill

Start Pythia as an MCP (Model Context Protocol) server for seamless LLM integration.

## Usage

```bash
# Start stdio server (default, for Claude Code)
pythia serve

# Start HTTP server for remote access
pythia serve --transport http --port 8080

# Specify host for network access
pythia serve --transport http --host 0.0.0.0 --port 8080
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
    "pythia": {
      "command": "pythia",
      "args": ["serve"]
    }
  }
}
```

Or with specific data directory:

```json
{
  "mcpServers": {
    "pythia": {
      "command": "pythia",
      "args": ["serve", "--config", "/path/to/pythia.toml"]
    }
  }
}
```

## HTTP Server

For remote or containerized deployments:

```bash
pythia serve --transport http --host 0.0.0.0 --port 8080
```

Access via HTTP SSE at `http://localhost:8080`.

## Docker Usage

```bash
docker run -v /path/to/data:/data ghcr.io/jon/pythia:latest serve
```

## Kubernetes

Deploy using the Helm chart:

```bash
helm install pythia ./deployments/helm/pythia
```

## Tips

- Use stdio transport for local Claude Code integration
- Use HTTP transport for remote or shared access
- Index codebases before starting the server
- Check server health with the version command
