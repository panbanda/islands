# MCP (Model Context Protocol) Research Notes

## Overview

MCP is a standardized protocol for LLM tool integration. It enables AI assistants like Claude to interact with external tools and data sources through a consistent interface.

## Architecture

### Transport Layers

1. **stdio**: Standard input/output (for local CLI tools)
2. **HTTP/SSE**: Server-Sent Events over HTTP (for remote servers)

### Core Concepts

- **Tools**: Callable functions with typed parameters
- **Resources**: Readable data with URIs
- **Prompts**: Pre-defined prompt templates

## Go SDK Implementation

Using the mcp-go library from mark3labs:

```go
// Create server
mcpServer := server.NewMCPServer("pythia", "1.0.0",
    server.WithToolCapabilities(true),
    server.WithResourceCapabilities(true, true),
)

// Register tool
mcpServer.AddTool(mcp.NewTool("search",
    mcp.WithDescription("Search codebases"),
    mcp.WithString("query", mcp.Required()),
), handleSearch)

// Start server
server.ServeStdio(mcpServer)
```

## Pythia MCP Tools

| Tool | Description |
|------|-------------|
| search | Semantic code search |
| ask | Question answering |
| list_indexes | List codebases |
| get_index | Get index info |

## Claude Code Integration

Configuration in Claude Code:

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

## References

- [MCP Specification](https://modelcontextprotocol.io/)
- [mcp-go SDK](https://github.com/mark3labs/mcp-go)
