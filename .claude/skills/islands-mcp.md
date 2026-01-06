---
name: islands-mcp
description: Start Islands as an MCP server for AI assistant integration
invocation: islands mcp
triggers:
  - start mcp server
  - islands server
  - mcp integration
---

# When to Use This Skill

Use `islands mcp` when:
- Configuring Islands as an MCP server for Claude
- Setting up semantic search as an AI tool
- Deploying Islands for AI assistant access

## Quick Reference

```bash
# Start MCP server (stdio transport)
islands mcp
```

The server communicates via stdin/stdout using JSON-RPC.

## MCP Tools Exposed

When running as MCP server, these tools become available:

| Tool | Purpose |
|------|---------|
| `islands_search` | Semantic search across indexed codebases |
| `islands_list` | List available indexed repositories |
| `islands_add_repo` | Add and index a new repository |
| `islands_sync` | Update an existing index |
| `islands_status` | System status and statistics |

## Claude Code Setup

```bash
claude mcp add islands -- islands mcp
```

## Claude Desktop Setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "islands": {
      "command": "islands",
      "args": ["mcp"]
    }
  }
}
```

## Prerequisites

1. Index repositories first: `islands add <url>`
2. Verify with: `islands list`
3. Then start server: `islands mcp`

## How It Helps

Once configured, the AI assistant can:
- Search across all indexed codebases semantically
- Find implementations and patterns without manual file navigation
- Gather context from multiple repositories simultaneously
