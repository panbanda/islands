# Pythia MCP API Reference

## Server Configuration

Start the MCP server:

```bash
# stdio mode (for Claude Code)
pythia serve

# HTTP mode (for remote access)
pythia serve --transport http --port 8080
```

## Tools

### search

Search indexed codebases using semantic search.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| query | string | yes | Natural language search query |
| index | string | no | Specific index to search |
| limit | number | no | Maximum results (default: 10) |

**Example:**
```json
{
  "query": "authentication middleware with JWT",
  "index": "my-backend",
  "limit": 20
}
```

**Response:**
```
[1] src/auth/jwt.go:42 (score: 0.92)
    func ValidateToken(token string) (*Claims, error) { ... }

[2] src/middleware/auth.go:15 (score: 0.87)
    AuthMiddleware validates JWT tokens for protected routes ...
```

### ask

Ask a question about the codebase.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| question | string | yes | Question about the code |
| index | string | no | Specific index to query |

**Example:**
```json
{
  "question": "How is user authentication implemented?",
  "index": "my-backend"
}
```

**Response:**
Relevant code snippets with context for answering the question.

### list_indexes

List all indexed codebases.

**Parameters:** None

**Response:**
```
- my-backend: /path/to/backend (128 files)
- my-frontend: /path/to/frontend (87 files)
```

### get_index

Get detailed information about a specific index.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| name | string | yes | Index name |

**Example:**
```json
{
  "name": "my-backend"
}
```

**Response:**
```
Index: my-backend
Path: /path/to/backend
Files: 128
Size: 1234567 bytes
Updated: 2025-01-03 10:30:00
```

## Resources

Resources are exposed for each indexed codebase:

**URI Format:** `pythia://index/{name}`

**Example:**
```
pythia://index/my-backend
pythia://index/my-frontend
```

## Error Handling

Errors are returned in the standard MCP error format:

```json
{
  "error": {
    "code": -32602,
    "message": "index not found: nonexistent"
  }
}
```

## HTTP Transport

When using HTTP transport, connect via SSE:

```
GET http://localhost:8080/sse
```

## Authentication

Currently, Pythia does not implement authentication. For production deployments, use a reverse proxy with authentication.
