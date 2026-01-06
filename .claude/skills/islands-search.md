---
name: islands-search
description: Search indexed codebases using Islands's semantic search capabilities
invocation: islands search
triggers:
  - search code
  - find in codebase
  - semantic search
  - search indexed
---

# Islands Search Skill

Search across indexed codebases using natural language semantic search powered by LEANN.

## Usage

```bash
# Basic search
islands search "authentication middleware"

# Search specific index
islands search --index myproject "error handling"

# Get more results
islands search --limit 20 "database connection"

# Output as JSON for processing
islands search --json "api endpoints"

# Set similarity threshold
islands search --threshold 0.7 "user validation"
```

## Parameters

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| --index | -i | Search only this index | all indexes |
| --limit | -l | Maximum results | 10 |
| --threshold | | Minimum similarity (0.0-1.0) | 0.5 |
| --json | | Output as JSON | false |
| --timeout | | Search timeout | 5m |

## Examples

### Find Authentication Code
```bash
islands search "JWT token validation and refresh"
```

### Search for Error Handling Patterns
```bash
islands search --index backend "error handling middleware express"
```

### Find Database Queries
```bash
islands search "SQL query builder with joins" --limit 15
```

## Output Format

Results show file path, line number, similarity score, and preview:

```
[1] src/auth/jwt.go:42 (0.92)
    func ValidateToken(token string) (*Claims, error) { ... }

[2] src/middleware/auth.go:15 (0.87)
    AuthMiddleware validates JWT tokens for protected routes ...
```

## Tips

- Use natural language queries for best results
- Higher thresholds (0.7+) give more precise matches
- Lower thresholds (0.3-0.5) find more diverse results
- Combine with grep for exact string matching after semantic search
