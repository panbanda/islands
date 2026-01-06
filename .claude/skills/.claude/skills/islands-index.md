---
name: islands-index
description: Index codebases for semantic search with Islands
invocation: islands index
triggers:
  - index codebase
  - index repository
  - add to islands
  - index project
---

# Islands Index Skill

Index local directories or Git repositories for semantic search.

## Usage

```bash
# Index current directory
islands index .

# Index with custom name
islands index --name myproject ./src

# Index multiple directories
islands index ./frontend ./backend ./shared

# Index from Git URL
islands index --git https://github.com/user/repo

# Force re-index
islands index --force --name myproject .
```

## Parameters

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| --name | -n | Custom index name | directory name |
| --git | | Treat path as Git URL | false |
| --exclude | | Glob patterns to exclude | (see defaults) |
| --include | | Glob patterns to include | source files |
| --force | | Force re-indexing | false |
| --timeout | | Indexing timeout | 30m |

## Default Exclusions

The following are excluded by default:
- `node_modules/`
- `.git/`
- `vendor/`
- `__pycache__/`
- `dist/`, `build/`
- `*.min.js`, `*.min.css`
- Lock files (package-lock.json, yarn.lock, go.sum)

## Default Inclusions

Source files indexed by default:
- Go: `*.go`
- Python: `*.py`
- JavaScript/TypeScript: `*.js`, `*.ts`, `*.jsx`, `*.tsx`
- Java: `*.java`
- Rust: `*.rs`
- C/C++: `*.c`, `*.cpp`, `*.h`, `*.hpp`
- Ruby: `*.rb`
- PHP: `*.php`
- C#: `*.cs`
- Markdown: `*.md`
- Config: `*.yaml`, `*.yml`, `*.json`, `*.toml`

## Examples

### Index a Monorepo
```bash
islands index --name monorepo-frontend ./packages/frontend
islands index --name monorepo-backend ./packages/backend
islands index --name monorepo-shared ./packages/shared
```

### Index with Custom Filters
```bash
islands index --include "**/*.go,**/*.proto" --exclude "**/testdata/**" .
```

### Index from GitHub
```bash
islands index --git https://github.com/kubernetes/kubernetes
```

## Output

Shows progress and statistics:

```
Indexing ./src as 'myproject'...
  cmd/main.go
  internal/server/server.go
  internal/handler/handler.go
Indexed 42 files (156 KB) in 2.3s
```

## Tips

- Use `--name` to give meaningful names for multiple indexes
- Use `--force` to update an existing index after code changes
- Index only relevant directories to improve search quality
- Check indexed files with `islands list`
