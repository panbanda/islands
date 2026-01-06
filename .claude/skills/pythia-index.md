---
name: pythia-index
description: Index codebases for semantic search with Pythia
invocation: pythia index
triggers:
  - index codebase
  - index repository
  - add to pythia
  - index project
---

# Pythia Index Skill

Index local directories or Git repositories for semantic search.

## Usage

```bash
# Index current directory
pythia index .

# Index with custom name
pythia index --name myproject ./src

# Index multiple directories
pythia index ./frontend ./backend ./shared

# Index from Git URL
pythia index --git https://github.com/user/repo

# Force re-index
pythia index --force --name myproject .
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
pythia index --name monorepo-frontend ./packages/frontend
pythia index --name monorepo-backend ./packages/backend
pythia index --name monorepo-shared ./packages/shared
```

### Index with Custom Filters
```bash
pythia index --include "**/*.go,**/*.proto" --exclude "**/testdata/**" .
```

### Index from GitHub
```bash
pythia index --git https://github.com/kubernetes/kubernetes
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
- Check indexed files with `pythia list`
