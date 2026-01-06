---
name: islands-index
description: Add and index repositories for semantic search
invocation: islands add
triggers:
  - index a repository
  - add codebase to islands
  - make repo searchable
---

# When to Use This Skill

Use `islands add` when:
- User wants to search a new repository
- Need to make code searchable for context gathering
- Setting up Islands for a project

## Quick Reference

```bash
# Add repository by URL
islands add github.com/owner/repo

# With authentication for private repos
islands add github.com/owner/repo --token $GITHUB_TOKEN
```

## What Gets Indexed

Source files are chunked and embedded for semantic search:
- Code files (go, py, js, ts, rs, java, etc.)
- Config files (yaml, json, toml)
- Documentation (md)

Automatically excluded:
- `node_modules/`, `vendor/`, `.git/`
- Build artifacts, lock files, minified code

## After Indexing

The repository becomes searchable via:
- `islands search "query"` - CLI search
- `islands_search` MCP tool - When running as MCP server

## Sync Updates

To update an existing index after code changes:

```bash
islands sync github/owner/repo
```

## Private Repositories

Set token via environment or flag:

```bash
export ISLANDS_GIT_TOKEN=ghp_xxxxx
islands add github.com/owner/private-repo
```
