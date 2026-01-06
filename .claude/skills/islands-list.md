---
name: islands-list
description: List indexed codebases to understand what repositories are available for semantic search
invocation: islands list
triggers:
  - what repos are indexed
  - show available codebases
  - what can I search
---

# When to Use This Skill

Use `islands list` to:
- Discover what codebases are available for search
- Verify a repository was indexed successfully
- Check index freshness before searching
- Understand the scope of searchable code

## Quick Reference

```bash
# List all indexed codebases
islands list
```

## Output

```
NAME         PATH                    FILES  SIZE     UPDATED
myproject    /home/user/myproject    42     156 KB   2025-01-03 10:30
backend      /home/user/backend      128    1.2 MB   2025-01-02 15:45
```

## What to Check

1. **Index exists**: Verify the repo you want to search is listed
2. **File count**: Low count may indicate indexing issues
3. **Updated date**: Stale indexes may miss recent code changes

## Decision Making

- **Index not listed**: Run `islands add <url>` first
- **Index stale**: Run `islands sync <name>` to update
- **Multiple indexes**: Use `-i <name>` flag in search to target specific ones
