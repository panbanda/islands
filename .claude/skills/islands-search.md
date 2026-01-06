---
name: islands-search
description: Semantic search across indexed codebases to find relevant code patterns, implementations, and examples
invocation: islands search
triggers:
  - find code examples
  - how is X implemented
  - search for patterns
  - find similar code
---

# When to Use This Skill

Use `islands search` when you need to:
- Find how a concept is implemented across the codebase
- Locate examples of a pattern (error handling, auth, API calls)
- Answer "how does this project do X?" questions
- Gather context before making changes to unfamiliar code

This is semantic search, not string matching. Query with concepts and intent, not exact code.

## Quick Reference

```bash
# Find implementations
islands search "authentication middleware"

# Narrow to specific index
islands search -i backend "database connection pooling"

# More results for broader context
islands search -k 20 "error handling patterns"
```

## Interpreting Results

Results show: `[rank] path:line (score) preview`

```
[1] src/auth/jwt.go:42 (0.92)
    func ValidateToken(token string) (*Claims, error) { ... }
```

**Scores:**
- 0.8+ = Highly relevant, likely what you need
- 0.6-0.8 = Related, may provide useful context
- 0.4-0.6 = Tangentially related, check if useful

## Search Strategy

1. **Start broad**: `islands search "user authentication"`
2. **If too many results**: Add specificity or use index filter
3. **If too few results**: Broaden terms, try synonyms
4. **For exact matches**: Use `rg` (ripgrep) after finding the right files

## Key Parameters

| Flag | Purpose |
|------|---------|
| `-i, --index` | Limit to specific codebase |
| `-k, --top-k` | Number of results (default: 10) |

## What You Can Learn

- How the project structures similar functionality
- Naming conventions and patterns in use
- Which files are central to a feature
- Dependencies and integrations between modules
