---
name: islands-list
description: List all indexed codebases in Islands
invocation: islands list
triggers:
  - list indexes
  - show indexes
  - list codebases
  - islands ls
---

# Islands List Skill

List all codebases that have been indexed by Islands.

## Usage

```bash
# List all indexes
islands list

# Short alias
islands ls

# Output as JSON
islands list --json
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| --json | Output as JSON | false |

## Output

Table format showing index details:

```
NAME         PATH                    FILES  SIZE     UPDATED
myproject    /home/user/myproject    42     156 KB   2025-01-03 10:30
backend      /home/user/backend      128    1.2 MB   2025-01-02 15:45
frontend     /home/user/frontend     87     890 KB   2025-01-01 09:00
```

## JSON Output

```json
[
  {
    "name": "myproject",
    "path": "/home/user/myproject",
    "fileCount": 42,
    "size": 159744,
    "createdAt": "2025-01-03T10:30:00Z",
    "updatedAt": "2025-01-03T10:30:00Z",
    "checksum": "abc123def456"
  }
]
```

## Tips

- Use `--json` for scripting and automation
- Check index status before searching
- Re-index with `islands index --force` to update
- Delete old indexes manually from `~/.islands/indexes/`
