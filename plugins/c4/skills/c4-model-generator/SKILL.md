---
name: "C4 Model Generator"
description: "This skill should be used when the user asks to 'generate C4 model', 'document architecture', 'create C4 diagrams', 'analyze codebase structure', 'generate architecture YAML', runs '/c4', or discusses C4 modeling, system context diagrams, container diagrams, or component diagrams. Provides procedural guidance for analyzing codebases and generating C4 model YAML documentation."
---

# C4 Model Generator

Generate C4 architecture model YAML by systematically analyzing codebases. Creates or updates a `c4-model/` workspace with schema-compliant YAML that renders in the c4 CLI tool.

## Quick Start

Run `/c4` to generate documentation for the current repository:

```bash
# Document current repo
/c4

# Document multiple services
/c4 ../auth-service ../order-service

# Document specific paths
/c4 ./backend ./frontend
```

This creates/updates the `c4-model/` workspace, analyzing each codebase and generating YAML files.

## What It Does

1. **Creates workspace** if `c4-model/` doesn't exist
2. **Analyzes codebases** to identify systems, containers, components
3. **Discovers relationships** between elements
4. **Generates YAML** files following the C4 schema
5. **Updates incrementally** - preserves manual edits, adds new discoveries

## Rendering

View the generated model with the c4 CLI:

```bash
c4 serve -C ./c4-model
```

---

## Manual Workflow

For more control, follow the phase-by-phase analysis below.

---

## Phase-by-Phase Analysis Workflow

### Phase 1: Identify Persons (Users/Actors)

**Goal**: Document who interacts with the system.

**Analysis Steps**:
1. Search for authentication/authorization code
2. Look for user role definitions
3. Check API documentation for client types
4. Review README for intended users

**Search Patterns**:
```bash
# Find auth-related code
rg -l "auth|role|user|permission|token" --type rust --type ts --type go

# Find API client references
rg "client|consumer|user" README.md docs/
```

**Output to** `shared/personas.yaml`:
```yaml
# yaml-language-server: $schema=../../_schema/c4.schema.json
persons:
  - id: developer
    name: Developer
    description: Uses the API to integrate with their applications
    tags:
      - external
      - api-consumer

  - id: admin
    name: Administrator
    description: Manages configuration and monitors system health
    tags:
      - internal
```

---

### Phase 2: Identify External Systems

**Goal**: Document systems this project depends on or integrates with.

**Analysis Steps**:
1. Check `Cargo.toml`, `package.json`, or equivalent for dependencies
2. Look for HTTP client/SDK usage
3. Search for external API URLs or service names
4. Review configuration files for service endpoints

**Search Patterns**:
```bash
# Find external service calls
rg "https?://|api\.|\.com|\.io" --type rust --type ts

# Find SDK/client imports
rg "use.*client|import.*sdk|require.*api" --type rust --type ts
```

**Output to** `shared/external-systems.yaml`:
```yaml
# yaml-language-server: $schema=../../_schema/c4.schema.json
systems:
  - id: openai
    name: OpenAI API
    description: LLM provider for embeddings and completions
    external: true
    tags:
      - external
      - ai
      - embedding
```

---

### Phase 3: Identify the Main System

**Goal**: Define the system being documented as a single box.

**Analysis Steps**:
1. Read project README for purpose statement
2. Check main entry point (main.rs, index.ts, main.go)
3. Review package/crate description
4. Identify the core value proposition

**Output to** `systems/<system-name>/system.yaml`:
```yaml
# yaml-language-server: $schema=../../../../_schema/c4.schema.json
systems:
  - id: islands
    name: Islands
    description: LEANN-based codebase indexing with MCP server and AI-powered search
    tags:
      - core
      - vector-search
    properties:
      repository: github.com/panbanda/islands
      language: Rust
```

---

### Phase 4: Identify Containers

**Goal**: Map deployable/runnable units within the system.

**Definition**: A container is something that runs (binary, service, database, queue).

**Analysis Steps**:
1. Check for multiple binaries in Cargo.toml/package.json
2. Look for Dockerfile/docker-compose definitions
3. Identify database schemas or connections
4. Find message queue integrations
5. Check for separate API/worker processes

**Search Patterns**:
```bash
# Find binary definitions
rg '\[\[bin\]\]' Cargo.toml
rg '"bin":|"main":' package.json

# Find database connections
rg "postgres|mysql|sqlite|mongodb|redis" --type rust --type ts

# Find queue integrations
rg "kafka|rabbitmq|sqs|nats" --type rust --type ts
```

**Output to** `systems/<system-name>/containers.yaml`:
```yaml
# yaml-language-server: $schema=../../../../_schema/c4.schema.json
containers:
  - id: cli
    name: Islands CLI
    description: Command-line interface for repository indexing and search
    technology: Rust, Clap
    tags:
      - cli
      - binary

  - id: mcp-server
    name: MCP Server
    description: Model Context Protocol server for AI assistant integration
    technology: Rust, stdio
    tags:
      - api
      - mcp
    properties:
      transport: stdio
```

---

### Phase 5: Identify Components

**Goal**: Map logical groupings within containers.

**Definition**: A component is a group of related functionality (module, package, class group).

**Analysis Steps**:
1. Review module structure (`src/`, `lib/`, `pkg/`)
2. Look for domain boundaries
3. Identify service layers (handlers, services, repositories)
4. Map major abstractions

**Search Patterns**:
```bash
# List module structure
eza --tree -L 2 src/

# Find module definitions
rg "^pub mod |^mod " src/lib.rs --type rust
rg "^export|^import.*from" src/index.ts --type ts
```

**Output to** `systems/<system-name>/components.yaml`:
```yaml
# yaml-language-server: $schema=../../../../_schema/c4.schema.json
components:
  - id: hnsw
    name: HNSW Graph
    description: Hierarchical Navigable Small World graph for approximate nearest neighbor search
    technology: Rust
    containerId: core

  - id: leann
    name: LEANN Index
    description: Low-storage embedding graph using CSR format, recomputes embeddings on-demand
    technology: Rust
    containerId: core
```

---

### Phase 6: Map Relationships

**Goal**: Document how elements communicate.

**Analysis Steps**:
1. Trace function calls between modules
2. Identify HTTP/gRPC/RPC calls
3. Map database read/write patterns
4. Document event publishing/subscribing
5. Note async vs sync communication

**Search Patterns**:
```bash
# Find cross-module imports
rg "^use crate::|^use super::" --type rust

# Find HTTP calls
rg "\.get\(|\.post\(|\.put\(|\.delete\(" --type rust --type ts

# Find database operations
rg "\.query\(|\.execute\(|\.insert\(|\.select\(" --type rust
```

**Output to** `systems/<system-name>/relationships.yaml`:
```yaml
# yaml-language-server: $schema=../../../../_schema/c4.schema.json
relationships:
  - from: islands.cli
    to: islands.core
    description: Uses LEANN/HNSW for vector operations
    technology: Rust function calls

  - from: islands.indexer
    to: openai
    description: Generates embeddings for code chunks
    technology: REST API, HTTPS
    tags:
      - async
```

---

### Phase 7: Document Flows (Optional)

**Goal**: Capture important runtime sequences.

**Analysis Steps**:
1. Identify critical user journeys
2. Trace request handling from entry to response
3. Document async processing chains
4. Map error handling flows

**Output to** `flows/<flow-name>.yaml`:
```yaml
# yaml-language-server: $schema=../../_schema/c4.schema.json
flows:
  - id: search-flow
    name: Semantic Search Flow
    description: User performs semantic search across indexed repositories
    tags:
      - critical-path
    steps:
      - seq: 1
        from: developer
        to: islands.mcp-server
        description: Sends search query via MCP tool
        technology: MCP, JSON-RPC

      - seq: 2
        from: islands.mcp-server
        to: islands.core
        description: Performs LEANN search
        technology: Rust
```

---

### Phase 8: Model Deployments (Optional)

**Goal**: Document infrastructure topology.

**Analysis Steps**:
1. Review Kubernetes manifests / Helm charts
2. Check docker-compose files
3. Analyze CI/CD pipelines for deploy targets
4. Document cloud resources (AWS, GCP, etc.)

**Output to** `deployments/<env>.yaml`:
```yaml
# yaml-language-server: $schema=../../_schema/c4.schema.json
deployments:
  - id: kubernetes
    name: Kubernetes Deployment
    description: Production deployment on Kubernetes
    nodes:
      - id: cluster
        name: EKS Cluster
        technology: AWS EKS
        children:
          - id: islands-pod
            name: Islands Pod
            technology: Kubernetes Pod
            instances:
              - container: islands.mcp-server
                replicas: 2
```

---

## Schema Reference

### Element Properties

All elements support:
- `id` (required): Unique identifier, lowercase with hyphens
- `name` (required): Display name
- `description`: Purpose and responsibility
- `tags`: Array of classification labels
- `properties`: Key-value metadata

### Container-Specific
- `technology`: Implementation stack (e.g., "Go, Chi", "PostgreSQL 15")

### Component-Specific
- `technology`: Implementation details
- `containerId`: Parent container reference

### Relationship-Specific
- `from`: Source element (dot-path: `system.container.component`)
- `to`: Target element
- `description`: What is communicated
- `technology`: Protocol/format

---

## Best Practices

### What Adds Value

1. **Rich descriptions** - Explain purpose, not just name
2. **Technology annotations** - Help readers understand implementation
3. **Tags for filtering** - Enable diagram customization
4. **Properties for metadata** - Team ownership, SLA, ports
5. **Flows for critical paths** - Document what matters most

### Common Mistakes to Avoid

1. **Mixing abstraction levels** - Keep containers and components separate
2. **Over-detailing external systems** - Focus on boundaries
3. **Missing relationships** - The connections are the architecture
4. **Stale documentation** - Update when code changes
5. **No personas** - Always show who uses the system

---

## Additional Resources

### Reference Files
- **`references/c4-schema.md`** - Complete JSON schema documentation
- **`references/element-patterns.md`** - Common modeling patterns

### Templates
- **`resources/templates/`** - Starter YAML templates for each element type

### Scripts
- **`scripts/analyze-rust.sh`** - Analyze Rust codebase structure
- **`scripts/validate-c4.sh`** - Validate C4 YAML against schema
