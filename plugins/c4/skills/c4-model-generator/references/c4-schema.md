# C4 Model YAML Schema Reference

Complete schema documentation for the C4 model YAML format.

## Module File (c4.mod.yaml)

The root manifest file that defines the workspace.

```yaml
# yaml-language-server: $schema=../../_schema/mod.schema.json
version: "1.0"           # Required: Schema version (only "1.0" supported)
name: project-name       # Required: Workspace identifier

schema: ./_schema        # Optional: Path to schema directory

include:                 # Optional: Glob patterns for YAML files to include
  - shared/*.yaml
  - systems/**/system.yaml
  - systems/**/containers.yaml
  - systems/**/components.yaml
  - systems/**/relationships.yaml
  - flows/*.yaml
  - deployments/*.yaml

imports:                 # Optional: External model imports
  external-lib:
    source: "https://github.com/org/repo"
    ref: "v1.0.0"
    path: "c4-model"

options:                 # Optional: Display options
  showMinimap: true
```

---

## Person Schema

A human user of the software system.

```yaml
persons:
  - id: string              # Required: ^[a-z][a-z0-9-]*$ pattern
    name: string            # Required: Display name, min 1 char
    description: string     # Optional: Role, responsibilities, goals
    tags:                   # Optional: Classification labels
      - string
    properties:             # Optional: Extensible metadata
      key: value
```

### Example
```yaml
persons:
  - id: developer
    name: Application Developer
    description: Integrates vector search into their applications via API
    tags:
      - external
      - api-consumer
    properties:
      segment: enterprise
```

---

## Software System Schema

A software system that delivers value to users.

```yaml
systems:
  - id: string              # Required: Unique identifier
    name: string            # Required: Display name
    description: string     # Optional: Purpose and value delivered
    external: boolean       # Optional: Default false, true for dependencies
    tags:                   # Optional: Classification labels
      - string
    properties:             # Optional: Extensible metadata
      key: value
```

### Example
```yaml
systems:
  - id: islands
    name: Islands
    description: LEANN-based codebase indexing with AI-powered semantic search
    tags:
      - core
      - vector-search
    properties:
      team: platform
      repository: github.com/panbanda/islands

  - id: openai
    name: OpenAI API
    description: LLM provider for embeddings and completions
    external: true
    tags:
      - external
      - ai
```

---

## Container Schema

A deployable/runnable unit within a software system.

```yaml
containers:
  - id: string              # Required: Unique within system
    name: string            # Required: Display name
    description: string     # Optional: Responsibility and purpose
    technology: string      # Optional: Implementation technology stack
    tags:                   # Optional: Classification labels
      - string
    properties:             # Optional: Extensible metadata
      key: value
```

### Example
```yaml
containers:
  - id: mcp-server
    name: MCP Server
    description: Model Context Protocol server for AI assistant integration
    technology: Rust, async-std
    tags:
      - api
      - mcp
    properties:
      port: stdio
      protocol: JSON-RPC 2.0

  - id: vector-store
    name: Vector Store
    description: Persistent storage for HNSW graph and embeddings
    technology: SQLite, HNSW
    tags:
      - database
      - storage
```

---

## Component Schema

A grouping of related functionality within a container.

```yaml
components:
  - id: string              # Required: ^[a-z][a-z0-9-]*$ pattern
    name: string            # Required: Display name
    description: string     # Optional: Responsibility
    technology: string      # Optional: Implementation details
    systemId: string        # Optional: Parent system ID
    containerId: string     # Optional: Parent container ID
    tags:                   # Optional: Classification labels
      - string
    properties:             # Optional: Extensible metadata
      key: value
```

### Example
```yaml
components:
  - id: hnsw-graph
    name: HNSW Graph
    description: Hierarchical Navigable Small World graph for ANN search
    technology: Rust
    containerId: core

  - id: embedding-provider
    name: Embedding Provider
    description: Abstract interface for embedding generation
    technology: Rust trait
    containerId: core
    properties:
      implementations: [openai, local]
```

---

## Relationship Schema

A connection between two elements.

```yaml
relationships:
  - from: string            # Required: Source element (dot-path reference)
    to: string              # Required: Target element (dot-path reference)
    description: string     # Optional: What is communicated/used
    technology: string      # Optional: Protocol, format, mechanism
    tags:                   # Optional: Classification labels
      - string
    properties:             # Optional: Extensible metadata
      key: value
```

### Dot-Path References

Elements are referenced using dot notation:
- `person-id` - Reference to a person
- `system-id` - Reference to a system
- `system-id.container-id` - Reference to a container within a system
- `system-id.container-id.component-id` - Reference to a component

### Example
```yaml
relationships:
  - from: developer
    to: islands.mcp-server
    description: Sends search queries
    technology: MCP, JSON-RPC
    tags:
      - sync

  - from: islands.mcp-server
    to: islands.core.hnsw-graph
    description: Performs approximate nearest neighbor search
    technology: Rust function calls

  - from: islands.indexer
    to: openai
    description: Generates embeddings for code chunks
    technology: REST API, HTTPS
    tags:
      - async
      - external
```

---

## Flow Schema

A dynamic diagram showing runtime interactions.

```yaml
flows:
  - id: string              # Required: ^[a-z][a-z0-9-]*$ pattern
    name: string            # Required: Display name
    description: string     # Optional: What the flow represents
    tags:                   # Optional: Classification labels
      - string
    steps:                  # Required: Sequence of interactions
      - seq: integer        # Required: Step sequence number (1+)
        from: string        # Required: Source element
        to: string          # Required: Target element
        description: string # Optional: What happens in this step
        technology: string  # Optional: Protocol/mechanism used
```

### Example
```yaml
flows:
  - id: search-flow
    name: Semantic Search
    description: User performs semantic search across indexed repositories
    tags:
      - critical-path
      - user-facing
    steps:
      - seq: 1
        from: developer
        to: islands.mcp-server
        description: Sends search query with natural language
        technology: MCP tool call

      - seq: 2
        from: islands.mcp-server
        to: openai
        description: Generates query embedding
        technology: REST API

      - seq: 3
        from: islands.mcp-server
        to: islands.core.hnsw-graph
        description: Performs approximate nearest neighbor search
        technology: Rust

      - seq: 4
        from: islands.mcp-server
        to: developer
        description: Returns ranked results with code snippets
        technology: MCP response
```

---

## Deployment Schema

Deployment environment topology.

```yaml
deployments:
  - id: string              # Required: ^[a-z][a-z0-9-]*$ pattern
    name: string            # Required: Display name
    description: string     # Optional: Environment description
    nodes:                  # Optional: Deployment nodes
      - id: string          # Required: Node identifier
        name: string        # Required: Display name
        technology: string  # Optional: Infrastructure technology
        children:           # Optional: Nested nodes (recursive)
          - ... (same structure)
        instances:          # Optional: Container instances
          - container: string    # Required: Reference to container
            replicas: integer    # Optional: Default 1
            properties:          # Optional: Instance-specific config
              key: value
        properties:         # Optional: Node metadata
          key: value
```

### Example
```yaml
deployments:
  - id: kubernetes
    name: Kubernetes Production
    description: Production deployment on AWS EKS
    nodes:
      - id: aws
        name: AWS Cloud
        technology: AWS
        properties:
          region: us-east-1
        children:
          - id: eks-cluster
            name: EKS Cluster
            technology: AWS EKS
            children:
              - id: islands-namespace
                name: islands namespace
                technology: Kubernetes Namespace
                instances:
                  - container: islands.mcp-server
                    replicas: 3
                    properties:
                      cpu: 500m
                      memory: 1Gi
                  - container: islands.vector-store
                    replicas: 1
                    properties:
                      storage: 100Gi
```

---

## File Organization

Recommended directory structure:

```
c4-model/
├── c4.mod.yaml              # Workspace manifest
├── shared/
│   ├── personas.yaml        # Person definitions
│   └── external-systems.yaml # External dependencies
├── systems/
│   └── <system-name>/
│       ├── system.yaml      # System definition
│       ├── containers.yaml  # Container definitions
│       ├── components.yaml  # Component definitions
│       └── relationships.yaml # Relationships
├── flows/
│   ├── search-flow.yaml     # Dynamic flow diagrams
│   └── index-flow.yaml
└── deployments/
    ├── production.yaml      # Production environment
    └── staging.yaml         # Staging environment
```

---

## Validation

Validate YAML files against schema:

```bash
# Using the c4 CLI
c4 validate -C ./c4-model

# Using ajv-cli (if available)
ajv validate -s _schema/c4.schema.json -d "systems/**/*.yaml"
```

---

## Tips

### ID Naming Convention
- Use lowercase
- Use hyphens for word separation
- Start with a letter
- Pattern: `^[a-z][a-z0-9-]*$`

### Tag Conventions
Common tag patterns:
- `external` / `internal` - System boundary
- `core` / `supporting` - Business criticality
- `sync` / `async` - Communication pattern
- `database` / `cache` / `queue` - Container type
- `api` / `worker` / `cli` - Container role

### Property Conventions
Common properties:
- `team` - Owning team
- `oncall` - Contact for issues
- `repository` - Source code location
- `port` - Network port
- `sla` - Service level agreement
