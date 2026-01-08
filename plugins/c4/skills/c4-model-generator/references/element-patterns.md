# C4 Element Patterns

Common patterns for modeling different types of systems and architectures.

## Rust Crate Patterns

### Single Binary Crate

```yaml
# System
systems:
  - id: my-tool
    name: My Tool
    description: CLI tool for doing X

# Container = the binary
containers:
  - id: cli
    name: CLI Binary
    technology: Rust, Clap

# Components = top-level modules
components:
  - id: commands
    name: Commands
    containerId: cli
  - id: config
    name: Configuration
    containerId: cli
```

### Library + Binary Crate

```yaml
# Two containers: library and binary
containers:
  - id: lib
    name: Core Library
    technology: Rust
    tags:
      - library
  - id: cli
    name: CLI
    technology: Rust, Clap
    tags:
      - binary

# Relationship: CLI uses library
relationships:
  - from: my-crate.cli
    to: my-crate.lib
    description: Uses core functionality
```

### Workspace with Multiple Crates

```yaml
# Each crate = container
containers:
  - id: core
    name: Core
    technology: Rust
  - id: server
    name: Server
    technology: Rust, Axum
  - id: client
    name: Client
    technology: Rust, reqwest
```

---

## Microservices Patterns

### API Gateway Pattern

```yaml
systems:
  - id: api-gateway
    name: API Gateway

containers:
  - id: gateway
    name: Gateway
    technology: Nginx, Kong
  - id: auth
    name: Auth Service
    technology: Go
  - id: rate-limiter
    name: Rate Limiter
    technology: Redis

relationships:
  - from: customer
    to: api-gateway.gateway
    description: All API requests
  - from: api-gateway.gateway
    to: api-gateway.auth
    description: Token validation
  - from: api-gateway.gateway
    to: order-service.api
    description: Routes order requests
```

### Event-Driven Pattern

```yaml
containers:
  - id: api
    name: Order API
    technology: Go
  - id: events
    name: Order Events
    technology: Apache Kafka
  - id: processor
    name: Event Processor
    technology: Go

relationships:
  - from: order-service.api
    to: order-service.events
    description: Publishes OrderCreated events
    technology: Kafka Producer
    tags:
      - async
  - from: order-service.processor
    to: order-service.events
    description: Consumes order events
    technology: Kafka Consumer
    tags:
      - async
```

### CQRS Pattern

```yaml
containers:
  - id: command-api
    name: Command API
    description: Handles writes
    technology: Go
  - id: query-api
    name: Query API
    description: Handles reads
    technology: Go
  - id: write-db
    name: Write Database
    technology: PostgreSQL
  - id: read-db
    name: Read Database
    technology: Elasticsearch
  - id: sync
    name: Data Sync
    technology: Debezium
```

---

## Database Patterns

### Primary + Read Replica

```yaml
containers:
  - id: primary-db
    name: Primary Database
    technology: PostgreSQL 15
    tags:
      - database
      - primary
    properties:
      storage: 500GB

  - id: read-replica
    name: Read Replica
    technology: PostgreSQL 15
    tags:
      - database
      - replica
```

### Database + Cache

```yaml
containers:
  - id: db
    name: Database
    technology: PostgreSQL
  - id: cache
    name: Cache
    technology: Redis

relationships:
  - from: my-service.api
    to: my-service.cache
    description: Checks cache first
    technology: Redis protocol
  - from: my-service.api
    to: my-service.db
    description: Falls back to database
    technology: SQL
```

---

## External Integration Patterns

### Third-Party API

```yaml
# External system (external: true)
systems:
  - id: stripe
    name: Stripe
    description: Payment processing
    external: true
    tags:
      - external
      - payment

# Relationship to external
relationships:
  - from: payment-service.processor
    to: stripe
    description: Processes card payments
    technology: REST API, HTTPS
    tags:
      - external
      - pci-dss
```

### Webhook Integration

```yaml
relationships:
  # Outbound webhook subscription
  - from: my-service.webhook-handler
    to: github
    description: Receives push events
    technology: Webhook, HTTPS
    tags:
      - async
      - inbound

  # Outbound webhook delivery
  - from: my-service.notifier
    to: slack
    description: Sends notifications
    technology: Webhook, HTTPS
    tags:
      - async
      - outbound
```

---

## MCP Server Pattern

For Model Context Protocol servers:

```yaml
containers:
  - id: mcp-server
    name: MCP Server
    description: Model Context Protocol server for AI assistant integration
    technology: Rust, stdio
    tags:
      - api
      - mcp
    properties:
      transport: stdio
      protocol: JSON-RPC 2.0

components:
  - id: tools
    name: MCP Tools
    description: Exposed tool definitions
    containerId: mcp-server

  - id: resources
    name: MCP Resources
    description: Exposed resources
    containerId: mcp-server

  - id: prompts
    name: MCP Prompts
    description: Prompt templates
    containerId: mcp-server

# AI assistant as a person
persons:
  - id: ai-assistant
    name: AI Assistant
    description: Claude or other LLM using MCP tools
    tags:
      - ai
      - mcp-client

relationships:
  - from: ai-assistant
    to: my-system.mcp-server
    description: Invokes tools via MCP
    technology: MCP, stdio
```

---

## Deployment Patterns

### Kubernetes

```yaml
deployments:
  - id: k8s-prod
    name: Kubernetes Production
    nodes:
      - id: cluster
        name: EKS Cluster
        technology: AWS EKS
        children:
          - id: namespace
            name: my-app
            technology: Kubernetes Namespace
            instances:
              - container: my-system.api
                replicas: 3
```

### Docker Compose (Development)

```yaml
deployments:
  - id: docker-dev
    name: Docker Development
    description: Local development environment
    nodes:
      - id: docker-host
        name: Docker Host
        technology: Docker Desktop
        children:
          - id: app-network
            name: App Network
            technology: Docker Network
            instances:
              - container: my-system.api
                replicas: 1
                properties:
                  ports: "8080:8080"
              - container: my-system.db
                replicas: 1
                properties:
                  ports: "5432:5432"
```

### Serverless

```yaml
deployments:
  - id: serverless-prod
    name: AWS Serverless
    nodes:
      - id: aws
        name: AWS
        technology: AWS
        children:
          - id: api-gateway
            name: API Gateway
            technology: AWS API Gateway
          - id: lambda
            name: Lambda Functions
            technology: AWS Lambda
            instances:
              - container: my-system.api
                properties:
                  runtime: nodejs18.x
                  memory: 512
          - id: dynamodb
            name: DynamoDB
            technology: AWS DynamoDB
            instances:
              - container: my-system.db
```

---

## Flow Patterns

### Request-Response Flow

```yaml
flows:
  - id: api-request
    name: API Request Flow
    steps:
      - seq: 1
        from: client
        to: api-gateway.gateway
        description: HTTP request
      - seq: 2
        from: api-gateway.gateway
        to: api-gateway.auth
        description: Validates token
      - seq: 3
        from: api-gateway.gateway
        to: service.api
        description: Forwards request
      - seq: 4
        from: service.api
        to: service.db
        description: Queries data
      - seq: 5
        from: service.api
        to: client
        description: Returns response
```

### Async Processing Flow

```yaml
flows:
  - id: async-job
    name: Async Job Processing
    steps:
      - seq: 1
        from: user
        to: service.api
        description: Submits job
      - seq: 2
        from: service.api
        to: service.queue
        description: Queues job
        technology: RabbitMQ
      - seq: 3
        from: service.api
        to: user
        description: Returns job ID
      - seq: 4
        from: service.worker
        to: service.queue
        description: Picks up job
        technology: RabbitMQ
      - seq: 5
        from: service.worker
        to: service.db
        description: Updates status
      - seq: 6
        from: service.worker
        to: user
        description: Sends notification
        technology: Email/Webhook
```

---

## Anti-Patterns to Avoid

### Don't Mix Levels

```yaml
# BAD: Container and component mixed
containers:
  - id: api
    name: API
  - id: auth-handler  # This is a component, not a container!
    name: Auth Handler
```

### Don't Over-Detail External Systems

```yaml
# BAD: Too much detail about external system
systems:
  - id: stripe
    external: true

containers:
  - id: stripe-api      # DON'T detail external internals
  - id: stripe-webhook
  - id: stripe-dashboard

# GOOD: Just the boundary
systems:
  - id: stripe
    external: true
    description: Payment processing platform
```

### Don't Forget Relationships

```yaml
# BAD: Orphaned containers
containers:
  - id: api
  - id: db
  - id: cache
# No relationships = meaningless diagram

# GOOD: Connected architecture
relationships:
  - from: my-service.api
    to: my-service.cache
  - from: my-service.api
    to: my-service.db
```
