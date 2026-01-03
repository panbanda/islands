# ADR-001: LEANN Integration Architecture

**Status**: Proposed
**Date**: 2026-01-03
**Author**: System Architect

## Context

Pythia is a Go-based codebase analysis tool that provides semantic search via MCP (Model Context Protocol). The current implementation has a placeholder LEANN client (`pkg/leann/client.go`) that returns mock results. We need to integrate the actual LEANN Python library for production-grade vector search.

### Current Architecture

```
+------------------+     +------------------+     +------------------+
|   Claude / LLM   |---->|   MCP Server     |---->|    Analyzer      |
|                  |     |  (Go, stdio/HTTP)|     |                  |
+------------------+     +------------------+     +--------+---------+
                                                          |
                         +------------------+     +-------v--------+
                         |    Storage       |<----|  LEANN Client  |
                         |   (SQLite)       |     |  (placeholder) |
                         +------------------+     +----------------+
```

### Requirements

1. **Self-contained Docker deployment** - Single container, no compose/orchestration
2. **Production-ready** - Health checks, graceful shutdown, error handling
3. **Efficient IPC** - Minimize latency for embedding generation and search
4. **Support operations**: Index building, semantic search, question answering
5. **Maintain distroless security posture** where possible

## Decision

After evaluating four integration options, **Option 2 (Unix Socket IPC)** is recommended for production deployment, with **Option 3 (JSON-RPC subprocess)** as a simpler alternative for development.

---

## Option 1: HTTP/gRPC Sidecar

### Architecture

```
+-------------------------------------------------------------------+
|                         Docker Container                           |
|                                                                    |
|  +---------------------+         +---------------------------+    |
|  |     Go Binary       |  HTTP   |     Python Service        |    |
|  |     (pythia)        |<------->|     (FastAPI/Flask)       |    |
|  |                     |  :8081  |                           |    |
|  |  +---------------+  |         |  +---------------------+  |    |
|  |  | MCP Server    |  |         |  | LEANN Engine        |  |    |
|  |  | (port 8080)   |  |         |  | - Embeddings        |  |    |
|  |  +---------------+  |         |  | - Vector Store      |  |    |
|  |                     |         |  | - Search            |  |    |
|  |  +---------------+  |         |  +---------------------+  |    |
|  |  | LEANN Client  +--+---------+->                         |    |
|  |  | (HTTP)        |  |         |                           |    |
|  |  +---------------+  |         +---------------------------+    |
|  +---------------------+                                          |
|                                                                    |
|  Process Manager: supervisord / s6-overlay / custom init          |
+-------------------------------------------------------------------+
```

### Data Flow

```
1. Index Request
   Claude --> MCP Server --> Analyzer --> LEANN Client (HTTP)
                                              |
                                              v
                                    POST /v1/index
                                    {documents: [...]}
                                              |
                                              v
                                    Python FastAPI --> LEANN Engine
                                              |
                                              v
                                    200 OK {chunks: 1234}

2. Search Request
   Claude --> MCP Server --> Analyzer --> LEANN Client (HTTP)
                                              |
                                              v
                                    POST /v1/search
                                    {query: "...", limit: 10}
                                              |
                                              v
                                    Python FastAPI --> LEANN Engine
                                              |
                                              v
                                    200 OK [{id, score, content}, ...]
```

### Pros
- **Clear separation** - Independent processes, easy to debug
- **Language-agnostic** - Could swap Python for Rust implementation later
- **Standard tooling** - HTTP clients, OpenAPI specs, load testing
- **Independent scaling** - Can tune resources per service
- **Health checks** - Native HTTP health endpoints

### Cons
- **Process management** - Requires supervisord or similar
- **Serialization overhead** - JSON encoding/decoding on every call
- **Startup complexity** - Must wait for Python service before Go can serve
- **Port conflicts** - Internal port management
- **Larger image** - Need Python runtime + Go binary + process manager

### Performance Characteristics
- **Latency**: ~1-5ms overhead per request (HTTP + JSON)
- **Throughput**: Limited by HTTP connection pool
- **Memory**: Higher due to separate process heaps
- **Cold start**: 3-10s (Python + model loading)

### Docker Implications
```dockerfile
# Cannot use distroless - need Python runtime
FROM python:3.11-slim AS python-base
# ... install LEANN, FastAPI ...

FROM golang:1.23-alpine AS go-builder
# ... build Go binary ...

FROM python:3.11-slim AS runtime
# Copy Go binary
COPY --from=go-builder /pythia /usr/local/bin/
# Copy Python environment
COPY --from=python-base /app /app
# Need supervisord for multi-process
RUN apt-get update && apt-get install -y supervisor
COPY supervisord.conf /etc/supervisor/conf.d/
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
```

### Implementation Complexity
- **Go changes**: Moderate - Replace placeholder with HTTP client
- **Python service**: New - FastAPI app with LEANN wrapper
- **Docker**: Complex - Multi-process management
- **Testing**: Easy - Mock HTTP endpoints

**Estimated effort**: 3-4 days

---

## Option 2: Unix Socket IPC (RECOMMENDED)

### Architecture

```
+-------------------------------------------------------------------+
|                         Docker Container                           |
|                                                                    |
|  +---------------------+                                          |
|  |     Go Binary       |                                          |
|  |     (pythia)        |                                          |
|  |                     |                                          |
|  |  +---------------+  |                                          |
|  |  | MCP Server    |  |                                          |
|  |  | (port 8080)   |  |                                          |
|  |  +---------------+  |                                          |
|  |                     |                                          |
|  |  +---------------+  |    Unix Socket    +-------------------+  |
|  |  | LEANN Client  +--+------------------->| Python Process   |  |
|  |  | (socket IPC)  |<-+-------------------+| (long-running)   |  |
|  |  +---------------+  |  /tmp/leann.sock  |                   |  |
|  |                     |                   | +---------------+ |  |
|  |  +---------------+  |                   | | LEANN Engine  | |  |
|  |  | Process Mgr   +--+--- spawns ------->| +---------------+ |  |
|  |  +---------------+  |                   +-------------------+  |
|  +---------------------+                                          |
+-------------------------------------------------------------------+
```

### Protocol Design

```
Message Format (length-prefixed JSON):
+--------+------------------+
| 4 bytes|   JSON payload   |
| (len)  |                  |
+--------+------------------+

Request:
{
  "id": "uuid",
  "method": "search",
  "params": {
    "query": "authentication logic",
    "limit": 10,
    "threshold": 0.5
  }
}

Response:
{
  "id": "uuid",
  "result": [...],
  "error": null
}
```

### Startup Sequence

```
1. Go binary starts
2. Go spawns Python subprocess with socket path
3. Python creates Unix socket, loads LEANN model
4. Python sends "ready" message
5. Go begins accepting MCP connections
6. On shutdown: Go sends "shutdown", waits for Python exit

Timeline:
|----Go init----|----Python spawn----|----Model load----|----Ready----|
0s              0.1s                 0.5s               3-5s
```

### Pros
- **Single entry point** - Go manages everything
- **Low latency** - Unix sockets faster than TCP/HTTP
- **No port conflicts** - Socket file, not network port
- **Clean shutdown** - Go controls Python lifecycle
- **Simpler container** - No process manager needed
- **Better security** - No network exposure for internal comms

### Cons
- **Tighter coupling** - Go must handle Python process management
- **Error handling** - Must handle Python crashes, restart logic
- **Debugging** - Harder to test Python service in isolation
- **Platform-specific** - Unix sockets not available on Windows

### Performance Characteristics
- **Latency**: ~0.1-0.5ms overhead per request
- **Throughput**: Higher than HTTP (no connection overhead)
- **Memory**: Slightly lower than HTTP (no HTTP server overhead)
- **Cold start**: Same 3-10s (dominated by model loading)

### Docker Implications
```dockerfile
# Multi-stage build
FROM python:3.11-slim AS python-deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY leann_server.py .

FROM golang:1.23-alpine AS go-builder
# ... build Go binary ...

FROM python:3.11-slim AS runtime
# Copy Python environment
COPY --from=python-deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=python-deps /app /app
# Copy Go binary
COPY --from=go-builder /pythia /usr/local/bin/
# Single entry point
USER nonroot
ENTRYPOINT ["/usr/local/bin/pythia"]
CMD ["serve"]
```

### Implementation Complexity
- **Go changes**: Moderate - Process spawning + socket client
- **Python service**: Simple - Socket server with LEANN wrapper
- **Docker**: Moderate - Single process entry, Python runtime
- **Testing**: Moderate - Need integration tests

**Estimated effort**: 2-3 days

---

## Option 3: JSON-RPC Subprocess (on-demand)

### Architecture

```
+-------------------------------------------------------------------+
|                         Docker Container                           |
|                                                                    |
|  +---------------------+                                          |
|  |     Go Binary       |                                          |
|  |     (pythia)        |                                          |
|  |                     |                                          |
|  |  +---------------+  |                                          |
|  |  | MCP Server    |  |                                          |
|  |  +---------------+  |                                          |
|  |                     |                                          |
|  |  +---------------+  |   stdin/stdout   +-------------------+   |
|  |  | LEANN Client  +--+----------------->| Python Process   |   |
|  |  | (subprocess)  |<-+------------------| (per-request OR  |   |
|  |  +---------------+  |   JSON-RPC       | connection-pooled)|   |
|  |                     |                  +-------------------+   |
|  +---------------------+                                          |
+-------------------------------------------------------------------+
```

### Protocol (JSON-RPC 2.0 over stdio)

```
Go --> Python (stdin):
{"jsonrpc":"2.0","id":1,"method":"index","params":{"docs":[...]}}

Python --> Go (stdout):
{"jsonrpc":"2.0","id":1,"result":{"chunks":1234}}
```

### Process Lifecycle Options

**Option A: Per-request spawning**
```
Request --> Spawn Python --> Load model --> Process --> Exit
           (expensive, 3-5s cold start per request)
```

**Option B: Connection pooling (recommended)**
```
First request --> Spawn Python --> Keep alive --> Reuse for N requests
                                                      |
                Idle timeout (5min) ------------------>  Exit
```

### Pros
- **Simplest implementation** - Standard I/O, no sockets/ports
- **Cross-platform** - Works on Windows, Mac, Linux
- **Easy debugging** - Can test Python script directly
- **No dependencies** - No socket libraries needed
- **Natural process isolation**

### Cons
- **Startup cost** - Model loading on each spawn (if per-request)
- **Complexity shifts** - Need connection pooling for performance
- **Buffer management** - Must handle large JSON payloads
- **No multiplexing** - One request at a time per subprocess

### Performance Characteristics
- **Latency**: ~0.5-1ms (warm), 3-5s (cold start)
- **Throughput**: Limited by process pool size
- **Memory**: Efficient with pooling, wasteful per-request
- **Cold start**: 3-5s per subprocess

### Docker Implications
```dockerfile
# Same as Option 2, simpler Python server
FROM python:3.11-slim AS runtime
COPY --from=go-builder /pythia /usr/local/bin/
COPY leann_worker.py /app/
ENTRYPOINT ["/usr/local/bin/pythia"]
```

### Implementation Complexity
- **Go changes**: Simple - exec.Cmd with stdin/stdout pipes
- **Python service**: Simplest - Read JSON from stdin, write to stdout
- **Docker**: Simple - Same as Option 2
- **Testing**: Easy - Can test Python script standalone

**Estimated effort**: 1-2 days

---

## Option 4: Embedded Python (cgo)

### Architecture

```
+-------------------------------------------------------------------+
|                         Docker Container                           |
|                                                                    |
|  +---------------------------------------------------------------+|
|  |                     Go Binary (with cgo)                       ||
|  |                                                                ||
|  |  +---------------+     +-----------------------------------+  ||
|  |  | MCP Server    |     |        Python Interpreter         |  ||
|  |  +---------------+     |        (embedded via cgo)         |  ||
|  |                        |                                   |  ||
|  |  +---------------+     |  +-----------------------------+  |  ||
|  |  | LEANN Client  +-----+->| LEANN Module (in-process)   |  |  ||
|  |  | (cgo calls)   |<----+--| - No serialization overhead |  |  ||
|  |  +---------------+     |  +-----------------------------+  |  ||
|  |                        +-----------------------------------+  ||
|  +---------------------------------------------------------------+|
+-------------------------------------------------------------------+
```

### Pros
- **Lowest latency** - No IPC overhead, direct function calls
- **Single process** - Simplest runtime model
- **Shared memory** - Can pass pointers, avoid copies

### Cons
- **Extreme complexity** - cgo + Python C API
- **GIL contention** - Python's GIL blocks Go goroutines
- **Build complexity** - Need Python headers, CGO_ENABLED=1
- **Debugging nightmare** - Mixed Go/Python stack traces
- **Version coupling** - Tied to specific Python version
- **Cannot use distroless** - Need glibc, Python shared libs
- **Memory leaks** - Reference counting across language boundary

### Performance Characteristics
- **Latency**: ~0.01ms (lowest possible)
- **Throughput**: Limited by Python GIL
- **Memory**: Shared heap (complex management)
- **Cold start**: Same model loading time

### Docker Implications
```dockerfile
# Complex multi-stage with shared libraries
FROM python:3.11 AS python-dev
# Need Python headers and shared libraries
RUN apt-get install -y python3-dev

FROM golang:1.23 AS go-builder
COPY --from=python-dev /usr/include/python3.11 /usr/include/python3.11
COPY --from=python-dev /usr/lib/x86_64-linux-gnu/libpython3.11.so* /usr/lib/
ENV CGO_ENABLED=1
# Complex build with Python linking...

FROM debian:bookworm-slim AS runtime
# Need full glibc runtime
COPY --from=python-dev /usr/lib/x86_64-linux-gnu/libpython3.11.so* /usr/lib/
# ... many more shared libraries ...
```

### Implementation Complexity
- **Go changes**: Extreme - cgo bindings, Python C API
- **Python service**: N/A - Direct module loading
- **Docker**: Very complex - Shared library management
- **Testing**: Difficult - Need full Python environment

**Estimated effort**: 2-3 weeks (not recommended)

---

## Comparison Matrix

| Criterion              | HTTP Sidecar | Unix Socket | JSON-RPC | Embedded |
|------------------------|--------------|-------------|----------|----------|
| **Latency**            | 1-5ms        | 0.1-0.5ms   | 0.5-1ms  | 0.01ms   |
| **Implementation**     | Moderate     | Moderate    | Simple   | Extreme  |
| **Docker complexity**  | High         | Moderate    | Simple   | Extreme  |
| **Debugging**          | Easy         | Moderate    | Easy     | Hard     |
| **Production-ready**   | Yes          | Yes         | Yes      | No       |
| **Process management** | Supervisor   | Go-managed  | Go-managed | N/A    |
| **Health checks**      | Native HTTP  | Custom      | Custom   | In-proc  |
| **Graceful shutdown**  | Moderate     | Easy        | Easy     | Complex  |
| **Security**           | Port exposure| Socket-only | stdio    | N/A      |
| **Cross-platform**     | Yes          | Unix only   | Yes      | Complex  |

---

## Recommendation

### For Production: Option 2 (Unix Socket IPC)

**Rationale**:
1. **Best latency/complexity tradeoff** - ~10x faster than HTTP with similar complexity
2. **Clean process model** - Go manages lifecycle, single entry point
3. **No network exposure** - Internal communication via filesystem socket
4. **Graceful shutdown** - Go can cleanly terminate Python subprocess
5. **Health checks** - Go can ping Python process and report status

### For Development/Prototyping: Option 3 (JSON-RPC)

**Rationale**:
1. Fastest to implement
2. Easy to test Python script independently
3. Good stepping stone to Option 2

### Migration Path

```
Phase 1 (Week 1): JSON-RPC subprocess
  - Get LEANN working end-to-end
  - Validate embedding quality
  - Test with real codebases

Phase 2 (Week 2): Unix Socket upgrade
  - Add connection pooling
  - Implement health checks
  - Add graceful shutdown

Phase 3 (Week 3): Production hardening
  - Retry logic for Python crashes
  - Metrics collection
  - Resource limits
```

---

## Implementation Details for Option 2

### Go LEANN Client Interface

```go
// pkg/leann/client.go

type Client struct {
    socketPath string
    conn       net.Conn
    cmd        *exec.Cmd
    mu         sync.Mutex
    ready      chan struct{}
}

type ClientOptions struct {
    PythonPath    string        // Path to Python interpreter
    ScriptPath    string        // Path to leann_server.py
    SocketPath    string        // Unix socket path
    ModelName     string        // Embedding model name
    StartupTimeout time.Duration
}

func NewClient(opts ClientOptions) (*Client, error)
func (c *Client) Start(ctx context.Context) error
func (c *Client) Stop() error
func (c *Client) Health() error
func (c *Client) AddDocuments(ctx context.Context, docs []Document) error
func (c *Client) Search(ctx context.Context, opts SearchOptions) ([]SearchResult, error)
```

### Python Server Skeleton

```python
# leann_server.py

import socket
import json
import sys
from leann import LEANN

def handle_request(engine: LEANN, request: dict) -> dict:
    method = request.get("method")
    params = request.get("params", {})

    if method == "health":
        return {"status": "ok", "documents": engine.count()}
    elif method == "index":
        chunks = engine.add_documents(params["documents"])
        return {"chunks": chunks}
    elif method == "search":
        results = engine.search(
            query=params["query"],
            limit=params.get("limit", 10),
            threshold=params.get("threshold", 0.5)
        )
        return {"results": results}
    elif method == "shutdown":
        return {"status": "shutting_down"}
    else:
        return {"error": f"unknown method: {method}"}

def main():
    socket_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "all-MiniLM-L6-v2"

    # Load model (slow, ~3-5s)
    engine = LEANN(model=model_name)

    # Create socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_path)
    sock.listen(1)

    # Signal ready
    print(json.dumps({"status": "ready"}), flush=True)

    # Accept connection from Go
    conn, _ = sock.accept()

    while True:
        # Read length-prefixed message
        length_bytes = conn.recv(4)
        if not length_bytes:
            break
        length = int.from_bytes(length_bytes, 'big')
        data = conn.recv(length)

        request = json.loads(data)
        response = handle_request(engine, request)

        # Write length-prefixed response
        response_bytes = json.dumps(response).encode()
        conn.send(len(response_bytes).to_bytes(4, 'big'))
        conn.send(response_bytes)

        if request.get("method") == "shutdown":
            break

    conn.close()
    sock.close()

if __name__ == "__main__":
    main()
```

### Dockerfile Structure

```dockerfile
# syntax=docker/dockerfile:1

#--------------------------------------------------------------
# Stage 1: Python dependencies
#--------------------------------------------------------------
FROM python:3.11-slim AS python-deps

WORKDIR /app

# Install LEANN and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# Copy Python server script
COPY scripts/leann_server.py .

#--------------------------------------------------------------
# Stage 2: Go builder
#--------------------------------------------------------------
FROM golang:1.23-alpine AS go-builder

ARG VERSION=dev
WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build \
    -ldflags="-s -w -X main.Version=${VERSION}" \
    -o /pythia .

#--------------------------------------------------------------
# Stage 3: Runtime
#--------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Security: Create non-root user
RUN useradd -r -u 65532 -s /sbin/nologin pythia

# Install minimal Python runtime (no pip, no dev tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        tzdata && \
    rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY --from=python-deps /app/deps /app/deps
COPY --from=python-deps /app/leann_server.py /app/

# Copy Go binary
COPY --from=go-builder /pythia /usr/local/bin/

# Set Python path
ENV PYTHONPATH=/app/deps

# Create data directory
WORKDIR /data
RUN chown pythia:pythia /data

USER pythia

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/pythia", "health"]

ENTRYPOINT ["/usr/local/bin/pythia"]
CMD ["serve"]
```

---

## Appendix: Health Check Implementation

```go
// cmd/pythia/health.go

var healthCmd = &cobra.Command{
    Use:   "health",
    Short: "Check service health",
    RunE: func(cmd *cobra.Command, args []string) error {
        client, err := leann.NewClient(leann.ClientOptions{...})
        if err != nil {
            return fmt.Errorf("leann unavailable: %w", err)
        }
        defer client.Close()

        if err := client.Health(); err != nil {
            return fmt.Errorf("leann unhealthy: %w", err)
        }

        fmt.Println("healthy")
        return nil
    },
}
```

---

## Open Questions

1. **Model persistence**: Should embeddings persist across container restarts?
   - Current: In-memory only
   - Option: Add persistence layer (SQLite, LevelDB)

2. **Multiple indexes**: How to handle multiple codebases?
   - Option A: Single LEANN instance with namespace prefixes
   - Option B: Separate LEANN instance per index

3. **Resource limits**: How to constrain Python memory usage?
   - Option: cgroups limits in Docker
   - Option: Python-level memory monitoring

4. **GPU support**: Should we support GPU acceleration?
   - Would require NVIDIA runtime, CUDA libraries
   - Significant Docker image size increase

---

## References

- [LEANN Documentation](https://github.com/example/leann)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Go subprocess management](https://pkg.go.dev/os/exec)
- [Python socket programming](https://docs.python.org/3/library/socket.html)
