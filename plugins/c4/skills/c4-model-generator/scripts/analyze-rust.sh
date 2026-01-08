#!/usr/bin/env bash
# Analyze Rust codebase structure for C4 modeling
# Usage: ./analyze-rust.sh [path-to-crate]

set -euo pipefail

CRATE_PATH="${1:-.}"

echo "=== Rust Crate Analysis for C4 Modeling ==="
echo "Path: $CRATE_PATH"
echo ""

# Check if it's a Rust project
if [[ ! -f "$CRATE_PATH/Cargo.toml" ]]; then
    echo "Error: No Cargo.toml found at $CRATE_PATH"
    exit 1
fi

echo "=== Package Info ==="
grep -E "^name|^version|^description" "$CRATE_PATH/Cargo.toml" | head -10

echo ""
echo "=== Binary Targets (Containers) ==="
grep -A2 '\[\[bin\]\]' "$CRATE_PATH/Cargo.toml" 2>/dev/null || echo "No explicit binaries (default: src/main.rs)"
if [[ -f "$CRATE_PATH/src/main.rs" ]]; then
    echo "  - Default binary from src/main.rs"
fi

echo ""
echo "=== Library Target ==="
if [[ -f "$CRATE_PATH/src/lib.rs" ]]; then
    echo "  - Library: src/lib.rs"
else
    echo "  - No library target"
fi

echo ""
echo "=== Top-Level Modules (Potential Components) ==="
if [[ -f "$CRATE_PATH/src/lib.rs" ]]; then
    echo "From lib.rs:"
    grep -E "^pub mod |^mod " "$CRATE_PATH/src/lib.rs" 2>/dev/null | sed 's/^/  /'
fi
if [[ -f "$CRATE_PATH/src/main.rs" ]]; then
    echo "From main.rs:"
    grep -E "^pub mod |^mod " "$CRATE_PATH/src/main.rs" 2>/dev/null | sed 's/^/  /'
fi

echo ""
echo "=== Source Directory Structure ==="
if command -v eza &> /dev/null; then
    eza --tree -L 2 "$CRATE_PATH/src" 2>/dev/null || tree -L 2 "$CRATE_PATH/src" 2>/dev/null
else
    find "$CRATE_PATH/src" -maxdepth 2 -type d | sed 's/^/  /'
fi

echo ""
echo "=== Feature Flags ==="
grep -A20 '^\[features\]' "$CRATE_PATH/Cargo.toml" 2>/dev/null | head -20 || echo "No feature flags"

echo ""
echo "=== Key Dependencies (Potential External Systems) ==="
grep -E "^(tokio|axum|actix|warp|hyper|reqwest|sqlx|diesel|redis|kafka|pulsar|nats|openai|anthropic)" "$CRATE_PATH/Cargo.toml" | sed 's/^/  /'

echo ""
echo "=== Database/Storage Dependencies ==="
grep -E "(postgres|mysql|sqlite|mongodb|redis|rocksdb|sled|dynamodb)" "$CRATE_PATH/Cargo.toml" | sed 's/^/  /'

echo ""
echo "=== HTTP Client Usage (External Integrations) ==="
rg -l "reqwest|hyper::Client|surf|ureq" "$CRATE_PATH/src" 2>/dev/null | sed 's/^/  /' || echo "  No HTTP clients found"

echo ""
echo "=== gRPC/Protobuf Usage ==="
rg -l "tonic|prost|grpc" "$CRATE_PATH/src" 2>/dev/null | sed 's/^/  /' || echo "  No gRPC found"

echo ""
echo "=== Async Runtime ==="
grep -E "^tokio|^async-std|^smol" "$CRATE_PATH/Cargo.toml" | head -1 || echo "  Sync only"

echo ""
echo "=== Analysis Complete ==="
echo "Use this information to populate your C4 model:"
echo "  - Package name -> System ID"
echo "  - Binaries -> Containers"
echo "  - Top-level modules -> Components"
echo "  - External dependencies -> External Systems"
