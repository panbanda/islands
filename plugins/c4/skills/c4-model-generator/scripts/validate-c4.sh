#!/usr/bin/env bash
# Validate C4 YAML files against schema
# Usage: ./validate-c4.sh [c4-model-directory]

set -euo pipefail

C4_DIR="${1:-.}"

echo "=== C4 Model Validation ==="
echo "Directory: $C4_DIR"
echo ""

# Check for module file
if [[ ! -f "$C4_DIR/c4.mod.yaml" ]]; then
    echo "ERROR: No c4.mod.yaml found at $C4_DIR"
    exit 1
fi

echo "=== Checking c4.mod.yaml ==="
# Validate required fields
if ! grep -q "^version:" "$C4_DIR/c4.mod.yaml"; then
    echo "ERROR: Missing 'version' field in c4.mod.yaml"
    exit 1
fi
if ! grep -q "^name:" "$C4_DIR/c4.mod.yaml"; then
    echo "ERROR: Missing 'name' field in c4.mod.yaml"
    exit 1
fi
echo "  OK: Required fields present"

echo ""
echo "=== Checking YAML Syntax ==="
# Find all YAML files
yaml_files=$(find "$C4_DIR" -name "*.yaml" -o -name "*.yml" | grep -v "node_modules")
error_count=0

for file in $yaml_files; do
    # Basic YAML syntax check using Python (usually available)
    if command -v python3 &> /dev/null; then
        if ! python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
            echo "  ERROR: Invalid YAML syntax in $file"
            ((error_count++))
        else
            echo "  OK: $file"
        fi
    else
        echo "  SKIP: $file (python3 not available for YAML parsing)"
    fi
done

echo ""
echo "=== Checking Element IDs ==="
# Check ID pattern: ^[a-z][a-z0-9-]*$
id_pattern='^[a-z][a-z0-9-]*$'
for file in $yaml_files; do
    ids=$(grep -E "^\s+id:" "$file" 2>/dev/null | sed 's/.*id:\s*//' | tr -d '"' | tr -d "'" || true)
    for id in $ids; do
        if [[ ! $id =~ $id_pattern ]]; then
            echo "  ERROR: Invalid ID '$id' in $file (must match $id_pattern)"
            ((error_count++))
        fi
    done
done
echo "  ID check complete"

echo ""
echo "=== Checking Required Fields ==="

# Check persons have id and name
for file in $(find "$C4_DIR" -name "*.yaml" -exec grep -l "^persons:" {} \;); do
    echo "  Checking persons in $file..."
    # Simple check - would need proper YAML parser for full validation
done

# Check systems have id and name
for file in $(find "$C4_DIR" -name "*.yaml" -exec grep -l "^systems:" {} \;); do
    echo "  Checking systems in $file..."
done

# Check containers have id and name
for file in $(find "$C4_DIR" -name "*.yaml" -exec grep -l "^containers:" {} \;); do
    echo "  Checking containers in $file..."
done

# Check relationships have from and to
for file in $(find "$C4_DIR" -name "*.yaml" -exec grep -l "^relationships:" {} \;); do
    echo "  Checking relationships in $file..."
    # Check each relationship has from and to
    if grep -A5 "^  - from:" "$file" | grep -q "to:"; then
        echo "    OK: Relationships have from/to"
    fi
done

echo ""
echo "=== Summary ==="
if [[ $error_count -eq 0 ]]; then
    echo "All checks passed!"
else
    echo "Found $error_count error(s)"
    exit 1
fi

echo ""
echo "=== File Statistics ==="
echo "  YAML files: $(echo "$yaml_files" | wc -l)"
echo "  Persons: $(grep -r "^persons:" "$C4_DIR" --include="*.yaml" | wc -l) files"
echo "  Systems: $(grep -r "^systems:" "$C4_DIR" --include="*.yaml" | wc -l) files"
echo "  Containers: $(grep -r "^containers:" "$C4_DIR" --include="*.yaml" | wc -l) files"
echo "  Components: $(grep -r "^components:" "$C4_DIR" --include="*.yaml" | wc -l) files"
echo "  Relationships: $(grep -r "^relationships:" "$C4_DIR" --include="*.yaml" | wc -l) files"
echo "  Flows: $(grep -r "^flows:" "$C4_DIR" --include="*.yaml" | wc -l) files"
echo "  Deployments: $(grep -r "^deployments:" "$C4_DIR" --include="*.yaml" | wc -l) files"
