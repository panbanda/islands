.PHONY: all build test lint clean install docker helm

VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE ?= $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
LDFLAGS := -s -w \
	-X github.com/jon/pythia/cmd/pythia.Version=$(VERSION) \
	-X github.com/jon/pythia/cmd/pythia.Commit=$(COMMIT) \
	-X github.com/jon/pythia/cmd/pythia.BuildDate=$(BUILD_DATE)

all: lint test build

build:
	CGO_ENABLED=0 go build -ldflags="$(LDFLAGS)" -o pythia .

install:
	CGO_ENABLED=0 go install -ldflags="$(LDFLAGS)" .

test:
	go test -v -race -coverprofile=coverage.out -covermode=atomic ./...

test-coverage: test
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

lint:
	golangci-lint run --timeout=5m

lint-fix:
	golangci-lint run --fix --timeout=5m

clean:
	rm -f pythia coverage.out coverage.html
	rm -rf dist/

# Docker targets
docker-build:
	docker build -f deployments/docker/Dockerfile \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		-t pythia:$(VERSION) .

docker-push:
	docker tag pythia:$(VERSION) ghcr.io/jon/pythia:$(VERSION)
	docker push ghcr.io/jon/pythia:$(VERSION)

# Helm targets
helm-lint:
	helm lint deployments/helm/pythia

helm-template:
	helm template pythia deployments/helm/pythia

helm-package:
	helm package deployments/helm/pythia -d dist/

# Development helpers
dev:
	go run . serve

run-index:
	go run . index .

run-search:
	go run . search "$(QUERY)"

# Security scanning
security:
	gosec ./...
	govulncheck ./...

# Complexity analysis
complexity:
	gocyclo -over 10 .
	gocognit -over 15 .

# Generate mocks (if needed)
generate:
	go generate ./...

# Format code
fmt:
	gofumpt -w .
	goimports -w .

# Check if code is formatted
check-fmt:
	@test -z "$$(gofumpt -l .)" || (echo "Code not formatted. Run 'make fmt'" && exit 1)

# Verify go.mod is tidy
tidy:
	go mod tidy
	@git diff --exit-code go.mod go.sum || (echo "go.mod is not tidy. Run 'go mod tidy'" && exit 1)

# Run all checks (CI simulation)
ci: tidy check-fmt lint test security complexity
	@echo "All CI checks passed!"
