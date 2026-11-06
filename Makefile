# ==============================================================================
# BurnNotice Root Makefile
# ==============================================================================
# Real-time latent governance, alignment monitoring, and dynamic activation
# steering for autonomous LLM agents.
# ==============================================================================

SHELL           := /bin/bash
BINARY_NAME     := burnnotice
DASHBOARD_NAME  := burnnotice-dashboard
BIN_DIR         := bin
DIST_DIR        := dist
CMD_BURN        := ./cmd/burnnotice
CMD_DASH        := ./cmd/dashboard

# Versioning and build metadata
VERSION         ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "0.8.0-dev")
COMMIT          ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE      ?= $(shell date -u +'%Y-%m-%dT%H:%M:%SZ')
GO_VERSION      ?= $(shell go version 2>/dev/null | awk '{print $$3}')

# Compiler flags
LDFLAGS_COMMON  := -X "main.version=$(VERSION)" -X "main.commit=$(COMMIT)" -X "main.date=$(BUILD_DATE)"
LDFLAGS_DEV     := -ldflags '$(LDFLAGS_COMMON)'
LDFLAGS_RELEASE := -ldflags '-s -w $(LDFLAGS_COMMON)' -trimpath
GCFLAGS_DEBUG   := -gcflags 'all=-N -l'

# Tooling
GO              ?= go
GOLANGCI_LINT   ?= $(shell command -v golangci-lint 2>/dev/null || echo /home/rsd/go/bin/golangci-lint)
GOSEC           ?= $(shell command -v gosec 2>/dev/null || echo /home/rsd/go/bin/gosec)
GOVULNCHECK     ?= $(shell command -v govulncheck 2>/dev/null || echo /home/rsd/go/bin/govulncheck)
DEADCODE        ?= $(shell command -v deadcode 2>/dev/null || echo /home/rsd/go/bin/deadcode)
TEMPL           ?= $(shell command -v templ 2>/dev/null)
PROTOC          ?= $(shell command -v protoc 2>/dev/null)

# Test knobs
FUZZ_TIME       ?= 10
COVER_FILE      ?= coverage.out
COVER_HTML      ?= coverage.html

# Terminal colors
COLOR_RESET     := \033[0m
COLOR_BOLD      := \033[1m
COLOR_CYAN      := \033[36m
COLOR_GREEN     := \033[32m
COLOR_YELLOW    := \033[33m

.PHONY: all help build build-burnnotice build-dashboard build-release build-debug \
        run run-dashboard docker-build \
        clean fmt fmt-check vet lint lint-full staticcheck deadcode \
        templ proto \
        test test-unit test-race test-coverage test-bench test-fuzz test-fuzz-all \
        test-all test-shuffle test-oracle test-e2e-playwright \
        security sec-gosec sec-vulncheck sec-audit \
        profile profile-cpu profile-mem \
        accuracy-gate oracle-gate quality-gate \
        vet-cross-platform \
        verify pre-commit makehooks install-hooks install-commit-hook

all: lint test build

##@ Help
help: ## Display available Make targets
	@echo -e "$(COLOR_BOLD)BurnNotice Build System$(COLOR_RESET)"
	@echo -e "Version: $(COLOR_GREEN)$(VERSION)$(COLOR_RESET) | Commit: $(COLOR_CYAN)$(COMMIT)$(COLOR_RESET) | Go: $(COLOR_YELLOW)$(GO_VERSION)$(COLOR_RESET)"
	@echo -e "Usage:   make $(COLOR_GREEN)<target>$(COLOR_RESET)"
	@awk '/^[a-zA-Z0-9_-]+:.*##/ { \
		match($$0, /^[a-zA-Z0-9_-]+/); \
		target=substr($$0, RSTART, RLENGTH); \
		match($$0, /## *(.*)$$/, desc); \
		printf "  $(COLOR_GREEN)%-24s$(COLOR_RESET) %s\n", target, desc[1]; \
	} /^##@/ { \
		printf "\n$(COLOR_BOLD)$(COLOR_CYAN)%s$(COLOR_RESET)\n", substr($$0, 5); \
	}' $(MAKEFILE_LIST)

##@ Build
build: build-burnnotice build-dashboard ## Build burnnotice and dashboard binaries
	@echo -e "$(COLOR_GREEN)Build complete.$(COLOR_RESET)"

build-burnnotice: ## Compile the burnnotice CLI binary
	@echo -e "$(COLOR_CYAN)Building $(BINARY_NAME) ($(VERSION))...$(COLOR_RESET)"
	@mkdir -p $(BIN_DIR)
	$(GO) build $(LDFLAGS_DEV) -o $(BIN_DIR)/$(BINARY_NAME) $(CMD_BURN)
	@echo -e "$(COLOR_GREEN)Built $(BIN_DIR)/$(BINARY_NAME)$(COLOR_RESET)"

build-dashboard: ## Compile the burnnotice dashboard binary
	@echo -e "$(COLOR_CYAN)Building $(DASHBOARD_NAME) ($(VERSION))...$(COLOR_RESET)"
	@mkdir -p $(BIN_DIR)
	$(GO) build $(LDFLAGS_DEV) -o $(BIN_DIR)/$(DASHBOARD_NAME) $(CMD_DASH)
	@echo -e "$(COLOR_GREEN)Built $(BIN_DIR)/$(DASHBOARD_NAME)$(COLOR_RESET)"

build-release: ## Compile stripped, optimized release binaries
	@echo -e "$(COLOR_CYAN)Building release binaries...$(COLOR_RESET)"
	@mkdir -p $(BIN_DIR)
	$(GO) build $(LDFLAGS_RELEASE) -o $(BIN_DIR)/$(BINARY_NAME) $(CMD_BURN)
	$(GO) build $(LDFLAGS_RELEASE) -o $(BIN_DIR)/$(DASHBOARD_NAME) $(CMD_DASH)
	@echo -e "$(COLOR_GREEN)Release builds ready in $(BIN_DIR)/$(COLOR_RESET)"

build-debug: ## Compile binaries with debug symbols (-N -l)
	@echo -e "$(COLOR_CYAN)Building debug binaries...$(COLOR_RESET)"
	@mkdir -p $(BIN_DIR)
	$(GO) build $(GCFLAGS_DEBUG) $(LDFLAGS_DEV) -o $(BIN_DIR)/$(BINARY_NAME)-debug $(CMD_BURN)
	$(GO) build $(GCFLAGS_DEBUG) $(LDFLAGS_DEV) -o $(BIN_DIR)/$(DASHBOARD_NAME)-debug $(CMD_DASH)

run: build-burnnotice ## Build and run the burnnotice CLI
	./$(BIN_DIR)/$(BINARY_NAME)

run-dashboard: build-dashboard ## Build and run the dashboard
	./$(BIN_DIR)/$(DASHBOARD_NAME) --port 8080

docker-build: ## Build the standard container image
	@echo -e "$(COLOR_CYAN)Building Docker image burnnotice:$(VERSION)...$(COLOR_RESET)"
	docker build -t burnnotice:$(VERSION) -t burnnotice:latest .
	@echo -e "$(COLOR_GREEN)Docker image built.$(COLOR_RESET)"

##@ Code Generation
templ: ## Generate Go code from .templ files (if templ is installed)
	@if [ -n "$(TEMPL)" ]; then \
		echo -e "$(COLOR_CYAN)Running templ generate...$(COLOR_RESET)"; \
		templ generate; \
	else \
		echo "templ not installed; skipping"; \
	fi

proto: ## Generate Go protobuf bindings from internal/onnxpb/onnx.in.proto (if present)
	@if [ -n "$(PROTOC)" ] && [ -f internal/onnxpb/onnx.in.proto ]; then \
		echo -e "$(COLOR_CYAN)Running protoc...$(COLOR_RESET)"; \
		protoc --go_out=. --go-grpc_out=. internal/onnxpb/onnx.in.proto; \
	else \
		echo "protoc or onnx.in.proto not available; skipping"; \
	fi

##@ Quality: Formatting, Vet, Lint
fmt: ## Format Go sources with gofmt -s
	@echo -e "$(COLOR_CYAN)Formatting Go sources...$(COLOR_RESET)"
	gofmt -s -w .
	@echo -e "$(COLOR_GREEN)Formatting complete.$(COLOR_RESET)"

fmt-check: ## Verify Go formatting without modifying files
	@echo -e "$(COLOR_CYAN)Checking Go formatting...$(COLOR_RESET)"
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then \
		echo "Unformatted files:"; \
		echo "$$unformatted"; \
		exit 1; \
	fi
	@echo -e "$(COLOR_GREEN)All files formatted.$(COLOR_RESET)"

vet: ## Run go vet across all packages
	@echo -e "$(COLOR_CYAN)Running go vet ./...$(COLOR_RESET)"
	$(GO) vet ./...
	@echo -e "$(COLOR_GREEN)go vet passed.$(COLOR_RESET)"

lint: ## Run golangci-lint
	@echo -e "$(COLOR_CYAN)Running golangci-lint...$(COLOR_RESET)"
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	elif [ -x "$(GOLANGCI_LINT)" ]; then \
		"$(GOLANGCI_LINT)" run ./...; \
	else \
		echo "golangci-lint not installed; skipping"; \
	fi

lint-full: fmt-check vet lint ## Run formatting, vet, and golangci-lint
	@echo -e "$(COLOR_GREEN)Full lint pass complete.$(COLOR_RESET)"

staticcheck: ## Run staticcheck via golangci-lint
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run -E staticcheck,unused ./...; \
	fi

deadcode: ## Run deadcode reachability analysis
	@if [ -x "$(DEADCODE)" ]; then \
		$(DEADCODE) ./cmd/... ./internal/...; \
	else \
		echo "deadcode not installed; skipping"; \
	fi

##@ Security
security: sec-gosec sec-vulncheck ## Run full Go security audit
	@echo -e "$(COLOR_GREEN)Security analysis passed.$(COLOR_RESET)"

sec-gosec: ## Run gosec AST security analysis
	@echo -e "$(COLOR_CYAN)Running gosec...$(COLOR_RESET)"
	$(GOSEC) ./...

sec-vulncheck: ## Run govulncheck against the Go vulnerability database
	@echo -e "$(COLOR_CYAN)Running govulncheck...$(COLOR_RESET)"
	$(GOVULNCHECK) ./...

sec-audit: ## Audit module dependencies
	@echo -e "$(COLOR_CYAN)Auditing module dependencies...$(COLOR_RESET)"
	$(GO) list -m all > /dev/null

##@ Testing
test: ## Run the default unit test suite
	$(GO) test ./...

test-unit: ## Run all package unit tests
	$(GO) test -count=1 ./...

test-race: ## Run all tests with the race detector
	$(GO) test -race ./...

test-coverage: ## Generate coverage profile and HTML report
	$(GO) test -coverprofile=$(COVER_FILE) ./...
	$(GO) tool cover -html=$(COVER_FILE) -o $(COVER_HTML)
	@echo -e "$(COLOR_GREEN)Coverage report: $(COVER_HTML)$(COLOR_RESET)"

test-bench: ## Run Go benchmarks with allocation reporting
	$(GO) test -bench=. -benchmem ./...

test-fuzz: ## Run smoke fuzz tests across fuzz-enabled packages
	@echo -e "$(COLOR_CYAN)Running smoke fuzz tests...$(COLOR_RESET)"
	@set -e; \
	for pkg in internal/math internal/extract internal/scoring internal/steer internal/metrics internal/zjson internal/converter internal/policy; do \
		for fn in $$($(GO) test -list 'Fuzz' ./$$pkg 2>/dev/null | grep '^Fuzz'); do \
			echo "  $$pkg :: $$fn"; \
			$(GO) test -run='^$$' -fuzz="^$${fn}$$" -fuzztime=$(FUZZ_TIME)s ./$$pkg; \
		done; \
	done
	@echo -e "$(COLOR_GREEN)Fuzz smoke tests passed.$(COLOR_RESET)"

test-fuzz-all: test-fuzz ## Run all discovered native Go fuzz targets
	@echo -e "$(COLOR_GREEN)All fuzz targets executed.$(COLOR_RESET)"

test-shuffle: ## Run unit tests with randomized execution order
	$(GO) test -shuffle=on ./...

test-oracle: ## Run deterministic unit tests twice for flaky/oracle drift
	$(GO) test -count=2 ./...

test-all: test test-race test-bench test-fuzz ## Run the exhaustive test matrix
	@echo -e "$(COLOR_GREEN)Exhaustive test matrix complete.$(COLOR_RESET)"

test-e2e-playwright: ## Run Playwright dashboard E2E tests
	@if [ -d e2e ]; then \
		cd e2e && npm test; \
	fi

##@ Performance & Memory Profiling
profile: profile-cpu profile-mem ## Run CPU and memory profiling benchmarks

profile-cpu: ## Run benchmarks with CPU profiling
	@echo -e "$(COLOR_CYAN)Running CPU profiling...$(COLOR_RESET)"
	$(GO) test -run='^$$' -bench=. -cpuprofile=cpu.out -benchtime=1s ./internal/scoring ./internal/steer ./internal/math
	@echo -e "$(COLOR_GREEN)CPU profile: cpu.out$(COLOR_RESET)"

profile-mem: ## Run benchmarks with memory profiling
	@echo -e "$(COLOR_CYAN)Running memory profiling...$(COLOR_RESET)"
	$(GO) test -run='^$$' -bench=. -memprofile=mem.out -benchtime=1s ./internal/scoring ./internal/steer ./internal/math
	@echo -e "$(COLOR_GREEN)Memory profile: mem.out$(COLOR_RESET)"

##@ Accuracy & Oracle Quality Gates
accuracy-gate: ## Run accuracy-focused tests and invariants
	@echo -e "$(COLOR_CYAN)Running accuracy gate...$(COLOR_RESET)"
	$(GO) test -run='Accuracy|Invariant|Cosine|Recall|Rank|Threshold|Calibration' -timeout 10m ./...
	@echo -e "$(COLOR_GREEN)Accuracy gate passed.$(COLOR_RESET)"

oracle-gate: ## Run deterministic oracle/golden tests
	@echo -e "$(COLOR_CYAN)Running oracle gate...$(COLOR_RESET)"
	$(GO) test -count=2 -run='Oracle|Golden|Determin' -timeout 10m ./...
	@echo -e "$(COLOR_GREEN)Oracle gate passed.$(COLOR_RESET)"

quality-gate: lint-full security accuracy-gate oracle-gate ## Run all quality gates
	@echo -e "$(COLOR_GREEN)Quality gate passed.$(COLOR_RESET)"

##@ Cross-Platform Verification
vet-cross-platform: ## Vet packages for windows/darwin/linux targets
	@echo -e "$(COLOR_CYAN)Vetting cross-platform targets...$(COLOR_RESET)"
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 $(GO) vet ./internal/... ./cmd/...
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 $(GO) vet ./internal/... ./cmd/...
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 $(GO) vet ./internal/... ./cmd/...
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 $(GO) vet ./internal/... ./cmd/...
	@echo -e "$(COLOR_GREEN)Cross-platform vet passed.$(COLOR_RESET)"

##@ Git Hooks
verify: ## Run scripts/verify.sh quality gate
	bash scripts/verify.sh $(FUZZ_TIME)

pre-commit: verify ## Alias for verify

makehooks: ## Install scripts/verify.sh into .git/hooks/pre-commit
	@echo "Installing Git pre-commit hook (scripts/verify.sh -> .git/hooks/pre-commit)..."
	@mkdir -p .git/hooks
	@chmod +x scripts/verify.sh
	@ln -sf ../../scripts/verify.sh .git/hooks/pre-commit || cp scripts/verify.sh .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hook installed."

install-hooks: makehooks ## Alias for makehooks

install-commit-hook: makehooks ## Alias for makehooks

##@ Maintenance
clean: ## Remove build artifacts, coverage, and profiling outputs
	@echo -e "$(COLOR_YELLOW)Cleaning...$(COLOR_RESET)"
	rm -rf $(BIN_DIR) $(DIST_DIR) coverage $(COVER_FILE) $(COVER_HTML) *.out perf-*.out
	@echo -e "$(COLOR_GREEN)Clean complete.$(COLOR_RESET)"
