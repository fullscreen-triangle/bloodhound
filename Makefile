# Bloodhound Virtual Machine Build System
# Consciousness-Aware Scientific Computing Platform

# === Configuration ===
RUST_VERSION := 1.70
PYTHON_VERSION := 3.11
PROJECT_NAME := bloodhound-vm
DOCKER_IMAGE := $(PROJECT_NAME):latest

# Directories
SRC_DIR := src
DOCS_DIR := docs
PYTHON_DIR := python
TESTS_DIR := tests
TARGET_DIR := target
BUILD_DIR := build

# === Default Target ===
.PHONY: all
all: check build test docs

# === Help ===
.PHONY: help
help:
	@echo "Bloodhound Virtual Machine Build System"
	@echo "========================================"
	@echo ""
	@echo "Core Commands:"
	@echo "  build          - Build the consciousness-aware virtual machine"
	@echo "  test           - Run comprehensive test suite"
	@echo "  check          - Run code quality checks"
	@echo "  docs           - Generate documentation"
	@echo "  clean          - Clean build artifacts"
	@echo ""
	@echo "Development:"
	@echo "  dev            - Start development environment"
	@echo "  fmt            - Format all code"
	@echo "  lint           - Run linters"
	@echo "  bench          - Run performance benchmarks"
	@echo ""
	@echo "Virtual Machine:"
	@echo "  vm-start       - Start Bloodhound VM"
	@echo "  vm-stop        - Stop Bloodhound VM"
	@echo "  vm-status      - Check VM status"
	@echo "  vm-reset       - Reset VM to initial state"
	@echo ""
	@echo "Scientific Computing:"
	@echo "  test-genomics  - Test genomics analysis capabilities"
	@echo "  test-proteomics - Test proteomics analysis capabilities"
	@echo "  test-metabolomics - Test metabolomics analysis capabilities"
	@echo "  test-consciousness - Test consciousness-level processing"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build   - Build Docker container"
	@echo "  docker-run     - Run VM in container"
	@echo "  docker-compose - Start full development stack"
	@echo ""
	@echo "Installation:"
	@echo "  install        - Install Bloodhound VM system-wide"
	@echo "  install-deps   - Install all dependencies"
	@echo "  setup-dev      - Setup development environment"

# === Build Commands ===
.PHONY: build
build: install-deps
	@echo "🔧 Building Consciousness-Aware Virtual Machine..."
	@echo "=================================================="
	cargo build --release --all-features
	@echo "✅ Bloodhound VM built successfully"

.PHONY: build-debug
build-debug: install-deps
	@echo "🔧 Building Debug Version..."
	cargo build --all-features
	@echo "✅ Debug build complete"

.PHONY: build-components
build-components:
	@echo "🔧 Building Individual Components..."
	@echo "Building Kwasa-Kwasa (Metacognitive Orchestrator)..."
	cargo build -p kwasa-kwasa --release
	@echo "Building Kambuzuma (Neural Stack)..."
	cargo build -p kambuzuma --release
	@echo "Building Buhera (VPOS)..."
	cargo build -p buhera --release
	@echo "Building Musande (S-Entropy Solver)..."
	cargo build -p musande --release
	@echo "Building Four-Sided Triangle (Bayesian Optimizer)..."
	cargo build -p four-sided-triangle --release
	@echo "Building Purpose Framework..."
	cargo build -p purpose-framework --release
	@echo "Building Combine Harvester..."
	cargo build -p combine-harvester --release
	@echo "✅ All components built successfully"

# === Testing ===
.PHONY: test
test: build
	@echo "🧪 Running Comprehensive Test Suite..."
	@echo "======================================"
	cargo test --all-features --workspace
	@echo "✅ All tests passed"

.PHONY: test-unit
test-unit:
	@echo "🧪 Running Unit Tests..."
	cargo test --lib --all-features

.PHONY: test-integration
test-integration:
	@echo "🧪 Running Integration Tests..."
	cargo test --test '*' --all-features

.PHONY: test-consciousness
test-consciousness:
	@echo "🧠 Testing Consciousness-Level Processing..."
	cargo test consciousness --all-features --release
	cargo run --bin test-consciousness --release

.PHONY: test-s-entropy
test-s-entropy:
	@echo "🌊 Testing S-Entropy Navigation..."
	cargo test s_entropy --all-features --release
	cargo run --bin test-s-entropy --release

.PHONY: test-scientific
test-scientific: test-genomics test-proteomics test-metabolomics

.PHONY: test-genomics
test-genomics:
	@echo "🧬 Testing Genomics Analysis..."
	cargo test genomics --all-features --release
	python $(PYTHON_DIR)/tests/test_genomics_integration.py

.PHONY: test-proteomics
test-proteomics:
	@echo "🧬 Testing Proteomics Analysis..."
	cargo test proteomics --all-features --release
	python $(PYTHON_DIR)/tests/test_proteomics_integration.py

.PHONY: test-metabolomics
test-metabolomics:
	@echo "🧬 Testing Metabolomics Analysis..."
	cargo test metabolomics --all-features --release
	python $(PYTHON_DIR)/tests/test_metabolomics_integration.py

# === Code Quality ===
.PHONY: check
check: fmt lint test-unit
	@echo "✅ Code quality checks complete"

.PHONY: fmt
fmt:
	@echo "🎨 Formatting Code..."
	cargo fmt --all
	black $(PYTHON_DIR)
	isort $(PYTHON_DIR)
	@echo "✅ Code formatting complete"

.PHONY: lint
lint:
	@echo "🔍 Running Linters..."
	cargo clippy --all-features --workspace -- -D warnings
	flake8 $(PYTHON_DIR)
	mypy $(PYTHON_DIR)
	@echo "✅ Linting complete"

.PHONY: audit
audit:
	@echo "🔒 Security Audit..."
	cargo audit
	safety check -r $(PYTHON_DIR)/requirements.txt

# === Performance ===
.PHONY: bench
bench: build
	@echo "⚡ Running Performance Benchmarks..."
	@echo "==================================="
	cargo bench --all-features
	@echo "✅ Benchmarks complete"

.PHONY: bench-consciousness
bench-consciousness:
	@echo "🧠 Benchmarking Consciousness Processing..."
	cargo bench consciousness_processing

.PHONY: bench-s-entropy
bench-s-entropy:
	@echo "🌊 Benchmarking S-Entropy Navigation..."
	cargo bench s_entropy_navigation

.PHONY: bench-domain-learning
bench-domain-learning:
	@echo "📚 Benchmarking Domain Learning..."
	cargo bench domain_learning

.PHONY: profile
profile: build
	@echo "📊 Profiling Performance..."
	cargo build --release --bin bloodhound-vm
	perf record -g $(TARGET_DIR)/release/bloodhound-vm --profile
	perf report

# === Virtual Machine Operations ===
.PHONY: vm-start
vm-start: build
	@echo "🚀 Starting Bloodhound Virtual Machine..."
	@echo "========================================"
	cargo run --release --bin bloodhound-vm -- start --config bloodhound.toml

.PHONY: vm-stop
vm-stop:
	@echo "🛑 Stopping Bloodhound Virtual Machine..."
	cargo run --release --bin bloodhound-vm -- stop

.PHONY: vm-status
vm-status:
	@echo "📊 Checking VM Status..."
	cargo run --release --bin bloodhound-vm -- status

.PHONY: vm-reset
vm-reset:
	@echo "🔄 Resetting Virtual Machine..."
	cargo run --release --bin bloodhound-vm -- reset --confirm

.PHONY: vm-interactive
vm-interactive: build
	@echo "🎮 Starting Interactive VM Session..."
	cargo run --release --bin bvm-cli

# === Documentation ===
.PHONY: docs
docs:
	@echo "📚 Generating Documentation..."
	@echo "============================="
	cargo doc --all-features --workspace --no-deps
	mdbook build $(DOCS_DIR)
	@echo "✅ Documentation generated"

.PHONY: docs-open
docs-open: docs
	@echo "🌐 Opening Documentation..."
	cargo doc --all-features --open

.PHONY: docs-serve
docs-serve:
	@echo "🌐 Serving Documentation..."
	mdbook serve $(DOCS_DIR) --open

# === Python Integration ===
.PHONY: python-build
python-build:
	@echo "🐍 Building Python Interface..."
	cd $(PYTHON_DIR) && python -m build

.PHONY: python-install
python-install:
	@echo "🐍 Installing Python Interface..."
	pip install -e $(PYTHON_DIR)/

.PHONY: python-test
python-test:
	@echo "🐍 Testing Python Interface..."
	cd $(PYTHON_DIR) && python -m pytest tests/ -v

# === Docker ===
.PHONY: docker-build
docker-build:
	@echo "🐳 Building Docker Image..."
	@echo "=========================="
	docker build -t $(DOCKER_IMAGE) .
	@echo "✅ Docker image built: $(DOCKER_IMAGE)"

.PHONY: docker-run
docker-run: docker-build
	@echo "🐳 Running VM in Docker..."
	docker run -it --rm \
		-p 8080:8080 \
		-p 9090:9090 \
		-v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE)

.PHONY: docker-compose
docker-compose:
	@echo "🐳 Starting Development Stack..."
	docker-compose up -d

.PHONY: docker-compose-down
docker-compose-down:
	@echo "🐳 Stopping Development Stack..."
	docker-compose down

# === Installation ===
.PHONY: install
install: build
	@echo "📦 Installing Bloodhound VM..."
	@echo "=============================="
	cargo install --path . --force
	@echo "✅ Installation complete"

.PHONY: install-deps
install-deps:
	@echo "📦 Installing Dependencies..."
	@echo "============================"
	@if ! command -v cargo >/dev/null 2>&1; then \
		echo "❌ Rust not found. Installing..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh; \
		source $$HOME/.cargo/env; \
	fi
	@if ! command -v python$(PYTHON_VERSION) >/dev/null 2>&1; then \
		echo "❌ Python $(PYTHON_VERSION) not found. Please install it."; \
		exit 1; \
	fi
	rustup update
	pip install -r $(PYTHON_DIR)/requirements.txt
	@echo "✅ Dependencies installed"

.PHONY: setup-dev
setup-dev: install-deps
	@echo "🛠️  Setting up Development Environment..."
	@echo "========================================"
	rustup component add clippy rustfmt
	pip install pre-commit black isort flake8 mypy safety
	pre-commit install
	@echo "✅ Development environment ready"

# === Cleanup ===
.PHONY: clean
clean:
	@echo "🧹 Cleaning Build Artifacts..."
	@echo "============================="
	cargo clean
	rm -rf $(BUILD_DIR)
	rm -rf $(PYTHON_DIR)/build
	rm -rf $(PYTHON_DIR)/dist
	rm -rf $(PYTHON_DIR)/*.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

.PHONY: clean-all
clean-all: clean
	@echo "🧹 Deep Cleaning..."
	docker system prune -f
	cargo cache -a

# === Research and Development ===
.PHONY: research-validate
research-validate:
	@echo "🔬 Validating Research Claims..."
	@echo "==============================="
	cargo test --release research_validation
	python scripts/validate_s_entropy_theory.py
	python scripts/validate_consciousness_metrics.py

.PHONY: research-reproduce
research-reproduce:
	@echo "🔬 Reproducing Research Results..."
	python scripts/reproduce_paper_results.py

# === Utility Targets ===
.PHONY: version
version:
	@echo "Bloodhound VM Version Information"
	@echo "================================"
	@echo "Version: $(shell cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "bloodhound-vm") | .version')"
	@echo "Rust: $(shell rustc --version)"
	@echo "Python: $(shell python --version)"
	@echo "Features: consciousness-processing, s-entropy-navigation, purpose-framework, combine-harvester"

.PHONY: deps-update
deps-update:
	@echo "🔄 Updating Dependencies..."
	cargo update
	pip-review --auto

# === Integration Tests ===
.PHONY: test-full-integration
test-full-integration: build
	@echo "🔗 Running Full Integration Tests..."
	@echo "==================================="
	./scripts/integration_tests.sh

# Make sure intermediate files are not deleted
.PRECIOUS: $(TARGET_DIR)/%.o

# Performance monitoring
.PHONY: monitor
monitor:
	@echo "📊 Starting Performance Monitor..."
	htop -p $(shell pgrep bloodhound-vm)