# Bloodhound Virtual Machine - Just Commands
# Consciousness-Aware Scientific Computing Platform
# Alternative to Makefile using the Just command runner

# Configuration
project_name := "bloodhound-vm"
rust_version := "1.70"
python_version := "3.11"

# Default recipe
default:
    @just --list

# === Core Development Commands ===

# Build the consciousness-aware virtual machine
build:
    @echo "🔧 Building Consciousness-Aware Virtual Machine..."
    cargo build --release --all-features
    @echo "✅ Bloodhound VM built successfully"

# Run comprehensive test suite
test:
    @echo "🧪 Running Comprehensive Test Suite..."
    cargo test --all-features --workspace
    @echo "✅ All tests passed"

# Check code quality
check: fmt lint test-unit
    @echo "✅ Code quality checks complete"

# Format all code
fmt:
    @echo "🎨 Formatting Code..."
    cargo fmt --all
    black python/
    isort python/
    @echo "✅ Code formatting complete"

# Run linters
lint:
    @echo "🔍 Running Linters..."
    cargo clippy --all-features --workspace -- -D warnings
    flake8 python/
    mypy python/
    @echo "✅ Linting complete"

# Clean build artifacts
clean:
    @echo "🧹 Cleaning Build Artifacts..."
    cargo clean
    rm -rf build/
    rm -rf python/build/
    rm -rf python/dist/
    find . -name "__pycache__" -type d -exec rm -rf {} +
    @echo "✅ Cleanup complete"

# === Virtual Machine Operations ===

# Start Bloodhound VM
vm-start: build
    @echo "🚀 Starting Bloodhound Virtual Machine..."
    cargo run --release --bin bloodhound-vm -- start --config bloodhound.toml

# Stop Bloodhound VM
vm-stop:
    @echo "🛑 Stopping Bloodhound Virtual Machine..."
    cargo run --release --bin bloodhound-vm -- stop

# Check VM status
vm-status:
    @echo "📊 Checking VM Status..."
    cargo run --release --bin bloodhound-vm -- status

# Reset VM to initial state
vm-reset:
    @echo "🔄 Resetting Virtual Machine..."
    cargo run --release --bin bloodhound-vm -- reset --confirm

# Start interactive VM session
vm-interactive: build
    @echo "🎮 Starting Interactive VM Session..."
    cargo run --release --bin bvm-cli

# === Consciousness and S-Entropy Testing ===

# Test consciousness-level processing
test-consciousness:
    @echo "🧠 Testing Consciousness-Level Processing..."
    cargo test consciousness --all-features --release
    cargo run --bin test-consciousness --release

# Test S-entropy navigation
test-s-entropy:
    @echo "🌊 Testing S-Entropy Navigation..."
    cargo test s_entropy --all-features --release
    cargo run --bin test-s-entropy --release

# Test Purpose Framework domain learning
test-purpose:
    @echo "📚 Testing Purpose Framework..."
    cargo test purpose_framework --all-features --release

# Test Combine Harvester integration
test-combine-harvester:
    @echo "🔄 Testing Combine Harvester..."
    cargo test combine_harvester --all-features --release

# === Scientific Computing Tests ===

# Test genomics analysis
test-genomics:
    @echo "🧬 Testing Genomics Analysis..."
    cargo test genomics --all-features --release
    python python/tests/test_genomics_integration.py

# Test proteomics analysis
test-proteomics:
    @echo "🧬 Testing Proteomics Analysis..."
    cargo test proteomics --all-features --release
    python python/tests/test_proteomics_integration.py

# Test metabolomics analysis
test-metabolomics:
    @echo "🧬 Testing Metabolomics Analysis..."
    cargo test metabolomics --all-features --release
    python python/tests/test_metabolomics_integration.py

# Run all scientific tests
test-scientific: test-genomics test-proteomics test-metabolomics

# === Performance and Benchmarking ===

# Run performance benchmarks
bench: build
    @echo "⚡ Running Performance Benchmarks..."
    cargo bench --all-features
    @echo "✅ Benchmarks complete"

# Benchmark consciousness processing
bench-consciousness:
    @echo "🧠 Benchmarking Consciousness Processing..."
    cargo bench consciousness_processing

# Benchmark S-entropy navigation
bench-s-entropy:
    @echo "🌊 Benchmarking S-Entropy Navigation..."
    cargo bench s_entropy_navigation

# Benchmark domain learning
bench-domain-learning:
    @echo "📚 Benchmarking Domain Learning..."
    cargo bench domain_learning

# Profile performance
profile: build
    @echo "📊 Profiling Performance..."
    cargo build --release --bin bloodhound-vm
    perf record -g target/release/bloodhound-vm --profile
    perf report

# === Documentation ===

# Generate all documentation
docs:
    @echo "📚 Generating Documentation..."
    cargo doc --all-features --workspace --no-deps
    mdbook build docs/
    @echo "✅ Documentation generated"

# Open documentation in browser
docs-open: docs
    @echo "🌐 Opening Documentation..."
    cargo doc --all-features --open

# Serve documentation locally
docs-serve:
    @echo "🌐 Serving Documentation..."
    mdbook serve docs/ --open

# === Python Integration ===

# Build Python interface
python-build:
    @echo "🐍 Building Python Interface..."
    cd python && python -m build

# Install Python interface
python-install:
    @echo "🐍 Installing Python Interface..."
    pip install -e python/

# Test Python interface
python-test:
    @echo "🐍 Testing Python Interface..."
    cd python && python -m pytest tests/ -v

# === Docker Operations ===

# Build Docker image
docker-build:
    @echo "🐳 Building Docker Image..."
    docker build -t {{project_name}}:latest .
    @echo "✅ Docker image built: {{project_name}}:latest"

# Run VM in Docker
docker-run: docker-build
    @echo "🐳 Running VM in Docker..."
    docker run -it --rm \
        -p 8080:8080 \
        -p 9090:9090 \
        -v $(pwd)/data:/app/data \
        {{project_name}}:latest

# Start development stack
docker-compose:
    @echo "🐳 Starting Development Stack..."
    docker-compose up -d

# Stop development stack
docker-compose-down:
    @echo "🐳 Stopping Development Stack..."
    docker-compose down

# === Installation and Setup ===

# Install Bloodhound VM system-wide
install: build
    @echo "📦 Installing Bloodhound VM..."
    cargo install --path . --force
    @echo "✅ Installation complete"

# Install all dependencies
install-deps:
    @echo "📦 Installing Dependencies..."
    @if ! command -v cargo >/dev/null 2>&1; then \
        echo "❌ Rust not found. Installing..."; \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh; \
        source $HOME/.cargo/env; \
    fi
    rustup update
    pip install -r requirements.txt
    @echo "✅ Dependencies installed"

# Setup development environment
setup-dev: install-deps
    @echo "🛠️  Setting up Development Environment..."
    rustup component add clippy rustfmt
    pip install pre-commit black isort flake8 mypy safety
    pre-commit install
    @echo "✅ Development environment ready"

# === Research and Validation ===

# Validate research claims
research-validate:
    @echo "🔬 Validating Research Claims..."
    cargo test --release research_validation
    python scripts/validate_s_entropy_theory.py
    python scripts/validate_consciousness_metrics.py

# Reproduce research results
research-reproduce:
    @echo "🔬 Reproducing Research Results..."
    python scripts/reproduce_paper_results.py

# === Utility Commands ===

# Show version information
version:
    @echo "Bloodhound VM Version Information"
    @echo "================================"
    @echo "Version: $(cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "bloodhound-vm") | .version')"
    @echo "Rust: $(rustc --version)"
    @echo "Python: $(python --version)"
    @echo "Features: consciousness-processing, s-entropy-navigation, purpose-framework, combine-harvester"

# Update dependencies
deps-update:
    @echo "🔄 Updating Dependencies..."
    cargo update
    pip-review --auto

# Security audit
audit:
    @echo "🔒 Security Audit..."
    cargo audit
    safety check -r requirements.txt

# Run full integration tests
test-full-integration: build
    @echo "🔗 Running Full Integration Tests..."
    ./scripts/integration_tests.sh

# Monitor VM performance
monitor:
    @echo "📊 Starting Performance Monitor..."
    htop -p $(pgrep bloodhound-vm)

# === Advanced Operations ===

# Build individual components
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

# Test specific component
test-component component:
    @echo "🧪 Testing Component: {{component}}"
    cargo test -p {{component}} --all-features --release

# Run consciousness-level analysis on data
analyze-data data_path analysis_type:
    @echo "🔬 Running Consciousness-Level Analysis..."
    cargo run --release --bin bloodhound-vm -- analyze \
        --data {{data_path}} \
        --type {{analysis_type}} \
        --consciousness-level full

# Generate synthetic test data
generate-test-data:
    @echo "🧬 Generating Synthetic Test Data..."
    python scripts/generate_synthetic_data.py

# Validate VM consciousness coherence
validate-consciousness:
    @echo "🧠 Validating Consciousness Coherence..."
    cargo run --release --bin validate-consciousness

# === Help and Information ===

# Show detailed help
help:
    @echo "Bloodhound Virtual Machine - Just Commands"
    @echo "=========================================="
    @echo ""
    @echo "Core Commands:"
    @echo "  just build          - Build the consciousness-aware virtual machine"
    @echo "  just test           - Run comprehensive test suite"
    @echo "  just check          - Run code quality checks"
    @echo "  just docs           - Generate documentation"
    @echo "  just clean          - Clean build artifacts"
    @echo ""
    @echo "Virtual Machine:"
    @echo "  just vm-start       - Start Bloodhound VM"
    @echo "  just vm-stop        - Stop Bloodhound VM"
    @echo "  just vm-status      - Check VM status"
    @echo "  just vm-reset       - Reset VM to initial state"
    @echo ""
    @echo "Scientific Computing:"
    @echo "  just test-genomics  - Test genomics analysis"
    @echo "  just test-proteomics - Test proteomics analysis"
    @echo "  just test-metabolomics - Test metabolomics analysis"
    @echo "  just test-consciousness - Test consciousness processing"
    @echo ""
    @echo "Docker:"
    @echo "  just docker-build   - Build Docker container"
    @echo "  just docker-run     - Run VM in container"
    @echo "  just docker-compose - Start full development stack"
    @echo ""
    @echo "For a complete list of commands, run: just --list"

# Show system status
status:
    @echo "Bloodhound VM System Status"
    @echo "=========================="
    @echo "Rust: $(rustc --version)"
    @echo "Cargo: $(cargo --version)"
    @echo "Python: $(python --version)"
    @echo "Docker: $(docker --version)"
    @echo "Git: $(git --version)"
    @echo ""
    @echo "Project Status:"
    @echo "- Configuration: bloodhound.toml"
    @echo "- Build Status: $(if [ -f target/release/bloodhound-vm ]; then echo "✅ Built"; else echo "❌ Not built"; fi)"
    @echo "- VM Status: $(if pgrep -f bloodhound-vm > /dev/null; then echo "🟢 Running"; else echo "🔴 Stopped"; fi)"