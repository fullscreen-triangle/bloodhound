# Bloodhound Virtual Machine - Consciousness-Aware Scientific Computing
# Multi-stage Docker build for optimal performance and size

# === Builder Stage ===
FROM rust:1.75-slim as rust-builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libsqlite3-dev \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./
COPY src/core/*/Cargo.toml src/core/
COPY src/integration/*/Cargo.toml src/integration/
COPY src/interfaces/*/Cargo.toml src/interfaces/

# Create dummy source files for dependency caching
RUN mkdir -p src/core/kwasa-kwasa/src \
    && echo "fn main() {}" > src/core/kwasa-kwasa/src/main.rs \
    && echo "pub fn dummy() {}" > src/core/kwasa-kwasa/src/lib.rs

RUN mkdir -p src/core/kambuzuma/src \
    && echo "fn main() {}" > src/core/kambuzuma/src/main.rs \
    && echo "pub fn dummy() {}" > src/core/kambuzuma/src/lib.rs

RUN mkdir -p src/core/buhera/src \
    && echo "fn main() {}" > src/core/buhera/src/main.rs \
    && echo "pub fn dummy() {}" > src/core/buhera/src/lib.rs

RUN mkdir -p src/core/musande/src \
    && echo "fn main() {}" > src/core/musande/src/main.rs \
    && echo "pub fn dummy() {}" > src/core/musande/src/lib.rs

RUN mkdir -p src/core/four-sided-triangle/src \
    && echo "fn main() {}" > src/core/four-sided-triangle/src/main.rs \
    && echo "pub fn dummy() {}" > src/core/four-sided-triangle/src/lib.rs

RUN mkdir -p src/integration/purpose-framework/src \
    && echo "pub fn dummy() {}" > src/integration/purpose-framework/src/lib.rs

RUN mkdir -p src/integration/combine-harvester/src \
    && echo "pub fn dummy() {}" > src/integration/combine-harvester/src/lib.rs

RUN mkdir -p src/integration/biological-quantum/src \
    && echo "pub fn dummy() {}" > src/integration/biological-quantum/src/lib.rs

RUN mkdir -p src/interfaces/cli/src \
    && echo "fn main() {}" > src/interfaces/cli/src/main.rs

RUN mkdir -p src/interfaces/web/src \
    && echo "pub fn dummy() {}" > src/interfaces/web/src/lib.rs

RUN mkdir -p src/interfaces/api/src \
    && echo "pub fn dummy() {}" > src/interfaces/api/src/lib.rs

RUN mkdir -p src && echo "fn main() {}" > src/main.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release --all-features
RUN rm -rf src

# Copy actual source code
COPY src/ src/
COPY docs/ docs/
COPY README.md LICENSE ./

# Build the actual application
RUN cargo build --release --all-features

# === Python Builder Stage ===
FROM python:3.11-slim as python-builder

# Install Python build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install dependencies
COPY python/requirements.txt python/
RUN pip install --user --no-cache-dir -r python/requirements.txt

# Copy Python source and build
COPY python/ python/
RUN cd python && python -m build

# === Final Runtime Stage ===
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    python3.11 \
    python3-pip \
    curl \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create application user for security
RUN useradd -m -u 1000 bloodhound \
    && mkdir -p /app/data /app/logs /app/config \
    && chown -R bloodhound:bloodhound /app

WORKDIR /app

# Copy built Rust binaries
COPY --from=rust-builder /app/target/release/bloodhound-vm /usr/local/bin/
COPY --from=rust-builder /app/target/release/bvm-cli /usr/local/bin/

# Copy Python packages
COPY --from=python-builder /root/.local /home/bloodhound/.local
ENV PATH="/home/bloodhound/.local/bin:$PATH"

# Copy configuration files
COPY bloodhound.toml config/
COPY docker/entrypoint.sh /usr/local/bin/
COPY docker/healthcheck.sh /usr/local/bin/

# Make scripts executable
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/healthcheck.sh

# Create directory structure
RUN mkdir -p \
    /app/data/genomics \
    /app/data/proteomics \
    /app/data/metabolomics \
    /app/logs/vm \
    /app/logs/consciousness \
    /app/config/domains \
    /app/tmp

# Set ownership
RUN chown -R bloodhound:bloodhound /app

# Switch to non-root user
USER bloodhound

# Environment variables
ENV BLOODHOUND_CONFIG="/app/config/bloodhound.toml"
ENV BLOODHOUND_DATA_DIR="/app/data"
ENV BLOODHOUND_LOG_DIR="/app/logs"
ENV RUST_LOG="info"
ENV RUST_BACKTRACE="1"

# Consciousness-aware VM configuration
ENV CONSCIOUSNESS_LEVEL="full"
ENV S_ENTROPY_NAVIGATION="true"
ENV PURPOSE_FRAMEWORK="enabled"
ENV COMBINE_HARVESTER="enabled"
ENV FEMTOSECOND_PROCESSORS="true"

# Performance optimization
ENV RAYON_NUM_THREADS="0"  # Use all available cores
ENV TOKIO_WORKER_THREADS="0"  # Use all available cores

# Expose ports
EXPOSE 8080   # Main VM interface
EXPOSE 9090   # Monitoring and metrics
EXPOSE 9091   # Consciousness monitoring
EXPOSE 3000   # Web dashboard

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Volume mounts for persistent data
VOLUME ["/app/data", "/app/logs", "/app/config"]

# Entry point
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["bloodhound-vm", "start", "--config", "/app/config/bloodhound.toml"]

# === Labels for metadata ===
LABEL maintainer="Kundai Farai Sachikonye <kundai.sachikonye@wzw.tum.de>"
LABEL description="Bloodhound Virtual Machine - First consciousness-aware computational environment"
LABEL version="0.1.0"
LABEL org.opencontainers.image.title="Bloodhound VM"
LABEL org.opencontainers.image.description="Consciousness-aware virtual machine for distributed scientific computing"
LABEL org.opencontainers.image.vendor="Independent Research"
LABEL org.opencontainers.image.authors="Kundai Farai Sachikonye"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/username/bloodhound-vm"
LABEL org.opencontainers.image.documentation="https://github.com/username/bloodhound-vm/docs"

# === Build Arguments ===
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.opencontainers.image.created=${BUILD_DATE}
LABEL org.opencontainers.image.revision=${VCS_REF}
LABEL org.opencontainers.image.version=${VERSION}

# === Special capabilities ===
LABEL bloodhound.consciousness.level="full"
LABEL bloodhound.s-entropy.navigation="enabled"
LABEL bloodhound.purpose.framework="47-domain-models"
LABEL bloodhound.combine.harvester="multi-domain-synthesis"
LABEL bloodhound.processing.type="zero-memory-femtosecond"