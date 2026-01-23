# PolymarketBot - Optimized Docker Image
# Multi-stage build for minimal image size and fast startup

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 polybot

# Copy Python packages from builder
COPY --from=builder /root/.local /home/polybot/.local

# Copy application code
COPY --chown=polybot:polybot . .

# Set environment variables
ENV PATH=/home/polybot/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8765

# Create data directory for persistence
RUN mkdir -p /app/data && chown polybot:polybot /app/data

# Switch to non-root user
USER polybot

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Default command
CMD ["python", "run_ghost_mode.py"]
