# Production Dockerfile for ProteinSynergyDock FastAPI Backend
FROM python:3.11-slim

# Set working directory & environment variables
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install system dependencies needed for RDKit & C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements-api.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application source files
COPY api.py redis_cache.py core.py core_fhir.py admet_utils.py audit_log.py model_bridge.py /app/
COPY precomputed_scores.json /app/

# Expose HTTP port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server via uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
