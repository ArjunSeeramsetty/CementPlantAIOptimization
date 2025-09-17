# Production Dockerfile for Cement Plant AI Digital Twin
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CEMENT_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY demo/ ./demo/

# Create secrets directory for service account key
RUN mkdir -p /app/secrets

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user for security
RUN groupadd -r cementuser && useradd -r -g cementuser cementuser
RUN chown -R cementuser:cementuser /app
USER cementuser

# Expose port
EXPOSE 8080

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "src.cement_ai_platform.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
