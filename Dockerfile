# Production-ready Dockerfile for Cement Plant AI Digital Twin
FROM python:3.9-slim

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/
COPY .env .env

# Create required directories
RUN mkdir -p logs data artifacts

# Create health check script
COPY healthcheck.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python healthcheck.py

# Run application
CMD ["streamlit", "run", "src/cement_ai_platform/dashboard/unified_dashboard.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]