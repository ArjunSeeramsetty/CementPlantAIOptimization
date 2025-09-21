# Dockerfile - Production Ready for JK Cement Digital Twin Platform
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install tools
RUN pip install --no-cache-dir --upgrade pip==23.3.2 setuptools==69.0.3 wheel==0.42.0

# Copy requirements
COPY requirements.txt .

# Install all dependencies from consolidated requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY main.py .
COPY demo_unified_dashboard.py .

# Create necessary directories
RUN mkdir -p logs data config

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "demo_unified_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]