FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (including libgomp1 for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the rest of the workspace files
COPY . .

# Change ownership of /app to appuser
RUN chown -R appuser:appgroup /app

# Add healthcheck to verify core ML libraries are functional
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import lightgbm; import pandas; import sklearn" || exit 1

# Switch to non-root user
USER appuser

# Default command is to run the validation script
CMD ["python", "debug_scripts/end_to_end_test_optimized.py"]
