FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (including libgomp1 for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to match the hardcoded paths in the scripts
WORKDIR /app/realtime_credit_card_1507

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the workspace files
COPY . .

# Default command is to run the validation script
CMD ["python", "debug_scripts/VALIDATION_SCRIPT.py"]
