# Deployment & MLOps Guide: Ingestion to Verification

This guide outlines the operations, containerization, environment setup, and verification routines for running the Credit Card Fraud Detection Pipeline.

---

## 1. Containerized Deployment (Docker)

To simplify cross-environment deployment, the pipeline is fully Dockerized using a multi-stage approach.

### Dockerfile Details
We use `python:3.11-slim` as the base image. To support LightGBM training on CPU, the container installs essential build packages and the OpenMP compiler runtime (`libgomp1`):

```dockerfile
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
RUN pip install --no-cache-dir -r requirements.txt

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
CMD ["python", "debug_scripts/VALIDATION_SCRIPT.py"]
```

### Orchestrating with Docker Compose
The `docker-compose.yml` mounts the host's workspace directory to `/app` to allow evaluation outputs, charts, and metrics JSON to persist outside the container lifecycle:

```yaml
version: '3.8'

services:
  pipeline:
    build: .
    image: creditcard-fraud-pipeline:latest
    container_name: creditcard_fraud_container
    volumes:
      - .:/app
    environment:
      - PYTHONUNBUFFERED=1
    command: python debug_scripts/VALIDATION_SCRIPT.py
```

---

## 2. Portability & Host Adjustments

During the local host migration, the following adjustments were made to ensure absolute portability:

### A. Relative File Paths
- **Issue**: Source code contained hardcoded absolute directory strings referring to `/app/realtime_credit_card_1507/...`.
- **Resolution**: Refactored **177 paths across 27 files** to use relative paths (`./`). All scripts now execute cleanly on both local Windows/macOS hosts and inside Linux containers without modifications.

### B. Windows Unicode CLI Rendering
- **Issue**: Running validation scripts on Windows hosts threw `UnicodeEncodeError` when writing terminal progress indicators like `✓` and `❌` to the default Windows console (`cp1252` encoding).
- **Resolution**: Implemented fallback progress symbols: `[OK]`, `[FAIL]`, and `[SUCCESS]` to maintain shell rendering compatibility across all platforms.

---

## 3. Operations & Validation Workflows

### Environment Variables
Configure model operational flags using these variables:

| Variable | Scope | Description | Default Value |
| :--- | :--- | :--- | :--- |
| `PIPELINE_LOG_LEVEL` | Logging | Granularity of logs (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `OUTPUT_DIR` | Storage | Directory where serialized models and json files are stored | `./models` |
| `DATA_DIR` | Storage | Directory containing raw csv transactions and splits | `./data` |

### Pipeline Validation Test Harness
We run E2E testing using `debug_scripts/end_to_end_test_optimized.py`.
This script performs a 3-step check:
1. **Model Validation**: Confirms model weights can be deserialized (`optimized_lightgbm.pkl`) and feature alignment matches the expected signature (72 features).
2. **Latency Benchmarking**: Performs 1,000 single transaction inference runs, recording latency in a high-resolution timer to output mean, median, 95th, and 99th percentile metrics.
3. **Accuracy Verification**: Calculates the model's F1-score, Precision, Recall, and ROC AUC on the test dataset and outputs a JSON summary to `./reports/end_to_end_optimized_results.json`.

---

## 4. Running the Production Stack

### One-Command Start

```bash
cp .env.example .env          # Fill in real values
make stack                    # Starts all services, migrates DB, provisions Metabase
```

### Services & Ports

| Service   | URL                      | Purpose                      |
|-----------|--------------------------|------------------------------|
| API       | http://localhost:8000    | `/predict`, `/stream`, `/health` |
| MLflow    | http://localhost:5001    | Experiment tracking, model registry |
| Metabase  | http://localhost:3000    | Business dashboards          |
| PostgreSQL| localhost:5432           | Predictions ledger           |
| Redis     | localhost:6379           | Feature cache, WAL, Pub/Sub  |

### Running the Smoke Test

```bash
# In Windows PowerShell:
$env:RUN_INTEGRATION_TESTS="true"
.venv\Scripts\pytest tests/integration/test_full_stack_smoke.py -v

# In Bash / Linux:
RUN_INTEGRATION_TESTS=true .venv/bin/pytest tests/integration/test_full_stack_smoke.py -v
```

### Adding a New API Key

To manually insert a new API key for `/predict` authorization, hash the raw API key with SHA-256 and insert it into the database:

```sql
INSERT INTO api_keys (key_hash, label, is_active)
VALUES (encode(sha256('your-raw-key'::bytea), 'hex'), 'my-key-label', true);
```

