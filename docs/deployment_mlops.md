# Deployment & MLOps Guide: Ingestion to Verification

This guide outlines the operations, containerization, environment setup, and verification routines for running the Credit Card Fraud Detection Pipeline.

---

## 1. Containerized Deployment (Docker)

To simplify cross-environment deployment, the pipeline is fully Dockerized using a multi-stage approach.

### Dockerfile Details
We use `python:3.11-slim` as the base image. To support LightGBM training on CPU, the container installs essential build packages and the OpenMP compiler runtime (`libgomp1`):

```dockerfile
FROM python:3.11-slim

# Install system dependencies needed for compiler compilation and LightGBM CPU runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/realtime_credit_card_1507

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run E2E validation script on start
CMD ["python", "debug_scripts/end_to_end_test_optimized.py"]
```

### Orchestrating with Docker Compose
The `docker-compose.yml` mounts the host's `./logs` and `./reports` directories to allow evaluation outputs, charts, and metrics JSON to persist outside the container container lifecycle:

```yaml
services:
  pipeline:
    build: .
    volumes:
      - ./logs:/app/realtime_credit_card_1507/logs
      - ./reports:/app/realtime_credit_card_1507/reports
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
