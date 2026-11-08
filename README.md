# Real-Time Credit Card Fraud Detection Pipeline

A containerized fraud detection prototype with an offline LightGBM benchmark and a transaction scoring API.

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-red?style=for-the-badge&logo=apache)](LICENSE)

</div>

## What is this?

In real-time credit card transaction authorization systems, classification latency and false-positive rates directly dictate business profitability and customer churn. A model that misses fraud costs millions in chargebacks and recovery costs; a model that is too slow (>10ms) gets bypassed by gateway routers to avoid transaction abandonment; and a model with poor precision triggers false alarms that degrade the user experience for legitimate cardholders.

The offline benchmark uses the [ULB credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains 284,807 transactions by European cardholders in September 2013. The benchmark uses the dataset's 28 anonymized PCA features. The scoring API accepts a different input contract, so these quality and latency measurements apply to offline model evaluation.

---

## Model Performance

Both runs used the same chronological holdout of 42,722 transactions, including 52 frauds.

| Offline run | F1 | Precision | Recall |
| :--- | ---: | ---: | ---: |
| Historical model at `d501607`, rescored from a clean checkout plus the CSV | 0.8041 | 0.8667 | 0.7500 |
| Corrected focal-loss run, using stored hyperparameters and an uncalibrated threshold | 0.7723 | 0.7959 | 0.7500 |

The historical run reused validation rows during model fitting and used later transactions to fit some preprocessing statistics. The corrected run fits statistics on training data, builds transaction windows from real transactions, and uses separate chronological validation slices for early stopping and threshold selection. Both runs miss the F1 target of 0.85 and the recall gate of 0.85. The historical [benchmark report](reports/end_to_end_optimized_results.json) measures inference on precomputed features.

### Reproduce the corrected run

Use Python 3.13 with the pinned dependencies in `requirements.txt`. Download `creditcard.csv` using the [dataset instructions](data/raw/README.md). The CSV used for the audit has SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.

```sh
git show d5016073f702b75a8c27934003e32f85d0a6689e:reports/hyperparameter_optimization.json > historical-params.json
python data/src/feature_engineering.py
python data/src/advanced_feature_engineering.py
python model/src/hyperparameter_tuning.py --params-report historical-params.json
```

Run in a scratch checkout because the scripts write model files and reports. The command reuses a previous parameter search. The bootstrap intervals below describe the historical model.

### Visual Performance & Cost Curves

| Precision-Recall Curve (Generalization) | Cost vs. Decision Threshold Sweep |
| :---: | :---: |
| ![Precision-Recall Curve](reports/precision_recall_curve.png) | ![Cost vs. Threshold Curve](reports/cost_vs_threshold_curve.png) |

### Global Benchmark Standing & Statistical Rigor

Rather than presenting nominal point estimates that suffer from high evaluation variance under extreme class imbalance, our model's performance is qualified using **Bootstrap Resampling ($B=10,000$)** and a simulation-based **Statistical Power Analysis**:

- **Bootstrap F1-Score Distribution**: While our point estimate F1-score is **0.8041** (95% CI F1 of `[0.7073, 0.8833]`), we also bootstrap our primary Recall metric, yielding a **95% Confidence Interval (CI) of `[0.6250, 0.8628]`** (mean Recall of `0.7488`).
- **Statistical Insignificance ($p=0.9565$)**: A hypothesis test comparing our optimized model against the target ($0.85$ Recall) yields a p-value of `0.9565`. This confirms that the Recall difference is not statistically significant at $\alpha=0.05$ due to the small sample size of the positive class in temporal test partitions (52 fraud cases).
- **Underpowered Point Comparisons (24.8% Power)**: A simulation-based power analysis shows that a test set with **52 fraud transactions** only has a **24.8% statistical power** to detect an F1-score difference. The probability of a Type II error (failing to detect a real difference) remains high.
- **Data Scale Constraints**: To reach the standard **80% statistical power**, a test partition must contain **325 fraud transactions**. Under the natural $0.172\%$ fraud occurrence rate, this requires a test split of over 188,000 transactions, translating to a total dataset of **over 944,000 transactions** under a 60/20/20 partition.
- **Generalization limits**: This dataset covers two days of anonymized European card transactions. The small number of holdout frauds limits comparisons, and performance on other transaction streams needs separate evaluation.
- **Hypothetical p-value Scaling**: By projecting statistical significance across test partition scales (assuming constant precision and recall), we show that our F1-score comparison achieves significance ($p < 0.05$) at the target scale of **$N_{fraud} = 325$** ($p \approx 0.0250$).
- **Historical model latency**: The original report recorded **1.41 ms p95** and **0.95 ms median** for in-process scoring on precomputed features.

#### F1-Score Statistical Validation Visualizations:

| Empirical F1 Distribution (Bootstrap) | Statistical Power Curve ($N_{fraud}$ vs. Power) | Hypothetical p-value Curve ($N_{fraud}$ vs. p-value) |
| :---: | :---: | :---: |
| ![Bootstrap F1-Score Distribution](reports/bootstrap_f1_distribution.png) | ![Statistical Power Curve](reports/statistical_power_analysis.png) | ![Hypothetical p-value Curve](reports/hypothetical_p_value_curve.png) |

---

## Pipeline Architecture

The pipeline follows a modular architecture from raw transaction intake to real-time model serving and analytics:

```mermaid
graph TD
    A[Raw Transactions: creditcard.csv] --> B[Data Quality & Exploration: data_exploration.py]
    A --> C[Preprocessing & Robust Scaling: feature_engineering.py]
    C --> D[Temporal Train/Val/Test Split: chronological cutoff]
    D --> E[Class Imbalance Solver: SMOTE + Undersampling]
    E --> F[Advanced Feature Engineering: advanced_feature_engineering.py]
    F --> G[Optuna Hyperparameter Tuning: latency-constrained]
    G --> H[Model Serialization: optimized_lightgbm.pkl]
    H --> I[Probability Calibration: calibrate_probabilities.py]
    I --> J[Calibrated Model Wrapper: calibrated_model.pkl]
    J --> K[Pareto Threshold Sweep: optimize_threshold_pareto.py]
    K --> L[Optimal Threshold & DB Sync: optimal_threshold_v2.json]
    L --> M[FastAPI Predict Endpoint: app/main.py]
    M --> N[Redis WAL queue: app/services/prediction_writer.py]
    N --> O[DLQ Worker: workers/dlq_worker.py]
    O --> P[(PostgreSQL Database)]
    P --> Q[Metabase BI Dashboards]
    M --> R[Alert Worker: workers/alert_worker.py]
    R --> S[Webhook Alerts]
```

1. **Preprocessing & Resampling**: Scaled using `RobustScaler` to guard against transaction outliers. Class imbalance is resolved in training by combining SMOTE oversampling (synthetic minority generation) with random undersampling to achieve a stable 1:5 ratio of fraud to legitimate samples.
2. **Feature Engineering**: Generates 72 total features, including cyclically encoded hour dimensions, interaction terms between predictive PCA components and amount variables, rolling transaction behavior windows (mean, standard deviation, and Z-scores over 3, 5, and 10 transactions), and expanding cumulative spending statistics.
3. **Optuna Optimization**: Searches for hyperparameter combinations maximizing the validation F1-score while pruning trials that violate the strict <8ms average inference constraint.
4. **Probability Calibration**: Platt scaling (logistic regression) is applied to validation predictions, producing well-calibrated probabilities that reflect actual empirical frequencies.
5. **Threshold Optimization**: Decides the decision boundary to minimize transaction cost and enforce target Recall levels, storing config in the DB/JSON.

### Data Distributions & Fraud Patterns

The following visualizations (generated via `data/src/data_exploration.py`) show the heavy class imbalance, transaction amount patterns (log scale), and the fraud rate distributed over time bins:

![EDA Visualizations](reports/eda_visualizations.png)

---

## Project Structure

The codebase is structured to separate concerns between model exploration, pipeline engineering, real-time prediction serving, frontend interfaces, and database migrations:

```
├── .github/                  # GitHub Actions workflow specifications
│   └── workflows/
│       └── ci.yml            # CI validation workflow (flake8, pytest, Docker build)
├── alembic/                  # Database schema migrations & version scripts
│   ├── versions/
│   │   ├── 001_canonical_schema.py             # Database tables schema creation
│   │   └── 002_seed_feature_explanations.py    # Seed data for explainers
│   ├── env.py                # Configuration for SQLAlchemy async migration environment
│   └── script.py.mako        # Alembic migration template
├── app/                      # Production FastAPI Application (Serving Path)
│   ├── db/
│   │   └── engine.py         # Async SQLAlchemy database engine using asyncpg connection pool
│   ├── middleware/
│   │   └── auth.py           # Header API key validation security middleware
│   ├── routes/
│   │   ├── predict.py        # POST /predict - real-time inference (supports calibrated wrapper)
│   │   └── stream.py         # GET /stream/transactions - WebSocket event streaming
│   ├── services/
│   │   ├── config_service.py     # Dynamic DB configuration fetch with Redis TTL cache
│   │   ├── prediction_writer.py  # Redis WAL queuing and DB batch prediction writer
│   │   ├── redis_cache.py        # Redis connection & rolling history state client
│   │   ├── shap_service.py       # Selective async local TreeSHAP explanations
│   │   └── stream_publisher.py   # Publishes transaction streams via WebSockets
│   ├── config.py             # Centralized environment configurations & parser
│   ├── limiter.py            # SlowAPI rate limiter configuration
│   └── main.py               # FastAPI server entrypoint (model lifecycle management)
├── data/                     # Dataset files
│   ├── raw/
│   │   └── README.md         # Data source info
│   └── src/                  # Pipeline feature engineering & resampling modules
│       ├── advanced_feature_engineering.py
│       ├── data_exploration.py
│       ├── feature_engineering.py
│       ├── handle_imbalance.py
│       ├── lightweight_feature_engineering.py
│       └── minimal_feature_engineering.py
├── debug_scripts/            # Diagnostic and end-to-end benchmarking runner scripts
│   └── end_to_end_test_optimized.py            # SLA benchmark validation runner
├── docs/                     # System design, research, & operations docs
│   ├── plans/                # Stage-by-stage implementation plans
│   │   ├── 2026-06-08-metric-optimization-and-production-hardening.md
│   │   ├── 2026-06-08-phase1-foundation.md
│   │   ├── 2026-06-08-phase2-inference-path.md
│   │   ├── 2026-06-08-phase3-observability.md
│   │   ├── 2026-06-08-phase4-business-intelligence.md
│   │   ├── 2026-06-08-remediation-documentation.md
│   │   ├── 2026-06-08-remediation-ml-science.md
│   │   └── 2026-06-08-remediation-pipeline-infrastructure.md
│   │   └── README.md
│   ├── 2026-06-08-phase1-3-audit-report.md
│   ├── 2026-06-08-phase1-4-audit-report.md
│   ├── 2026-06-08-phase1-implementation-record.md
│   ├── 2026-06-08-phase4-walkthrough.md
│   ├── 2026-06-08-production-audit-report.md
│   ├── business_impact.md
│   ├── deployment_mlops.md
│   ├── design_decisions.md
│   ├── model_architecture.md
│   └── research_notes.md
├── frontend/                 # Interactive Next.js Dashboard Client
│   ├── app/
│   │   ├── analyst/
│   │   │   └── page.tsx      # Metabase metrics & transaction viewer
│   │   ├── api/
│   │   │   ├── backend/
│   │   │   │   └── stream-token/
│   │   │   │       └── route.ts  # Token generator for WebSocket routing
│   │   │   └── stream-token/
│   │   │       └── route.ts
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   ├── page.tsx          # Real-time transaction feed UI
│   │   ├── providers.tsx
│   │   └── replay/
│   │       └── page.tsx      # Past transaction replay console
│   ├── components/
│   │   ├── ui/
│   │   │   ├── badge.tsx
│   │   │   ├── button.tsx
│   │   │   └── card.tsx
│   │   ├── latency-chart.tsx
│   │   ├── live-feed.tsx
│   │   ├── metric-card.tsx
│   │   ├── navigation.tsx
│   │   ├── predict-form.tsx
│   │   ├── shap-waterfall.tsx
│   │   ├── threshold-tuner.tsx
│   │   └── transaction-card.tsx
│   ├── lib/
│   │   ├── api.ts
│   │   ├── auth-context.tsx
│   │   ├── store.ts
│   │   ├── utils.ts
│   │   └── ws-client.ts
│   ├── Dockerfile
│   ├── next.config.js
│   ├── postcss.config.js
│   ├── tailwind.config.ts
│   └── tsconfig.json
├── infographic/              # Pipeline overview content & design assets
│   └── pipeline-overview/
│       ├── prompts/
│       │   └── infographic.md
│       ├── analysis.md
│       ├── source.md
│       └── structured-content.md
├── logs/                     # Local execution logs folder
│   └── .gitkeep
├── metabase/                 # Metabase dashboards configuration mounts
│   └── dashboards/
│       ├── 01_operations.json
│       ├── 02_financial_impact.json
│       └── 03_model_health.json
├── model/                    # Model training, calibration, and sweep logic
│   └── src/
│       ├── bootstrap_analysis.py       # Bootstrap power & CI sweep script
│       ├── feature_coverage_check.py   # Asserts feature coverage in naming maps
│       ├── final_model_evaluation.py   # Evaluates calibrated model on test split
│       ├── hyperparameter_tuning.py    # Optuna HP tuning script
│       ├── proxy_clustering.py         # KMeans clusterer for customer segments
│       └── train_baseline_model.py     # Baseline LightGBM model trainer
├── models/                   # Serialized model, preprocessor, and threshold configs
│   ├── balancing_preprocessor.pkl
│   ├── baseline_lightgbm.pkl
│   ├── baseline_lightgbm.txt
│   ├── feature_list.json
│   ├── feature_names.json
│   ├── features_scaled.txt
│   ├── optimal_threshold.json
│   ├── optimal_threshold_v2.json
│   ├── optimized_lightgbm.pkl
│   ├── optimized_lightgbm.txt
│   └── preprocessor.pkl
├── reports/                  # Plot charts and evaluation output files
│   ├── bootstrap_f1_distribution.png
│   ├── bootstrap_statistical_results.json
│   ├── component_inventory.json
│   ├── cost_vs_threshold_curve.png
│   ├── dataset_validation_report.json
│   ├── diagnostic_test_results.json
│   ├── eda_report.html
│   ├── eda_visualizations.png
│   ├── end_to_end_optimized_results.json
│   ├── hyperparameter_optimization.json
│   ├── hypothetical_p_value_analysis.json
│   ├── hypothetical_p_value_curve.png
│   ├── import_validation_report.json
│   ├── precision_recall_curve.png
│   ├── statistical_power_analysis.png
│   ├── statistical_power_report.json
│   └── threshold_pareto_results.json
├── src/                      # Shared core logic utilities
│   ├── basic_feature_engineering.py
│   └── feature_engineering.py
├── tests/                    # Unit and integration test suites
│   ├── integration/
│   │   ├── test_artifact_sla.py
│   │   └── test_full_stack_smoke.py
│   ├── unit/
│   │   ├── test_alert_worker.py
│   │   ├── test_app_startup.py
│   │   ├── test_auth_middleware.py
│   │   ├── test_config_service.py
│   │   ├── test_dlq_worker.py
│   │   ├── test_feature_coverage_check.py
│   │   ├── test_feature_engineering.py
│   │   ├── test_handle_imbalance.py
│   │   ├── test_model_pipeline.py
│   │   ├── test_predict_endpoint.py
│   │   ├── test_prediction_writer.py
│   │   ├── test_redis_cache.py
│   │   ├── test_setup_metabase.py
│   │   ├── test_shap_service.py
│   │   ├── test_stream_endpoint.py
│   │   └── test_stream_publisher.py
│   └── conftest.py
├── utils/                    # Drift monitoring and validation scripts
│   ├── check_files.py
│   ├── dataset_validation_summary.py
│   ├── drift_monitor.py      # KS-test & Wasserstein Distance Amount drift monitor
│   ├── environment_detection.py
│   └── validate_results.py
├── workers/                  # Background event workers
│   ├── alert_worker.py       # Redis pub/sub webhook dispatcher
│   └── dlq_worker.py         # Batch prediction database logger
├── .dockerignore
├── .env.example
├── .flake8
├── .gitignore
├── alembic.ini               # Alembic configuration
├── docker-compose.frontend.yml # Frontend dashboard compose
├── docker-compose.yml        # Core services Docker compose file
├── Dockerfile                # Core FastAPI serving container config
├── LICENSE                   # Apache 2.0 LICENSE
├── Makefile                  # Build and test CLI command shortcuts
├── openhop-pipeline.yaml     # Hop pipeline config
├── pyrightconfig.json        # Pyright configuration for import resolving
├── pytest.ini                # Pytest configurations
└── requirements.txt          # Python dependencies manifest
```

---

## Documentation Index

| Resource | Description | Target Audience |
| :--- | :--- | :--- |
| [Business Impact Report](docs/business_impact.md) | ROI, false positive/negative trade-offs, and transaction cost metrics. | **Business & Product Stakeholders** |
| [Model Architecture Guide](docs/model_architecture.md) | SMOTE balancing, engineered feature definitions, and LightGBM tuning. | **Data Scientists & ML Engineers** |
| [Deployment & MLOps Guide](docs/deployment_mlops.md) | Docker containers, OpenMP setup, host environment fixes, and E2E validation. | **DevOps & MLOps Engineers** |
| [Research Notes](docs/research_notes.md) | Deep-dive research report covering local setups, unicode fixes, and unicode charts. | **Core Developers** |
| [E2E Evaluation JSON](reports/end_to_end_optimized_results.json) | Raw test run metrics and percentile latencies. | **Infrastructure Engineers** |
| [EDA Visualizations](reports/eda_visualizations.png) | Log-scale amount distributions and fraud rate binned over time. | **Technical Reviewers** |

---

## Quick Start

### Prerequisites
- Docker & Docker Compose (v2.0+)
- Python 3.11.x (if running locally)
- Node.js 18+ & npm/pnpm (for local frontend work)

---

### Option A: Running with Docker (Recommended)

This approach runs the entire validation suite in a sandboxed, zero-dependency environment.

1. **Verify Docker and Docker Compose are installed**:
   ```bash
   docker --version
   docker compose version
   ```

2. **Copy environment configuration**:
   ```bash
   cp .env.example .env
   ```

3. **Build and spin up the pipeline verification service**:
   ```bash
   docker compose up --build
   ```
   This will spin up a `python:3.11-slim` container, compile necessary system libraries (e.g. `libgomp1`), and execute the validation checks. Model training logs and reports will be mapped to the `./logs` and `./reports` directories on your host.

---

### Option B: Local Host Setup (.venv)

Follow these instructions to run the training, tuning, or evaluation scripts directly on your local machine.

1. **Clone and navigate to the project directory**:
   ```bash
   git clone https://github.com/ai-ml/creditcard-fraud.git
   cd creditcard-fraud
   ```

2. **Create and activate a virtual environment**:
   - **Windows PowerShell**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Linux/macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install the dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify the environment and run the End-to-End test suite**:
   ```bash
   python debug_scripts/end_to_end_test_optimized.py
   ```

---

## Environment Variables Reference

The project runs successfully with default local paths. You may override configurations using the following environment variables:

### Core Serving & ML Pipeline Variables (`.env`)

| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `DATABASE_URL` | PostgreSQL connection string using asyncpg | `postgresql+asyncpg://fraud_user:dev_password@localhost:5432/fraud_db` |
| `REDIS_URL` | Redis URL for caching transaction histories and WAL queuing | `redis://:dev_password@localhost:6379/0` |
| `POSTGRES_PASSWORD` | PostgreSQL password for containerized database | `dev_password` |
| `REDIS_PASSWORD` | Redis password for containerized instance | `dev_password` |
| `MLFLOW_TRACKING_URI` | HTTP URI for the MLflow tracking server | `http://localhost:5001` |
| `MODEL_URI` | MLflow model artifact identifier | `models:/fraud-lgbm/Production` |
| `API_KEY_HEADER` | Request header name used to pass authorization key | `X-API-Key` |
| `STREAM_STALE_THRESHOLD_SECONDS` | Limit for stale websocket streams (seconds) | `5` |
| `STREAM_DEFAULT_REPLAY_SECONDS` | Default timeframe back for transactions replay (seconds) | `60` |
| `STREAM_MAX_REPLAY_SECONDS` | Maximum timeframe back for transactions replay (seconds) | `600` |
| `SHAP_TRIGGER_THRESHOLD` | Threshold above which async TreeSHAP explanations are computed | `0.50` (overridden by DB sync) |
| `ALERT_THRESHOLD` | Fraud probability threshold triggering alert webhooks | `0.90` |
| `ALERT_WEBHOOK_URL` | Webhook URL to dispatch Slack/Discord notifications | `""` |
| `ALERT_ENABLED` | Enable webhook calls for high-confidence alerts | `false` |
| `PIPELINE_LOG_LEVEL` | Level of logging granularity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `OUTPUT_DIR` | Target folder for serialized models and JSON reports | `./models` |
| `DATA_DIR` | Ingestion directory for raw and processed datasets | `./data` |

### Frontend Application Variables (`frontend/.env.example`)

| Variable | Description | Example Value |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_API_URL` | Endpoint of the FastAPI backend serving prediction path | `http://localhost:8000` |
| `NEXT_PUBLIC_API_KEY` | Backend authorization token for X-API-Key verification | `development_api_key_token` |

---

## Detailed Scripts Reference

| Directory | Script | Purpose |
| :--- | :--- | :--- |
| `data/src/` | `data_exploration.py` | Runs raw dataset checks, missing value checks, and generates EDA reports. |
| `data/src/` | `feature_engineering.py` | Handles scaling using `RobustScaler` and temporal splits. |
| `data/src/` | `handle_imbalance.py` | Applies SMOTE + RandomUnderSampler to balance the training split. |
| `data/src/` | `advanced_feature_engineering.py` | Builds z-scores, cyclic encodes hours, and creates PCA interaction features. |
| `model/src/` | `train_baseline_model.py` | Trains baseline LightGBM model and saves optimal thresholds. |
| `model/src/` | `hyperparameter_tuning.py` | Performs Optuna hyperparameter optimization with inference latency constraints. |
| `model/src/` | `calibrate_probabilities.py` | Calibrates model predictions using Platt Scaling (Logistic Regression). |
| `model/src/` | `optimize_threshold_pareto.py` | Sweeps decision thresholds to minimize total cost and satisfy Recall >= 0.85. |
| `model/src/` | `final_model_evaluation.py` | Performs comparative evaluation of models on the test set & runs latency benchmarks. |
| `model/src/` | `bootstrap_analysis.py` | Runs bootstrap resamples to qualify Recall confidence intervals and significance. |
| `workers/` | `alert_worker.py` | Listens to Redis high-fraud alerts & executes webhook dispatches. |
| `workers/` | `dlq_worker.py` | Reads prediction WAL queues & batches writes to PostgreSQL database. |
| `debug_scripts/` | `end_to_end_test_optimized.py` | Benchmarks 1000 single transaction inferences and prints model metrics. |
| `utils/` | `dataset_validation_summary.py` | Performs schema mapping checks on local files. |
| `utils/` | `drift_monitor.py` | Tracks AUPRC trends and Amount distribution changes using Wasserstein/KS metrics. |
| `scripts/` | `setup_metabase.py` | Automates Metabase configurations and dashboards bootstrap. |

---

## Testing & Code Quality

Our testing suite uses `pytest` and enforces coverage minimums to maintain production code quality:

```bash
# Run the complete test suite
pytest

# Run tests with coverage reporting
pytest --cov=app --cov=workers --cov=data --cov=model --cov-report=term-missing

# Run a specific unit test file
pytest tests/unit/test_predict_endpoint.py
```

Code linting and format compliance are validated with `flake8`:
```bash
# Check formatting
flake8 app workers data model tests
```

---

## Deployment & Production MLOps

In production environments, the pipeline components run in isolated containers orchestrated via Docker Compose.

```bash
# Spin up production database, Redis queue, workers, FastAPI serving path, and Next.js frontend
docker compose -f docker-compose.yml -f docker-compose.frontend.yml up -d
```

### Batch Data Sync & WAL Architecture
To guarantee low latency on the prediction path while maintaining data persistence, the system utilizes a **Write-Ahead Logging (WAL)** system on Redis:
1. The **FastAPI predict endpoint** receives a transaction, executes the inference with the calibrated LightGBM model, and instantly pushes the predicted values to a Redis list.
2. A background **DLQ worker** (`workers/dlq_worker.py`) consumes records from the list in batches and writes them to the PostgreSQL database async using bulk insertion.
3. This architecture decouples transactional database writes from the API response loop.

---

## Troubleshooting

### Database Connection Issues
- **Error:** `could not connect to server: Connection refused`
- **Solution:** Verify the database container is running (`docker ps`) and that the `DATABASE_URL` matches your local port forwarding settings. Ensure database tables are created using:
  ```bash
  alembic upgrade head
  ```

### Redis Connection / WAL Queue Overflows
- **Error:** Connection timeouts to Redis or high memory usage.
- **Solution:** Check Redis health with `ping` using `redis-cli`. Check queue backlog lengths in Redis by querying `LLEN WAL_QUEUE`. Restart workers using `docker compose restart dlq_worker`.

---

## Ethical Disclaimer

> [!WARNING]
> **Automated Decision-Making and Financial Access Risks**
>
> 1. **No Fully Automated Deployment**: This system should **never** be deployed as a fully automated blocker of credit access or account suspension without human-in-the-loop oversight. False positives in fraud detection can lead to unfair exclusion of legitimate consumers from essential financial services.
> 2. **Evaluation Balance & Bias**: The dataset used here contains anonymized transaction data. In a live system, models may exhibit differential error rates across demographic cohorts or geographic regions if not regularly audited for fairness.
> 3. **Transparency & Redress**: Any decision to reject or delay a transaction should be logged with clear explanation codes, and cardholders must be provided a simple, fast redress mechanism to appeal automated decisions.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---
