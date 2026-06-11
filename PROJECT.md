# Project: Credit Card Fraud Detection Pipeline System Ideation

## Architecture
The system consists of:
- **FastAPI backend (`app/`)** that loads a LightGBM model once at startup and serves endpoints like `/predict` and `/stream`.
- **ML pipelines (`data/`, `model/`)** for chronological data splitting, SMOTE and RUS balancing, baseline training, and latency-constrained Optuna tuning.
- **Background tasks (`workers/`)** executing TreeSHAP explainability calculations (`alert_worker.py`) and dead letter queue routing (`dlq_worker.py`).
- **Client dashboard (`frontend/`)** built with Next.js, displaying transaction feeds, metrics, and SHAP waterfalls.
- **Persistence & Caching Layer**: PostgreSQL (ledger) and Redis (features, rate limits, streams).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Grounded Codebase Scanning | Scan codebase directories and verify files/parameters | None | PLANNED |
| 2 | Topic Axes Decomposition | Identify and select 3-5 orthogonal axes for system optimization | M1 | PLANNED |
| 3 | Divergent Idea Generation | Brainstorm 30+ raw candidate ideas using structured frames | M2 | PLANNED |
| 4 | Adversarial Filtering | Critique candidates and compile Rejection Summary | M3 | PLANNED |
| 5 | Ranked Optimization Artifact | Save final report to `docs/ideation/2026-06-11-system-ideation.md` | M4 | PLANNED |

## Code Layout
- `app/` — FastAPI application code, routes, services, middleware
- `data/` — Data exploration, processing, and raw dataset storage
- `model/` — Model training, tuning, and evaluation scripts
- `workers/` — Background task workers (Alerts and DLQ)
- `frontend/` — Next.js client-side application
- `utils/` — Common script utilities and validators
- `debug_scripts/` — E2E validation and test runners
- `tests/` — Unit and integration tests
- `docs/` — Architectural and design documentation
