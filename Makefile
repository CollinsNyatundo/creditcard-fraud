# Makefile for Credit Card Fraud Detection Pipeline

PYTHON = python
PIP = pip
PYTEST = pytest
FLAKE8 = flake8

.PHONY: setup run-pipeline test lint docker-build clean help up down migrate setup-metabase stack

help:
	@echo "Available targets:"
	@echo "  setup         - Upgrade pip and install all project dependencies"
	@echo "  run-pipeline  - Run feature engineering, imbalance handling, model training, tuning, and evaluation"
	@echo "  test          - Run the unit test suite with pytest"
	@echo "  lint          - Run flake8 style checks on all source and test files"
	@echo "  docker-build  - Build the production-grade Docker image"
	@echo "  clean         - Clean python cache files and test artifacts"
	@echo "  up            - Start docker compose services and wait for PG readiness"
	@echo "  down          - Stop and remove docker compose containers"
	@echo "  migrate       - Run Alembic database migrations"
	@echo "  setup-metabase- Provision Metabase connection and dashboards"
	@echo "  stack         - Full orchestration setup (up + migrate + setup-metabase)"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run-pipeline:
	$(PYTHON) data/src/feature_engineering.py
	$(PYTHON) data/src/handle_imbalance.py
	$(PYTHON) model/src/train_baseline_model.py
	$(PYTHON) model/src/hyperparameter_tuning_fixed.py
	$(PYTHON) model/src/final_model_evaluation.py

test:
	$(PYTEST)

lint:
	$(FLAKE8) data/src/ model/src/ src/ utils/ tests/

docker-build:
	docker build -t creditcard-fraud:latest .

clean:
	rm -rf .pytest_cache .coverage htmlcov .web_artifacts_builder_cache
	# Recursive cleaning of pycache files (compatible with POSIX shells)
	-find . -type d -name "__pycache__" -exec rm -rf {} +
	-find . -type f -name "*.pyc" -delete

up:
	docker compose up -d
	@echo "Waiting for postgres to be healthy..."
	@until docker compose exec postgres pg_isready -U fraud_user -d fraud_db; do sleep 1; done
	@echo "All services up."

down:
	docker compose down

migrate:
	alembic upgrade head
	@echo "Migrations complete."

setup-metabase:
	$(PYTHON) scripts/setup_metabase.py
	@echo "Metabase provisioned."

stack: up migrate setup-metabase
	@echo "Full stack ready."

