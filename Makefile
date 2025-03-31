# Makefile for Churn Model Project (Portable Version for Linux/macOS/Git Bash)

# Define directories using forward slashes
PROCESSED_DATA_DIR = data/processed
MODELS_DIR = models
REPORTS_DIR = reports
LOGS_DIR = logs
MLFLOW_DIR = mlruns
PYTHON_CACHE_PATTERN = -name '__pycache__' -o -name '*.py[co]'

# Phony targets don't represent files (commands, not files)
.PHONY: all clean clean-data clean-models clean-reports clean-mlflow clean-logs clean-cache clean-venv \
        setup run-preprocess run-tune run-train run-evaluate run-api run-all \
        run-train-rf run-train-svc run-train-xgb run-train-lr \
        run-evaluate-rf run-evaluate-svc run-evaluate-xgb run-evaluate-lr \
        lint format test

# Default target runs the main workflow up to evaluation
all: run-all

# --- Cleaning Targets ---
# Removes generated artifacts
clean: clean-data clean-models clean-reports clean-mlflow clean-logs clean-cache
	@echo "Project artifacts cleaned (processed data, models, reports, logs, mlflow runs, cache)."

clean-data:
	@echo "Cleaning processed data..."
	rm -rf $(PROCESSED_DATA_DIR)/*
	@# Use 'mkdir -p' to ensure the directory exists, ignoring errors if it already does
	@mkdir -p $(PROCESSED_DATA_DIR)

clean-models:
	@echo "Cleaning saved models and tuning results..."
	rm -rf $(MODELS_DIR)/*
	@mkdir -p $(MODELS_DIR)

clean-reports:
	@echo "Cleaning reports and figures..."
	rm -rf $(REPORTS_DIR)/*
	@mkdir -p $(REPORTS_DIR)/figures/shap

clean-mlflow:
	@echo "Cleaning MLflow runs..."
	rm -rf $(MLFLOW_DIR)
	rm -f .mlflow # Remove cache file if it exists

clean-logs:
	@echo "Cleaning log files..."
	rm -rf $(LOGS_DIR)/*
	@mkdir -p $(LOGS_DIR)

clean-cache:
	@echo "Cleaning Python cache files..."
	find . \( $(PYTHON_CACHE_PATTERN) \) -exec rm -rf {} +

clean-venv:
	@echo "Removing Poetry virtual environment (.venv)..."
	rm -rf .venv
	@echo "Run 'poetry install --with dev' to recreate."

# --- Setup Target ---
setup:
	@echo "Installing dependencies via Poetry..."
	poetry install --with dev

# --- Main Workflow Targets ---
# These depend on each other to ensure correct order
run-preprocess:
	@echo "Running data preprocessing..."
	poetry run python -m src.churn_model.processing --run

run-tune: run-preprocess
	@echo "Running hyperparameter tuning..."
	poetry run python -m src.churn_model.tune

run-train: run-tune
	@echo "Running final model training (overall best identified by tuning)..."
	poetry run python -m src.churn_model.train

run-evaluate: run-train
	@echo "Running final model evaluation (overall best identified by tuning)..."
	poetry run python -m src.churn_model.evaluate

run-all: run-evaluate
	@echo "Main workflow (preprocess, tune, train, evaluate) complete."

# --- Targets for Specific Models ---
# Assumes run-tune has completed successfully before running these

# Training specific models
run-train-rf: run-tune
	@echo "Running final model training (RandomForest)..."
	poetry run python -m src.churn_model.train --model-name RandomForest

run-train-svc: run-tune
	@echo "Running final model training (SVC)..."
	poetry run python -m src.churn_model.train --model-name SVC

run-train-xgb: run-tune
	@echo "Running final model training (XGBoost)..."
	poetry run python -m src.churn_model.train --model-name XGBoost

run-train-lr: run-tune
	@echo "Running final model training (LogisticRegression)..."
	poetry run python -m src.churn_model.train --model-name LogisticRegression

# Evaluating specific models (assumes they have been trained)
run-evaluate-rf: # Doesn't strictly depend on run-train-rf here, assumes file exists
	@echo "Running final model evaluation (RandomForest)..."
	poetry run python -m src.churn_model.evaluate --model-path $(MODELS_DIR)/final_churn_model_RandomForest.joblib

run-evaluate-svc:
	@echo "Running final model evaluation (SVC)..."
	poetry run python -m src.churn_model.evaluate --model-path $(MODELS_DIR)/final_churn_model_SVC.joblib

run-evaluate-xgb:
	@echo "Running final model evaluation (XGBoost)..."
	poetry run python -m src.churn_model.evaluate --model-path $(MODELS_DIR)/final_churn_model_XGBoost.joblib

run-evaluate-lr:
	@echo "Running final model evaluation (LogisticRegression)..."
	poetry run python -m src.churn_model.evaluate --model-path $(MODELS_DIR)/final_churn_model_LogisticRegression.joblib


# --- API Target ---
run-api: # Assumes at least one model (preferably the best) has been trained
	@echo "Starting API server (Press Ctrl+C to stop)..."
	poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# --- Quality Assurance Targets ---
lint:
	@echo "Running linters (flake8)..."
	poetry run flake8 src tests

format:
	@echo "Running formatters (black, isort)..."
	poetry run black src tests
	poetry run isort src tests

test:
	@echo "Running tests (pytest)..."
	poetry run pytest tests/ # Assumes tests are in tests/ directory
