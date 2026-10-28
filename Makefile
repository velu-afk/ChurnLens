.PHONY: help install env download validate train evaluate run-synthetic test lint format clean

PYTHON ?= python3
PYTEST ?= pytest
RUFF ?= ruff

help:
	@echo "ChurnLens Makefile Commands (Linux/macOS/Git Bash):"
	@echo "  make install        Install churnlens in editable mode with dev dependencies"
	@echo "  make env            Inspect Python, OS, and library versions"
	@echo "  make download       Download IBM Telco Customer Churn CSV to data/raw/"
	@echo "  make validate       Validate dataset schema, missing values, and class balance"
	@echo "  make train          Run full training and evaluation on real dataset"
	@echo "  make run-synthetic  Run full pipeline on synthetic fixture (offline verification)"
	@echo "  make test           Run automated pytest unit test suite"
	@echo "  make lint           Run Ruff linter checks"
	@echo "  make format         Run Ruff format check"
	@echo "  make clean          Remove caches and temporary build files"

install:
	$(PYTHON) -m pip install -e ".[dev]"

env:
	$(PYTHON) -m churnlens inspect-env

download:
	$(PYTHON) scripts/download_data.py

validate:
	$(PYTHON) -m churnlens validate-data --config configs/default.yaml

train:
	$(PYTHON) scripts/run_experiment.py --config configs/default.yaml

run-synthetic:
	$(PYTHON) scripts/run_experiment.py --config configs/default.yaml --use-synthetic --output-dir results/synthetic_verification

test:
	$(PYTEST) tests -v

lint:
	$(RUFF) check src tests scripts

format:
	$(RUFF) format --check src tests scripts

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build dist src/*.egg-info
