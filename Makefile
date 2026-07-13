# OpenJSpace developer tasks. Uses the active Python environment.
PYTHON ?= python

.PHONY: help install install-dev web lint format typecheck test test-integration \
        check doctor serve clean

help:
	@echo "OpenJSpace make targets:"
	@echo "  install           install the package"
	@echo "  install-dev       install with dev + datasets extras"
	@echo "  web               build the React web UI into web/dist"
	@echo "  lint              ruff lint (src, tests, examples)"
	@echo "  format            ruff format"
	@echo "  typecheck         mypy"
	@echo "  test              unit tests (skips integration/real-weight tests)"
	@echo "  test-integration  integration tests (downloads model weights)"
	@echo "  check             lint + typecheck + test"
	@echo "  doctor            openjspace doctor"
	@echo "  serve             start the local web UI"
	@echo "  clean             remove caches and build artifacts"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,datasets]"

web:
	cd web && npm install && npm run build

lint:
	ruff check src tests examples

format:
	ruff format src tests examples

typecheck:
	mypy

test:
	pytest

test-integration:
	pytest -m integration

check: lint typecheck test

doctor:
	openjspace doctor

serve:
	openjspace serve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ *.egg-info build dist
