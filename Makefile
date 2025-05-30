# Makefile for cryoblob development

.PHONY: help install test test-fast test-cov lint format clean build docs docs-serve

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and all dependencies
	uv sync --all-extras

install-dev:  ## Install development dependencies only
	uv sync --extra dev

test:  ## Run full test suite
	uv run pytest tests/ -v

test-fast:  ## Run tests without coverage (faster)
	uv run pytest tests/ -v --no-cov -x

test-cov:  ## Run tests with coverage report
	uv run pytest tests/ --cov=src/cryoblob --cov-report=html --cov-report=term-missing

test-ci:  ## Run tests as in CI (with coverage and XML output)
	uv run pytest tests/ \
		--cov=src/cryoblob \
		--cov-report=xml \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-fail-under=80 \
		--junitxml=test-results.xml \
		-v

lint:  ## Run linting checks
	uv run black --check --diff src/ tests/

format:  ## Format code with black
	uv run black src/ tests/

type-check:  ## Run runtime type checking
	uv run python -c "import cryoblob; print('✅ Runtime type checking passed')"

clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf test-results.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	uv build

docs:  ## Build documentation
	cd docs && uv run make html

docs-serve:  ## Serve documentation locally with auto-reload
	cd docs && uv run make livehtml

docs-clean:  ## Clean documentation build
	cd docs && uv run make clean

# Development workflow targets
dev-setup: install format lint test  ## Complete development setup

check: lint type-check test-fast  ## Quick check before commit

ci: lint type-check test-ci build  ## Full CI pipeline locally