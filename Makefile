.PHONY: help install test test-fast test-cov lint format clean build docs docs-serve

help:  
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  
	uv sync --all-extras

install-dev: 
	uv sync --extra dev

test:  
	uv run pytest tests/ -v

test-fast:  
	uv run pytest tests/ -v --no-cov -x

test-cov:  
	uv run pytest tests/ --cov=src/cryoblob --cov-report=html --cov-report=term-missing

test-ci:  
	uv run pytest tests/ \
		--cov=src/cryoblob \
		--cov-report=xml \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-fail-under=80 \
		--junitxml=test-results.xml \
		-v

lint:  
	uv run black --check --diff src/ tests/

format:  
	uv run black src/ tests/

type-check:  
	uv run python -c "import cryoblob; print('✅ Runtime type checking passed')"

clean:  
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf test-results.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	uv build

docs:
	cd docs && uv run make html

docs-serve:
	cd docs && uv run make livehtml

docs-clean:
	cd docs && uv run make clean

dev-setup: install format lint test

check: lint type-check test-fast

ci: lint type-check test-ci build 