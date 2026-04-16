# Simple Makefile for stickler

# Define color codes for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m  # No Color

install:
	uv sync --no-dev --frozen

install-dev:
	uv sync --frozen

test:
	uv run pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Run linting checks and automatically fix issues
lint:
	uv run ruff check --fix
	uv run ruff format

.PHONY: docs docs-build docs-install

# Install docs dependencies (mkdocs, material theme, etc.)
docs-install:
	uv sync --group docs --frozen

# Start local docs site with live reload (http://127.0.0.1:8000)
docs:
	$(MAKE) -C docs docs

# Build docs site (validates links and generates static site)
docs-build:
	$(MAKE) -C docs build

# CI/CD version of lint that only checks but doesn't modify files
# Used in CI pipelines to verify code quality without making changes
lint-cicd:
	@echo "Running code quality checks..."
	@if ! uv run ruff check; then \
		echo -e "$(RED)ERROR: Ruff linting failed!$(NC)"; \
		echo -e "$(YELLOW)Please run 'make lint' locally to fix these issues.$(NC)"; \
		exit 1; \
	fi
	@if ! uv run ruff format --check; then \
		echo -e "$(RED)ERROR: Code formatting check failed!$(NC)"; \
		echo -e "$(YELLOW)Please run 'make lint' locally to fix these issues.$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(GREEN)All code quality checks passed!$(NC)"
