# Simple Makefile for stickler

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
