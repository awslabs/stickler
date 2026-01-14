#!/bin/bash
# Test dependencies in a clean virtual environment to avoid cached dependencies

set -e

echo "Creating clean test environment..."
python3 -m venv test_venv
source test_venv/bin/activate

echo "Installing package with new dependency constraints..."
pip install -e ".[dev]"

echo "Running tests..."
pytest tests/ -v

echo ""
echo "✅ Tests passed in clean environment!"
echo ""
echo "Cleaning up..."
deactivate
rm -rf test_venv

echo "Done!"
