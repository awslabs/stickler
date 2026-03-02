---
title: Installation
---

# Installation

## Quick Install

```bash
pip install stickler-eval
```

## Requirements

- Python 3.12+
- conda (recommended)

## Conda Setup

```bash
# Create a dedicated environment
conda create -n stickler python=3.12 -y
conda activate stickler

# Install from PyPI
pip install stickler-eval
```

## Development Install

If you want to contribute or run from source:

```bash
git clone https://github.com/awslabs/stickler.git
cd stickler

# Create conda environment
conda create -n stickler python=3.12 -y
conda activate stickler

# Install with dev dependencies
pip install -e ".[dev]"
```

## Optional Dependencies

Stickler's core comparators (Exact, Levenshtein, Numeric, Fuzzy) work out of the box. The AI-powered comparators require additional setup:

### SemanticComparator

Uses AWS Bedrock Titan embeddings for cosine similarity.

- AWS credentials configured (`aws configure` or environment variables)
- Access to Amazon Bedrock with Titan embedding models enabled

### BERTComparator

Uses BERTScore for contextual similarity. Runs locally -- no cloud services needed.

- `torch` and `bert-score` packages (installed automatically with stickler-eval)
- GPU recommended for performance, but CPU works

### LLMComparator

Uses AWS Bedrock via strands-agents for LLM-powered comparison.

- AWS credentials configured
- Access to Amazon Bedrock with your chosen model enabled
- `strands-agents` and `strands-agents-tools` packages

## Verify Installation

```bash
python -c "import stickler; print('Stickler installed successfully')"
```

Run the quick start example:

```bash
python examples/scripts/quick_start.py
```

Run the test suite:

```bash
pytest tests/
```

## Troubleshooting

See [Known Issues](known-issues.md) for platform-specific problems (e.g., NumPy/GCC compatibility on RHEL).
