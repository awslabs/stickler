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
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

## Development Install

If you want to contribute or run from source:

```bash
git clone https://github.com/awslabs/stickler.git
cd stickler

# uv handles Python version, venv creation, and dependency installation
uv sync
```

Or with pip (you manage your own Python + venv):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Optional Dependencies

Stickler's core comparators (Exact, Levenshtein, Numeric, Fuzzy) work out of the box. The AI-powered comparators require additional packages, installable via extras:

### SemanticComparator

Uses AWS Bedrock Titan embeddings for cosine similarity.

```bash
pip install stickler-eval[semantic]
```

- AWS credentials configured (`aws configure` or environment variables)
- Access to Amazon Bedrock with Titan embedding models enabled

### BERTComparator

Uses BERTScore for contextual similarity. Runs locally -- no cloud services needed.

```bash
pip install stickler-eval[bert]
```

- GPU recommended for performance, but CPU works

### LLMComparator

Uses AWS Bedrock via strands-agents for LLM-powered comparison.

```bash
pip install stickler-eval[llm]
```

- AWS credentials configured
- Access to Amazon Bedrock with your chosen model enabled

### All optional dependencies

Install everything at once:

```bash
pip install stickler-eval[bert,semantic,llm]
```

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
