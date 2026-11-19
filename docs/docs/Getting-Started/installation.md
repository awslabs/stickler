---
title: Installation
---

# Installation

## Quick Install

```bash
pip install stickler-eval

# Using uv? Works the same way:
uv pip install stickler-eval
```

## Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

## Development Install

See [Development Setup](Contributing/development-setup.md).

## Optional Dependencies

Stickler's core comparators (Exact, Levenshtein, Numeric, Fuzzy, Date, Phone, Normalized, ANLS\* and BBox IoU) work out of the box. The comparators below need an extra:

### SemanticComparator

Uses AWS Bedrock Titan embeddings for cosine similarity.

```bash
pip install "stickler-eval[semantic]"
```

- AWS credentials configured (`aws configure` or environment variables)
- Access to Amazon Bedrock with Titan embedding models enabled

### BERTComparator

Uses BERTScore for contextual similarity. Runs locally, but downloads `roberta-large` (about 1.4 GB) from Hugging Face on first use.

```bash
pip install "stickler-eval[bert]"
```

- Network access to Hugging Face on first use
- GPU recommended for performance, but CPU works

### LLMComparator

Uses AWS Bedrock via strands-agents for LLM-powered comparison.

```bash
pip install "stickler-eval[llm]"
```

- AWS credentials configured
- Access to Amazon Bedrock with your chosen model enabled

### Other extras

| Extra | Adds |
|---|---|
| `docsplit` | document packet splitting |
| `reporting` | pandas, for passing a DataFrame as `ProcessEvaluation.matrix` (HTML reports need no extra) |
| `all` | every extra except `bert` |

```bash
pip install "stickler-eval[all]"
```

## Verify Installation

```bash
python -c "import stickler; print('Stickler installed successfully')"
```

The example script and the test suite need a clone, see [Development Setup](Contributing/development-setup.md).

## Troubleshooting

See [Known Issues](known-issues.md) for platform-specific problems (e.g., NumPy/GCC compatibility on RHEL).
