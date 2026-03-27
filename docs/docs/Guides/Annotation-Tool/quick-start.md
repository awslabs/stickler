# Quick Start

## Prerequisites

- Python 3.12+
- `poppler-utils` for PDF rendering: `brew install poppler` (macOS)
- AWS credentials for LLM features (optional)

## Install

```bash
pip install -e ".[annotator,dev]"
```

## Launch

```bash
streamlit run src/stickler/annotator/app.py --server.port 8501
```

Or use the Makefile:

```bash
make annotate
```

## First Annotation

1. Enter a dataset directory path (folder containing PDFs)
2. Choose a schema source — JSON Schema file, Pydantic import, or the built-in Schema Builder
3. Select an operating mode:
    - **Zero Start** — manual annotation from scratch
    - **LLM Inference** — Bedrock pre-fills all fields for batch review
    - **HITL** — field-by-field review of LLM predictions
4. Navigate documents with Prev/Next or the document picker
5. Fill in field values or mark fields as N/A
6. Annotations auto-save on every change

## Deep Links

Share a URL that skips configuration:

```
# New session
http://localhost:8501/?dataset=./files&schema=./files/schema.json&mode=zero_start

# Resume session (schema loaded from manifest)
http://localhost:8501/?dataset=./files&session=<guid>&doc=invoice.pdf
```

The `doc` parameter preserves the selected document across page refreshes.

## AWS Credentials

For LLM auto-annotate and localization, set credentials in `.env`:

```bash
AWS_PROFILE=your-profile
```

Or export `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` directly.
