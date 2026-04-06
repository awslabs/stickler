# Quick Start

Point the tool at a folder of PDFs, pick a schema, and start annotating. LLM features are optional — you can annotate entirely by hand.

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
3. Navigate documents with Prev/Next or the document picker
4. Fill in field values or mark fields as N/A
5. Use **Auto-annotate** to pre-fill fields via LLM (optional, requires AWS credentials)
6. Use **Locate** to find where field values appear in the PDF (optional)
7. Annotations auto-save on every change

## Deep Links

Share a URL that skips configuration:

```
# New session
http://localhost:8501/?dataset=./files&schema=./files/schema.json

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
