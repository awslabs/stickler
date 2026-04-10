# KIE Annotation Tool

Streamlit web app for annotating PDF documents with structured key-value data. Creates golden datasets for evaluating KIE models with Stickler.

## Quick Start

```bash
# Install
pip install -e ".[annotator,dev]"
brew install poppler  # macOS — needed for PDF rendering

# Launch
make annotate
```

See the [User Guide](../../../docs/docs/Guides/Annotation-Tool/user-guide.md) for the full walkthrough.

## Architecture

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit entry point, layout, session management, deep links |
| `config.py` | Config dialog, session state, query param auto-apply |
| `dataset.py` | PDF discovery (excludes dotdirs), session-aware document status |
| `schema_loader.py` | Load schema from JSON file, Pydantic import, or builder output |
| `schema_builder.py` | In-app schema builder UI |
| `pdf_viewer.py` | Lazy PDF page rendering via pdf2image, bounding box overlay |
| `annotation_panel.py` | Field entry UI — scalar inputs, array tables, N/A toggles |
| `serializer.py` | `AnnotationManifest` (atomic writes), `AnnotationSession`, `AnnotationSerializer` |
| `llm_backend.py` | AWS Bedrock — extraction (Strands agent) + localization (raw Converse) |
| `models.py` | Domain models: `AnnotationState`, `FieldAnnotation`, `FieldProvenance`, `FieldLocation` |
| `styles.py` | CSS/JS injection for tooltips and custom styling |

## Storage Layout

```
<dataset_dir>/
  *.pdf
  .annotations/
    manifest.json              ← schema + all session metadata
    <session-guid>/
      <pdf-stem>.json          ← per-doc annotation (data + provenance)
```

The manifest embeds the full JSON Schema so the dataset is self-contained — zip and share.

## Tests

```bash
pytest tests/annotator/ -v
```
