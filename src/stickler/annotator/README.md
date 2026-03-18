# KIE Annotation Tool

A Streamlit-based web app for annotating PDF documents with structured key-value data. The primary use case is creating **golden datasets** for evaluating KIE (Key Information Extraction) models using stickler.

## Quick Start

```bash
# Install with annotator dependencies
pip install -e ".[annotator,dev]"

# Launch
streamlit run src/stickler/annotator/app.py --server.port 8501

# Or one-click via deep link (see below)
```

Requires `poppler-utils` for PDF rendering: `brew install poppler` (macOS).

---

## Core Concepts

### Annotation Session

Every annotation run is a **session** — a UUID-identified subdirectory under `.annotations/` that belongs to one annotator working against one schema. Sessions are tracked in a top-level `manifest.json`.

```
<dataset_dir>/
  .annotations/
    manifest.json              ← schema embedded + all session metadata
    <session-guid>/
      <pdf-stem>.json          ← per-doc annotation (data + provenance)
```

The manifest embeds the full JSON Schema so the dataset is self-contained — zip it up and share it, no external schema file needed.

### Manifest Structure

```json
{
  "schema": { ...full JSON Schema... },
  "schema_hash": "dff112...",
  "sessions": {
    "42fac85a-...": {
      "annotator": "sromo",
      "created_at": "2026-03-18T...",
      "updated_at": "2026-03-18T...",
      "doc_count": 370,
      "completed_count": 7
    }
  }
}
```

### Per-Doc Annotation File

```json
{
  "data": {
    "station_name": "COX MEDIA - WEST",
    "invoice_id": "1277139",
    "line_items": [
      {"air_date": "02/29/2016", "program": "Evening News", "gross_rate": "$2,249.00"}
    ]
  },
  "metadata": {
    "schema_hash": "dff112...",
    "created_at": "2026-03-18T...",
    "updated_at": "2026-03-18T...",
    "fields": {
      "station_name": {"source": "human", "checked": false},
      "invoice_id": {"source": "llm", "checked": true}
    }
  }
}
```

`data` is directly loadable: `Model(**annotation["data"])`. `metadata.fields` tracks provenance (human vs LLM, reviewed or not).

---

## Configuration

Click the **⚙️** gear icon in the top-right to configure:

| Setting | Description |
|---|---|
| Dataset directory | Path to folder containing PDFs |
| Schema source | JSON Schema file, Pydantic import path, or Schema Builder |
| Operating mode | Zero Start, LLM Inference, or HITL |

Once applied, the gear shows a summary: `📁 dataset · 📋 schema · ⚡ mode`.

---

## Deep Links (One-Click Start)

Share a URL that skips configuration entirely.

**New session** (schema file required):
```
http://localhost:8501/?dataset=./files&schema=./files/invoice_schema.json&mode=zero_start
```

**Resume existing session** (schema loaded from manifest):
```
http://localhost:8501/?dataset=./files&session=42fac85a-e74c-4760-8cca-1e177bbf4886
```

The active deep link is always shown in the header. Copy it to share your session with another annotator.

---

## Annotation Modes

### Zero Start
Manual annotation from scratch. All fields shown with text inputs and N/A checkboxes. Progress bar tracks completion. Auto-saves on every field change (💾 toast confirms).

### LLM Inference
Send the PDF to AWS Bedrock (Claude) to pre-fill all fields. Review predictions in batch — Accept All, Reject All, or edit individual fields. LLM values shown with 🤖 prefix.

### HITL (Human-in-the-Loop)
Same as LLM Inference but field-by-field review. Each prediction shown one at a time with Accept / Reject / Edit controls. Review progress tracked separately from annotation progress.

---

## Schema Sources

### JSON Schema File
Point to a `.json` file. The schema is embedded in the manifest on first use — subsequent sessions don't need the file.

```json
{
  "type": "object",
  "properties": {
    "invoice_id": {"type": "string"},
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "air_date": {"type": "string"},
          "gross_rate": {"type": "string"}
        }
      }
    }
  }
}
```

Array fields render as inline tables — one row per item, add/remove rows with ＋/✕ buttons.

### Schema Builder
Build a schema in the UI without writing JSON. Add fields with name + type (string, number, integer, boolean, object, array). Export to file when done.

### Pydantic Import
Provide a dotted import path to a `StructuredModel` subclass: `mypackage.models.InvoiceModel`.

---

## Creating Golden Data for KIE Evaluation

The annotation tool is designed to produce ground truth data for stickler evaluation. Here's the full workflow:

### 1. Annotate

Run the tool against your PDF dataset, annotate each document field by field. Use LLM Inference or HITL to speed up annotation — the provenance metadata tracks which values were human-verified.

### 2. Load Annotations

```python
import json
from pathlib import Path
from stickler.annotator.serializer import AnnotationManifest, AnnotationSession

# Load a session
manifest = AnnotationManifest(Path("./files"))
session = manifest.get_session("42fac85a-...")

# Load one document's annotation
annotation_path = session.annotation_path_for(Path("./files/invoice.pdf"))
annotation = json.loads(annotation_path.read_text())

# Construct a StructuredModel instance directly from the data section
from stickler.annotator.schema_loader import SchemaLoader
_, ModelClass = SchemaLoader.from_builder_schema(session.schema)
ground_truth = ModelClass(**annotation["data"])
```

### 3. Evaluate Against Model Predictions

```python
# Your KIE model's output
prediction_json = my_kie_model.extract(pdf_path)
prediction = ModelClass(**prediction_json)

# Compare using stickler
result = ground_truth.compare_with(prediction)
print(f"Overall score: {result['overall_score']:.3f}")
print(f"Field scores: {result['field_scores']}")
```

### 4. Bulk Evaluation Across a Session

```python
from pathlib import Path
import json

manifest = AnnotationManifest(Path("./files"))
session = manifest.get_session(session_id)
_, ModelClass = SchemaLoader.from_builder_schema(session.schema)

scores = []
for pdf_path in Path("./files").glob("*.pdf"):
    ann_path = session.annotation_path_for(pdf_path)
    if not ann_path.exists():
        continue
    annotation = json.loads(ann_path.read_text())
    ground_truth = ModelClass(**annotation["data"])
    prediction = ModelClass(**my_model.extract(pdf_path))
    result = ground_truth.compare_with(prediction)
    scores.append(result["overall_score"])

print(f"Mean score across {len(scores)} docs: {sum(scores)/len(scores):.3f}")
```

### Provenance Filtering

Only use human-verified annotations for high-confidence evaluation:

```python
annotation = json.loads(ann_path.read_text())
fields_meta = annotation["metadata"]["fields"]

# Filter to only human-entered or LLM-checked fields
verified_data = {
    k: v for k, v in annotation["data"].items()
    if fields_meta.get(k, {}).get("source") == "human"
    or fields_meta.get(k, {}).get("checked") is True
}
```

---

## Module Reference

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit entry point, layout, deep link handling |
| `config.py` | Config dialog, session state, query param auto-apply |
| `dataset.py` | PDF discovery, session-aware document status |
| `schema_loader.py` | Load schema from file, import path, or builder output |
| `schema_builder.py` | In-app schema builder UI |
| `pdf_viewer.py` | Lazy PDF page rendering via pdf2image |
| `annotation_panel.py` | Field entry UI, type-aware rendering (scalar + array), all three modes |
| `serializer.py` | `AnnotationManifest`, `AnnotationSession`, `AnnotationSerializer` |
| `llm_backend.py` | AWS Bedrock (Claude) integration |
| `models.py` | Pydantic models: `AnnotationState`, `FieldAnnotation`, `FieldProvenance` |

---

## Tests

```bash
pytest tests/annotator/ -v
```

Visual acceptance tests: [`docs/testing/visual-acceptance-testing.md`](../../../docs/testing/visual-acceptance-testing.md)
