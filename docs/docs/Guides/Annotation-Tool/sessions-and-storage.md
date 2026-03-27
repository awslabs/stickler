# Sessions & Storage

## Directory Layout

```
<dataset_dir>/
  *.pdf                        ← source documents
  .annotations/
    manifest.json              ← schema + session metadata
    <session-guid>/
      <pdf-stem>.json          ← per-doc annotation
```

## Manifest

The manifest embeds the full JSON Schema so the dataset is self-contained:

```json
{
  "schema": { "type": "object", "properties": { ... } },
  "schema_hash": "dff112...",
  "sessions": {
    "42fac85a-...": {
      "annotator": "sromo",
      "created_at": "2026-03-18T...",
      "updated_at": "2026-03-27T...",
      "doc_count": 370,
      "completed_count": 12
    }
  }
}
```

## Per-Document Annotation

Each annotated document produces a JSON file with `data` (field values) and `metadata` (provenance + locations):

```json
{
  "data": {
    "station_name": "COX MEDIA - WEST",
    "invoice_id": "1277139",
    "line_items": [
      {"air_date": "02/29/2016", "program": "Evening News"}
    ]
  },
  "metadata": {
    "schema_hash": "dff112...",
    "created_at": "2026-03-18T...",
    "updated_at": "2026-03-27T...",
    "fields": {
      "station_name": {"source": "human", "checked": false},
      "invoice_id": {
        "source": "llm",
        "checked": true,
        "location": {"page": 1, "bbox": [780, 55, 950, 80]}
      }
    }
  }
}
```

The `data` section is directly loadable into a Pydantic model: `Model(**annotation["data"])`.

## Completion Logic

A document is "complete" when every field in `data` has a corresponding entry in `metadata.fields` (provenance metadata). Fields with `null` values that were explicitly marked N/A count as annotated.

## Loading Annotations Programmatically

```python
from pathlib import Path
from stickler.annotator.serializer import AnnotationManifest

manifest = AnnotationManifest(Path("./files"))
session = manifest.get_session("42fac85a-...")

# Load one document
state = session.load(Path("./files/invoice.pdf"))
print(state.fields["invoice_id"].value)       # "1277139"
print(state.fields["invoice_id"].location)     # FieldLocation(page=1, bbox=[780, 55, 950, 80])
```

## Deep Links

URLs encode dataset, session, and document position:

```
http://localhost:8501/?dataset=./files&session=42fac85a-...&doc=invoice.pdf
```

The `doc` parameter matches against PDF filenames in the dataset directory. On page refresh, the app restores the exact document position.
