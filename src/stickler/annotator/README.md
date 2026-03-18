# annotator — KIE Annotation Tool

Streamlit-based web app for annotating PDF documents with structured key-value data against a JSON Schema. Lives at `src/stickler/annotator/`.

## Modules

| Module | Purpose |
|---|---|
| `models.py` | Pydantic data models: `AnnotationMode`, `DocumentStatus`, `FieldProvenance`, `FieldAnnotation`, `AnnotationState` |
| `app.py` | Streamlit entry point, layout orchestration, mode routing |
| `config.py` | Session configuration: dataset path, schema source, mode selection |
| `dataset.py` | Recursive PDF discovery, document status tracking, queue ordering |
| `schema_loader.py` | Load schema from JSON Schema file or Pydantic import path |
| `schema_builder.py` | In-app UI for building schemas without writing code |
| `pdf_viewer.py` | PDF page rendering with lazy loading via `pdf2image` |
| `annotation_panel.py` | Field entry/review UI, mode-specific workflows (Zero Start, LLM Inference, HITL) |
| `serializer.py` | Annotation file I/O (JSON), auto-save, round-trip integrity |
| `llm_backend.py` | AWS Bedrock integration for LLM pre-filling |

## Data Models (`models.py`)

All models use Pydantic v2.

- **`AnnotationMode`** — `str` enum: `zero_start`, `llm_inference`, `hitl`
- **`DocumentStatus`** — `str` enum: `not_started`, `in_progress`, `complete`. Derived from annotation file state, not stored separately.
- **`FieldProvenance`** — Tracks `source` (`"human"` | `"llm"`) and `checked` (bool, whether an LLM value was reviewed).
- **`FieldAnnotation`** — Single field: `value` (Any), `is_none` (bool, explicit None marker), `provenance` (FieldProvenance).
- **`AnnotationState`** — Full document state: `schema_hash`, `fields` (dict of path → FieldAnnotation), `created_at`, `updated_at` (ISO 8601).

## Configuration Sidebar (`config.py`)

`render_config_sidebar()` draws the sidebar and returns a `ConfigResult` (or `None` if not yet configured / validation failed).

Sidebar widgets:
- `st.text_input` for dataset directory path.
- `st.radio` for schema source: *JSON Schema file*, *Pydantic import path*, or *Schema Builder*. Conditional text inputs appear for the first two; Schema Builder shows an info message directing users to the main panel.
- `st.selectbox` for operating mode: *Zero Start*, *LLM Inference*, *HITL*.
- *Apply Configuration* button triggers validation.

Validation on apply:
- Dataset directory must exist and be a directory (`st.error()` on failure).
- JSON Schema file must exist and parse via `SchemaLoader.from_json_schema_file()`.
- Pydantic import path must resolve via `SchemaLoader.from_pydantic_import()`.
- Schema Builder defers validation — the builder UI in the main panel finalizes the schema.

Session state keys: `config_dataset_dir`, `config_schema_source`, `config_schema_path`, `config_pydantic_import`, `config_mode`, `config_schema`, `config_model_class`, `config_validated`.

`ConfigResult` is a dataclass with `dataset_dir: Path`, `schema_source: str`, `mode: AnnotationMode`, `schema: dict`, `model_class: Type`.

## Dataset Manager (`dataset.py`)

`DatasetManager` discovers PDFs and derives their annotation status:

- Constructor validates the directory exists and contains at least one PDF. Raises `FileNotFoundError`, `NotADirectoryError`, or `ValueError` with descriptive messages.
- `discover()` recursively finds `*.pdf` files (case-insensitive extension) and returns sorted `PDFDocument` list.
- `get_status(pdf_path, schema_fields)` reads the co-located `.json` annotation file and compares `metadata.fields` keys against `schema_fields` to derive `DocumentStatus`. No annotation file → `NOT_STARTED`; all fields present → `COMPLETE`; partial → `IN_PROGRESS`.

`PDFDocument` is a dataclass with `path: Path` and `status: DocumentStatus`.

## PDF Viewer (`pdf_viewer.py`)

`PDFViewer` renders a single PDF with lazy per-page loading and prev/next navigation.

- Constructor takes a `pdf_path: Path` and optional `pages_per_batch: int` (reserved for future batch loading; v1 renders one page at a time).
- `total_pages` property lazily fetches the page count via `pdf2image.pdfinfo_from_path()` without rendering any pages.
- `render_page(page_num)` converts a single 1-indexed page to a PIL Image using `pdf2image.convert_from_path()` with `first_page`/`last_page` params. Raises `RuntimeError` if poppler is missing, `ValueError` for out-of-range pages.
- `render()` displays the current page via `st.image()` with prev/next navigation buttons. Page number is stored in `st.session_state` (keyed by PDF path) to persist across Streamlit reruns.
- Gracefully handles missing `poppler-utils` — shows an `st.error()` instead of crashing.

Requires `pdf2image` (MIT) and `poppler-utils` system package.

## Annotation Panel (`annotation_panel.py`)

`AnnotationPanel` renders schema fields with mode-specific workflows. Constructor takes `schema` (JSON Schema dict), `mode` (AnnotationMode), `annotation_state` (AnnotationState), and `pdf_path` (Path for auto-save).

- `render()` dispatches to the mode-specific renderer.
- `render_zero_start()` — All fields shown with text inputs and "Mark as None" checkboxes. Progress indicator shows "X of Y fields annotated". Provenance is always `source="human"`.
- `render_llm_inference()` — Pre-fill button (wiring deferred to app.py), batch Accept All / Reject All, individual field editing. LLM values shown with 🤖 prefix. Editing an LLM value changes provenance to `source="human"`.
- `render_hitl()` — Fields presented one at a time after pre-fill. Accept (keeps value, `checked=True`), Reject (clears value, manual entry), Edit (change value, `source="human"`). Review progress: "K of M fields reviewed".

Auto-saves on every field change via `AnnotationSerializer.save()`. Fields extracted from `schema["properties"]` keys. Uses `st.session_state` for edit mode tracking in HITL.

## Annotation Serializer (`serializer.py`)

`AnnotationSerializer` handles reading/writing annotation JSON files:

- `annotation_path_for(pdf_path)` replaces `.pdf` extension with `.json` (bijective mapping).
- `save(annotation, pdf_path)` writes 4-space-indented JSON with `data` (raw field values) and `metadata` (schema_hash, timestamps, per-field provenance) sections. The `data` section is directly constructable via `Model(**loaded_json["data"])`.
- `load(pdf_path)` reads the JSON file and reconstructs an `AnnotationState`. Returns `None` for missing or corrupted files (logs a warning). Fields without provenance metadata default to `source="human"`, `checked=False`.

## LLM Backend (`llm_backend.py`)

`BedrockLLMBackend` sends PDFs to AWS Bedrock (Claude) for annotation pre-filling.

- Constructor takes `model_id` (default: `anthropic.claude-sonnet-4-20250514`) and `region` (default: `us-east-1`). The boto3 client is lazily created on first use.
- `prefill(pdf_path, schema)` reads the PDF as base64, sends it with the JSON Schema as a structured prompt to Bedrock's `invoke_model` API, and parses the JSON response into field values. Returns a dict whose keys are a subset of the schema's properties. Partial results (missing keys) are OK.
- `estimate_cost(pdf_path)` estimates USD cost based on file size using approximate token-per-byte ratios. The caller (`app.py`) should display a warning when the estimate exceeds `COST_WARNING_THRESHOLD` ($100).
- `_parse_response()` strips markdown code fences if present, then parses JSON. Raises `ValueError` for invalid JSON.

Error handling — all designed for caller-side fallback to manual annotation:
- `RuntimeError` for Bedrock API errors (`ClientError`, `BotoCoreError`, `EndpointConnectionError`), unreachable service, or missing boto3.
- `ValueError` for invalid JSON responses or non-object responses.
- `FileNotFoundError` if the PDF path doesn't exist.

Requires `boto3` (Apache 2.0) and valid AWS credentials with Bedrock access.

## Annotation File Format

One JSON file per PDF, co-located in the same directory with `.json` extension:

```json
{
  "data": { "field_name": "value", ... },
  "metadata": {
    "schema_hash": "abc123",
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T11:00:00Z",
    "fields": {
      "field_name": { "source": "human", "checked": true }
    }
  }
}
```

The `data` section is directly constructable via `Model(**loaded_json["data"])`. The `metadata` section stores provenance without polluting the data.

## Three Operating Modes

1. **Zero Start** — All fields empty, user types values or marks None.
2. **LLM Inference** — Bedrock pre-fills all fields. Batch accept/reject + individual edit.
3. **HITL** — Bedrock pre-fills, fields presented one at a time for accept/reject/edit.

## Launch

```bash
stickler-annotate
# or
streamlit run src/stickler/annotator/app.py
```

## Tests

Tests live in `tests/annotator/`. Property-based tests use Hypothesis.
