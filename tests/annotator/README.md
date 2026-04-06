# annotator tests

Unit and property-based tests for `src/stickler/annotator/`.

## Files

| File | Covers |
|---|---|
| `conftest.py` | Shared Hypothesis strategies and pytest fixtures (see below) |
| `test_dataset.py` | `DatasetManager` init validation, `discover()` PDF finding, `get_status()` status derivation |
| `test_schema_builder.py` | `SchemaBuilder` field-to-schema conversion, export, StructuredModel compatibility |
| `test_schema_loader.py` | `SchemaLoader` file loading, Pydantic import, builder schema, extension preservation |
| `test_serializer.py` | `AnnotationSerializer` path derivation, save/load, round-trip integrity, corrupted file handling |

## Shared Strategies (`conftest.py`)

Hypothesis strategies for property-based tests. Import `st_hyp` (not `st`) to avoid Streamlit collision.

| Strategy | Produces | Used by Properties |
|---|---|---|
| `st_field_name()` | Valid field names (letter + alphanumeric/underscore) | All schema-related |
| `st_primitive_type()` | One of `"string"`, `"number"`, `"integer"`, `"boolean"` | 3, 5, 6 |
| `st_json_schema(max_depth, min_fields, max_fields, with_extensions)` | Valid JSON Schemas with nested objects, arrays, stickler extensions | 3, 4, 5, 6, 11, 12 |
| `st_field_provenance()` | `FieldProvenance` instances | 8, 9, 14 |
| `st_field_annotation()` | `FieldAnnotation` with random values and provenance | 2, 7, 14 |
| `st_annotation_state(schema_fields)` | `AnnotationState` instances | 2, 7, 8, 9, 14 |
| `st_pdf_filename()` | Valid PDF filenames | 10 |
| `st_directory_tree(tmp_path, ...)` | Temp directory with PDF/non-PDF files, returns `(dir, pdf_paths)` | 1, 13 |

Fixtures: `sample_schema` (simple 4-field schema), `sample_annotation_state` (mixed provenance state).

## Running

```bash
conda run -n stickler-dev python -m pytest tests/annotator/ -v
```
