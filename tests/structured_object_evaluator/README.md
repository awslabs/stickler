# Structured evaluation tests

These tests exercise model configuration, recursive scoring, list matching,
confusion matrices, and reporting through `StructuredModel.compare_with()`.
Run this directory with `python -m pytest tests/structured_object_evaluator/`.

`test_nested_plain_basemodel.py` covers plain Pydantic model dispatch and class
gating. The cross-collector list-reporting regression matrix for issue #332 is
in `../test_list_reporting.py`. See `../README.md` for general test conventions.
