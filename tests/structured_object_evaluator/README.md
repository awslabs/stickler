# Structured evaluation tests

These tests exercise model configuration, recursive scoring, list matching,
confusion matrices, and reporting through `StructuredModel.compare_with()`.
Run this directory with `python -m pytest tests/structured_object_evaluator/`.

`test_nested_plain_basemodel.py` covers plain Pydantic model dispatch and class
gating. The cross-collector list-reporting regression matrix for issue #332 is
in `../test_list_reporting.py`. `test_threshold_validation.py` checks that a
threshold or `match_threshold` that is not an `int` or `float` from 0.0 to 1.0
raises on every path, with one message.
See `../README.md` for general test conventions.
