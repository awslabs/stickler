# Structured evaluation tests

These tests exercise model configuration, recursive scoring, list matching,
confusion matrices, and reporting through `StructuredModel.compare_with()`.
Run this directory with `python -m pytest tests/structured_object_evaluator/`.

`test_nested_plain_basemodel.py` covers plain Pydantic model dispatch and class
gating. `test_jsonl_nonfinite.py` checks the `individual_results_jsonl` writer
emits strict JSON, NaN and infinities as `null` except a confidence, bbox-map
value, or `doc_id`, which is written as a string (#450, #600). The cross-collector
list-reporting regression matrix for issue #332 is in
`../test_list_reporting.py`. See `../README.md` for general test conventions.
