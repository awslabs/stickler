# Structured Object Evaluator — Models

Core building blocks for defining evaluatable schemas and comparing two
`StructuredModel` instances field-by-field. `StructuredModel` (a Pydantic
`BaseModel` subclass) is the public entry point; everything else here supports
its `from_json()` ingestion and `compare_with()` comparison pipeline.

## Orientation

**Schema definition**
- `structured_model.py` — `StructuredModel`, the base class users subclass. Owns
  `from_json()` (rich-value unwrapping) and `compare_with()` (delegates to the engine).
- `comparable_field.py`, `field.py`, `comparison_info.py` — field declaration
  (`ComparableField`), per-field comparator/threshold/weight metadata.
- `model_factory.py`, `field_converter.py`, `json_schema_field_converter.py`,
  `type_resolver.py`, `configuration_helper.py` — building models from JSON / schema.

**Comparison pipeline**
- `comparison_engine.py` — `ComparisonEngine`, the orchestrator behind
  `compare_with()`. Runs a single recursive traversal, then layers on confusion
  matrix, non-matches, field comparisons, and optional metrics.
- `comparison_dispatcher.py`, `field_comparator.py`,
  `primitive_list_comparator.py`, `structured_list_comparator.py`,
  `hungarian_helper.py` — dispatch and per-type comparison, including Hungarian
  matching for lists.
- `field_comparison_collector.py` / `field_comparison_helper.py` — produce the
  `field_comparisons` rows (each carries a GT-side `expected_key` and a
  prediction-side `actual_key`, which diverge for reordered list items).
- `non_match_collector.py`, `null_helper.py`,
  `evaluator_format_helper.py` — result shaping. Standard result-dict factories
  (true negative / false alarm / false negative / empty-list cases) are inlined
  as private functions in `comparison_dispatcher.py` (their only caller) rather
  than a separate helper module.

**Metrics**
- `confusion_matrix_builder.py`, `confusion_matrix_calculator.py`,
  `derived_metrics_calculator.py`, `aggregate_metrics_calculator.py`,
  `metrics_helper.py`, `threshold_helper.py` — confusion-matrix and derived/
  aggregate metric rollups.
- `rich_value_helper.py` — unwraps the Rich Value Pattern
  (`{"_value": ..., "_confidence": ..., "_bbox": ...}`) during `from_json()`,
  returning `(data, confidences, extras)`.
- `post_comparison_accumulator.py` — `PostComparisonAccumulator`, the interface
  for bulk metric accumulators consumed by `BulkStructuredModelEvaluator`.

## Metric sub-packages

Pluggable post-comparison metrics live in their own packages, each with a
calculator (the math) and a `PostComparisonAccumulator` (bulk aggregation):

- `confidence/` — confidence calibration metrics (AUROC, Brier, ECE, ...).
- `bbox/` — bounding-box localization via mean Average Precision (mAP).

Both consume metadata extracted by `rich_value_helper.py` and plug into bulk
evaluation through `post_comparison_accumulator.py`.
