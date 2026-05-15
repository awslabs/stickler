# Changelog

All notable changes to Stickler are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pluggable confidence metrics framework under
  `stickler.structured_object_evaluator.models.confidence` (AUROC, Brier
  Score, ECE with reliability bins, Error Capture at Review Budget).
- `BulkStructuredModelEvaluator` now accumulates confidence pairs across
  documents and exposes `confidence_metrics` in the aggregated result.
- `StructuredModel.compare_with(..., add_confidence_metrics=True)` adds
  a structured `confidence_metrics` block to the single-doc result with
  `overall`, `fields`, and `coverage` keys.
- `PostComparisonAccumulator` interface so future accumulators (e.g.
  bbox mAP) can plug into the bulk pipeline.
- `prediction_raw` round-tripping through comparison results enables
  lossless map/reduce aggregation from JSONL corpora.
- `CHANGELOG.md` (this file).

### Changed
- **Rich Value Pattern** renames the convention keys from
  `{"value", "confidence"}` to `{"_value", "_confidence"}`. The
  underscore prefix prevents collision with user data fields that may
  legitimately be named `value` or `confidence`. Any dict with a
  `_value` key is treated as a rich value wrapper; extra
  underscore-prefixed keys (e.g. `_bbox`, `_source_span`) are preserved
  on the instance via `get_field_extras()`.
- `StructuredModel.from_json(process_confidence=...)` renamed to
  `from_json(process_rich_values=...)` to reflect the broader scope of
  unwrapping.
- Single-doc confidence result key renamed from
  `auroc_confidence_metric` (a single float) to `confidence_metrics`
  (a nested dict with `overall`, `fields`, and `coverage`).
- **Reserved namespace.** Underscore-prefixed keys (`_*`) inside a rich
  value wrapper are now reserved for stickler. Any dict containing a
  `_value` key is treated as a wrapper, and stickler may extract
  additional `_`-prefixed keys (`_confidence`, `_bbox`, `_source_span`,
  ...) into typed accessors as the schema grows. Don't ship
  `{"_value": x}` payloads where `_value` is intended as user data.

### Deprecated
One release of deprecation shims covers the Rich Value rename so
existing callers and JSONL corpora continue to work on upgrade. Each
shim emits a `DeprecationWarning` on use and will be removed in the
next release:

- `StructuredModel.from_json(process_confidence=...)` still accepted as
  an alias for `process_rich_values=...`.
- `compare_with(add_confidence_metrics=True)` still populates the legacy
  `auroc_confidence_metric` key alongside the new `confidence_metrics`
  block, mirroring the nested AUROC value (or the pre-rename `0.5`
  sentinel when undefined).
- `{"value": ..., "confidence": ...}` rich value payloads are still
  detected and unwrapped, with the `DeprecationWarning` naming the
  offending field path.

### Fixed
- `ConfidencePair` now validates that `confidence` and `similarity` are
  finite floats in `[0.0, 1.0]`, preventing NaN/out-of-range inputs from
  silently corrupting Brier/AUROC/ECE results.
- `ErrorCaptureAtBudgetMetric` now reports gain against the actual
  reviewed fraction (`k/n`) rather than the requested budget. At small
  `n` or tight budgets, rounding forces `k = max(1, int(n * budget))` to
  review more than the requested budget; the previous calculation
  reported spurious inflated gains (e.g. `10x` for `n=1, budget=0.1`)
  when the actual review fraction was 100%.
- `ConfidenceCalculator.extract` now skips field comparison rows with
  `actual_key=None` (list FN entries where the prediction has fewer
  items than ground truth). These rows previously inflated
  `fields_total` without contributing to `fields_with_confidence`,
  biasing the coverage ratio by the prediction miss rate.
- `StructuredModel` instances always capture the raw input JSON
  (internally on `__stickler_raw_json__`) when
  `process_rich_values=True`, so map/reduce aggregation works even on
  corpora that gain `_confidence` annotations after the fact.

### Migration Guide
For users upgrading from the pre-rename API:

1. Update JSONL/JSON payloads from `{"value": x, "confidence": c}` to
   `{"_value": x, "_confidence": c}`. Old payloads still work for one
   release but emit `DeprecationWarning`.
2. Replace `Model.from_json(..., process_confidence=...)` with
   `Model.from_json(..., process_rich_values=...)`.
3. Replace `result["auroc_confidence_metric"]` with
   `result["confidence_metrics"]["overall"]["auroc"]["value"]`. The
   nested form also exposes per-field metrics and coverage stats.

## [0.2.0] and earlier
See the git history prior to the introduction of this changelog.
