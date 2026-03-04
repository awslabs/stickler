---
title: Engine Architecture
---

# Comparison Engine Architecture

Internal architecture reference for contributors and maintainers working on the Stickler comparison engine. For user-facing feature documentation, see [Advanced](../../Advanced/README.md).

## Overview

The comparison engine evaluates how well a predicted structured object matches a ground truth object. It produces similarity scores, confusion matrix metrics, and match/non-match documentation — all in a **single traversal** of the model tree.

The system follows a **delegation pattern**: `StructuredModel` exposes the public API, but all comparison work is delegated to specialized helper classes with single responsibilities. Helpers receive the model instance as a parameter (composition over inheritance) and are lazily initialized to avoid circular imports.

## Component Map

```
StructuredModel
│
├── compare_with()            # Public API entry point
│   └── ComparisonEngine      # Orchestrator — single-traversal loop
│       ├── ComparisonDispatcher        # 5-step field routing
│       │   ├── NullHelper              # Null/empty detection
│       │   ├── ResultHelper            # Standard result factories
│       │   ├── FieldComparator         # Primitives & nested models
│       │   ├── PrimitiveListComparator # List[str/int/float]
│       │   └── StructuredListComparator  # List[StructuredModel]
│       │       ├── HungarianHelper     # Optimal bipartite matching
│       │       └── MetricsHelper       # Derived metrics (precision, recall, F1)
│       ├── NonMatchCollector           # Non-match documentation
│       ├── FieldComparisonCollector    # Field-level comparison docs
│       └── ConfusionMatrixBuilder      # Aggregate metrics
│
├── ComparisonHelper          # List metrics & Hungarian integration
│   └── ThresholdHelper       # Threshold comparison logic
└── MetricsHelper             # Also used directly for score→metrics conversion
```

### File Path Reference

| Component | File |
|-----------|------|
| StructuredModel | `models/structured_model.py` |
| ComparisonEngine | `models/comparison_engine.py` |
| ComparisonDispatcher | `models/comparison_dispatcher.py` |
| FieldComparator | `models/field_comparator.py` |
| PrimitiveListComparator | `models/primitive_list_comparator.py` |
| StructuredListComparator | `models/structured_list_comparator.py` |
| HungarianHelper | `models/hungarian_helper.py` |
| NonMatchCollector | `models/non_match_collector.py` |
| FieldComparisonCollector | `models/field_comparison_collector.py` |
| ConfusionMatrixBuilder | `models/confusion_matrix_builder.py` |
| NullHelper | `models/null_helper.py` |
| ResultHelper | `models/result_helper.py` |
| ThresholdHelper | `models/threshold_helper.py` |
| MetricsHelper | `models/metrics_helper.py` |
| ComparisonHelper | `models/comparison_helper.py` |
| ComparableField (function) | `models/comparable_field.py` |

All paths are relative to `src/stickler/structured_object_evaluator/`.

## Core Recursive Engine

### Entry Point

`StructuredModel.compare_with()` creates a `ComparisonEngine` and delegates to it. The engine's `compare_recursive()` method is the heart of the system.

### Single-Traversal Design

The engine iterates through every field in the ground truth model **once**, dispatching each comparison and collecting scores, confusion matrix counts, and non-match data in the same pass.

```python
# See: comparison_engine.py:128-192
# Simplified flow — see source for full implementation

result = {"overall": {metrics}, "fields": {}, "non_matches": []}
total_score = 0.0
total_weight = 0.0
threshold_matched_fields = set()

for field_name in model.model_fields:
    field_result = dispatcher.dispatch_field_comparison(field_name, gt_val, pred_val)
    result["fields"][field_name] = field_result
    _aggregate_to_overall(field_result, result["overall"])
    # Score percolation: accumulate weighted scores
    total_score += field_result["threshold_applied_score"] * weight
    total_weight += weight
```

### Score Percolation Variables

Three tracking variables accumulate during traversal:

- **`total_score`** — Running sum of `threshold_applied_score * weight` per field
- **`total_weight`** — Running sum of field weights (denominator for weighted average)
- **`threshold_matched_fields`** — Set of fields where `raw_similarity_score >= threshold`

### Overall Score Determination

After the field loop completes:

```python
# See: comparison_engine.py:181-191
overall_score = total_score / total_weight  # Weighted average

# all_fields_matched is True only when EVERY field meets its threshold
all_fields_matched = len(threshold_matched_fields) == len(model_fields)
```

### Extra Field Handling

After the main loop, `_count_extra_fields_as_false_alarms()` recursively checks the prediction for hallucinated fields (via `__pydantic_extra__`) and adds them as False Alarms. This catches fields the prediction invented that don't exist in the ground truth schema.

```python
# See: comparison_engine.py:175-178
extra_fields_fa = self._count_extra_fields_as_false_alarms(other)
result["overall"]["fa"] += extra_fields_fa
result["overall"]["fp"] += extra_fields_fa
```

## Field Dispatch System

### ComparisonDispatcher 5-Step Cascade

The dispatcher routes each field comparison through a 5-step decision tree:

| Step | Logic | Early Exit? |
|------|-------|-------------|
| 1. **Get field config** | Extract weight, threshold, comparator from `_get_comparison_info()` | No |
| 2. **Determine types** | Check `_is_list_field()` and `_should_use_hierarchical_structure()` | No |
| 3. **List null cases** | Match on `(gt_null, pred_null)` → TN/FA/FN | Yes, if null case |
| 4. **Primitive null cases** | Match on `(gt_null, pred_null)` for non-hierarchical fields | Yes, if null case |
| 5. **Type-based dispatch** | Route by runtime types to specialized comparator | Terminal |

```python
# See: comparison_dispatcher.py:65-225
# Step 5 type routing:
#   (str|int|float, str|int|float) → FieldComparator
#   (list, list) where list[0] is StructuredModel → StructuredListComparator
#   (list, list) otherwise → PrimitiveListComparator
#   (StructuredModel, StructuredModel) → FieldComparator
#   Mismatched types → FD result (score=0.0)
```

### Lazy Initialization

Both `ComparisonEngine` and `ComparisonDispatcher` use `@property` with `None`-guard patterns to lazily create sub-components. This avoids circular imports between `structured_model.py` and the helper modules.

```python
# See: comparison_dispatcher.py:41-63
@property
def field_comparator(self):
    if self._field_comparator is None:
        from .field_comparator import FieldComparator
        self._field_comparator = FieldComparator(self.model)
    return self._field_comparator
```

### Match-Statement Routing

Null cases use Python 3.10+ `match` statements for clarity:

```python
# See: comparison_dispatcher.py:157-169
match (gt_effectively_null, pred_effectively_null):
    case (True, True):   return ResultHelper.create_true_negative_result(weight)
    case (True, False):  return ResultHelper.create_false_alarm_result(weight)
    case (False, True):  return ResultHelper.create_false_negative_result(weight)
    case _:              pass  # Continue to type dispatch
```

## Specialized Comparators

### FieldComparator

Handles two cases:

- **Primitives** (`compare_primitive_with_scores`): Uses the field's configured `BaseComparator` (e.g., Levenshtein, exact match) to produce a similarity score, then applies threshold and weight.
- **Nested StructuredModel** (`compare_structured_field`): Recursively calls `compare_recursive` on the nested model, wrapping the result with the parent field's weight and threshold.

### PrimitiveListComparator

Compares `List[str]`, `List[int]`, etc. using Hungarian matching for optimal element pairing.

**Universal hierarchical structure:** Returns `{"overall": {...}, "fields": {...}}` even for primitive lists. This ensures all list fields use the same access pattern (`result["fields"][name]["overall"]`), which simplifies consumers and test assertions.

For details on Hungarian matching mechanics, see [Advanced > Hungarian Matching](../../Advanced/hungarian-matching.md).

### StructuredListComparator

The most complex comparator — handles `List[StructuredModel]` with three phases:

1. **Object-level metrics** — Hungarian matching determines TP/FD/FA/FN at the *object* level (counting whole objects, not individual fields)
2. **Similarity scoring** — Threshold-corrected individual comparisons for each matched pair
3. **Nested field metrics** — Threshold-gated recursive analysis of matched pairs

#### Known Bugs (from source header)

The source header documents preserved behavioral bugs:

```python
# See: structured_list_comparator.py:8-12
# Current Behavior Preserved (including bugs):
# - Uses parent field threshold instead of object match_threshold (bug)
# - Generates nested metrics for all matched pairs regardless of threshold (bug)
# - Object-level counting discrepancies in some scenarios (bug)
```

> **Note:** Phase 3 fixes have addressed some of these. The `match_threshold` fix is implemented at line 59-67. The header comments may be stale — verify against current behavior before assuming bugs are present.

#### Threshold-Gated Recursion Internals

Field-level detail is only generated for object pairs with `similarity >= match_threshold`. Poor matches are treated as atomic failures without recursive field analysis. This is both a correctness decision (poor matches don't have meaningful field-level breakdowns) and a performance optimization.

```python
# See: structured_list_comparator.py:243-249
good_matched_pairs = [
    (gt_idx, pred_idx, similarity)
    for gt_idx, pred_idx, similarity in matched_pairs
    if similarity >= match_threshold
]
```

For user-facing threshold-gated evaluation documentation, see [Advanced > Threshold-Gated Evaluation](../../Advanced/threshold-gated-evaluation.md).

## Score Aggregation

### Score Types

Every field comparison produces three score variants:

| Score | Description |
|-------|-------------|
| `raw_similarity_score` | Direct comparator output (0.0–1.0) |
| `similarity_score` | Same as raw — maintained for API compatibility |
| `threshold_applied_score` | Raw score with `clip_under_threshold` applied (0.0 if below threshold and clipping is enabled) |

### Weight-Based Formula

The overall similarity score is a weighted average:

```
overall_score = Σ(threshold_applied_score_i × weight_i) / Σ(weight_i)
```

Weights are configured per-field via `ComparableField` metadata. Default weight is 1.0.

### `clip_under_threshold`

When enabled on a field (the default), `FieldComparator` clips scores below the threshold to 0.0 when producing `threshold_applied_score`. The engine's percolation loop then uses the already-clipped score. Lists are exempt — both `PrimitiveListComparator` and `StructuredListComparator` always preserve partial match scores:

```python
# See: field_comparator.py:73-76 — where clipping happens for primitives
threshold_applied_score = (
    0.0 if info.clip_under_threshold else raw_similarity
)

# See: structured_list_comparator.py:84-85 — lists bypass clipping
# CRITICAL FIX: For structured lists, we NEVER clip under threshold
threshold_applied_score = raw_similarity  # Always use raw score for lists
```

### `_aggregate_to_overall()`

Sums confusion matrix counts from each field result into the overall totals:

```python
# See: comparison_engine.py:315-328
for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
    if metric in field_result:
        overall[metric] += field_result[metric]
    elif "overall" in field_result and metric in field_result["overall"]:
        overall[metric] += field_result["overall"][metric]
```

This handles both flat results (primitives) and hierarchical results (lists/nested models).

## Performance

### Single-Traversal Benefits

Before the current architecture, comparison required multiple passes:

1. Pass 1: Calculate similarity scores
2. Pass 2: Generate confusion matrix
3. Pass 3: Collect non-matches

The single-traversal design collects all three in **one pass** through `compare_recursive()`. The `compare_with()` method then optionally post-processes the result (confusion matrix formatting, non-match collection, field comparison docs) without re-traversing.

### Lazy Evaluation

Optional features are computed only when requested:

```python
# See: comparison_engine.py:279-306
if include_confusion_matrix:
    confusion_matrix = self.confusion_matrix_builder.build_confusion_matrix(...)
if document_non_matches:
    non_matches = self.non_match_collector.collect_enhanced_non_matches(...)
if document_field_comparisons:
    field_comparisons = self.field_comparison_collector.collect_field_comparisons(...)
if add_confidence_metrics:
    auroc = ConfidenceCalculator().calculate_overall_auroc(...)
```

### Threshold Gating as Optimization

Threshold-gated recursion in `StructuredListComparator` avoids expensive recursive field analysis for poor matches. For a list of N objects where K pairs are below threshold, this saves K full recursive comparisons.

### Hungarian Algorithm Complexity

The Hungarian algorithm runs in **O(n³)** time. For `StructuredListComparator`, this is applied at the object level (not field level), so `n` is the number of list items — typically small. For large lists, this can become a bottleneck.

### Known TODOs

From `ComparisonDispatcher` source:

```python
# See: comparison_dispatcher.py:177-184
# TODO: Refactor to use a cleaner match-based dispatch pattern that separates
#       list handling from singleton handling more explicitly.
```

## Debugging & Troubleshooting

### Common Issues

| Symptom | Where to Look | What to Check |
|---------|---------------|---------------|
| Wrong similarity scores | `FieldComparator` | `ComparableField` comparator/threshold config |
| TP + FP ≠ expected | `StructuredListComparator` | Object-level vs field-level counting |
| Wrong comparator used | `ComparisonDispatcher` | `_is_list_field()` / `_should_use_hierarchical_structure()` return values |
| Slow on large objects | `StructuredListComparator` | Threshold gate effectiveness; Hungarian O(n³) |
| High memory usage | `ComparisonEngine` | Result structure depth; helper instantiation count |

### Tracing Dispatch Decisions

To trace which comparator handles a field:

```python
from stickler.structured_object_evaluator.models.comparison_dispatcher import ComparisonDispatcher

dispatcher = ComparisonDispatcher(gt_model)
info = gt_model._get_comparison_info(field_name)
print(f"Field: {field_name}")
print(f"  is_list_field: {gt_model._is_list_field(field_name)}")
print(f"  comparator: {info.comparator.__class__.__name__}")
print(f"  threshold: {info.threshold}, weight: {info.weight}")

result = dispatcher.dispatch_field_comparison(field_name, gt_val, pred_val)
print(f"  result keys: {list(result.keys())}")
```

### Tracing Score Percolation

To understand how the overall score is computed:

```python
from stickler.structured_object_evaluator.models.comparison_engine import ComparisonEngine

engine = ComparisonEngine(gt_model)
result = engine.compare_recursive(pred_model)

for name, field_result in result["fields"].items():
    raw = field_result.get("raw_similarity_score", "N/A")
    applied = field_result.get("threshold_applied_score", "N/A")
    weight = field_result.get("weight", "N/A")
    print(f"  {name}: raw={raw}, applied={applied}, weight={weight}")

print(f"Overall: {result['overall']['similarity_score']}")
```

### Profiling Tips

```python
import tracemalloc
from stickler.structured_object_evaluator.models.comparison_engine import ComparisonEngine

tracemalloc.start()
engine = ComparisonEngine(gt_model)
result = engine.compare_recursive(pred_model)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
```

For CPU profiling, wrap `compare_recursive` with `cProfile` and sort by cumulative time — the dispatch and Hungarian matching methods will typically dominate.

## Maintenance Guidelines

### Delegation Pattern Rules

1. **Keep helpers focused** — each class has one responsibility. Don't add unrelated logic to an existing helper.
2. **Preserve lazy initialization** — all cross-module imports inside `@property` methods to avoid circular import chains.
3. **Consistent result structures** — all comparators must return dicts with `overall`, `raw_similarity_score`, `similarity_score`, `threshold_applied_score`, and `weight` keys. Hierarchical comparators also include `fields`.

### Single-Traversal Integrity

Any new feature that needs comparison data **must** integrate into the existing `compare_recursive` loop or post-process its result — never add a second traversal.

### API Compatibility

- `StructuredModel.compare_with()` and `StructuredModel.compare()` are the public API. Their signatures and return structure must remain backward-compatible.
- Internal helpers (`ComparisonEngine`, `ComparisonDispatcher`, etc.) are not public API but are used extensively in tests. Changes require updating corresponding test files in `tests/structured_object_evaluator/`.
