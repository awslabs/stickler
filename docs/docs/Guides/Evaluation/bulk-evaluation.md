---
title: Bulk Evaluation
---

# Bulk Evaluation

When you need to evaluate many document pairs -- hundreds to thousands -- use `BulkStructuredModelEvaluator` for memory-efficient streaming evaluation. Instead of holding all results in memory, it accumulates confusion matrix counts incrementally and computes aggregate metrics at the end.

---

## When to Use Bulk Evaluation

Use `BulkStructuredModelEvaluator` when you need to:

- **Evaluate large datasets** that would be impractical to process with individual `compare_with()` calls and manual aggregation.
- **Compute aggregate metrics** (precision, recall, F1, accuracy) across an entire corpus.
- **Stream results** through a pipeline without loading everything into memory at once.
- **Write per-document results** to JSONL for downstream analysis.

For comparing a single pair of documents, use `compare_with()` directly (see [Customizing Your Evaluation](README.md)).

---

## Basic Setup

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

evaluator = BulkStructuredModelEvaluator(target_schema=YourModel)
```

The `target_schema` argument accepts any `StructuredModel` subclass. The evaluator uses it to validate inputs and label output metrics.

---

## The Update / Compute Pattern

The core workflow follows two steps:

1. **`update()`** -- Feed in one ground-truth/prediction pair at a time. Each call runs `compare_with()` internally and accumulates the confusion matrix counts.
2. **`compute()`** -- Calculate final aggregate metrics from the accumulated state.

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

evaluator = BulkStructuredModelEvaluator(target_schema=Invoice)

# Accumulate results one pair at a time
for gt_data, pred_data in dataset:
    gt_model = Invoice(**gt_data)
    pred_model = Invoice(**pred_data)
    evaluator.update(gt_model, pred_model, doc_id=gt_data.get("id"))

# Compute final metrics
result = evaluator.compute()
print(f"Precision: {result.metrics['cm_precision']:.3f}")
print(f"Recall:    {result.metrics['cm_recall']:.3f}")
print(f"F1:        {result.metrics['cm_f1']:.3f}")
```

The `update()` method accepts an optional `doc_id` string for tracking errors and labeling per-document output.

---

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_schema` | `Type[StructuredModel]` | `None` | The StructuredModel class for validation. Required when using `update()`. |
| `verbose` | `bool` | `False` | Print progress information (every 1,000 documents and at completion). |
| `document_non_matches` | `bool` | `True` | Track detailed non-match information for every failed field. |
| `elide_errors` | `bool` | `False` | When `True`, skip documents that raise errors silently. When `False`, record the error and count a false negative. |
| `individual_results_jsonl` | `str` | `None` | File path for appending per-document comparison results as JSONL. |

---

## Batch Processing

For processing multiple pairs in a single call, use `update_batch()`:

```python
# Prepare a list of (ground_truth, prediction, doc_id) tuples
batch = [
    (gt_model_1, pred_model_1, "doc_001"),
    (gt_model_2, pred_model_2, "doc_002"),
    (gt_model_3, pred_model_3, "doc_003"),
]

evaluator.update_batch(batch)
```

This calls `update()` for each tuple and triggers garbage collection for batches of 1,000 or more items.

---

## Monitoring Progress

You can check intermediate metrics at any point without resetting the evaluator state:

```python
# After processing some documents...
current = evaluator.get_current_metrics()
print(f"Documents so far: {current.document_count}")
print(f"Current F1: {current.metrics['cm_f1']:.3f}")

# Continue processing more documents...
evaluator.update(gt_model, pred_model)

# Final computation
final = evaluator.compute()
```

---

## Output

`compute()` returns a `ProcessEvaluation` object with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `document_count` | `int` | Total number of documents processed. |
| `metrics` | `dict` | Overall confusion matrix counts (`tp`, `fp`, `tn`, `fn`, `fd`, `fa`) plus derived metrics (`cm_precision`, `cm_recall`, `cm_f1`, `cm_accuracy`). |
| `field_metrics` | `dict` | Per-field metrics with the same structure as `metrics`, keyed by dotted field path (e.g., `"customer.name"`). |
| `errors` | `list` | Records for any documents that raised exceptions during processing. |
| `total_time` | `float` | Wall-clock time in seconds since the evaluator was created or last reset. |
| `non_matches` | `list` | Detailed non-match records (when `document_non_matches=True`), each tagged with `doc_id`. |

---

## JSONL Output

When you provide the `individual_results_jsonl` parameter, each call to `update()` appends a JSON line to the specified file:

```python
evaluator = BulkStructuredModelEvaluator(
    target_schema=Invoice,
    individual_results_jsonl="results.jsonl",
)

for gt_model, pred_model, doc_id in dataset:
    evaluator.update(gt_model, pred_model, doc_id)
```

Each line in `results.jsonl` contains:

```json
{"doc_id": "doc_001", "comparison_result": {"field_scores": {...}, "overall_score": 0.92, ...}}
```

This is the raw output of `compare_with(include_confusion_matrix=True)` for that pair, making it easy to analyze individual results after the fact.

Lines are appended only after the comparison has been successfully accumulated into the bulk evaluator's state, so the JSONL reflects "successfully accumulated" outcomes rather than "attempted" ones. Documents that error inside `update()` are recorded in the in-memory error list and do not produce a JSONL row.

---

## Saving and Loading Metrics

Save aggregate metrics to JSON for reporting or later analysis:

```python
evaluator.save_metrics("evaluation_metrics.json")
```

The output file includes overall metrics, field-level metrics, processing statistics, error summaries, and evaluator configuration.

---

## Checkpointing and Distributed Processing

For long-running jobs, you can checkpoint the evaluator state and restore it later:

```python
# Save checkpoint
state = evaluator.get_state()

# Later, restore and continue
evaluator.load_state(state)
```

To combine results from multiple evaluator instances (e.g., parallel workers processing different data shards):

```python
# Worker 1
worker1 = BulkStructuredModelEvaluator(target_schema=Invoice)
# ... process shard 1 ...
state1 = worker1.get_state()

# Worker 2
worker2 = BulkStructuredModelEvaluator(target_schema=Invoice)
# ... process shard 2 ...
state2 = worker2.get_state()

# Merge into a single evaluator
combined = BulkStructuredModelEvaluator(target_schema=Invoice)
combined.load_state(state1)
combined.merge_state(state2)

final_result = combined.compute()
```

---

## Map/Reduce: Single-Doc Compare, Bulk Aggregate

In production pipelines (like the IDP Accelerator), documents are often compared individually in a map step, with results saved to JSONL. A separate reduce step aggregates those results into bulk metrics. This pattern works for both confusion matrix metrics and confidence metrics.

### How It Works

When predictions are created via `from_json()` with rich values (e.g., `{"_value": "Widget", "_confidence": 0.95}`), the original prediction JSON is automatically stored in the comparison result as `prediction_raw`. This enables the reduce step to reconstruct confidence pairs without needing the original model instances.

### Map Step: Compare Individual Documents

```python
import json

results_file = "comparison_results.jsonl"

for doc_id, gt_json, pred_json in your_dataset:
    gt = Invoice(**gt_json)
    pred = Invoice.from_json(pred_json)  # from_json() preserves rich value metadata

    result = gt.compare_with(
        pred,
        include_confusion_matrix=True,
        document_field_comparisons=True,
    )

    # Save to JSONL. prediction_raw is included automatically.
    with open(results_file, "a") as f:
        record = {"doc_id": doc_id, "comparison_result": result}
        f.write(json.dumps(record, default=str) + "\n")
```

### Reduce Step: Aggregate from JSONL

```python
evaluator = BulkStructuredModelEvaluator(target_schema=Invoice)

with open(results_file) as f:
    for line in f:
        record = json.loads(line)
        evaluator.update_from_comparison_result(
            record["comparison_result"],
            doc_id=record["doc_id"],
        )

result = evaluator.compute()
print(f"F1: {result.metrics['cm_f1']:.3f}")
print(f"AUROC: {result.confidence_metrics['overall']['auroc']['value']}")
```

### Requirements

For confidence metrics to survive the JSONL round-trip:

1. Predictions must be created via `from_json()` (not direct construction)
2. Rich values must use the `_value`/`_confidence` convention
3. `compare_with()` must be called with `document_field_comparisons=True`

When these conditions are met, `update_from_comparison_result()` produces identical confidence metrics to the direct `update()` path.

### Alternative: aggregate_from_comparisons()

For simpler cases where you have a list of comparison results in memory:

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    aggregate_from_comparisons,
)

results = [comp1, comp2, comp3]  # list of compare_with() outputs
evaluation = aggregate_from_comparisons(results)
```

---

## Pretty Printing

For quick terminal output of accumulated metrics, use:

```python
evaluator.pretty_print_metrics()
```

This displays a formatted summary including overall confusion matrix counts, derived metrics, field-level performance sorted by F1 score, error summaries, and processing statistics.

---

## HTML Reports

Stickler can generate interactive HTML reports from evaluation results. See [Understanding Results](understanding-results.md#html-reports) for details on the `EvaluationHTMLReporter`.

---

## Complete Example

```python
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)


class Document(StructuredModel):
    doc_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )
    title: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )
    author: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.5
    )


# Create the evaluator
evaluator = BulkStructuredModelEvaluator(
    target_schema=Document,
    verbose=True,
    document_non_matches=True,
    individual_results_jsonl="document_results.jsonl",
)

# Process your dataset
for gt_data, pred_data in your_dataset:
    gt = Document(**gt_data)
    pred = Document(**pred_data)
    evaluator.update(gt, pred)

# Get results
result = evaluator.compute()

# Print summary
evaluator.pretty_print_metrics()

# Save for later analysis
evaluator.save_metrics("metrics.json")
```
