---
title: Map/Reduce Evaluation
---

# Map/Reduce Evaluation

In production pipelines, documents are often compared individually in a distributed map step, with results saved to JSONL. A separate reduce step aggregates those results into bulk metrics. This pattern supports all metric types including confidence (AUROC, ECE, etc.) and will support future metadata-based metrics (bounding box MAP, etc.).

## How It Works

1. **Map step**: For each document, call `compare_with()` on the ground truth and prediction. Save the result dict to JSONL.
2. **Reduce step**: Read the JSONL file, feed each result through `update_from_comparison_result()`, then call `compute()` for aggregate metrics.

The key mechanism: when predictions are created via `from_json()` with rich values (`_value`, `_confidence`, etc.), the original prediction JSON is automatically included in the comparison result as `prediction_raw`. This enables the reduce step to reconstruct confidence pairs without needing the original model instances.

## Map Step

```python
import json

for doc_id, gt_json, pred_json in your_dataset:
    gt = Invoice(**gt_json)
    pred = Invoice.from_json(pred_json)  # from_json() preserves rich value metadata

    result = gt.compare_with(
        pred,
        include_confusion_matrix=True,
        document_field_comparisons=True,  # required for confidence metrics
    )

    # prediction_raw is included automatically when pred has rich values
    with open("results.jsonl", "a") as f:
        record = {"doc_id": doc_id, "comparison_result": result}
        f.write(json.dumps(record, default=str) + "\n")
```

## Reduce Step

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

evaluator = BulkStructuredModelEvaluator(target_schema=Invoice)

with open("results.jsonl") as f:
    for line in f:
        record = json.loads(line)
        evaluator.update_from_comparison_result(
            record["comparison_result"],
            doc_id=record["doc_id"],
        )

result = evaluator.compute()
print(f"F1: {result.metrics['cm_f1']:.3f}")
print(f"AUROC: {result.confidence_metrics['overall']['auroc']['value']}")
print(f"Coverage: {result.confidence_metrics['coverage']}")
```

## Requirements

For confidence metrics to survive the JSONL round-trip:

1. **Predictions must use `from_json()`** with rich values (`_value`, `_confidence`). Direct construction (`Invoice(name="Widget")`) does not store raw JSON.
2. **`compare_with()` must include `document_field_comparisons=True`**. This provides the field-level comparison data needed to reconstruct confidence pairs.
3. **`compare_with()` must include `include_confusion_matrix=True`**. This is required by `update_from_comparison_result()`.

When these conditions are met, `update_from_comparison_result()` produces identical metrics to the direct `update()` path.

## Parity Guarantee

Both evaluation paths produce the same results:

| Path | How it works | Confidence support |
|------|-------------|-------------------|
| `update(gt, pred)` | Direct model comparison | Reads confidence from model instance |
| `update_from_comparison_result(result)` | From serialized comparison dict | Reconstructs confidence from `prediction_raw` |

This is tested explicitly. See `test_prediction_raw_roundtrip.py::TestBulkVsJsonlReplay`.

## Alternative: aggregate_from_comparisons()

For simpler cases where comparison results are already in memory:

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    aggregate_from_comparisons,
)

comparisons = [result1, result2, result3]
evaluation = aggregate_from_comparisons(comparisons)
```

This also supports confidence metrics when the comparison results contain `prediction_raw`.

## Notebook

See [`Map_Reduce_Evaluation.ipynb`](https://github.com/awslabs/stickler/blob/main/examples/notebooks/Map_Reduce_Evaluation.ipynb) for a runnable end-to-end example that demonstrates the full flow and verifies parity between direct and JSONL-replay paths.

## See Also

- [Rich Value Pattern](rich-value-pattern.md): the `_value`/`_confidence` convention that enables metadata round-tripping
- [Confidence Metrics](confidence-metrics.md): the metrics computed from confidence data
- [Bulk Evaluation Guide](../Guides/Evaluation/bulk-evaluation.md): the bulk evaluator that powers aggregation
