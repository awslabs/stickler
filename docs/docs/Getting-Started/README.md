---
title: Getting Started
---

# Getting Started

## What is Stickler?

You're building an invoice extraction pipeline. Your AI reads scanned documents and produces structured JSON — invoice IDs, amounts, line items. How accurate is it? Do the errors matter? A wrong total is a billing error. A wrong ID routes a package to the wrong warehouse. A minor typo in a vendor name is cosmetic.

Stickler answers these questions. It compares structured AI output against ground truth using specialized comparators tailored to each data type (exact, numeric, fuzzy, semantic), business-weighted scoring so critical fields count more than cosmetic ones, and Hungarian algorithm matching for lists regardless of order. The result is a single weighted score that reflects real operational impact, not just raw accuracy.

## Your First Evaluation in 30 Seconds

```python
# pip install stickler-eval
from typing import List
from stickler import StructuredModel, ComparableField
from stickler.comparators import ExactComparator, NumericComparator, LevenshteinComparator

# Define your models
class LineItem(StructuredModel):
    product: str = ComparableField(comparator=LevenshteinComparator(), weight=1.0)
    quantity: int = ComparableField(weight=0.8)
    price: float = ComparableField(comparator=NumericComparator(tolerance=0.01), weight=1.2)

class Invoice(StructuredModel):
    shipment_id: str = ComparableField(comparator=ExactComparator(), weight=3.0)  # Critical
    amount: float = ComparableField(comparator=NumericComparator(tolerance=0.01), weight=2.0)
    line_items: List[LineItem] = ComparableField(weight=2.0)  # Hungarian matching!

# JSON from your systems (agent output, ground truth, etc.)
ground_truth_json = {
    "shipment_id": "SHP-2024-001",
    "amount": 1247.50,
    "line_items": [
        {"product": "Wireless Mouse", "quantity": 2, "price": 29.99},
        {"product": "USB Cable", "quantity": 5, "price": 12.99}
    ]
}

prediction_json = {
    "shipment_id": "SHP-2024-001",  # Perfect match
    "amount": 1247.48,  # Within tolerance
    "line_items": [
        {"product": "USB Cord", "quantity": 5, "price": 12.99},  # Name variation
        {"product": "Wireless Mouse", "quantity": 2, "price": 29.99}  # Reordered
    ]
}

# Construct from JSON and compare
ground_truth = Invoice(**ground_truth_json)
prediction = Invoice(**prediction_json)
result = ground_truth.compare_with(prediction)

print(f"Overall Score: {result['overall_score']:.3f}")  # 0.693
print(f"Shipment ID: {result['field_scores']['shipment_id']:.3f}")  # 1.000 - exact match
print(f"Line Items: {result['field_scores']['line_items']:.3f}")  # 0.926 - Hungarian optimal matching
```

??? example "Sample Output"

    **Console output:**
    ```
    Overall Score: 0.693
    Shipment ID: 1.000
    Line Items: 0.926
    ```

    **Full result dictionary:**
    ```json
    {
      "field_scores": {
        "shipment_id": 1.0,
        "amount": 0.0,
        "line_items": 0.926
      },
      "overall_score": 0.693,
      "all_fields_matched": false
    }
    ```

    The `amount` field scores 0.0 because the default `clip_under_threshold` behavior zeros out the score — the difference between 1247.50 and 1247.48 exceeds the `NumericComparator`'s default absolute tolerance of 0.01, and the resulting score falls below the default threshold.

## What You Just Did

- **Defined models with `ComparableField`**: Each field declares its own comparator and weight, turning a plain data class into an evaluation-aware structure.
- **Chose specialized comparators**: `ExactComparator` for the shipment ID that must match perfectly, `NumericComparator` with a tolerance for currency amounts, and `LevenshteinComparator` for product names that may have minor variations.
- **Applied business-weighted scoring**: The shipment ID carries a weight of 3.0 because a wrong ID routes packages to the wrong warehouse. Lower-priority fields have smaller weights. The overall score is a weighted average that reflects operational impact.
- **Used Hungarian matching for lists**: The `line_items` field contains a list of `LineItem` objects. Stickler uses the Hungarian algorithm to find the optimal one-to-one pairing between ground truth and prediction items, regardless of order.
- **Compared and got results**: `compare_with` returns a dictionary with an `overall_score` and per-field `field_scores`, so you can see exactly where the differences are and how much they matter.

## Evaluate a Test Set

In production, you'll compare many document pairs at once. `BulkStructuredModelEvaluator` handles this with streaming aggregation and progress reporting:

```python
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import BulkStructuredModelEvaluator

evaluator = BulkStructuredModelEvaluator(target_schema=Invoice, verbose=True)

for gt_json, pred_json, doc_id in your_test_set:
    gt = Invoice(**gt_json)
    pred = Invoice(**pred_json)
    evaluator.update(gt, pred, doc_id)

result = evaluator.compute()
print(f"Aggregate F1: {result.overall_metrics['f1']:.3f}")
```

See [Bulk Evaluation](../Guides/Evaluation/bulk-evaluation.md) for the full guide.

