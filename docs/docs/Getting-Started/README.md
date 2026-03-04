---
title: Getting Started
---

# Getting Started

## What is Stickler?

Stickler is a Python library for structured JSON comparison and evaluation, designed for generative AI workflows. It lets you assign business weights to fields so that critical identifiers count more than cosmetic text, and it uses specialized comparators (exact, numeric, fuzzy, semantic) tailored to each data type. The Hungarian algorithm optimally matches list elements regardless of order, while the recursive evaluation engine handles arbitrarily nested structures. The result is a single weighted score that reflects real operational impact, not just raw accuracy.

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

## What You Just Did

- **Defined models with `ComparableField`**: Each field declares its own comparator and weight, turning a plain data class into an evaluation-aware structure.
- **Chose specialized comparators**: `ExactComparator` for the shipment ID that must match perfectly, `NumericComparator` with a tolerance for currency amounts, and `LevenshteinComparator` for product names that may have minor variations.
- **Applied business-weighted scoring**: The shipment ID carries a weight of 3.0 because a wrong ID routes packages to the wrong warehouse. Lower-priority fields have smaller weights. The overall score is a weighted average that reflects operational impact.
- **Used Hungarian matching for lists**: The `line_items` field contains a list of `LineItem` objects. Stickler uses the Hungarian algorithm to find the optimal one-to-one pairing between ground truth and prediction items, regardless of order.
- **Compared and got results**: `compare_with` returns a dictionary with an `overall_score` and per-field `field_scores`, so you can see exactly where the differences are and how much they matter.

## Next Steps

- [Installation](installation.md) -- detailed setup instructions, conda environment, and optional dependencies.
- [Comparators](../Guides/Comparators/README.md) -- understand the full set of comparison algorithms and when to use each one.
- [Evaluation](../Guides/Evaluation/README.md) -- customize thresholds, clipping, aggregation, and the evaluation engine.
- [Use Cases](../Guides/Use-Cases/README.md) -- real-world patterns for document extraction, OCR, ML evaluation, and more.
