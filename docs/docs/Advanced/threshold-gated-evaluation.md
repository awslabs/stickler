---
title: Threshold-Gated Recursive Evaluation
---

# Threshold-Gated Recursive Evaluation

When comparing `List[StructuredModel]` fields, Stickler only performs detailed nested-field analysis on object pairs whose overall similarity meets a configurable threshold. Pairs that fall below the threshold are classified as False Discovery (FD) and treated as atomic units -- no field-by-field breakdown is generated for them.

## Core Principle

**Only recurse into nested field evaluation for object pairs that meet the similarity threshold.**

This keeps metrics focused on meaningful comparisons and avoids generating misleading field-level statistics for object pairs that are fundamentally different.

## Algorithm Flow

### 1. Hungarian Matching

Use the [Hungarian algorithm](hungarian-matching.md) to find optimal pairings between GT and Pred lists based on overall object similarity.

### 2. Threshold Classification

For each matched pair, compare the similarity score against `StructuredModel.match_threshold`:

- **similarity >= threshold** -- **TP**: recurse into nested fields for aggregate metric only (this recursion does not affect object level metrics)
- **similarity < threshold** -- **FD**: recurse into nested fields for aggregate metric only (this recursion does not affect object level metrics)

### 3. Unmatched Items

- **GT extras** -- **FN**: recurse into nested fields for aggregate metric only (this recursion does not affect object level metrics)
- **Pred extras** -- **FA**: recurse into nested fields for aggregate metric only (this recursion does not affect object level metrics)

## Code Example

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator
from typing import List

class Product(StructuredModel):
    product_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=3.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )
    price: float = ComparableField(threshold=0.9, weight=1.0)

    match_threshold = 0.8  # Gates recursive evaluation

class Order(StructuredModel):
    order_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )
    products: List[Product] = ComparableField(weight=3.0)
```

## Worked Scenarios

Given three GT products and two Pred products. Hungarian matching pairs `min(N_gt, N_pred) = 2` items optimally, leaving the remaining GT product unmatched.

### Good Match (similarity >= 0.8)

**GT:** `Product("PROD-001", "Premium Laptop", 999.99)`
**Pred:** `Product("PROD-001", "Premium Laptop X", 999.99)`

- Classification: **TP**
- Nested field analysis contributes to both per-field `overall` and `aggregate`:
    - `product_id`: TP (exact match)
    - `name`: TP (similarity ~0.875)
    - `price`: TP (exact match)

### Poor Match (similarity < 0.8)

**GT:** `Product("PROD-002", "Mouse", 29.99)`
**Pred:** `Product("PROD-002", "Different Product", 99.99)`

- Classification: **FD**
- Nested field analysis is performed for **aggregate** metrics only. The per-field `overall` does not include this pair's field-level breakdown, since the objects are too dissimilar for those counts to be meaningful alongside good matches.

### Unmatched Item

**GT:** `Product("PROD-003", "Cable", 14.99)` -- **FN** (no counterpart in Pred)

Nested field analysis is performed for both per-field `overall` and `aggregate` (each non-null field on the FN item counts as FN). Per-field `overall` includes this contribution since the classification is unambiguous.

If the Pred list had been longer than the GT list, any unpaired Pred items would be classified as **FA** by the same rule (each non-null field on an FA item counts as FA in both per-field `overall` and `aggregate`).

## Result Structure

```json
{
  "products": {
    "overall": {
      "tp": 1, "fd": 1, "fn": 1, "fa": 0,
      "derived": { "cm_precision": 0.5, "cm_recall": 0.5, "cm_f1": 0.5 }
    },
    "aggregate": {
      "tp": 4, "fd": 2, "fn": 3, "fa": 0,
      "derived": { "cm_precision": 0.67, "cm_recall": 0.57, "cm_f1": 0.62 }
    },
    "fields": {
      "product_id": { "overall": { "tp": 1, "fn": 1 },
                      "aggregate": { "tp": 2, "fn": 1 } },
      "name":       { "overall": { "tp": 1, "fn": 1 },
                      "aggregate": { "tp": 1, "fd": 1, "fn": 1 } },
      "price":      { "overall": { "tp": 1, "fn": 1 },
                      "aggregate": { "tp": 1, "fd": 1, "fn": 1 } }
    },
    "non_matches": [
      {
        "type": "FD",
        "gt_object": "Product(PROD-002, Mouse, 29.99)",
        "pred_object": "Product(PROD-002, Different Product, 99.99)",
        "similarity": 0.3
      },
      { "type": "FN", "gt_object": "Product(PROD-003, Cable, 14.99)" }
    ]
  }
}
```

Per-field `overall` metrics include contributions from the TP pair and the unmatched FN item, but not the FD pair. The `aggregate` at each field includes contributions from all pairs -- good matches, poor matches, and unmatched items alike. The `non_matches` list documents every FD, FN, and FA for diagnostic purposes.

## Delegation Pattern

Under the hood, comparison logic is distributed across specialized components:

| Component | Responsibility |
|-----------|---------------|
| **ComparisonEngine** | Orchestrates the single-traversal comparison; manages score percolation |
| **ComparisonDispatcher** | Routes each field to the correct comparator based on type and null state |
| **FieldComparator** | Handles primitives and single nested `StructuredModel` fields |
| **PrimitiveListComparator** | Handles `List[str]`, `List[int]`, etc. via Hungarian matching |
| **StructuredListComparator** | Handles `List[StructuredModel]` with threshold-gated recursion |

The dispatcher uses pattern matching on null states for early exits, then routes non-null values by type to the appropriate comparator.

## Score Aggregation

Scores percolate upward from leaf fields to the top-level result using weighted averaging:

1. Each field comparison produces a raw similarity score (0.0 -- 1.0).
2. The score is optionally clipped to 0 if below the field threshold (`clip_under_threshold`).
3. Clipped scores are multiplied by the field weight and summed.
4. The overall similarity is `total_weighted_score / total_weight`.

The `all_fields_matched` flag is `True` only when every field's raw similarity meets its individual threshold.

## Edge Cases

**Empty lists** -- `[] vs []` is TN. `[] vs [items]` generates one FA per item. `[items] vs []` generates one FN per item.

**Threshold boundary** -- `similarity >= threshold` counts as TP and triggers recursion. Values exactly at the boundary are matches.

**Different thresholds per model** -- Each `StructuredModel` subclass can define its own `match_threshold`. A `Product` with `match_threshold = 0.8` and an `Address` with `match_threshold = 0.6` are each evaluated independently.

**Nested lists** -- When a `StructuredModel` contains another `List[StructuredModel]`, the same threshold-gating applies recursively at each nesting level, using the inner model's `match_threshold`.

## See Also

- [Hungarian Matching](hungarian-matching.md) -- the assignment algorithm that produces pairings
- [Classification Logic](classification-logic.md) -- full definitions of TP, FD, FA, FN, TN
- [Aggregate Metrics](aggregate-metrics.md) -- how metrics roll up through the result tree
