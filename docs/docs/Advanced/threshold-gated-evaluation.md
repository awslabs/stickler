---
title: How Below-Threshold Pairs Are Classified
---

# How Below-Threshold Pairs Are Classified

When comparing `List[StructuredModel]` fields, Stickler only performs detailed nested-field analysis on object pairs whose overall similarity meets a configurable threshold. Pairs that fall below the threshold are classified as False Discovery (FD) and treated as atomic units -- no field-by-field breakdown is generated for them.

> **New to thresholds?** Start with
> [Thresholds and Metrics](../Getting-Started/thresholds-and-metrics.md), which
> explains all four thresholds, which one wins where, and what each confusion-matrix
> category means. This page is the mechanism reference for the gating behaviour
> itself: what happens to a pair once it falls below the threshold.

## Core Principle

**Only recurse into nested field evaluation for object pairs that meet the similarity threshold.**

This keeps metrics focused on meaningful comparisons and avoids generating misleading field-level statistics for object pairs that are fundamentally different.

The gate applies to nested confusion-matrix metrics, `non_matches`, and
`field_comparisons`. A rejected or unmatched object produces one object-level
entry in those reports, not one entry per leaf. To inspect leaves for weaker
pairs, lower the list element model's `match_threshold` to a small positive
value; this intentionally broadens which pairs count as the same object.

## Algorithm Flow

### 1. Hungarian Matching

Use the [Hungarian algorithm](hungarian-matching.md) to find optimal pairings between GT and Pred lists based on overall object similarity.

### 2. Threshold Classification

For each matched pair, compare the similarity score against `StructuredModel.match_threshold`:

- **similarity >= threshold** -- **TP**: recurse into nested fields
- **similarity < threshold** -- **FD**: stop recursion, treat as atomic

!!! warning "Do not set a list field's `threshold` to `0.0`"
    The comparison is `>=`, so a threshold of `0.0` is satisfied by *every*
    score including `0.0` itself. Every pair the algorithm assigns becomes a
    true positive, and a wholly wrong prediction reports perfect metrics:

    ```python
    class Doc(StructuredModel):
        tags: List[str] = ComparableField(comparator=ExactComparator(), threshold=0.0)

    Doc(tags=["X"]).compare_with(Doc(tags=["A"]), include_confusion_matrix=True)
    # tp=1, fa=0, fn=0 -- recall, precision, F1 and accuracy all 1.000
    ```

    Use a small positive threshold instead if the intent is "accept weak
    matches." `0.0` means "accept everything," which is rarely what a
    threshold is for. Note that Stickler uses `match_threshold=0.0`
    internally as a deliberate capture-all sentinel when it needs every pair
    for scoring, and reclassifies those pairs itself rather than reading
    their `tp`.

### 3. Unmatched Items

- **GT extras** -- **FN**: stop recursion
- **Pred extras** -- **FA**: stop recursion

## Code Example

```python
from stickler import StructuredModel, ComparableField, LevenshteinComparator, ExactComparator
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

Given three GT products and three Pred products:

### Good Match (similarity >= 0.8)

**GT:** `Product("PROD-001", "Laptop", 999.99)`
**Pred:** `Product("PROD-001", "Laptop Computer", 999.99)`

- Classification: **TP**
- Nested field analysis is performed:
    - `product_id`: TP (exact match)
    - `name`: TP (similarity ~0.9)
    - `price`: TP (exact match)

### Poor Match (similarity < 0.8)

**GT:** `Product("PROD-002", "Mouse", 29.99)`
**Pred:** `Product("PROD-002", "Different Product", 99.99)`

- Classification: **FD**
- No nested field analysis -- the objects are too dissimilar for field-level breakdown to be useful.

### Unmatched Items

**GT:** `Product("PROD-003", "Cable", 14.99)` -- **FN** (no counterpart in Pred)
**Pred:** `Product("PROD-004", "New Product", 19.99)` -- **FA** (no counterpart in GT)

No nested analysis for either.

## Result Structure

```json
{
  "products": {
    "overall": {
      "tp": 1, "fd": 1, "fn": 1, "fa": 1,
      "derived": { "cm_precision": 0.5, "cm_recall": 0.5, "cm_f1": 0.5 }
    },
    "fields": {
      "product_id": { "tp": 1 },
      "name":       { "tp": 1 },
      "price":      { "tp": 1 }
    },
    "non_matches": [
      {
        "type": "FD",
        "gt_object": "Product(PROD-002, Mouse, 29.99)",
        "pred_object": "Product(PROD-002, Different Product, 99.99)",
        "similarity": 0.3
      },
      { "type": "FN", "gt_object": "Product(PROD-003, Cable, 14.99)" },
      { "type": "FA", "pred_object": "Product(PROD-004, New Product, 19.99)" }
    ]
  }
}
```

Field-level metrics appear only for the single TP pair. The `non_matches` list
documents each FD, FN, and FA once at object level; `field_comparisons` uses the
same boundary. Neither report expands a rejected or unmatched object into
leaves.

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

To ask whether anything failed, read `confusion_matrix.aggregate.fd`, which
counts leaf comparisons that fell below their threshold at any depth.
`overall_score` is the scalar summary, and `EvalResult.matched` from
`stickler.evaluate()` is the object-level verdict
(`overall_score >= match_threshold`).

For the raw object similarity used by Hungarian matching, fields absent on both sides are omitted from both totals. They remain TNs in the confusion matrix, but do not help a pair clear `match_threshold`. If no fields remain, the similarity is defined as `1.0`, because nothing disagreed.

## Edge Cases

**Empty lists** -- `[] vs []` is TN. `[] vs [items]` generates one FA per item. `[items] vs []` generates one FN per item.

**Threshold boundary** -- `similarity >= threshold` counts as TP and triggers recursion. Values exactly at the boundary are matches.

**Different thresholds per model** -- Each `StructuredModel` subclass can define its own `match_threshold`. A `Product` with `match_threshold = 0.8` and an `Address` with `match_threshold = 0.6` are each evaluated independently.

**Nested lists** -- When a `StructuredModel` contains another `List[StructuredModel]`, the same threshold-gating applies recursively at each nesting level, using the inner model's `match_threshold`.

## See Also

- [Thresholds and Metrics](../Getting-Started/thresholds-and-metrics.md) -- the explainer: all four thresholds, precedence, the metrics glossary, and the zero-threshold trap
- [Hungarian Matching](hungarian-matching.md) -- the assignment algorithm that produces pairings
- [Classification Logic](classification-logic.md) -- full definitions of TP, FD, FA, FN, TN
- [Aggregate Metrics](aggregate-metrics.md) -- how metrics roll up through the result tree
