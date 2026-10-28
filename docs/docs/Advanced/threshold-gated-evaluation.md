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

Three GT products against three predictions, measured with the model above:

| GT | Pred | Pair similarity | Classification |
|----|------|-----------------|----------------|
| `("PROD-001", "Laptop", 999.99)` | `("PROD-001", "Laptop Computer", 999.99)` | `0.7999999999999999` | **FD** |
| `("PROD-002", "Mouse", 29.99)` | `("PROD-002", "Different Product", 99.99)` | `0.6725` | **FD** |
| `("PROD-003", "Cable", 14.99)` | `("PROD-004", "New Product", 19.99)` | `0.1333` | **FD** |

`products` reads `overall fd=3` with no `fields`, and none of the three is descended into.

- **The Laptop pair misses by a rounding error.** It prints as `0.800` but is just below `0.8`, and the object gate is a bare `>=` ([#417](https://github.com/awslabs/stickler/issues/417)). Its `name` similarity is only `0.4`; `product_id` and `price` carry it.
- **Equal-length lists cannot produce FN or FA.** Every element gets a partner, so `Cable` is an FD against `New Product`, not an FN plus an FA.

Lower `match_threshold` to `0.79` and drop the `PROD-004` prediction. The Laptop pair is now a TP and is descended into, the Mouse pair stays an FD, and `Cable` has no partner, so it is an FN:

- `product_id`: TP
- `name`: FD (`0.4` is below its `threshold=0.7`), a failing leaf inside an accepted item
- `price`: TP

## Result Structure

That second run, measured:

```python
cm["fields"]["products"]
{
    "overall":   {"tp": 1, "fd": 1, "fn": 1, "fa": 0, "fp": 1,
                  "derived": {"cm_precision": 0.5, "cm_recall": 0.5, "cm_f1": 0.5, ...}},
    "aggregate": {"tp": 2, "fd": 1, "fn": 0, ...},   # leaves of the TP item only
    "fields": {
        "product_id": {"overall": {"tp": 1, ...}, ...},
        "name":       {"overall": {"fd": 1, ...}, ...},
        "price":      {"overall": {"tp": 1, ...}, ...},
    },
}

result["non_matches"]
[
    {"field_path": "products[0].name", "non_match_type": "false_discovery",
     "ground_truth_value": "Laptop", "prediction_value": "Laptop Computer",
     "similarity_score": 0.4, "reason": "field mismatch"},
    {"field_path": "products[1]", "non_match_type": "false_discovery",
     "ground_truth_value": {"product_id": "PROD-002", "name": "Mouse", "price": 29.99},
     "prediction_value": {"product_id": "PROD-002", "name": "Different Product", "price": 99.99},
     "similarity_score": 0.6725, "similarity": 0.6725,
     "reason": "below threshold (0.673 < 0.79)"},
    {"field_path": "products[2]", "non_match_type": "false_negative",
     "ground_truth_value": {"product_id": "PROD-003", "name": "Cable", "price": 14.99},
     "prediction_value": None, "reason": "unmatched ground truth"},
]
```

Field-level metrics appear only for the TP pair. The `non_matches` list
documents each FD and FN item once at object level, and the TP item's failing
leaf by its own path; `field_comparisons` uses the same boundary. Neither report
expands a rejected or unmatched object into leaves. The missed `Cable` is on
`overall` only: next to an accepted sibling, `aggregate` leaves it out
([Recursion](classification-logic.md#list-contract)).

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
2. The score is optionally clipped to 0 if below the field threshold (`clip_under_threshold`). A list field applies this per element, not to its score as a whole ([Score](classification-logic.md#list-contract)).
3. Clipped scores are multiplied by the field weight and summed.
4. The overall similarity is `total_weighted_score / total_weight`.

This gating is also why asking "did anything fail" reads both rollup nodes. A
list item scoring below `match_threshold` is a spurious non-match: it is recorded
as one `fd` on `overall` and is **not descended into**, so it contributes no leaf
rows to `aggregate`. Reporting the leaves of an object already rejected as a
whole would score something the comparison declared not comparable.

```python
clean = (
    cm['aggregate']['fp'] + cm['aggregate']['fn'] == 0
    and cm['overall']['fp'] + cm['overall']['fn'] == 0
)
```

Read **both nodes**. That is the fix: the earlier version of this snippet summed
`fd` on `aggregate` but took `fa` from `overall` only, so a value invented where the
ground truth is null went unseen. Such a value is `fa` at that leaf and rolls into
`aggregate`, while `overall` stays clean because the item still paired.

`fa + fd` on a node would work just as well, since `FP = FA + FD` by construction.
`fp` is preferred only because it is one term instead of two and cannot go stale if a
class is ever added -- not because the classes are unsafe to read.

So `match_threshold` is also the knob for how much leaf detail you get, for a
**list item**. If you want a marginal item's leaves scored individually, lower it
until that item qualifies as comparable. It does nothing for a single nested
`StructuredModel` field, which is never gated and always reports its leaves; the
field's own `threshold` decides that verdict. See
[Aggregate Metrics](aggregate-metrics.md#which-node-answers-which-question).

`overall_score` is the scalar summary, and `EvalResult.matched` from
`stickler.evaluate()` is the object-level verdict
(`overall_score >= match_threshold`).

For the raw object similarity used by Hungarian matching, fields absent on both sides are omitted from both totals. They remain TNs in the confusion matrix, but do not help a pair clear `match_threshold`. If no fields remain, the similarity is defined as `1.0`, because nothing disagreed.

## Edge Cases

**Empty lists** -- `[] vs []` is TN. `[] vs [items]` generates one FA per item. `[items] vs []` generates one FN per item.

**Threshold boundary** -- `similarity >= threshold` counts as TP and triggers recursion. Values exactly at the boundary are matches. A float that prints as the threshold can sit just below it, as the Laptop pair above does ([#417](https://github.com/awslabs/stickler/issues/417)).

**Different thresholds per model** -- Each `StructuredModel` subclass can define its own `match_threshold`. A `Product` with `match_threshold = 0.8` and an `Address` with `match_threshold = 0.6` are each evaluated independently.

**Nested lists** -- When a `StructuredModel` contains another `List[StructuredModel]`, the same threshold-gating applies recursively at each nesting level, using the inner model's `match_threshold`.

## See Also

- [Thresholds and Metrics](../Getting-Started/thresholds-and-metrics.md) -- the explainer: all four thresholds, precedence, the metrics glossary, and the zero-threshold trap
- [Hungarian Matching](hungarian-matching.md) -- the assignment algorithm that produces pairings
- [Classification Logic](classification-logic.md) -- full definitions of TP, FD, FA, FN, TN
- [Aggregate Metrics](aggregate-metrics.md) -- how metrics roll up through the result tree
