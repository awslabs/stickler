---
title: Aggregate Metrics
---

# Aggregate Metrics

Stickler automatically includes an `aggregate` field at every node in the confusion-matrix result tree. This provides a hierarchical rollup of all primitive-field metrics below each node, without any per-field configuration.

## Key Features

- **Automatic** -- Every node gets an `aggregate` field. No `aggregate=True` parameter needed.
- **Hierarchical** -- Parent nodes sum metrics from all child primitive fields.
- **Consistent** -- The same access pattern works at every level: `result['confusion_matrix']['aggregate']` or `result['confusion_matrix']['fields']['contact']['aggregate']`.
- **Derived metrics included** -- Each aggregate contains precision, recall, F1, and accuracy.

## Example 1: Primitive + List of Primitives + Nested Structure

This example covers three node types in one model: a primitive field (`name`), a list of primitives (`tags`), and a nested `StructuredModel` (`contact`).

```python
from typing import List
from stickler import StructuredModel, ComparableField
from stickler.comparators.exact import ExactComparator

class Contact(StructuredModel):
    phone: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    email: str = ComparableField(comparator=ExactComparator(), threshold=1.0)

class Person(StructuredModel):
    name: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    tags: List[str] = ComparableField(comparator=ExactComparator(), threshold=1.0)
    contact: Contact = ComparableField(comparator=ExactComparator(), threshold=1.0)

gt = Person(name="John", tags=["vip", "active", "premium"],
            contact=Contact(phone="123", email="john@test.com"))
pred = Person(name="John", tags=["vip", "premium"],
              contact=Contact(phone="456", email="john@test.com"))

result = gt.compare_with(pred, include_confusion_matrix=True)
cm = result['confusion_matrix']

# Top-level aggregate (all primitive fields across the entire model)
print(cm['aggregate'])

# Tags aggregate (list-of-primitives -- aggregate equals overall)
print(cm['fields']['tags']['aggregate'])

# Contact-level aggregate (phone + email)
print(cm['fields']['contact']['aggregate'])
```

### Output Structure

```json
{
  "confusion_matrix": {
    "overall": {
      "tp": 3, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 1,
      "derived": { "cm_precision": 0.75, "cm_recall": 0.75, "cm_f1": 0.75 }
    },
    "aggregate": {
      "tp": 4, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 1,
      "derived": { "cm_precision": 0.8, "cm_recall": 0.8, "cm_f1": 0.8 }
    },
    "fields": {
      "name": {
        "overall":   { "tp": 1, "fd": 0, "fa": 0, "fn": 0, "tn": 0 },
        "aggregate": { "tp": 1, "fd": 0, "fa": 0, "fn": 0, "tn": 0 }
      },
      "tags": {
        "overall":   { "tp": 2, "fd": 0, "fa": 0, "fn": 1, "tn": 0 },
        "aggregate": { "tp": 2, "fd": 0, "fa": 0, "fn": 1, "tn": 0 },
        "fields": {}
      },
      "contact": {
        "overall":   { "tp": 0, "fd": 1, "fa": 0, "fn": 0, "tn": 0 },
        "aggregate": { "tp": 1, "fd": 1, "fa": 0, "fn": 0, "tn": 0 },
        "fields": {
          "phone": {
            "overall":   { "tp": 0, "fd": 1, "fa": 0, "fn": 0, "tn": 0 },
            "aggregate": { "tp": 0, "fd": 1, "fa": 0, "fn": 0, "tn": 0 }
          },
          "email": {
            "overall":   { "tp": 1, "fd": 0, "fa": 0, "fn": 0, "tn": 0 },
            "aggregate": { "tp": 1, "fd": 0, "fa": 0, "fn": 0, "tn": 0 }
          }
        }
      }
    }
  }
}
```

Key observations:

- `name` is a primitive leaf -- `aggregate` equals `overall`.
- `tags` is a `List[str]` -- Hungarian matching produces 2 TP ("vip", "premium") and 1 FN ("active" has no pred counterpart). `aggregate` equals `overall` because it's a leaf.
- `contact` is a nested structure -- `overall` is FD (phone mismatch), but `aggregate` sums the child fields (1 TP from email + 1 FD from phone).
- The top-level `aggregate` sums all four leaf-level counts: name(1 TP) + tags(2 TP, 1 FN) + phone(1 FD) + email(1 TP) = 4 TP, 1 FD, 1 FN.

Note the difference between `overall` and `aggregate`:

- **`overall`** reflects this node's own direct classification (e.g., was this object a TP or FD?).
- **`aggregate`** sums all leaf-level classifications beneath this node (including itself if it is a leaf).

## Example 2: List of StructuredModel -- FD Recursion and Unmatched Items

This example illustrates two important behaviors:

1. An object pair classified as FD (below `match_threshold`) still has its fields recursed for aggregate metrics.
2. An unmatched GT item (FN) contributes its populated fields to the aggregate.

```python
from typing import List
from stickler import StructuredModel, ComparableField
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator

class LineItem(StructuredModel):
    match_threshold = 0.6
    sku: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=2.0)
    description: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    qty: int = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)

class Invoice(StructuredModel):
    invoice_id: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    items: List[LineItem] = ComparableField(weight=1.0)

gt = Invoice(
    invoice_id="INV-001",
    items=[
        LineItem(sku="AAA", description="Widget", qty=10),
        LineItem(sku="BBB", description="Gadget", qty=5),
        LineItem(sku="CCC", description="Cable", qty=2),   # no pred counterpart
    ],
)
pred = Invoice(
    invoice_id="INV-001",
    items=[
        LineItem(sku="AAA", description="Widget", qty=10),            # TP (similarity 1.0)
        LineItem(sku="BBB", description="Completely Wrong", qty=99),  # FD (similarity 0.53)
    ],
)

result = gt.compare_with(pred, include_confusion_matrix=True)
cm = result['confusion_matrix']

# Object-level: 1 TP, 1 FD, 1 FN
print(cm['fields']['items']['overall'])

# Aggregate still recurses into FD and FN fields
print(cm['fields']['items']['aggregate'])
```

### Output Structure

```json
{
  "confusion_matrix": {
    "overall": {
      "tp": 2, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 1,
      "derived": { "cm_precision": 0.67, "cm_recall": 0.67, "cm_f1": 0.67 }
    },
    "aggregate": {
      "tp": 5, "fa": 0, "fd": 2, "fp": 2, "tn": 0, "fn": 3,
      "derived": { "cm_precision": 0.71, "cm_recall": 0.63, "cm_f1": 0.67 }
    },
    "fields": {
      "invoice_id": {
        "overall":   { "tp": 1, "fd": 0, "fa": 0, "fn": 0, "tn": 0 },
        "aggregate": { "tp": 1, "fd": 0, "fa": 0, "fn": 0, "tn": 0 }
      },
      "items": {
        "overall": {
          "tp": 1, "fd": 1, "fa": 0, "fn": 1,
          "derived": { "cm_precision": 0.5, "cm_recall": 0.5, "cm_f1": 0.5 }
        },
        "aggregate": {
          "tp": 4, "fd": 2, "fa": 0, "fn": 3,
          "derived": { "cm_precision": 0.67, "cm_recall": 0.57, "cm_f1": 0.62 }
        },
        "fields": {
          "sku":         { "overall": { "tp": 2, "fd": 0, "fn": 1 },
                           "aggregate": { "tp": 2, "fd": 0, "fn": 1 } },
          "description": { "overall": { "tp": 1, "fd": 1, "fn": 1 },
                           "aggregate": { "tp": 1, "fd": 1, "fn": 1 } },
          "qty":         { "overall": { "tp": 1, "fd": 1, "fn": 1 },
                           "aggregate": { "tp": 1, "fd": 1, "fn": 1 } }
        }
      }
    }
  }
}
```

Key observations:

- `items.overall` has 1 TP (AAA pair, similarity 1.0 >= 0.6), 1 FD (BBB pair, similarity 0.53 < 0.6), and 1 FN (CCC, unmatched in pred). These are object-level counts.
- `items.aggregate` recurses into all three items' fields regardless of the threshold outcome:
    - AAA pair (TP): sku TP, description TP, qty TP → 3 TP
    - BBB pair (FD): sku TP, description FD, qty FD → 1 TP + 2 FD
    - CCC (FN, unmatched): each populated field counts as FN → 3 FN
- The threshold gates only the object-level classification. Aggregate metrics always drill down to the leaf fields.

## Node Types and Aggregation Behavior

Stickler's comparison tree is built from four distinct node types. The node type determines how metrics are computed and how `overall` and `aggregate` relate at each level.

### 1. Primitive (`str`, `int`, `float`)

Leaf node. `aggregate` equals `overall`. The field is compared directly and classified as TP, FD, FA, FN, or TN.

### 2. List of Primitives (`List[str]`, `List[int]`)

Also a leaf from the aggregate tree's perspective. Elements are matched via the [Hungarian algorithm](hungarian-matching.md) and each element-level classification (TP/FD/FA/FN) rolls into `overall`. The result has an empty `fields` dict, so `aggregate` equals `overall`.

### 3. Nested StructuredModel (e.g., `contact: Contact`)

Parent node. The `overall` reflects the object-level classification of the nested model as a whole. `aggregate` is the sum of all child field aggregates within the nested model — it recurses into the child model's fields.

### 4. List of StructuredModel (`List[Product]`)

Also a parent node and the most complex case. [Threshold-gating](threshold-gated-evaluation.md) controls the object-level classification, but aggregate metrics always recurse through nested fields to the leaf nodes regardless of the threshold outcome.

- **`overall`**: Object-level counts — one TP/FD/FA/FN per list item, determined by Hungarian matching against `match_threshold`. The threshold gates this classification only.
- **`fields`**: Per-sub-field metrics aggregated across all matched and unmatched items. Every pair (TP, FD) and every unmatched item (FN, FA) is recursed into for aggregate purposes — this recursion does not affect object-level metrics.
- **`aggregate`**: Sum of child field aggregates from the `fields` dict.

Within each pair, sub-fields are dispatched by their own type — primitives are classified directly, nested `List[StructuredModel]` fields recurse again with the inner model's `match_threshold`, and so on to arbitrary depth.

Matched and unmatched items contribute to aggregate metrics differently. For matched pairs (TP or FD), every child field is fully evaluated whether populated or not — both-null fields produce a TN, mismatches produce FD, etc. For unmatched items (FN or FA), only populated fields are counted: each non-null field on an unmatched GT item counts as FN, each non-null field on an unmatched Pred item counts as FA. Null fields on unmatched items are skipped entirely and do not produce a TN. This avoids inflating the TN count when a long predicted list contains mostly-empty objects.

## Calculation Summary

1. **Leaf nodes** (primitives and primitive lists): `aggregate` equals `overall`.
2. **Parent nodes** (nested models and structured lists): `aggregate` is the sum of all child `aggregate` values.
3. **Derived metrics**: Precision, recall, F1, and accuracy are recomputed at each level from the summed counts.

## Hierarchical Reporting Example

```python
def print_metrics(node, path=""):
    if 'aggregate' in node:
        a = node['aggregate']
        p = a.get('derived', {}).get('cm_precision', 0)
        r = a.get('derived', {}).get('cm_recall', 0)
        f1 = a.get('derived', {}).get('cm_f1', 0)
        print(f"{path or 'root'}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    for name, child in node.get('fields', {}).items():
        print_metrics(child, f"{path}.{name}" if path else name)

result = gt.compare_with(pred, include_confusion_matrix=True)
print_metrics(result['confusion_matrix'])
```

## See Also

- [Classification Logic](classification-logic.md) -- definitions of TP, FD, FA, FN, TN
- [Threshold-Gated Evaluation](threshold-gated-evaluation.md) -- how list comparisons feed into aggregation
