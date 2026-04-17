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

## Usage

```python
from stickler import StructuredModel, ComparableField
from stickler.comparators.exact import ExactComparator

class Contact(StructuredModel):
    phone: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    email: str = ComparableField(comparator=ExactComparator(), threshold=1.0)

class Person(StructuredModel):
    name: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    contact: Contact = ComparableField(comparator=ExactComparator(), threshold=1.0)

gt = Person(name="John", contact=Contact(phone="123", email="john@test.com"))
pred = Person(name="John", contact=Contact(phone="456", email="john@test.com"))

result = gt.compare_with(pred, include_confusion_matrix=True)
cm = result['confusion_matrix']

# Top-level aggregate (all primitive fields across the entire model)
print(cm['aggregate'])

# Contact-level aggregate (phone + email)
print(cm['fields']['contact']['aggregate'])
```

## Output Structure

```json
{
  "confusion_matrix": {
    "overall": {
      "tp": 1, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0,
      "derived": { "cm_precision": 0.5, "cm_recall": 1.0, "cm_f1": 0.67 }
    },
    "aggregate": {
      "tp": 2, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0,
      "derived": { "cm_precision": 0.67, "cm_recall": 1.0, "cm_f1": 0.8 }
    },
    "fields": {
      "name": {
        "overall":   { "tp": 1, "fd": 0, "fa": 0, "fn": 0, "tn": 0 },
        "aggregate": { "tp": 1, "fd": 0, "fa": 0, "fn": 0, "tn": 0 }
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

Note the difference between `overall` and `aggregate`:

- **`overall`** reflects this node's own direct classification (e.g., was this object a TP or FD?).
- **`aggregate`** sums all leaf-level classifications beneath this node (including itself if it is a leaf).

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
