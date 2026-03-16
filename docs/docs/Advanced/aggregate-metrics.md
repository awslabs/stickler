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
        "overall":   { "tp": 1, "fd": 0, "fa": 0, "fn": 0 },
        "aggregate": { "tp": 1, "fd": 0, "fa": 0, "fn": 0 }
      },
      "contact": {
        "overall":   { "tp": 0, "fd": 1, "fa": 0, "fn": 0 },
        "aggregate": { "tp": 1, "fd": 1, "fa": 0, "fn": 0 },
        "fields": {
          "phone": {
            "overall":   { "tp": 0, "fd": 1 },
            "aggregate": { "tp": 0, "fd": 1 }
          },
          "email": {
            "overall":   { "tp": 1, "fd": 0 },
            "aggregate": { "tp": 1, "fd": 0 }
          }
        }
      }
    }
  }
}
```

Note the difference between `overall` and `aggregate`:

- **`overall`** reflects this node's own direct classification.
- **`aggregate`** sums all primitive-field classifications beneath this node (including itself if it is a leaf).

## Calculation Logic

1. **Leaf nodes** (primitive fields): `aggregate` equals `overall`.
2. **Parent nodes**: `aggregate` is the sum of all child `aggregate` values.
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
