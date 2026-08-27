---
title: Aggregate Metrics
---

# Aggregate Metrics

Stickler automatically includes an `aggregate` field at every node in the confusion-matrix result tree. This provides a hierarchical rollup of all primitive-field metrics below each node, without any per-field configuration.

## Key Features

- **Automatic** -- Every node gets an `aggregate` field, with no per-field configuration.
- **Hierarchical** -- Parent nodes sum metrics from all child primitive fields.
- **Consistent** -- The same access pattern works at every level: `result['confusion_matrix']['aggregate']` or `result['confusion_matrix']['fields']['contact']['aggregate']`.
- **Derived metrics included** -- Each aggregate contains precision, recall, F1, and accuracy.

## Usage

```python
from stickler import StructuredModel, ComparableField
from stickler import ExactComparator

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

### Which node answers which question

`overall` reports **object verdicts**: were these two objects comparable, or was the pairing spurious. `aggregate` reports **leaf detail for the objects that were comparable**. `match_threshold` is the line between the two.

An object scoring below `match_threshold` is classified as a single false discovery and is not descended into. Its leaves are not enumerated, because reporting the parts of an object already rejected as a whole would be scoring something declared not comparable.

Five line items of six fields each, one field of one item wrong. That item scores 5/6, clears a 0.7 threshold, and is comparable:

```
overall_score   0.9667

cm['overall']     tp=5   fd=0   P=1.0000  R=1.0000  F1=1.0000
cm['aggregate']   tp=29  fd=1   P=0.9667  R=1.0000  F1=0.9831
```

Five comparable objects and no spurious pairings, which is what `overall` measures. The wrong field is one of 30 leaves, which is what `aggregate` measures. Both numbers are correct for their own question.

Drop the same item below the threshold (two fields wrong, so 4/6) and it becomes a rejected object:

```
cm['overall']     tp=0   fd=1
cm['aggregate']   tp=0   fd=1
```

The two nodes now agree, because there is no accepted subtree to expand. They coincide in exactly two situations: a model with no nesting, and a subtree that was rejected outright.

#### Getting leaf detail for a marginal object

`match_threshold` controls how much leaf detail you get. Lower it so the object qualifies as comparable, and its leaves are scored individually:

```
match_threshold   comparable?   overall            aggregate
0.70              no            tp=0 fd=1          tp=0 fd=1
0.66              yes           tp=1 fd=0          tp=4 fd=2
```

#### Asking whether anything failed

Because the nodes scope different things, a complete check reads both. `overall` is also the only node carrying `fa`, since an invented field corresponds to no ground truth leaf:

```python
clean = (
    cm['aggregate']['fd'] + cm['aggregate']['fn'] == 0
    and cm['overall']['fd'] + cm['overall']['fn'] + cm['overall']['fa'] == 0
)
```

The `overall` name predates the aggregate rollup and reads as "the whole document" when it means "object verdicts at this level". Renaming is breaking, so it is under consideration for 1.0 in [#288](https://github.com/awslabs/stickler/issues/288).

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
- [How Below-Threshold Pairs Are Classified](threshold-gated-evaluation.md) -- how list comparisons feed into aggregation
