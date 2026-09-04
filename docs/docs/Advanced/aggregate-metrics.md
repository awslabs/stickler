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

The two nodes are the two stages of the evaluation:

- **`overall` is detection.** The unit is the object. Did we find the right things? Five line items paired, none spurious.
- **`aggregate` is extraction.** The unit is the leaf. Among the objects established to be the same object, how many field values were correct? 29 of 30.

`match_threshold` is the handoff, and it is really the definition of "the same object". Above it, the pair is the same thing, so grading its fields is meaningful. Below it, it is not the same thing, so grading its fields would be scoring the fields of a *different* object. Such an object is classified as a single false discovery and is not descended into.

This is the same two-stage structure as mean Average Precision, which Stickler also implements for bounding boxes: an IoU threshold decides whether a detection matched, and only matched pairs are evaluated further. See [Bounding Box mAP Metrics](bbox-map-metrics.md#iou-thresholds), where a below-threshold detection is likewise a failure at the matching stage rather than a source of per-attribute errors. Nobody expects an unmatched detection to contribute attribute-level accuracy, and the reasoning for objects is the same.

The two paths differ on recall, though. A below-threshold detection counts as both FP and FN, so mAP recall falls; a below-threshold object is an `fd` only, so `overall` recall still reads `1.0` on a document with a spurious pairing. Pass `recall_with_fd=True` to `compare_with()` for the mAP convention.

Five line items of six fields each, one field of one item wrong. That item scores 5/6, clears a 0.7 threshold, and is comparable:

```
overall_score   0.9667

cm['overall']     tp=5   fd=0   P=1.0000  R=1.0000  F1=1.0000
cm['aggregate']   tp=29  fd=1   P=0.9667  R=1.0000  F1=0.9831
```

Five comparable objects and no spurious pairings, which is what `overall` measures. The wrong field is one of 30 leaves, which is what `aggregate` measures. Both numbers are correct for their own question.

Drop the same item below the threshold (two fields wrong, so 4/6) and it becomes a rejected object. The nodes do not converge, they diverge further:

```
overall_score   0.9333

cm['overall']     tp=4   fd=1   P=0.8000  R=1.0000  F1=0.8889
cm['aggregate']   tp=24  fd=0   P=1.0000  R=1.0000  F1=1.0000
```

`aggregate` now reports a flawless `P=1.0000` precisely because the rejected item contributes no leaf rows: 24 leaves from the four accepted items, all correct. Reading `aggregate` alone here is the same trap as reading `overall` alone one level up.

The two nodes coincide only where there is no accepted subtree to expand at all: a model with no nesting, or a document in which *every* subtree was rejected. Reject all five items and both nodes read `tp=0 fd=5`.

#### Getting leaf detail for a marginal object

`match_threshold` controls how much leaf detail you get. Lower it so the object qualifies as comparable, and its leaves are scored individually. Same five items, with the third still at 4/6:

```
match_threshold   comparable?   overall            aggregate
0.70              no            tp=4 fd=1          tp=24 fd=0
0.66              yes           tp=5 fd=0          tp=28 fd=2
```

At `0.66` the marginal item's six leaves join the other 24, and the two wrong ones finally appear as `fd`.

#### Asking whether anything failed

Because the nodes scope different things, a complete check reads both:

```python
clean = (
    cm['aggregate']['fp'] + cm['aggregate']['fn'] == 0
    and cm['overall']['fp'] + cm['overall']['fn'] == 0
)
```

Sum `fp` rather than `fa + fd`. `FP = FA + FD` by construction, so the two are equivalent today, and reading `fp` cannot go stale if a class is ever added.

Both nodes carry `fa`. A value invented where the ground truth is null is `fa` at that leaf and rolls into `aggregate`, leaving `overall` clean because the item still paired; an item invented wholesale is `fa` on `overall`. A check that names only one node reports clean on a hallucinated value:

```
one item, ground truth total=None, prediction total="INVENTED", five other leaves exact

overall     tp=1  fp=0  fn=0  fa=0  fd=0
aggregate   tp=5  fp=1  fn=0  fa=1  fd=0
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
