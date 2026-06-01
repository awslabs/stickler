---
title: Aggregate Metrics
---

# Aggregate Metrics

Stickler automatically includes an `aggregate` field at every node in the confusion-matrix result tree. This provides a hierarchical rollup of all primitive-field metrics below each node, without any per-field configuration.

## Why aggregates?

The default `overall` metrics answer the question *"how did the model do at this level of the object?"* For a `List[Product]`, that means one TP/FD/FA/FN per item — a five-item GT compared against five predictions yields five object-level events, each gated by `match_threshold`. That view is correct for object-level evaluation, but it hides the field-level signal: a TP item with three of four fields wrong still counts as a single TP, and an FD or unmatched item contributes nothing to per-field accuracy because the per-field counts are threshold-gated.

`aggregate` answers a different question: *"how did the model do across **every** field of **every** item, regardless of whether each item cleared the match threshold?"* The aggregate path always recurses into the nested fields of TP and FD pairs, and counts populated fields on unmatched items as FN (for missing GT) or FA (for spurious predictions). The result is a single per-field rollup at the parent level that tells you, e.g., "across this entire `List[LineItem]`, sku had 2 TP / 0 FD / 1 FN; description had 1 TP / 1 FD / 1 FN."

In short:

- Use **`overall`** when you want object-level pass/fail performance gated by `match_threshold` (e.g., "what fraction of predicted line items matched a GT line item?").
- Use **`aggregate`** when you want field-level performance rolled up across an entire list or nested structure (e.g., "what's the precision and recall of the `sku` field across all line items in this invoice?").

The two are complementary: `overall` and `aggregate` agree exactly for primitive fields and primitive-list fields (no recursion to do), and they diverge in interesting ways for `List[StructuredModel]`. The [Common pitfalls](#common-pitfalls) section near the end of this page covers the most frequent confusions.

## Key Features

- **Automatic** -- Every node gets an `aggregate` field. No `aggregate=True` parameter needed.
- **Hierarchical** -- Parent nodes sum metrics from all child primitive fields.
- **Consistent** -- The same access pattern works at every level: `result['confusion_matrix']['aggregate']` or `result['confusion_matrix']['fields']['contact']['aggregate']`.
- **Derived metrics included** -- Each aggregate contains precision, recall, F1, and accuracy.

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

- `items.overall` has 1 TP (AAA, similarity 1.0 >= 0.6), 1 FD (BBB, similarity 0.53 < 0.6), and 1 FN (CCC, unmatched). These are object-level counts.
- Even though BBB is classified as FD at the object level, its fields are still recursed for aggregate purposes. The table below shows how each item's fields contribute to `items.aggregate`:

| Item | Object classification | `sku` | `description` | `qty` |
|------|----------------------|-------|---------------|-------|
| AAA pair | TP (sim 1.0 >= 0.6) | TP (exact match) | TP (exact match) | TP (exact match) |
| BBB pair | FD (sim 0.53 < 0.6) | TP (exact match) | FD (low similarity) | FD (5 ≠ 99) |
| CCC | FN (unmatched) | FN | FN | FN |

- Summing the columns: `sku` = 2 TP + 1 FN, `description` = 1 TP + 1 FD + 1 FN, `qty` = 1 TP + 1 FD + 1 FN. Grand total across all sub-fields: 4 TP, 2 FD, 3 FN — which is exactly what `items.aggregate` reports.
- The threshold gates only the object-level classification. Aggregate metrics always drill down to the leaf fields.

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

## Common pitfalls

A handful of behaviors trip up users when they first start consuming `aggregate` metrics in earnest. Each subsection below is anchored so you can link directly to the specific footgun.

### Aggregate doesn't equal the sum of `overall` counts {#aggregate-not-sum-of-overall}

For `List[StructuredModel]` parents, `aggregate` is **not** derived from the parent's `overall` — it is a separately-accumulated rollup that recurses through every Hungarian-paired item, regardless of `match_threshold`. The object-level `overall` (TP/FD/FA/FN per item) is threshold-gated; `aggregate` pre-seeds its leaf counts from the full ungated set of pairs and then sums upward. If your numbers don't add up, this is almost always why — see the FD-recursion table in [Example 2](#example-2-list-of-structuredmodel-fd-recursion-and-unmatched-items) and the [threshold-gated drill-down explanation](threshold-gated-evaluation.md).

### Zero-similarity item pairs are unmatched, not FD {#zero-similarity-pairs}

Hungarian can return a "pairing" with similarity exactly `0.0` (e.g., two list items that share no signal at all). At the **object level**, those pairs are treated as unmatched and contribute one FN to GT plus one FA to Pred — not a single FD. Only pairs with `0.0 < similarity < match_threshold` count as FD. If you're tuning `match_threshold` and expect to see FD for very different items, you'll instead see FN+FA. See `_calculate_object_level_metrics` in `structured_list_comparator.py` (the `elif similarity > 0.0:` branch around line 205).

### Empty list comparisons {#empty-lists}

When **both** the GT and Pred lists are empty, the list field's `overall` is recorded as `tn: 1` (object-level true negative, similarity `1.0`). When **one** side is empty and the other is populated, every item on the non-empty side counts at the field level — populated GT items become FN, populated Pred items become FA. This matters in IDP scenarios where many document fields are optional: a model that hallucinates a 5-item table when GT is empty will produce 5 items' worth of FA across all sub-fields in `aggregate`, not just one object-level FA.

### Bulk evaluator does not yet aggregate the `aggregate` field {#bulk-aggregate-not-rolled-up}

`BulkStructuredModelEvaluator._accumulate_confusion_matrix` walks `cm_result["overall"]` and `cm_result["fields"]` (recursively into `fields` and `nested_fields`), but it **does not** roll up the per-document `aggregate` key into a corpus-level aggregate. Each per-document result still contains `aggregate`, but the bulk-rolled-up output exposes only `overall` and per-field metrics. If you need a corpus-level aggregate today, sum the per-document `aggregate` blocks yourself from `evaluate_pair` results. (Tracked as a follow-up.)

### `recall_with_fd` parameter {#recall-with-fd}

Derived metrics support two recall formulas, controlled by `recall_with_fd`:

```text
recall_with_fd=False (default):  TP / (TP + FN)
recall_with_fd=True:             TP / (TP + FN + FD)
```

The default matches the textbook definition and treats FDs (partial/below-threshold matches) as neither rewards nor penalties for recall. Set `recall_with_fd=True` when a soft-but-wrong prediction should still count against you — i.e., when your downstream consumer can't tell the difference between "not retrieved" and "retrieved but below quality bar." Precision, F1, and accuracy are unaffected by this flag; only `cm_recall` (and the `cm_f1` derived from it) change.

## See Also

- [Classification Logic](classification-logic.md) -- definitions of TP, FD, FA, FN, TN
- [Threshold-Gated Evaluation](threshold-gated-evaluation.md) -- how list comparisons feed into aggregation
