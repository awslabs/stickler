---
title: Hungarian Matching
---

# Hungarian Matching

When Stickler compares `List[StructuredModel]` fields, it uses the Hungarian algorithm to find the optimal one-to-one pairing between ground-truth (GT) and prediction (Pred) elements. This page walks through the algorithm and a concrete example.

## Overview

The algorithm has three phases:

1. **Pairwise similarity** -- Compute a similarity score for every GT/Pred combination.
2. **Optimal assignment** -- Use the Hungarian algorithm to find the pairing that maximizes total similarity.
3. **Threshold-gated classification** -- Classify each pair as TP or FD based on `match_threshold`, then handle unmatched items.

## Algorithm Steps

### 1. Pairwise Similarity

For each (GT[i], Pred[j]) pair, Stickler calls `GT[i].compare(Pred[j])` to obtain a raw weighted similarity score. Fields absent on both sides are excluded because a true negative is not match evidence. If every field is absent, the score is `1.0`. The result is an N x M cost matrix.

### 2. Hungarian Assignment

The Hungarian algorithm solves the assignment problem in O(n^3) time, producing a one-to-one mapping that maximizes total similarity. When lists differ in length, some items remain unmatched.

### 3. Threshold-Gated Classification

Each matched pair is classified using `StructuredModel.match_threshold`:

| Condition | Classification | Nested analysis? |
|-----------|---------------|-----------------|
| similarity >= match_threshold | **TP** | Yes -- recurse into fields |
| similarity < match_threshold | **FD** | No -- treated as atomic |
| GT item unmatched | **FN** | No |
| Pred item unmatched | **FA** | No |

The threshold splits matched pairs into TP and FD; it does not un-match them.
A pair the algorithm assigned is a match, so **similarity magnitude never
changes the classification** -- a pair at similarity `0.0` is still an assigned
pair and is therefore FD, not FN + FA. Only items with no partner at all
become FN or FA. This holds identically for a one-item list and a hundred-item
one; see `tests/common/algorithms/test_hungarian_path_parity.py`.

`HungarianMatcher.calculate_metrics` returns these same four counts if you call
it directly. Before
[#231](https://github.com/awslabs/stickler/issues/231) it did not have an `fd`
key and reported every low score pair in `fn` as well, which contradicted the
rule above. `tests/common/algorithms/test_hungarian_fd_contract.py` now pins the
agreement. Its `recall` key, however, follows the `recall_with_fd=True` row of
the table below rather than the default, so do not recompute it from the `fn` in
the same dict.

### FD and recall

Whether an FD counts against recall is a separate decision, controlled by
`recall_with_fd` (see
[Understanding Results](../Guides/Evaluation/understanding-results.md)).

This is worth understanding before comparing recall across versions or
thresholds, because **the default excludes FD from the recall denominator**:

| `recall_with_fd` | recall | effect |
|---|---|---|
| `False` (default) | `TP / (TP + FN)` | an FD is invisible to recall |
| `True` | `TP / (TP + FN + FD)` | an FD counts as a miss |

A wrong-but-paired prediction is an FD, not an FN, so under the default it is
absent from the recall denominator entirely. The effect shows up when a
document mixes right and wrong fields: three single-item list fields with one
wholly wrong scores recall `0.667` if that field is FN + FA, but `1.000` if it
is an FD, because the FD leaves both numerator and denominator untouched.
Precision is unaffected either way, since FD is part of FP.

Two different moves are easy to conflate here, and they push recall in
opposite directions:

| move | numerator | denominator | default recall |
|---|---|---|---|
| TP → FD (raise `match_threshold`) | `-1` | `-1` | falls, or stays equal |
| FN → FD (what changed for 1-vs-1 lists) | unchanged | `-1` | **rises** |

Raising `match_threshold` moves pairs from TP to FD, which lowers both default
recall and precision. The move that *raises* default recall is reclassifying an
unmatched item (FN) as a matched-but-below-threshold pair (FD), because that
drops a denominator term while leaving the numerator alone. That is the change
1-vs-1 lists saw in
[#224](https://github.com/awslabs/stickler/issues/224).

Either way an FD is invisible to default recall, which is the reason to set
`recall_with_fd=True` if you track recall over time.

## Concrete Example: Transaction Matching

### Model Definition

```python
class Transaction(StructuredModel):
    transaction_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=3.0
    )
    description: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )
    amount: float = ComparableField(threshold=0.9, weight=1.0)

    match_threshold = 0.8  # Controls Hungarian recursion gating

class Account(StructuredModel):
    account_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )
    transactions: List[Transaction] = ComparableField(weight=3.0)
```

### Test Data

**Ground Truth:**

| Index | transaction_id | description | amount |
|-------|---------------|-------------|--------|
| 0 | TXN-001 | Coffee shop payment | 4.95 |
| 1 | TXN-002 | Grocery store | 127.43 |
| 2 | TXN-003 | Gas station | 45.67 |

**Prediction:**

| Index | transaction_id | description | amount |
|-------|---------------|-------------|--------|
| 0 | TXN-001 | Coffee shop | 4.95 |
| 1 | TXN-002 | Online purchase | 89.99 |
| 2 | TXN-004 | Restaurant | 23.45 |

### Step 1: Pairwise Similarity

| GT | Pred | Similarity | >= 0.8? |
|----|------|-----------|---------|
| GT[0] | Pred[0] | 0.860 | Yes |
| GT[0] | Pred[1] | 0.137 | No |
| GT[0] | Pred[2] | 0.154 | No |
| GT[1] | Pred[0] | 0.130 | No |
| GT[1] | Pred[1] | 0.572 | No |
| GT[1] | Pred[2] | 0.135 | No |
| GT[2] | Pred[0] | 0.097 | No |
| GT[2] | Pred[1] | 0.056 | No |
| GT[2] | Pred[2] | 0.124 | No |

### Step 2: Optimal Assignment

The Hungarian algorithm produces:

- GT[0] -> Pred[0]: 0.860
- GT[1] -> Pred[1]: 0.572
- GT[2] -> Pred[2]: 0.124

### Step 3: Classification

| Pair | Similarity | vs. threshold (0.8) | Classification | Nested analysis |
|------|-----------|---------------------|---------------|----------------|
| GT[0] -> Pred[0] | 0.860 | Above | **TP** | Yes |
| GT[1] -> Pred[1] | 0.572 | Below | **FD** | No |
| GT[2] -> Pred[2] | 0.124 | Below | **FD** | No |

**Result:** TP=1, FD=2, FN=0, FA=0

Because the lists are equal length, every element gets paired. Only the TP pair (GT[0] -> Pred[0]) receives field-level analysis; the FD pairs are treated as atomic mismatches.

### Result Structure

```python
"transactions": {
    "overall": {
        "tp": 1, "fd": 2, "fa": 0, "fn": 0, "fp": 2
    },
    "fields": {
        # Only from the TP pair (GT[0] -> Pred[0])
        "transaction_id": {"tp": 1, "fd": 0, "fa": 0, "fn": 0},
        "description":    {"tp": 1, "fd": 0, "fa": 0, "fn": 0},
        "amount":         {"tp": 1, "fd": 0, "fa": 0, "fn": 0}
    }
}
```

## Key Architectural Principles

### Threshold Source

The threshold for Hungarian classification comes from the element model's `match_threshold` class attribute (default 0.7). It is **not** taken from the `ComparableField` on the parent list field.

### Metric Separation

- **Object-level metrics** count whole objects (TP=1 means one matched object, not three matched fields).
- **Field-level metrics** count individual field comparisons within TP-matched objects only.

### Recursion Gating

```
IF object_similarity >= match_threshold:
    classification = TP
    recurse into nested field analysis
ELSE:
    classification = FD
    stop (treat as atomic)
```

### Equal vs. Unequal Length Lists

- **Equal length** -- Every element gets paired. Only TP and FD are possible.
- **Unequal length** -- Extra GT items become FN; extra Pred items become FA.

## See Also

- [Classification Logic](classification-logic.md) -- full definitions of TP, FD, FA, FN, TN
- [How Below-Threshold Pairs Are Classified](threshold-gated-evaluation.md) -- the recursive evaluation model in detail
