---
title: Classification Logic
---

# Classification Logic

Stickler classifies every field comparison into one of five confusion-matrix categories. These categories drive all derived metrics (precision, recall, F1) and aggregate reporting.

## Core Definitions

| Category | Abbr. | Definition |
|----------|-------|------------|
| True Positive | TP | GT and EST are both non-null and match above the threshold |
| False Alarm | FA | GT is null, EST is non-null |
| True Negative | TN | GT and EST are both null |
| False Negative | FN | GT is non-null, EST is null |
| False Discovery | FD | GT and EST are both non-null but match below the threshold |

False Alarm (FA) and False Discovery (FD) together make up the broader False Positive (FP) count:

**FP = FA + FD**

## Classification by Data Type

### Simple Values (Strings, Numbers, Booleans)

| Ground Truth | Prediction | Classification | Notes |
|--------------|------------|----------------|-------|
| `"value"` | `"value"` | TP | Exact match |
| `"value"` | `"similar"` | FD | Both non-null, below threshold |
| `"value"` | `null` | FN | Missing prediction |
| `null` | `"value"` | FA | Spurious prediction |
| `null` | `null` | TN | Correctly absent |
| `""` | `null` | TN | Empty string treated as null |

### Lists

Lists use the [Hungarian algorithm](hungarian-matching.md) for optimal element pairing.

1. **Empty lists**: `[] vs []` is TN; `[] vs [items]` produces one FA per item; `[items] vs []` produces one FN per item.
2. **Matched elements**: similarity >= threshold is TP; below threshold is FD.
3. **Unmatched elements**: leftover GT elements are FN; leftover EST elements are FA.

#### Example: Mixed Matching

```
GT  = ["red", "blue", "green"]
EST = ["red", "yellow", "orange", "blue"]
```

- "red" matches "red" -- TP
- "blue" matches "blue" -- TP
- "green" unmatched -- FN
- "yellow" unmatched -- FA
- "orange" unmatched -- FA

Result: TP=2, FA=2, FN=1, FD=0

#### Example: Below-Threshold Matches

```
GT  = ["apple", "banana", "cherry"]
EST = ["appx", "bnn", "chry"]        (threshold = 0.7)
```

All pairs match below 0.7 -- each is FD.

Result: TP=0, FA=0, FN=0, FD=3

### Nested Objects

Nested objects are evaluated recursively, field by field.

| Condition | Classification |
|-----------|---------------|
| Both have the field, similarity >= threshold | TP |
| Both have the field, similarity < threshold | FD |
| Only GT has the field | FN |
| Only EST has the field | FA |

#### Example

```
GT  = {name: "John", age: 30, address: "123 Main St"}
EST = {name: "John", age: 31, phone: "555-1234"}
```

- `name`: exact match -- TP
- `age`: both present, mismatch -- FD
- `address`: only in GT -- FN
- `phone`: only in EST -- FA

Result: TP=1, FA=1, FN=1, FD=1

## Derived Metrics

From the base counts:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Precision | TP / (TP + FP) | Fraction of predictions that are correct |
| Recall | TP / (TP + FN) | Fraction of ground-truth values found |
| F1 Score | 2 * Precision * Recall / (Precision + Recall) | Harmonic mean of precision and recall |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |

## Edge Cases

**Null vs. empty equivalence** -- Empty strings (`""`), empty lists (`[]`), and empty objects (`{}`) are treated as null. Comparing any of these with `null` yields TN.

**Threshold boundary** -- A similarity score exactly equal to the threshold counts as a match (TP).

**List order** -- Order does not matter. The Hungarian algorithm finds the optimal pairing regardless of element position.

**Nested lists** -- For `List[StructuredModel]`, the Hungarian algorithm pairs objects at the list level, then each matched pair is evaluated recursively.

**Missing vs. null fields** -- A missing field and a field explicitly set to `null` are handled the same way: if the other side has a non-null value, the result is FN or FA accordingly.

## See Also

- [Understanding Results](../Evaluation/understanding-results.md) -- interpreting the full result dictionary
- [Hungarian Matching](hungarian-matching.md) -- details on the list-pairing algorithm
