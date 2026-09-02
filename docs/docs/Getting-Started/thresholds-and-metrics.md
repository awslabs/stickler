---
title: Thresholds and Metrics
---

# Thresholds and Metrics

This guide explains the four thresholds in Stickler, how they interact, and provides a glossary of confusion-matrix metrics with worked examples.

## The Four Thresholds

Stickler has four distinct thresholds that control different aspects of evaluation. Understanding what each one does—and which takes precedence—is essential for interpreting results correctly.

### 1. Comparator Threshold

**What it is:** The threshold built into a comparator instance.

```python
comparator = LevenshteinComparator(threshold=0.8)
```

**What it gates:** The comparator's `binary_compare()` method uses this to return `(1, 0)` for TP or `(0, 1)` for FP.

**When it applies:** Only when calling `comparator.binary_compare()` directly. In normal evaluation, this threshold is **ignored**—the field threshold is used instead.

### 2. Field Threshold

**What it is:** The threshold set on a `ComparableField`.

```python
class Invoice(StructuredModel):
    vendor_name: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.85,  # <-- Field threshold
    )
```

**What it gates:**

- **TP vs FD classification:** If `similarity >= threshold`, the field is a True Positive. Otherwise, it's a False Discovery.
- **Score clipping:** When `clip_under_threshold=True` (the default), scores below the threshold are zeroed out in the weighted average.

**Default:** `0.5` when you write `ComparableField(...)` and omit `threshold`.

**When you name no comparator at all,** the threshold is not `0.5`. It is inferred
alongside the comparator, from the field's type and name, and it is tuned to
whatever was chosen: exact matching demands `1.0`, loose text matching accepts
`0.85`. See [Unspecified fields](#unspecified-fields) below.

### 3. Model Match Threshold

**What it is:** A class-level attribute on a `StructuredModel`.

```python
class LineItem(StructuredModel):
    match_threshold = 0.8  # <-- Model match threshold

    sku: str = ComparableField(comparator=ExactComparator())
    description: str = ComparableField(threshold=0.7)
```

**What it gates:**

- **Hungarian matching classification:** When comparing `List[StructuredModel]`, the Hungarian algorithm pairs ground-truth and prediction objects. Each pair's overall similarity is compared against `match_threshold` to classify as TP or FD.
- **Recursive evaluation:** Only pairs meeting the threshold get field-by-field breakdown. Below-threshold pairs are treated as atomic FD.

**Default:** `0.7`

### 4. Runtime Match Threshold

**What it is:** The `match_threshold` parameter passed to `evaluate()` or `EvalSpec`.

```python
result = stickler.evaluate(gt, pred, match_threshold=0.8)

# Or via EvalSpec
spec = stickler.eval_for(Invoice, match_threshold=0.8)
```

**What it gates:** Same as Model Match Threshold, but applied at runtime. Overrides the class-level `match_threshold` for this evaluation.

**Default:** `0.7`

## Unspecified fields

If you tell Stickler how to compare a field, it does what you said. If you
don't, Stickler picks for you, by looking at the field's type and its name.

There is one rule, and it does not depend on how you built the model. All four
of these say the same thing, so all four get the same answer:

```python
sku: str                                      # plain annotation
sku: str = ComparableField()                  # no comparator named
{"sku": {"type": "str"}}                      # JSON config, no comparator key
{"sku": {"type": "string"}}                   # JSON Schema, no extension
```

Stickler reads the type first, then the name. A number is compared as a number.
A date as a date. A true/false exactly. Text loosely, so a small typo still
counts. But text in a field *named* like an identifier, `sku`, `invoice_id`,
`code`, is compared exactly, because being one character off on an ID is being
wrong, not being close.

Each choice carries a threshold suited to it, which is why there is no single
default to quote. `auto/README.md` holds the full table.

To see what was chosen for a given model:

```python
stickler.eval_for(MyModel).explain()
```

That reports the comparator, the threshold, and why each was picked. Nothing is
decided invisibly.

Inference is a default, not a verdict. Stickler guesses from a name and a type
and cannot know your data, so if `reference` is a fuzzy human-entered string in
your world, declare it explicitly.

## Threshold Precedence

When multiple thresholds could apply, here's which one wins:

| Situation | Which threshold applies |
|-----------|------------------------|
| **Primitive field comparison** (str, int, float) | Field threshold (`ComparableField(threshold=...)`) |
| **Primitive field with no comparator named** | Inferred threshold, from the field's type and name |
| **Nested object comparison** | Recurses into fields; each field uses its own field threshold |
| **List element pairing** (`List[StructuredModel]`) | Runtime match threshold if provided, else Model match threshold |
| **Score clipping** | Field threshold (when `clip_under_threshold=True`) |
| **Binary classification (TP/FD)** for primitives | Field threshold |
| **Binary classification (TP/FD)** for list pairs | Match threshold (runtime or model) |

### Key Rules

1. **Field threshold** controls primitive-level TP/FD classification and score clipping.
2. **Match threshold** controls object-level TP/FD classification for list matching.
3. **Runtime match threshold** overrides model-level match threshold.
4. **Comparator threshold** is only used by `binary_compare()`; evaluation ignores it.
5. **`match_threshold` never sets a field's threshold.** It answers one question:
   are these two objects in a list the same object? Whether a field matched is a
   different question, and a single number cannot mean both. To make one field
   lenient, say so on that field: `ComparableField(threshold=0.6)`.

### Example: All Four in Action

```python
from stickler import StructuredModel, ComparableField, LevenshteinComparator

class Product(StructuredModel):
    match_threshold = 0.75  # (3) Model match threshold

    name: str = ComparableField(
        comparator=LevenshteinComparator(threshold=0.5),  # (1) Comparator threshold - IGNORED
        threshold=0.8,  # (2) Field threshold - USED
    )

class Order(StructuredModel):
    products: List[Product] = ComparableField(weight=2.0)

# (4) Runtime match threshold - overrides Product.match_threshold
result = stickler.evaluate(gt_order, pred_order, match_threshold=0.8)
```

In this example:

- `name` similarity is compared against **0.8** (field threshold) for TP/FD
- Product pairs are compared against **0.8** (runtime threshold) for list matching
- The comparator's **0.5** is never used

---

## Metrics Glossary

Stickler uses a five-category confusion matrix that splits False Positives into two meaningful subcategories.

### Base Categories

| Category | Abbr | Definition | Intuition |
|----------|------|------------|-----------|
| **True Positive** | TP | Both GT and Pred are non-null and match above threshold | Correct prediction |
| **True Negative** | TN | Both GT and Pred are null | Correctly absent |
| **False Negative** | FN | GT is non-null, Pred is null | Missing prediction |
| **False Alarm** | FA | GT is null, Pred is non-null | Spurious prediction |
| **False Discovery** | FD | Both non-null but similarity < threshold | Wrong prediction |

### The FP Split

Traditional confusion matrices have a single False Positive (FP) category. Stickler splits this into **FA** (False Alarm) and **FD** (False Discovery):

```
FP = FA + FD
```

**Why the split?** FA and FD represent different failure modes:

- **FA** (False Alarm): The model hallucinated a value where none should exist.
- **FD** (False Discovery): The model produced a value, but it was wrong.

These require different remediation strategies, so Stickler tracks them separately.

### Derived Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | `TP / (TP + FA + FD)` | Of all predictions made, how many were correct? |
| **Recall** | `TP / (TP + FN)` | Of all ground-truth values, how many were found? |
| **F1 Score** | `2 × P × R / (P + R)` | Harmonic mean of precision and recall |
| **Accuracy** | `(TP + TN) / Total` | Overall correctness rate |

### The `recall_with_fd` Option

Standard recall only penalizes missing predictions (FN). But what if the model produces a value that's *wrong* (FD)? Should that count against recall?

**`recall_with_fd=True`** changes the recall formula:

```
Standard:      Recall = TP / (TP + FN)
With FD:       Recall = TP / (TP + FN + FD)
```

**When to use it:**

- **Use standard recall** when you want to measure coverage—did the model attempt to extract the field?
- **Use `recall_with_fd=True`** when wrong values are as bad as missing values for your use case.

```python
result = gt.compare_with(pred, recall_with_fd=True)
```

#### Two moves that push recall in opposite directions

Because the default excludes FD from the recall denominator, *becoming* an FD
affects recall differently depending on what the pair was before. These two are
easy to conflate, and they go opposite ways:

| move | numerator | denominator | default recall |
|---|---|---|---|
| TP → FD (you raised `match_threshold`) | `-1` | `-1` | falls, or stays equal |
| FN → FD (the pair got matched instead of missed) | unchanged | `-1` | **rises** |

A TP that becomes an FD loses a correct answer, so recall falls. An FN that
becomes an FD was never in the numerator to begin with, so removing it from the
denominator only makes the remaining hits look better.

The practical consequence: **raising `match_threshold` does not reliably lower
reported recall.** Checked exhaustively over `tp ∈ 1..4` and `fn ∈ 0..3`, a
TP → FD move raised recall in 0 cases, left it equal in 6, and lowered it in 34.
But an FN → FD move raises it. If you track recall across releases or across
threshold changes, set `recall_with_fd=True` so both moves count against you and
the number means one thing.

---

## The Zero-Threshold Trap

Setting a `threshold` or `match_threshold` to exactly `0.0` reports perfect
scores for wholly incorrect output. Stickler emits a `UserWarning` when it sees
one.

The threshold test is `>=`, so `0.0` is satisfied by **every** score, including
`0.0` itself. Every compared pair becomes a true positive:

```python
class Doc(StructuredModel):
    match_threshold = 0.0          # UserWarning
    name: str

gt   = Doc(name="Acme Corporation")
pred = Doc(name="totally wrong")

result = gt.compare_with(pred, include_confusion_matrix=True)
# precision 1.0, recall 1.0, f1 1.0 -- for an answer that is entirely wrong
```

Nothing errors and the numbers look ideal, which makes this the hardest
misconfiguration to notice.

**It is a cliff, not a slope.** `0.01` classifies correctly; only exactly `0.0`
misbehaves. This is one broken value rather than a "low thresholds are risky"
heuristic, which is why the warning fires only for `0.0`.

What is genuinely invariant at `0.0` is that **no false discovery can ever be
reported**, since FD means "compared and scored below threshold" and nothing
scores below `0.0`. Perfect precision and recall are *not* invariant, and it is
worth knowing why, because it is easy to over-claim here. Unmatched items are not
subject to any threshold, so:

- 2 ground-truth objects against 3 predictions still yields an FA, giving
  precision `0.667`
- 2 against 1 still yields an FN, giving recall `0.5`

**`match_threshold = 0.0` warns unconditionally**, because a model that is not a
list element today may become one tomorrow, and the warning is cheaper than the
silent `1.0` that follows.

It gates object pairing only. It does not set the threshold of any field, so a
standalone model with no list anywhere is unaffected by it. If you probe a
plainly annotated `name: str` and find it carrying the `match_threshold` value,
that is [#237](https://github.com/awslabs/stickler/issues/237), a defect against
the contract stated above, not the contract itself.

**Fix:** use a small positive value.

```python
match_threshold = 0.01    # accepts weak matches, still classifies correctly
```

The warning fires once per configured site, so a bulk run over many documents
does not emit one per document.

---

## Worked Example

Let's trace through a complete evaluation to see how thresholds and metrics interact.

### Setup

```python
from typing import List, Optional
from stickler import StructuredModel, ComparableField, ExactComparator, LevenshteinComparator

class LineItem(StructuredModel):
    match_threshold = 0.7

    sku: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=2.0)
    description: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.6)

class Invoice(StructuredModel):
    invoice_id: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    vendor: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
    notes: Optional[str] = ComparableField(threshold=0.6, default=None)
    line_items: List[LineItem] = ComparableField(weight=2.0)
```

### Data

```python
gt = Invoice(
    invoice_id="INV-001",
    vendor="Acme Corporation",
    notes="Net 30",
    line_items=[
        LineItem(sku="SKU-A", description="Widget Alpha"),
        LineItem(sku="SKU-B", description="Widget Beta"),
        LineItem(sku="SKU-C", description="Widget Gamma"),
    ]
)

pred = Invoice(
    invoice_id="INV-001",           # Exact match
    vendor="Acme Corp",             # 0.5625 similarity (below 0.8 threshold)
    notes=None,                     # Missing
    line_items=[
        LineItem(sku="SKU-A", description="Widget Alpha"),  # Match
        LineItem(sku="SKU-B", description="Widget Bet"),    # SKU match, desc ~0.9
        LineItem(sku="SKU-X", description="New Item"),      # Pairs with SKU-C, far below threshold
    ]
)
```

### Field-by-Field Analysis

| Field | GT Value | Pred Value | Similarity | Threshold | Classification |
|-------|----------|------------|------------|-----------|----------------|
| `invoice_id` | "INV-001" | "INV-001" | 1.0 | 1.0 | **TP** |
| `vendor` | "Acme Corporation" | "Acme Corp" | 0.5625 | 0.8 | **FD** |
| `notes` | "Net 30" | null | — | 0.6 | **FN** |

### List Matching (line_items)

Hungarian algorithm pairs:

| GT Item | Pred Item | Overall Similarity | vs `match_threshold=0.7` | Result |
|---------|-----------|-------------------|--------------------------|--------|
| SKU-A, "Widget Alpha" | SKU-A, "Widget Alpha" | 1.0 | ≥ 0.7 | **TP** → recurse |
| SKU-B, "Widget Beta" | SKU-B, "Widget Bet" | ~0.95 | ≥ 0.7 | **TP** → recurse |
| SKU-C, "Widget Gamma" | SKU-X, "New Item" | 0.083 | < 0.7 | **FD** → atomic |

Both lists hold three items, so the Hungarian algorithm pairs every one of them and
no item is left over. **FN and FA are impossible for an equal-length list** --
`SKU-C` does not go missing, it gets paired with the only prediction left, scores
`0.083`, and lands below `match_threshold` as a false discovery. Unmatched items
appear only when the two lists differ in length.

Within each matched pair that *meets* the threshold, the fields (sku, description)
are evaluated and contribute to the aggregate counts. A below-threshold pair is
treated as atomic and contributes one FD at the object level, with no field-level
breakdown -- see
[Threshold-Gated Evaluation](../Advanced/threshold-gated-evaluation.md).

### Aggregate Confusion Matrix

| Category | Count | Details |
|----------|-------|---------|
| TP | 5 | invoice_id + line_items fields from matched pairs |
| TN | 0 | — |
| FN | 1 | notes |
| FA | 0 | — |
| FD | 1 | vendor |

### Derived Metrics

```
Precision = TP / (TP + FA + FD) = 5 / (5 + 0 + 1) = 0.833
Recall    = TP / (TP + FN)      = 5 / (5 + 1)     = 0.833
F1        = 2 × 0.833 × 0.833 / (0.833 + 0.833)   = 0.833
Accuracy  = (TP + TN) / Total   = 5 / 7           = 0.714

Recall (with FD) = TP / (TP + FN + FD) = 5 / (5 + 1 + 1) = 0.714
```

---

## Quick Reference

### Threshold Cheat Sheet

| Threshold | Where Set | What It Controls | Default |
|-----------|-----------|------------------|---------|
| Comparator | `Comparator(threshold=...)` | `binary_compare()` only | varies |
| Field | `ComparableField(threshold=...)` | TP/FD for primitives, clipping | 0.5 |
| Field (inferred) | nothing; no comparator named | same as above | per type and name |
| Model | `match_threshold = ...` on class | Hungarian pairing, and nothing else | 0.7 |
| Runtime | `evaluate(..., match_threshold=...)` | Overrides model threshold | 0.7 |

### Metric Formulas

| Metric | Formula |
|--------|---------|
| Precision | `TP / (TP + FA + FD)` |
| Recall | `TP / (TP + FN)` |
| Recall (with FD) | `TP / (TP + FN + FD)` |
| F1 | `2 × Precision × Recall / (Precision + Recall)` |
| FP (total) | `FA + FD` |

## See Also

- [Classification Logic](../Advanced/classification-logic.md) — detailed definitions
- [How Below-Threshold Pairs Are Classified](../Advanced/threshold-gated-evaluation.md) — the gating mechanism in detail
- [Hungarian Matching](../Advanced/hungarian-matching.md) — list pairing algorithm
