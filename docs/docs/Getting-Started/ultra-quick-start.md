---
title: Ultra Quick Start
---

# Ultra Quick Start

**Already have a Pydantic model? You can evaluate with it as-is, with no configuration required.**

If you have a [Pydantic](https://docs.pydantic.dev/) model (say, the `response_model` your agent produces structured output with) you do **not** need to define a `StructuredModel`, pick comparators, choose thresholds, or write a JSON schema. Hand Stickler your two objects and it picks sensible defaults for you, then shows you exactly what it chose.

```python
# pip install stickler-eval
import stickler

result = stickler.evaluate(ground_truth, prediction)

print(result.overall_score)   # weighted similarity in [0, 1]
print(result.f1)              # precision/recall/F1 are right there too
print(result.field_scores)    # per-field breakdown
```

That's the whole integration. `ground_truth` and `prediction` are ordinary Pydantic instances of the same model.

## The full example

Here is a realistic model with the types that usually make evaluation annoying: an enum, a date, an optional field, and a nested list. You configure none of it.

```python
import datetime
import enum
from typing import List, Optional

from pydantic import BaseModel

import stickler


class Priority(str, enum.Enum):
    LOW = "low"
    HIGH = "high"


class LineItem(BaseModel):
    sku: str
    description: str
    quantity: int
    unit_price: float


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    invoice_date: datetime.date
    total_amount: float
    priority: Priority
    notes: Optional[str] = None
    line_items: List[LineItem] = []


ground_truth = Invoice(
    invoice_id="INV-2024-0042",
    vendor_name="Acme Corporation",
    invoice_date=datetime.date(2024, 3, 15),
    total_amount=1247.50,
    priority=Priority.HIGH,
    notes="Net 30 payment terms",
    line_items=[
        LineItem(sku="WM-100", description="Wireless Mouse", quantity=2, unit_price=29.99),
        LineItem(sku="UC-050", description="USB-C Cable 1m", quantity=5, unit_price=12.99),
    ],
)

# What the model actually produced: minor variations, reordered list.
prediction = Invoice(
    invoice_id="INV-2024-0042",
    vendor_name="Acme Corp",                       # abbreviated
    invoice_date=datetime.date(2024, 3, 15),
    total_amount=1247.50,
    priority=Priority.HIGH,
    notes="net 30 terms",                          # reworded
    line_items=[
        LineItem(sku="UC-050", description="USB-C Cable 1 m", quantity=5, unit_price=12.99),
        LineItem(sku="WM-100", description="Wireless Mouse", quantity=2, unit_price=29.99),
    ],
)

result = stickler.evaluate(ground_truth, prediction)

print(f"Overall: {result.overall_score:.3f}")   # 0.857
print(f"F1:      {result.f1:.3f}")               # 0.933
print(result.field_scores)
```

```text
Overall: 0.857
F1:      0.933
{'invoice_id': 1.0, 'vendor_name': 0.0, 'invoice_date': 1.0,
 'total_amount': 1.0, 'priority': 1.0, 'notes': 1.0, 'line_items': 0.996}
```

Notice what happened with **zero configuration**:

- `invoice_id` was matched **exactly** (an ID typo is never "close enough").
- `total_amount` matched despite being a float, using a small numeric tolerance.
- `invoice_date` matched as a real date, not string edit-distance.
- `priority` (an enum) matched exactly.
- `notes` ("Net 30 payment terms" vs "net 30 terms") matched with **fuzzy** text matching, so reworded free text still counts.
- `line_items` scored **0.996** even though the list was reordered. Stickler pairs list elements optimally (Hungarian matching) rather than comparing by position.
- `vendor_name` scored **0.0**, and that's the interesting one. Read on.

## Every decision is inspectable

You didn't make any of these choices, so you shouldn't have to defend them from memory. `.explain()` tells you exactly what Stickler decided and why, for when a reviewer (or future you) asks:

```python
for field, info in result.explain().items():
    print(f"{field:22} {info['comparator']:40} threshold={info['threshold']} src={info['source']}")
```

```text
invoice_id             ExactComparator                          threshold=1.0  src=name-token
vendor_name            LevenshteinComparator                    threshold=0.85 src=name-token
invoice_date           DateComparator                           threshold=0.95 src=name-token
total_amount           NumericComparator                        threshold=0.95 src=name-token
priority               ExactComparator                          threshold=1.0  src=type
notes                  FuzzyComparator                          threshold=0.6  src=name-token
line_items             Hungarian (per-element StructuredModel)  threshold=0.7  src=type
line_items.sku         ExactComparator                          threshold=1.0  src=name-token
line_items.description FuzzyComparator                          threshold=0.6  src=name-token
line_items.quantity    NumericComparator                        threshold=1.0  src=name-token
line_items.unit_price  NumericComparator                        threshold=0.95 src=name-token
```

Nested fields appear under dotted paths (`line_items.sku`), so every decision at every depth is auditable.

And the answer to "why did `vendor_name` score 0.0 for *this* pair?" is right in the result:

```python
result.explain()["vendor_name"]["verdict"]
# 'raw 0.56 < threshold 0.85 -> clipped to 0.0'
```

"Acme Corporation" vs "Acme Corp" was 56% similar, **below** the `0.85` similarity threshold Stickler picked for a name field, and by default scores under the threshold are clipped to zero. A near-miss and a total mismatch are distinguishable at a glance. If you disagree with a decision, graduate that model to a hand-authored [`StructuredModel`](README.md) where you set the comparator, threshold, and weight per field explicitly.

`src` tells you where each decision came from:

- **`type`**: inferred from the Python type. For example, `priority: Priority` is an enum, so it gets `ExactComparator`.
- **`name-token`**: sharpened by the field name. A field named `..._id` gets `ExactComparator`, `..._amount` gets `NumericComparator`, and `notes` gets `FuzzyComparator`.

## How it decides (the 10-second version)

| Your field looks like | Stickler uses | Why |
|---|---|---|
| `bool`, `Enum`, `Literal` | `ExactComparator` | must match exactly |
| `int` | `NumericComparator` (exact) | counts are exact |
| `float` | `NumericComparator` (small tolerance) | `1247.50` is close enough to `1247.5001` |
| `date` / `datetime` | `DateComparator` | real date semantics, not text |
| `str` | `LevenshteinComparator` | tolerate typos |
| a field named `*_id`, `sku`, `code` | `ExactComparator` | IDs must be exact |
| a field named `*amount`, `price`, `total` | `NumericComparator` | money |
| a field named `notes`, `description` | `FuzzyComparator` | free text, reworded is fine |
| name and type disagree (`amount: str`) | the type wins | a comparator that cannot parse the type would mis-score |
| a nested model | recurse into it | field-by-field |
| a `List[Model]` | Hungarian matching | order-independent |

Full rules and rationale live in [`src/stickler/auto/README.md`](https://github.com/awslabs/stickler/blob/main/src/stickler/auto/README.md).

## The two knobs (still optional)

The defaults are meant to be enough, but if you want a little more control without defining a `StructuredModel`:

```python
# Reuse the compiled evaluator across a whole dataset (faster):
spec = stickler.eval_for(Invoice)
scores = [spec.evaluate(gt, pred).overall_score for gt, pred in dataset]

# Let field names hint at business importance (id/amount weigh more):
result = stickler.evaluate(ground_truth, prediction, weight_hints=True)
```

By default every field weighs the same (`weight=1.0`), because business-criticality is not something a plain model encodes and we would rather not guess it silently. `weight_hints=True` turns on name-based weighting (for example, `invoice_id` at 3x and `total_amount` at 2.5x), and, as always, `.explain()` shows you exactly what changed.

!!! tip "Want full control?"
    When you outgrow inference, graduate to a hand-authored [`StructuredModel`](README.md) with explicit `ComparableField` comparators, thresholds, and weights. `stickler.evaluate` is the on-ramp, not a ceiling.

## Next steps

- **Evaluating a Strands agent?** See [Evaluating a Strands Agent](../Guides/Use-Cases/evaluate-strands-agent.md).
- **Scoring a whole test set?** See [Bulk Evaluation](../Guides/Evaluation/bulk-evaluation.md).
- **Want to tune comparators by hand?** See [Getting Started](README.md).
