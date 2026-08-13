---
title: Evaluating a Strands Agent
---

# Evaluating a Strands Agent

[Strands](https://strandsagents.com/) agents produce structured output by handing the agent a Pydantic model:

```python
from strands import Agent

agent = Agent(model="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
result = agent("Extract the invoice from this document: ...", structured_output_model=Invoice)
invoice = result.structured_output
# `invoice` is a validated Invoice instance
```

The question this guide answers: **how accurate is that output, and do the errors matter?**

There are two supported ways to wire Stickler into this, and neither is second-class:

| | Flow 1: plain Pydantic | Flow 2: configured `StructuredModel` |
|---|---|---|
| agent gets | your existing `BaseModel` | your `StructuredModel`, directly |
| evaluation | `stickler.evaluate()` infers comparators | your explicit comparators, thresholds, weights |
| choose it when | you want a baseline in one line | you need per-field control over scoring |

Start with Flow 1; graduate to Flow 2 when you outgrow the inferred defaults. With Flow 1 the integration is a single line — you do not write a `StructuredModel`, pick comparators, or annotate anything.

## Flow 1: the whole integration

```python
import stickler

prediction = agent(prompt, structured_output_model=Invoice).structured_output  # your agent, unchanged
result = stickler.evaluate(ground_truth, prediction)     # <-- the entire integration

print(result.overall_score, result.f1, result.field_scores)
```

`ground_truth` is your labeled expected output (same `Invoice` type). Everything else is inferred from the model: which comparator each field gets, sensible thresholds, and order-independent list matching. See [Ultra Quick Start](../../Getting-Started/ultra-quick-start.md) for how that inference works.

## End-to-end example

```python
import datetime
from typing import List, Optional

from pydantic import BaseModel
from strands import Agent

import stickler


# 1. The response_model your agent already uses for structured output.
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
    notes: Optional[str] = None
    line_items: List[LineItem] = []


# 2. Run the agent (this is normal Strands usage).
agent = Agent(model="us.anthropic.claude-sonnet-4-5-20250929-v1:0")

DOCUMENT = """
INVOICE  #INV-2024-0042
Acme Corporation      Date: 2024-03-15
2x Wireless Mouse (WM-100) @ $29.99
5x USB-C Cable 1m (UC-050) @ $12.99
Total: $1,247.50    Terms: Net 30
"""

prediction = agent(
    f"Extract the invoice:\n{DOCUMENT}", structured_output_model=Invoice
).structured_output

# 3. Your labeled ground truth for this document.
ground_truth = Invoice(
    invoice_id="INV-2024-0042",
    vendor_name="Acme Corporation",
    invoice_date=datetime.date(2024, 3, 15),
    total_amount=1247.50,
    notes="Net 30 payment terms",
    line_items=[
        LineItem(sku="WM-100", description="Wireless Mouse", quantity=2, unit_price=29.99),
        LineItem(sku="UC-050", description="USB-C Cable 1m", quantity=5, unit_price=12.99),
    ],
)

# 4. Score it.
result = stickler.evaluate(ground_truth, prediction)
print(f"Overall: {result.overall_score:.3f}")
print(f"F1:      {result.f1:.3f}")
for field, score in result.field_scores.items():
    print(f"  {field:14} {score:.3f}")
```

## Scoring a whole evaluation set

You usually have many labeled documents, not one. Turn the agent's
`response_model` into a regular `StructuredModel` with
`StructuredModel.from_pydantic()` (the same inference `stickler.evaluate`
uses), then the standard
[Bulk Evaluation](../Evaluation/bulk-evaluation.md) update/compute pattern
applies exactly as documented:

```python
from stickler import StructuredModel
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

InvoiceEval = StructuredModel.from_pydantic(Invoice)
evaluator = BulkStructuredModelEvaluator(target_schema=InvoiceEval)

for doc, expected in labeled_dataset:           # your (document, ground_truth) pairs
    prediction = agent(
        f"Extract the invoice:\n{doc}", structured_output_model=Invoice
    ).structured_output
    evaluator.update(
        InvoiceEval.from_json(expected.model_dump()),
        InvoiceEval.from_json(prediction.model_dump()),
    )

corpus = evaluator.compute()
print(f"Corpus F1: {corpus.metrics['cm_f1']:.3f} over {corpus.document_count} docs")
```

This accumulates true corpus-level precision/recall/F1 (confusion-matrix
counts across every document, not a mean of per-document scores), and
everything from the [Bulk Evaluation guide](../Evaluation/bulk-evaluation.md)
applies unchanged: `doc_id` tracking, JSONL per-document output,
checkpointing, and error accumulation. And because `InvoiceEval` is an
ordinary `StructuredModel`, you can export its inferred config
(`to_stickler_config()`), edit any comparator or threshold, and rebuild, no
zero-config-specific API needed. For a closer look at any single pair,
`stickler.evaluate(expected, prediction)` returns the per-document
[`EvalResult`](#defending-the-numbers).

## Defending the numbers

When someone asks "why did `vendor_name` score zero?", you don't guess. You
ask the single-pair result from the end-to-end example above:

```python
result.explain()["vendor_name"]
# {'comparator': 'LevenshteinComparator', 'threshold': 0.85, 'weight': 1.0,
#  'source': 'name-token', 'score': 0.0, 'raw_similarity': 0.5625,
#  'verdict': 'raw 0.56 < threshold 0.85 -> clipped to 0.0', 'why': [...]}
```

"Acme Corporation" vs "Acme Corp" fell below the `0.85` similarity threshold Stickler chose for a name field. If that is too strict for your use case, graduate to a hand-authored [`StructuredModel`](../../Getting-Started/README.md) where you set the comparator, threshold, and weight per field explicitly. `stickler.evaluate` gets you a defensible baseline in one line; the full API is there when you outgrow it. That graduation is Flow 2, below.

## Flow 2: one configured class, two jobs

Because `StructuredModel` extends `pydantic.BaseModel`, the class that carries your comparison configuration can *also* be the agent's `structured_output_model`. Define it once; it drives extraction and evaluation:

```python
from typing import List, Optional

from stickler import (
    ComparableField,
    ExactComparator,
    NumericComparator,
    StructuredModel,
)


class LineItem(StructuredModel):
    product: str = ComparableField()
    qty: int = ComparableField()


class Invoice(StructuredModel):
    shipment_id: str = ComparableField(
        comparator=ExactComparator(),                 # IDs must match exactly
        weight=3.0,                                   # and matter most
        description="The carrier shipment tracking identifier",
        examples=["1Z999AA10123456784"],
    )
    amount: float = ComparableField(comparator=NumericComparator())
    line_items: List[LineItem] = ComparableField()
    notes: Optional[str] = ComparableField(default=None)


# The SAME class drives the agent...
prediction = agent(prompt, structured_output_model=Invoice).structured_output

# ...and the evaluation, with your configured comparators.
result = ground_truth.compare_with(prediction)
print(result["overall_score"], result["field_scores"])
```

The schema the agent receives is clean: `shipment_id`, `amount`, and `line_items` are `required` exactly as a plain `BaseModel` twin would declare them, no comparison metadata rides along, and `description` / `examples` / `alias` reach the model, where they genuinely help extraction. Requiredness in the schema follows your annotations — `notes: Optional[str]` is optional, everything else is required — while evaluation stays tolerant: a prediction that omits a field still constructs and scores, it just scores as missing.

Which fields are required is decided by the **annotation**, so write `Optional[...]` on fields the model may legitimately skip and leave the rest bare. This is ordinary Pydantic; there is nothing Stickler-specific to learn.

## Requirements

Strands is an optional dependency:

```bash
pip install "stickler-eval[llm]"     # brings in strands-agents
```

`stickler.evaluate` itself has **no** dependency on Strands; it works on any Pydantic instances. You only need `[llm]` to run the agent that produces them.
