---
title: Evaluating a Strands Agent
---

# Evaluating a Strands Agent

[Strands](https://strandsagents.com/) agents produce structured output by handing the agent a Pydantic `response_model`:

```python
from strands import Agent

agent = Agent(model="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
invoice = agent.structured_output(Invoice, "Extract the invoice from this document: ...")
# `invoice` is a validated Invoice instance
```

The question this guide answers: **how accurate is that output, and do the errors matter?** With `stickler.evaluate` the integration is a single line. You do not write a `StructuredModel`, pick comparators, or annotate anything.

## The whole integration

```python
import stickler

prediction = agent.structured_output(Invoice, prompt)   # your agent, unchanged
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

prediction = agent.structured_output(Invoice, f"Extract the invoice:\n{DOCUMENT}")

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

You usually have many labeled documents, not one. Compile the evaluator once with `eval_for`, then loop:

```python
spec = stickler.eval_for(Invoice)

scores = []
for doc, expected in labeled_dataset:          # your (document, ground_truth) pairs
    prediction = agent.structured_output(Invoice, f"Extract the invoice:\n{doc}")
    result = spec.evaluate(expected, prediction)
    scores.append(result.overall_score)

print(f"Mean score over {len(scores)} docs: {sum(scores) / len(scores):.3f}")
```

For corpus-level aggregate metrics (precision/recall/F1 accumulated across every
document, not a mean of per-document scores), use the standard
[Bulk Evaluation](../Evaluation/bulk-evaluation.md) update/compute pattern.
`spec.bulk_evaluator()` returns a `BulkStructuredModelEvaluator` wired to the
compiled model, and `spec.to_model()` converts your Pydantic instances (or plain
dicts) into what it accepts:

```python
spec = stickler.eval_for(Invoice)
evaluator = spec.bulk_evaluator()               # standard update/compute evaluator

for doc, expected in labeled_dataset:
    prediction = agent.structured_output(Invoice, f"Extract the invoice:\n{doc}")
    evaluator.update(spec.to_model(expected), spec.to_model(prediction))

result = evaluator.compute()
print(f"Corpus F1: {result.metrics['cm_f1']:.3f} over {result.document_count} docs")
```

Everything from the [Bulk Evaluation guide](../Evaluation/bulk-evaluation.md)
applies unchanged: `doc_id` tracking, JSONL per-document output, checkpointing,
and error accumulation.

## Defending the numbers

When someone asks "why did `vendor_name` score zero?", you don't guess. You ask:

```python
result.explain()["vendor_name"]
# {'comparator': 'LevenshteinComparator', 'threshold': 0.85, 'weight': 1.0,
#  'source': 'name-token', 'why': [...]}
```

"Acme Corporation" vs "Acme Corp" fell below the `0.85` similarity threshold Stickler chose for a name field. If that is too strict for your use case, graduate to a hand-authored [`StructuredModel`](../../Getting-Started/README.md) where you set the comparator, threshold, and weight per field explicitly. `stickler.evaluate` gets you a defensible baseline in one line; the full API is there when you outgrow it.

## Requirements

Strands is an optional dependency:

```bash
pip install "stickler-eval[llm]"     # brings in strands-agents
```

`stickler.evaluate` itself has **no** dependency on Strands; it works on any Pydantic instances. You only need `[llm]` to run the agent that produces them.
