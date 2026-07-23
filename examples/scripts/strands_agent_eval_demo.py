"""Evaluate a Strands agent's structured output with zero configuration.

This demo shows the *entire* integration for scoring a Strands agent that emits
structured output via a Pydantic ``response_model``:

    prediction = agent.structured_output(Invoice, prompt)   # your agent, unchanged
    result = stickler.evaluate(ground_truth, prediction)    # the whole integration

No ``StructuredModel`` subclass, no comparators, no thresholds, no schema.
``stickler.evaluate`` infers a sensible comparator/threshold per field from the
Pydantic model itself (see ``src/stickler/auto/README.md``).

Running the agent requires the optional ``[llm]`` extra and AWS Bedrock
credentials:

    pip install "stickler-eval[llm]"

If Strands is not installed (or the model call fails), this script falls back to
a *fabricated* prediction so it stays runnable and demonstrates the same
evaluation path. The Stickler side does not depend on Strands at all.

Usage:
    python strands_agent_eval_demo.py
"""

import datetime
from typing import List, Optional

from pydantic import BaseModel

import stickler

# --- 1. The response_model your agent already uses --------------------------


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


DOCUMENT = """
INVOICE  #INV-2024-0042
Acme Corporation      Date: 2024-03-15
2x Wireless Mouse (WM-100) @ $29.99
5x USB-C Cable 1m (UC-050) @ $12.99
Total: $1,247.50    Terms: Net 30
"""


# --- 2. Get a prediction (real agent, or a fabricated fallback) -------------


def get_prediction() -> Invoice:
    """Run a Strands agent if available; otherwise fabricate a plausible output."""
    try:
        from strands import Agent

        agent = Agent(model="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        print("Running Strands agent for structured output...")
        return agent.structured_output(Invoice, f"Extract the invoice:\n{DOCUMENT}")
    except Exception as exc:  # noqa: BLE001 - demo fallback for any failure
        print(f"(Strands unavailable: {type(exc).__name__}. Using a fabricated prediction.)")
        # Stand-in for what an agent might return: mostly right, with the kind of
        # variations real extraction produces.
        return Invoice(
            invoice_id="INV-2024-0042",
            vendor_name="Acme Corp",  # abbreviated
            invoice_date=datetime.date(2024, 3, 15),
            total_amount=1247.50,
            notes="net 30 terms",  # reworded
            line_items=[
                # Reordered relative to ground truth, minor description drift.
                LineItem(sku="UC-050", description="USB-C Cable 1 m", quantity=5, unit_price=12.99),
                LineItem(sku="WM-100", description="Wireless Mouse", quantity=2, unit_price=29.99),
            ],
        )


# --- 3. Your labeled ground truth -------------------------------------------


GROUND_TRUTH = Invoice(
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


def main() -> None:
    prediction = get_prediction()

    # --- 4. Score it: the entire integration is one call --------------------
    result = stickler.evaluate(GROUND_TRUTH, prediction)

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Overall score: {result.overall_score:.3f}")
    print(f"Precision:     {result.precision:.3f}")
    print(f"Recall:        {result.recall:.3f}")
    print(f"F1:            {result.f1:.3f}\n")

    print("Per-field scores:")
    for field, score in result.field_scores.items():
        print(f"  {field:14} {score:.3f}")

    # --- 5. Defend every decision -------------------------------------------
    print("\n" + "=" * 60)
    print("WHY (inferred configuration — nothing was hand-written)")
    print("=" * 60)
    for field, info in result.explain().items():
        print(
            f"  {field:14} {info['comparator']:26} "
            f"threshold={info['threshold']:<5} src={info['source']}"
        )


if __name__ == "__main__":
    main()
