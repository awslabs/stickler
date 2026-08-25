"""Tool-spec parity between StructuredModel and plain BaseModel (issue #188).

``StructuredModel`` extends ``pydantic.BaseModel``, so one configured class
can drive a Strands agent's structured output *and* carry stickler's
comparison configuration. That premise only holds if the schema a
``StructuredModel`` hands to the model matches what an equivalent plain
``BaseModel`` would send. These tests pin the three degradations from #188:

1. ``required`` disappeared (ComparableField's ``default=None`` made every
   field optional in Pydantic's eyes).
2. Types widened (a required ``str`` rendered as ``["string", "null"]``).
3. Comparison config leaked into the tool spec.

And the property to preserve: ``description``, ``examples``, and ``alias``
must keep reaching the rendered schema, since those are genuinely useful to
the model.

The whole module goes through Strands' own ``convert_pydantic_to_tool_spec``
and skips when strands is not installed (the ``llm`` extra; CI installs it).
The equivalent schema-shape assertions that need no strands live in
``test_model_schema.py`` and ``test_structured_model_schema.py``.
"""

import json
from typing import List, Optional

import pytest
from pydantic import BaseModel

from stickler.comparators.exact import ExactComparator
from stickler.comparators.numeric import NumericComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)

strands_structured_output = pytest.importorskip(
    "strands.tools.structured_output",
    reason="strands-agents not installed (llm extra)",
)
convert_pydantic_to_tool_spec = strands_structured_output.convert_pydantic_to_tool_spec


# --- Equivalent model pairs -------------------------------------------------


class PlainLineItem(BaseModel):
    product: str
    qty: int


class PlainInvoice(BaseModel):
    shipment_id: str
    amount: float
    line_items: List[PlainLineItem]
    memo: Optional[str] = None


class LineItem(StructuredModel):
    product: str = ComparableField()
    qty: int = ComparableField()


class Invoice(StructuredModel):
    shipment_id: str = ComparableField(comparator=ExactComparator(), weight=3.0)
    amount: float = ComparableField(comparator=NumericComparator())
    line_items: List[LineItem] = ComparableField()
    memo: Optional[str] = ComparableField(default=None)


def _tool_spec_json(model_cls) -> dict:
    return convert_pydantic_to_tool_spec(model_cls)["inputSchema"]["json"]


class TestToolSpecParity:
    """Flow 2: a configured StructuredModel sends the same schema as its twin."""

    def test_required_lists_match(self):
        plain = _tool_spec_json(PlainInvoice)
        configured = _tool_spec_json(Invoice)

        assert sorted(configured.get("required") or []) == sorted(
            plain.get("required") or []
        )
        # And they are the right fields, not coincidentally-equal empties.
        assert sorted(plain.get("required") or []) == [
            "amount",
            "line_items",
            "shipment_id",
        ]

    def test_required_field_types_are_not_null_widened(self):
        configured = _tool_spec_json(Invoice)

        assert configured["properties"]["shipment_id"]["type"] == "string"
        assert configured["properties"]["amount"]["type"] == "number"

    def test_optional_field_stays_nullable(self):
        plain = _tool_spec_json(PlainInvoice)
        configured = _tool_spec_json(Invoice)

        # memo is genuinely Optional on both models; the annotation, not the
        # ComparableField default, is what makes it nullable.
        assert configured["properties"]["memo"]["type"] == plain["properties"][
            "memo"
        ]["type"]

    def test_no_comparison_config_in_tool_spec(self):
        configured = _tool_spec_json(Invoice)

        assert "x-comparison" not in json.dumps(configured)

    def test_payload_size_is_comparable(self):
        # #188 measured 141% bloat. Identical shape means near-identical size;
        # allow slack only for the model title/description strings.
        plain = len(json.dumps(_tool_spec_json(PlainInvoice)))
        configured = len(json.dumps(_tool_spec_json(Invoice)))

        assert configured <= plain * 1.15, (
            f"tool spec for the configured model is {configured} chars vs "
            f"{plain} for the plain twin; comparison metadata is leaking again"
        )


class TestFlowOnePlainPydantic:
    """Flow 1: plain Pydantic through the tool spec, evaluated by inference."""

    def test_plain_model_tool_spec_and_evaluation(self):
        import stickler

        spec = _tool_spec_json(PlainInvoice)
        assert sorted(spec.get("required") or []) == [
            "amount",
            "line_items",
            "shipment_id",
        ]

        ground_truth = PlainInvoice(
            shipment_id="SHP-1",
            amount=10.0,
            line_items=[PlainLineItem(product="mouse", qty=2)],
        )
        prediction = PlainInvoice(
            shipment_id="SHP-1",
            amount=10.0,
            line_items=[PlainLineItem(product="mouse", qty=2)],
        )

        result = stickler.evaluate(ground_truth, prediction)
        assert result.overall_score == 1.0


class TestFlowTwoConfiguredModel:
    """Flow 2 end to end: the same class drives the agent and the comparison."""

    def test_configured_comparators_still_honored(self):
        # The schema fix must not disturb the comparison config. ExactComparator
        # on shipment_id means a one-character difference scores 0, where the
        # default Levenshtein would score high.
        gt = Invoice.from_json(
            {
                "shipment_id": "SHP-2024-001",
                "amount": 10.0,
                "line_items": [{"product": "mouse", "qty": 2}],
            }
        )
        off_by_one = Invoice.from_json(
            {
                "shipment_id": "SHP-2024-002",
                "amount": 10.0,
                "line_items": [{"product": "mouse", "qty": 2}],
            }
        )

        result = gt.compare_with(off_by_one)
        assert result["field_scores"]["shipment_id"] == 0.0

    def test_partial_prediction_still_constructs(self):
        # The schema-only fix must not change runtime tolerance: predictions
        # that omit fields still construct (the engine depends on this).
        partial = Invoice.from_json({"shipment_id": "SHP-1"})
        assert partial.amount is None

        gt = Invoice.from_json(
            {
                "shipment_id": "SHP-1",
                "amount": 10.0,
                "line_items": [{"product": "mouse", "qty": 2}],
            }
        )
        result = gt.compare_with(partial)
        assert result["field_scores"]["shipment_id"] == 1.0
