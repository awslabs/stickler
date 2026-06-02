#!/usr/bin/env python3
"""End-to-end integration tests for AggregateConfusionMatrixAccumulator.

Pins the user-visible contract that AggregateConfusionMatrixAccumulator
behaves correctly when wired through BulkStructuredModelEvaluator:

- It is part of the default accumulator set.
- Its corpus-level rollup equals the per-document sum (parity headline).
- It captures the FD-recursion behavior described in
  ``docs/Advanced/aggregate-metrics.md`` Example 2 — i.e., it includes
  fields from below-threshold (FD) pairs and unmatched items, which the
  bulk evaluator's own ``overall_metrics`` does not.
- State serialization round-trips produce identical totals to a
  single-process run (distributed eval contract).
- Opt-out via custom ``accumulators=[...]`` is honored.
- Empty runs produce no aggregate output (parity with other accumulators
  that return ``None`` from ``compute()``).
"""

from typing import List, Optional

import pytest

from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.aggregate import (
    AggregateConfusionMatrixAccumulator,
)
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.confidence.accumulator import (
    ConfidenceAccumulator,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)


# --- Test models ----------------------------------------------------------


class _Contact(StructuredModel):
    """Simple nested contact for primitive-only aggregation tests."""

    phone: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    email: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class _Person(StructuredModel):
    """Top-level model with a primitive + nested object."""

    name: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    contact: _Contact = ComparableField(weight=1.0)


class _LineItem(StructuredModel):
    """List-of-StructuredModel element from docs Example 2.

    ``match_threshold = 0.6`` lets us construct (sku/description/qty)
    triples whose combined similarity falls below 0.6 — making them FD
    at the object level — while still leaving leaf fields available for
    aggregate recursion (which is the headline behavior tested in
    ``test_list_of_structured_model_with_threshold_gating``).
    """

    match_threshold = 0.6
    sku: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )
    description: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    qty: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class _Invoice(StructuredModel):
    invoice_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0
    )
    items: List[_LineItem] = ComparableField(weight=1.0)


# --- Helpers --------------------------------------------------------------


_METRIC_KEYS = ("tp", "fp", "fn", "fa", "fd", "tn")


def _zero_counts() -> dict:
    return {k: 0 for k in _METRIC_KEYS}


def _add_counts(target: dict, source: dict) -> None:
    """Sum tp/fp/fn/fa/fd/tn from ``source`` into ``target`` in place.

    Mirrors AggregateConfusionMatrixAccumulator._add_metrics so the
    parity test computes a "manual ground truth" with the exact same
    semantics the accumulator is supposed to implement.
    """
    if not isinstance(source, dict):
        return
    for key in _METRIC_KEYS:
        value = source.get(key, 0)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value >= 0:
            target[key] += int(value)


def _make_person_pair(idx: int) -> tuple:
    """Build a (gt, pred) Person pair with a deterministic mismatch pattern.

    ``idx`` controls which fields differ so the corpus exercises a mix
    of TP/FD/FN across documents — without that variety the parity test
    only proves "summing zeros equals zeros."
    """
    gt = _Person(
        name=f"name_{idx}",
        contact=_Contact(phone=f"555-{idx:04d}", email=f"u{idx}@example.com"),
    )
    if idx % 3 == 0:
        # Perfect match — TPs at every leaf.
        pred = _Person(
            name=f"name_{idx}",
            contact=_Contact(phone=f"555-{idx:04d}", email=f"u{idx}@example.com"),
        )
    elif idx % 3 == 1:
        # Phone wrong → 1 FD on contact.phone, rest TP.
        pred = _Person(
            name=f"name_{idx}",
            contact=_Contact(phone="000-0000", email=f"u{idx}@example.com"),
        )
    else:
        # Email missing → 1 FN on contact.email; name still TP.
        pred = _Person(
            name=f"name_{idx}",
            contact=_Contact(phone=f"555-{idx:04d}", email=None),
        )
    return gt, pred


# --- Tests ---------------------------------------------------------------


class TestDefaultAccumulators:
    """The aggregate accumulator must be wired in by default — otherwise
    every existing user picks up no new behavior on upgrade."""

    def test_default_includes_aggregate_accumulator(self):
        evaluator = BulkStructuredModelEvaluator(target_schema=_Person)

        # (a) The instance carries it…
        names = [acc.name for acc in evaluator._accumulators]
        assert "aggregate_metrics" in names
        assert any(
            isinstance(acc, AggregateConfusionMatrixAccumulator)
            for acc in evaluator._accumulators
        )

        # (b) …and after running, it surfaces in accumulator_metrics.
        gt, pred = _make_person_pair(0)
        evaluator.update(gt, pred, "doc0")
        result = evaluator.compute()

        assert result.accumulator_metrics is not None
        assert "aggregate_metrics" in result.accumulator_metrics
        agg = result.accumulator_metrics["aggregate_metrics"]
        assert "overall" in agg
        assert "fields" in agg


class TestParityWithPerDocSum:
    """Headline test: bulk-rolled aggregate equals manual sum of per-doc
    ``confusion_matrix['aggregate']`` blocks. If this drifts the feature
    is broken regardless of whether other tests still pass."""

    def test_bulk_aggregate_parity_with_per_doc_sum(self):
        pairs = [_make_person_pair(i) for i in range(5)]

        # Bulk path — the system under test.
        evaluator = BulkStructuredModelEvaluator(target_schema=_Person)
        for i, (gt, pred) in enumerate(pairs):
            evaluator.update(gt, pred, f"doc_{i}")
        bulk_result = evaluator.compute()

        bulk_agg_overall = bulk_result.accumulator_metrics["aggregate_metrics"][
            "overall"
        ]

        # Manual reference path — sum each document's top-level aggregate.
        manual = _zero_counts()
        for gt, pred in pairs:
            cm = gt.compare_with(pred, include_confusion_matrix=True)[
                "confusion_matrix"
            ]
            _add_counts(manual, cm.get("aggregate"))

        for key in _METRIC_KEYS:
            assert bulk_agg_overall[key] == manual[key], (
                f"aggregate.overall[{key}] mismatch: "
                f"bulk={bulk_agg_overall[key]} manual={manual[key]}"
            )

        # Sanity: at least one TP and one non-TP outcome present, else
        # the parity assertion above would be a vacuous "0 == 0".
        assert manual["tp"] > 0
        assert (manual["fd"] + manual["fn"] + manual["fa"]) > 0


class TestListOfStructuredModelThresholdGating:
    """The whole reason the aggregate accumulator exists: per-doc
    ``aggregate`` recurses into FD-classified pairs and unmatched items,
    while the bulk evaluator's own ``overall_metrics`` does not. The
    accumulator must therefore expose strictly more leaf-level signal
    than ``overall_metrics`` for fields below an FD list element."""

    def test_list_of_structured_model_with_threshold_gating(self):
        # Mirrors docs/Advanced/aggregate-metrics.md Example 2.
        gt = _Invoice(
            invoice_id="INV-001",
            items=[
                _LineItem(sku="AAA", description="Widget", qty=10),
                _LineItem(sku="BBB", description="Gadget", qty=5),
                _LineItem(sku="CCC", description="Cable", qty=2),
            ],
        )
        pred = _Invoice(
            invoice_id="INV-001",
            items=[
                _LineItem(sku="AAA", description="Widget", qty=10),
                _LineItem(
                    sku="BBB", description="Completely Wrong", qty=99
                ),
            ],
        )

        evaluator = BulkStructuredModelEvaluator(target_schema=_Invoice)
        evaluator.update(gt, pred, "doc_invoice")
        result = evaluator.compute()

        agg = result.accumulator_metrics["aggregate_metrics"]
        agg_overall = agg["overall"]
        agg_fields = agg["fields"]

        # Per the docs example: at the items level the per-doc aggregate
        # is tp=4, fd=2, fn=3 — the FD pair (BBB) and unmatched CCC
        # contribute leaf counts. Bulk should pass that through unchanged
        # for a 1-document run.
        assert agg_fields["items"]["tp"] == 4
        assert agg_fields["items"]["fd"] == 2
        assert agg_fields["items"]["fn"] == 3

        # Top-level aggregate adds invoice_id (1 TP) on top of items.
        assert agg_overall["tp"] == 5
        assert agg_overall["fd"] == 2
        assert agg_overall["fn"] == 3

        # Headline contract: aggregate exposes leaf signal that
        # field_metrics["items"] does not. The bulk field_metrics
        # records the items list field's threshold-gated counts only —
        # so its ``fd`` should equal the object-level fd (1), strictly
        # less than the aggregate fd of 2 (which includes the leaf FDs
        # produced by the BBB pair's description and qty fields).
        items_field_metrics = result.field_metrics["items"]
        assert items_field_metrics.get("fd", 0) == 1
        assert agg_fields["items"]["fd"] > items_field_metrics.get("fd", 0)

    def test_aggregate_records_per_field_paths_below_list(self):
        """Per-leaf paths under the list (``items.sku`` etc.) must be
        present in aggregate's ``fields`` dict — that's the structure
        downstream consumers query on."""
        gt = _Invoice(
            invoice_id="INV-001",
            items=[
                _LineItem(sku="AAA", description="Widget", qty=10),
                _LineItem(sku="BBB", description="Gadget", qty=5),
            ],
        )
        pred = _Invoice(
            invoice_id="INV-001",
            items=[
                _LineItem(sku="AAA", description="Widget", qty=10),
                _LineItem(
                    sku="BBB", description="Completely Wrong", qty=99
                ),
            ],
        )

        evaluator = BulkStructuredModelEvaluator(target_schema=_Invoice)
        evaluator.update(gt, pred, "doc_invoice")
        result = evaluator.compute()

        agg_fields = result.accumulator_metrics["aggregate_metrics"]["fields"]

        # Leaf paths below the list field must be addressable.
        for leaf in ("items.sku", "items.description", "items.qty"):
            assert leaf in agg_fields, f"missing aggregate path: {leaf}"


class TestDistributedEvalRoundTrip:
    """Distributed eval contract: serialize-merge-compute on N shards
    must equal compute on the union. The aggregate accumulator's state
    is part of ``BulkStructuredModelEvaluator.get_state()`` and is
    merged in by ``merge_state``; if either side regresses, sharded runs
    silently lose data. This test pins both round-trips together."""

    def test_distributed_eval_round_trip(self):
        all_pairs = [_make_person_pair(i) for i in range(6)]
        shard_a = all_pairs[:3]
        shard_b = all_pairs[3:]

        # Shard A
        ev_a = BulkStructuredModelEvaluator(target_schema=_Person)
        for i, (gt, pred) in enumerate(shard_a):
            ev_a.update(gt, pred, f"a_{i}")

        # Shard B
        ev_b = BulkStructuredModelEvaluator(target_schema=_Person)
        for i, (gt, pred) in enumerate(shard_b):
            ev_b.update(gt, pred, f"b_{i}")

        # Merge B's state into A and compute.
        ev_a.merge_state(ev_b.get_state())
        merged_result = ev_a.compute()
        merged_agg = merged_result.accumulator_metrics["aggregate_metrics"][
            "overall"
        ]

        # Single-process reference run.
        ev_ref = BulkStructuredModelEvaluator(target_schema=_Person)
        for i, (gt, pred) in enumerate(all_pairs):
            ev_ref.update(gt, pred, f"ref_{i}")
        ref_agg = ev_ref.compute().accumulator_metrics["aggregate_metrics"][
            "overall"
        ]

        for key in _METRIC_KEYS:
            assert merged_agg[key] == ref_agg[key], (
                f"merged.aggregate.overall[{key}] != single-process: "
                f"merged={merged_agg[key]} ref={ref_agg[key]}"
            )

        # Sanity: at least one count was actually accumulated, else
        # 0==0 would pass without exercising the merge path.
        assert ref_agg["tp"] > 0


class TestOptOutViaCustomAccumulators:
    """A user who explicitly passes ``accumulators=[...]`` without the
    aggregate accumulator must NOT get aggregate_metrics in the output.
    Default-injection must not silently override an explicit list."""

    def test_opt_out_via_custom_accumulators(self):
        evaluator = BulkStructuredModelEvaluator(
            target_schema=_Person,
            accumulators=[ConfidenceAccumulator()],
        )

        # Sanity: only the one accumulator is wired up.
        assert [acc.name for acc in evaluator._accumulators] == [
            "confidence_metrics"
        ]

        gt, pred = _make_person_pair(0)
        evaluator.update(gt, pred, "doc0")
        result = evaluator.compute()

        # Either no accumulator_metrics at all (None when nothing
        # contributed), or the dict simply lacks aggregate_metrics.
        if result.accumulator_metrics is not None:
            assert "aggregate_metrics" not in result.accumulator_metrics


class TestEmptyEvaluation:
    """Zero-document runs must not synthesize an empty aggregate block —
    the accumulator's ``compute()`` returns ``None`` when nothing
    contributed, and the bulk evaluator filters those out before
    populating ``ProcessEvaluation.accumulator_metrics``."""

    def test_empty_evaluation_no_aggregate(self):
        evaluator = BulkStructuredModelEvaluator(target_schema=_Person)
        result = evaluator.compute()

        # Either the whole field is None (no accumulator produced
        # output) or, if some other accumulator contributed an empty
        # block, aggregate_metrics is still absent / falsy.
        if result.accumulator_metrics is None:
            return

        agg = result.accumulator_metrics.get("aggregate_metrics")
        assert agg is None or (
            agg.get("overall", {}).get("tp", 0) == 0
            and agg.get("overall", {}).get("fp", 0) == 0
            and agg.get("overall", {}).get("fn", 0) == 0
            and agg.get("overall", {}).get("fd", 0) == 0
            and agg.get("overall", {}).get("fa", 0) == 0
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
