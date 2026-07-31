"""
Tests for prediction_raw round-trip through comparison results.

Verifies that raw prediction JSON (with rich value metadata) is preserved
in compare_with() results, survives JSONL serialization, and enables
update_from_comparison_result() to produce identical confidence metrics
as the direct update() path.

This is the PRIMARY consumption path for confidence metrics (and future
bbox/MAP metrics) in production, where comparison results are serialized
to JSONL and aggregated later.
"""

import json
from typing import List

from stickler.comparators import (
    ExactComparator,
    LevenshteinComparator,
    NumericComparator,
)
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
    aggregate_from_comparisons,
)
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

# ── Test models ──


class Product(StructuredModel):
    name: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
    price: float = ComparableField(comparator=NumericComparator(), threshold=0.5)
    sku: str = ComparableField(comparator=ExactComparator(), threshold=1.0)


class Address(StructuredModel):
    street: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.7)
    city: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.7)


class Customer(StructuredModel):
    name: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
    address: Address = ComparableField()


class Order(StructuredModel):
    order_id: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    items: List[Product] = ComparableField()


# ── 1. compare_with includes prediction_raw ──


class TestPredictionRawInResult:
    def test_prediction_raw_present_with_rich_values(self):
        """compare_with includes prediction_raw when prediction has rich value metadata."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC123", "_confidence": 0.7},
        })
        result = gt.compare_with(pred, document_field_comparisons=True)
        assert "prediction_raw" in result
        assert result["prediction_raw"]["name"]["_value"] == "Widget"
        assert result["prediction_raw"]["name"]["_confidence"] == 0.9

    def test_prediction_raw_with_partial_confidence(self):
        """prediction_raw preserves the mix of rich and plain values."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": 29.99,  # plain value
            "sku": {"_value": "ABC123"},  # value-only rich value
        })
        result = gt.compare_with(pred, document_field_comparisons=True)
        assert "prediction_raw" in result
        raw = result["prediction_raw"]
        assert raw["name"]["_confidence"] == 0.9
        assert raw["price"] == 29.99
        assert raw["sku"] == {"_value": "ABC123"}

    def test_prediction_raw_absent_without_rich_values(self):
        """Plain predictions (no rich values) don't include prediction_raw."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product(name="Widget", price=29.99, sku="ABC123")
        result = gt.compare_with(pred, document_field_comparisons=True)
        assert "prediction_raw" not in result

    def test_prediction_raw_with_nested_model(self):
        """Nested model prediction_raw preserves the full nested structure."""
        gt = Customer(
            name="Jane",
            address=Address(street="123 Main", city="Boston"),
        )
        pred = Customer.from_json({
            "name": {"_value": "Jane", "_confidence": 0.96},
            "address": {
                "street": {"_value": "123 Main St", "_confidence": 0.85},
                "city": {"_value": "Chicago", "_confidence": 0.40},
            },
        })
        result = gt.compare_with(pred, document_field_comparisons=True)
        assert "prediction_raw" in result
        raw = result["prediction_raw"]
        assert raw["address"]["street"]["_confidence"] == 0.85
        assert raw["address"]["city"]["_confidence"] == 0.40

    def test_prediction_raw_with_list_fields(self):
        """List field prediction_raw preserves array structure and indices."""
        gt = Order(
            order_id="ORD-1",
            items=[
                Product(name="Mouse", price=29.99, sku="MOU001"),
            ],
        )
        pred = Order.from_json({
            "order_id": {"_value": "ORD-1", "_confidence": 0.99},
            "items": [
                {
                    "name": {"_value": "Mouse", "_confidence": 0.90},
                    "price": {"_value": 29.99, "_confidence": 0.85},
                    "sku": {"_value": "MOU001", "_confidence": 0.95},
                },
            ],
        })
        result = gt.compare_with(pred, document_field_comparisons=True)
        raw = result["prediction_raw"]
        assert raw["items"][0]["name"]["_confidence"] == 0.90


# ── 2. JSON serialization round-trip ──


class TestJsonRoundTrip:
    def test_prediction_raw_survives_json_serialization(self):
        """prediction_raw survives json.dumps/json.loads."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })
        result = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )

        # Round-trip through JSON
        serialized = json.dumps(result, default=str)
        deserialized = json.loads(serialized)

        assert "prediction_raw" in deserialized
        assert deserialized["prediction_raw"]["name"]["_confidence"] == 0.9
        assert deserialized["prediction_raw"]["price"]["_confidence"] == 0.3


# ── 3. update_from_comparison_result with confidence ──


class TestUpdateFromComparisonResultConfidence:
    def test_accumulates_confidence_from_prediction_raw(self):
        """update_from_comparison_result extracts confidence from prediction_raw."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })
        comparison = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )

        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator.update_from_comparison_result(comparison)
        result = evaluator.compute()

        assert result.confidence_metrics is not None
        assert result.confidence_metrics["overall"]["auroc"]["value"] is not None
        assert result.confidence_metrics["coverage"]["fields_with_confidence"] > 0

    def test_matches_update_path(self):
        """update_from_comparison_result produces same confidence metrics as update()."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred_json = {
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        }
        pred = Product.from_json(pred_json)

        # Path A: update()
        eval_a = BulkStructuredModelEvaluator(target_schema=Product)
        eval_a.update(gt, pred)
        result_a = eval_a.compute()

        # Path B: update_from_comparison_result()
        comparison = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )
        eval_b = BulkStructuredModelEvaluator(target_schema=Product)
        eval_b.update_from_comparison_result(comparison)
        result_b = eval_b.compute()

        # Confidence metrics should match
        assert (
            result_a.confidence_metrics["overall"]
            == result_b.confidence_metrics["overall"]
        )
        assert (
            result_a.confidence_metrics["coverage"]
            == result_b.confidence_metrics["coverage"]
        )

    def test_partial_confidence(self):
        """update_from_comparison_result handles partial confidence correctly."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": 29.99,  # no confidence
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })
        comparison = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )

        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator.update_from_comparison_result(comparison)
        result = evaluator.compute()

        cov = result.confidence_metrics["coverage"]
        assert cov["fields_with_confidence"] == 2
        assert cov["fields_total"] == 3

    def test_no_prediction_raw_no_confidence(self):
        """Without prediction_raw, no confidence metrics are produced."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product(name="Widget", price=29.99, sku="ABC123")
        comparison = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )

        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator.update_from_comparison_result(comparison)
        result = evaluator.compute()

        # Should have coverage from field_comparisons but zero confidence
        assert result.confidence_metrics is not None
        assert result.confidence_metrics["coverage"]["fields_with_confidence"] == 0

    def test_prediction_confidences_absent_fallback(self):
        """Replay of pre-cache JSONL: no prediction_confidences key, only prediction_raw.

        Simulates a comparison_result captured before the
        ``prediction_confidences`` cache existed. The accumulator must
        fall back to walking ``prediction_raw`` via
        ``process_rich_values`` and still emit
        confidence metrics.
        """
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })
        comparison = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )
        assert "prediction_confidences" in comparison
        assert "prediction_raw" in comparison
        # Drop the cache to force the fallback walk.
        del comparison["prediction_confidences"]

        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator.update_from_comparison_result(comparison)
        result = evaluator.compute()

        assert result.confidence_metrics is not None
        assert result.confidence_metrics["coverage"]["fields_with_confidence"] == 3
        assert result.confidence_metrics["coverage"]["fields_total"] == 3

    def test_multiple_docs_accumulate(self):
        """Multiple update_from_comparison_result calls accumulate confidence."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")

        pred1 = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC123", "_confidence": 0.7},
        })
        pred2 = Product.from_json({
            "name": {"_value": "Wrong", "_confidence": 0.2},
            "price": {"_value": 99.99, "_confidence": 0.15},
            "sku": {"_value": "XYZ", "_confidence": 0.1},
        })

        comp1 = gt.compare_with(pred1, include_confusion_matrix=True, document_field_comparisons=True)
        comp2 = gt.compare_with(pred2, include_confusion_matrix=True, document_field_comparisons=True)

        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator.update_from_comparison_result(comp1)
        evaluator.update_from_comparison_result(comp2)
        result = evaluator.compute()

        assert result.confidence_metrics["coverage"]["fields_with_confidence"] == 6
        assert result.confidence_metrics["coverage"]["fields_total"] == 6
        # With both matches and non-matches, AUROC should be computable
        assert result.confidence_metrics["overall"]["auroc"]["value"] is not None


# ── 4. aggregate_from_comparisons ──


class TestAggregateFromComparisons:
    def test_produces_confidence_metrics(self):
        """aggregate_from_comparisons works end-to-end with confidence."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")

        pred1 = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC123", "_confidence": 0.7},
        })
        pred2 = Product.from_json({
            "name": {"_value": "Wrong", "_confidence": 0.2},
            "price": {"_value": 99.99, "_confidence": 0.15},
            "sku": {"_value": "XYZ", "_confidence": 0.1},
        })

        results = [
            gt.compare_with(pred1, include_confusion_matrix=True, document_field_comparisons=True),
            gt.compare_with(pred2, include_confusion_matrix=True, document_field_comparisons=True),
        ]

        evaluation = aggregate_from_comparisons(results)
        assert evaluation.confidence_metrics is not None
        assert evaluation.confidence_metrics["overall"]["auroc"]["value"] is not None


# ── 5. JSONL round-trip ──


class TestJsonlRoundTrip:
    def test_jsonl_preserves_confidence(self, tmp_path):
        """Write comparison results to JSONL, read back, aggregate. Confidence survives."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })

        comparison = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )

        # Write to JSONL
        jsonl_path = tmp_path / "results.jsonl"
        with open(jsonl_path, "w") as f:
            record = {"doc_id": "doc_0", "comparison_result": comparison}
            f.write(json.dumps(record, default=str) + "\n")

        # Read back and aggregate
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                evaluator.update_from_comparison_result(record["comparison_result"])

        result = evaluator.compute()
        assert result.confidence_metrics is not None
        assert result.confidence_metrics["coverage"]["fields_with_confidence"] == 3
        assert result.confidence_metrics["coverage"]["fields_total"] == 3


# ── 6. Nested model round-trip ──


class TestNestedRoundTrip:
    def test_nested_model_confidence_via_comparison_result(self):
        """Nested model confidence survives the comparison result path."""
        gt = Customer(
            name="Jane",
            address=Address(street="123 Main", city="Boston"),
        )
        pred = Customer.from_json({
            "name": {"_value": "Jane", "_confidence": 0.96},
            "address": {
                "street": {"_value": "123 Main St", "_confidence": 0.85},
                "city": {"_value": "Chicago", "_confidence": 0.40},
            },
        })

        comparison = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )

        evaluator = BulkStructuredModelEvaluator(target_schema=Customer)
        evaluator.update_from_comparison_result(comparison)
        result = evaluator.compute()

        assert result.confidence_metrics is not None
        fields = result.confidence_metrics["fields"]
        # Should have dot-notation paths
        assert any("address.street" in k for k in fields)
        assert any("address.city" in k for k in fields)


# ── 7. Full end-to-end: bulk update() vs JSONL replay via update_from_comparison_result() ──


class TestBulkVsJsonlReplay:
    def test_bulk_update_matches_jsonl_replay(self, tmp_path):
        """Run multiple docs through update(), save JSONL, replay via
        update_from_comparison_result(). Both paths must produce identical
        confidence metrics, confusion matrix metrics, and coverage."""

        gt1 = Product(name="Widget", price=29.99, sku="ABC123")
        gt2 = Product(name="Gadget", price=49.99, sku="DEF456")
        gt3 = Product(name="Doohickey", price=9.99, sku="GHI789")

        pred1_json = {
            "name": {"_value": "Widget", "_confidence": 0.95},
            "price": {"_value": 29.99, "_confidence": 0.88},
            "sku": {"_value": "ABC123", "_confidence": 0.92},
        }
        pred2_json = {
            "name": {"_value": "Wrong", "_confidence": 0.20},
            "price": {"_value": 49.99, "_confidence": 0.85},
            "sku": {"_value": "DEF456", "_confidence": 0.90},
        }
        pred3_json = {
            "name": {"_value": "Doohickey", "_confidence": 0.88},
            "price": {"_value": 99.99, "_confidence": 0.30},
            "sku": "GHI789",  # no confidence on this field
        }

        pred1 = Product.from_json(pred1_json)
        pred2 = Product.from_json(pred2_json)
        pred3 = Product.from_json(pred3_json)

        gts = [gt1, gt2, gt3]
        preds = [pred1, pred2, pred3]

        # Path A: bulk update() with JSONL output
        jsonl_path = tmp_path / "results.jsonl"
        eval_a = BulkStructuredModelEvaluator(
            target_schema=Product,
            individual_results_jsonl=str(jsonl_path),
        )
        for gt, pred in zip(gts, preds):
            eval_a.update(gt, pred)
        result_a = eval_a.compute()

        # Path B: replay from JSONL via update_from_comparison_result()
        eval_b = BulkStructuredModelEvaluator(target_schema=Product)
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                eval_b.update_from_comparison_result(
                    record["comparison_result"],
                    doc_id=record["doc_id"],
                )
        result_b = eval_b.compute()

        # Confusion matrix metrics must match
        assert result_a.metrics == result_b.metrics

        # Confidence metrics must match
        assert result_a.confidence_metrics is not None
        assert result_b.confidence_metrics is not None

        assert (
            result_a.confidence_metrics["overall"]
            == result_b.confidence_metrics["overall"]
        )
        assert (
            result_a.confidence_metrics["coverage"]
            == result_b.confidence_metrics["coverage"]
        )

        # Per-field confidence metrics must match
        assert (
            set(result_a.confidence_metrics["fields"].keys())
            == set(result_b.confidence_metrics["fields"].keys())
        )
        for field in result_a.confidence_metrics["fields"]:
            assert (
                result_a.confidence_metrics["fields"][field]
                == result_b.confidence_metrics["fields"][field]
            ), f"Field {field} metrics differ between bulk and JSONL replay"

    def test_bulk_update_matches_aggregate_from_comparisons(self):
        """aggregate_from_comparisons() produces same confidence metrics
        as bulk update() for the same data."""

        gt = Product(name="Widget", price=29.99, sku="ABC123")

        pred1 = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })
        pred2 = Product.from_json({
            "name": {"_value": "Wrong", "_confidence": 0.2},
            "price": {"_value": 29.99, "_confidence": 0.85},
            "sku": {"_value": "XYZ", "_confidence": 0.1},
        })

        # Path A: bulk update()
        eval_a = BulkStructuredModelEvaluator(target_schema=Product)
        eval_a.update(gt, pred1)
        eval_a.update(gt, pred2)
        result_a = eval_a.compute()

        # Path B: aggregate_from_comparisons()
        comparisons = [
            gt.compare_with(
                pred1,
                include_confusion_matrix=True,
                document_field_comparisons=True,
            ),
            gt.compare_with(
                pred2,
                include_confusion_matrix=True,
                document_field_comparisons=True,
            ),
        ]
        result_b = aggregate_from_comparisons(comparisons)

        # Confidence metrics must match
        assert (
            result_a.confidence_metrics["overall"]
            == result_b.confidence_metrics["overall"]
        )
        assert (
            result_a.confidence_metrics["coverage"]
            == result_b.confidence_metrics["coverage"]
        )


# ── 8. Extras (_bbox, _source_span) survive the JSONL round-trip ──


class TestExtrasRoundTrip:
    """Future accumulators (BBoxMAPAccumulator, source-span attribution, ...)
    are the design motivation for ``prediction_raw``. The data contract is
    that arbitrary ``_*``-prefixed metadata round-trips through JSONL without
    loss, so a downstream accumulator can consume it later. Confidence is the
    only metric using extras today; these tests pin the contract for the
    rest."""

    def test_bbox_extras_survive_compare_with(self):
        """compare_with preserves _bbox in prediction_raw alongside _confidence."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {
                "_value": "Widget",
                "_confidence": 0.9,
                "_bbox": [0.1, 0.2, 0.3, 0.4],
            },
            "price": {
                "_value": 29.99,
                "_confidence": 0.8,
                "_bbox": [0.5, 0.6, 0.7, 0.8],
            },
            "sku": {"_value": "ABC123", "_confidence": 0.7},
        })
        result = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )

        raw = result["prediction_raw"]
        assert raw["name"]["_bbox"] == [0.1, 0.2, 0.3, 0.4]
        assert raw["price"]["_bbox"] == [0.5, 0.6, 0.7, 0.8]
        # _confidence still flows through as well
        assert raw["name"]["_confidence"] == 0.9

    def test_bbox_survives_jsonl_serialization(self):
        """Serialize a comparison result containing _bbox to JSONL, read it
        back, and confirm the metadata is still accessible. This is the
        primary path a future BBoxMAPAccumulator would walk."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {
                "_value": "Widget",
                "_confidence": 0.9,
                "_bbox": [0.1, 0.2, 0.3, 0.4],
                "_source_span": [10, 16],
            },
            "price": 29.99,
            "sku": {"_value": "ABC123", "_confidence": 0.7},
        })
        result = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )

        # JSONL is what production pipelines persist comparisons as
        line = json.dumps({"doc_id": "doc1", "comparison_result": result}, default=str)
        record = json.loads(line)

        raw = record["comparison_result"]["prediction_raw"]
        assert raw["name"]["_bbox"] == [0.1, 0.2, 0.3, 0.4]
        assert raw["name"]["_source_span"] == [10, 16]
        assert raw["price"] == 29.99
        assert raw["sku"]["_confidence"] == 0.7

    def test_bbox_does_not_break_update_from_comparison_result(self):
        """A comparison result carrying _bbox alongside _confidence must
        still aggregate cleanly through update_from_comparison_result —
        the confidence accumulator ignores unknown metadata, and the
        bulk evaluator must not fail on extras it doesn't yet consume."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {
                "_value": "Widget",
                "_confidence": 0.9,
                "_bbox": [0.1, 0.2, 0.3, 0.4],
            },
            "price": {
                "_value": 99.99,
                "_confidence": 0.3,
                "_bbox": [0.5, 0.6, 0.7, 0.8],
            },
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })
        comparison = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )

        # Round-trip through JSONL the way a real reduce step would
        line = json.dumps({"doc_id": "doc1", "comparison_result": comparison}, default=str)
        record = json.loads(line)

        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator.update_from_comparison_result(
            record["comparison_result"], doc_id=record["doc_id"]
        )
        result = evaluator.compute()

        # Confidence still flows
        assert result.confidence_metrics is not None
        assert result.confidence_metrics["coverage"]["fields_with_confidence"] == 3
        # No errors — the extras are tolerated, not fatal
        assert result.errors == []
