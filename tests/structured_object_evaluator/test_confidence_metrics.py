"""
Tests for the confidence evaluation module.

Covers:
- Keyed pair extraction correctness (the critical join)
- Nested and double-nested field path correctness
- List field path correctness with Hungarian reordering
- Partial confidence coverage
- Per-field metric computation
- Metric result structure validation
- ECE bin correctness
- Single-doc compare_with integration
- Bulk evaluator accumulation, levels, state, and merge
"""

from typing import List

import pytest

from stickler.comparators import (
    ExactComparator,
    LevenshteinComparator,
    NumericComparator,
)
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.confidence import (
    AUROCMetric,
    BrierScoreMetric,
    ConfidenceCalculator,
    ConfidencePair,
    ECEMetric,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

# ── Helper ──

def cp(is_match, confidence, similarity=0.0):
    """Shorthand for creating ConfidencePair."""
    return ConfidencePair(is_match=is_match, confidence=confidence, similarity=similarity)


# ── Test models ──


class Product(StructuredModel):
    name: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
    price: float = ComparableField(comparator=NumericComparator(), threshold=0.5)
    sku: str = ComparableField(comparator=ExactComparator(), threshold=1.0)


class Address(StructuredModel):
    street: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.7)
    city: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.7)


class ContactInfo(StructuredModel):
    email: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    address: Address = ComparableField()


class Customer(StructuredModel):
    name: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.8)
    contact: ContactInfo = ComparableField()


class Order(StructuredModel):
    order_id: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    items: List[Product] = ComparableField()


# ── 1. Keyed pair extraction correctness ──


class TestKeyedPairExtraction:
    def test_basic_field_pairing(self):
        """Verify exact field paths, match labels, confidence, and similarity."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.6},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })

        result = gt.compare_with(pred, document_field_comparisons=True)
        calc = ConfidenceCalculator()
        keyed = calc.extract_keyed_pairs(result, pred)

        assert set(keyed.keys()) == {"name", "price", "sku"}

        # name matches (Widget == Widget), conf 0.9
        assert len(keyed["name"]) == 1
        p = keyed["name"][0]
        assert p.is_match is True
        assert p.confidence == 0.9
        assert p.similarity > 0.0  # should have the raw score

        # price doesn't match (29.99 vs 99.99), conf 0.6
        assert keyed["price"][0].is_match is False
        assert keyed["price"][0].confidence == 0.6

        # sku matches exactly, conf 0.8
        assert keyed["sku"][0].is_match is True
        assert keyed["sku"][0].confidence == 0.8

    def test_similarity_score_captured(self):
        """The similarity field carries the raw comparator score."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "XYZ999", "_confidence": 0.5},
        })

        result = gt.compare_with(pred, document_field_comparisons=True)
        calc = ConfidenceCalculator()
        keyed = calc.extract_keyed_pairs(result, pred)

        # name: exact match -> similarity ~1.0
        assert keyed["name"][0].similarity >= 0.99
        # sku: completely wrong -> similarity 0.0 (ExactComparator)
        assert keyed["sku"][0].similarity == 0.0

    def test_raises_without_field_comparisons(self):
        """extract_keyed_pairs requires field_comparisons in the result."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
        })
        result = gt.compare_with(pred)
        calc = ConfidenceCalculator()
        with pytest.raises(ValueError, match="No field comparisons"):
            calc.extract_keyed_pairs(result, pred)


# ── 2. Nested field path correctness ──


class TestNestedPaths:
    def test_single_nesting(self):
        """Nested fields use dot notation: address.street, address.city."""
        gt = ContactInfo(
            email="a@b.com",
            address=Address(street="123 Main St", city="Boston"),
        )
        pred = ContactInfo.from_json({
            "email": {"_value": "a@b.com", "_confidence": 0.95},
            "address": {
                "street": {"_value": "123 Main St", "_confidence": 0.85},
                "city": {"_value": "Chicago", "_confidence": 0.40},
            },
        })

        result = gt.compare_with(pred, document_field_comparisons=True)
        calc = ConfidenceCalculator()
        keyed = calc.extract_keyed_pairs(result, pred)

        assert "email" in keyed
        assert "address.street" in keyed
        assert "address.city" in keyed
        assert keyed["address.street"][0].is_match is True
        assert keyed["address.city"][0].is_match is False

    def test_double_nesting(self):
        """Double-nested: contact.address.street, contact.address.city."""
        gt = Customer(
            name="Jane",
            contact=ContactInfo(
                email="jane@test.com",
                address=Address(street="456 Oak Ave", city="Boston"),
            ),
        )
        pred = Customer.from_json({
            "name": {"_value": "Jane", "_confidence": 0.96},
            "contact": {
                "email": {"_value": "jane@test.com", "_confidence": 0.90},
                "address": {
                    "street": {"_value": "456 Oak Avenue", "_confidence": 0.80},
                    "city": {"_value": "New York", "_confidence": 0.35},
                },
            },
        })

        result = gt.compare_with(pred, document_field_comparisons=True)
        calc = ConfidenceCalculator()
        keyed = calc.extract_keyed_pairs(result, pred)

        assert "name" in keyed
        assert "contact.email" in keyed
        assert "contact.address.street" in keyed
        assert "contact.address.city" in keyed

        assert keyed["contact.address.city"][0].is_match is False
        assert keyed["contact.address.city"][0].confidence == 0.35


# ── 3. List field paths with Hungarian reordering ──


class TestListPaths:
    def test_list_field_paths_use_prediction_indices(self):
        """List items use array notation with prediction indices."""
        gt = Order(
            order_id="ORD-1",
            items=[
                Product(name="Mouse", price=29.99, sku="MOU001"),
                Product(name="Keyboard", price=79.99, sku="KEY001"),
            ],
        )
        pred = Order.from_json({
            "order_id": {"_value": "ORD-1", "_confidence": 0.99},
            "items": [
                {
                    "name": {"_value": "Keyboard", "_confidence": 0.92},
                    "price": {"_value": 79.99, "_confidence": 0.88},
                    "sku": {"_value": "KEY001", "_confidence": 0.95},
                },
                {
                    "name": {"_value": "Mouse", "_confidence": 0.90},
                    "price": {"_value": 29.99, "_confidence": 0.85},
                    "sku": {"_value": "WRONG", "_confidence": 0.30},
                },
            ],
        })

        result = gt.compare_with(pred, document_field_comparisons=True)
        calc = ConfidenceCalculator()
        keyed = calc.extract_keyed_pairs(result, pred)

        assert "order_id" in keyed
        item_keys = [k for k in keyed if k.startswith("items[")]
        assert len(item_keys) > 0
        for k in item_keys:
            assert "[0]." in k or "[1]." in k

    def test_hungarian_reorder_preserves_confidence_match_pairing(self):
        """After Hungarian matching, confidence must stay paired with the
        prediction it came from, not the ground-truth slot it landed in.

        gt[0]=Mouse, gt[1]=Keyboard. pred[0]=Keyboard (conf 0.92),
        pred[1]=Mouse (conf 0.90). A bug that re-pairs pred[0]'s
        confidence with gt[0]'s match outcome would be invisible in the
        previous assertion style (both are matches). This test also
        includes a wrong sku on pred[1] so is_match=False has to line up
        with the 0.30 confidence, not Keyboard's 0.95.
        """
        gt = Order(
            order_id="ORD-1",
            items=[
                Product(name="Mouse", price=29.99, sku="MOU001"),
                Product(name="Keyboard", price=79.99, sku="KEY001"),
            ],
        )
        pred = Order.from_json({
            "order_id": {"_value": "ORD-1", "_confidence": 0.99},
            "items": [
                {
                    "name": {"_value": "Keyboard", "_confidence": 0.92},
                    "price": {"_value": 79.99, "_confidence": 0.88},
                    "sku": {"_value": "KEY001", "_confidence": 0.95},
                },
                {
                    "name": {"_value": "Mouse", "_confidence": 0.90},
                    "price": {"_value": 29.99, "_confidence": 0.85},
                    "sku": {"_value": "WRONG", "_confidence": 0.30},
                },
            ],
        })

        result = gt.compare_with(pred, document_field_comparisons=True)
        calc = ConfidenceCalculator()
        keyed = calc.extract_keyed_pairs(result, pred)

        # pred[0] = Keyboard (matches gt[1]=Keyboard). Confidence 0.92
        # must follow pred[0] into the items[0] slot.
        assert keyed["items[0].name"][0].confidence == 0.92
        assert keyed["items[0].name"][0].is_match is True
        assert keyed["items[0].price"][0].confidence == 0.88
        assert keyed["items[0].price"][0].is_match is True
        assert keyed["items[0].sku"][0].confidence == 0.95
        assert keyed["items[0].sku"][0].is_match is True

        # pred[1] = Mouse (matches gt[0]=Mouse). The sku field is wrong
        # (WRONG vs MOU001); is_match=False must pair with conf 0.30,
        # not with Keyboard's sku conf of 0.95.
        assert keyed["items[1].name"][0].confidence == 0.90
        assert keyed["items[1].name"][0].is_match is True
        assert keyed["items[1].sku"][0].confidence == 0.30
        assert keyed["items[1].sku"][0].is_match is False


# ── 4. Partial confidence coverage ──


class TestPartialCoverage:
    def test_fields_without_confidence_excluded(self):
        """Fields without confidence are not in keyed pairs."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": 29.99,  # no confidence
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })

        result = gt.compare_with(pred, document_field_comparisons=True)
        calc = ConfidenceCalculator()
        keyed = calc.extract_keyed_pairs(result, pred)

        assert "name" in keyed
        assert "sku" in keyed
        assert "price" not in keyed


# ── 5. Per-field metric computation ──


class TestComputeMetrics:
    def test_overall_and_per_field_structure(self):
        """compute_metrics returns overall + fields with correct structure."""
        keyed = {
            "field_a": [cp(True, 0.9), cp(False, 0.3)],
            "field_b": [cp(True, 0.8), cp(True, 0.7)],
        }
        calc = ConfidenceCalculator(metrics=[AUROCMetric()])
        result = calc.compute_metrics(keyed)

        assert "overall" in result
        assert "fields" in result
        assert "auroc" in result["overall"]
        assert "field_a" in result["fields"]
        assert "field_b" in result["fields"]
        assert "auroc" in result["fields"]["field_a"]

    def test_overall_uses_all_pairs(self):
        """Overall metric is computed from all pairs across all fields."""
        keyed = {
            "a": [cp(True, 0.95)],
            "b": [cp(False, 0.10)],
        }
        calc = ConfidenceCalculator(metrics=[AUROCMetric()])
        result = calc.compute_metrics(keyed)
        assert result["overall"]["auroc"]["value"] == 1.0

    def test_per_field_single_class_returns_none(self):
        """Per-field AUROC is None when a field only has one class."""
        keyed = {
            "always_right": [cp(True, 0.9), cp(True, 0.8)],
            "mixed": [cp(True, 0.9), cp(False, 0.2)],
        }
        calc = ConfidenceCalculator(metrics=[AUROCMetric()])
        result = calc.compute_metrics(keyed)

        assert result["fields"]["always_right"]["auroc"]["value"] is None
        assert result["fields"]["mixed"]["auroc"]["value"] is not None


# ── 6. Metric result structure validation ──


class TestMetricResultStructure:
    def test_auroc_result_shape(self):
        result = AUROCMetric().compute([cp(True, 0.9), cp(False, 0.1)])
        assert "value" in result
        assert isinstance(result["value"], float)

    def test_auroc_empty_returns_none(self):
        assert AUROCMetric().compute([])["value"] is None

    def test_auroc_single_class_returns_none(self):
        assert AUROCMetric().compute([cp(True, 0.9), cp(True, 0.8)])["value"] is None

    def test_brier_result_shape(self):
        result = BrierScoreMetric().compute([cp(True, 0.9), cp(False, 0.1)])
        assert "value" in result
        assert isinstance(result["value"], float)

    def test_brier_empty_returns_none(self):
        assert BrierScoreMetric().compute([])["value"] is None

    def test_brier_matches_closed_form(self):
        # ((1 - 0.95)**2 + (0 - 0.10)**2) / 2 == 0.00625
        pairs = [
            ConfidencePair(is_match=True, confidence=0.95, similarity=1.0),
            ConfidencePair(is_match=False, confidence=0.10, similarity=0.0),
        ]
        assert BrierScoreMetric().compute(pairs)["value"] == pytest.approx(0.00625)

    def test_ece_result_shape(self):
        result = ECEMetric(n_bins=5).compute([cp(True, 0.9), cp(False, 0.1)])
        assert "value" in result
        assert "bins" in result
        assert len(result["bins"]) == 5

    def test_ece_empty_returns_none_and_empty_bins(self):
        result = ECEMetric().compute([])
        assert result["value"] is None
        assert result["bins"] == []

    def test_ece_bin_structure(self):
        result = ECEMetric(n_bins=10).compute([cp(True, 0.85), cp(False, 0.15)])
        for b in result["bins"]:
            assert "range" in b
            assert "count" in b
            assert "accuracy" in b
            assert "mean_confidence" in b
            assert len(b["range"]) == 2


# ── 7. ECE bin correctness ──


class TestECEBins:
    def test_pairs_land_in_correct_bins(self):
        """Hand-crafted pairs with known bin assignments."""
        pairs = [
            cp(False, 0.15),  # bin [0.1, 0.2)
            cp(True, 0.35),   # bin [0.3, 0.4)
            cp(True, 0.75),   # bin [0.7, 0.8)
            cp(False, 0.95),  # bin [0.9, 1.0]
        ]
        result = ECEMetric(n_bins=10).compute(pairs)
        bins = result["bins"]

        assert bins[1]["count"] == 1
        assert bins[1]["accuracy"] == 0.0
        assert abs(bins[1]["mean_confidence"] - 0.15) < 0.001

        assert bins[3]["count"] == 1
        assert bins[3]["accuracy"] == 1.0

        assert bins[7]["count"] == 1
        assert bins[7]["accuracy"] == 1.0

        assert bins[9]["count"] == 1
        assert bins[9]["accuracy"] == 0.0

        assert bins[0]["count"] == 0
        assert bins[5]["count"] == 0

    def test_ece_value_is_weighted_gap(self):
        """ECE = weighted average of |accuracy - mean_confidence| per bin."""
        pairs = [cp(True, 0.75), cp(False, 0.72)]
        result = ECEMetric(n_bins=10).compute(pairs)

        bin7 = result["bins"][7]
        assert bin7["count"] == 2
        assert bin7["accuracy"] == 0.5
        expected_mc = (0.75 + 0.72) / 2
        assert abs(bin7["mean_confidence"] - expected_mc) < 0.001

        expected_ece = abs(0.5 - expected_mc)
        assert abs(result["value"] - expected_ece) < 0.001

    def test_ece_bin_assignment_at_lower_boundary(self):
        # Pins bisect_right behaviour at confidence == 0.0: lands in bin 0.
        result = ECEMetric(n_bins=10).compute(
            [ConfidencePair(is_match=False, confidence=0.0, similarity=0.0)]
        )
        assert result["bins"][0]["count"] == 1
        assert all(b["count"] == 0 for b in result["bins"][1:])

    def test_ece_bin_assignment_at_upper_boundary(self):
        # Pins the min(idx, n_bins-1) clamp: confidence == 1.0 lands in
        # the last bin rather than overflowing past it.
        result = ECEMetric(n_bins=10).compute(
            [ConfidencePair(is_match=True, confidence=1.0, similarity=1.0)]
        )
        assert result["bins"][9]["count"] == 1
        assert all(b["count"] == 0 for b in result["bins"][:9])


# ── 8. Single-doc compare_with integration ──


class TestSingleDocIntegration:
    pytestmark = pytest.mark.filterwarnings(
        "ignore:Single-document confidence metrics:UserWarning"
    )

    def test_confidence_metrics_in_result(self):
        """compare_with with add_confidence_metrics populates confidence_metrics."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })

        result = gt.compare_with(
            pred, add_confidence_metrics=True, document_field_comparisons=True,
        )

        assert "confidence_metrics" in result
        cm = result["confidence_metrics"]
        assert "overall" in cm
        assert "fields" in cm
        assert "auroc" in cm["overall"]
        assert "name" in cm["fields"]
        assert "price" in cm["fields"]
        assert "sku" in cm["fields"]

    def test_confidence_metrics_absent_without_flag(self):
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({"name": {"_value": "Widget", "_confidence": 0.9}})
        result = gt.compare_with(pred, document_field_comparisons=True)
        assert "confidence_metrics" not in result

    def test_no_confidence_data_still_works(self):
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product(name="Widget", price=29.99, sku="ABC123")
        result = gt.compare_with(
            pred, add_confidence_metrics=True, document_field_comparisons=True,
        )
        cm = result.get("confidence_metrics")
        if cm is not None:
            assert cm["overall"]["auroc"]["value"] is None

    def test_nested_model_paths_in_single_doc(self):
        gt = Customer(
            name="Jane",
            contact=ContactInfo(
                email="j@t.com",
                address=Address(street="123 Main", city="Boston"),
            ),
        )
        pred = Customer.from_json({
            "name": {"_value": "Jane", "_confidence": 0.95},
            "contact": {
                "email": {"_value": "j@t.com", "_confidence": 0.90},
                "address": {
                    "street": {"_value": "123 Main", "_confidence": 0.85},
                    "city": {"_value": "Wrong City", "_confidence": 0.30},
                },
            },
        })

        result = gt.compare_with(
            pred, add_confidence_metrics=True, document_field_comparisons=True,
        )
        fields = result["confidence_metrics"]["fields"]
        assert "contact.address.street" in fields
        assert "contact.address.city" in fields


# ── 9. Bulk evaluator accumulation ──


class TestBulkAccumulation:
    def test_keyed_pairs_accumulate_across_documents(self):
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)

        gt1 = Product(name="Widget", price=29.99, sku="ABC")
        pred1 = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC", "_confidence": 0.7},
        })
        gt2 = Product(name="Gadget", price=49.99, sku="DEF")
        pred2 = Product.from_json({
            "name": {"_value": "Gadget", "_confidence": 0.85},
            "price": {"_value": 99.99, "_confidence": 0.4},
            "sku": {"_value": "DEF", "_confidence": 0.95},
        })

        evaluator.update(gt1, pred1)
        evaluator.update(gt2, pred2)

        for field in ["name", "price", "sku"]:
            assert len(evaluator._accumulators[0]._keyed_pairs[field]) == 2
            assert all(isinstance(p, ConfidencePair) for p in evaluator._accumulators[0]._keyed_pairs[field])

    def test_bulk_metrics_match_manual_computation(self):
        gt1 = Product(name="Widget", price=29.99, sku="ABC")
        pred1 = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC", "_confidence": 0.8},
        })
        gt2 = Product(name="Gadget", price=49.99, sku="DEF")
        pred2 = Product.from_json({
            "name": {"_value": "Wrong", "_confidence": 0.2},
            "price": {"_value": 49.99, "_confidence": 0.85},
            "sku": {"_value": "DEF", "_confidence": 0.95},
        })

        metrics = [AUROCMetric(), BrierScoreMetric(), ECEMetric(n_bins=5)]
        evaluator = BulkStructuredModelEvaluator(
            target_schema=Product, confidence_metrics=metrics
        )
        evaluator.update(gt1, pred1)
        evaluator.update(gt2, pred2)
        bulk_result = evaluator.compute()

        calc = ConfidenceCalculator(
            metrics=[AUROCMetric(), BrierScoreMetric(), ECEMetric(n_bins=5)]
        )
        r1 = gt1.compare_with(pred1, document_field_comparisons=True)
        r2 = gt2.compare_with(pred2, document_field_comparisons=True)
        keyed1 = calc.extract_keyed_pairs(r1, pred1)
        keyed2 = calc.extract_keyed_pairs(r2, pred2)

        merged_keyed = {}
        for k, v in keyed1.items():
            merged_keyed.setdefault(k, []).extend(v)
        for k, v in keyed2.items():
            merged_keyed.setdefault(k, []).extend(v)
        manual_result = calc.compute_metrics(merged_keyed)

        for metric_name in ("auroc", "brier_score", "ece"):
            assert (
                bulk_result.confidence_metrics["overall"][metric_name]["value"]
                == manual_result["overall"][metric_name]["value"]
            )

    def test_no_confidence_data_returns_coverage_only(self):
        """When update() runs on docs without confidence, we still report
        coverage (0/N) so users can see they have no confidence data."""
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred = Product(name="Widget", price=29.99, sku="ABC")
        evaluator.update(gt, pred)
        result = evaluator.compute()
        # confidence_metrics should be present with coverage showing zero
        assert result.confidence_metrics is not None
        assert result.confidence_metrics["coverage"]["fields_with_confidence"] == 0
        assert result.confidence_metrics["coverage"]["fields_total"] > 0
        assert result.confidence_metrics["coverage"]["ratio"] == 0.0
        # Overall metrics are None (no pairs to compute from)
        assert result.confidence_metrics["overall"]["auroc"]["value"] is None

    def test_completely_unprocessed_returns_none(self):
        """An evaluator that never ran update() returns confidence_metrics=None."""
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        result = evaluator.compute()
        assert result.confidence_metrics is None


# ── 10. State serialization round-trip ──


class TestStateSerialization:
    def test_keyed_pairs_survive_round_trip(self):
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC", "_confidence": 0.8},
        })
        evaluator.update(gt, pred)
        state = evaluator.get_state()

        evaluator2 = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator2.load_state(state)

        assert evaluator2._accumulators[0]._keyed_pairs == evaluator._accumulators[0]._keyed_pairs

    def test_compute_after_load_matches_original(self):
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC", "_confidence": 0.8},
        })
        evaluator.update(gt, pred)
        original = evaluator.compute()

        evaluator2 = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator2.load_state(evaluator.get_state())
        restored = evaluator2.compute()

        assert original.confidence_metrics["overall"] == restored.confidence_metrics["overall"]


# ── 11. Merge preserves keyed pair integrity ──


class TestMerge:
    def test_merge_equals_single_pass(self):
        gts = [
            Product(name="Widget", price=29.99, sku="ABC"),
            Product(name="Gadget", price=49.99, sku="DEF"),
            Product(name="Doohickey", price=9.99, sku="GHI"),
            Product(name="Thingamajig", price=99.99, sku="JKL"),
        ]
        preds = [
            Product.from_json({"name": {"_value": "Widget", "_confidence": 0.9}, "price": {"_value": 29.99, "_confidence": 0.8}, "sku": {"_value": "ABC", "_confidence": 0.7}}),
            Product.from_json({"name": {"_value": "Wrong", "_confidence": 0.2}, "price": {"_value": 49.99, "_confidence": 0.85}, "sku": {"_value": "DEF", "_confidence": 0.95}}),
            Product.from_json({"name": {"_value": "Doohickey", "_confidence": 0.88}, "price": {"_value": 9.99, "_confidence": 0.92}, "sku": {"_value": "WRONG", "_confidence": 0.15}}),
            Product.from_json({"name": {"_value": "Thingamajig", "_confidence": 0.91}, "price": {"_value": 999.99, "_confidence": 0.25}, "sku": {"_value": "JKL", "_confidence": 0.93}}),
        ]

        single = BulkStructuredModelEvaluator(target_schema=Product)
        for gt, pred in zip(gts, preds):
            single.update(gt, pred)

        wa = BulkStructuredModelEvaluator(target_schema=Product)
        wb = BulkStructuredModelEvaluator(target_schema=Product)
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            (wa if i < 2 else wb).update(gt, pred)

        wa.merge_state(wb.get_state())

        assert single.compute().confidence_metrics["overall"] == wa.compute().confidence_metrics["overall"]
        assert set(single.compute().confidence_metrics["fields"].keys()) == set(wa.compute().confidence_metrics["fields"].keys())

    def test_merge_accumulates_field_pairs(self):
        wa = BulkStructuredModelEvaluator(target_schema=Product)
        wb = BulkStructuredModelEvaluator(target_schema=Product)

        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC", "_confidence": 0.7},
        })

        wa.update(gt, pred)
        wb.update(gt, pred)
        wa.merge_state(wb.get_state())

        for field in ["name", "price", "sku"]:
            assert len(wa._accumulators[0]._keyed_pairs[field]) == 2


# ── 12. Multiple metrics in bulk evaluator ──


class TestMultipleMetrics:
    def test_bulk_with_multiple_metrics(self):
        evaluator = BulkStructuredModelEvaluator(
            target_schema=Product,
            confidence_metrics=[AUROCMetric(), BrierScoreMetric(), ECEMetric(n_bins=5)],
        )
        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC", "_confidence": 0.8},
        })
        evaluator.update(gt, pred)
        result = evaluator.compute()

        overall = result.confidence_metrics["overall"]
        assert "auroc" in overall
        assert "brier_score" in overall
        assert "ece" in overall
        assert "bins" in overall["ece"]
        assert len(overall["ece"]["bins"]) == 5


# ── 13. Coverage tracking ──


class TestCoverage:
    pytestmark = pytest.mark.filterwarnings(
        "ignore:Single-document confidence metrics:UserWarning"
    )

    def test_single_doc_coverage(self):
        """Single-doc result includes coverage stats."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": 29.99,  # no confidence
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })
        result = gt.compare_with(
            pred, add_confidence_metrics=True, document_field_comparisons=True,
        )
        cov = result["confidence_metrics"]["coverage"]
        assert cov["fields_with_confidence"] == 2
        assert cov["fields_total"] == 3
        assert abs(cov["ratio"] - 2 / 3) < 0.001

    def test_full_coverage(self):
        """All fields have confidence -> ratio 1.0."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC123", "_confidence": 0.7},
        })
        result = gt.compare_with(
            pred, add_confidence_metrics=True, document_field_comparisons=True,
        )
        cov = result["confidence_metrics"]["coverage"]
        assert cov["fields_with_confidence"] == 3
        assert cov["fields_total"] == 3
        assert cov["ratio"] == 1.0

    def test_bulk_coverage_accumulates(self):
        """Bulk evaluator accumulates coverage across documents."""
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)

        gt = Product(name="Widget", price=29.99, sku="ABC")

        # Doc 1: 2 of 3 fields have confidence
        pred1 = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": 29.99,
            "sku": {"_value": "ABC", "_confidence": 0.8},
        })
        # Doc 2: 3 of 3 fields have confidence
        pred2 = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC", "_confidence": 0.7},
        })

        evaluator.update(gt, pred1)
        evaluator.update(gt, pred2)
        result = evaluator.compute()

        cov = result.confidence_metrics["coverage"]
        assert cov["fields_with_confidence"] == 5  # 2 + 3
        assert cov["fields_total"] == 6  # 3 + 3
        assert abs(cov["ratio"] - 5 / 6) < 0.001

    def test_coverage_survives_state_round_trip(self):
        """Coverage counts survive get_state/load_state."""
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": 29.99,
            "sku": {"_value": "ABC", "_confidence": 0.8},
        })
        evaluator.update(gt, pred)

        state = evaluator.get_state()
        evaluator2 = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator2.load_state(state)

        r1 = evaluator.compute()
        r2 = evaluator2.compute()
        assert r1.confidence_metrics["coverage"] == r2.confidence_metrics["coverage"]

    def test_coverage_merges_correctly(self):
        """Coverage counts merge across workers."""
        wa = BulkStructuredModelEvaluator(target_schema=Product)
        wb = BulkStructuredModelEvaluator(target_schema=Product)

        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred_partial = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": 29.99,
            "sku": {"_value": "ABC", "_confidence": 0.8},
        })
        pred_full = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC", "_confidence": 0.7},
        })

        wa.update(gt, pred_partial)  # 2/3
        wb.update(gt, pred_full)     # 3/3
        wa.merge_state(wb.get_state())

        result = wa.compute()
        cov = result.confidence_metrics["coverage"]
        assert cov["fields_with_confidence"] == 5
        assert cov["fields_total"] == 6


# -- 14. Error Capture at Review Budget --


class TestErrorCaptureAtBudget:
    def test_basic_computation(self):
        """Well-separated confidence: low-conf fields are errors, high-conf are correct."""
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )

        # 10 pairs: 3 errors with low confidence, 7 correct with high confidence
        pairs = [
            cp(False, 0.10), cp(False, 0.15), cp(False, 0.20),  # errors
            cp(True, 0.60), cp(True, 0.65), cp(True, 0.70),
            cp(True, 0.75), cp(True, 0.80), cp(True, 0.85), cp(True, 0.90),
        ]

        metric = ErrorCaptureAtBudgetMetric(budgets=[0.10, 0.30, 0.50])
        result = metric.compute(pairs)

        assert result["value"] is not None
        assert "budgets" in result

        # At 30% budget (3 fields), all 3 errors should be found
        b30 = result["budgets"]["0.30"]
        assert b30["fields_reviewed"] == 3
        assert b30["errors_found"] == 3
        assert b30["pct_errors_caught"] == 1.0
        assert b30["gain"] > 1.0

    def test_random_confidence_gain_near_one(self):
        """Random confidence should produce gain near 1.0."""
        import random as rng

        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )
        rng.seed(99)

        pairs = [
            cp(rng.random() < 0.3, rng.random())
            for _ in range(500)
        ]

        metric = ErrorCaptureAtBudgetMetric(budgets=[0.30])
        result = metric.compute(pairs)

        # Gain should be close to 1.0 for random confidence
        assert 0.7 < result["budgets"]["0.30"]["gain"] < 1.5

    def test_empty_pairs(self):
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )
        result = ErrorCaptureAtBudgetMetric().compute([])
        assert result["value"] is None
        assert result["budgets"] == {}

    def test_no_errors(self):
        """All correct: no errors to find."""
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )
        pairs = [cp(True, 0.5), cp(True, 0.6), cp(True, 0.7)]
        result = ErrorCaptureAtBudgetMetric().compute(pairs)
        assert result["value"] is None

    def test_custom_budgets(self):
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )
        pairs = [
            cp(False, 0.1), cp(False, 0.2),
            cp(True, 0.8), cp(True, 0.9),
        ]
        metric = ErrorCaptureAtBudgetMetric(budgets=[0.25, 0.50, 0.75])
        result = metric.compute(pairs)
        assert set(result["budgets"].keys()) == {"0.25", "0.50", "0.75"}

    def test_headline_is_middle_budget(self):
        """Headline value should be the gain at the middle budget level."""
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )
        pairs = [
            cp(False, 0.1), cp(False, 0.2), cp(False, 0.3),
            cp(True, 0.7), cp(True, 0.8), cp(True, 0.9),
            cp(True, 0.91), cp(True, 0.92), cp(True, 0.93), cp(True, 0.94),
        ]
        metric = ErrorCaptureAtBudgetMetric(budgets=[0.10, 0.30, 0.50])
        result = metric.compute(pairs)
        # Middle budget is 0.30
        assert result["value"] == result["budgets"]["0.30"]["gain"]

    def test_bulk_evaluator_integration(self):
        """ErrorCaptureAtBudgetMetric works through the bulk evaluator."""
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )

        evaluator = BulkStructuredModelEvaluator(
            target_schema=Product,
            confidence_metrics=[ErrorCaptureAtBudgetMetric(budgets=[0.30, 0.50])],
        )

        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred_good = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC", "_confidence": 0.7},
        })
        pred_bad = Product.from_json({
            "name": {"_value": "Wrong", "_confidence": 0.2},
            "price": {"_value": 99.99, "_confidence": 0.15},
            "sku": {"_value": "XYZ", "_confidence": 0.1},
        })

        evaluator.update(gt, pred_good)
        evaluator.update(gt, pred_bad)
        result = evaluator.compute()

        ecab = result.confidence_metrics["overall"]["error_capture_at_budget"]
        assert "budgets" in ecab
        assert "0.30" in ecab["budgets"]
        assert "0.50" in ecab["budgets"]


# -- 15. Input validation (Copilot PR review fixes) --


class TestInputValidation:
    def test_ece_rejects_zero_bins(self):
        from stickler.structured_object_evaluator.models.confidence import ECEMetric
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            ECEMetric(n_bins=0)

    def test_ece_rejects_negative_bins(self):
        from stickler.structured_object_evaluator.models.confidence import ECEMetric
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            ECEMetric(n_bins=-5)

    def test_ecab_rejects_zero_budget(self):
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )
        with pytest.raises(ValueError, match="must be in the range"):
            ErrorCaptureAtBudgetMetric(budgets=[0.0, 0.5])

    def test_ecab_rejects_empty_budgets(self):
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )
        with pytest.raises(ValueError, match="budgets must not be empty"):
            ErrorCaptureAtBudgetMetric(budgets=[])

    def test_ecab_rejects_negative_budget(self):
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )
        with pytest.raises(ValueError, match="must be in the range"):
            ErrorCaptureAtBudgetMetric(budgets=[-0.1])

    def test_ecab_rejects_budget_over_one(self):
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )
        with pytest.raises(ValueError, match="must be in the range"):
            ErrorCaptureAtBudgetMetric(budgets=[0.5, 1.5])

    def test_ecab_accepts_budget_of_one(self):
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )
        metric = ErrorCaptureAtBudgetMetric(budgets=[1.0])
        assert metric.budgets == [1.0]


# -- 16. Coverage accounts for docs without confidence --


class TestCoverageAccountsForAllDocs:
    def test_coverage_includes_docs_without_confidence(self):
        """Coverage totals should include fields from docs that have no confidence."""
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)

        gt = Product(name="Widget", price=29.99, sku="ABC")

        # Doc 1: has confidence (3 fields with, 3 total)
        pred_with = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC", "_confidence": 0.7},
        })

        # Doc 2: no confidence at all (0 fields with, 3 total)
        pred_without = Product(name="Widget", price=29.99, sku="ABC")

        evaluator.update(gt, pred_with)
        evaluator.update(gt, pred_without)
        result = evaluator.compute()

        cov = result.confidence_metrics["coverage"]
        assert cov["fields_with_confidence"] == 3  # only from doc 1
        assert cov["fields_total"] == 6  # 3 from each doc
        assert abs(cov["ratio"] - 0.5) < 0.01


# -- 17. Single-doc path: warning and configurable metrics --


class TestSingleDocWarningAndConfig:
    def test_add_confidence_metrics_emits_warning(self):
        """Single-doc confidence should warn that bulk is recommended."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC123", "_confidence": 0.7},
        })
        with pytest.warns(UserWarning, match="Single-document confidence metrics"):
            gt.compare_with(
                pred, add_confidence_metrics=True, document_field_comparisons=True,
            )

    def test_confidence_metrics_kwarg_configures_metrics(self):
        """Passing confidence_metrics=[...] to compare_with threads through."""
        from stickler.structured_object_evaluator.models.confidence import (
            AUROCMetric,
            BrierScoreMetric,
        )

        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })
        with pytest.warns(UserWarning):
            result = gt.compare_with(
                pred,
                add_confidence_metrics=True,
                document_field_comparisons=True,
                confidence_metrics=[AUROCMetric(), BrierScoreMetric()],
            )

        overall = result["confidence_metrics"]["overall"]
        assert "auroc" in overall
        assert "brier_score" in overall


# -- 18. Deterministic ECAB headline regardless of budget order --


class TestECABDeterministicHeadline:
    def test_unsorted_budgets_still_deterministic(self):
        """Headline gain should match regardless of input budget order."""
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )

        pairs = [
            cp(False, 0.10), cp(False, 0.15), cp(False, 0.20),
            cp(True, 0.60), cp(True, 0.70), cp(True, 0.80),
            cp(True, 0.85), cp(True, 0.90), cp(True, 0.95), cp(True, 0.99),
        ]

        sorted_metric = ErrorCaptureAtBudgetMetric(budgets=[0.10, 0.30, 0.50])
        unsorted_metric = ErrorCaptureAtBudgetMetric(budgets=[0.50, 0.10, 0.30])

        r1 = sorted_metric.compute(pairs)
        r2 = unsorted_metric.compute(pairs)

        # Headline should be the same
        assert r1["value"] == r2["value"]
        # Both should have budgets dict keyed by the same strings
        assert set(r1["budgets"].keys()) == set(r2["budgets"].keys())


# -- 19. default_metrics returns fresh instances --


class TestDefaultMetricsFactory:
    def test_each_call_returns_new_list(self):
        """default_metrics() must return independent instances so state doesn't leak."""
        from stickler.structured_object_evaluator.models.confidence import (
            default_metrics,
        )

        a = default_metrics()
        b = default_metrics()
        # Different list objects
        assert a is not b
        # Different metric instances inside (not shared references)
        assert a[0] is not b[0]


# -- 20. Deprecation shim: legacy `auroc_confidence_metric` result key --


class TestLegacyAurocKeyShim:
    """The pre-rename ``auroc_confidence_metric`` result key is still
    populated for one release so callers doing
    ``result["auroc_confidence_metric"]`` keep working on upgrade."""

    def test_legacy_key_present_with_deprecation(self):
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })
        with pytest.warns(DeprecationWarning, match="auroc_confidence_metric"):
            result = gt.compare_with(
                pred,
                add_confidence_metrics=True,
                document_field_comparisons=True,
            )

        assert "auroc_confidence_metric" in result
        legacy = result["auroc_confidence_metric"]
        nested = result["confidence_metrics"]["overall"]["auroc"]["value"]
        # Legacy field mirrors the new structured value when present, and
        # falls back to the pre-rename 0.5 sentinel when AUROC is undefined.
        if nested is None:
            assert legacy == 0.5
        else:
            assert legacy == nested

    def test_legacy_key_falls_back_to_half_when_undefined(self):
        """When all fields match (single class), AUROC is None; legacy key
        uses the pre-rename 0.5 sentinel."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC123", "_confidence": 0.7},
        })
        with pytest.warns(DeprecationWarning):
            result = gt.compare_with(
                pred,
                add_confidence_metrics=True,
                document_field_comparisons=True,
            )
        assert result["auroc_confidence_metric"] == 0.5
        assert result["confidence_metrics"]["overall"]["auroc"]["value"] is None

    def test_legacy_key_absent_without_flag(self):
        """Without add_confidence_metrics=True the legacy key is not added."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
        })
        result = gt.compare_with(pred, document_field_comparisons=True)
        assert "auroc_confidence_metric" not in result
        assert "confidence_metrics" not in result


# -- 21. Boundary: ConfidencePair rejects invalid inputs --


class TestConfidencePairValidation:
    """ConfidencePair is the single producer of pair objects for every
    metric, so validating at construction prevents NaN/out-of-range
    inputs from corrupting downstream Brier/AUROC/ECE calculations."""

    def test_nan_confidence_rejected(self):
        import math

        with pytest.raises(ValueError, match="confidence"):
            ConfidencePair(is_match=True, confidence=math.nan, similarity=0.5)

    def test_inf_confidence_rejected(self):
        import math

        with pytest.raises(ValueError, match="confidence"):
            ConfidencePair(is_match=True, confidence=math.inf, similarity=0.5)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            ConfidencePair(is_match=True, confidence=1.5, similarity=0.5)

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            ConfidencePair(is_match=True, confidence=-0.1, similarity=0.5)

    def test_similarity_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="similarity"):
            ConfidencePair(is_match=False, confidence=0.5, similarity=1.1)

    def test_boundaries_accepted(self):
        """Both zero and one are valid."""
        ConfidencePair(is_match=True, confidence=0.0, similarity=0.0)
        ConfidencePair(is_match=False, confidence=1.0, similarity=1.0)


# -- 22. Boundary: ECARB gain against actual reviewed fraction --


class TestECARBGainBaseline:
    """Gain is computed against the actual reviewed fraction (k/n) rather
    than the requested budget, so tight budgets on small datasets don't
    report spuriously inflated gains."""

    def test_small_n_does_not_inflate_gain(self):
        """n=1 with budget=0.1 forces k=1 (review 100% of data). Gain
        should reflect the actual 100% review, not the 10% budget."""
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )

        pairs = [cp(False, 0.1)]  # 1 field, 1 error
        metric = ErrorCaptureAtBudgetMetric(budgets=[0.10])
        result = metric.compute(pairs)
        budget_entry = result["budgets"]["0.10"]
        # Reviewed 100% of data → caught 100% of errors → gain should be 1.0,
        # not 10.0 (which is what comparing against the requested 0.10 budget
        # would produce).
        assert budget_entry["fields_reviewed"] == 1
        assert budget_entry["pct_errors_caught"] == 1.0
        assert budget_entry["pct_errors_random"] == 1.0
        assert budget_entry["gain"] == 1.0

    def test_k_exceeds_budget_uses_actual_fraction(self):
        """n=9 with budget=0.1 forces k=1 (review ~11%). Gain reports
        against the actual 1/9 fraction rather than the 10% budget."""
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )

        # 9 fields: 1 error at the lowest confidence, 8 correct at higher.
        pairs = [cp(False, 0.05)] + [cp(True, 0.9) for _ in range(8)]
        metric = ErrorCaptureAtBudgetMetric(budgets=[0.10])
        result = metric.compute(pairs)
        entry = result["budgets"]["0.10"]
        assert entry["fields_reviewed"] == 1
        # k=1 out of 9
        assert entry["pct_errors_random"] == pytest.approx(1 / 9)
        # Caught the single error
        assert entry["pct_errors_caught"] == 1.0
        # gain = 1.0 / (1/9) ≈ 9.0 — matches reality, not 10.0
        assert entry["gain"] == pytest.approx(9.0)

    def test_budget_one_reviews_everything(self):
        """Budget=1.0 reviews 100%, catches 100% of errors, gain=1.0."""
        from stickler.structured_object_evaluator.models.confidence import (
            ErrorCaptureAtBudgetMetric,
        )

        pairs = [cp(False, 0.1), cp(True, 0.9), cp(False, 0.2)]
        metric = ErrorCaptureAtBudgetMetric(budgets=[1.0])
        result = metric.compute(pairs)
        entry = result["budgets"]["1.00"]
        assert entry["fields_reviewed"] == 3
        assert entry["pct_errors_caught"] == 1.0
        assert entry["pct_errors_random"] == 1.0
        assert entry["gain"] == 1.0


# -- 23. Boundary: calculator skips list FN entries without a key --


class TestExtractSkipsNullKeyRows:
    """Field comparisons with actual_key=None arrive when a prediction
    has fewer list items than ground truth. They can't be joined to any
    confidence score, so they must be skipped rather than inflating
    fields_total."""

    def test_list_fn_rows_do_not_inflate_coverage(self):
        """Synthesize a field_comparisons list with a null-key row and
        confirm extract_from_dicts skips it."""
        calc = ConfidenceCalculator()
        field_comparisons = [
            {"actual_key": "name", "match": True, "score": 1.0},
            # Simulated list FN: pred is missing an item for this gt row
            {"actual_key": None, "match": False, "score": 0.0},
            {"actual_key": "sku", "match": False, "score": 0.0},
        ]
        confidences = {"name": 0.9}  # sku has no confidence

        extraction = calc.extract_from_dicts(field_comparisons, confidences)

        # fields_total counts only the two keyed rows, not the FN entry
        assert extraction.fields_total == 2
        assert extraction.fields_with_confidence == 1
        assert set(extraction.keyed_pairs.keys()) == {"name"}
        # Sanity: the None key didn't leak into the dict
        assert None not in extraction.keyed_pairs



# -- 24. compare_with auto-enables field_comparisons for confidence --


class TestConfidenceAutoEnablesFieldComparisons:
    """``add_confidence_metrics=True`` should be usable without the caller
    also remembering ``document_field_comparisons=True``; the underlying
    field-level join data is wired in automatically so the common case
    'just give me confidence metrics' no longer crashes."""

    def test_confidence_without_field_comparisons_flag(self):
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })

        # No document_field_comparisons flag on purpose.
        with pytest.warns(UserWarning, match="sanity check"):
            result = gt.compare_with(pred, add_confidence_metrics=True)

        assert "confidence_metrics" in result
        # Auto-enabled field_comparisons should be populated since the
        # calculator depends on them.
        assert "field_comparisons" in result
        coverage = result["confidence_metrics"]["coverage"]
        assert coverage["fields_with_confidence"] == 3


# -- 25. Bulk evaluator construction conflicts --


class TestBulkEvaluatorConstructionConflicts:
    """Passing both ``accumulators`` and ``confidence_metrics`` used to
    silently drop ``confidence_metrics``. Now it raises so the conflict
    shows up at construction time."""

    def test_both_kwargs_raises(self):
        from stickler.structured_object_evaluator.models.confidence.accumulator import (
            ConfidenceAccumulator,
        )

        with pytest.raises(ValueError, match="not both"):
            BulkStructuredModelEvaluator(
                target_schema=Product,
                confidence_metrics=[AUROCMetric()],
                accumulators=[ConfidenceAccumulator()],
            )

    def test_accumulators_alone_ok(self):
        from stickler.structured_object_evaluator.models.confidence.accumulator import (
            ConfidenceAccumulator,
        )

        ev = BulkStructuredModelEvaluator(
            target_schema=Product,
            accumulators=[ConfidenceAccumulator(metrics=[AUROCMetric()])],
        )
        assert len(ev._accumulators) == 1

    def test_confidence_metrics_alone_ok(self):
        ev = BulkStructuredModelEvaluator(
            target_schema=Product,
            confidence_metrics=[AUROCMetric(), BrierScoreMetric()],
        )
        # Defaults: ConfidenceAccumulator + AggregateConfusionMatrixAccumulator.
        assert [acc.name for acc in ev._accumulators] == [
            "confidence_metrics",
            "aggregate_metrics",
        ]

    def test_duplicate_accumulator_names_raise(self):
        """Two accumulators with the same .name silently overwrite each
        other in compute().accumulator_metrics; the constructor should
        surface that conflict instead."""
        from stickler.structured_object_evaluator.models.confidence.accumulator import (
            ConfidenceAccumulator,
        )

        with pytest.raises(ValueError, match="confidence_metrics"):
            BulkStructuredModelEvaluator(
                target_schema=Product,
                accumulators=[
                    ConfidenceAccumulator(),
                    ConfidenceAccumulator(metrics=[AUROCMetric()]),
                ],
            )


# -- 26. Per-accumulator error isolation --


class TestAccumulatorErrorIsolation:
    """A bug in a single post-comparison accumulator must not corrupt the
    confusion matrix of every doc it touches. Before the isolation fix
    a failing accumulator flowed into the outer except, which rolled up
    as ``fn += 1`` on the overall cm for that doc."""

    def test_failing_accumulator_does_not_increment_fn(self):
        from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
            PostComparisonAccumulator,
        )

        class BoomAccumulator(PostComparisonAccumulator):
            @property
            def name(self) -> str:
                return "boom"

            def reset(self):
                pass

            def accumulate(self, comparison_result, prediction_raw):
                raise RuntimeError("synthetic accumulator failure")

            def compute(self):
                return None

            def get_state(self):
                return {}

            def load_state(self, state):
                pass

            def merge_state(self, other_state):
                pass

        ev = BulkStructuredModelEvaluator(
            target_schema=Product,
            accumulators=[BoomAccumulator()],
        )

        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product(name="Widget", price=29.99, sku="ABC123")
        ev.update(gt, pred)

        # The comparison itself is a perfect match: all tp, no fn.
        assert ev._confusion_matrix["overall"]["fn"] == 0
        # The accumulator error was recorded with the offender named.
        assert len(ev._errors) == 1
        assert ev._errors[0]["accumulator"] == "boom"
        assert ev._errors[0]["error_type"] == "RuntimeError"
        # Processed count still advances so pipeline metrics are intact.
        assert ev._processed_count == 1


# -- 27. Accumulator warning on missing field_comparisons --


class TestAccumulatorMissingFieldComparisonsWarning:
    """When the caller supplies ``prediction_raw`` but forgets to enable
    ``document_field_comparisons``, the confidence accumulator now warns
    so the silent-zero-confidence failure mode is visible in logs."""

    def test_warns_when_prediction_raw_but_no_field_comparisons(self):
        from stickler.structured_object_evaluator.models.confidence.accumulator import (
            ConfidenceAccumulator,
        )

        acc = ConfidenceAccumulator()
        # Simulated comparison_result missing the field_comparisons key.
        comparison_result = {
            "overall_score": 1.0,
            "confusion_matrix": {"overall": {"tp": 1}, "fields": {}},
        }
        prediction_raw = {"name": {"_value": "Widget", "_confidence": 0.9}}

        with pytest.warns(UserWarning, match="document_field_comparisons"):
            acc.accumulate(comparison_result, prediction_raw)

    def test_no_warning_without_prediction_raw(self):
        import warnings

        from stickler.structured_object_evaluator.models.confidence.accumulator import (
            ConfidenceAccumulator,
        )

        acc = ConfidenceAccumulator()
        comparison_result = {"overall_score": 1.0}

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn warnings into errors
            # Should not warn when confidence data isn't expected.
            acc.accumulate(comparison_result, None)


# -- 28. ProcessEvaluation surfaces accumulator_metrics --


class TestProcessEvaluationAccumulatorMetrics:
    """The bulk evaluator now exposes every accumulator's output on
    ``ProcessEvaluation.accumulator_metrics`` so a future
    BBoxMAPAccumulator's results don't get dropped on the floor."""

    def test_accumulator_metrics_dict_populated(self):
        ev = BulkStructuredModelEvaluator(
            target_schema=Product,
            confidence_metrics=[AUROCMetric()],
        )
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 29.99, "_confidence": 0.8},
            "sku": {"_value": "ABC123", "_confidence": 0.7},
        })
        ev.update(gt, pred)
        result = ev.compute()

        assert result.accumulator_metrics is not None
        assert "confidence_metrics" in result.accumulator_metrics
        # Back-compat: the dedicated field mirrors the accumulator entry.
        # Using equality rather than identity because ProcessEvaluation is
        # a pydantic BaseModel and copies dict values on assignment.
        assert (
            result.confidence_metrics
            == result.accumulator_metrics["confidence_metrics"]
        )


# -- 29. save_metrics includes confidence + accumulator data --


class TestSaveMetricsIncludesAccumulatorData:
    def test_save_metrics_json_includes_confidence(self, tmp_path):
        import json as _json

        ev = BulkStructuredModelEvaluator(
            target_schema=Product,
            confidence_metrics=[AUROCMetric()],
        )
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"_value": "Widget", "_confidence": 0.9},
            "price": {"_value": 99.99, "_confidence": 0.3},
            "sku": {"_value": "ABC123", "_confidence": 0.8},
        })
        ev.update(gt, pred)

        outfile = tmp_path / "metrics.json"
        ev.save_metrics(str(outfile))

        payload = _json.loads(outfile.read_text())
        assert "confidence_metrics" in payload
        assert "accumulator_metrics" in payload
        assert payload["accumulator_metrics"]["confidence_metrics"] is not None
