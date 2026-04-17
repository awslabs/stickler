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
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 99.99, "confidence": 0.6},
            "sku": {"value": "ABC123", "confidence": 0.8},
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
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 29.99, "confidence": 0.8},
            "sku": {"value": "XYZ999", "confidence": 0.5},
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
            "name": {"value": "Widget", "confidence": 0.9},
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
            "email": {"value": "a@b.com", "confidence": 0.95},
            "address": {
                "street": {"value": "123 Main St", "confidence": 0.85},
                "city": {"value": "Chicago", "confidence": 0.40},
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
            "name": {"value": "Jane", "confidence": 0.96},
            "contact": {
                "email": {"value": "jane@test.com", "confidence": 0.90},
                "address": {
                    "street": {"value": "456 Oak Avenue", "confidence": 0.80},
                    "city": {"value": "New York", "confidence": 0.35},
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
            "order_id": {"value": "ORD-1", "confidence": 0.99},
            "items": [
                {
                    "name": {"value": "Keyboard", "confidence": 0.92},
                    "price": {"value": 79.99, "confidence": 0.88},
                    "sku": {"value": "KEY001", "confidence": 0.95},
                },
                {
                    "name": {"value": "Mouse", "confidence": 0.90},
                    "price": {"value": 29.99, "confidence": 0.85},
                    "sku": {"value": "WRONG", "confidence": 0.30},
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


# ── 4. Partial confidence coverage ──


class TestPartialCoverage:
    def test_fields_without_confidence_excluded(self):
        """Fields without confidence are not in keyed pairs."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": 29.99,  # no confidence
            "sku": {"value": "ABC123", "confidence": 0.8},
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


# ── 8. Single-doc compare_with integration ──


class TestSingleDocIntegration:
    def test_confidence_metrics_in_result(self):
        """compare_with with add_confidence_metrics populates confidence_metrics."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 99.99, "confidence": 0.3},
            "sku": {"value": "ABC123", "confidence": 0.8},
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
        pred = Product.from_json({"name": {"value": "Widget", "confidence": 0.9}})
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
            "name": {"value": "Jane", "confidence": 0.95},
            "contact": {
                "email": {"value": "j@t.com", "confidence": 0.90},
                "address": {
                    "street": {"value": "123 Main", "confidence": 0.85},
                    "city": {"value": "Wrong City", "confidence": 0.30},
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
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 29.99, "confidence": 0.8},
            "sku": {"value": "ABC", "confidence": 0.7},
        })
        gt2 = Product(name="Gadget", price=49.99, sku="DEF")
        pred2 = Product.from_json({
            "name": {"value": "Gadget", "confidence": 0.85},
            "price": {"value": 99.99, "confidence": 0.4},
            "sku": {"value": "DEF", "confidence": 0.95},
        })

        evaluator.update(gt1, pred1)
        evaluator.update(gt2, pred2)

        for field in ["name", "price", "sku"]:
            assert len(evaluator._keyed_confidence_pairs[field]) == 2
            assert all(isinstance(p, ConfidencePair) for p in evaluator._keyed_confidence_pairs[field])

    def test_bulk_metrics_match_manual_computation(self):
        gt1 = Product(name="Widget", price=29.99, sku="ABC")
        pred1 = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 99.99, "confidence": 0.3},
            "sku": {"value": "ABC", "confidence": 0.8},
        })
        gt2 = Product(name="Gadget", price=49.99, sku="DEF")
        pred2 = Product.from_json({
            "name": {"value": "Wrong", "confidence": 0.2},
            "price": {"value": 49.99, "confidence": 0.85},
            "sku": {"value": "DEF", "confidence": 0.95},
        })

        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator.update(gt1, pred1)
        evaluator.update(gt2, pred2)
        bulk_result = evaluator.compute()

        calc = ConfidenceCalculator()
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

        assert bulk_result.confidence_metrics["overall"]["auroc"]["value"] == manual_result["overall"]["auroc"]["value"]

    def test_no_confidence_data_returns_none(self):
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred = Product(name="Widget", price=29.99, sku="ABC")
        evaluator.update(gt, pred)
        assert evaluator.compute().confidence_metrics is None


# ── 10. State serialization round-trip ──


class TestStateSerialization:
    def test_keyed_pairs_survive_round_trip(self):
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 99.99, "confidence": 0.3},
            "sku": {"value": "ABC", "confidence": 0.8},
        })
        evaluator.update(gt, pred)
        state = evaluator.get_state()

        evaluator2 = BulkStructuredModelEvaluator(target_schema=Product)
        evaluator2.load_state(state)

        assert evaluator2._keyed_confidence_pairs == evaluator._keyed_confidence_pairs

    def test_compute_after_load_matches_original(self):
        evaluator = BulkStructuredModelEvaluator(target_schema=Product)
        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 99.99, "confidence": 0.3},
            "sku": {"value": "ABC", "confidence": 0.8},
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
            Product.from_json({"name": {"value": "Widget", "confidence": 0.9}, "price": {"value": 29.99, "confidence": 0.8}, "sku": {"value": "ABC", "confidence": 0.7}}),
            Product.from_json({"name": {"value": "Wrong", "confidence": 0.2}, "price": {"value": 49.99, "confidence": 0.85}, "sku": {"value": "DEF", "confidence": 0.95}}),
            Product.from_json({"name": {"value": "Doohickey", "confidence": 0.88}, "price": {"value": 9.99, "confidence": 0.92}, "sku": {"value": "WRONG", "confidence": 0.15}}),
            Product.from_json({"name": {"value": "Thingamajig", "confidence": 0.91}, "price": {"value": 999.99, "confidence": 0.25}, "sku": {"value": "JKL", "confidence": 0.93}}),
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
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 29.99, "confidence": 0.8},
            "sku": {"value": "ABC", "confidence": 0.7},
        })

        wa.update(gt, pred)
        wb.update(gt, pred)
        wa.merge_state(wb.get_state())

        for field in ["name", "price", "sku"]:
            assert len(wa._keyed_confidence_pairs[field]) == 2


# ── 12. Multiple metrics in bulk evaluator ──


class TestMultipleMetrics:
    def test_bulk_with_multiple_metrics(self):
        evaluator = BulkStructuredModelEvaluator(
            target_schema=Product,
            confidence_metrics=[AUROCMetric(), BrierScoreMetric(), ECEMetric(n_bins=5)],
        )
        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 99.99, "confidence": 0.3},
            "sku": {"value": "ABC", "confidence": 0.8},
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
    def test_single_doc_coverage(self):
        """Single-doc result includes coverage stats."""
        gt = Product(name="Widget", price=29.99, sku="ABC123")
        pred = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": 29.99,  # no confidence
            "sku": {"value": "ABC123", "confidence": 0.8},
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
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 29.99, "confidence": 0.8},
            "sku": {"value": "ABC123", "confidence": 0.7},
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
            "name": {"value": "Widget", "confidence": 0.9},
            "price": 29.99,
            "sku": {"value": "ABC", "confidence": 0.8},
        })
        # Doc 2: 3 of 3 fields have confidence
        pred2 = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 29.99, "confidence": 0.8},
            "sku": {"value": "ABC", "confidence": 0.7},
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
            "name": {"value": "Widget", "confidence": 0.9},
            "price": 29.99,
            "sku": {"value": "ABC", "confidence": 0.8},
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
            "name": {"value": "Widget", "confidence": 0.9},
            "price": 29.99,
            "sku": {"value": "ABC", "confidence": 0.8},
        })
        pred_full = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 29.99, "confidence": 0.8},
            "sku": {"value": "ABC", "confidence": 0.7},
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
        from stickler.structured_object_evaluator.models.confidence import ErrorCaptureAtBudgetMetric

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
        b30 = result["budgets"][0.30]
        assert b30["fields_reviewed"] == 3
        assert b30["errors_found"] == 3
        assert b30["pct_errors_caught"] == 1.0
        assert b30["gain"] > 1.0

    def test_random_confidence_gain_near_one(self):
        """Random confidence should produce gain near 1.0."""
        from stickler.structured_object_evaluator.models.confidence import ErrorCaptureAtBudgetMetric
        import random as rng
        rng.seed(99)

        pairs = [
            cp(rng.random() < 0.3, rng.random())
            for _ in range(500)
        ]

        metric = ErrorCaptureAtBudgetMetric(budgets=[0.30])
        result = metric.compute(pairs)

        # Gain should be close to 1.0 for random confidence
        assert 0.7 < result["budgets"][0.30]["gain"] < 1.5

    def test_empty_pairs(self):
        from stickler.structured_object_evaluator.models.confidence import ErrorCaptureAtBudgetMetric
        result = ErrorCaptureAtBudgetMetric().compute([])
        assert result["value"] is None
        assert result["budgets"] == {}

    def test_no_errors(self):
        """All correct: no errors to find."""
        from stickler.structured_object_evaluator.models.confidence import ErrorCaptureAtBudgetMetric
        pairs = [cp(True, 0.5), cp(True, 0.6), cp(True, 0.7)]
        result = ErrorCaptureAtBudgetMetric().compute(pairs)
        assert result["value"] is None

    def test_custom_budgets(self):
        from stickler.structured_object_evaluator.models.confidence import ErrorCaptureAtBudgetMetric
        pairs = [
            cp(False, 0.1), cp(False, 0.2),
            cp(True, 0.8), cp(True, 0.9),
        ]
        metric = ErrorCaptureAtBudgetMetric(budgets=[0.25, 0.50, 0.75])
        result = metric.compute(pairs)
        assert set(result["budgets"].keys()) == {0.25, 0.50, 0.75}

    def test_headline_is_middle_budget(self):
        """Headline value should be the gain at the middle budget level."""
        from stickler.structured_object_evaluator.models.confidence import ErrorCaptureAtBudgetMetric
        pairs = [
            cp(False, 0.1), cp(False, 0.2), cp(False, 0.3),
            cp(True, 0.7), cp(True, 0.8), cp(True, 0.9),
            cp(True, 0.91), cp(True, 0.92), cp(True, 0.93), cp(True, 0.94),
        ]
        metric = ErrorCaptureAtBudgetMetric(budgets=[0.10, 0.30, 0.50])
        result = metric.compute(pairs)
        # Middle budget is 0.30
        assert result["value"] == result["budgets"][0.30]["gain"]

    def test_bulk_evaluator_integration(self):
        """ErrorCaptureAtBudgetMetric works through the bulk evaluator."""
        from stickler.structured_object_evaluator.models.confidence import ErrorCaptureAtBudgetMetric

        evaluator = BulkStructuredModelEvaluator(
            target_schema=Product,
            confidence_metrics=[ErrorCaptureAtBudgetMetric(budgets=[0.30, 0.50])],
        )

        gt = Product(name="Widget", price=29.99, sku="ABC")
        pred_good = Product.from_json({
            "name": {"value": "Widget", "confidence": 0.9},
            "price": {"value": 29.99, "confidence": 0.8},
            "sku": {"value": "ABC", "confidence": 0.7},
        })
        pred_bad = Product.from_json({
            "name": {"value": "Wrong", "confidence": 0.2},
            "price": {"value": 99.99, "confidence": 0.15},
            "sku": {"value": "XYZ", "confidence": 0.1},
        })

        evaluator.update(gt, pred_good)
        evaluator.update(gt, pred_bad)
        result = evaluator.compute()

        ecab = result.confidence_metrics["overall"]["error_capture_at_budget"]
        assert "budgets" in ecab
        assert 0.30 in ecab["budgets"]
        assert 0.50 in ecab["budgets"]
