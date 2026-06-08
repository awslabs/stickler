"""
Tests for bounding box mAP (Mean Average Precision) scoring.

Covers:
- BBoxIoUComparator: IoU math, format handling, edge cases.
- MAPCalculator: extraction from dicts, per-field metrics, mean AP, coverage.
- BBoxMAPAccumulator: bulk accumulation, compute, state round-trip/merge,
  coexistence with ConfidenceAccumulator.
- compare_with(add_bbox_metrics=True): single-document integration and the
  pre-extracted ground_truth_bboxes / prediction_bboxes result keys.

Bounding boxes ride on the underscore rich-value keys (_value / _bbox).
"""

import warnings
from typing import Optional

import pytest

from stickler.comparators import LevenshteinComparator, NumericComparator
from stickler.comparators.bbox import BBoxIoUComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.bbox_map_accumulator import (
    BBoxMAPAccumulator,
)
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.confidence.accumulator import (
    ConfidenceAccumulator,
)
from stickler.structured_object_evaluator.models.map_calculator import (
    BBoxPair,
    MAPCalculator,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)

# ── Test model ──


class DocumentField(StructuredModel):
    vendor_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8
    )
    invoice_number: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9
    )
    total_amount: Optional[float] = ComparableField(
        comparator=NumericComparator(), threshold=0.95
    )


# ══════════════════════════════════════════════════════════════════════
# BBoxIoUComparator
# ══════════════════════════════════════════════════════════════════════


class TestBBoxIoUComparator:
    def setup_method(self):
        self.cmp = BBoxIoUComparator(threshold=0.5)

    def test_identical_boxes_two_point(self):
        assert self.cmp.compare([[0, 0], [10, 10]], [[0, 0], [10, 10]]) == 1.0

    def test_identical_boxes_flat(self):
        assert self.cmp.compare([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_mixed_formats(self):
        assert self.cmp.compare([[0, 0], [10, 10]], [0, 0, 10, 10]) == 1.0

    def test_no_overlap(self):
        assert self.cmp.compare([[0, 0], [5, 5]], [[10, 10], [20, 20]]) == 0.0

    def test_touching_edges(self):
        assert self.cmp.compare([[0, 0], [5, 5]], [[5, 0], [10, 5]]) == 0.0

    def test_partial_overlap(self):
        # Box1 (0,0)-(10,10), Box2 (5,5)-(15,15): inter 25, union 175.
        iou = self.cmp.compare([[0, 0], [10, 10]], [[5, 5], [15, 15]])
        assert abs(iou - 25 / 175) < 1e-6

    def test_one_box_inside_another(self):
        # Box1 area 400, Box2 area 25 fully inside: inter 25, union 400.
        iou = self.cmp.compare([[0, 0], [20, 20]], [[5, 5], [10, 10]])
        assert abs(iou - 25 / 400) < 1e-6

    def test_reversed_coordinates(self):
        assert self.cmp.compare([[10, 10], [0, 0]], [[0, 0], [10, 10]]) == 1.0

    def test_both_none(self):
        assert self.cmp.compare(None, None) == 1.0

    def test_one_none(self):
        assert self.cmp.compare(None, [[0, 0], [10, 10]]) == 0.0
        assert self.cmp.compare([[0, 0], [10, 10]], None) == 0.0

    def test_invalid_format_returns_zero(self):
        assert self.cmp.compare("not a bbox", [[0, 0], [10, 10]]) == 0.0
        assert self.cmp.compare([[0, 0], [10, 10]], [1, 2, 3]) == 0.0
        assert self.cmp.compare(42, [[0, 0], [10, 10]]) == 0.0

    def test_empty_list_returns_zero(self):
        assert self.cmp.compare([], [[0, 0], [10, 10]]) == 0.0

    def test_zero_area_box(self):
        assert self.cmp.compare([[5, 5], [5, 5]], [[0, 0], [10, 10]]) == 0.0

    def test_default_config_is_none(self):
        assert BBoxIoUComparator().config is None

    def test_custom_config(self):
        assert BBoxIoUComparator(margin_percent=10.0).config == {"margin_percent": 10.0}


# ══════════════════════════════════════════════════════════════════════
# MAPCalculator (extract_from_dicts + compute_metrics)
# ══════════════════════════════════════════════════════════════════════


def _fc(*keys):
    """Build minimal field_comparisons rows for the given field paths."""
    return [{"actual_key": k, "match": True, "score": 1.0} for k in keys]


class TestMAPCalculator:
    def test_bboxes_from_extras(self):
        extras = {
            "vendor_name": {"_bbox": [[0, 0], [10, 10]]},
            "invoice_number": {"_confidence": 0.9},  # no bbox
        }
        bboxes = MAPCalculator.bboxes_from_extras(extras)
        assert bboxes == {"vendor_name": [[0, 0], [10, 10]]}

    def test_perfect_match(self):
        calc = MAPCalculator(iou_threshold=0.5)
        fc = _fc("vendor_name", "invoice_number")
        gt = {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[0, 25], [100, 45]],
        }
        pred = dict(gt)
        extraction = calc.extract_from_dicts(fc, gt, pred)
        metrics = calc.compute_metrics(
            extraction.keyed_pairs,
            fields_with_bbox=extraction.fields_with_bbox,
            fields_total=extraction.fields_total,
        )
        assert metrics["mean_ap"] == 1.0
        assert metrics["coverage"]["fields_with_bbox"] == 2
        assert metrics["coverage"]["fields_total"] == 2
        for f in metrics["fields"].values():
            assert f["iou"] == 1.0
            assert f["ap"] == 1.0

    def test_no_overlap(self):
        calc = MAPCalculator(iou_threshold=0.5)
        fc = _fc("vendor_name")
        gt = {"vendor_name": [[0, 0], [50, 20]]}
        pred = {"vendor_name": [[200, 200], [300, 220]]}
        extraction = calc.extract_from_dicts(fc, gt, pred)
        metrics = calc.compute_metrics(extraction.keyed_pairs, 1, 1)
        assert metrics["mean_ap"] == 0.0
        assert metrics["fields"]["vendor_name"]["iou"] == 0.0

    def test_partial_one_hit_one_miss(self):
        calc = MAPCalculator(iou_threshold=0.5)
        fc = _fc("vendor_name", "invoice_number")
        gt = {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[0, 25], [100, 45]],
        }
        pred = {
            "vendor_name": [[0, 0], [100, 20]],  # hit
            "invoice_number": [[200, 200], [300, 220]],  # miss
        }
        extraction = calc.extract_from_dicts(fc, gt, pred)
        metrics = calc.compute_metrics(extraction.keyed_pairs, 2, 2)
        assert metrics["mean_ap"] == 0.5

    def test_missing_pred_bbox_is_miss(self):
        calc = MAPCalculator(iou_threshold=0.5)
        fc = _fc("vendor_name", "invoice_number")
        gt = {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[0, 25], [100, 45]],
        }
        pred = {"vendor_name": [[0, 0], [100, 20]]}  # invoice_number absent
        extraction = calc.extract_from_dicts(fc, gt, pred)
        metrics = calc.compute_metrics(extraction.keyed_pairs, 2, 2)
        assert metrics["fields"]["vendor_name"]["ap"] == 1.0
        assert metrics["fields"]["invoice_number"]["ap"] == 0.0
        assert metrics["mean_ap"] == 0.5

    def test_no_gt_bbox_field_skipped_but_counted(self):
        calc = MAPCalculator(iou_threshold=0.5)
        fc = _fc("vendor_name", "invoice_number")
        gt = {"vendor_name": [[0, 0], [100, 20]]}  # no GT for invoice_number
        pred = {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[0, 25], [100, 45]],
        }
        extraction = calc.extract_from_dicts(fc, gt, pred)
        metrics = calc.compute_metrics(
            extraction.keyed_pairs,
            fields_with_bbox=extraction.fields_with_bbox,
            fields_total=extraction.fields_total,
        )
        assert list(metrics["fields"].keys()) == ["vendor_name"]
        assert metrics["coverage"]["fields_with_bbox"] == 1
        assert metrics["coverage"]["fields_total"] == 2
        assert metrics["mean_ap"] == 1.0

    def test_no_bbox_data_returns_none_mean_ap(self):
        calc = MAPCalculator(iou_threshold=0.5)
        fc = _fc("vendor_name", "invoice_number")
        extraction = calc.extract_from_dicts(fc, {}, {})
        metrics = calc.compute_metrics(
            extraction.keyed_pairs,
            fields_with_bbox=extraction.fields_with_bbox,
            fields_total=extraction.fields_total,
        )
        assert metrics["mean_ap"] is None
        assert metrics["coverage"]["fields_with_bbox"] == 0
        assert metrics["coverage"]["fields_total"] == 2

    def test_non_string_actual_key_skipped(self):
        calc = MAPCalculator(iou_threshold=0.5)
        fc = [
            {"actual_key": "vendor_name", "match": True},
            {"actual_key": None, "match": False},  # list FN placeholder
        ]
        gt = {"vendor_name": [[0, 0], [10, 10]]}
        pred = {"vendor_name": [[0, 0], [10, 10]]}
        extraction = calc.extract_from_dicts(fc, gt, pred)
        assert extraction.fields_total == 1

    def test_threshold_changes_classification(self):
        # IoU = 5000/15000 ≈ 0.333.
        fc = _fc("vendor_name")
        gt = {"vendor_name": [[0, 0], [100, 100]]}
        pred = {"vendor_name": [[50, 0], [150, 100]]}

        strict = MAPCalculator(iou_threshold=0.5)
        ex = strict.extract_from_dicts(fc, gt, pred)
        assert strict.compute_metrics(ex.keyed_pairs, 1, 1)["mean_ap"] == 0.0

        lenient = MAPCalculator(iou_threshold=0.3)
        ex = lenient.extract_from_dicts(fc, gt, pred)
        assert lenient.compute_metrics(ex.keyed_pairs, 1, 1)["mean_ap"] == 1.0

    def test_extract_raises_without_field_comparisons(self):
        calc = MAPCalculator()
        gt = DocumentField.from_json(
            {"vendor_name": {"_value": "A", "_bbox": [[0, 0], [1, 1]]}}
        )
        with pytest.raises(ValueError, match="No field comparisons"):
            calc.extract({"field_comparisons": []}, gt, gt)


# ══════════════════════════════════════════════════════════════════════
# BBoxMAPAccumulator
# ══════════════════════════════════════════════════════════════════════


def _gt_pred_pair():
    gt = DocumentField.from_json(
        {
            "vendor_name": {"_value": "Acme", "_bbox": [[0, 0], [100, 20]]},
            "invoice_number": {"_value": "INV-001", "_bbox": [[0, 25], [100, 45]]},
        }
    )
    pred = DocumentField.from_json(
        {
            "vendor_name": {
                "_value": "Acme",
                "_bbox": [[0, 0], [100, 20]],  # hit
                "_confidence": 0.9,
            },
            "invoice_number": {
                "_value": "INV-001",
                "_bbox": [[200, 200], [300, 220]],  # miss
                "_confidence": 0.8,
            },
        }
    )
    return gt, pred


class TestBBoxMAPAccumulator:
    def test_name(self):
        assert BBoxMAPAccumulator().name == "bbox_map_metrics"

    def test_empty_compute_is_none(self):
        assert BBoxMAPAccumulator().compute() is None

    def test_accumulate_and_compute(self):
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(pred, document_field_comparisons=True)
        acc = BBoxMAPAccumulator(iou_threshold=0.5)
        acc.accumulate(result, result.get("prediction_raw"))
        metrics = acc.compute()
        assert metrics["mean_ap"] == 0.5
        assert metrics["coverage"]["fields_with_bbox"] == 2
        # total_amount is a compared field with no bbox -> counted in total only.
        assert metrics["coverage"]["fields_total"] == 3

    def test_accumulate_across_documents(self):
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(pred, document_field_comparisons=True)
        acc = BBoxMAPAccumulator(iou_threshold=0.5)
        for _ in range(3):
            acc.accumulate(result, result.get("prediction_raw"))
        metrics = acc.compute()
        assert metrics["mean_ap"] == 0.5
        assert metrics["coverage"]["fields_total"] == 9

    def test_no_field_comparisons_with_prediction_raw_warns(self):
        acc = BBoxMAPAccumulator()
        with pytest.warns(UserWarning, match="document_field_comparisons"):
            acc.accumulate({"field_comparisons": []}, {"vendor_name": "x"})
        assert acc.compute() is None

    def test_reset(self):
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(pred, document_field_comparisons=True)
        acc = BBoxMAPAccumulator()
        acc.accumulate(result, result.get("prediction_raw"))
        acc.reset()
        assert acc.compute() is None

    def test_state_round_trip(self):
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(pred, document_field_comparisons=True)
        acc = BBoxMAPAccumulator(iou_threshold=0.5)
        acc.accumulate(result, result.get("prediction_raw"))

        restored = BBoxMAPAccumulator(iou_threshold=0.5)
        restored.load_state(acc.get_state())
        assert restored.compute()["mean_ap"] == acc.compute()["mean_ap"]
        assert restored.compute()["coverage"] == acc.compute()["coverage"]

    def test_merge_state(self):
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(pred, document_field_comparisons=True)
        acc = BBoxMAPAccumulator(iou_threshold=0.5)
        acc.accumulate(result, result.get("prediction_raw"))
        state = acc.get_state()

        merged = BBoxMAPAccumulator(iou_threshold=0.5)
        merged.merge_state(state)
        merged.merge_state(state)
        metrics = merged.compute()
        assert metrics["coverage"]["fields_total"] == 6
        assert metrics["mean_ap"] == 0.5

    def test_get_state_pairs_are_serializable(self):
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(pred, document_field_comparisons=True)
        acc = BBoxMAPAccumulator()
        acc.accumulate(result, result.get("prediction_raw"))
        state = acc.get_state()
        # Pairs serialize to plain dicts and rehydrate into BBoxPair.
        for pairs in state["keyed_bbox_pairs"].values():
            for p in pairs:
                assert set(p.keys()) == {"iou", "has_pred"}
                assert isinstance(BBoxPair(**p), BBoxPair)


# ══════════════════════════════════════════════════════════════════════
# Bulk evaluator integration
# ══════════════════════════════════════════════════════════════════════


class TestBulkIntegration:
    def test_bulk_bbox_map_metrics(self):
        gt, pred = _gt_pred_pair()
        evaluator = BulkStructuredModelEvaluator(
            accumulators=[BBoxMAPAccumulator(iou_threshold=0.5)]
        )
        for _ in range(3):
            evaluator.update(gt, pred)
        proc = evaluator.compute()
        assert "bbox_map_metrics" in proc.accumulator_metrics
        bm = proc.accumulator_metrics["bbox_map_metrics"]
        assert bm["mean_ap"] == 0.5
        assert bm["coverage"]["fields_total"] == 9

    def test_bbox_and_confidence_accumulators_coexist(self):
        gt, pred = _gt_pred_pair()
        evaluator = BulkStructuredModelEvaluator(
            accumulators=[
                ConfidenceAccumulator(),
                BBoxMAPAccumulator(iou_threshold=0.5),
            ]
        )
        for _ in range(2):
            evaluator.update(gt, pred)
        proc = evaluator.compute()
        assert "confidence_metrics" in proc.accumulator_metrics
        assert "bbox_map_metrics" in proc.accumulator_metrics
        assert proc.accumulator_metrics["bbox_map_metrics"]["mean_ap"] == 0.5


# ══════════════════════════════════════════════════════════════════════
# compare_with(add_bbox_metrics=True) single-document path
# ══════════════════════════════════════════════════════════════════════


class TestCompareWithBBoxMetrics:
    def test_pre_extracted_bbox_maps_present(self):
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(pred, document_field_comparisons=True)
        assert result["ground_truth_bboxes"] == {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[0, 25], [100, 45]],
        }
        assert result["prediction_bboxes"] == {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[200, 200], [300, 220]],
        }

    def test_add_bbox_metrics_single_doc(self):
        gt, pred = _gt_pred_pair()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = gt.compare_with(
                pred, add_bbox_metrics=True, document_field_comparisons=True
            )
        assert "bbox_metrics" in result
        assert result["bbox_metrics"]["mean_ap"] == 0.5

    def test_add_bbox_metrics_auto_enables_field_comparisons(self):
        gt = DocumentField.from_json(
            {"vendor_name": {"_value": "Acme", "_bbox": [[0, 0], [100, 20]]}}
        )
        pred = DocumentField.from_json(
            {"vendor_name": {"_value": "Acme", "_bbox": [[0, 0], [100, 20]]}}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = gt.compare_with(pred, add_bbox_metrics=True)
        assert result["bbox_metrics"]["mean_ap"] == 1.0

    def test_add_bbox_metrics_emits_warning(self):
        gt, pred = _gt_pred_pair()
        with pytest.warns(UserWarning, match="Single-document mAP"):
            gt.compare_with(
                pred, add_bbox_metrics=True, document_field_comparisons=True
            )

    def test_custom_threshold_single_doc(self):
        gt = DocumentField.from_json(
            {"vendor_name": {"_value": "Acme", "_bbox": [[0, 0], [100, 100]]}}
        )
        pred = DocumentField.from_json(
            {"vendor_name": {"_value": "Acme", "_bbox": [[50, 0], [150, 100]]}}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            strict = gt.compare_with(
                pred,
                add_bbox_metrics=True,
                document_field_comparisons=True,
                bbox_iou_threshold=0.5,
            )
            lenient = gt.compare_with(
                pred,
                add_bbox_metrics=True,
                document_field_comparisons=True,
                bbox_iou_threshold=0.3,
            )
        assert strict["bbox_metrics"]["mean_ap"] == 0.0
        assert lenient["bbox_metrics"]["mean_ap"] == 1.0

    def test_bbox_and_confidence_coexist_single_doc(self):
        gt, pred = _gt_pred_pair()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = gt.compare_with(
                pred,
                add_bbox_metrics=True,
                add_confidence_metrics=True,
                document_field_comparisons=True,
            )
        assert "bbox_metrics" in result
        assert "confidence_metrics" in result

    def test_flat_bbox_format_single_doc(self):
        gt = DocumentField.from_json(
            {"vendor_name": {"_value": "Acme", "_bbox": [0, 0, 100, 20]}}
        )
        pred = DocumentField.from_json(
            {"vendor_name": {"_value": "Acme", "_bbox": [0, 0, 100, 20]}}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = gt.compare_with(
                pred, add_bbox_metrics=True, document_field_comparisons=True
            )
        assert result["bbox_metrics"]["mean_ap"] == 1.0
