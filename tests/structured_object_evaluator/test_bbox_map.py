"""
Tests for bounding box mAP (Mean Average Precision) scoring.

Covers:
- BBoxIoUComparator: IoU math, format handling, NaN/inf and edge cases.
- class_key: list-index normalization for per-field-type grouping.
- MAPCalculator: GT/pred join by expected/actual key, real confidence-ranked
  Average Precision, coverage.
- Reordered list fields and FN rows (the join-correctness regression).
- BBoxMAPAccumulator: bulk accumulation, compute, state round-trip/merge,
  coexistence with ConfidenceAccumulator, JSONL round-trip.
- compare_with(add_bbox_metrics=True): single-document integration, the
  pre-extracted bbox maps, evaluator_format warning, empty-list coverage.

Bounding boxes ride on the underscore rich-value keys (_value / _bbox /
_confidence).
"""

import warnings
from typing import List, Optional

import pytest

from stickler.comparators import LevenshteinComparator, NumericComparator
from stickler.comparators.bbox import BBoxIoUComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.bbox import (
    BBoxMAPAccumulator,
    BBoxObservation,
    MAPCalculator,
    class_key,
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

# ── Test models ──


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


class LineItem(StructuredModel):
    description: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8
    )


class ItemizedInvoice(StructuredModel):
    items: List[LineItem] = ComparableField(weight=1.0)


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
        iou = self.cmp.compare([[0, 0], [10, 10]], [[5, 5], [15, 15]])
        assert abs(iou - 25 / 175) < 1e-6

    def test_one_box_inside_another(self):
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

    def test_nan_coordinates_score_zero(self):
        nan = float("nan")
        iou = self.cmp.compare([[0, 0], [nan, 10]], [[0, 0], [10, 10]])
        assert iou == 0.0

    def test_inf_coordinates_score_zero(self):
        inf = float("inf")
        iou = self.cmp.compare([0, 0, inf, 10], [0, 0, 10, 10])
        assert iou == 0.0


# ══════════════════════════════════════════════════════════════════════
# class_key
# ══════════════════════════════════════════════════════════════════════


class TestClassKey:
    def test_flat_path_unchanged(self):
        assert class_key("vendor_name") == "vendor_name"

    def test_single_index_normalized(self):
        assert class_key("items[2].description") == "items[].description"

    def test_multiple_indices_normalized(self):
        assert class_key("a[0].b[3].c") == "a[].b[].c"


# ══════════════════════════════════════════════════════════════════════
# MAPCalculator — real Average Precision
# ══════════════════════════════════════════════════════════════════════


def _fc(*pairs):
    """field_comparisons rows from (expected_key, actual_key) pairs."""
    return [
        {"expected_key": e, "actual_key": a, "match": True, "score": 1.0}
        for e, a in pairs
    ]


class TestAveragePrecision:
    def test_all_correct_ap_is_one(self):
        calc = MAPCalculator(0.5)
        assert calc._average_precision([(0.9, True), (0.8, True)], 2) == 1.0

    def test_undefined_when_no_gt(self):
        calc = MAPCalculator(0.5)
        assert calc._average_precision([(0.9, True)], 0) is None

    def test_no_detections_is_zero(self):
        calc = MAPCalculator(0.5)
        assert calc._average_precision([], 2) == 0.0

    def test_confidence_ranking_matters(self):
        """Same TP/FP counts, different ordering by confidence -> different AP."""
        calc = MAPCalculator(0.5)
        tp_first = calc._average_precision([(0.9, True), (0.6, False)], 2)
        fp_first = calc._average_precision([(0.9, False), (0.6, True)], 2)
        assert tp_first == 0.5
        assert fp_first == 0.25
        assert tp_first != fp_first


class TestMAPCalculator:
    def test_bboxes_from_extras(self):
        extras = {
            "vendor_name": {"_bbox": [[0, 0], [10, 10]]},
            "invoice_number": {"_confidence": 0.9},
        }
        assert MAPCalculator.bboxes_from_extras(extras) == {
            "vendor_name": [[0, 0], [10, 10]]
        }

    def test_perfect_match(self):
        calc = MAPCalculator(0.5)
        fc = _fc(("vendor_name", "vendor_name"), ("invoice_number", "invoice_number"))
        gt = {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[0, 25], [100, 45]],
        }
        ex = calc.extract_from_dicts(fc, gt, dict(gt), {})
        metrics = calc.compute_metrics(
            ex.keyed_pairs, ex.fields_with_bbox, ex.fields_total
        )
        assert metrics["mean_ap"] == 1.0
        assert metrics["coverage"]["fields_with_bbox"] == 2
        assert metrics["coverage"]["fields_total"] == 2

    def test_one_hit_one_miss(self):
        calc = MAPCalculator(0.5)
        fc = _fc(("vendor_name", "vendor_name"), ("invoice_number", "invoice_number"))
        gt = {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[0, 25], [100, 45]],
        }
        pred = {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[200, 200], [300, 220]],
        }
        ex = calc.extract_from_dicts(fc, gt, pred, {})
        metrics = calc.compute_metrics(
            ex.keyed_pairs, ex.fields_with_bbox, ex.fields_total
        )
        # vendor AP 1.0, invoice AP 0.0 -> mean 0.5
        assert metrics["mean_ap"] == 0.5
        assert metrics["fields"]["vendor_name"]["ap"] == 1.0
        assert metrics["fields"]["invoice_number"]["ap"] == 0.0

    def test_fn_row_records_miss(self):
        """A GT box the prediction missed (actual_key None) is a recall miss."""
        calc = MAPCalculator(0.5)
        fc = [
            {"expected_key": "vendor_name", "actual_key": "vendor_name", "match": True},
            {"expected_key": "invoice_number", "actual_key": None, "match": False},
        ]
        gt = {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[0, 25], [100, 45]],
        }
        pred = {"vendor_name": [[0, 0], [100, 20]]}
        ex = calc.extract_from_dicts(fc, gt, pred, {})
        metrics = calc.compute_metrics(
            ex.keyed_pairs, ex.fields_with_bbox, ex.fields_total
        )
        assert ex.fields_with_bbox == 2
        assert metrics["fields"]["invoice_number"]["num_gt"] == 1
        assert metrics["fields"]["invoice_number"]["num_detections"] == 0
        assert metrics["fields"]["invoice_number"]["ap"] == 0.0

    def test_no_gt_field_skipped_but_counted(self):
        calc = MAPCalculator(0.5)
        fc = _fc(("vendor_name", "vendor_name"), ("invoice_number", "invoice_number"))
        gt = {"vendor_name": [[0, 0], [100, 20]]}  # no GT for invoice_number
        pred = {
            "vendor_name": [[0, 0], [100, 20]],
            "invoice_number": [[0, 25], [100, 45]],
        }
        ex = calc.extract_from_dicts(fc, gt, pred, {})
        metrics = calc.compute_metrics(
            ex.keyed_pairs, ex.fields_with_bbox, ex.fields_total
        )
        # invoice_number has a spurious prediction (no GT) -> FP, num_gt 0 -> AP None
        assert metrics["fields"]["invoice_number"]["num_gt"] == 0
        assert metrics["fields"]["invoice_number"]["ap"] is None
        # mean_ap only averages fields with GT boxes
        assert metrics["mean_ap"] == 1.0

    def test_no_bbox_data_returns_none(self):
        calc = MAPCalculator(0.5)
        fc = _fc(("vendor_name", "vendor_name"))
        ex = calc.extract_from_dicts(fc, {}, {}, {})
        metrics = calc.compute_metrics(
            ex.keyed_pairs, ex.fields_with_bbox, ex.fields_total
        )
        assert metrics["mean_ap"] is None
        assert metrics["coverage"]["fields_with_bbox"] == 0
        assert metrics["coverage"]["fields_total"] == 1

    def test_threshold_changes_classification(self):
        # IoU = 5000/15000 ≈ 0.333
        fc = _fc(("vendor_name", "vendor_name"))
        gt = {"vendor_name": [[0, 0], [100, 100]]}
        pred = {"vendor_name": [[50, 0], [150, 100]]}

        strict = MAPCalculator(0.5)
        ex = strict.extract_from_dicts(fc, gt, pred, {})
        assert strict.compute_metrics(ex.keyed_pairs, 1, 1)["mean_ap"] == 0.0

        lenient = MAPCalculator(0.3)
        ex = lenient.extract_from_dicts(fc, gt, pred, {})
        assert lenient.compute_metrics(ex.keyed_pairs, 1, 1)["mean_ap"] == 1.0

    def test_extract_raises_without_field_comparisons(self):
        calc = MAPCalculator()
        gt = DocumentField.from_json(
            {"vendor_name": {"_value": "A", "_bbox": [[0, 0], [1, 1]]}}
        )
        with pytest.raises(ValueError, match="No field comparisons"):
            calc.extract({"field_comparisons": []}, gt, gt)


# ══════════════════════════════════════════════════════════════════════
# Reordered list fields — the join-correctness regression (B1)
# ══════════════════════════════════════════════════════════════════════


class TestListFieldJoin:
    def _itemized(self, order):
        """Build an ItemizedInvoice with items in the given description order."""
        bbox = {
            "Apple": [[0, 0], [10, 10]],
            "Banana": [[0, 20], [10, 30]],
            "Cherry": [[0, 40], [10, 50]],
        }
        return ItemizedInvoice.from_json(
            {
                "items": [
                    {"description": {"_value": d, "_bbox": bbox[d], "_confidence": 0.9}}
                    for d in order
                ]
            }
        )

    def test_reordered_items_score_correctly(self):
        """Reordered-but-matching list items must score mAP 1.0, not 0.0."""
        gt = self._itemized(["Apple", "Banana", "Cherry"])
        pred = self._itemized(["Cherry", "Banana", "Apple"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = gt.compare_with(
                pred, add_bbox_metrics=True, document_field_comparisons=True
            )
        bm = result["bbox_metrics"]
        # All three items localize perfectly under the index-normalized class.
        assert "items[].description" in bm["fields"]
        assert bm["fields"]["items[].description"]["num_gt"] == 3
        assert bm["mean_ap"] == 1.0

    def test_missing_item_records_recall_miss(self):
        gt = self._itemized(["Apple", "Banana"])
        pred = self._itemized(["Apple"])  # Banana missing entirely
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = gt.compare_with(
                pred, add_bbox_metrics=True, document_field_comparisons=True
            )
        field = result["bbox_metrics"]["fields"]["items[].description"]
        assert field["num_gt"] == 2
        assert field["num_detections"] == 1
        # 1 TP out of 2 GT, single detection -> AP 0.5
        assert result["bbox_metrics"]["mean_ap"] == 0.5


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
        acc = BBoxMAPAccumulator(0.5)
        acc.accumulate(result, result.get("prediction_raw"))
        metrics = acc.compute()
        assert metrics["mean_ap"] == 0.5
        assert metrics["coverage"]["fields_with_bbox"] == 2
        assert metrics["coverage"]["fields_total"] == 3

    def test_accumulate_across_documents(self):
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(pred, document_field_comparisons=True)
        acc = BBoxMAPAccumulator(0.5)
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
        acc = BBoxMAPAccumulator(0.5)
        acc.accumulate(result, result.get("prediction_raw"))

        restored = BBoxMAPAccumulator(0.5)
        restored.load_state(acc.get_state())
        # Assert against the known value, not just equality of two computeds.
        assert restored.compute()["mean_ap"] == 0.5
        assert restored.compute()["coverage"]["fields_total"] == 3

    def test_merge_state(self):
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(pred, document_field_comparisons=True)
        acc = BBoxMAPAccumulator(0.5)
        acc.accumulate(result, result.get("prediction_raw"))
        state = acc.get_state()

        merged = BBoxMAPAccumulator(0.5)
        merged.merge_state(state)
        merged.merge_state(state)
        metrics = merged.compute()
        assert metrics["coverage"]["fields_total"] == 6
        assert metrics["mean_ap"] == 0.5

    def test_get_state_observations_serializable(self):
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(pred, document_field_comparisons=True)
        acc = BBoxMAPAccumulator()
        acc.accumulate(result, result.get("prediction_raw"))
        state = acc.get_state()
        for obs in state["keyed_bbox_pairs"].values():
            for o in obs:
                assert set(o.keys()) == {"has_gt", "has_pred", "iou", "confidence"}
                assert isinstance(BBoxObservation(**o), BBoxObservation)


# ══════════════════════════════════════════════════════════════════════
# Bulk evaluator integration
# ══════════════════════════════════════════════════════════════════════


class TestBulkIntegration:
    def test_bulk_bbox_map_metrics(self):
        gt, pred = _gt_pred_pair()
        evaluator = BulkStructuredModelEvaluator(accumulators=[BBoxMAPAccumulator(0.5)])
        for _ in range(3):
            evaluator.update(gt, pred)
        proc = evaluator.compute()
        bm = proc.accumulator_metrics["bbox_map_metrics"]
        assert bm["mean_ap"] == 0.5
        assert bm["coverage"]["fields_total"] == 9

    def test_bbox_and_confidence_accumulators_coexist(self):
        gt, pred = _gt_pred_pair()
        evaluator = BulkStructuredModelEvaluator(
            accumulators=[ConfidenceAccumulator(), BBoxMAPAccumulator(0.5)]
        )
        for _ in range(2):
            evaluator.update(gt, pred)
        metrics = evaluator.compute().accumulator_metrics
        assert "confidence_metrics" in metrics
        assert "bbox_map_metrics" in metrics
        assert metrics["bbox_map_metrics"]["mean_ap"] == 0.5

    def test_jsonl_round_trip_parity(self, tmp_path):
        """update_from_comparison_result reproduces the update() metrics."""
        gt, pred = _gt_pred_pair()
        result = gt.compare_with(
            pred,
            include_confusion_matrix=True,
            document_field_comparisons=True,
        )

        from_update = BulkStructuredModelEvaluator(
            accumulators=[BBoxMAPAccumulator(0.5)]
        )
        from_update.update(gt, pred)
        expected = from_update.compute().accumulator_metrics["bbox_map_metrics"]

        from_jsonl = BulkStructuredModelEvaluator(
            accumulators=[BBoxMAPAccumulator(0.5)]
        )
        from_jsonl.update_from_comparison_result(result, doc_id="d1")
        got = from_jsonl.compute().accumulator_metrics["bbox_map_metrics"]

        assert got["mean_ap"] == expected["mean_ap"]
        assert got["coverage"] == expected["coverage"]


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
        assert result["bbox_metrics"]["mean_ap"] == 0.5

    def test_add_bbox_metrics_auto_enables_field_comparisons(self):
        gt = DocumentField.from_json(
            {"vendor_name": {"_value": "Acme", "_bbox": [[0, 0], [100, 20]]}}
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {
                    "_value": "Acme",
                    "_bbox": [[0, 0], [100, 20]],
                    "_confidence": 0.9,
                }
            }
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

    def test_evaluator_format_warns_and_drops_metrics(self):
        gt, pred = _gt_pred_pair()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = gt.compare_with(
                pred,
                add_bbox_metrics=True,
                evaluator_format=True,
                document_field_comparisons=True,
            )
        assert any("evaluator_format=True" in str(w.message) for w in caught)
        # evaluator format rebuilds from a fixed key set; bbox_metrics is dropped.
        assert "bbox_metrics" not in result

    def test_empty_field_comparisons_is_coverage_only(self):
        """A model whose only list field is empty on both sides must not raise."""
        gt = ItemizedInvoice.from_json({"items": []})
        pred = ItemizedInvoice.from_json({"items": []})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = gt.compare_with(
                pred, add_bbox_metrics=True, document_field_comparisons=True
            )
        assert result["bbox_metrics"]["mean_ap"] is None
        assert result["bbox_metrics"]["coverage"]["fields_with_bbox"] == 0

    def test_flat_bbox_format_single_doc(self):
        gt = DocumentField.from_json(
            {"vendor_name": {"_value": "Acme", "_bbox": [0, 0, 100, 20]}}
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {
                    "_value": "Acme",
                    "_bbox": [0, 0, 100, 20],
                    "_confidence": 0.9,
                }
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = gt.compare_with(
                pred, add_bbox_metrics=True, document_field_comparisons=True
            )
        assert result["bbox_metrics"]["mean_ap"] == 1.0
