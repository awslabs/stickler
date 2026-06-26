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
# BBoxIoUComparator — page-aware comparison
# ══════════════════════════════════════════════════════════════════════


class TestBBoxIoUComparatorPageAware:
    """page_aware=True forces a miss when boxes disagree about their page."""

    def setup_method(self):
        self.aware = BBoxIoUComparator(threshold=0.5, page_aware=True)
        self.plain = BBoxIoUComparator(threshold=0.5)

    # --- page numbers are accepted in both formats (regardless of flag) -----

    def test_two_point_with_page_parses(self):
        # Same coords, same page -> normal IoU even when page-aware.
        assert self.aware.compare([[0, 0], [10, 10], 1], [[0, 0], [10, 10], 1]) == 1.0

    def test_flat_with_page_parses(self):
        assert self.aware.compare([0, 0, 10, 10, 3], [0, 0, 10, 10, 3]) == 1.0

    def test_mixed_formats_same_page(self):
        assert self.aware.compare([[0, 0], [10, 10], 2], [0, 0, 10, 10, 2]) == 1.0

    def test_integer_valued_float_page_coerced(self):
        # 2.0 is treated as page 2, so it matches an int page 2.
        assert self.aware.compare([[0, 0], [10, 10], 2.0], [[0, 0], [10, 10], 2]) == 1.0

    def test_partial_overlap_same_page_matches_plain_iou(self):
        # When pages agree, the page-aware comparator must return the EXACT
        # same partial IoU as the plain comparator on the same coordinates --
        # i.e. the page suffix didn't disturb the geometry math.
        aware_iou = self.aware.compare([[0, 0], [10, 10], 1], [[5, 5], [15, 15], 1])
        plain_iou = self.plain.compare([[0, 0], [10, 10]], [[5, 5], [15, 15]])
        assert aware_iou == plain_iou
        assert abs(aware_iou - 25 / 175) < 1e-6  # genuinely partial, not 0 or 1

    def test_partial_overlap_flat_same_page_matches_plain_iou(self):
        # Same invariant via the flat (5-element) format.
        aware_iou = self.aware.compare([0, 0, 20, 20, 3], [5, 5, 10, 10, 3])
        plain_iou = self.plain.compare([0, 0, 20, 20], [5, 5, 10, 10])
        assert aware_iou == plain_iou
        assert abs(aware_iou - 25 / 400) < 1e-6

    # --- page disagreement forces 0.0 when page-aware ----------------------

    def test_different_pages_force_miss(self):
        # Identical coordinates, different pages -> automatic miss.
        assert self.aware.compare([[0, 0], [10, 10], 1], [[0, 0], [10, 10], 2]) == 0.0

    def test_gt_has_page_pred_missing_forces_miss(self):
        assert self.aware.compare([[0, 0], [10, 10]], [[0, 0], [10, 10], 1]) == 0.0

    def test_pred_has_page_gt_missing_forces_miss(self):
        assert self.aware.compare([[0, 0], [10, 10], 1], [[0, 0], [10, 10]]) == 0.0

    def test_both_missing_page_forces_miss(self):
        # In page-aware mode a box MUST declare its page, so a page-less box is
        # wrong even against another page-less box.
        assert self.aware.compare([[0, 0], [10, 10]], [[0, 0], [10, 10]]) == 0.0

    def test_flat_without_page_forces_miss(self):
        # A four-element (page-less) box is wrong 100% of the time when aware.
        assert self.aware.compare([0, 0, 10, 10], [0, 0, 10, 10, 1]) == 0.0

    # --- default (page_aware=False) ignores pages entirely -----------------

    def test_default_ignores_different_pages(self):
        # Without the flag, the page suffix is parsed but does not affect IoU.
        assert self.plain.compare([[0, 0], [10, 10], 1], [[0, 0], [10, 10], 2]) == 1.0

    def test_default_ignores_present_absent_page(self):
        assert self.plain.compare([[0, 0], [10, 10], 1], [[0, 0], [10, 10]]) == 1.0

    # --- malformed pages are treated as malformed boxes (score 0.0) --------

    def test_non_integer_float_page_is_malformed(self):
        assert self.aware.compare([[0, 0], [10, 10], 1.5], [[0, 0], [10, 10], 1]) == 0.0

    def test_non_numeric_page_is_malformed(self):
        assert self.aware.compare([[0, 0], [10, 10], "1"], [[0, 0], [10, 10], 1]) == 0.0

    def test_nan_page_is_malformed(self):
        nan = float("nan")
        assert self.aware.compare([[0, 0], [10, 10], nan], [[0, 0], [10, 10], 1]) == 0.0

    def test_three_scalars_still_invalid(self):
        # [1, 2, 3] is NOT a two-point+page box (first two aren't points);
        # it stays malformed under both flag settings.
        assert self.aware.compare([1, 2, 3], [[0, 0], [10, 10], 1]) == 0.0
        assert self.plain.compare([1, 2, 3], [[0, 0], [10, 10]]) == 0.0


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
        # detections are (confidence, iou); both clear the 0.5 threshold.
        assert calc._average_precision([(0.9, 0.9), (0.8, 0.8)], 2, 0.5) == 1.0

    def test_undefined_when_no_gt(self):
        calc = MAPCalculator(0.5)
        assert calc._average_precision([(0.9, 0.9)], 0, 0.5) is None

    def test_no_detections_is_zero(self):
        calc = MAPCalculator(0.5)
        assert calc._average_precision([], 2, 0.5) == 0.0

    def test_confidence_ranking_matters(self):
        """Same TP/FP counts, different ordering by confidence -> different AP."""
        calc = MAPCalculator(0.5)
        # TP (iou 0.9) ranked above FP (iou 0.0) vs. the reverse.
        tp_first = calc._average_precision([(0.9, 0.9), (0.6, 0.0)], 2, 0.5)
        fp_first = calc._average_precision([(0.9, 0.0), (0.6, 0.9)], 2, 0.5)
        # COCO 101-point values: 51/101 vs 25.5/101.
        assert tp_first == pytest.approx(51 / 101)
        assert fp_first == pytest.approx(25.5 / 101)
        assert tp_first > fp_first

    def test_multi_detection_envelope_interpolation(self):
        """Multi-detection curve where the precision envelope changes the result.

        Detections ranked by confidence: FP, TP, TP with num_gt=2.

            rank  label  tp fp  recall  precision
              1   FP     0  1    0.0     0.0
              2   TP     1  1    0.5     0.5
              3   TP     2  1    1.0     0.667

        COCO removes zig-zags (precision envelope) before sampling, lifting the
        precision at recall 0.5 to the later max of 0.667, so the 101-point mean
        is 2/3. Without the envelope (sklearn-style) the same curve gives
        7/12 ≈ 0.583, so this asserts the interpolation is actually applied.
        """
        calc = MAPCalculator(0.5)
        ap = calc._average_precision([(0.9, 0.0), (0.8, 0.9), (0.7, 0.8)], 2, 0.5)
        assert ap == pytest.approx(2 / 3)
        assert ap > 0.5833

    def test_larger_pr_curve(self):
        """A richer alternating curve, hand-derived against COCO 101-point.

        Detections ranked: TP, FP, TP, FP, TP with num_gt=3. After the envelope,
        precision is 1.0 up to recall 1/3, 2/3 up to recall 2/3, and 0.6 up to
        recall 1.0. Sampling at 101 recall points (34 / 33 / 34 split) gives:

            AP = (34*1.0 + 33*(2/3) + 34*0.6) / 101 ≈ 0.7564
        """
        calc = MAPCalculator(0.5)
        ap = calc._average_precision(
            [(0.9, 0.9), (0.8, 0.0), (0.7, 0.9), (0.6, 0.0), (0.5, 0.9)], 3, 0.5
        )
        assert ap == pytest.approx((34 + 33 * (2 / 3) + 34 * 0.6) / 101)


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

    def test_object_level_fn_row_coverage_stays_consistent(self):
        """An object-level FN row hiding several nested GT boxes must count each
        missed box toward both fields_with_bbox and fields_total, so the coverage
        ratio cannot exceed 1.0."""
        calc = MAPCalculator(0.5)
        # item[0] matched; item[1] missing entirely (object-level FN row) while
        # carrying two nested GT boxes.
        fc = [
            {
                "expected_key": "items[0].description",
                "actual_key": "items[0].description",
                "match": True,
            },
            {
                "expected_key": "items[0].amount",
                "actual_key": "items[0].amount",
                "match": True,
            },
            {"expected_key": "items[1]", "actual_key": None, "match": False},
        ]
        gt = {
            "items[0].description": [[0, 0], [10, 10]],
            "items[0].amount": [[0, 10], [10, 20]],
            "items[1].description": [[0, 20], [10, 30]],
            "items[1].amount": [[0, 30], [10, 40]],
        }
        pred = {
            "items[0].description": [[0, 0], [10, 10]],
            "items[0].amount": [[0, 10], [10, 20]],
        }
        ex = calc.extract_from_dicts(fc, gt, pred, {})
        metrics = calc.compute_metrics(
            ex.keyed_pairs, ex.fields_with_bbox, ex.fields_total
        )
        cov = metrics["coverage"]
        # 4 GT boxes total, all carried a bbox; every one of them counts toward
        # both numerator and denominator.
        assert cov["fields_with_bbox"] == 4
        assert cov["fields_total"] == 4
        assert cov["ratio"] == 1.0
        # The two missed boxes still deflate recall on their field-types.
        assert metrics["fields"]["items[].description"]["num_gt"] == 2
        assert metrics["fields"]["items[].description"]["num_detections"] == 1
        assert metrics["fields"]["items[].amount"]["num_gt"] == 2
        assert metrics["fields"]["items[].amount"]["num_detections"] == 1

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

    def test_multi_threshold_averaging_and_map_50_75(self):
        """Default COCO IoU range: mean_ap averages over thresholds; map_50/75.

        A single detection with IoU exactly 0.7 is a TP at thresholds
        0.50-0.70 (5 of the 10 COCO thresholds) and an FP at 0.75-0.95. With
        one GT box, AP is 1.0 where it's a TP and 0.0 where it's an FP, so:

            mean_ap = 5/10 = 0.5,  map_50 = 1.0,  map_75 = 0.0
        """
        calc = MAPCalculator()  # COCO default (0.50 ... 0.95)
        assert calc.iou_thresholds == tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
        obs = [BBoxObservation(has_gt=True, has_pred=True, iou=0.7, confidence=0.9)]
        m = calc.compute_metrics({"field": obs}, fields_with_bbox=1, fields_total=1)
        assert m["map_50"] == 1.0
        assert m["map_75"] == 0.0
        assert m["mean_ap"] == pytest.approx(0.5)
        assert m["fields"]["field"]["ap_50"] == 1.0
        assert m["fields"]["field"]["ap_75"] == 0.0
        assert m["fields"]["field"]["ap"] == pytest.approx(0.5)

    def test_single_threshold_has_no_map_75(self):
        calc = MAPCalculator(0.5)
        obs = [BBoxObservation(has_gt=True, has_pred=True, iou=0.9, confidence=0.9)]
        m = calc.compute_metrics({"field": obs}, 1, 1)
        assert m["iou_thresholds"] == [0.5]
        assert m["map_50"] == 1.0
        assert m["map_75"] is None
        assert m["mean_ap"] == 1.0


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
        # 1 TP out of 2 GT, single detection. Under COCO 101-point sampling this
        # is 51/101 at every IoU threshold (the detection has IoU 1.0), so the
        # averaged mean_ap is 51/101 ≈ 0.505 rather than an exact 0.5.
        assert result["bbox_metrics"]["mean_ap"] == pytest.approx(51 / 101)


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
