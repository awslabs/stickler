"""
Tests for bounding box mAP (Mean Average Precision) scoring.

Tests cover:
- BBoxIoUComparator: IoU calculation, format handling, edge cases
- MAPCalculator: extraction, per-field metrics, mean AP
- End-to-end: compare_with(add_bbox_metrics=True) integration
- Rich value pattern: bbox extraction from JSON
"""

from typing import Optional

from stickler.comparators import LevenshteinComparator, NumericComparator
from stickler.comparators.bbox import BBoxIoUComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

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


# ══════════════════════════════════════════════════════════════════════
# BBoxIoUComparator tests
# ══════════════════════════════════════════════════════════════════════


class TestBBoxIoUComparator:
    """Tests for the BBoxIoUComparator."""

    def setup_method(self):
        self.cmp = BBoxIoUComparator(threshold=0.5)

    # ── Perfect overlap ──

    def test_identical_boxes_two_point(self):
        """Identical boxes in two-point format should return IoU 1.0."""
        assert self.cmp.compare([[0, 0], [10, 10]], [[0, 0], [10, 10]]) == 1.0

    def test_identical_boxes_flat(self):
        """Identical boxes in flat format should return IoU 1.0."""
        assert self.cmp.compare([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_mixed_formats(self):
        """Two-point vs flat format should still compute correctly."""
        iou = self.cmp.compare([[0, 0], [10, 10]], [0, 0, 10, 10])
        assert iou == 1.0

    # ── No overlap ──

    def test_no_overlap(self):
        """Non-overlapping boxes should return IoU 0.0."""
        assert self.cmp.compare([[0, 0], [5, 5]], [[10, 10], [20, 20]]) == 0.0

    def test_touching_edges(self):
        """Boxes that touch at an edge have zero-area intersection."""
        assert self.cmp.compare([[0, 0], [5, 5]], [[5, 0], [10, 5]]) == 0.0

    # ── Partial overlap ──

    def test_partial_overlap(self):
        """Partially overlapping boxes should return IoU between 0 and 1."""
        # Box1: (0,0)-(10,10), area=100
        # Box2: (5,5)-(15,15), area=100
        # Intersection: (5,5)-(10,10), area=25
        # Union: 100+100-25=175
        # IoU: 25/175 ≈ 0.1429
        iou = self.cmp.compare([[0, 0], [10, 10]], [[5, 5], [15, 15]])
        assert abs(iou - 25 / 175) < 1e-6

    def test_one_box_inside_another(self):
        """Small box fully inside large box."""
        # Box1: (0,0)-(20,20), area=400
        # Box2: (5,5)-(10,10), area=25
        # Intersection: 25
        # Union: 400+25-25=400
        # IoU: 25/400 = 0.0625
        iou = self.cmp.compare([[0, 0], [20, 20]], [[5, 5], [10, 10]])
        assert abs(iou - 25 / 400) < 1e-6

    # ── Coordinate normalization ──

    def test_reversed_coordinates(self):
        """Coordinates with x1>x2 or y1>y2 should be normalized."""
        iou = self.cmp.compare([[10, 10], [0, 0]], [[0, 0], [10, 10]])
        assert iou == 1.0

    # ── Null handling ──

    def test_both_none(self):
        assert self.cmp.compare(None, None) == 1.0

    def test_one_none(self):
        assert self.cmp.compare(None, [[0, 0], [10, 10]]) == 0.0
        assert self.cmp.compare([[0, 0], [10, 10]], None) == 0.0

    # ── Invalid input ──

    def test_invalid_format_returns_zero(self):
        assert self.cmp.compare("not a bbox", [[0, 0], [10, 10]]) == 0.0
        assert self.cmp.compare([[0, 0], [10, 10]], [1, 2, 3]) == 0.0
        assert self.cmp.compare(42, [[0, 0], [10, 10]]) == 0.0

    def test_empty_list_returns_zero(self):
        assert self.cmp.compare([], [[0, 0], [10, 10]]) == 0.0

    # ── Binary classification ──

    def test_binary_compare_above_threshold(self):
        """IoU above threshold should be classified as TP."""
        tp, fp = self.cmp.binary_compare([[0, 0], [10, 10]], [[0, 0], [10, 10]])
        assert tp == 1
        assert fp == 0

    def test_binary_compare_below_threshold(self):
        """IoU below threshold should be classified as FP."""
        tp, fp = self.cmp.binary_compare([[0, 0], [5, 5]], [[10, 10], [20, 20]])
        assert tp == 0
        assert fp == 1

    # ── Zero-area boxes ──

    def test_zero_area_box(self):
        """A point (zero-area box) should return IoU 0.0."""
        assert self.cmp.compare([[5, 5], [5, 5]], [[0, 0], [10, 10]]) == 0.0

    # ── Config serialization ──

    def test_default_config_is_none(self):
        """Default margin_percent should produce no config."""
        cmp = BBoxIoUComparator()
        assert cmp.config is None

    def test_custom_config(self):
        cmp = BBoxIoUComparator(margin_percent=10.0)
        assert cmp.config == {"margin_percent": 10.0}


# ══════════════════════════════════════════════════════════════════════
# MAPCalculator tests
# ══════════════════════════════════════════════════════════════════════


class TestMAPCalculator:
    """Tests for the MAPCalculator."""

    def test_perfect_bbox_match(self):
        """All bboxes match perfectly — mAP should be 1.0."""
        gt = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [100, 20]]},
                "invoice_number": {"value": "INV-001", "bbox": [[0, 25], [100, 45]]},
                "total_amount": {"value": 1500.00, "bbox": [[0, 50], [100, 70]]},
            }
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [100, 20]]},
                "invoice_number": {"value": "INV-001", "bbox": [[0, 25], [100, 45]]},
                "total_amount": {"value": 1500.00, "bbox": [[0, 50], [100, 70]]},
            }
        )

        result = gt.compare_with(
            pred, add_bbox_metrics=True, document_field_comparisons=True
        )

        assert "bbox_metrics" in result
        bm = result["bbox_metrics"]
        assert bm["mean_ap"] == 1.0
        assert bm["iou_threshold"] == 0.5
        assert len(bm["field_results"]) == 3
        for field_data in bm["field_results"].values():
            assert field_data["iou"] == 1.0
            assert field_data["ap"] == 1.0

    def test_no_bbox_overlap(self):
        """Bboxes don't overlap at all — mAP should be 0.0."""
        gt = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [50, 20]]},
                "invoice_number": {"value": "INV-001", "bbox": [[0, 25], [50, 45]]},
            }
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[200, 200], [300, 220]]},
                "invoice_number": {
                    "value": "INV-001",
                    "bbox": [[200, 225], [300, 245]],
                },
            }
        )

        result = gt.compare_with(
            pred, add_bbox_metrics=True, document_field_comparisons=True
        )

        bm = result["bbox_metrics"]
        assert bm["mean_ap"] == 0.0
        for field_data in bm["field_results"].values():
            assert field_data["iou"] == 0.0
            assert field_data["ap"] == 0.0

    def test_partial_bbox_overlap(self):
        """Some bboxes match, some don't — mAP should be between 0 and 1."""
        gt = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [100, 20]]},
                "invoice_number": {"value": "INV-001", "bbox": [[0, 25], [100, 45]]},
            }
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {
                    "value": "Acme Corp",
                    "bbox": [[0, 0], [100, 20]],  # Perfect match
                },
                "invoice_number": {
                    "value": "INV-001",
                    "bbox": [[200, 200], [300, 220]],  # No overlap
                },
            }
        )

        result = gt.compare_with(
            pred, add_bbox_metrics=True, document_field_comparisons=True
        )

        bm = result["bbox_metrics"]
        assert bm["mean_ap"] == 0.5  # 1 match + 1 miss = 0.5

    def test_missing_pred_bbox(self):
        """Prediction has no bbox for a field — treated as miss."""
        gt = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [100, 20]]},
                "invoice_number": {"value": "INV-001", "bbox": [[0, 25], [100, 45]]},
            }
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {
                    "value": "Acme Corp",
                    "bbox": [[0, 0], [100, 20]],
                },
                "invoice_number": {"value": "INV-001"},  # No bbox
            }
        )

        result = gt.compare_with(
            pred, add_bbox_metrics=True, document_field_comparisons=True
        )

        bm = result["bbox_metrics"]
        assert bm["field_results"]["vendor_name"]["ap"] == 1.0
        assert bm["field_results"]["invoice_number"]["ap"] == 0.0
        assert bm["mean_ap"] == 0.5

    def test_no_gt_bbox_fields_skipped(self):
        """Fields without GT bbox are not counted in mAP."""
        gt = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [100, 20]]},
                "invoice_number": {"value": "INV-001"},  # No GT bbox
            }
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {
                    "value": "Acme Corp",
                    "bbox": [[0, 0], [100, 20]],
                },
                "invoice_number": {
                    "value": "INV-001",
                    "bbox": [[0, 25], [100, 45]],  # Has pred bbox but no GT
                },
            }
        )

        result = gt.compare_with(
            pred, add_bbox_metrics=True, document_field_comparisons=True
        )

        bm = result["bbox_metrics"]
        # Only vendor_name should be in field_results
        assert len(bm["field_results"]) == 1
        assert "vendor_name" in bm["field_results"]
        assert bm["mean_ap"] == 1.0

    def test_no_bbox_data_at_all(self):
        """When no fields have bbox data, metrics should be None."""
        gt = DocumentField(
            vendor_name="Acme Corp", invoice_number="INV-001", total_amount=1500.00
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": "Acme Corp",
                "invoice_number": "INV-001",
                "total_amount": 1500.00,
            }
        )

        result = gt.compare_with(
            pred, add_bbox_metrics=True, document_field_comparisons=True
        )

        bm = result["bbox_metrics"]
        assert bm["mean_ap"] is None
        assert bm["coverage"]["fields_with_bbox"] == 0

    def test_custom_iou_threshold(self):
        """Custom IoU threshold changes what counts as a match."""
        gt = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [100, 100]]},
            }
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {
                    "value": "Acme Corp",
                    # Overlaps partially: intersection area / union area
                    "bbox": [[50, 0], [150, 100]],
                },
            }
        )

        # IoU = 50*100 / (100*100 + 100*100 - 50*100) = 5000/15000 ≈ 0.333
        # At threshold 0.5 → miss
        result_strict = gt.compare_with(
            pred,
            add_bbox_metrics=True,
            document_field_comparisons=True,
            bbox_iou_threshold=0.5,
        )
        assert (
            result_strict["bbox_metrics"]["field_results"]["vendor_name"]["ap"] == 0.0
        )

        # At threshold 0.3 → match
        result_lenient = gt.compare_with(
            pred,
            add_bbox_metrics=True,
            document_field_comparisons=True,
            bbox_iou_threshold=0.3,
        )
        assert (
            result_lenient["bbox_metrics"]["field_results"]["vendor_name"]["ap"] == 1.0
        )

    def test_coverage_tracking(self):
        """Coverage should report how many fields had bbox data."""
        gt = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [100, 20]]},
                "invoice_number": {"value": "INV-001"},
                "total_amount": {"value": 1500.00, "bbox": [[0, 50], [100, 70]]},
            }
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [100, 20]]},
                "invoice_number": {"value": "INV-001"},
                "total_amount": {"value": 1500.00, "bbox": [[0, 50], [100, 70]]},
            }
        )

        result = gt.compare_with(
            pred, add_bbox_metrics=True, document_field_comparisons=True
        )

        coverage = result["bbox_metrics"]["coverage"]
        assert coverage["fields_with_bbox"] == 2
        assert coverage["fields_total"] == 3

    def test_bbox_with_confidence(self):
        """Bbox and confidence can coexist in the same rich value."""
        gt = DocumentField.from_json(
            {
                "vendor_name": {
                    "value": "Acme Corp",
                    "bbox": [[0, 0], [100, 20]],
                    "confidence": 0.95,
                },
                "invoice_number": {
                    "value": "INV-001",
                    "bbox": [[0, 25], [100, 45]],
                    "confidence": 0.8,
                },
            }
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {
                    "value": "Acme Corp",
                    "bbox": [[0, 0], [100, 20]],
                    "confidence": 0.9,
                },
                "invoice_number": {
                    "value": "INV-001",
                    "bbox": [[0, 25], [100, 45]],
                    "confidence": 0.7,
                },
            }
        )

        result = gt.compare_with(
            pred,
            add_bbox_metrics=True,
            add_confidence_metrics=True,
            document_field_comparisons=True,
        )

        # Both metrics should be present
        assert "bbox_metrics" in result
        assert "confidence_metrics" in result
        assert result["bbox_metrics"]["mean_ap"] == 1.0

    def test_flat_bbox_format(self):
        """Flat [x1, y1, x2, y2] format works in rich values."""
        gt = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [0, 0, 100, 20]},
            }
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [0, 0, 100, 20]},
            }
        )

        result = gt.compare_with(
            pred, add_bbox_metrics=True, document_field_comparisons=True
        )

        assert result["bbox_metrics"]["mean_ap"] == 1.0


# ══════════════════════════════════════════════════════════════════════
# Auto-enable field_comparisons tests
# ══════════════════════════════════════════════════════════════════════


class TestAutoEnableFieldComparisons:
    """Verify that add_bbox_metrics auto-enables document_field_comparisons."""

    def test_bbox_metrics_without_explicit_field_comparisons(self):
        """add_bbox_metrics=True should work without explicit document_field_comparisons."""
        gt = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [100, 20]]},
            }
        )
        pred = DocumentField.from_json(
            {
                "vendor_name": {"value": "Acme Corp", "bbox": [[0, 0], [100, 20]]},
            }
        )

        # Should not raise — field_comparisons auto-enabled
        result = gt.compare_with(pred, add_bbox_metrics=True)
        assert "bbox_metrics" in result
        assert result["bbox_metrics"]["mean_ap"] == 1.0
