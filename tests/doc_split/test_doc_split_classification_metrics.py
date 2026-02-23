"""Tests for DocSplitClassificationMetrics.

Covers the three classical metrics (page-level accuracy, split accuracy
without order, split accuracy with order), load_sections input parsing,
edge cases, and markdown report generation.

Edge case scenarios are drawn from Appendix A of the DocSplit paper
(arXiv:2602.15958, Table 3).
"""

import pytest

from stickler.doc_split import DocSplitClassificationMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section(section_id, document_class, page_indices):
    """Shorthand for creating a section dict."""
    return {
        "section_id": section_id,
        "document_class": document_class,
        "page_indices": page_indices,
    }


def _make_nested_section(section_id, doc_class_type, page_indices):
    """Create a section dict in the accelerator nested format."""
    return {
        "section_id": section_id,
        "document_class": {"type": doc_class_type},
        "split_document": {"page_indices": page_indices},
    }


# ---------------------------------------------------------------------------
# Perfect match
# ---------------------------------------------------------------------------


class TestPerfectMatch:
    """All predictions match ground truth exactly."""

    def setup_method(self):
        self.gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "form", [3, 4]),
        ]
        self.pred = [
            _make_section("p1", "invoice", [0, 1, 2]),
            _make_section("p2", "form", [3, 4]),
        ]

    def test_page_level_accuracy(self):
        m = DocSplitClassificationMetrics()
        m.load_sections(self.gt, self.pred)
        result = m.calculate_page_level_accuracy()
        assert result["accuracy"] == 1.0
        assert result["total_pages"] == 5
        assert result["correct_pages"] == 5

    def test_split_without_order(self):
        m = DocSplitClassificationMetrics()
        m.load_sections(self.gt, self.pred)
        result = m.calculate_split_accuracy_without_order()
        assert result["accuracy"] == 1.0
        assert result["correct_sections"] == 2

    def test_split_with_order(self):
        m = DocSplitClassificationMetrics()
        m.load_sections(self.gt, self.pred)
        result = m.calculate_split_accuracy_with_order()
        assert result["accuracy"] == 1.0
        assert result["correct_sections"] == 2

    def test_calculate_all_metrics(self):
        m = DocSplitClassificationMetrics()
        m.load_sections(self.gt, self.pred)
        results = m.calculate_all_metrics()
        assert results["page_level_accuracy"]["accuracy"] == 1.0
        assert results["split_accuracy_without_order"]["accuracy"] == 1.0
        assert results["split_accuracy_with_order"]["accuracy"] == 1.0
        assert results["errors"] == []


# ---------------------------------------------------------------------------
# Paper edge cases (Table 3, Appendix A)
# ---------------------------------------------------------------------------


class TestMisclassificationOnly:
    """Correct grouping and ordering, but class labels swapped."""

    def test_page_level_zero(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "form", [3, 4]),
        ]
        pred = [
            _make_section("p1", "form", [0, 1, 2]),
            _make_section("p2", "invoice", [3, 4]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        result = m.calculate_page_level_accuracy()
        assert result["accuracy"] == 0.0

    def test_split_without_order_zero(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "form", [3, 4]),
        ]
        pred = [
            _make_section("p1", "form", [0, 1, 2]),
            _make_section("p2", "invoice", [3, 4]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        result = m.calculate_split_accuracy_without_order()
        assert result["accuracy"] == 0.0


class TestWrongGroupingOnly:
    """Correct classification and ordering, but group IDs swapped.

    Classical metrics treat this as correct because page-level class is right
    and the page sets still match (just with swapped section IDs).
    """

    def test_page_level_perfect(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "invoice", [3, 4]),
        ]
        # Prediction swaps which pages go in which group, but same class
        pred = [
            _make_section("p1", "invoice", [3, 4]),
            _make_section("p2", "invoice", [0, 1, 2]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        result = m.calculate_page_level_accuracy()
        assert result["accuracy"] == 1.0

    def test_split_without_order_perfect(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "invoice", [3, 4]),
        ]
        pred = [
            _make_section("p1", "invoice", [3, 4]),
            _make_section("p2", "invoice", [0, 1, 2]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        result = m.calculate_split_accuracy_without_order()
        assert result["accuracy"] == 1.0


class TestWrongOrderingOnly:
    """Correct classification and grouping, but page order scrambled."""

    def test_page_level_perfect(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "form", [3, 4]),
        ]
        pred = [
            _make_section("p1", "invoice", [2, 0, 1]),
            _make_section("p2", "form", [4, 3]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        result = m.calculate_page_level_accuracy()
        assert result["accuracy"] == 1.0

    def test_split_without_order_perfect(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "form", [3, 4]),
        ]
        pred = [
            _make_section("p1", "invoice", [2, 0, 1]),
            _make_section("p2", "form", [4, 3]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        result = m.calculate_split_accuracy_without_order()
        assert result["accuracy"] == 1.0

    def test_split_with_order_zero(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "form", [3, 4]),
        ]
        pred = [
            _make_section("p1", "invoice", [2, 0, 1]),
            _make_section("p2", "form", [4, 3]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        result = m.calculate_split_accuracy_with_order()
        assert result["accuracy"] == 0.0
        # order_matched should be False for both
        for detail in result["section_details"]:
            assert detail["order_matched"] is False


class TestSplitGroups:
    """One document split into two groups (over-segmentation)."""

    def test_split_accuracy_drops(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "form", [3, 4]),
        ]
        # Invoice split into two predicted sections
        pred = [
            _make_section("p1", "invoice", [0, 1]),
            _make_section("p2", "invoice", [2]),
            _make_section("p3", "form", [3, 4]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)

        # Page level still correct (all pages classified correctly)
        page_result = m.calculate_page_level_accuracy()
        assert page_result["accuracy"] == 1.0

        # Split without order: invoice section won't match (different page sets)
        split_result = m.calculate_split_accuracy_without_order()
        assert split_result["correct_sections"] == 1  # only form matches
        assert split_result["accuracy"] == 0.5


class TestMergedGroups:
    """Two documents merged into one group (under-segmentation)."""

    def test_split_accuracy_drops(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "form", [3, 4]),
        ]
        # Both merged into one predicted section
        pred = [
            _make_section("p1", "invoice", [0, 1, 2, 3, 4]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)

        # Page level: pages 3,4 are classified as invoice instead of form
        page_result = m.calculate_page_level_accuracy()
        assert page_result["correct_pages"] == 3  # only 0,1,2 correct
        assert page_result["accuracy"] == pytest.approx(3 / 5)

        # Split: neither section matches
        split_result = m.calculate_split_accuracy_without_order()
        assert split_result["correct_sections"] == 0


class TestPartialMisclassification:
    """One page misclassified within otherwise correct group."""

    def test_page_level_partial(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "form", [3, 4]),
        ]
        # Page 2 classified as form instead of invoice
        pred = [
            _make_section("p1", "invoice", [0, 1]),
            _make_section("p2", "form", [2, 3, 4]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)

        page_result = m.calculate_page_level_accuracy()
        assert page_result["correct_pages"] == 4  # page 2 wrong
        assert page_result["accuracy"] == pytest.approx(4 / 5)


# ---------------------------------------------------------------------------
# Input format variations
# ---------------------------------------------------------------------------


class TestNestedInputFormat:
    """Test the accelerator nested format with document_class.type and
    split_document.page_indices."""

    def test_nested_format_works(self):
        gt = [_make_nested_section("s1", "invoice", [0, 1, 2])]
        pred = [_make_nested_section("p1", "invoice", [0, 1, 2])]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        result = m.calculate_all_metrics()
        assert result["page_level_accuracy"]["accuracy"] == 1.0
        assert result["split_accuracy_with_order"]["accuracy"] == 1.0


class TestMixedInputFormats:
    """Test mixing flat and nested formats."""

    def test_mixed_formats(self):
        gt = [_make_section("s1", "invoice", [0, 1])]
        pred = [_make_nested_section("p1", "invoice", [0, 1])]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        result = m.calculate_page_level_accuracy()
        assert result["accuracy"] == 1.0


class TestAutoGeneratedSectionIds:
    """Section IDs are auto-generated when not provided."""

    def test_missing_section_id(self):
        gt = [{"document_class": "invoice", "page_indices": [0]}]
        pred = [{"document_class": "invoice", "page_indices": [0]}]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        assert m.sections_gt[0]["section_id"] == "gt_0"
        assert m.sections_pred[0]["section_id"] == "pred_0"
        result = m.calculate_all_metrics()
        assert result["page_level_accuracy"]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    """Empty section lists."""

    def test_empty_gt_and_pred(self):
        m = DocSplitClassificationMetrics()
        m.load_sections([], [])
        results = m.calculate_all_metrics()
        assert results["page_level_accuracy"]["accuracy"] == 0.0
        assert results["page_level_accuracy"]["total_pages"] == 0
        assert results["split_accuracy_without_order"]["accuracy"] == 0.0
        assert results["split_accuracy_with_order"]["accuracy"] == 0.0

    def test_empty_pred(self):
        gt = [_make_section("s1", "invoice", [0, 1])]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, [])
        page_result = m.calculate_page_level_accuracy()
        # GT pages exist but pred pages are "Missing"
        assert page_result["accuracy"] == 0.0
        assert page_result["total_pages"] == 2

    def test_empty_gt(self):
        pred = [_make_section("p1", "invoice", [0, 1])]
        m = DocSplitClassificationMetrics()
        m.load_sections([], pred)
        page_result = m.calculate_page_level_accuracy()
        assert page_result["accuracy"] == 0.0
        assert page_result["total_pages"] == 2


class TestSinglePageDocument:
    """Single-page documents."""

    def test_single_page_perfect(self):
        gt = [_make_section("s1", "letter", [0])]
        pred = [_make_section("p1", "letter", [0])]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        results = m.calculate_all_metrics()
        assert results["page_level_accuracy"]["accuracy"] == 1.0
        assert results["split_accuracy_with_order"]["accuracy"] == 1.0


class TestEmptyPageIndices:
    """Sections with no page indices."""

    def test_empty_page_indices_handled(self):
        gt = [_make_section("s1", "invoice", [])]
        pred = [_make_section("p1", "invoice", [])]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        results = m.calculate_all_metrics()
        assert results["page_level_accuracy"]["total_pages"] == 0


class TestInvalidPageIndices:
    """Non-integer page indices are handled gracefully."""

    def test_string_indices_converted(self):
        gt = [_make_section("s1", "invoice", ["0", "1"])]
        pred = [_make_section("p1", "invoice", [0, 1])]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        result = m.calculate_page_level_accuracy()
        assert result["accuracy"] == 1.0


class TestNonSequentialPageIndices:
    """Page indices that are not sequential (e.g., interleaved documents)."""

    def test_non_sequential(self):
        gt = [
            _make_section("s1", "invoice", [0, 2, 4]),
            _make_section("s2", "form", [1, 3]),
        ]
        pred = [
            _make_section("p1", "invoice", [0, 2, 4]),
            _make_section("p2", "form", [1, 3]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        results = m.calculate_all_metrics()
        assert results["page_level_accuracy"]["accuracy"] == 1.0
        assert results["split_accuracy_with_order"]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    """Test markdown report generation."""

    def test_report_contains_key_sections(self):
        gt = [
            _make_section("s1", "invoice", [0, 1, 2]),
            _make_section("s2", "form", [3, 4]),
        ]
        pred = [
            _make_section("p1", "invoice", [0, 1, 2]),
            _make_section("p2", "form", [3, 4]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        metrics = m.calculate_all_metrics()
        report = m.generate_markdown_report(metrics)

        assert "# Document Split Classification Evaluation" in report
        assert "Split Classification Summary" in report
        assert "Split Classification Metrics" in report
        assert "Section Split Analysis" in report
        assert "Metrics Explanation" in report
        assert "1.0000" in report  # perfect accuracy

    def test_report_with_errors(self):
        m = DocSplitClassificationMetrics()
        m.errors = ["Something went wrong"]
        m.load_sections([], [])
        metrics = m.calculate_all_metrics()
        report = m.generate_markdown_report(metrics)
        assert "Errors Encountered" in report
        assert "Something went wrong" in report

    def test_report_shows_unmatched_predictions(self):
        gt = [_make_section("s1", "invoice", [0, 1])]
        pred = [
            _make_section("p1", "invoice", [0, 1]),
            _make_section("p2", "form", [2, 3]),  # extra prediction
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(gt, pred)
        metrics = m.calculate_all_metrics()
        report = m.generate_markdown_report(metrics)
        # The unmatched prediction should appear in the report
        assert "p2" in report


# ---------------------------------------------------------------------------
# Multi-document realistic scenario
# ---------------------------------------------------------------------------


class TestRealisticPacket:
    """Realistic 10-page packet with 3 document types."""

    def setup_method(self):
        # A lending packet: 1099 (3 pages), W2 (4 pages), pay stub (3 pages)
        self.gt = [
            _make_section("1099", "1099", [0, 1, 2]),
            _make_section("w2", "w2", [3, 4, 5, 6]),
            _make_section("paystub", "pay_stub", [7, 8, 9]),
        ]

    def test_perfect_prediction(self):
        pred = [
            _make_section("p1", "1099", [0, 1, 2]),
            _make_section("p2", "w2", [3, 4, 5, 6]),
            _make_section("p3", "pay_stub", [7, 8, 9]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(self.gt, pred)
        results = m.calculate_all_metrics()
        assert results["page_level_accuracy"]["accuracy"] == 1.0
        assert results["split_accuracy_with_order"]["accuracy"] == 1.0

    def test_boundary_error(self):
        """Page 6 (last W2 page) incorrectly assigned to pay_stub."""
        pred = [
            _make_section("p1", "1099", [0, 1, 2]),
            _make_section("p2", "w2", [3, 4, 5]),
            _make_section("p3", "pay_stub", [6, 7, 8, 9]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(self.gt, pred)
        results = m.calculate_all_metrics()

        # Page 6 is classified as pay_stub instead of w2
        assert results["page_level_accuracy"]["correct_pages"] == 9
        assert results["page_level_accuracy"]["accuracy"] == pytest.approx(9 / 10)

        # Only 1099 section matches (w2 and pay_stub have wrong page sets)
        assert results["split_accuracy_without_order"]["correct_sections"] == 1

    def test_correct_pages_wrong_order(self):
        """Correct grouping but pages within W2 are shuffled."""
        pred = [
            _make_section("p1", "1099", [0, 1, 2]),
            _make_section("p2", "w2", [6, 3, 5, 4]),  # shuffled
            _make_section("p3", "pay_stub", [7, 8, 9]),
        ]
        m = DocSplitClassificationMetrics()
        m.load_sections(self.gt, pred)
        results = m.calculate_all_metrics()

        assert results["page_level_accuracy"]["accuracy"] == 1.0
        assert results["split_accuracy_without_order"]["accuracy"] == 1.0
        # W2 order doesn't match
        assert results["split_accuracy_with_order"]["correct_sections"] == 2
        assert results["split_accuracy_with_order"]["accuracy"] == pytest.approx(2 / 3)
