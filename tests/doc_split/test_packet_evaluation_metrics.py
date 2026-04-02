"""Tests for packet evaluation metrics (proposed metrics from DocSplit paper).

Edge case scenarios and expected values are from Appendix A of the DocSplit
paper (arXiv:2602.15958, Tables 3-4) and the edge_cases_evaluation.ipynb
notebook. All tests use strict_clustering=True to match the paper results.

Each test case uses 5 pages: 3 "invoice" pages (group inv-01) and
2 "form" pages (group form-01).
"""

import os
import tempfile

import pandas as pd
import pytest

from stickler.doc_split.packet_evaluation_metrics import (
    calculate_average_ordering_score,
    calculate_clustering_score,
    calculate_final_score,
    calculate_ordering_score_per_group,
    evaluate_packet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page(
    class_label,
    group_id,
    page_number,
    class_label_predicted,
    group_id_predicted,
    page_number_predicted,
):
    """Shorthand for creating a page dict."""
    return {
        "class_label": class_label,
        "group_id": group_id,
        "page_number": page_number,
        "class_label_predicted": class_label_predicted,
        "group_id_predicted": group_id_predicted,
        "page_number_predicted": page_number_predicted,
    }


# ---------------------------------------------------------------------------
# Edge case data (from edge_cases_test_data.csv)
# ---------------------------------------------------------------------------

PERFECT = [
    _page("invoice", "inv-01", 1, "invoice", "inv-01", 1),
    _page("invoice", "inv-01", 2, "invoice", "inv-01", 2),
    _page("invoice", "inv-01", 3, "invoice", "inv-01", 3),
    _page("form", "form-01", 4, "form", "form-01", 4),
    _page("form", "form-01", 5, "form", "form-01", 5),
]

MISCLASSIFICATION_ONLY = [
    _page("invoice", "inv-01", 1, "form", "inv-01", 1),
    _page("invoice", "inv-01", 2, "form", "inv-01", 2),
    _page("invoice", "inv-01", 3, "form", "inv-01", 3),
    _page("form", "form-01", 4, "invoice", "form-01", 4),
    _page("form", "form-01", 5, "invoice", "form-01", 5),
]

WRONG_GROUPING_ONLY = [
    _page("invoice", "inv-01", 1, "invoice", "form-01", 1),
    _page("invoice", "inv-01", 2, "invoice", "form-01", 2),
    _page("invoice", "inv-01", 3, "invoice", "form-01", 3),
    _page("form", "form-01", 4, "form", "inv-01", 4),
    _page("form", "form-01", 5, "form", "inv-01", 5),
]

WRONG_ORDERING_ONLY = [
    _page("invoice", "inv-01", 1, "invoice", "inv-01", 3),
    _page("invoice", "inv-01", 2, "invoice", "inv-01", 1),
    _page("invoice", "inv-01", 3, "invoice", "inv-01", 2),
    _page("form", "form-01", 4, "form", "form-01", 5),
    _page("form", "form-01", 5, "form", "form-01", 4),
]

SPLIT_GROUPS = [
    _page("invoice", "inv-01", 1, "invoice", "inv-01", 1),
    _page("invoice", "inv-01", 2, "invoice", "inv-02", 2),
    _page("invoice", "inv-01", 3, "invoice", "inv-01", 3),
    _page("form", "form-01", 4, "form", "form-01", 4),
    _page("form", "form-01", 5, "form", "form-02", 5),
]

MERGED_GROUPS = [
    _page("invoice", "inv-01", 1, "invoice", "mixed-01", 1),
    _page("invoice", "inv-01", 2, "invoice", "mixed-01", 2),
    _page("invoice", "inv-01", 3, "invoice", "mixed-01", 3),
    _page("form", "form-01", 4, "form", "mixed-01", 4),
    _page("form", "form-01", 5, "form", "mixed-01", 5),
]

PARTIAL_MISCLASS = [
    _page("invoice", "inv-01", 1, "invoice", "inv-01", 1),
    _page("invoice", "inv-01", 2, "form", "inv-01", 2),
    _page("invoice", "inv-01", 3, "invoice", "inv-01", 3),
    _page("form", "form-01", 4, "form", "form-01", 4),
    _page("form", "form-01", 5, "form", "form-01", 5),
]

MULTIPLE_ERRORS = [
    _page("invoice", "inv-01", 1, "form", "form-01", 3),
    _page("invoice", "inv-01", 2, "form", "form-01", 1),
    _page("invoice", "inv-01", 3, "invoice", "inv-02", 2),
    _page("form", "form-01", 4, "invoice", "inv-01", 5),
    _page("form", "form-01", 5, "invoice", "inv-01", 4),
]

DUPLICATE_PAGE_NUMS = [
    _page("invoice", "inv-01", 1, "invoice", "inv-01", 1),
    _page("invoice", "inv-01", 2, "invoice", "inv-01", 1),
    _page("invoice", "inv-01", 3, "invoice", "inv-01", 3),
    _page("form", "form-01", 4, "form", "form-01", 4),
    _page("form", "form-01", 5, "form", "form-01", 5),
]

REVERSE_ORDER = [
    _page("invoice", "inv-01", 1, "invoice", "inv-01", 3),
    _page("invoice", "inv-01", 2, "invoice", "inv-01", 2),
    _page("invoice", "inv-01", 3, "invoice", "inv-01", 1),
    _page("form", "form-01", 4, "form", "form-01", 5),
    _page("form", "form-01", 5, "form", "form-01", 4),
]


# ---------------------------------------------------------------------------
# Test evaluate_packet with all 10 edge cases (strict mode)
# Expected values from paper Table 4 and notebook outputs
# ---------------------------------------------------------------------------


class TestPerfect:
    def test_all_scores_perfect(self):
        r = evaluate_packet(PERFECT, strict_clustering=True)
        assert r["final_score"] == pytest.approx(1.0, abs=1e-4)
        assert r["clustering_score"] == pytest.approx(1.0, abs=1e-4)
        assert r["v_measure"] == pytest.approx(1.0, abs=1e-4)
        assert r["rand_index"] == pytest.approx(1.0, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(1.0, abs=1e-4)


class TestMisclassificationOnly:
    """Correct grouping and ordering, but class labels swapped."""

    def test_scores(self):
        r = evaluate_packet(MISCLASSIFICATION_ONLY, strict_clustering=True)
        assert r["final_score"] == pytest.approx(0.7974, abs=1e-4)
        assert r["clustering_score"] == pytest.approx(0.5949, abs=1e-4)
        assert r["v_measure"] == pytest.approx(0.5897, abs=1e-4)
        assert r["rand_index"] == pytest.approx(0.6000, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(1.0, abs=1e-4)


class TestWrongGroupingOnly:
    """Group IDs swapped but structure preserved — full credit."""

    def test_scores(self):
        r = evaluate_packet(WRONG_GROUPING_ONLY, strict_clustering=True)
        assert r["final_score"] == pytest.approx(1.0, abs=1e-4)
        assert r["clustering_score"] == pytest.approx(1.0, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(1.0, abs=1e-4)


class TestWrongOrderingOnly:
    """Correct grouping but pages scrambled within groups."""

    def test_scores(self):
        r = evaluate_packet(WRONG_ORDERING_ONLY, strict_clustering=True)
        assert r["final_score"] == pytest.approx(0.1667, abs=1e-4)
        assert r["clustering_score"] == pytest.approx(1.0, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(-0.6667, abs=1e-4)


class TestSplitGroups:
    """Over-segmentation: one document split into two groups."""

    def test_scores(self):
        r = evaluate_packet(SPLIT_GROUPS, strict_clustering=True)
        assert r["final_score"] == pytest.approx(0.8428, abs=1e-4)
        assert r["clustering_score"] == pytest.approx(0.6856, abs=1e-4)
        assert r["v_measure"] == pytest.approx(0.6713, abs=1e-4)
        assert r["rand_index"] == pytest.approx(0.7000, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(1.0, abs=1e-4)


class TestMergedGroups:
    """Under-segmentation: two documents merged into one group."""

    def test_scores(self):
        r = evaluate_packet(MERGED_GROUPS, strict_clustering=True)
        assert r["final_score"] == pytest.approx(0.6000, abs=1e-4)
        assert r["clustering_score"] == pytest.approx(0.2000, abs=1e-4)
        assert r["v_measure"] == pytest.approx(0.0, abs=1e-4)
        assert r["rand_index"] == pytest.approx(0.4000, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(1.0, abs=1e-4)


class TestPartialMisclass:
    """One page misclassified within otherwise correct group."""

    def test_scores(self):
        r = evaluate_packet(PARTIAL_MISCLASS, strict_clustering=True)
        assert r["final_score"] == pytest.approx(0.8947, abs=1e-4)
        assert r["clustering_score"] == pytest.approx(0.7895, abs=1e-4)
        assert r["v_measure"] == pytest.approx(0.7790, abs=1e-4)
        assert r["rand_index"] == pytest.approx(0.8000, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(1.0, abs=1e-4)


class TestMultipleErrors:
    """Combination of classification, grouping, and ordering errors."""

    def test_scores(self):
        r = evaluate_packet(MULTIPLE_ERRORS, strict_clustering=True)
        assert r["final_score"] == pytest.approx(-0.0359, abs=1e-4)
        assert r["clustering_score"] == pytest.approx(0.5949, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(-0.6667, abs=1e-4)


class TestDuplicatePageNums:
    """Same page number predicted for multiple pages."""

    def test_scores(self):
        r = evaluate_packet(DUPLICATE_PAGE_NUMS, strict_clustering=True)
        assert r["final_score"] == pytest.approx(0.9541, abs=1e-4)
        assert r["clustering_score"] == pytest.approx(1.0, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(0.9082, abs=1e-4)


class TestReverseOrder:
    """Pages in completely reversed sequence."""

    def test_scores(self):
        r = evaluate_packet(REVERSE_ORDER, strict_clustering=True)
        assert r["final_score"] == pytest.approx(0.0, abs=1e-4)
        assert r["clustering_score"] == pytest.approx(1.0, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(-1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test non-strict mode (standard clustering without class penalty)
# ---------------------------------------------------------------------------


class TestNonStrictMode:
    """Non-strict mode should not penalize misclassification in clustering."""

    def test_misclassification_no_penalty(self):
        r = evaluate_packet(MISCLASSIFICATION_ONLY, strict_clustering=False)
        # Without strict mode, clustering only looks at group_id
        # which is correct, so clustering should be perfect
        assert r["clustering_score"] == pytest.approx(1.0, abs=1e-4)
        assert r["avg_ordering_score"] == pytest.approx(1.0, abs=1e-4)
        assert r["final_score"] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test individual functions
# ---------------------------------------------------------------------------


class TestCalculateClusteringScore:
    def test_perfect_clustering(self):
        df = pd.DataFrame(PERFECT)
        score, v, ri = calculate_clustering_score(df)
        assert score == pytest.approx(1.0, abs=1e-4)
        assert v == pytest.approx(1.0, abs=1e-4)
        assert ri == pytest.approx(1.0, abs=1e-4)

    def test_custom_weight(self):
        df = pd.DataFrame(PERFECT)
        # All v_measure weight
        score, v, ri = calculate_clustering_score(df, v_measure_weight=1.0)
        assert score == pytest.approx(v, abs=1e-4)
        # All rand index weight
        score, v, ri = calculate_clustering_score(df, v_measure_weight=0.0)
        assert score == pytest.approx(ri, abs=1e-4)

    def test_invalid_weight_raises(self):
        df = pd.DataFrame(PERFECT)
        with pytest.raises(ValueError):
            calculate_clustering_score(df, v_measure_weight=1.5)

    def test_missing_columns_raises(self):
        df = pd.DataFrame([{"foo": 1}])
        with pytest.raises(KeyError):
            calculate_clustering_score(df)


class TestCalculateOrderingScore:
    def test_perfect_ordering(self):
        df = pd.DataFrame(PERFECT)
        scores = calculate_ordering_score_per_group(df)
        for group_id, tau in scores.items():
            assert tau == pytest.approx(1.0, abs=1e-4)

    def test_reverse_ordering(self):
        df = pd.DataFrame(REVERSE_ORDER)
        scores = calculate_ordering_score_per_group(df)
        avg = calculate_average_ordering_score(scores)
        assert avg == pytest.approx(-1.0, abs=1e-4)

    def test_single_page_groups_score_perfect(self):
        """Single-page groups should receive a perfect ordering score of 1.0."""
        data = [
            _page("invoice", "inv-01", 1, "invoice", "inv-01", 1),
            _page("form", "form-01", 2, "form", "form-01", 2),
        ]
        df = pd.DataFrame(data)
        scores = calculate_ordering_score_per_group(df)
        assert len(scores) == 2
        assert all(v == 1.0 for v in scores.values())
        assert calculate_average_ordering_score(scores) == 1.0

    def test_missing_columns_raises(self):
        df = pd.DataFrame([{"foo": 1}])
        with pytest.raises(KeyError):
            calculate_ordering_score_per_group(df)


class TestCalculateFinalScore:
    def test_default_weights(self):
        score = calculate_final_score(0.8, 0.8, 0.9)
        assert score == pytest.approx(0.85, abs=1e-4)

    def test_custom_weights(self):
        score = calculate_final_score(0.8, 0.8, 0.9, alpha=0.7, beta=0.3)
        assert score == pytest.approx(0.83, abs=1e-4)

    def test_zero_weights(self):
        score = calculate_final_score(0.8, 0.8, 0.9, alpha=0.0, beta=1.0)
        assert score == pytest.approx(0.9, abs=1e-4)


# ---------------------------------------------------------------------------
# Test input format flexibility
# ---------------------------------------------------------------------------


class TestInputFormats:
    """evaluate_packet should accept list-of-dicts, CSV path, or DataFrame."""

    def test_list_of_dicts(self):
        r = evaluate_packet(PERFECT)
        assert r["final_score"] == pytest.approx(1.0, abs=1e-4)

    def test_dataframe(self):
        df = pd.DataFrame(PERFECT)
        r = evaluate_packet(df)
        assert r["final_score"] == pytest.approx(1.0, abs=1e-4)

    def test_csv_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(PERFECT)
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            r = evaluate_packet(csv_path)
            assert r["final_score"] == pytest.approx(1.0, abs=1e-4)
        finally:
            os.unlink(csv_path)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            evaluate_packet(12345)


# ---------------------------------------------------------------------------
# Test result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_result_keys(self):
        r = evaluate_packet(PERFECT)
        expected_keys = {
            "final_score",
            "clustering_score",
            "v_measure",
            "rand_index",
            "avg_ordering_score",
            "group_ordering_scores",
            "strict_clustering",
        }
        assert set(r.keys()) == expected_keys

    def test_strict_flag_in_result(self):
        r1 = evaluate_packet(PERFECT, strict_clustering=False)
        assert r1["strict_clustering"] is False
        r2 = evaluate_packet(PERFECT, strict_clustering=True)
        assert r2["strict_clustering"] is True
