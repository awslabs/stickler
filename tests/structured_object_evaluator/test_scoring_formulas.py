"""Focused unit tests for scoring_formulas (precision, recall, F1, accuracy).

Previously these formulas lived only inside MetricsHelper and were exercised
solely through end-to-end compare_with() integration tests. Per issue #134's
testing guidance, the rename is a good moment to add direct formula tests,
including the zero-denominator edge cases the integration tests don't
specifically target.
"""

from stickler.structured_object_evaluator.models.scoring_formulas import (
    calculate_derived_metrics,
    convert_score_to_binary_metrics,
)


class TestCalculateDerivedMetrics:
    def test_all_true_positive(self):
        metrics = {"tp": 5, "fp": 0, "tn": 0, "fn": 0, "fd": 0, "fa": 0}
        result = calculate_derived_metrics(metrics)
        assert result["cm_precision"] == 1.0
        assert result["cm_recall"] == 1.0
        assert result["cm_f1"] == 1.0
        assert result["cm_accuracy"] == 1.0

    def test_all_false_negative(self):
        metrics = {"tp": 0, "fp": 0, "tn": 0, "fn": 5, "fd": 0, "fa": 0}
        result = calculate_derived_metrics(metrics)
        assert result["cm_precision"] == 0.0
        assert result["cm_recall"] == 0.0
        assert result["cm_f1"] == 0.0
        assert result["cm_accuracy"] == 0.0

    def test_zero_denominator_everything_zero(self):
        # No predictions and no ground truth at all - every ratio's
        # denominator is 0. Must return 0.0, not raise ZeroDivisionError.
        metrics = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "fd": 0, "fa": 0}
        result = calculate_derived_metrics(metrics)
        assert result["cm_precision"] == 0.0
        assert result["cm_recall"] == 0.0
        assert result["cm_f1"] == 0.0
        assert result["cm_accuracy"] == 0.0

    def test_mixed_counts_traditional_recall(self):
        # precision uses fa+fd (not the fp field): 8 / (8 + 1 + 1) = 8/10
        # recall (traditional) = 8 / (8 + 1) = 8/9
        metrics = {"tp": 8, "fp": 2, "tn": 0, "fn": 1, "fd": 1, "fa": 1}
        result = calculate_derived_metrics(metrics, recall_with_fd=False)
        assert result["cm_precision"] == 8 / 10
        assert result["cm_recall"] == 8 / 9

    def test_recall_with_fd_includes_fd_in_denominator(self):
        # recall_with_fd: TP / (TP + FN + FD) = 8 / (8 + 1 + 1)
        metrics = {"tp": 8, "fp": 2, "tn": 0, "fn": 1, "fd": 1, "fa": 1}
        result = calculate_derived_metrics(metrics, recall_with_fd=True)
        assert result["cm_recall"] == 8 / 10

    def test_accuracy_includes_true_negatives(self):
        metrics = {"tp": 3, "fp": 1, "tn": 4, "fn": 2, "fd": 1, "fa": 0}
        result = calculate_derived_metrics(metrics)
        assert result["cm_accuracy"] == (3 + 4) / (3 + 4 + 1 + 2)


class TestConvertScoreToBinaryMetrics:
    def test_perfect_score(self):
        result = convert_score_to_binary_metrics(1.0, threshold=0.5)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_zero_score(self):
        result = convert_score_to_binary_metrics(0.0, threshold=0.5)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_score_above_threshold_gets_partial_credit(self):
        result = convert_score_to_binary_metrics(0.8, threshold=0.5)
        assert result["anls_score"] == 0.8
        assert result["precision"] > 0.0

    def test_score_below_threshold(self):
        # Below threshold, tp is always 0, so both ratios are 0 - only
        # anls_score carries the raw similarity through.
        result = convert_score_to_binary_metrics(0.3, threshold=0.5)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["anls_score"] == 0.3

    def test_score_exactly_at_threshold_counts_as_match(self):
        # fn is always 0 once score >= threshold, so recall is exactly 1.0
        # regardless of how close the score is to a perfect match; precision
        # reflects the partial credit instead (0.5 here, not 1.0).
        result = convert_score_to_binary_metrics(0.5, threshold=0.5)
        assert result["recall"] == 1.0
        assert result["precision"] == 0.5
