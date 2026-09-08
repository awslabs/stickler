"""Equivalence tests for the dependency-free runtime statistics."""

import numpy as np
import pytest
from scipy.stats import kendalltau
from sklearn.metrics import (
    homogeneity_completeness_v_measure as sklearn_v_measure,
)
from sklearn.metrics import rand_score, roc_auc_score

from stickler.utils.statistics import (
    binary_roc_auc,
    homogeneity_completeness_v_measure,
    kendall_tau_b,
    rand_index,
)


def test_binary_roc_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(216)
    for _ in range(250):
        sample_count = int(rng.integers(2, 80))
        labels = rng.integers(0, 2, size=sample_count)
        labels[0] = 0
        labels[-1] = 1
        scores = rng.integers(0, 5, size=sample_count) / 4

        assert binary_roc_auc(labels, scores) == pytest.approx(
            roc_auc_score(labels, scores), abs=1e-12
        )


def test_binary_roc_auc_all_equal_scores_is_half():
    assert binary_roc_auc([False, True, False, True], [0.4] * 4) == 0.5


def test_binary_roc_auc_rejects_a_single_class():
    with pytest.raises(ValueError, match="both positive and negative"):
        binary_roc_auc([True, True], [0.2, 0.8])


def test_clustering_metrics_match_sklearn():
    rng = np.random.default_rng(217)
    for _ in range(250):
        sample_count = int(rng.integers(1, 80))
        labels_true = rng.integers(0, int(rng.integers(1, 10)), size=sample_count)
        labels_pred = rng.integers(0, int(rng.integers(1, 10)), size=sample_count)

        expected_h, expected_c, expected_v = sklearn_v_measure(
            labels_true, labels_pred
        )
        actual_h, actual_c, actual_v = homogeneity_completeness_v_measure(
            labels_true, labels_pred
        )
        assert actual_h == pytest.approx(expected_h, abs=1e-12)
        assert actual_c == pytest.approx(expected_c, abs=1e-12)
        assert actual_v == pytest.approx(expected_v, abs=1e-12)
        assert rand_index(labels_true, labels_pred) == rand_score(
            labels_true, labels_pred
        )


def test_empty_clusterings_are_perfect():
    assert rand_index([], []) == 1.0
    assert homogeneity_completeness_v_measure([], []) == (1.0, 1.0, 1.0)


def test_kendall_tau_b_matches_scipy_with_ties():
    rng = np.random.default_rng(218)
    for _ in range(250):
        sample_count = int(rng.integers(2, 60))
        values_x = rng.integers(0, 10, size=sample_count)
        values_y = rng.integers(0, 10, size=sample_count)

        expected = kendalltau(values_x, values_y).statistic
        actual = kendall_tau_b(values_x, values_y)
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected, abs=1e-12)


def test_kendall_tau_b_all_tied_is_nan():
    assert np.isnan(kendall_tau_b([1, 1, 1], [2, 2, 2]))
