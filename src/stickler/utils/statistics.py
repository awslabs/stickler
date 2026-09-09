"""Small statistical primitives used by Stickler metrics."""

from math import sqrt
from typing import Any, Sequence

import numpy as np


def binary_roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Return binary ROC AUC using average ranks for tied scores."""
    labels_array = np.asarray(labels, dtype=bool)
    scores_array = np.asarray(scores, dtype=float)
    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if labels_array.size != scores_array.size:
        raise ValueError("labels and scores must have the same length")

    positive_count = int(labels_array.sum())
    negative_count = labels_array.size - positive_count
    if not positive_count or not negative_count:
        raise ValueError("ROC AUC requires both positive and negative labels")

    order = np.argsort(scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    ranks = np.empty(labels_array.size, dtype=float)
    start = 0
    while start < labels_array.size:
        end = start + 1
        while end < labels_array.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based. The tied group spans start + 1 through end.
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end

    positive_rank_sum = float(ranks[labels_array].sum())
    mann_whitney_u = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return mann_whitney_u / (positive_count * negative_count)


def _contingency_matrix(
    labels_true: Sequence[Any], labels_pred: Sequence[Any]
) -> np.ndarray:
    true_array = np.asarray(labels_true)
    pred_array = np.asarray(labels_pred)
    if true_array.ndim != 1 or pred_array.ndim != 1:
        raise ValueError("clustering labels must be one-dimensional")
    if true_array.size != pred_array.size:
        raise ValueError("clustering label arrays must have the same length")
    if not true_array.size:
        return np.zeros((0, 0), dtype=np.int64)

    _, true_inverse = np.unique(true_array, return_inverse=True)
    _, pred_inverse = np.unique(pred_array, return_inverse=True)
    contingency = np.zeros(
        (int(true_inverse.max()) + 1, int(pred_inverse.max()) + 1),
        dtype=np.int64,
    )
    np.add.at(contingency, (true_inverse, pred_inverse), 1)
    return contingency


def rand_index(labels_true: Sequence[Any], labels_pred: Sequence[Any]) -> float:
    """Return the Rand index for two cluster assignments."""
    contingency = _contingency_matrix(labels_true, labels_pred)
    sample_count = int(contingency.sum())
    pair_count = sample_count * (sample_count - 1) // 2
    if not pair_count:
        return 1.0

    true_counts = contingency.sum(axis=1)
    pred_counts = contingency.sum(axis=0)
    true_positive = int(((contingency * (contingency - 1)) // 2).sum())
    same_true = int(((true_counts * (true_counts - 1)) // 2).sum())
    same_pred = int(((pred_counts * (pred_counts - 1)) // 2).sum())
    true_negative = pair_count - same_true - same_pred + true_positive
    return (true_positive + true_negative) / pair_count


def homogeneity_completeness_v_measure(
    labels_true: Sequence[Any], labels_pred: Sequence[Any]
) -> tuple[float, float, float]:
    """Return homogeneity, completeness and their harmonic mean."""
    contingency = _contingency_matrix(labels_true, labels_pred).astype(float)
    sample_count = float(contingency.sum())
    if not sample_count:
        return 1.0, 1.0, 1.0

    true_counts = contingency.sum(axis=1)
    pred_counts = contingency.sum(axis=0)
    nonzero = contingency > 0
    expected = np.outer(true_counts, pred_counts)
    mutual_information = float(
        (
            (contingency[nonzero] / sample_count)
            * np.log(
                contingency[nonzero]
                * sample_count
                / expected[nonzero]
            )
        ).sum()
    )

    def entropy(counts: np.ndarray) -> float:
        probabilities = counts[counts > 0] / sample_count
        return float(-(probabilities * np.log(probabilities)).sum())

    true_entropy = entropy(true_counts)
    pred_entropy = entropy(pred_counts)
    homogeneity = 1.0 if true_entropy == 0 else mutual_information / true_entropy
    completeness = 1.0 if pred_entropy == 0 else mutual_information / pred_entropy
    if homogeneity + completeness == 0:
        v_measure = 0.0
    else:
        v_measure = 2 * homogeneity * completeness / (homogeneity + completeness)
    return homogeneity, completeness, v_measure


def kendall_tau_b(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    """Return Kendall's tau-b, including correction for ties."""
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Kendall inputs must be one-dimensional")
    if x.size != y.size:
        raise ValueError("Kendall inputs must have the same length")

    numerator = 0.0
    untied_x = 0
    untied_y = 0
    for index in range(x.size - 1):
        x_sign = np.sign(x[index + 1 :] - x[index])
        y_sign = np.sign(y[index + 1 :] - y[index])
        numerator += float(np.dot(x_sign, y_sign))
        untied_x += int(np.count_nonzero(x_sign))
        untied_y += int(np.count_nonzero(y_sign))

    denominator = sqrt(untied_x * untied_y)
    return numerator / denominator if denominator else float("nan")
