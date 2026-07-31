"""Derived-metric scoring formulas for StructuredModel comparisons."""

from typing import Any, Dict


def calculate_derived_metrics(
    metrics: Dict[str, int], recall_with_fd: bool = False
) -> Dict[str, float]:
    """Calculate derived metrics from confusion matrix counts.

    Args:
        metrics: Dictionary with TP, FP, TN, FN, FD counts
        recall_with_fd: If True, include FD in recall denominator (TP/(TP+FN+FD))
                        If False, use traditional recall (TP/(TP+FN))

    Returns:
        Dictionary with precision, recall, F1, and accuracy
    """
    tp = metrics["tp"]
    fp = metrics["fp"]
    tn = metrics["tn"]
    fn = metrics["fn"]
    fd = metrics["fd"]
    fa = metrics["fa"]

    # Calculate precision: TP / (TP + FP) where FP includes both FA and FD
    # Note: fp field should already equal fa + fd from individual classifications
    total_fp = fa + fd  # Total False Positives = False Alarms + False Discoveries
    precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0

    # Calculate recall based on the selected formula
    if recall_with_fd:
        # Alternative recall: TP / (TP + FN + FD)
        recall = tp / (tp + fn + fd) if (tp + fn + fd) > 0 else 0.0
    else:
        # Traditional recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Calculate F1 score
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "cm_precision": precision,
        "cm_recall": recall,
        "cm_f1": f1,
        "cm_accuracy": accuracy,
    }


def convert_score_to_binary_metrics(
    score: float, threshold: float = 0.5
) -> Dict[str, Any]:
    """Convert similarity score to binary classification metrics.

    Args:
        score: Similarity score [0-1]
        threshold: Threshold for considering a match

    Returns:
        Dictionary with TP, FP, FN, TN counts converted to metrics
    """
    # For single field comparison: if score >= threshold, it's TP, otherwise FP/FN
    if score >= threshold:
        tp = score  # Proportional TP credit
        fp = 1 - score if score < 1.0 else 0  # Small FP for imperfect matches
        fn = 0
        tn = 0
    else:
        tp = 0
        fp = score  # Partial FP credit for some similarity
        fn = 1 - score  # Higher FN for very different values
        tn = 0

    # Calculate derived metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "anls_score": score,
    }
