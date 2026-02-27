# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Document Packet Evaluation Metrics.

Provides clustering + ordering metrics for document packet splitting tasks,
as proposed in the DocSplit paper (arXiv:2602.15958).

Metrics:
- V-measure: Harmonic mean of homogeneity and completeness for clustering
- Rand Index: Pairwise clustering similarity
- Kendall's Tau: Page ordering correlation per document group
- Combined Packet Score: α * S_clustering + β * S_ordering

Ported from the IDP accelerator's packet_evaluation_metrics.py with:
- matplotlib/visualization functions removed
- Flexible input: accepts List[Dict], CSV path, or pd.DataFrame

Reference: "DocSplit: A Comprehensive Benchmark Dataset and Evaluation
Approach for Document Packet Recognition and Splitting" (arXiv:2602.15958)
"""

import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import (
    homogeneity_completeness_v_measure,
    rand_score,
)

logger = logging.getLogger(__name__)


def _to_dataframe(data: Union[List[Dict[str, Any]], str, pd.DataFrame]) -> pd.DataFrame:
    """
    Normalize input to a pandas DataFrame.

    Accepts:
        - List[Dict]: list of page dicts (JSON-style)
        - str: path to a CSV file
        - pd.DataFrame: pass-through

    Returns:
        pd.DataFrame with page-level evaluation data
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, str):
        return pd.read_csv(data)
    raise TypeError(
        f"data must be List[Dict], CSV path (str), or pd.DataFrame, got {type(data).__name__}"
    )


def create_composite_group_ids(
    data: pd.DataFrame, strict_clustering: bool = False
) -> Tuple[pd.Series, pd.Series]:
    """
    Prepare group identifiers for clustering evaluation.

    In strict mode, misclassified pages get unique error IDs so they
    receive no clustering credit.

    Args:
        data: DataFrame with group_id, group_id_predicted columns.
            For strict mode, also needs class_label and class_label_predicted.
        strict_clustering: If True, penalize misclassification in clustering.

    Returns:
        (group_id_for_eval, group_id_predicted_for_eval)
    """
    if strict_clustering:
        required = ["class_label", "class_label_predicted"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise KeyError(f"Strict clustering requires columns: {missing}")

        group_id_predicted_eval = data["group_id_predicted"].copy()
        misclassified = data["class_label"] != data["class_label_predicted"]

        if misclassified.any():
            error_ids = "ERROR_ROW_" + data.index[misclassified].astype(str)
            group_id_predicted_eval.loc[misclassified] = error_ids

        return data["group_id"], group_id_predicted_eval

    return data["group_id"], data["group_id_predicted"]


def calculate_clustering_score(
    data: pd.DataFrame,
    v_measure_weight: float = 0.5,
    strict_clustering: bool = False,
) -> Tuple[float, float, float]:
    """
    Calculate clustering score as weighted V-measure + Rand Index.

    Formula: S_clustering = w * V-measure + (1 - w) * Rand Index

    Args:
        data: DataFrame with group_id and group_id_predicted columns.
        v_measure_weight: Weight for V-measure in [0, 1].
        strict_clustering: If True, penalize misclassification.

    Returns:
        (clustering_score, v_measure, rand_index)
    """
    required = ["group_id", "group_id_predicted"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if not 0 <= v_measure_weight <= 1:
        raise ValueError(f"v_measure_weight must be in [0, 1], got {v_measure_weight}")

    gt_ids, pred_ids = create_composite_group_ids(data, strict_clustering)

    _h, _c, v_measure = homogeneity_completeness_v_measure(gt_ids, pred_ids)
    ri = rand_score(gt_ids, pred_ids)

    clustering_score = v_measure_weight * v_measure + (1 - v_measure_weight) * ri
    return clustering_score, v_measure, ri


def calculate_ordering_score_per_group(
    data: pd.DataFrame,
) -> Dict[Any, float]:
    """
    Calculate Kendall's Tau for each document group.

    Single-page groups are excluded (ordering undefined).

    Args:
        data: DataFrame with group_id, page_number, page_number_predicted.

    Returns:
        Dict mapping group_id to Kendall's Tau score.
    """
    required = ["group_id", "page_number", "page_number_predicted"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    group_scores: Dict[Any, float] = {}

    for group_id, group_data in data.groupby("group_id"):
        if len(group_data) <= 1:
            continue

        tau, _p_value = kendalltau(
            group_data["page_number"], group_data["page_number_predicted"]
        )
        group_scores[group_id] = tau if not np.isnan(tau) else 0

    return group_scores


def calculate_average_ordering_score(group_scores: Dict[Any, float]) -> float:
    """
    Mean Kendall's Tau across all multi-page groups.

    Returns 0 if no multi-page groups exist.
    """
    if not group_scores:
        return 0
    return sum(group_scores.values()) / len(group_scores)


def calculate_final_score(
    clustering_score: float,
    v_measure: float,
    ordering_score: float,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Combined packet score: S_packet = α * S_clustering + β * S_ordering.

    With α + β = 1, range is [-0.5, 1.0] (1.0 = perfect).

    Args:
        clustering_score: Weighted V-measure + Rand Index.
        v_measure: V-measure (included for API compatibility, not used in formula).
        ordering_score: Average Kendall's Tau.
        alpha: Weight for clustering.
        beta: Weight for ordering.

    Returns:
        Combined packet score.
    """
    return alpha * clustering_score + beta * ordering_score


def evaluate_packet(
    data: Union[List[Dict[str, Any]], str, pd.DataFrame],
    v_measure_weight: float = 0.5,
    alpha: float = 0.5,
    beta: float = 0.5,
    strict_clustering: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate document packet splitting performance.

    Accepts flexible input (list of dicts, CSV path, or DataFrame) and
    computes clustering + ordering metrics.

    Required columns:
        - group_id: Ground truth document group
        - group_id_predicted: Predicted document group
        - page_number: Ground truth page order within document
        - page_number_predicted: Predicted page order

    For strict_clustering=True, also requires:
        - class_label: Ground truth document type
        - class_label_predicted: Predicted document type

    Args:
        data: Page-level evaluation data.
        v_measure_weight: Weight for V-measure in clustering score.
        alpha: Weight for clustering in final score.
        beta: Weight for ordering in final score.
        strict_clustering: If True, penalize misclassification in clustering.

    Returns:
        Dict with final_score, clustering_score, v_measure, rand_index,
        avg_ordering_score, group_ordering_scores, strict_clustering.
    """
    df = _to_dataframe(data)

    clustering_score, v_measure, ri = calculate_clustering_score(
        df, v_measure_weight, strict_clustering
    )

    group_scores = calculate_ordering_score_per_group(df)
    avg_ordering = calculate_average_ordering_score(group_scores)

    final = calculate_final_score(
        clustering_score, v_measure, avg_ordering, alpha, beta
    )

    return {
        "final_score": final,
        "clustering_score": clustering_score,
        "v_measure": v_measure,
        "rand_index": ri,
        "avg_ordering_score": avg_ordering,
        "group_ordering_scores": group_scores,
        "strict_clustering": strict_clustering,
    }
