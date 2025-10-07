"""Utility functions for structured object evaluation.

This module provides helper functions for working with structured models.
"""

from typing import Any, Dict, Type, Union

from .models.structured_model import StructuredModel


def compare_structured_models(
    gt: StructuredModel, pred: StructuredModel
) -> Dict[str, Any]:
    """Compare a ground truth model with a prediction.

    This function wraps the compare_with method of StructuredModel for
    a more explicit API.

    Args:
        gt: Ground truth model
        pred: Prediction model

    Returns:
        Comparison result dictionary
    """
    return gt.compare_with(pred)


def anls_score(
    gt: StructuredModel, pred: StructuredModel, return_field_scores: bool = False
) -> Union[float, Dict[str, Any]]:
    """Calculate ANLS* score for structured models.

    This function provides a simple API for getting an ANLS* score
    from structured models, similar to the regular anls_score function.

    Args:
        gt: Ground truth model
        pred: Prediction model
        return_field_scores: Whether to return detailed field scores

    Returns:
        Either just the overall score (float) or the full comparison results (dict)
    """
    # Use the structured comparison
    result = gt.compare_with(pred)

    # Return either just the score or the detailed results
    if return_field_scores:
        return result
    else:
        return result["overall_score"]


def compare_json(
    gt_json: Dict[str, Any], pred_json: Dict[str, Any], model_cls: Type[StructuredModel]
) -> Dict[str, Any]:
    """Compare JSON objects using a StructuredModel.

    This function is a utility for comparing raw JSON objects using a
    StructuredModel class. It handles missing fields and extra fields gracefully.

    Args:
        gt_json: Ground truth JSON
        pred_json: Prediction JSON
        model_cls: StructuredModel class to use for comparison

    Returns:
        Dictionary with comparison results
    """
    try:
        # Try to convert both JSONs to structured models
        gt_model = model_cls.from_json(gt_json)
        pred_model = model_cls.from_json(pred_json)

        # Compare the models
        return gt_model.compare_with(pred_model)
    except Exception as e:
        # Return error details if conversion fails
        return {
            "error": str(e),
            "overall_score": 0.0,
            "field_scores": {},
            "all_fields_matched": False,
        }
