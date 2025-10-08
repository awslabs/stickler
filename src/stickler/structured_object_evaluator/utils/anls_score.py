"""ANLS score calculation for structured objects."""

from typing import Any, Dict, List, Tuple, Type, Union, Optional

from ..models.structured_model import StructuredModel


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
    gt: Any, pred: Any, return_gt: bool = False, return_key_scores: bool = False
) -> Union[float, Tuple[float, Any], Tuple[float, Any, Dict[str, Any]]]:
    """Calculate ANLS* score between two objects.

    This function provides a simple API for getting an ANLS* score
    between any two objects, similar to the original anls_score function.

    Args:
        gt: Ground truth object
        pred: Prediction object
        return_gt: Whether to return the closest ground truth
        return_key_scores: Whether to return detailed key scores

    Returns:
        Either just the overall score (float), or a tuple with the score and
        closest ground truth, or a tuple with the score, closest ground truth,
        and key scores.
    """
    import warnings
    from ..trees.base import ANLSTree

    # Store original gt object for possible return
    original_gt = gt

    # Handle classical QA dataset compatibility
    gt_is_list_str = isinstance(gt, list) and all(isinstance(x, str) for x in gt)
    pred_is_str = isinstance(pred, str)
    if gt_is_list_str and pred_is_str:
        warnings.warn(
            "Treating ground truth as a list of options. This is a compatibility mode for ST-VQA-like datasets."
        )
        gt = tuple(gt)

    # Create trees from the objects
    gt_tree = ANLSTree.make_tree(gt, is_gt=True)
    pred_tree = ANLSTree.make_tree(pred, is_gt=False)

    # Calculate ANLS score
    score, closest_gt, key_scores = gt_tree.anls(pred_tree)

    # Determine what to return for gt (smart detection)
    gt_to_return = original_gt if hasattr(original_gt, "model_dump") else closest_gt

    # Return the requested information
    if return_gt and return_key_scores:
        from .key_scores import construct_nested_dict

        key_scores_dict = construct_nested_dict(key_scores)
        return score, gt_to_return, key_scores_dict
    elif return_gt:
        return score, gt_to_return
    elif return_key_scores:
        from .key_scores import construct_nested_dict

        key_scores_dict = construct_nested_dict(key_scores)
        return score, key_scores_dict
    else:
        return score


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
