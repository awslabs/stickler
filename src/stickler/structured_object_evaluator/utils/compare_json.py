from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from typing import Dict, Any, Type


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
