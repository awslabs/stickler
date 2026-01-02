"""
Confidence metrics calculator for StructuredModel comparison evaluation.

This module provides the ConfidenceCalculator class for evaluating how well
prediction confidence scores correlate with actual field match results in
StructuredModel comparisons. The primary metric calculated is AUROC (Area
Under ROC Curve), which measures confidence calibration quality.

Usage:
------
The ConfidenceCalculator integrates with StructuredModel.compare_with() to provide
confidence evaluation metrics alongside standard comparison results.

Example:
    >>> gt = Product(name="Widget", price=29.99)
    >>> pred = Product.from_json({
    ...     "name": {"value": "Widget", "confidence": 0.95},
    ...     "price": {"value": 30.00, "confidence": 0.7}
    ... })
    >>> result = gt.compare_with(pred, add_confidence_metrics=True, document_field_comparisons=True)
    >>> auroc = result['auroc_confidence_metric']
    >>> print(f"Confidence calibration AUROC: {auroc}")
"""
from typing import Dict
from sklearn.metrics import roc_auc_score

from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class ConfidenceCalculator:
    """Calculate confidence-based metrics like AUROC for structured model comparisons."""
    
    def calculate_overall_auroc(self, comparison_result: Dict, pred_instance: StructuredModel) -> float:
        """Calculate AUROC using field_comparisons for exact path matching."""
        
        y_true = []
        y_scores = []
        
        pred_confidences = pred_instance.get_all_confidences()
        field_comparisons = comparison_result.get('field_comparisons', [])
        print(field_comparisons)
        for field_comparison in field_comparisons:
            field_path = field_comparison['actual_key']
            is_match = field_comparison['match'] 
            confidence = pred_confidences.get(field_path)

            
            if confidence is not None:
                y_true.append(1 if is_match else 0)
                y_scores.append(confidence)

        if len(y_true) == 0 or len(set(y_true)) < 2:
            return 0.5
      
        return roc_auc_score(y_true, y_scores)
