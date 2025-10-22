"""Comparison dispatcher for StructuredModel field comparisons.

This module provides the ComparisonDispatcher class that routes field comparisons
to appropriate handlers based on field type and null states.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class ComparisonDispatcher:
    """Dispatches field comparisons to appropriate handlers based on field type.
    
    This class is responsible for routing field comparisons to the correct
    comparison handler based on:
    - Field type (primitive, list, structured)
    - Null states (None, empty lists, etc.)
    - Hierarchical structure requirements
    
    It uses match-statement based dispatch for clear, traceable logic flow.
    """

    def __init__(self, model: "StructuredModel"):
        """Initialize dispatcher with the ground truth model.
        
        Args:
            model: The ground truth StructuredModel instance
        """
        self.model = model
        
        # Initialize comparators lazily to avoid circular imports
        self._field_comparator = None
        self._primitive_list_comparator = None
        self._structured_list_comparator = None

    @property
    def field_comparator(self):
        """Lazy initialization of FieldComparator."""
        if self._field_comparator is None:
            from .field_comparator import FieldComparator
            self._field_comparator = FieldComparator(self.model)
        return self._field_comparator

    @property
    def primitive_list_comparator(self):
        """Lazy initialization of PrimitiveListComparator."""
        if self._primitive_list_comparator is None:
            from .primitive_list_comparator import PrimitiveListComparator
            self._primitive_list_comparator = PrimitiveListComparator(self.model)
        return self._primitive_list_comparator

    @property
    def structured_list_comparator(self):
        """Lazy initialization of StructuredListComparator."""
        if self._structured_list_comparator is None:
            from .structured_list_comparator import StructuredListComparator
            self._structured_list_comparator = StructuredListComparator(self.model)
        return self._structured_list_comparator

    def dispatch_field_comparison(
        self, 
        field_name: str, 
        gt_val: Any, 
        pred_val: Any
    ) -> Dict[str, Any]:
        """Dispatch field comparison using match-based routing.
        
        This is the core dispatch logic that routes to the appropriate
        comparison handler based on field type and null states.
        
        The dispatch follows this decision tree:
        1. Check if field is a list type → handle list-specific null cases
        2. Check for primitive null cases → handle TN/FA/FN
        3. Route based on value types:
           - Primitive types → FieldComparator
           - List types → PrimitiveListComparator or StructuredListComparator
           - StructuredModel types → FieldComparator
           - Mismatched types → FD result
        
        Args:
            field_name: Name of the field being compared
            gt_val: Ground truth value
            pred_val: Predicted value
            
        Returns:
            Comparison result with structure:
            {
                "overall": {"tp": int, "fa": int, "fd": int, "fp": int, "tn": int, "fn": int},
                "fields": dict,  # Present for hierarchical fields
                "raw_similarity_score": float,
                "similarity_score": float,
                "threshold_applied_score": float,
                "weight": float
            }
        """
        from .structured_model import StructuredModel
        
        # Get field configuration for scoring
        info = self.model._get_comparison_info(field_name)
        weight = info.weight
        threshold = info.threshold

        # Check if this field is ANY list type (including Optional[List[str]], Optional[List[StructuredModel]], etc.)
        is_list_field = self.model._is_list_field(field_name)

        # Get null states and hierarchical needs
        gt_is_null = self.model._is_truly_null(gt_val)
        pred_is_null = self.model._is_truly_null(pred_val)
        gt_needs_hierarchy = self.model._should_use_hierarchical_structure(gt_val, field_name)
        pred_needs_hierarchy = self.model._should_use_hierarchical_structure(
            pred_val, field_name
        )

        # Handle list fields with match statements
        if is_list_field:
            list_result = self.handle_list_field_dispatch(gt_val, pred_val, weight)
            if list_result is not None:
                return list_result
            # If None returned, continue to regular type-based dispatch

        # Handle non-hierarchical primitive null cases with match statements
        if not (gt_needs_hierarchy or pred_needs_hierarchy):
            gt_effectively_null_prim = self.model._is_effectively_null_for_primitives(gt_val)
            pred_effectively_null_prim = self.model._is_effectively_null_for_primitives(
                pred_val
            )

            match (gt_effectively_null_prim, pred_effectively_null_prim):
                case (True, True):
                    return self.model._create_true_negative_result(weight)
                case (True, False):
                    return self.model._create_false_alarm_result(weight)
                case (False, True):
                    return self.model._create_false_negative_result(weight)
                case _:
                    # Both non-null, continue to type-based dispatch
                    pass

        # Type-based dispatch - delegate to appropriate comparator
        if isinstance(gt_val, (str, int, float)) and isinstance(
            pred_val, (str, int, float)
        ):
            # Delegate to FieldComparator for primitive comparison
            return self.field_comparator.compare_primitive_with_scores(gt_val, pred_val, field_name)
        elif isinstance(gt_val, list) and isinstance(pred_val, list):
            # Check if this should be structured list
            if gt_val and isinstance(gt_val[0], StructuredModel):
                return self.model._compare_struct_list_with_scores(
                    gt_val, pred_val, field_name
                )
            else:
                # Delegate to PrimitiveListComparator for primitive list comparison
                return self.primitive_list_comparator.compare_primitive_list_with_scores(
                    gt_val, pred_val, field_name
                )
        elif isinstance(gt_val, list) and len(gt_val) == 0:
            # Handle empty GT list - check if it should be structured
            field_info = self.model.__class__.model_fields.get(field_name)
            if field_info and self.model._is_structured_field_type(field_info):
                # Empty structured list - should still return hierarchical structure
                return self.model._compare_struct_list_with_scores(
                    gt_val, pred_val, field_name
                )
            else:
                # Delegate to PrimitiveListComparator for primitive list comparison
                return self.primitive_list_comparator.compare_primitive_list_with_scores(
                    gt_val, pred_val, field_name
                )
        elif isinstance(pred_val, list) and len(pred_val) == 0:
            # Handle empty pred list - check if it should be structured
            field_info = self.model.__class__.model_fields.get(field_name)
            if field_info and self.model._is_structured_field_type(field_info):
                # Empty structured list - should still return hierarchical structure
                return self.model._compare_struct_list_with_scores(
                    gt_val, pred_val, field_name
                )
            else:
                # Delegate to PrimitiveListComparator for primitive list comparison
                return self.primitive_list_comparator.compare_primitive_list_with_scores(
                    gt_val, pred_val, field_name
                )
        elif isinstance(gt_val, StructuredModel) and isinstance(
            pred_val, StructuredModel
        ):
            # Delegate to FieldComparator for structured field comparison
            return self.field_comparator.compare_structured_field(gt_val, pred_val, field_name, threshold)
        else:
            # Mismatched types
            return {
                "overall": {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "tn": 0, "fn": 0},
                "fields": {},
                "raw_similarity_score": 0.0,
                "similarity_score": 0.0,
                "threshold_applied_score": 0.0,
                "weight": weight,
            }

    def handle_list_field_dispatch(
        self, 
        gt_val: Any, 
        pred_val: Any, 
        weight: float
    ) -> Optional[Dict[str, Any]]:
        """Handle list field comparison with early exit for null cases.
        
        This method handles special cases for list fields:
        - Both None/empty → True Negative
        - GT None/empty, Pred populated → False Alarm
        - GT populated, Pred None/empty → False Negative
        - Both populated → Return None to continue processing
        
        Args:
            gt_val: Ground truth list value (may be None or empty)
            pred_val: Predicted list value (may be None or empty)
            weight: Field weight for scoring
            
        Returns:
            Comparison result dictionary if early exit needed (null cases),
            None if both lists are populated and should continue to type-based dispatch
        """
        gt_effectively_null = self.model._is_effectively_null_for_lists(gt_val)
        pred_effectively_null = self.model._is_effectively_null_for_lists(pred_val)

        match (gt_effectively_null, pred_effectively_null):
            case (True, True):
                # Both None or empty lists → True Negative
                return {
                    "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0},
                    "fields": {},
                    "raw_similarity_score": 1.0,
                    "similarity_score": 1.0,
                    "threshold_applied_score": 1.0,
                    "weight": weight,
                }
            case (True, False):
                # GT=None/empty, Pred=populated list → False Alarm
                pred_list = pred_val if isinstance(pred_val, list) else []
                fa_count = (
                    len(pred_list) if pred_list else 1
                )  # At least 1 FA for the field itself
                return {
                    "overall": {
                        "tp": 0,
                        "fa": fa_count,
                        "fd": 0,
                        "fp": fa_count,
                        "tn": 0,
                        "fn": 0,
                    },
                    "fields": {},
                    "raw_similarity_score": 0.0,
                    "similarity_score": 0.0,
                    "threshold_applied_score": 0.0,
                    "weight": weight,
                }
            case (False, True):
                # GT=populated list, Pred=None/empty → False Negative
                gt_list = gt_val if isinstance(gt_val, list) else []
                fn_count = (
                    len(gt_list) if gt_list else 1
                )  # At least 1 FN for the field itself
                return {
                    "overall": {
                        "tp": 0,
                        "fa": 0,
                        "fd": 0,
                        "fp": 0,
                        "tn": 0,
                        "fn": fn_count,
                    },
                    "fields": {},
                    "raw_similarity_score": 0.0,
                    "similarity_score": 0.0,
                    "threshold_applied_score": 0.0,
                    "weight": weight,
                }
            case _:
                # Both non-null and non-empty, return None to continue processing
                return None
