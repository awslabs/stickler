"""Non-match collector for StructuredModel comparisons.

This module provides the NonMatchCollector class that handles the collection
and documentation of non-matching fields during structured object comparison.
"""

from typing import List, Dict, Any, TYPE_CHECKING
from .non_matches_helper import NonMatchesHelper
from .non_match_field import NonMatchField, NonMatchType

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class NonMatchCollector:
    """Collects non-matching fields during comparison for detailed analysis.
    
    This class is responsible for collecting and documenting fields that don't
    match between compared StructuredModel instances. It provides two collection
    methods:
    - collect_enhanced_non_matches: Object-level granularity for list fields
    - collect_non_matches: Field-level granularity (legacy format)
    """

    def __init__(self, model: "StructuredModel"):
        """Initialize collector with the ground truth model.
        
        Args:
            model: The ground truth StructuredModel instance
        """
        self.model = model
        self.helper = NonMatchesHelper()

    def collect_enhanced_non_matches(
        self, 
        recursive_result: dict, 
        other: "StructuredModel"
    ) -> List[Dict[str, Any]]:
        """Collect enhanced non-matches with object-level granularity.
        
        This method walks through the recursive comparison result and collects
        non-matches at the object level for list fields, providing more detailed
        information about which specific objects in lists don't match.
        
        Args:
            recursive_result: Result from compare_recursive containing field comparison details
            other: The predicted StructuredModel instance
            
        Returns:
            List of non-match dictionaries with enhanced object-level information
        """
        all_non_matches = []

        # Walk through the recursive result and collect non-matches
        for field_name, field_result in recursive_result.get("fields", {}).items():
            gt_val = getattr(self.model, field_name)
            pred_val = getattr(other, field_name, None)

            # Check if this is a list field that should use object-level collection
            # Import here to avoid circular dependency
            from .structured_model import StructuredModel
            if (
                isinstance(gt_val, list)
                and isinstance(pred_val, list)
                and gt_val
                and isinstance(gt_val[0], StructuredModel)
            ):
                # Use NonMatchesHelper for object-level collection
                object_non_matches = self.helper.collect_list_non_matches(
                    field_name, gt_val, pred_val
                )
                all_non_matches.extend(object_non_matches)

            # Handle null list cases
            elif (
                (gt_val is None or (isinstance(gt_val, list) and len(gt_val) == 0))
                and isinstance(pred_val, list)
                and len(pred_val) > 0
            ):
                # GT empty, pred has items → use helper for FA entries
                null_non_matches = self.helper.add_non_matches_for_null_cases(
                    field_name, gt_val, pred_val
                )
                all_non_matches.extend(null_non_matches)

            elif (
                isinstance(gt_val, list)
                and len(gt_val) > 0
                and (
                    pred_val is None
                    or (isinstance(pred_val, list) and len(pred_val) == 0)
                )
            ):
                # GT has items, pred empty → use helper for FN entries
                null_non_matches = self.helper.add_non_matches_for_null_cases(
                    field_name, gt_val, pred_val
                )
                all_non_matches.extend(null_non_matches)

            else:
                # Use existing field-level logic for non-list fields
                # Extract metrics from field result to determine non-match type
                if isinstance(field_result, dict) and "overall" in field_result:
                    metrics = field_result["overall"]
                elif isinstance(field_result, dict):
                    metrics = field_result
                else:
                    continue  # Skip if we can't extract metrics

                # Create field-level non-match entries based on metrics (legacy format for backward compatibility)
                if metrics.get("fa", 0) > 0:  # False Alarm
                    entry = {
                        "field_path": field_name,
                        "non_match_type": NonMatchType.FALSE_ALARM,  # Use enum value
                        "ground_truth_value": gt_val,
                        "prediction_value": pred_val,
                        "details": {"reason": "unmatched prediction"},
                    }
                    all_non_matches.append(entry)
                elif metrics.get("fn", 0) > 0:  # False Negative
                    entry = {
                        "field_path": field_name,
                        "non_match_type": NonMatchType.FALSE_NEGATIVE,  # Use enum value
                        "ground_truth_value": gt_val,
                        "prediction_value": pred_val,
                        "details": {"reason": "unmatched ground truth"},
                    }
                    all_non_matches.append(entry)
                elif metrics.get("fd", 0) > 0:  # False Discovery
                    similarity = field_result.get("raw_similarity_score")
                    entry = {
                        "field_path": field_name,
                        "non_match_type": NonMatchType.FALSE_DISCOVERY,  # Use enum value
                        "ground_truth_value": gt_val,
                        "prediction_value": pred_val,
                        "similarity_score": similarity,
                        "details": {"reason": "below threshold"},
                    }
                    if similarity is not None:
                        info = self.model._get_comparison_info(field_name)
                        entry["details"]["reason"] = (
                            f"below threshold ({similarity:.3f} < {info.threshold})"
                        )
                    all_non_matches.append(entry)

                # ADDITIONAL: Handle nested StructuredModel objects for detailed non-match collection
                # Import here to avoid circular dependency
                from .structured_model import StructuredModel
                if (
                    isinstance(gt_val, StructuredModel)
                    and isinstance(pred_val, StructuredModel)
                    and "fields" in field_result
                ):
                    # Recursively collect non-matches from nested objects
                    nested_collector = NonMatchCollector(gt_val)
                    nested_non_matches = nested_collector.collect_enhanced_non_matches(
                        field_result, pred_val
                    )
                    # Prefix nested field paths with the parent field name
                    for nested_nm in nested_non_matches:
                        nested_nm["field_path"] = (
                            f"{field_name}.{nested_nm['field_path']}"
                        )
                        all_non_matches.append(nested_nm)

        return all_non_matches

    def collect_non_matches(
        self, 
        other: "StructuredModel", 
        base_path: str = ""
    ) -> List[NonMatchField]:
        """Collect non-matches for detailed analysis (legacy format).
        
        This method performs a field-by-field comparison and collects non-matches
        in the legacy NonMatchField format. It recursively processes nested
        StructuredModel fields.
        
        Args:
            other: Other model to compare with
            base_path: Base path for field naming (e.g., "address")
            
        Returns:
            List of NonMatchField objects documenting non-matches
        """
        non_matches = []

        # Handle null cases
        if other is None:
            non_matches.append(
                NonMatchField(
                    field_path=base_path or "root",
                    non_match_type=NonMatchType.FALSE_NEGATIVE,
                    ground_truth_value=self.model,
                    prediction_value=None,
                )
            )
            return non_matches

        # Compare each field
        for field_name in self.model.__class__.model_fields:
            if field_name == "extra_fields":
                continue

            field_path = f"{base_path}.{field_name}" if base_path else field_name
            gt_value = getattr(self.model, field_name)
            pred_value = getattr(other, field_name, None)

            # Use existing field classification logic
            if isinstance(pred_value, list):
                classification = self.model._calculate_list_confusion_matrix(
                    field_name, pred_value
                )
            else:
                classification = self.model._classify_field_for_confusion_matrix(
                    field_name, pred_value
                )

            # Document non-matches based on classification
            if classification["fa"] > 0:  # False Alarm
                non_matches.append(
                    NonMatchField(
                        field_path=field_path,
                        non_match_type=NonMatchType.FALSE_ALARM,
                        ground_truth_value=gt_value,
                        prediction_value=pred_value,
                        similarity_score=classification.get("similarity_score"),
                    )
                )
            elif classification["fn"] > 0:  # False Negative
                non_matches.append(
                    NonMatchField(
                        field_path=field_path,
                        non_match_type=NonMatchType.FALSE_NEGATIVE,
                        ground_truth_value=gt_value,
                        prediction_value=pred_value,
                    )
                )
            elif classification["fd"] > 0:  # False Discovery
                non_matches.append(
                    NonMatchField(
                        field_path=field_path,
                        non_match_type=NonMatchType.FALSE_DISCOVERY,
                        ground_truth_value=gt_value,
                        prediction_value=pred_value,
                        similarity_score=classification.get("similarity_score"),
                    )
                )

            # Handle nested models recursively
            # Import here to avoid circular dependency
            from .structured_model import StructuredModel
            if isinstance(gt_value, StructuredModel) and isinstance(
                pred_value, StructuredModel
            ):
                nested_collector = NonMatchCollector(gt_value)
                nested_non_matches = nested_collector.collect_non_matches(
                    pred_value, field_path
                )
                non_matches.extend(nested_non_matches)

        return non_matches
