"""
Dedicated class for handling comparison of List[primitive] fields.

This class extracts the primitive list comparison logic from StructuredModel to improve
code organization and maintainability. The extraction preserves existing behavior
exactly, maintaining the hierarchical structure for API consistency.

Design Decision: Universal Hierarchical Structure
=================================================
This comparator returns a hierarchical structure {"overall": {...}, "fields": {...}} even for
primitive lists (List[str], List[int], etc.) to maintain API consistency across all field types.

Why this approach:
- CONSISTENCY: All list fields use the same access pattern: cm["fields"][name]["overall"]
- TEST COMPATIBILITY: Multiple test files expect this pattern for both primitive and structured lists
- PREDICTABLE API: Consumers don't need to check field type before accessing metrics

Trade-offs:
- Creates vestigial "fields": {} objects for primitive lists that will never be populated
- Slightly more verbose structure than necessary for leaf nodes
- Architecturally less pure than type-based structure (primitives flat, structured hierarchical)
"""

from typing import List, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class PrimitiveListComparator:
    """Handles comparison of List[primitive] fields using Hungarian matching."""

    def __init__(self, parent_model: "StructuredModel"):
        """Initialize the comparator with reference to parent model.

        Args:
            parent_model: The StructuredModel instance that owns the list field
        """
        self.parent_model = parent_model

    def compare_primitive_list_with_scores(
        self, gt_list: List[Any], pred_list: List[Any], field_name: str
    ) -> dict:
        """Enhanced primitive list comparison that returns both metrics AND scores.

        This is the main entry point for comparing primitive lists (List[str], List[int], etc.).
        Returns a hierarchical structure for API consistency with structured list comparisons.

        DESIGN DECISION: Universal Hierarchical Structure
        ===============================================
        This method returns a hierarchical structure {"overall": {...}, "fields": {...}} even for
        primitive lists (List[str], List[int], etc.) to maintain API consistency across all field types.

        Why this approach:
        - CONSISTENCY: All list fields use the same access pattern: cm["fields"][name]["overall"]
        - TEST COMPATIBILITY: Multiple test files expect this pattern for both primitive and structured lists
        - PREDICTABLE API: Consumers don't need to check field type before accessing metrics

        Trade-offs:
        - Creates vestigial "fields": {} objects for primitive lists that will never be populated
        - Slightly more verbose structure than necessary for leaf nodes
        - Architecturally less pure than type-based structure (primitives flat, structured hierarchical)

        Alternative considered but rejected:
        - Type-based structure where List[primitive] → flat, List[StructuredModel] → hierarchical
        - Would require updating multiple test files and consumer code to handle mixed access patterns
        - More architecturally pure but breaks backward compatibility

        Future consideration: If we ever refactor the entire confusion matrix API, we could move to
        type-based structure where the presence of "fields" key indicates structured vs primitive.

        Args:
            gt_list: Ground truth list of primitive values
            pred_list: Predicted list of primitive values
            field_name: Name of the list field being compared

        Returns:
            Dictionary with overall metrics, empty fields dict, and scores
            Structure: {
                "overall": {"tp": int, "fa": int, "fd": int, "fp": int, "tn": int, "fn": int},
                "fields": {},  # Always empty for primitive lists
                "raw_similarity_score": float,
                "similarity_score": float,
                "threshold_applied_score": float,
                "weight": float
            }
        """
        # Get field configuration
        info = self.parent_model.__class__._get_comparison_info(field_name)
        weight = info.weight
        threshold = info.threshold

        # CRITICAL FIX: Handle None values before checking length
        # Convert None to empty list for consistent handling
        if gt_list is None:
            gt_list = []
        if pred_list is None:
            pred_list = []

        # Handle empty/null list cases first - FIXED: Empty lists should be TN=1
        if len(gt_list) == 0 and len(pred_list) == 0:
            # Both empty lists should be TN=1
            return {
                "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0},
                "fields": {},  # Empty for primitive lists
                "raw_similarity_score": 1.0,  # Perfect match
                "similarity_score": 1.0,
                "threshold_applied_score": 1.0,
                "weight": weight,
            }
        elif len(gt_list) == 0:
            # GT empty, pred has items → False Alarms
            return {
                "overall": {
                    "tp": 0,
                    "fa": len(pred_list),
                    "fd": 0,
                    "fp": len(pred_list),
                    "tn": 0,
                    "fn": 0,
                },
                "fields": {},
                "raw_similarity_score": 0.0,
                "similarity_score": 0.0,
                "threshold_applied_score": 0.0,
                "weight": weight,
            }
        elif len(pred_list) == 0:
            # GT has items, pred empty → False Negatives
            return {
                "overall": {
                    "tp": 0,
                    "fa": 0,
                    "fd": 0,
                    "fp": 0,
                    "tn": 0,
                    "fn": len(gt_list),
                },
                "fields": {},
                "raw_similarity_score": 0.0,
                "similarity_score": 0.0,
                "threshold_applied_score": 0.0,
                "weight": weight,
            }

        # For primitive lists, use the comparison logic from _compare_unordered_lists
        # which properly handles the threshold-based matching
        comparator = info.comparator
        match_result = self.parent_model._compare_unordered_lists(
            gt_list, pred_list, comparator, threshold
        )

        # Extract the counts from the match result
        tp = match_result.get("tp", 0)
        fd = match_result.get("fd", 0)
        fa = match_result.get("fa", 0)
        fn = match_result.get("fn", 0)

        # Use the overall_score from the match result for raw similarity
        raw_similarity = match_result.get("overall_score", 0.0)

        # CRITICAL FIX: For lists, we NEVER clip under threshold - partial matches are important
        threshold_applied_score = raw_similarity  # Always use raw score for lists

        # Return hierarchical structure expected by tests
        return {
            "overall": {"tp": tp, "fa": fa, "fd": fd, "fp": fa + fd, "tn": 0, "fn": fn},
            "fields": {},  # Empty for primitive lists - no nested structure
            "raw_similarity_score": raw_similarity,
            "similarity_score": raw_similarity,
            "threshold_applied_score": threshold_applied_score,
            "weight": weight,
        }
