"""
Dedicated class for handling Hungarian matching of List[StructuredModel] fields.

Encapsulates the comparison logic for a list whose elements are themselves
StructuredModel instances. It runs Hungarian matching to pair items between
the GT and predicted lists, then produces two parallel views of the result:
a threshold-gated view ("overall") and an ungated view ("aggregate") used
for drill-down field-level metrics.

Behavior summary:

- Object-level threshold: the element model's own ``match_threshold``
  (``StructuredModel.match_threshold`` of the list's item type) is used to
  classify each Hungarian-matched pair. Pairs at or above threshold are
  TP; matched pairs below threshold are FD; unmatched GT/Pred items are
  FN/FA.
- Below-threshold pairs are FD, not unmatched: a pair the Hungarian
  algorithm assigns counts as a match. The threshold splits matched pairs
  into TP (>= threshold) and FD (< threshold); it does not un-match them.
  For the general multi-item matching this holds regardless of similarity
  magnitude — a pair at similarity 0.0 is still an assigned match and
  therefore FD, not FN+FA. (Exception: the len==1-vs-len==1 fast path in
  ``HungarianMatcher.calculate_metrics`` drops a zero-similarity pair, so a
  single-item list yields FN+FA at 0.0. This predates and is independent of
  this comparator.) Whether FD counts against recall is controlled by the
  ``recall_with_fd`` knob.
- Per-field ``overall`` metrics are threshold-gated at every nesting
  level: only TP pairs and unmatched FN/FA items contribute. FD pairs are
  excluded so per-field ``overall`` reflects the same threshold decision
  as the object level. Nested ``List[StructuredModel]`` fields apply
  the same gating recursively using the inner model's ``match_threshold``.
- Per-field ``aggregate`` metrics recurse through every matched pair
  (TP and FD alike) and every unmatched item (FN and FA), providing a
  complete drill-down view independent of the threshold gate.
- Empty-list cases: empty-vs-empty yields a single list-level TN with no
  nested field metrics.
- Nested field metrics are computed by delegating into each child model's
  ``compare_recursive`` rather than re-implementing the recursion here.
"""

from typing import TYPE_CHECKING, Any, Dict, List

from .comparable_field import ComparableField
from .comparison_helper import ComparisonHelper
from .hungarian_helper import HungarianHelper

if TYPE_CHECKING:
    from .structured_model import StructuredModel


class StructuredListComparator:
    """Handles comparison of List[StructuredModel] fields using Hungarian matching."""

    def __init__(self, parent_model: "StructuredModel"):
        """Initialize the comparator with reference to parent model.

        Args:
            parent_model: The StructuredModel instance that owns the list field
        """
        self.parent_model = parent_model

    def compare_struct_list_with_scores(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        field_name: str,
    ) -> dict:
        """Enhanced structural list comparison that returns both metrics AND scores.

        CRITICAL: This is the main entry point extracted from StructuredModel.
        Maintains identical behavior including current bugs for Phase 2 compatibility.

        Args:
            gt_list: Ground truth list of StructuredModel objects
            pred_list: Predicted list of StructuredModel objects
            field_name: Name of the list field being compared

        Returns:
            Dictionary with overall metrics, nested field details, and scores
        """
        # Get field configuration - same as original
        info = self.parent_model.__class__._get_comparison_info(field_name)
        weight = info.weight

        # PHASE 3 FIX: Use correct threshold source for Hungarian matching decisions
        # Should use the list element model's match_threshold, not the parent field's threshold
        if gt_list and hasattr(gt_list[0].__class__, "match_threshold"):
            match_threshold = gt_list[0].__class__.match_threshold
        else:
            # Fallback to default if no match_threshold defined
            match_threshold = getattr(
                self.parent_model.__class__, "match_threshold", 0.7
            )

        # Handle empty list cases with beautiful match statements
        early_exit_result = self._handle_struct_list_empty_cases(
            gt_list, pred_list, weight, field_name
        )
        if early_exit_result is not None:
            return early_exit_result

        # Normalize None to empty lists for consistent processing below
        gt_list = gt_list or []
        pred_list = pred_list or []

        # Calculate object-level metrics using extracted method
        (
            object_level_metrics,
            matched_pairs,
            matched_gt_indices,
            matched_pred_indices,
        ) = self._calculate_object_level_metrics(gt_list, pred_list, match_threshold)

        # Calculate raw similarity score using extracted method
        raw_similarity = self._calculate_struct_list_similarity(
            matched_pairs, gt_list, pred_list, info
        )

        # CRITICAL FIX: For structured lists, we NEVER clip under threshold - partial matches are important
        threshold_applied_score = raw_similarity  # Always use raw score for lists

        # Get field-level details for nested structure (but DON'T aggregate to list level)
        # THRESHOLD-GATED RECURSION: Only generate field details for good matches
        field_details = self._calculate_nested_field_metrics(
            field_name,
            gt_list,
            pred_list,
            matched_pairs,
            matched_gt_indices,
            matched_pred_indices,
            match_threshold,
        )

        # Build final result structure
        final_result = {
            "overall": object_level_metrics,  # Count OBJECTS, not fields
            "fields": field_details,  # Field-level details kept separate
            "raw_similarity_score": raw_similarity,
            "similarity_score": raw_similarity,
            "threshold_applied_score": threshold_applied_score,
            "weight": weight,
        }
        return final_result

    def _handle_struct_list_empty_cases(
            self,
            gt_list: List["StructuredModel"],
            pred_list: List["StructuredModel"],
            weight: float,
            field_name: str,
        ) -> dict:
            """Handle empty list cases with beautiful match statements.

            Args:
                gt_list: Ground truth list (may be None)
                pred_list: Predicted list (may be None)
                weight: Field weight for scoring

            Returns:
                Result dictionary if early exit needed, None if should continue processing
            """
            # Normalize None to empty lists for consistent handling
            gt_len = len(gt_list or [])
            pred_len = len(pred_list or [])

            match (gt_len, pred_len):
                case (0, 0):
                    # Both empty lists → True Negative at list level.
                    # No nested field metrics are emitted (matches dev behavior).
                    return {
                        "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 1, "fn": 0},
                        "fields": {},
                        "raw_similarity_score": 1.0,
                        "similarity_score": 1.0,
                        "threshold_applied_score": 1.0,
                        "weight": weight,
                    }

                case _:
                    # Both non-empty, continue processing
                    return None
                
    def _calculate_object_level_metrics(
        self,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        match_threshold: float,
    ) -> tuple:
        """Calculate object-level metrics using Hungarian matching.

        Args:
            gt_list: Ground truth list
            pred_list: Predicted list
            match_threshold: Threshold for considering objects as matches

        Returns:
            Tuple of (object_metrics_dict, matched_pairs, matched_gt_indices, matched_pred_indices)
        """
        # Use Hungarian matching for OBJECT-LEVEL counts
        hungarian_helper = HungarianHelper()
        hungarian_info = hungarian_helper.get_complete_matching_info(gt_list, pred_list)
        matched_pairs = hungarian_info["matched_pairs"]

        # Count OBJECTS, not individual fields.
        # A Hungarian-matched pair below match_threshold is FD regardless of
        # whether its similarity is exactly 0.0. The match/no-match decision is
        # binary (the pair exists in the Hungarian assignment), and the
        # threshold gate then splits matched pairs into TP vs FD. The
        # recall_with_fd knob controls whether FD counts against recall.
        tp_objects = 0  # Objects with similarity >= match_threshold
        fd_objects = 0  # Objects with similarity < match_threshold
        for gt_idx, pred_idx, similarity in matched_pairs:
            if similarity >= match_threshold:
                tp_objects += 1
            else:
                fd_objects += 1

        # Count unmatched objects
        matched_gt_indices = {idx for idx, _, _ in matched_pairs}
        matched_pred_indices = {idx for _, idx, _ in matched_pairs}
        fn_objects = len(gt_list) - len(matched_gt_indices)  # Unmatched GT objects
        fa_objects = len(pred_list) - len(matched_pred_indices)  # Unmatched pred objects

        # Build list-level metrics counting OBJECTS (not fields)
        object_level_metrics = {
            "tp": tp_objects,
            "fa": fa_objects,
            "fd": fd_objects,
            "fp": fa_objects + fd_objects,  # Total false positives
            "tn": 0,  # No true negatives at object level for non-empty lists
            "fn": fn_objects,
        }

        return (
            object_level_metrics,
            matched_pairs,
            matched_gt_indices,
            matched_pred_indices,
        )

    def _calculate_struct_list_similarity(
        self,
        matched_pairs: List[Any],
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        info: "ComparableField",
    ) -> float:
        """Calculate raw similarity score for structured list.

        Args:
            gt_list: Ground truth list
            pred_list: Predicted list
            info: Field comparison info

        Returns:
            Raw similarity score between 0.0 and 1.0
        """
        # Updated code to not use helper that was calling Hungarian match again, and instead use already generated matched pairs
        threshold_corrected_pairs = []
        for gt_idx, pred_idx, raw_score in matched_pairs:
            if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                gt_item = gt_list[gt_idx]
                pred_item = pred_list[pred_idx]

                # Use individual comparison with threshold application (same as .compare_with())
                individual_result = gt_item.compare_with(pred_item)
                threshold_applied_score = individual_result["overall_score"]

                threshold_corrected_pairs.append(
                    (gt_idx, pred_idx, threshold_applied_score)
                )
            else:
                threshold_corrected_pairs.append((gt_idx, pred_idx, raw_score))
        
        classification_threshold = (
                0.01  # Almost everything that's not 0.0 should be TP
            )
        
        match_result = ComparisonHelper.unordered_list_metrics(
            threshold_corrected_pairs, gt_list, pred_list, classification_threshold
        )

        return match_result.get("overall_score", 0.0)

    def _calculate_nested_field_metrics(
        self,
        list_field_name: str,
        gt_list: List["StructuredModel"],
        pred_list: List["StructuredModel"],
        matched_pairs: List,
        matched_gt_indices: set,
        matched_pred_indices: set,
        match_threshold: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate field-level details for nested structure with threshold-gated recursion.

        Implements proper threshold-gated recursion as documented: the same
        threshold-gating applies recursively at each nesting level, using the
        inner model's ``match_threshold``.

        Only generates nested field metrics in per-field 'overall' for object pairs
        with similarity >= match_threshold. Below-threshold pairs still contribute to
        'aggregate' metrics (which recurse into all leaf nodes regardless of threshold).

        Args:
            list_field_name: Name of the parent list field
            gt_list: Ground truth list
            pred_list: Predicted list
            matched_pairs: List of (gt_idx, pred_idx, similarity) tuples
            matched_gt_indices: Set of matched GT indices
            matched_pred_indices: Set of matched pred indices
            match_threshold: Match threshold for threshold-gating

        Returns:
            Dictionary mapping field names to their metrics
        """
        # Two result sets:
        # - gated_results: only above-threshold pairs + unmatched items (for per-field "overall")
        # - all_results: ALL pairs including below-threshold (for "aggregate" pre-seeding)
        all_results = []
        gated_results = []

        # Handle matched pairs - split by threshold.
        # Every Hungarian-matched pair recurses for aggregate purposes; the
        # threshold only gates whether the pair also contributes to per-field
        # "overall". This mirrors the object-level treatment where a matched
        # pair below threshold is FD (not unmatched), regardless of similarity.
        for gt_idx, pred_idx, similarity in matched_pairs:
            if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                gt_item = gt_list[gt_idx]
                pred_item = pred_list[pred_idx]
                field_details_tmp = gt_item.compare_recursive(pred_item)
                all_results.append(field_details_tmp)
                if similarity >= match_threshold:
                    gated_results.append(field_details_tmp)

        # Handle unmatched GT objects - count each element in the list
        for gt_idx, gt_item in enumerate(gt_list):
            if gt_idx not in matched_gt_indices:
                #compare against itself to count all non-null values
                field_details_tmp = gt_item.compare_recursive(gt_item)

                #take all the tp values and convert to fn
                field_details_tmp["fields"] = self._switch_metrics(field_details_tmp["fields"] , source_metric='tp', target_metric='fn')
                all_results.append(field_details_tmp)
                gated_results.append(field_details_tmp)

        # Handle unmatched pred objects - count each element in the list
        for pred_idx, pred_item in enumerate(pred_list):
            if pred_idx not in matched_pred_indices:
                #compare against itself to count all non-null values
                field_details_tmp = pred_item.compare_recursive(pred_item)

                #take all the tp values and convert to fa and fp
                target_result = field_details_tmp.copy()
                target_result["fields"]  = self._switch_metrics(field_details_tmp["fields"], source_metric='tp', target_metric='fa')
                field_details_tmp["fields"] = self._switch_metrics(field_details_tmp["fields"], target_result["fields"], source_metric='tp', target_metric='fp')
                all_results.append(field_details_tmp)
                gated_results.append(field_details_tmp)

        # Aggregate gated results for per-field "overall" values
        gated_aggregated = self._recursive_aggregate_metrics(gated_results)

        # Aggregate all results for pre-seeded "aggregate" values
        all_aggregated = self._recursive_aggregate_metrics(all_results)

        # Merge: use gated structure for "overall" but pre-seed "aggregate" from full results
        merged = self._preseed_aggregate_from_full(gated_aggregated, all_aggregated)

        return merged['fields']

    def _switch_metrics(self, source_result:dict, target_result: dict =None, source_metric: str ='tp', target_metric: str ='fp'):
        if not target_result:
            target_result={}

        # RECURSIVE CALL: Handle nested fields at arbitrary depth
        for field_name, field_metrics in source_result.items():
            if field_name not in target_result:
                target_result[field_name] = {}

            if "overall" in field_metrics:
                if "overall" not in target_result[field_name]:
                    target_result[field_name]["overall"] = {}

                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                    if metric not in target_result[field_name]["overall"]:
                        target_result[field_name]["overall"][metric] = 0

                target_result[field_name]["overall"][target_metric] += field_metrics["overall"].get(source_metric, 0)

            if "fields" in field_metrics:
                if "fields" not in target_result[field_name]: 
                    target_result[field_name]["fields"] = {}
                
                target_result[field_name]["fields"] = self._switch_metrics(field_metrics["fields"],
                                                                           target_result[field_name]["fields"],
                                                                           source_metric, target_metric)
        return target_result

    def _preseed_aggregate_from_full(
        self, gated: Dict[str, Any], full: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Pre-seed 'aggregate' keys into the gated structure using metrics from the full (ungated) structure.

        Only pre-seeds at LEAF nodes (fields with no children or empty children).
        Non-leaf nodes let the AggregateMetricsCalculator compute by summing child aggregates.

        The gated structure has per-field 'overall' counts that only include above-threshold
        entity pairs. The full structure includes ALL pairs. We copy the full structure's
        ungated counts into 'aggregate' keys at leaf levels so that the
        AggregateMetricsCalculator picks them up as pre-computed aggregates and sums them upward.

        When a leaf node in the full structure already has an 'aggregate' key (from an inner
        list comparison that also applied gating), we use that pre-computed aggregate rather
        than the gated 'overall', since it represents the true ungated counts.

        Args:
            gated: Aggregated result from above-threshold pairs only (used for 'overall')
            full: Aggregated result from ALL pairs (used for 'aggregate' pre-seeding)

        Returns:
            The gated structure with 'aggregate' keys pre-seeded at leaf nodes from full.
        """
        # Determine if this is a leaf node (no children or empty children)
        full_has_children = "fields" in full and full["fields"] and len(full["fields"]) > 0

        if not full_has_children:
            # Leaf node: pre-seed aggregate from full's aggregate (if present) or overall
            if "aggregate" in full:
                gated["aggregate"] = dict(full["aggregate"])
            elif "overall" in full:
                gated["aggregate"] = dict(full["overall"])
        else:
            # Non-leaf node: recurse into children but do NOT pre-seed aggregate here.
            # The AggregateMetricsCalculator will compute it by summing child aggregates.
            if "fields" not in gated:
                gated["fields"] = {}

            for field_name in full["fields"]:
                if field_name in gated.get("fields", {}):
                    gated["fields"][field_name] = self._preseed_aggregate_from_full(
                        gated["fields"][field_name], full["fields"][field_name]
                    )
                else:
                    # Field exists in full but not in gated (all from below-threshold pairs)
                    # Create a shell with zeroed overall and recurse
                    full_field = full["fields"][field_name]
                    shell = {
                        "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},
                        "fields": {},
                    }
                    # Recursively pre-seed into the shell
                    shell = self._preseed_aggregate_from_full(shell, full_field)
                    gated["fields"][field_name] = shell

        return gated

    def _recursive_aggregate_metrics(
        self, pair_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Recursively aggregate metrics from multiple pair results - handles arbitrary depth."""
        if not pair_results:
            return {
                "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},
                "fields": {},
            }

        # Initialize the aggregated result
        aggregated = {
            "overall": {"tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0},
            "fields": {},
        }

        for pair_result in pair_results:
            # Aggregate overall metrics
            if "overall" in pair_result:
                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                    aggregated["overall"][metric] += pair_result["overall"].get(
                        metric, 0
                    )

            # Recursively aggregate fields
            if "fields" in pair_result:
                aggregated["fields"] = self._recursive_merge_fields(
                    aggregated["fields"], pair_result["fields"]
                )

        return aggregated

    def _recursive_merge_fields(
        self, target_fields: Dict[str, Any], source_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge field metrics - TRUE recursion for arbitrary depth."""
        for field_name, field_metrics in source_fields.items():
            if field_name not in target_fields:
                # Initialize field in target with same structure as source
                if "overall" in field_metrics:
                    # Hierarchical structure
                    target_fields[field_name] = {
                        "overall": {
                            "tp": 0,
                            "fa": 0,
                            "fd": 0,
                            "fp": 0,
                            "tn": 0,
                            "fn": 0,
                        },
                        "fields": {},
                    }
                else:
                    # Flat structure
                    target_fields[field_name] = {
                        "tp": 0,
                        "fa": 0,
                        "fd": 0,
                        "fp": 0,
                        "tn": 0,
                        "fn": 0,
                    }

            # Aggregate metrics based on structure type
            if "overall" in field_metrics:
                # Hierarchical structure - aggregate overall and recurse into fields
                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                    target_fields[field_name]["overall"][metric] += field_metrics[
                        "overall"
                    ].get(metric, 0)

                # Propagate aggregate keys from inner list comparisons.
                # When an inner List[StructuredModel] has already been gated and
                # pre-seeded, its leaf nodes carry 'aggregate' keys that must be
                # summed across multiple pair results.
                if "aggregate" in field_metrics:
                    if "aggregate" not in target_fields[field_name]:
                        target_fields[field_name]["aggregate"] = {
                            "tp": 0, "fa": 0, "fd": 0, "fp": 0, "tn": 0, "fn": 0
                        }
                    for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                        target_fields[field_name]["aggregate"][metric] += field_metrics[
                            "aggregate"
                        ].get(metric, 0)

                # RECURSIVE CALL: Handle nested fields at arbitrary depth
                if "fields" in field_metrics:
                    if "fields" not in target_fields[field_name]:
                        target_fields[field_name]["fields"] = {}
                    target_fields[field_name]["fields"] = self._recursive_merge_fields(
                        target_fields[field_name]["fields"], field_metrics["fields"]
                    )
            else:
                # Flat structure - aggregate directly
                for metric in ["tp", "fa", "fd", "fp", "tn", "fn"]:
                    target_fields[field_name][metric] += field_metrics.get(metric, 0)

        return target_fields
