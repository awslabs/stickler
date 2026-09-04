"""Comparison helper for StructuredModel field comparison operations.

This module provides utilities for comparing fields, lists, and nested structures
within StructuredModel instances.
"""

from typing import Any, Dict, List

from stickler.comparators.base import BaseComparator

from .hungarian_helper import HungarianHelper
from .null_helper import NullHelper
from .threshold_helper import ThresholdHelper


class _ClassGatedComparator(BaseComparator):
    """Refuses two plain models of different classes, else delegates.

    List elements reach a comparator through the Hungarian cost matrix, which
    sees values and knows nothing about the field. Wrapping the comparator is
    what makes a list element obey the same class rule as the singular form:
    both then read the same configured comparator, with the gate in front of it.
    Without this, `Single(pet=Cat('rex'))` against `Dog('rev')` scored 0.0 while
    `Listed(pets=[Cat('rex')])` against `[Dog('rex')]` scored 1.0 -- the parity
    #319 exists to establish, broken in the opposite direction. See #321 for
    whether a refused element should pair at all.

    Applied only when an element is actually a plain model, so the ordinary
    primitive-list path keeps its cost-matrix hot loop unwrapped.
    """

    def __init__(self, inner: BaseComparator, model_cls=None, field_name: str = ""):
        super().__init__(getattr(inner, "threshold", 0.5))
        self._inner = inner
        self._model_cls = model_cls
        self._field_name = field_name

    def _compare(self, str1: Any, str2: Any) -> float:
        from .configuration_helper import ConfigurationHelper

        if not ConfigurationHelper.values_are_same_model_class(
            self._model_cls, self._field_name, str1, str2
        ):
            return 0.0
        return self._inner.compare(str1, str2)


def _holds_a_plain_model(items: List[Any]) -> bool:
    """Whether any element is a plain pydantic model.

    Scans the whole list rather than the first element: a heterogeneous list is
    exactly the case that needs the gate, and keying on `items[0]` would miss
    `[Cat(...), "text"]`.
    """
    from pydantic import BaseModel

    from .structured_model import StructuredModel

    return any(
        isinstance(item, BaseModel) and not isinstance(item, StructuredModel)
        for item in items
    )


def _maybe_absent(val: Any) -> bool:
    """Whether ``val`` could be absent under *either* of ``NullHelper``'s rules.

    A cheap over-approximation, used only to decide whether it is worth reading
    a field's annotation to find out which rule applies. It must stay a
    superset of both :meth:`NullHelper.is_effectively_null_for_lists` and
    :meth:`NullHelper.is_effectively_null_for_primitives`: returning ``False``
    for something either one calls absent would silently skip the true-negative
    and false-negative handling in
    :meth:`ComparisonHelper.compare_field_raw`. Returning ``True`` too often
    only costs the lookup it was meant to avoid.

    ``test_maybe_absent_is_a_superset_of_both_null_rules`` pins that property so
    adding a case to either predicate without widening this one fails loudly.
    """
    return val is None or (isinstance(val, (str, list, dict)) and len(val) == 0)


class ComparisonHelper:
    """Helper class for StructuredModel field comparison operations."""

    @staticmethod
    def compare_unordered_lists(
        gt_list: List[Any],
        pred_list: List[Any],
        comparator: BaseComparator,
        threshold: float,
        model_cls=None,
        field_name: str = "",
    ) -> Dict[str, Any]:
        """Compare two lists as unordered collections using Hungarian matching.

        Args:
            list1: First list
            list2: Second list
            comparator: Comparator to use for item comparison
            threshold: Minimum score to consider a match

        Returns:
            Dictionary with confusion matrix metrics including:
            - tp: True positives (matches >= threshold)
            - fd: False discoveries (matches < threshold)
            - fa: False alarms (unmatched prediction items)
            - fn: False negatives (unmatched ground truth items)
            - fp: Total false positives (fd + fa)
            - overall_score: Similarity score for backward compatibility
        """
        # Empty lists reach here and fall through to `unordered_list_metrics`,
        # which scores two of them 1.0. `ComparisonDispatcher` short-circuits an
        # absent list field before this function is called, but only for fields
        # `_is_list_field` recognizes -- a list held in an `Any`-annotated field
        # arrives here empty.

        # Use HungarianHelper for Hungarian matching operations
        hungarian_helper = HungarianHelper()
        from .structured_model import StructuredModel

        # Use the appropriate comparator based on item types
        # Import here to avoid circular import

        if all(isinstance(item, StructuredModel) for item in gt_list[:1]) and all(
            isinstance(item, StructuredModel) for item in pred_list[:1]
        ):
            # For StructuredModel lists, we need to use individual comparison scoring for consistency
            # Use HungarianHelper to get optimal pairings - OPTIMIZED: Single call gets all info
            hungarian_info = hungarian_helper.get_complete_matching_info(gt_list, pred_list)
            matched_pairs = hungarian_info["matched_pairs"]

            # CRITICAL FIX: Replace raw scores with threshold-applied scores from individual comparison
            # This ensures consistency between individual and list comparison results
            threshold_corrected_pairs = []
            for gt_idx, pred_idx, raw_score in matched_pairs:
                if gt_idx < len(gt_list) and pred_idx < len(pred_list):
                    gt_item = gt_list[gt_idx]
                    pred_item = pred_list[pred_idx]

                    if gt_item is None or pred_item is None:
                        # Nullable object elements (List[Optional[Model]]) can pair
                        # a None against a model; compare_with would crash on None,
                        # so score it directly: both-None matches, one-None does not.
                        threshold_applied_score = (
                            1.0 if gt_item is None and pred_item is None else 0.0
                        )
                    else:
                        # Use individual comparison with threshold application (same as .compare_with())
                        individual_result = gt_item.compare_with(pred_item)
                        threshold_applied_score = individual_result["overall_score"]

                    threshold_corrected_pairs.append(
                        (gt_idx, pred_idx, threshold_applied_score)
                    )
                else:
                    threshold_corrected_pairs.append((gt_idx, pred_idx, raw_score))

            # Replace matched_pairs with threshold-corrected version
            matched_pairs = threshold_corrected_pairs

            # Use a very low threshold since we've already applied thresholds in individual comparison
            classification_threshold = (
                0.01  # Almost everything that's not 0.0 should be TP
            )
        else:
            # Use the provided comparator for other types
            from stickler.algorithms.hungarian import HungarianMatcher

            # CRITICAL FIX: Use match_threshold=0.0 to capture ALL matches, not just those above threshold
            # This allows us to keep track of partial matches for scoring.
            #
            # `0.0` is a capture-all sentinel: only `matched_pairs` is read
            # below, and classification happens against
            # `classification_threshold` instead. Do not read `tp`/`fp`/`fn`
            # from a matcher built this way -- every pair satisfies
            # `score >= 0.0`, so its `tp` counts pairs, not true positives.
            #
            # `normalize_values=False` because comparators own their own
            # normalization: `ExactComparator.case_sensitive`,
            # `LevenshteinComparator._normalize`, `FuzzyComparator._normalize`.
            # The matcher's normalization is a legacy pre-comparator behavior
            # that lowercases, collapses whitespace and `str()`-coerces every
            # item before the comparator sees it, which silently overrides the
            # field's declared comparator -- it defeated #199 for every
            # list-typed field, and made `List[bbox]` unscoreable because
            # stringified coordinates cannot be parsed. Items must reach the
            # comparator exactly as the caller supplied them, so that a list
            # field and a scalar field score the same pair identically.
            # Gate the element comparator on class identity. See
            # _ClassGatedComparator; skipped entirely for ordinary primitive
            # lists so the cost matrix stays unwrapped.
            element_comparator = comparator
            if _holds_a_plain_model(gt_list) or _holds_a_plain_model(pred_list):
                element_comparator = _ClassGatedComparator(
                    comparator, model_cls=model_cls, field_name=field_name
                )

            hungarian = HungarianMatcher(
                element_comparator, match_threshold=0.0, normalize_values=False
            )
            classification_threshold = threshold

            # Get detailed metrics from HungarianMatcher
            metrics = hungarian.calculate_metrics(gt_list, pred_list)
            matched_pairs = metrics["matched_pairs"]

        return ComparisonHelper.unordered_list_metrics(matched_pairs=matched_pairs,
                                                       gt_list=gt_list,
                                                       pred_list=pred_list,
                                                       classification_threshold=classification_threshold)
    
    @staticmethod
    def unordered_list_metrics(matched_pairs:List[Any],
                        gt_list: List[Any],
                        pred_list: List[Any],
                        classification_threshold: float):
        """
        Compare two lists as unordered collections using Hungarian matching.

        Args:
            list1: First list
        Returns:
                Dictionary with confusion matrix metrics including:
                - tp: True positives (matches >= threshold)
                - fd: False discoveries (matches < threshold)
                - fa: False alarms (unmatched prediction items)
                - fn: False negatives (unmatched ground truth items)
                - fp: Total false positives (fd + fa)
                - overall_score: Similarity score for backward compatibility
        """
        tp = 0  # True positives (score >= threshold)
        fd = 0  # False discoveries (score < threshold, including 0)

        for i, j, score in matched_pairs:
            # Use ThresholdHelper for consistent threshold checking
            if ThresholdHelper.is_above_threshold(score, classification_threshold):
                tp += 1
            else:
                # All matches below threshold are False Discoveries, including 0.0 scores
                fd += 1

        # False negatives are unmatched ground truth items
        fn = len(gt_list) - len(matched_pairs)

        # False alarms are unmatched prediction items
        fa = len(pred_list) - len(matched_pairs)

        # Total false positives include both false discoveries and false alarms
        fp = fd + fa

        # CRITICAL FIX: Use threshold-applied scores for consistency with individual comparison
        # This ensures list comparison matches the same scoring logic as individual comparison
        if not matched_pairs:
            # Two empty lists agree perfectly: there is nothing to find and
            # nothing was found. Scoring that 0.0 made an object whose only
            # field is an empty list compare as a total mismatch against an
            # identical object, which then classified as a false discovery even
            # though `compare_with` reported a perfect match. The dispatcher
            # already treats both-empty as a true negative with score 1.0; this
            # keeps the raw path in agreement with it.
            # See https://github.com/awslabs/stickler/issues/233
            overall_score = 1.0 if not gt_list and not pred_list else 0.0
        else:
            # Apply threshold to each similarity score (same logic as individual comparison)
            threshold_applied_similarities = []
            for _, _, score in matched_pairs:
                # Use ThresholdHelper for consistent threshold checking
                if ThresholdHelper.is_above_threshold(score, classification_threshold):
                    threshold_applied_similarities.append(score)
                else:
                    # Below threshold gets 0.0 (same as individual comparison clipping)
                    threshold_applied_similarities.append(0.0)

            # Average the threshold-applied similarities
            avg_threshold_similarity = sum(threshold_applied_similarities) / len(
                threshold_applied_similarities
            )

            # Scale by coverage ratio (matched pairs / max list size)
            max_items = max(len(gt_list), len(pred_list))
            coverage_ratio = len(matched_pairs) / max_items if max_items > 0 else 1.0
            overall_score = avg_threshold_similarity * coverage_ratio

        return {
            "tp": tp,
            "fd": fd,
            "fa": fa,
            "fn": fn,
            "fp": fp,
            "overall_score": overall_score,
        }

    @staticmethod
    def compare_field_raw(
        structured_model_instance, field_name: str, other_value: Any
    ) -> float:
        """Compare a single field with a value WITHOUT applying thresholds.

        This version is used by the compare method to get raw similarity scores.

        Args:
            structured_model_instance: StructuredModel instance
            field_name: Name of the field to compare
            other_value: Value to compare with

        Returns:
            Raw similarity score between 0.0 and 1.0 without threshold filtering
        """
        # Import here to avoid circular import
        from .configuration_helper import ConfigurationHelper

        info = ConfigurationHelper.get_comparison_info(
            structured_model_instance.__class__, field_name
        )

        # We should always get a ComparableField object now
        comparator = info.comparator

        # Get field value from self
        self_value = getattr(structured_model_instance, field_name)

        # Read absence the same way `ComparisonDispatcher` does. It scores a
        # field absent on both sides as a true negative worth 1.0 and a field
        # absent on exactly one side as FN/FA worth 0.0 (STEP 3 for list fields,
        # STEP 4 for everything else). Without the same rule here the two score
        # readers disagree: `compare_with` reports a perfect match while the raw
        # similarity lands under `match_threshold`, so the same pair is a perfect
        # match and a false discovery at once.
        #
        # What counts as absent is per-kind, exactly as the dispatcher defines
        # it: for a list field `None` and `[]` both mean "no items"; for
        # everything else `None`, `""` and `{}` all mean "no value". Both
        # predicates treat `None` as absent, which is why this subsumes the bare
        # `is None` check that used to stand here.
        # See https://github.com/awslabs/stickler/issues/233
        #
        # `_maybe_absent` guards the annotation lookup rather than duplicating
        # it. This function runs once per field per pairwise comparison, which
        # is every cell of a Hungarian cost matrix -- 60x60 objects of 20 fields
        # is 72,000 calls -- and `_is_list_field` re-reads `model_fields` and
        # destructures the annotation on each one. The bare `is None` check this
        # replaced short-circuited before doing any of that, so consulting the
        # annotation unconditionally cost ~23% on that shape. The guard is the
        # union of the two predicates below, so anything either one would call
        # absent still reaches it and the outcome is unchanged; both sides being
        # populated, the overwhelmingly common case, now skips the lookup.
        if _maybe_absent(self_value) or _maybe_absent(other_value):
            if structured_model_instance._is_list_field(field_name):
                is_absent = NullHelper.is_effectively_null_for_lists
            else:
                is_absent = NullHelper.is_effectively_null_for_primitives

            self_is_null = is_absent(self_value)
            other_is_null = is_absent(other_value)
            if self_is_null or other_is_null:
                return 1.0 if self_is_null and other_is_null else 0.0

        # Handle lists with special processing
        if isinstance(self_value, list) and isinstance(other_value, list):
            threshold = 0.0  # Use zero threshold for raw comparisons
            result = ComparisonHelper.compare_unordered_lists(
                self_value,
                other_value,
                comparator,
                threshold,
                model_cls=structured_model_instance.__class__,
                field_name=field_name,
            )
            return result["overall_score"]

        # Handle nested StructuredModel objects
        from .structured_model import StructuredModel

        if isinstance(self_value, StructuredModel) and isinstance(
            other_value, StructuredModel
        ):
            return self_value.compare(other_value)

        # Handle dictionary objects using the field's configured comparator
        if isinstance(self_value, dict) and isinstance(other_value, dict):
            return comparator.compare(self_value, other_value)

        # Two plain models of different classes are not comparable, and this
        # function must agree with `compare_with` about that. Its own comment
        # above forbids the two readers disagreeing (#233), and without this
        # gate they did: `compare()` returned 1.0 for Cat/Dog and 0.4167 for
        # Base/Sub where `compare_with` reported 0.0 and a false discovery.
        # `compare()` also feeds the Hungarian cost matrix, so a List[Holder]
        # paired those items at zero cost and then called the field a mismatch.
        if not ConfigurationHelper.values_are_same_model_class(
            structured_model_instance.__class__, field_name, self_value, other_value
        ):
            return 0.0

        # Use the comparator to calculate raw similarity (no threshold)
        return comparator.compare(self_value, other_value)
