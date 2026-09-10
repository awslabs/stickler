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
    """Applies the object-pair gate to a list element, else delegates.

    List elements reach a comparator through the Hungarian cost matrix, which
    sees values and knows nothing about the field. Wrapping the comparator is
    what makes a list element obey the same rules as the singular form: both
    then read the same configured comparator, with the same gate in front of it.
    Without this, `Single(pet=Cat('rex'))` against `Dog('rev')` scored 0.0 while
    `Listed(pets=[Cat('rex')])` against `[Dog('rex')]` scored 1.0 -- the parity
    #319 exists to establish, broken in the opposite direction. See #321 for
    whether a refused element should pair at all.

    The gate itself is `ConfigurationHelper.can_compare_object_pair`, the same
    call `ComparisonDispatcher` CASE 4/CASE 5 and both `compare_field_raw`
    readers make, so an element and a singular value are judged by one rule.

    Installed when at least one element of EITHER list is a plain model, so the
    ordinary primitive-list path keeps its cost-matrix hot loop unwrapped. The
    LIST is what qualifies for wrapping; which PAIRS are then judged is
    `_judges`, and it is a pair with a plain model on at least one side. A pair
    with a plain model on neither -- dict against dict, `StructuredModel` against
    `StructuredModel` -- is delegated untouched.

    That narrowing is the whole correctness of this wrapper, and it was wrong in
    both directions before. Judging every pair because the list qualified made an
    element's verdict depend on what ELSE the list happened to hold:

        f: Optional[List[Any]]                          dev    list-wide gate
        [{'a':1},{'b':2},{'c':3}]                       1.0    1.0
        [{'a':1},{'b':2},{'c':3}, Plain('p')]           1.0    0.0   <- fd on all four
        [NoteSM]      vs [Note2SM]                      1.0    1.0
        [Plain, NoteSM] vs [Plain, Note2SM]             1.0    0.5

    The dict row zeroed three elements that score 1.0 on their own, in the same
    list, on `dev`, and here the moment the plain model is removed -- and it drew
    a "holds a mapping" warning on a field whose annotation says nothing about
    mappings. The `StructuredModel` row is worse than wrong-looking: it enforces
    the cross-class rule that #327 is explicitly holding open, so a user got the
    #327 answer or the documented one according to whether an unrelated plain
    model shared the list. Both rules now apply to exactly the population the
    wrapper was installed for.

    A dict element does not reach the gate -- neither because
    `_holds_a_plain_model` does not look for one, which was the old and only
    conditionally true reason, nor by accident: `_is_plain_model` is asked of both
    sides here.

    WARNING TIMING. The matcher asks this comparator for every cell of the cost
    matrix and keeps one per row, so warning from inside `_compare` announced an
    outcome that never happened -- an identical `[Plain('aaa'), Note('bbb')]` pair
    scored 1.0 with `tp=2` and still warned that a `Plain` had been compared
    against a `Note`, from the cross cell the matcher discarded. Worse, `warn_once`
    spends one message per field for the life of the process, so that probe cell
    silenced the genuine wrong-class prediction later in the same corpus. Probing
    is therefore silent, and `warn_for_selected` replays the gate on the pairs the
    matcher actually chose.
    """

    def __init__(self, inner: BaseComparator, model_cls=None, field_name: str = ""):
        super().__init__(getattr(inner, "threshold", 0.5))
        self._inner = inner
        self._model_cls = model_cls
        self._field_name = field_name

    def _judges(self, gt_val: Any, pred_val: Any) -> bool:
        """Whether this pair is the wrapper's business at all.

        EITHER side, not both. `and` is the tempting reading and it is wrong: it
        waves through a plain model against a `StructuredModel`, which is a
        documented mismatch that `dev` also reports, and which CASE 5 cannot
        dispatch because CASE 3 needs both sides to be a `StructuredModel`.

        `or` is already enough to exclude both defects this predicate exists for,
        because each is a pair with NEITHER side plain: dict against dict, and
        `StructuredModel` against `StructuredModel`.
        """
        return _is_plain_model(gt_val) or _is_plain_model(pred_val)

    def _refuses(self, gt_val: Any, pred_val: Any, *, warn: bool) -> bool:
        from .configuration_helper import ConfigurationHelper

        return not ConfigurationHelper.can_compare_object_pair(
            self._model_cls, self._field_name, self._inner, gt_val, pred_val, warn=warn
        )

    def _compare(self, str1: Any, str2: Any) -> float:
        if self._judges(str1, str2) and self._refuses(str1, str2, warn=False):
            return 0.0
        return self._inner.compare(str1, str2)

    def warn_for_selected(
        self, gt_list: List[Any], pred_list: List[Any], matched_pairs: List[Any]
    ) -> None:
        """Emit the gate's warnings for the pairs the matcher selected.

        Replays the gate rather than recording verdicts during probing: the
        verdict is a pure function of the pair, so a replay cannot disagree with
        what scoring did, while a cache keyed on `id()` can be wrong the moment a
        list holds two equal-but-distinct objects.
        """
        for gt_idx, pred_idx, *_ in matched_pairs:
            if not (0 <= gt_idx < len(gt_list) and 0 <= pred_idx < len(pred_list)):
                continue
            gt_val, pred_val = gt_list[gt_idx], pred_list[pred_idx]
            if self._judges(gt_val, pred_val):
                self._refuses(gt_val, pred_val, warn=True)


def _is_plain_model(value: Any) -> bool:
    """Whether a value is a pydantic model that is NOT a `StructuredModel`.

    The distinction the whole element gate turns on. A `StructuredModel` also
    passes `isinstance(v, BaseModel)`, and it is scored by recursion rather than
    by the field's comparator, so it is not what this gate is about.
    """
    from pydantic import BaseModel

    from .structured_model import StructuredModel

    return isinstance(value, BaseModel) and not isinstance(value, StructuredModel)


def _holds_a_plain_model(items: List[Any]) -> bool:
    """Whether any element is a plain pydantic model.

    Scans the whole list rather than the first element: a heterogeneous list is
    exactly the case that needs the gate, and keying on `items[0]` would miss
    `[Cat(...), "text"]`.

    Decides only whether the list is WRAPPED. Which pairs the wrapper then judges
    is `_ClassGatedComparator._judges`, and the two must not be conflated: reading
    this predicate as the gate's population is what made a dict element's score
    depend on whether a plain model shared its list.
    """
    return any(_is_plain_model(item) for item in items)


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
        clip_under_threshold: bool = True,
        model_cls=None,
        field_name: str = "",
    ) -> Dict[str, Any]:
        """Compare two lists as unordered collections using Hungarian matching.

        Args:
            gt_list: Ground truth list
            pred_list: Prediction list
            comparator: Comparator to use for item comparison
            threshold: Minimum score to consider a match
            clip_under_threshold: Whether a sub-threshold element contributes 0.0
                rather than its own score
            model_cls: Owning model, passed to the element gate. Names the field
                in the gate's warning; the verdict does not depend on it.
            field_name: Field being compared, likewise for the warning. Both
                default to a falsy value so a caller with no field in hand still
                gets the gate -- it refuses the same pairs either way, and only
                the warning is less specific.

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
            hungarian_info = hungarian_helper.get_complete_matching_info(
                gt_list, pred_list
            )
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

            # Only now is it known which pairs were chosen. The gate ran silently
            # over the whole cost matrix; warning from inside it described
            # discarded cells and spent `warn_once`'s single message per field on
            # them. See `_ClassGatedComparator.warn_for_selected`.
            if isinstance(element_comparator, _ClassGatedComparator):
                element_comparator.warn_for_selected(
                    gt_list, pred_list, matched_pairs
                )

        return ComparisonHelper.unordered_list_metrics(
            matched_pairs=matched_pairs,
            clip_under_threshold=clip_under_threshold,
            gt_list=gt_list,
            pred_list=pred_list,
            classification_threshold=classification_threshold,
        )

    @staticmethod
    def unordered_list_metrics(
        matched_pairs: List[Any],
        gt_list: List[Any],
        pred_list: List[Any],
        classification_threshold: float,
        clip_under_threshold: bool = True,
    ):
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
            # Apply threshold to each similarity score (same logic as individual
            # comparison), UNLESS the field turned clipping off.
            #
            # `clip_under_threshold` means one thing everywhere: a value that
            # missed its bar contributes 0.0 rather than its partial similarity.
            # This is the list spelling of it, per ELEMENT, and it used to be
            # unconditional here -- so a list field could not opt out while a
            # scalar field could, and `clip_under_threshold=False` was silently a
            # no-op on every list. Two fields declaring the same comparator, the
            # same threshold and the same values then disagreed:
            #
            #     raw 0.5625, threshold 0.9, clip_under_threshold=False
            #       scalar field   0.5625
            #       list field     0.0000     <- opted out, still clipped
            #
            # The default is True, so an ordinary `List[str]` is unchanged; only a
            # field that asked to keep partial credit now gets it. Classification
            # is deliberately NOT affected: the counts above already ran, so a
            # sub-threshold pair stays one `fd` either way. Only the score moves.
            threshold_applied_similarities = []
            for _, _, score in matched_pairs:
                # Use ThresholdHelper for consistent threshold checking
                if ThresholdHelper.is_above_threshold(score, classification_threshold):
                    threshold_applied_similarities.append(score)
                elif clip_under_threshold:
                    # Below threshold gets 0.0 (same as individual comparison clipping)
                    threshold_applied_similarities.append(0.0)
                else:
                    threshold_applied_similarities.append(score)

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

        # A pair of models this field cannot score is not comparable, and this
        # function must agree with `compare_with` about that. Its own comment
        # above forbids the two readers disagreeing (#233), and without the gate
        # they did, on BOTH of its rules in turn: `compare()` returned 1.0 for
        # Cat/Dog and 0.4167 for Base/Sub, then 1.0 for two identical models on
        # an `Any` field, where `compare_with` reported 0.0 and a false discovery
        # each time. `compare()` also feeds the Hungarian cost matrix, so a
        # List[Holder] paired those items at zero cost and then called the field
        # a mismatch.
        #
        # `can_compare_object_pair` rather than either rule spelled out here:
        # composing them in one place is what stops this reader adopting a new
        # rule's first half and missing its second, which is how both of the
        # above arrived. The mapping half of the same gate is applied one frame
        # up, in `StructuredModel.compare_field_raw`, before a dict pair can
        # reach here.
        if not ConfigurationHelper.can_compare_object_pair(
            structured_model_instance.__class__,
            field_name,
            comparator,
            self_value,
            other_value,
        ):
            return 0.0

        # Use the comparator to calculate raw similarity (no threshold)
        return comparator.compare(self_value, other_value)
