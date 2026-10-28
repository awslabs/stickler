"""Hungarian matching helper for StructuredModel comparisons."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List

from stickler.algorithms.hungarian import HungarianMatcher
from stickler.comparators.structured import StructuredModelComparator

from .null_helper import NullHelper

_DEFER_PAIR_WARNINGS: ContextVar[bool] = ContextVar(
    "stickler_defer_pair_warnings", default=False
)


def deferring_pair_warnings() -> bool:
    """Whether warnings about a scored pair wait for the caller's re-score.

    True while ``HungarianHelper.get_complete_matching_info`` scores and solves
    the cost matrix for a list of ``StructuredModel`` rows. The matcher scores
    every pair and keeps one per row, so a warning emitted then may describe a
    discarded pair and spend ``warn_once`` before the real one arrives (#336).
    The helper's caller scores the kept pairs again, so code that warns about a
    scored pair checks this and leaves the warning to that re-score. A bare
    ``HungarianMatcher`` does not set it, because nothing re-scores its pairs.

    A ``ContextVar`` does not carry into a worker thread unless the work is run
    in a ``contextvars.copy_context()``.
    """
    return _DEFER_PAIR_WARNINGS.get()


@contextmanager
def defer_pair_warnings() -> Iterator[None]:
    """Make :func:`deferring_pair_warnings` true for the body; see there.

    The previous value is restored on exit, including when the body raises or
    an interrupt arrives as the set returns.
    """
    previous = _DEFER_PAIR_WARNINGS.get()
    try:
        _DEFER_PAIR_WARNINGS.set(True)
        yield
    finally:
        _DEFER_PAIR_WARNINGS.set(previous)


class HungarianHelper:
    """Helper class for Hungarian matching operations with StructuredModel objects."""

    def __init__(self):
        self.hungarian = HungarianMatcher(StructuredModelComparator())

    def get_complete_matching_info(
        self, gt_list: List[Any], pred_list: List[Any]
    ) -> Dict[str, Any]:
        """Get all Hungarian matching information in one call to eliminate redundant calculations.

        This method performs Hungarian matching once and returns all derived information,
        eliminating the need for multiple calls that recalculate the same matching.

        Warnings about scored pairs are deferred while the matrix is scored; see
        :func:`deferring_pair_warnings`. A caller that needs them must score the
        kept pairs again outside this call.

        Args:
            gt_list: Ground truth list
            pred_list: Prediction list

        Returns:
            Dictionary containing:
            - matched_pairs: List of tuples (gt_idx, pred_idx, similarity_score)
            - assignments: List of tuples (gt_idx, pred_idx) without scores
            - unmatched_gt_indices: List of unmatched ground truth indices
            - unmatched_pred_indices: List of unmatched prediction indices
            - matched_gt_indices: Set of matched ground truth indices
            - matched_pred_indices: Set of matched prediction indices
        """
        # A None element is absent (#386): only present elements are matched,
        # and indices are mapped back to the caller's lists so reports keep
        # each element's original position. An absent element is neither
        # matched nor unmatched.
        gt_present = NullHelper.present_indices(gt_list)
        pred_present = NullHelper.present_indices(pred_list)

        if not gt_present or not pred_present:
            return {
                "matched_pairs": [],
                "assignments": [],
                "unmatched_gt_indices": gt_present,
                "unmatched_pred_indices": pred_present,
                "matched_gt_indices": set(),
                "matched_pred_indices": set(),
            }

        # Single Hungarian algorithm call - this is the only expensive operation.
        with defer_pair_warnings():
            metrics = self.hungarian.calculate_metrics(
                [gt_list[i] for i in gt_present], [pred_list[j] for j in pred_present]
            )
        matched_pairs = [
            (gt_present[i], pred_present[j], score)
            for i, j, score in metrics["matched_pairs"]
        ]

        # Derive all other information from this single result
        assignments = [(i, j) for i, j, _ in matched_pairs]
        matched_gt_indices = {i for i, _, _ in matched_pairs}
        matched_pred_indices = {j for _, j, _ in matched_pairs}

        unmatched_gt_indices = [i for i in gt_present if i not in matched_gt_indices]
        unmatched_pred_indices = [
            j for j in pred_present if j not in matched_pred_indices
        ]

        return {
            "matched_pairs": matched_pairs,
            "assignments": assignments,
            "unmatched_gt_indices": unmatched_gt_indices,
            "unmatched_pred_indices": unmatched_pred_indices,
            "matched_gt_indices": matched_gt_indices,
            "matched_pred_indices": matched_pred_indices,
        }
