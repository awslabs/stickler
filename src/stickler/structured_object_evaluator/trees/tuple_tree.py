"""Tuple node implementation for the ANLS* tree."""

import math
from typing import Any, Optional, Tuple, List, Dict, cast

from stickler.comparators.base import BaseComparator
from .base import ANLSTree


class ANLSTuple(ANLSTree):
    """Tuple node for representing multiple valid answers in the ANLS tree.

    This class handles tuples, which represent multiple valid ground truth options.
    For example, if a question has multiple valid answers, a tuple can be used.

    Attributes:
        obj: The original tuple.
        tree: A tuple of ANLSTree nodes representing each item in the tuple.
        _comparator: The comparator used for string similarity.
    """

    def __init__(
        self, obj: Any, *, is_gt: bool, comparator: Optional[BaseComparator] = None
    ):
        """Initialize a tuple node.

        Args:
            obj: The tuple object.
            is_gt: Whether this is a ground truth node. Only ground truths can be tuples.
            comparator: Optional comparator for string comparison.

        Raises:
            ValueError: If obj is not a tuple, or if is_gt is False (predictions cannot be tuples),
                      or if the tuple is empty (must have at least one valid option).
        """
        if not isinstance(obj, tuple):
            raise ValueError(f"ANLSTuple expects a tuple, got {type(obj)}")
        if not is_gt:
            raise ValueError(
                "Tuples are reserved for 1-of-n ground truths. Use lists as containers in predictions."
            )
        if len(obj) == 0:
            raise ValueError("Expected at least 1 valid ground truth option")

        super().__init__(obj, comparator)
        self.tree: Tuple[ANLSTree, ...] = tuple(
            ANLSTree.make_tree(x, is_gt=is_gt, comparator=comparator) for x in obj
        )

    def __len__(self) -> int:
        """Return the length of this tuple.

        Returns:
            The maximum length of any of the tuple's options.
        """
        return max(len(x) for x in self.tree)

    def _choose_best_item(
        self,
        other: ANLSTree,
        key_hierarchy: Tuple[str, ...],
        key_scores: List[Dict[Tuple[str, ...], float]],
    ):
        """Choose the best matching option from the tuple.

        Args:
            other: The other ANLSTree to compare with.
            key_hierarchy: The current key hierarchy for nested structures.
            key_scores: A list to store key-wise scores.

        Returns:
            A tuple containing:
            - The best NLS scores list
            - The best pairwise length
            - The chosen ground truth option
            - The updated key scores list
        """
        candidate_nlss: List[List[float]] = []
        lengths: List[int] = []
        gts: List[Any] = []
        new_key_scores_list: List[List[Dict[Tuple[str, ...], float]]] = []

        for gt in self.tree:
            # Check each option against the prediction
            cand_nlss, chosen_gt, new_key_scores = gt.nls_list(
                other, key_hierarchy, key_scores
            )
            candidate_nlss.append(cand_nlss)
            gts.append(chosen_gt)
            lengths.append(gt.pairwise_len(other))
            new_key_scores_list.append(new_key_scores)

        # Select the best matching choice
        def sort_avg_nls_then_eq(tuple_):
            """Sort by average NLS, then by ground truth equality in case of ties."""
            nls_list, length, gt_item, _ = tuple_
            avg = (sum(nls_list) / length) if length > 0 else 1.0
            gt_eq = 1 if gt_item == other.obj else 0
            return (avg, gt_eq)

        best_nls, best_length, chosen_gt, chosen_key_scores = max(
            zip(candidate_nlss, lengths, gts, new_key_scores_list),
            key=sort_avg_nls_then_eq,
        )
        return best_nls, best_length, chosen_gt, chosen_key_scores

    def pairwise_len(self, other: ANLSTree) -> int:
        """Calculate the pairwise length between this tuple and another tree.

        Args:
            other: The other ANLSTree to compare with.

        Returns:
            The pairwise length for the best matching option.
        """
        best_nls, best_length, chosen_gt, chosen_key_scores = self._choose_best_item(
            other, (), []
        )
        return best_length

    def nls_list(
        self,
        other: ANLSTree,
        key_hierarchy: Tuple[str, ...],
        key_scores: List[Dict[Tuple[str, ...], float]],
    ) -> Tuple[List[float], Any, List[Dict[Tuple[str, ...], float]]]:
        """Calculate the NLS score between this tuple and another tree.

        Args:
            other: The other ANLSTree to compare with.
            key_hierarchy: The current key hierarchy for nested structures.
            key_scores: A list to store key-wise scores.

        Returns:
            A tuple containing:
            - The NLS scores list for the best matching option
            - The chosen ground truth option
            - An updated list of key scores
        """
        key_scores = key_scores.copy()
        best_nls, best_length, chosen_gt, chosen_key_scores = self._choose_best_item(
            other, key_hierarchy, key_scores
        )
        return best_nls, chosen_gt, chosen_key_scores
