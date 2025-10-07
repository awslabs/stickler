"""Dict node implementation for the ANLS* tree."""

from typing import Any, Optional, Tuple, List, Dict as PyDict, Set, cast

from stickler.comparators.base import BaseComparator
from .base import ANLSTree
from .none_tree import ANLSNone


class ANLSDict(ANLSTree):
    """Dict node for representing key-value structures in the ANLS tree.

    This class handles dictionaries, which represent mappings from keys to values
    like JSON objects, configuration files, or extracted fields from documents.

    Attributes:
        obj: The original dictionary.
        tree: A dictionary of ANLSTree nodes representing each key-value pair.
        _comparator: The comparator used for string similarity.
    """

    def __init__(
        self, obj: Any, *, is_gt: bool, comparator: Optional[BaseComparator] = None
    ):
        """Initialize a dictionary node.

        Args:
            obj: The dictionary object.
            is_gt: Whether this is a ground truth node.
            comparator: Optional comparator for string comparison.

        Raises:
            ValueError: If obj is not a dictionary.
        """
        if not isinstance(obj, dict):
            raise ValueError(f"ANLSDict expects a dict, got {type(obj)}")

        super().__init__(obj, comparator)
        self.tree: PyDict[Any, ANLSTree] = {
            k: ANLSTree.make_tree(v, is_gt=is_gt, comparator=comparator)
            for k, v in obj.items()
        }

    def __len__(self) -> int:
        """Return the length of this dictionary.

        Returns:
            The sum of the lengths of all values in the dictionary.
        """
        return sum(len(x) for x in self.tree.values())

    def pairwise_len(self, other: ANLSTree) -> int:
        """Calculate the pairwise length between this dictionary and another tree.

        Args:
            other: The other ANLSTree to compare with.

        Returns:
            The pairwise length between the two trees.
        """
        if not isinstance(other, ANLSDict):
            return max(len(self), len(other))

        # Calculate pairwise length for each key in either dictionary
        pwl = 0
        for k in self.tree.keys() | other.tree.keys():
            self_value = self.tree.get(k, ANLSNone(self._comparator))
            other_value = other.tree.get(k, ANLSNone(self._comparator))
            pwl += self_value.pairwise_len(other_value)
        return pwl

    def nls_list(
        self,
        other: ANLSTree,
        key_hierarchy: Tuple[str, ...],
        key_scores: List[PyDict[Tuple[str, ...], float]],
    ) -> Tuple[List[float], Any, List[PyDict[Tuple[str, ...], float]]]:
        """Calculate the NLS score between this dictionary and another tree.

        Args:
            other: The other ANLSTree to compare with.
            key_hierarchy: The current key hierarchy for nested structures.
            key_scores: A list to store key-wise scores.

        Returns:
            A tuple containing:
            - A list of NLS scores for each key-value pair
            - The closest ground truth dictionary
            - An updated list of key scores
        """
        key_scores_copy = key_scores.copy()

        if not isinstance(other, ANLSDict):
            return [0.0], self.obj, key_scores_copy

        # Compute NLS scores for each key in either dictionary
        nlss = []
        chosen_gts = {}

        # Process all keys in either dictionary
        for k in list(self.tree.keys()) + [
            k for k in other.tree.keys() if k not in self.tree.keys()
        ]:
            self_value = self.tree.get(k, ANLSNone(self._comparator))
            other_value = other.tree.get(k, ANLSNone(self._comparator))

            # Skip hallucinated None keys
            is_hallucinated_none_key = (
                k not in self.tree
                and k in other.tree
                and ANLSNone.check_if_none(other_value.obj)
            )
            if is_hallucinated_none_key:
                continue

            # Calculate NLS for this key
            new_key_hierarchy = key_hierarchy + (str(k),)
            nls_list, chosen_gt, new_key_scores = self_value.nls_list(
                other_value, new_key_hierarchy, []
            )
            nlss.extend(nls_list)

            # Track chosen ground truths, excluding "no key" cases
            closest_gt_is_no_key = (
                ANLSNone.check_if_none(chosen_gt) and k not in other.tree
            )
            if not closest_gt_is_no_key:
                chosen_gts[k] = chosen_gt

            # Calculate mean NLS for this key and update key scores
            length = self_value.pairwise_len(other_value)
            mean_nls = sum(nls_list) / length if length > 0 else 1.0
            key_scores_copy.extend(new_key_scores)
            key_scores_copy.append({new_key_hierarchy: mean_nls})

        return nlss, chosen_gts, key_scores_copy
