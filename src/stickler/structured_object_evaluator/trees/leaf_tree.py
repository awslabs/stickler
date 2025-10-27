"""Leaf node implementation for the ANLS* tree."""

from typing import Any, Optional, Tuple, List, Dict

from stickler.comparators.base import BaseComparator
from .base import ANLSTree


class ANLSLeaf(ANLSTree):
    """Leaf node for primitive values in the ANLS tree.

    This class represents leaf nodes in the ANLSTree for primitive types
    (strings, numbers, booleans). It compares values using the provided
    string comparator.

    Attributes:
        obj: The primitive value represented by this leaf node.
        _comparator: The comparator used for string similarity.
    """

    def __init__(self, obj: Any, comparator: Optional[BaseComparator] = None):
        """Initialize a leaf node.

        Args:
            obj: The primitive value (str, float, int, bool).
            comparator: Optional comparator for string comparison.

        Raises:
            ValueError: If obj is not a primitive type.
        """
        if not isinstance(obj, (str, float, int, bool)):
            raise ValueError(f"Leaf must be a primitive type, got {type(obj)}")
        super().__init__(obj, comparator)

    def __len__(self) -> int:
        """Return the length of this leaf node.

        Returns:
            Always 1 for leaf nodes.
        """
        return 1

    def pairwise_len(self, other: ANLSTree) -> int:
        """Calculate the pairwise length between this leaf and another tree.

        Args:
            other: The other ANLSTree to compare with.

        Returns:
            The maximum length of the two trees (for leaf nodes, this is usually 1).
        """
        return max(len(self), len(other))

    def nls_list(
        self,
        other: ANLSTree,
        key_hierarchy: Tuple[str, ...],
        key_scores: List[Dict[Tuple[str, ...], float]],
    ) -> Tuple[List[float], Any, List[Dict[Tuple[str, ...], float]]]:
        """Calculate the NLS score between this leaf and another tree.

        Args:
            other: The other ANLSTree to compare with.
            key_hierarchy: The current key hierarchy for nested structures.
            key_scores: A list to store key-wise scores.

        Returns:
            A tuple containing:
            - A list of NLS scores
            - The closest ground truth object (the original leaf value)
            - An updated list of key scores
        """
        key_scores_copy = key_scores.copy()

        if not isinstance(other, ANLSLeaf):
            # Type mismatch, so the ANLS is 0. But we still return our object
            # as the closest ground truth.
            return [0.0], self.obj, key_scores_copy

        # Normalize strings: strip whitespace, convert to lowercase, normalize spaces
        this_str = " ".join(str(self.obj).strip().lower().split())
        other_str = " ".join(str(other.obj).strip().lower().split())

        # Use the configured comparator to calculate similarity
        similarity = self._comparator.compare(this_str, other_str)

        # Apply the ANLS threshold
        question_result = 0.0 if similarity < self.THRESHOLD else similarity

        return [question_result], self.obj, key_scores_copy
