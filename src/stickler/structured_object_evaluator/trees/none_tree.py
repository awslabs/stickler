"""None node implementation for the ANLS* tree."""

from typing import Any, Optional, Tuple, List, Dict

from stickler.comparators.base import BaseComparator
from .base import ANLSTree


class ANLSNone(ANLSTree):
    """None node for representing None values in the ANLS tree.

    This class represents None values, empty lists, empty dicts, and empty strings
    in the ANLSTree. It treats all these "None-y" values as equivalent.

    Attributes:
        obj: Always None for this class.
        _comparator: The comparator used for string similarity (though not used
                    directly by this class).
    """

    def __init__(self, comparator: Optional[BaseComparator] = None):
        """Initialize a None node.

        Args:
            comparator: Optional comparator for string comparison.
        """
        super().__init__(None, comparator)

    def __len__(self) -> int:
        """Return the length of this None node.

        Returns:
            Always 1 for None nodes.
        """
        return 1

    def pairwise_len(self, other: ANLSTree) -> int:
        """Calculate the pairwise length between this None and another tree.

        Args:
            other: The other ANLSTree to compare with.

        Returns:
            The maximum length of the two trees (for None nodes, this is usually 1).
        """
        return max(len(self), len(other))

    def nls_list(
        self,
        other: ANLSTree,
        key_hierarchy: Tuple[str, ...],
        key_scores: List[Dict[Tuple[str, ...], float]],
    ) -> Tuple[List[float], Any, List[Dict[Tuple[str, ...], float]]]:
        """Calculate the NLS score between this None and another tree.

        Args:
            other: The other ANLSTree to compare with.
            key_hierarchy: The current key hierarchy for nested structures.
            key_scores: A list to store key-wise scores.

        Returns:
            A tuple containing:
            - A list of NLS scores
            - The closest ground truth object
            - An updated list of key scores
        """
        key_scores_copy = key_scores.copy()

        if self.check_if_none(other.obj):
            # If the prediction is "None-y", it's a perfect match
            # Return the prediction as the closest ground truth
            return [1.0], other.obj, key_scores_copy
        else:
            # If the prediction is not "None-y", it's a complete mismatch
            # Return our None as the closest ground truth
            return [0.0], self.obj, key_scores_copy

    @classmethod
    def check_if_none(cls, value: Any) -> bool:
        """Check if a value is None-like (None, empty list, empty dict, etc.).

        Args:
            value: The value to check.

        Returns:
            True if the value is None-like, False otherwise.
        """
        return (
            isinstance(value, ANLSNone)
            or value in (None, {}, [], "")
            or (hasattr(value, "__len__") and len(value) == 0)
        )
