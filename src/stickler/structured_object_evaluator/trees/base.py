"""Base ANLSTree class for structured object evaluation.

This module provides the ANLSTree class for structured object evaluation.
"""

import abc
from typing import Any, Optional, Tuple, List, Dict, Type, Union, cast

from stickler.comparators.base import BaseComparator
from stickler.comparators.levenshtein import LevenshteinComparator


class ANLSTree(abc.ABC):
    """Base abstract class for ANLS tree nodes.

    This class defines the interface that all ANLS tree nodes must implement.
    It provides functionality for calculating ANLS scores between tree nodes.

    Attributes:
        THRESHOLD: ANLS threshold. 0.5 is a standard value.
        obj: The original object represented by this tree.
        tree: The tree structure representing the object.
        _comparator: The comparator used for string similarity.
    """

    THRESHOLD = 0.5  # ANLS threshold. 0.5 is a standard value.
    obj: Any
    tree: Any
    _comparator: BaseComparator

    def __init__(self, obj: Any, comparator: Optional[BaseComparator] = None):
        """Initialize the ANLSTree.

        Args:
            obj: The object represented by this tree.
            comparator: Optional comparator for string comparisons.
        """
        self.obj = obj
        self._comparator = comparator or LevenshteinComparator()

    @staticmethod
    def make_tree(
        obj: Any, *, is_gt: bool, comparator: Optional[BaseComparator] = None
    ) -> "ANLSTree":
        """Make an ANLS tree from a complex object.

        Args:
            obj: The object to make a tree from.
            is_gt: Whether the object is a ground truth object. Ground truths are allowed
                  to have multiple valid options via tuples. Predictions are not allowed
                  to have tuples.
            comparator: Optional comparator for string comparison.

        Returns:
            Parent node of the ANLS tree.

        Raises:
            ValueError: If the object type is unsupported or if a tuple is used in a prediction.
        """
        # Import locally to avoid circular imports
        # These classes will be implemented in separate files
        from .tuple_tree import ANLSTuple
        from .list_tree import ANLSList
        from .dict_tree import ANLSDict
        from .none_tree import ANLSNone
        from .leaf_tree import ANLSLeaf

        if isinstance(obj, tuple):
            return ANLSTuple(obj, is_gt=is_gt, comparator=comparator)
        elif isinstance(obj, list):
            return ANLSList(obj, is_gt=is_gt, comparator=comparator)
        elif isinstance(obj, dict):
            return ANLSDict(obj, is_gt=is_gt, comparator=comparator)
        elif obj is None:
            return ANLSNone(comparator=comparator)
        elif isinstance(obj, (str, float, int, bool)):
            return ANLSLeaf(obj, comparator=comparator)
        else:
            # Handle StructuredModel objects by converting them to dictionaries
            if hasattr(obj, "model_dump") and callable(obj.model_dump):
                return ANLSDict(obj.model_dump(), is_gt=is_gt, comparator=comparator)
            else:
                raise ValueError(
                    f"Found unsupported type {type(obj)} for {obj} while creating ANLS tree"
                )

    def anls(
        self, other: "ANLSTree"
    ) -> tuple[float, "ANLSTree", list[dict[tuple[str, ...], float]]]:
        """Calculate the ANLS score between this tree and another tree.

        Args:
            other: The other ANLSTree to compare with.

        Returns:
            A tuple containing:
            - The ANLS score [0-1]
            - The closest ground truth tree
            - A list of key scores for detailed analysis
        """
        nls_list, closest_gt, key_scores = self.nls_list(other, (), [])
        length = self.pairwise_len(other)
        return (sum(nls_list) / length) if length > 0 else 1.0, closest_gt, key_scores

    def __str__(self) -> str:
        """Return a string representation of the ANLSTree."""
        return f"{self.__class__.__name__}({repr(self.obj)})"

    def __repr__(self) -> str:
        """Return a string representation of the ANLSTree."""
        return f"{self.__class__.__name__}({repr(self.obj)})"

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of this tree.

        Returns:
            The number of leaf nodes in this tree.
        """
        pass

    @abc.abstractmethod
    def pairwise_len(self, other: "ANLSTree") -> int:
        """Calculate the pairwise length between this tree and another tree.

        This is used to normalize the ANLS score.

        Args:
            other: The other ANLSTree to compare with.

        Returns:
            The pairwise length between the two trees.
        """
        pass

    @abc.abstractmethod
    def nls_list(
        self,
        other: "ANLSTree",
        key_hierarchy: tuple[str, ...],
        key_scores: list[dict[tuple[str, ...], float]],
    ) -> tuple[list[float], Any, list[dict[tuple[str, ...], float]]]:
        """Calculate the NLS scores between this tree and another tree.

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
        pass
