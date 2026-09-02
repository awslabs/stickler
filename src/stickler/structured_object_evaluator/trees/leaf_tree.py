"""Leaf node implementation for the ANLS* tree."""

import re
from typing import Any, Dict, List, Optional, Tuple

from stickler.comparators.base import BaseComparator

from .base import ANLSTree

# The canonical ISO-8601 date and datetime forms, which is what
# `to_jsonable_python` produces for a `date`/`datetime` before this tree sees it.
# Matched against the already-lowercased leaf string, hence [t] not [T].
# Deliberately not general date parsing: the job is to recognise a type signal
# this layer destroyed, not to guess whether arbitrary text is a date.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([t ][\d:.+\-z]*)?$")


class ANLSLeaf(ANLSTree):
    """Leaf node for primitive values in the ANLS tree.

    This class represents leaf nodes in the ANLSTree for primitive types
    (strings, numbers, booleans). It compares values using the provided
    string comparator.

    Attributes:
        obj: The primitive value represented by this leaf node.
        _comparator: The comparator used for string similarity.
    """

    def __init__(
        self,
        obj: Any,
        comparator: Optional[BaseComparator] = None,
        threshold: Optional[float] = None,
    ):
        """Initialize a leaf node.

        Args:
            obj: The primitive value (str, float, int, bool).
            comparator: Optional comparator for string comparison.
            threshold: Tau, the cutoff below which this leaf's similarity
                is discarded as noise.

        Raises:
            ValueError: If obj is not a primitive type.
        """
        if not isinstance(obj, (str, float, int, bool)):
            raise ValueError(f"Leaf must be a primitive type, got {type(obj)}")
        super().__init__(obj, comparator, threshold)

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

        # Edit distance is a STRING metric, so it is only applied when both
        # leaves are strings. Character overlap between two numbers is not
        # partial correctness: `1000` against `9000` shares three of four
        # characters and would score 0.75, which then clears a 0.7 field
        # threshold and reports a nine-fold error as a true positive. The same
        # applies to dates, where `2024-01-01` against `2024-11-11` scored 0.80.
        #
        # A non-string leaf is therefore right or wrong, judged on its canonical
        # string form so that a value which crossed a JSON boundary still
        # matches its origin: `1000` and `"1000"` are the same extraction, which
        # is what keeps `model_dump()` and `model_dump(mode="json")` equivalent.
        #
        # Upstream anls_star stringifies everything and has the same flaw. The
        # divergence is deliberate, and narrows the metric to where it means
        # something rather than adding machinery.
        both_strings = isinstance(self.obj, str) and isinstance(other.obj, str)
        # A pair of ISO timestamps is a date comparison wearing a string, not
        # text. `datetime.date` reaches this leaf already serialised to ISO by
        # `to_jsonable_python`, so treating it as text would score a wrong date
        # on character overlap: `2024-01-01` against `2024-11-11` is 0.80, which
        # clears a 0.7 field threshold and reports a wrong date as a match.
        both_iso_dates = (
            both_strings
            and _ISO_DATE.match(this_str) is not None
            and _ISO_DATE.match(other_str) is not None
        )
        if both_strings and not both_iso_dates:
            similarity = self._comparator.compare(this_str, other_str)
        else:
            similarity = 1.0 if this_str == other_str else 0.0

        # Apply the ANLS threshold
        question_result = 0.0 if similarity < self.threshold else similarity

        return [question_result], self.obj, key_scores_copy
