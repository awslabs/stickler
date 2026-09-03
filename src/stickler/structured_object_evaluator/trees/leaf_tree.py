"""Leaf node implementation for the ANLS* tree."""

from typing import Any, Dict, List, Optional, Tuple

from stickler.comparators.anls import _UNSCOREABLE
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

        # Canonical ANLS*: every leaf is compared as text, whatever its Python
        # type. ANLS came from scene-text VQA, where every answer IS a string.
        #
        # The consequence is real and is documented rather than patched: incidental
        # character overlap on a long numeric or identifier value scores high, and
        # scores ABOVE a genuine text near-miss.
        #
        #     account number, 1 of 22 chars   0.9545
        #     date one day off            0.9000
        #     amount 2x wrong             0.8571
        #     "Acme Corporation"/"Acme Corp"   0.5625
        #
        # So no `leaf_threshold` separates them: a cutoff high enough to reject the
        # account number also rejects the partial credit this exists to award. A
        # field whose values you care about should be declared, where it gets a
        # comparator chosen for its type.
        #
        # FUTURE: `ANLSStarComparator(infer_types=True)` would make leaves
        # type-aware. It must delegate to `stickler.auto.inference`, NOT define its
        # own rules: a dict key is a field name, so `{"amount": 1000}` would route
        # through `infer_field_config("amount", ...)` and get the same
        # NumericComparator@0.95 with relative_tolerance a *declared* `amount:
        # float` gets. Hand-rolled leaf rules would be a second inference table
        # contradicting the first -- exactly the divergence
        # https://github.com/awslabs/stickler/issues/239 exists to remove.
        #
        # The open question that blocks it: inference needs a type, and the two
        # sides can disagree about theirs (ground truth `Decimal("10.50")` against
        # a prediction's string `"10.50"`), so something has to decide which side is
        # authoritative. That is coercion, and it is unresolved in
        # https://github.com/awslabs/stickler/issues/49.
        # A value with no JSON representation is outside ANLS*'s domain, so it is
        # refused rather than scored. `ANLSStarComparator` marks such a value
        # instead of inventing text for it; see `_UNSCOREABLE` there for what
        # inventing text cost (two unrelated objects scored 0.8684, because the
        # invented text was a memory address).
        #
        # Short-circuited here rather than left to the metric: two identical
        # markers would score 1.0, so every out-of-domain value would match every
        # other. Refusing is also why equality is not consulted -- an out-of-domain
        # value is not scored, whether or not the two happen to be equal.
        if _UNSCOREABLE in (this_str, other_str):
            return [0.0], self.obj, key_scores_copy

        similarity = self._comparator.compare(this_str, other_str)

        # Apply the ANLS threshold
        question_result = 0.0 if similarity < self.threshold else similarity

        return [question_result], self.obj, key_scores_copy
