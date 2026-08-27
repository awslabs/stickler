"""Hungarian algorithm implementation for optimal assignment problems.

This module provides a Hungarian algorithm implementation for matching elements
between two lists, which is commonly used for evaluating list-type fields in
key information extraction tasks.
"""

import traceback
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from munkres import Munkres, make_cost_matrix

from stickler.comparators.base import BaseComparator
from stickler.utils.canonical import canonicalize_json

# Memory threshold for warning in MB
HUNGARIAN_SIZE_WARNING_THRESHOLD = 10000  # Matrix size (product of dimensions)

# Global Munkres instance for optimization
_MUNKRES = Munkres()


class HungarianMatcher:
    """Hungarian algorithm matcher for optimal assignment problems.

    This class implements the Hungarian algorithm for finding the optimal assignment
    between two lists of elements, using a specified comparator to determine similarity
    between pairs of elements.
    """

    def __init__(
        self,
        comparator: Optional[Union[BaseComparator, Callable]] = None,
        size_threshold: int = HUNGARIAN_SIZE_WARNING_THRESHOLD,
        normalize_values: bool = True,
        match_threshold: float = 0.7,
    ):
        """Initialize the Hungarian matcher.

        Args:
            comparator: Function or BaseComparator instance to determine similarity
                        between elements. If None, exact matching is used.
            size_threshold: Maximum allowable matrix size (rows*cols) before warning
            normalize_values: Whether to normalize string values before comparison
                             (convert strings to lowercase, strip whitespace, etc.)

                             Legacy, and deliberately declined on the
                             list-of-values path.
                             This predates comparators owning their own
                             normalization; since 0.7.0 they do
                             (``ExactComparator.case_sensitive``,
                             ``LevenshteinComparator._normalize``,
                             ``FuzzyComparator._normalize``), so normalizing
                             here silently overrides the comparator a field
                             declared. ``ComparisonHelper.compare_unordered_lists``
                             therefore passes ``False``. The default stays
                             ``True`` for direct callers who relied on it; do
                             not "fix" that call site back.

                             ``HungarianHelper``, which matches
                             ``List[StructuredModel]`` via
                             ``StructuredModelComparator``, still takes the
                             default: it matches whole objects rather than
                             values, so a field's declared comparator is not
                             what gets overridden. It does still map ``None`` to
                             ``""`` before scoring candidate pairs, which can
                             change which pairs the algorithm assigns.
            match_threshold: Minimum similarity score to consider a match as TP
        """
        self.comparator = comparator or (lambda x, y: float(x == y))
        self.size_threshold = size_threshold
        self.normalize_values = normalize_values
        self.match_threshold = match_threshold

    def _normalize_value(self, value: Any) -> Any:
        """Normalize a value to improve string matching.

        Only reached when ``normalize_values`` is true. Note what it does
        beyond case folding: it ``str()``-coerces every primitive and maps
        ``None`` to ``""``. Both are lossy for a comparator that inspects the
        value -- ``BBoxIoUComparator`` cannot parse ``"[0, 0, 10, 10]"``, and
        the ``""`` substitution bypasses ``BaseComparator``'s ``None`` policy.
        See the ``normalize_values`` note in :meth:`__init__` for why the
        evaluator path opts out.

        Args:
            value: Value to normalize

        Returns:
            Normalized value (string for primitives, unchanged for StructuredModels)
        """
        if value is None:
            return ""

        # Don't normalize StructuredModel objects - keep them as-is
        if hasattr(value, "compare") and callable(getattr(value, "compare")):
            return value

        # Convert to string for primitive types
        value_str = str(value)

        # Strip punctuation and extra spaces if required
        if self.normalize_values:
            # Simple normalization: lowercase, strip, collapse spaces
            value_str = " ".join(value_str.lower().strip().split())
            # Remove punctuation if needed
            # This could be enhanced based on specific requirements

        return value_str

    def _prepare_lists(self, list1: Any, list2: Any) -> Tuple[List[Any], List[Any]]:
        """Prepare input values for matching.

        Handles conversion of various input types to lists and normalizes values
        if needed.

        Args:
            list1: First list or value
            list2: Second list or value

        Returns:
            Tuple of (normalized list1, normalized list2)
        """
        # Convert string representation of lists if needed
        try:
            if isinstance(list1, str) and list1.startswith("[") and list1.endswith("]"):
                import ast

                list1 = ast.literal_eval(list1)
            if isinstance(list2, str) and list2.startswith("[") and list2.endswith("]"):
                import ast

                list2 = ast.literal_eval(list2)
        except (ValueError, SyntaxError):
            # Keep original values if parsing fails
            pass

        # Ensure inputs are lists
        if not isinstance(list1, list):
            list1 = [list1]
        if not isinstance(list2, list):
            list2 = [list2]

        # Normalize values if needed
        if self.normalize_values:
            list1 = [self._normalize_value(x) for x in list1]
            list2 = [self._normalize_value(x) for x in list2]

        return list1, list2

    @staticmethod
    def _comparable_form(item: Any) -> Any:
        """Canonical form for an item no comparator can handle.

        Only reached from :meth:`_score` after a comparator raised
        ``TypeError``. A ``dict`` has no comparator: ``LevenshteinComparator``
        raises for one, and ``str(dict)`` makes key order significant. Sorted-key
        JSON removes both problems and matches what ``stickler.auto`` already
        chooses for a dict field, so the explicit and zero-config paths agree.

        Everything else passes through untouched, so :meth:`_score` re-raises
        rather than scoring: a bounding box IS a list, and a comparator that
        raises on anything but a dict has a bug worth surfacing.
        """
        if isinstance(item, dict):
            return canonicalize_json(item)
        return item

    def _score(self, item1: Any, item2: Any) -> float:
        """Score one pair, canonicalizing dicts only if the comparator refuses.

        The comparator sees the raw item first, so a dict-aware comparator keeps
        working (#277). Canonicalization is the fallback, not a pre-filter. A
        ``TypeError`` from anything but a dict pair propagates: it means a
        comparator bug, not an uncomparable value.
        """
        compare = (
            self.comparator.compare
            if hasattr(self.comparator, "compare")
            else self.comparator
        )
        try:
            return compare(item1, item2)
        except TypeError:
            if not (isinstance(item1, dict) or isinstance(item2, dict)):
                raise
            return compare(self._comparable_form(item1), self._comparable_form(item2))

    def match(self, list1: Any, list2: Any) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Find optimal assignments between two lists.

        Performs Hungarian matching to find optimal assignment between elements
        in list1 and list2, using the provided comparator to determine similarity.

        Args:
            list1: First list
            list2: Second list

        Returns:
            Tuple of (matched_indices, similarity_matrix) where:
                - matched_indices is list of (i, j) pairs for matches
                - similarity_matrix is the calculated similarity matrix

            An empty input on either side gives an empty list and an empty
            array. :meth:`calculate_metrics` counts on that, so it is part of
            the contract rather than an error.

        Raises:
            Exception: For other errors during matching
        """
        # Handle case of empty lists
        if not list1 or not list2:
            return [], np.array([])

        # Proceed with Hungarian matching
        try:
            # Create similarity matrix
            similarity_matrix = np.zeros((len(list1), len(list2)))

            # Fill the matrix with similarity scores
            for i, item1 in enumerate(list1):
                for j, item2 in enumerate(list2):
                    similarity_matrix[i, j] = self._score(item1, item2)

            # Check matrix size
            matrix_size = len(list1) * len(list2)
            if matrix_size > self.size_threshold:
                print(
                    f"[Warning] Large matrix for Hungarian algorithm: {len(list1)}x{len(list2)} = {matrix_size}"
                )

            # Convert to cost matrix for the Hungarian algorithm
            # Cost is 1 - similarity (because Hungarian minimizes cost)
            cost_matrix = make_cost_matrix(similarity_matrix, lambda x: 1 - x)

            # Compute the optimal assignment
            matched_indices = _MUNKRES.compute(cost_matrix)

            # Clean up to help with memory usage
            del cost_matrix
            # Let Python's automatic garbage collection handle cleanup
            # Explicit gc.collect() was causing 97% performance overhead

            return matched_indices, similarity_matrix

        except Exception as e:
            print(f"Error in Hungarian matching: {str(e)}")
            traceback.print_exc()
            raise

    def calculate_metrics(self, list1: Any, list2: Any) -> dict:
        """Calculate matching metrics between two lists.

        The assignment decides what is paired. ``match_threshold`` then splits
        the paired items into TP and FD. It never puts a pair back into ``fn``
        or ``fa``, so only an item left with no partner is counted there.

        Args:
            list1: First list, typically ground truth. A bare value counts as
                a list of one, and a JSON string holding a list is parsed into
                one. Every count below is over the prepared list.
            list2: Second list, typically prediction, prepared the same way.

        Returns:
            The same nine keys for every input, empty lists included. Below,
            ``m`` and ``n`` are the two prepared lengths and ``k`` is the pair
            count, which is always ``min(m, n)``.

                matched_pairs: ``k`` tuples of ``(i, j, score)``
                tp: pairs scoring at or above ``match_threshold``
                fa: predictions with no partner, ``n`` minus ``k``
                fd: pairs scoring below ``match_threshold``
                fp: the rollup ``fd`` plus ``fa``, equal to ``n`` minus ``tp``
                fn: ground truth items with no partner, ``m`` minus ``k``
                precision: ``tp / (tp + fp)``, and 1.0 if both lists are empty
                recall: ``tp / (tp + fn + fd)``, and 1.0 if list1 is empty
                f1: the harmonic mean of precision and recall

        Note:
            ``recall`` counts an FD against the score.
            :meth:`MetricsHelper.calculate_derived_metrics` defaults to the
            other convention, so the same counts read lower here than they do
            as ``cm_recall`` there.

            The threshold test here is a bare ``>=``. ``ThresholdHelper``,
            ``ComparisonHelper`` and ``ConfusionMatrixCalculator`` use a
            tolerant test that accepts a score up to ``1e-10`` under the
            threshold, so a score inside that window is a TP there and an FD
            here.

            ``match_threshold=0.0`` is used elsewhere as a capture all
            sentinel. Every score satisfies ``>= 0.0``, so ``tp`` then counts
            pairs rather than true positives and must not be read.
        """
        prepared_list1, prepared_list2 = self._prepare_lists(list1, list2)
        m, n = len(prepared_list1), len(prepared_list2)

        if m == 1 and n == 1:
            # Fast path for the only assignment there can be, which skips
            # match() and the solver. #224: a one item list has to classify
            # exactly like a longer one, so this branch produces the pair and
            # stops. The counting below is then shared, not repeated.
            score = self._score(prepared_list1[0], prepared_list2[0])
            matched_pairs = [(0, 0, score)]
        elif m == 0 or n == 0:
            matched_pairs = []
        else:
            matched_indices, similarity_matrix = self.match(
                prepared_list1, prepared_list2
            )
            matched_pairs = [
                (i, j, similarity_matrix[i, j]) for i, j in matched_indices
            ]

        # match() assigns exactly min(m, n) pairs, so the four counts below
        # cover max(m, n) items once each. The threshold splits the pairs into
        # tp and fd. It does not un-match them, which is why a paired item
        # never reaches fn or fa.
        tp = sum(1 for _, _, score in matched_pairs if score >= self.match_threshold)
        k = len(matched_pairs)
        fd, fn, fa = k - tp, m - k, n - k
        fp = fd + fa

        # tp + fp == n and tp + fn + fd == m, so the documented rates can use
        # the two lengths as their denominators. A side with no items has
        # nothing to be right or wrong about, so its rate is 1.0.
        precision = tp / n if n else (1.0 if m == 0 else 0.0)
        recall = tp / m if m else 1.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

        return {
            "matched_pairs": matched_pairs,
            "tp": tp,
            "fa": fa,
            "fd": fd,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def __call__(self, list1: Any, list2: Any) -> Tuple[int, int]:
        """Legacy interface for compatibility with traditional evaluator.

        Returns only tp and fp counts, which is the format expected by the
        traditional Hungarian class.

        Args:
            list1: First list (ground truth)
            list2: Second list (prediction)

        Returns:
            Tuple of (tp, fp) counts
        """
        metrics = self.calculate_metrics(list1, list2)
        return metrics["tp"], metrics["fp"]

    def binary_compare(self, list1: Any, list2: Any) -> Tuple[int, int]:
        """Utility method for binary comparison, aliases __call__ method.

        This method supports the binary comparison interface used by other comparators
        and returns true positives and false positives as counts.

        Args:
            list1: First list (ground truth)
            list2: Second list (prediction)

        Returns:
            Tuple of (tp, fp) counts
        """
        return self.__call__(list1, list2)
