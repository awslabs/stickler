"""Hungarian algorithm implementation for optimal assignment problems.

This module provides a Hungarian algorithm implementation for matching elements
between two lists, which is commonly used for evaluating list-type fields in
key information extraction tasks.
"""

import gc
import traceback
import numpy as np
from typing import Any, List, Tuple, Callable, Optional, Union
from munkres import Munkres, make_cost_matrix

from stickler.comparators.base import BaseComparator

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
            match_threshold: Minimum similarity score to consider a match as TP
        """
        self.comparator = comparator or (lambda x, y: float(x == y))
        self.size_threshold = size_threshold
        self.normalize_values = normalize_values
        self.match_threshold = match_threshold

    def _normalize_value(self, value: Any) -> Any:
        """Normalize a value to improve string matching.

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

        Raises:
            ValueError: If input lists are empty
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
                    # Handle callable function or object with compare method
                    if hasattr(self.comparator, "compare"):
                        similarity_matrix[i, j] = self.comparator.compare(item1, item2)
                    else:
                        similarity_matrix[i, j] = self.comparator(item1, item2)

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
            gc.collect()

            return matched_indices, similarity_matrix

        except Exception as e:
            print(f"Error in Hungarian matching: {str(e)}")
            traceback.print_exc()
            raise

    def calculate_metrics(self, list1: Any, list2: Any) -> dict:
        """Calculate matching metrics between two lists.

        Uses Hungarian matching to find optimal assignments and calculates
        metrics such as true positives, false positives, and false negatives.

        Args:
            list1: First list (typically ground truth)
            list2: Second list (typically prediction)

        Returns:
            Dictionary with metrics:
                - matched_pairs: List of (i, j, score) tuples for matches
                - tp: Count of true positives (good matches)
                - fp: Count of false positives (extra or wrong items in list2)
                - fn: Count of false negatives (missing items from list1)
                - precision: Precision score
                - recall: Recall score
                - f1: F1 score
        """
        # Prepare lists
        prepared_list1, prepared_list2 = self._prepare_lists(list1, list2)

        # Handle simple case efficiently: single items
        if len(prepared_list1) == 1 and len(prepared_list2) == 1:
            # Directly compare the single items
            if hasattr(self.comparator, "compare"):
                score = self.comparator.compare(prepared_list1[0], prepared_list2[0])
            else:
                score = self.comparator(prepared_list1[0], prepared_list2[0])

            if score > 0:
                return {
                    "matched_pairs": [(0, 0, score)],
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                }
            else:
                return {
                    "matched_pairs": [],
                    "tp": 0,
                    "fp": 1,
                    "fn": 1,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }

        # Handle empty lists
        if not prepared_list1 and not prepared_list2:
            return {
                "matched_pairs": [],
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
        elif not prepared_list1:
            return {
                "matched_pairs": [],
                "tp": 0,
                "fp": len(prepared_list2),
                "fn": 0,
                "precision": 0.0,
                "recall": 1.0,
                "f1": 0.0,
            }
        elif not prepared_list2:
            return {
                "matched_pairs": [],
                "tp": 0,
                "fp": 0,
                "fn": len(prepared_list1),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        # Get matched indices and similarity matrix
        matched_indices, similarity_matrix = self.match(prepared_list1, prepared_list2)

        # Calculate matched pairs with scores
        matched_pairs = []
        tp = 0
        for i, j in matched_indices:
            score = similarity_matrix[i, j]
            matched_pairs.append((i, j, score))
            # Only count as true positive if score meets threshold
            if score >= self.match_threshold:
                tp += 1

        # Calculate metrics
        fp = len(prepared_list2) - tp
        fn = len(prepared_list1) - tp

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "matched_pairs": matched_pairs,
            "tp": tp,
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
