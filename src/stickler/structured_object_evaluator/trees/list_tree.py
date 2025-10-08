"""List node implementation for the ANLS* tree."""

from typing import Any, Optional, Tuple, List as PyList, Dict, cast

from munkres import Munkres, make_cost_matrix
from stickler.comparators.base import BaseComparator
from .base import ANLSTree


class ANLSList(ANLSTree):
    """List node for representing sequences in the ANLS tree.

    This class handles lists, which represent ordered collections of items
    like elements in an array or items in a document. The Hungarian algorithm
    is used to find the optimal matching between ground truth and prediction lists.

    Attributes:
        obj: The original list.
        tree: A list of ANLSTree nodes representing each item in the list.
        _comparator: The comparator used for string similarity.
    """

    def __init__(
        self, obj: Any, *, is_gt: bool, comparator: Optional[BaseComparator] = None
    ):
        """Initialize a list node.

        Args:
            obj: The list object.
            is_gt: Whether this is a ground truth node.
            comparator: Optional comparator for string comparison.

        Raises:
            ValueError: If obj is not a list.
        """
        if not isinstance(obj, list):
            raise ValueError(f"ANLSList expects a list, got {type(obj)}")

        super().__init__(obj, comparator)
        self.tree: PyList[ANLSTree] = [
            ANLSTree.make_tree(x, is_gt=is_gt, comparator=comparator) for x in obj
        ]

    def __len__(self) -> int:
        """Return the length of this list.

        Returns:
            The sum of the lengths of all items in the list.
        """
        return sum(len(x) for x in self.tree)

    def _hungarian(
        self,
        other: "ANLSList",
        key_hierarchy: Tuple[str, ...],
        key_scores: PyList[Dict[Tuple[str, ...], float]],
    ):
        """
        Perform Hungarian algorithm matching between self and other ANLSList.

        This method computes the optimal matching between elements of self and other,
        using the Hungarian algorithm to minimize the total cost (maximize similarity).

        Args:
            other: The other ANLSList to match against.
            key_hierarchy: The current key hierarchy for nested structures.
            key_scores: List to store key-wise scores.

        Returns:
            tuple: A tuple containing:
                - mat: Matrix of NLS scores for each pair.
                - gts: Matrix of chosen ground truths for each pair.
                - indexes: Optimal matching indexes from Hungarian algorithm.
                - key_scores_mat: Matrix of key scores for each pair.
        """
        mat: PyList[PyList[PyList[float]]] = []
        avg_mat: PyList[PyList[float]] = []
        gts: PyList[PyList[Any]] = []
        key_scores_mat: PyList[PyList[PyList[Dict[Tuple[str, ...], float]]]] = []

        # Compute NLS scores and averages for all pairs of elements
        for gt in self.tree:
            row = []
            avg_row = []
            gts_row = []
            ks_row: PyList[PyList[Dict[Tuple[str, ...], float]]] = []
            for pred in other.tree:
                key_scores_copy = key_scores.copy()
                nls_list, chosen_gt, new_key_scores = gt.nls_list(
                    pred, key_hierarchy, key_scores_copy
                )
                length = gt.pairwise_len(pred)
                row.append(nls_list)
                avg = (sum(nls_list) / length) if length > 0 else 1.0
                if pred.obj == chosen_gt:
                    # Slightly favor exact matches to break ties in the Hungarian algorithm
                    avg = 1 + 1e-10  # Use math.nextafter if available
                avg_row.append(avg)
                gts_row.append(chosen_gt)
                ks_row.append(new_key_scores)
            mat.append(row)
            avg_mat.append(avg_row)
            gts.append(gts_row)
            key_scores_mat.append(ks_row)

        # Check for empty lists - Munkres fails on empty
        if len(mat) == 0 or len(mat[0]) == 0:
            return mat, gts, [], []

        # Run Hungarian algorithm
        m = Munkres()
        m_cost_matrix = make_cost_matrix(avg_mat)
        indexes = m.compute(m_cost_matrix)
        indexes = cast(PyList[Tuple[int, int]], indexes)
        return mat, gts, indexes, key_scores_mat

    def pairwise_len(self, other: ANLSTree) -> int:
        """Calculate the pairwise length between this list and another tree.

        Args:
            other: The other ANLSTree to compare with.

        Returns:
            The pairwise length between the two trees.
        """
        if not isinstance(other, ANLSList):
            return max(len(self), len(other))

        _, _, indexes, _ = self._hungarian(other, (), [])

        # Identify elements that weren't matched
        not_selected_self = {*range(len(self.tree))} - {row for row, _ in indexes}
        not_selected_other = {*range(len(other.tree))} - {col for _, col in indexes}

        # Calculate the pairwise length
        pwl = sum(self.tree[row].pairwise_len(other.tree[col]) for row, col in indexes)
        pwl += sum(len(self.tree[i]) for i in not_selected_self)
        pwl += sum(len(other.tree[j]) for j in not_selected_other)
        return pwl

    def nls_list(
        self,
        other: ANLSTree,
        key_hierarchy: Tuple[str, ...],
        key_scores: PyList[Dict[Tuple[str, ...], float]],
    ) -> Tuple[PyList[float], Any, PyList[Dict[Tuple[str, ...], float]]]:
        """Calculate the NLS score between this list and another tree.

        Args:
            other: The other ANLSTree to compare with.
            key_hierarchy: The current key hierarchy for nested structures.
            key_scores: A list to store key-wise scores.

        Returns:
            A tuple containing:
            - A list of NLS scores
            - The closest ground truth list
            - An updated list of key scores
        """
        key_scores = key_scores.copy()

        # If 'other' is not an ANLSList, return a default score of 0.0
        if not isinstance(other, ANLSList):
            return [0.0], self.obj, key_scores

        # Perform Hungarian algorithm matching
        mat, gts, indexes, key_scores_mat = self._hungarian(
            other, key_hierarchy, key_scores
        )

        # Extract NLS values for matched pairs
        values = [mat[row][column] for row, column in indexes]
        values = [item for sublist in values for item in sublist]  # Flatten the list

        # Process chosen ground truths
        chosen_gt_with_idx = [(gts[row][col], col) for row, col in indexes]
        chosen_gt_with_idx.sort(key=lambda x: x[1])  # Sort by column index
        chosen_gt = [gt for gt, idx in chosen_gt_with_idx]

        # Add ground truths for unmatched rows
        not_selected_rows = [
            i for i in range(len(self.tree)) if i not in {row for row, _ in indexes}
        ]
        chosen_gt.extend(self.tree[i].obj for i in not_selected_rows)

        # Process chosen key scores
        chosen_key_scores_with_idx = [
            (key_scores_mat[row][col], col) for row, col in indexes
        ]
        chosen_key_scores_with_idx.sort(key=lambda x: x[1])  # Sort by column index
        chosen_key_scores = [ks for ks, idx in chosen_key_scores_with_idx]

        # Flatten the chosen key scores
        flattened_chosen_key_scores: PyList[Dict[Tuple[str, ...], float]] = []
        for ks in chosen_key_scores:
            flattened_chosen_key_scores.extend(ks)

        return values, chosen_gt, flattened_chosen_key_scores
