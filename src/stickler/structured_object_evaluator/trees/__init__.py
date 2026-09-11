"""Tree module for structured object evaluation.

This module provides tree-based representations for structured objects
to support ANLS* evaluation.

ANLS* generalizes ANLS (Average Normalized Levenshtein Similarity, a string
metric from the scene-text VQA literature) to hierarchical values: leaves are
compared as strings, container scores are normalized over the union of both
sides, and list elements are paired optimally rather than positionally.

Attribution
-----------
The ANLS* metric, and the approach of walking structured output whose shape is
not declared in order to score it, are taken from the ``anls_star`` project
(https://pypi.org/project/anls_star/), Apache-2.0. See the repository NOTICE.

One divergence: the per-leaf cutoff (tau) is a parameter here rather than a fixed
constant, because whether a near match at a leaf should earn partial credit
depends on the data being evaluated. It threads through
:meth:`ANLSTree.make_tree`, ``anls_score(..., threshold=...)`` and
:class:`~stickler.comparators.anls.ANLSStarComparator`, defaulting to
:attr:`ANLSTree.THRESHOLD` in the first two so existing behaviour is unchanged.
"""

from .base import ANLSTree
from .dict_tree import ANLSDict
from .leaf_tree import ANLSLeaf
from .list_tree import ANLSList
from .none_tree import ANLSNone
from .tuple_tree import ANLSTuple

__all__ = [
    "ANLSTree",
    "ANLSDict",
    "ANLSLeaf",
    "ANLSList",
    "ANLSNone",
    "ANLSTuple",
]
