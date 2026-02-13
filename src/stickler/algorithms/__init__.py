"""Common algorithms for key information evaluation.

This package contains algorithms that are shared between the traditional
and ANLS Star evaluation systems.
"""

from stickler.algorithms.hungarian import (
    HUNGARIAN_SIZE_WARNING_THRESHOLD,
    HungarianMatcher,
)

__all__ = ["HungarianMatcher", "HUNGARIAN_SIZE_WARNING_THRESHOLD"]
