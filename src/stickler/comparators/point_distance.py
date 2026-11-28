"""Point comparator using Euclidean distance.

This comparator compares two 2D or 3D points and scores them by how far
apart they are, relative to a caller-supplied ``max_distance``.
"""

import math
import numbers
from typing import Any, Dict, Optional, Tuple

from stickler.comparators.base import BaseComparator


class PointDistanceComparator(BaseComparator):
    """Comparator for points using Euclidean distance.

    Scores two points as ``max(0.0, 1.0 - distance / max_distance)``: 1.0 for
    identical points, falling linearly to 0.0 at ``max_distance`` and beyond.
    The match gate is therefore a radius of ``max_distance * (1 - threshold)``
    around the ground truth, not the box that per-axis ``NumericComparator``
    tolerances give.

    Point formats accepted:
        - 2D: [x, y]
        - 3D: [x, y, z]

    Lists, tuples and 1D NumPy arrays are accepted. Both points must have the
    same number of coordinates; a 2D point against a 3D point is malformed
    input and scores 0.0. Coordinates must be finite ``numbers.Real`` values,
    which includes NumPy scalars and ``Fraction``. Strings, ``Decimal`` (not
    a ``numbers.Real``), non-finite values (NaN, inf) and values too large
    for a float score 0.0. Booleans are accepted as coordinates, as in
    ``BBoxIoUComparator`` (``bool`` is a subclass of ``int``: ``True`` == 1,
    ``False`` == 0), so guard upstream if that is not intended.

    On a ``StructuredModel``, annotate a single point as a ``Tuple`` such as
    ``Tuple[float, float, float]``. A ``list`` or ``List[float]`` point field
    always scores 0.0, even against an identical point, because the
    comparator never sees the whole point. A list of points is
    ``List[List[float]]`` or ``List[Tuple[float, float, float]]``.

    Args:
        max_distance: Distance at which the score reaches 0.0, in the units of
            the coordinates. Must be a finite number greater than 0.
        threshold: Score threshold for binary match classification
            (default: 0.5, a match within half of ``max_distance``).

    Raises:
        ValueError: If ``max_distance`` is not a finite number greater than 0.

    Example:
        >>> from stickler.comparators import PointDistanceComparator
        >>> cmp = PointDistanceComparator(max_distance=10.0)
        >>> cmp.compare([0, 0], [3, 4])
        0.5
        >>> cmp.compare([0, 0, 0], [0, 0, 20])
        0.0
    """

    # Same default as BBoxIoUComparator, which reads here as a match radius of
    # half of max_distance.
    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        max_distance: float,
        threshold: Optional[float] = None,
    ):
        # A bool, NaN, inf or non-positive max_distance would make every score
        # 0.0, 1.0 or undefined, so reject it instead of scoring with it.
        is_number = isinstance(max_distance, (int, float)) and not isinstance(
            max_distance, bool
        )
        try:
            max_distance_float = float(max_distance) if is_number else math.nan
        except OverflowError:
            # An int beyond float range, such as 10**400.
            max_distance_float = math.inf
        if not math.isfinite(max_distance_float) or max_distance_float <= 0:
            raise ValueError(
                f"max_distance must be a finite number greater than 0; "
                f"got {max_distance!r}"
            )
        super().__init__(threshold=threshold)
        self.max_distance = max_distance_float

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Round-trippable config for JSON-schema export.

        ``max_distance`` has no default, so it is always emitted; without it an
        exported schema could not be imported again.
        """
        return {"max_distance": self.max_distance}

    def _compare(self, point1: Any, point2: Any) -> float:
        """Compare two points and return a distance-based score.

        Args:
            point1: First point (prediction).
            point2: Second point (ground truth).

        Returns:
            Score between 0.0 and 1.0.
        """
        coords1 = self._normalize_point(point1)
        coords2 = self._normalize_point(point2)

        if coords1 is None or coords2 is None or len(coords1) != len(coords2):
            return 0.0

        distance = math.dist(coords1, coords2)
        return max(0.0, 1.0 - distance / self.max_distance)

    @staticmethod
    def _normalize_point(point: Any) -> Optional[Tuple[float, ...]]:
        """Normalize a point to a tuple of 2 or 3 finite floats.

        Returns:
            The coordinates, or None if the input is invalid.
        """
        # Duck-typed so a NumPy array is accepted without importing NumPy.
        if not isinstance(point, (list, tuple)) and callable(
            getattr(point, "tolist", None)
        ):
            point = point.tolist()
        if not isinstance(point, (list, tuple)) or len(point) not in (2, 3):
            return None
        if not all(isinstance(v, numbers.Real) for v in point):
            return None

        try:
            coords = tuple(float(v) for v in point)
        except OverflowError:
            # An int beyond float range, such as 10**400.
            return None
        # Reject non-finite coordinates (NaN, inf) as malformed input so they
        # score as a miss rather than poisoning the distance.
        if not all(math.isfinite(v) for v in coords):
            return None
        return coords
