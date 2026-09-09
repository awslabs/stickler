"""Bounding box comparator using Intersection over Union (IoU).

This comparator compares two bounding boxes and returns their IoU score
as the similarity measure. It supports both two-point format
([[x1, y1], [x2, y2]]) and four-element flat format ([x1, y1, x2, y2]).
"""

import math
from typing import Any, Optional, Tuple

from stickler.comparators.base import BaseComparator


class BBoxIoUComparator(BaseComparator):
    """Comparator for bounding boxes using Intersection over Union.

    Compares two bounding boxes and returns their IoU as a similarity
    score between 0.0 and 1.0.

    Bounding box formats accepted:
        - Two-point: [[x1, y1], [x2, y2]]
        - Flat: [x1, y1, x2, y2]

    Coordinates must be finite numbers; non-finite values (NaN, inf) are
    treated as malformed input and score 0.0. Booleans are accepted as
    coordinates (``bool`` is a subclass of ``int``: ``True`` == 1, ``False``
    == 0), so guard upstream if that is not intended. Note that a zero-area
    box (a point, e.g. ``[[5, 5], [5, 5]]``) has no area to intersect, so it
    scores IoU 0.0 even against an identical point — relevant when annotating
    point locations rather than regions.

    Args:
        threshold: IoU threshold for binary match classification (default: 0.5).

    Example:
        >>> from stickler.comparators.bbox import BBoxIoUComparator
        >>> cmp = BBoxIoUComparator(threshold=0.5)
        >>> cmp.compare([[0, 0], [10, 10]], [[0, 0], [10, 10]])
        1.0
        >>> cmp.compare([[0, 0], [5, 5]], [[5, 5], [10, 10]])
        0.0

    .. versionchanged:: 0.7.0
        Usable on a list-of-boxes field. Previously every ``List[bbox]`` field
        scored ``0.0``, even against identical input: the evaluator stringified
        each item before comparison and ``"[0, 0, 10, 10]"`` is not a list, so
        no box could be parsed.
    """

    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        threshold: Optional[float] = None,
    ):
        super().__init__(threshold=threshold)

    def _compare(self, bbox1: Any, bbox2: Any) -> float:
        """Compare two bounding boxes and return their IoU.

        Args:
            bbox1: First bounding box (prediction).
            bbox2: Second bounding box (ground truth).

        Returns:
            IoU score between 0.0 and 1.0.
        """
        coords1 = self._normalize_bbox(bbox1)
        coords2 = self._normalize_bbox(bbox2)

        if coords1 is None or coords2 is None:
            return 0.0

        return self._compute_iou(coords1, coords2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_bbox(
        bbox: Any,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Normalize a bounding box to (x1, y1, x2, y2) with x1<=x2, y1<=y2.

        Accepts:
            - [[x1, y1], [x2, y2]]
            - [x1, y1, x2, y2]

        Returns:
            (x_min, y_min, x_max, y_max) or None if the input is invalid.
        """
        try:
            if not isinstance(bbox, (list, tuple)):
                return None

            if len(bbox) == 2 and all(
                isinstance(p, (list, tuple)) and len(p) == 2 for p in bbox
            ):
                # Two-point format: [[x1, y1], [x2, y2]]
                x1, y1 = float(bbox[0][0]), float(bbox[0][1])
                x2, y2 = float(bbox[1][0]), float(bbox[1][1])
            elif len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                # Flat format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = (float(v) for v in bbox)
            else:
                return None

            # Reject non-finite coordinates (NaN, inf) as malformed input so
            # they score as a miss rather than poisoning IoU output.
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                return None

            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _compute_iou(
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two normalized boxes (x1, y1, x2, y2).

        Args:
            box1: (x_min, y_min, x_max, y_max)
            box2: (x_min, y_min, x_max, y_max)

        Returns:
            IoU value between 0.0 and 1.0.
        """
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        inter_area = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area
