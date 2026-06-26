"""Bounding box comparator using Intersection over Union (IoU).

This comparator compares two bounding boxes and returns their IoU score
as the similarity measure. It supports both two-point format
([[x1, y1], [x2, y2]]) and four-element flat format ([x1, y1, x2, y2]), each
optionally carrying a trailing page number ([[x1, y1], [x2, y2], page] or
[x1, y1, x2, y2, page]).
"""

import math
from typing import Any, Optional, Tuple

from stickler.comparators.base import BaseComparator

# Coordinate 4-tuple (x_min, y_min, x_max, y_max).
_Coords = Tuple[float, float, float, float]

# Sentinel marking "no page element was present" (distinct from page == None,
# which this code never produces, and from a malformed page).
_MISSING = object()


def _parse_page(value: Any) -> int:
    """Coerce a trailing page element to an int, raising on anything else.

    Lenient about integer-valued floats (``2.0`` -> ``2``) but rejects
    non-integer floats, non-finite values, strings, and bools so a malformed
    page is treated as a malformed box (scores 0.0) rather than silently
    matching. ``bool`` is rejected explicitly (it is an ``int`` subclass, but a
    boolean page number is nonsensical).
    """
    if isinstance(value, bool):
        raise ValueError("page number must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise ValueError(f"invalid page number: {value!r}")


class BBoxIoUComparator(BaseComparator):
    """Comparator for bounding boxes using Intersection over Union.

    Compares two bounding boxes and returns their IoU as a similarity
    score between 0.0 and 1.0.

    Bounding box formats accepted:
        - Two-point: [[x1, y1], [x2, y2]]
        - Flat: [x1, y1, x2, y2]
        - Two-point with page: [[x1, y1], [x2, y2], page]
        - Flat with page: [x1, y1, x2, y2, page]

    Coordinates must be finite numbers; non-finite values (NaN, inf) are
    treated as malformed input and score 0.0. Booleans are accepted as
    coordinates (``bool`` is a subclass of ``int``: ``True`` == 1, ``False``
    == 0), so guard upstream if that is not intended. Note that a zero-area
    box (a point, e.g. ``[[5, 5], [5, 5]]``) has no area to intersect, so it
    scores IoU 0.0 even against an identical point — relevant when annotating
    point locations rather than regions.

    The optional trailing page number identifies which page of a multi-page
    document a box lives on. It must be an integer (an integer-valued float
    like ``2.0`` is coerced to ``2``); a non-integer, non-finite, or
    non-numeric page makes the whole box malformed (scores 0.0). The page is
    ignored unless ``page_aware=True``.

    Args:
        threshold: IoU threshold for binary match classification (default: 0.5).
        page_aware: When True, a box MUST declare its page and the two pages
            must match for IoU to be computed; otherwise the comparison scores
            0.0. This means a box with no page number (a two- or four-element
            box) is wrong 100% of the time in page-aware mode — even against
            another page-less box — and boxes on different pages never match.
            When False (default), page numbers are parsed but ignored, so the
            page suffix is fully backward compatible and opt-in.

    Example:
        >>> from stickler.comparators.bbox import BBoxIoUComparator
        >>> cmp = BBoxIoUComparator(threshold=0.5)
        >>> cmp.compare([[0, 0], [10, 10]], [[0, 0], [10, 10]])
        1.0
        >>> cmp.compare([[0, 0], [5, 5]], [[5, 5], [10, 10]])
        0.0
        >>> paged = BBoxIoUComparator(threshold=0.5, page_aware=True)
        >>> paged.compare([[0, 0], [10, 10], 1], [[0, 0], [10, 10], 2])
        0.0
    """

    def __init__(
        self,
        threshold: float = 0.5,
        page_aware: bool = False,
    ):
        super().__init__(threshold=threshold)
        self.page_aware = page_aware

    def compare(self, bbox1: Any, bbox2: Any) -> float:
        """Compare two bounding boxes and return their IoU.

        Args:
            bbox1: First bounding box (prediction).
            bbox2: Second bounding box (ground truth).

        Returns:
            IoU score between 0.0 and 1.0.
        """
        if bbox1 is None and bbox2 is None:
            return 1.0
        if bbox1 is None or bbox2 is None:
            return 0.0

        norm1 = self._normalize_bbox(bbox1)
        norm2 = self._normalize_bbox(bbox2)

        if norm1 is None or norm2 is None:
            return 0.0

        coords1, page1 = norm1
        coords2, page2 = norm2

        # Page-aware short-circuit: in page-aware mode a box MUST declare its
        # page. A box with no page is wrong 100% of the time (even against
        # another page-less box), and two boxes on different pages never match,
        # regardless of how well their coordinates align.
        if self.page_aware:
            if page1 is _MISSING or page2 is _MISSING or page1 != page2:
                return 0.0

        return self._compute_iou(coords1, coords2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_bbox(
        bbox: Any,
    ) -> Optional[Tuple[_Coords, Any]]:
        """Normalize a bounding box to ((x1, y1, x2, y2), page) with x1<=x2, y1<=y2.

        Accepts:
            - [[x1, y1], [x2, y2]]                 (page = _MISSING)
            - [x1, y1, x2, y2]                     (page = _MISSING)
            - [[x1, y1], [x2, y2], page]
            - [x1, y1, x2, y2, page]

        Returns:
            ((x_min, y_min, x_max, y_max), page) where page is an int or the
            ``_MISSING`` sentinel, or None if the input is invalid. A malformed
            page (non-integer, non-finite, non-numeric) makes the whole box
            invalid.
        """
        try:
            if not isinstance(bbox, (list, tuple)):
                return None

            page: Any = _MISSING
            if len(bbox) in (2, 3) and all(
                isinstance(p, (list, tuple)) and len(p) == 2 for p in bbox[:2]
            ):
                # Two-point format: [[x1, y1], [x2, y2]] (+ optional page)
                x1, y1 = float(bbox[0][0]), float(bbox[0][1])
                x2, y2 = float(bbox[1][0]), float(bbox[1][1])
                if len(bbox) == 3:
                    page = _parse_page(bbox[2])
            elif len(bbox) in (4, 5) and all(
                isinstance(v, (int, float)) for v in bbox[:4]
            ):
                # Flat format: [x1, y1, x2, y2] (+ optional page)
                x1, y1, x2, y2 = (float(v) for v in bbox[:4])
                if len(bbox) == 5:
                    page = _parse_page(bbox[4])
            else:
                return None

            # Reject non-finite coordinates (NaN, inf) as malformed input so
            # they score as a miss rather than poisoning IoU output.
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                return None

            coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            return (coords, page)
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
