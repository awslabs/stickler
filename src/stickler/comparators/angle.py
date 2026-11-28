"""Angle comparator that measures the shortest difference around the circle."""

import math
from numbers import Real
from typing import Any, Dict, Optional

from stickler.comparators.base import BaseComparator

_FULL_TURN = {"degrees": 360.0, "radians": 2 * math.pi}


class AngleComparator(BaseComparator):
    """Comparator for angles such as heading or yaw, with wraparound.

    Two angles match when their shortest angular difference is at most
    ``max_angular_difference``, so ``359`` and ``1`` are 2 degrees apart, not
    358. Any multiple of a full turn is the same angle: ``-90``, ``270`` and
    ``630`` all match. The score is binary, as for ``NumericComparator``:
    1.0 within the tolerance, 0.0 beyond it.

    Inputs must be finite ``numbers.Real`` values, such as ``int``, ``float``
    or a NumPy float. A string, a boolean, a ``Decimal`` or an infinite value
    is malformed and scores 0.0. A float NaN counts as missing, as for every
    comparator.

    Args:
        threshold: Similarity threshold (default 1.0).
        max_angular_difference: Largest shortest difference that still
            matches, in ``angle_unit`` (default 0.0, exact). Must be at least
            0 and less than half a turn, because half a turn would match
            every pair.
        angle_unit: ``"degrees"`` (default) or ``"radians"``, for both the
            inputs and ``max_angular_difference``.

    Raises:
        ValueError: If ``angle_unit`` or ``max_angular_difference`` is
            invalid.

    Example:
        >>> from stickler.comparators import AngleComparator
        >>> heading = AngleComparator(max_angular_difference=5)
        >>> heading.compare(359, 1)
        1.0
        >>> heading.compare(0, 180)
        0.0
    """

    DEFAULT_THRESHOLD = 1.0

    def __init__(
        self,
        threshold: Optional[float] = None,
        max_angular_difference: float = 0.0,
        angle_unit: str = "degrees",
    ):
        super().__init__(threshold=threshold)
        # The type check first, because an unhashable value such as a list
        # raises TypeError on the dict lookup.
        if not isinstance(angle_unit, str) or angle_unit not in _FULL_TURN:
            raise ValueError(
                f"angle_unit must be 'degrees' or 'radians', got {angle_unit!r}"
            )
        half_turn = _FULL_TURN[angle_unit] / 2
        tolerance = _finite_float(max_angular_difference)
        if tolerance is None or not 0 <= tolerance < half_turn:
            # A tolerance of half a turn or more matches every pair, which is
            # usually a value in degrees passed with angle_unit="radians".
            raise ValueError(
                f"max_angular_difference must be at least 0 and less than "
                f"{half_turn:g} {angle_unit}, got {max_angular_difference!r}"
            )
        # A float, not a NumPy scalar, so the exported schema stays JSON.
        self.max_angular_difference = tolerance
        self.angle_unit = angle_unit

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Return the non-default configuration for serialization."""
        config: Dict[str, Any] = {}
        if self.max_angular_difference != 0.0:
            config["max_angular_difference"] = self.max_angular_difference
        if self.angle_unit != "degrees":
            config["angle_unit"] = self.angle_unit
        return config or None

    def _compare(self, angle1: Any, angle2: Any) -> float:
        """Return 1.0 if the angles are within tolerance, 0.0 otherwise."""
        # Imported here, as in BaseComparator.binary_compare, so this module
        # has no import-time dependency on the evaluator package.
        from stickler.structured_object_evaluator.models.threshold_helper import (
            ThresholdHelper,
        )

        float1 = _finite_float(angle1)
        float2 = _finite_float(angle2)
        if float1 is None or float2 is None:
            return 0.0
        # Reducing each angle to one turn first keeps the subtraction finite
        # and exact for large inputs. The outer remainder() maps the
        # difference into [-half turn, half turn], so its magnitude is the
        # shortest way around the circle.
        full_turn = _FULL_TURN[self.angle_unit]
        difference = abs(
            # requires-python is >=3.10, so math.remainder (3.7+) is always present.
            # nosemgrep: python.lang.compatibility.python37.python37-compatibility-math1
            math.remainder(
                # nosemgrep: python.lang.compatibility.python37.python37-compatibility-math1
                math.remainder(float1, full_turn) - math.remainder(float2, full_turn),
                full_turn,
            )
        )
        # The same float tolerance as a threshold check, so 359.9 against 0.1
        # meets a tolerance of 0.2.
        within = ThresholdHelper.is_above_threshold(
            self.max_angular_difference, difference
        )
        return 1.0 if within else 0.0


def _finite_float(value: Any) -> Optional[float]:
    """The value as a finite float, or None for a non-real, bool or overflow."""
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None
