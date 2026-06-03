"""Date comparison comparator."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional

from stickler.comparators.base import BaseComparator

try:
    from dateutil import parser as dateutil_parser

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

_DATE_ORDER_FLAGS: Dict[str, Dict[str, bool]] = {
    "YMD": {"yearfirst": True, "dayfirst": False},
    "MDY": {"yearfirst": False, "dayfirst": False},
    "DMY": {"yearfirst": False, "dayfirst": True},
}

_GRANULARITIES = ("year", "month", "day", "second")

# Fixed default keeps missing date parts deterministic (no today-pollution).
_DEFAULT_DATE = datetime(1, 1, 1)


class DateComparator(BaseComparator):
    """Comparator that parses two date strings and compares them as datetime objects.

    Handles canonical and non-canonical date formats (e.g., "2025-01-01",
    "Jan 1, 2025", "2025-01-01T00:00:00Z") by parsing both values with
    python-dateutil and comparing the resulting datetimes.

    Unparseable input is treated as an explicit scoring failure (returns 0.0),
    not a silent guess.

    Example:
        ```python
        comparator = DateComparator()

        # Returns 1.0 - same date, different formats
        comparator.compare("2025-01-01", "Jan 1, 2025")

        # Returns 1.0 - ISO with timezone vs plain ISO
        comparator.compare("2025-01-01T00:00:00Z", "2025-01-01")

        # Returns 0.0 - different dates
        comparator.compare("2025-01-01", "2025-01-02")

        # Returns 1.0 - different dates within 1-day tolerance
        DateComparator(tolerance=timedelta(days=1)).compare("2025-01-01", "2025-01-02")

        # Returns 1.0 - same year, granularity="year"
        DateComparator(granularity="year").compare("2025-01-01", "2025-06-15")
        ```
    """

    def __init__(
        self,
        threshold: float = 1.0,
        tolerance: timedelta = timedelta(0),
        granularity: Literal["year", "month", "day", "second"] = "day",
        date_order: Literal["YMD", "MDY", "DMY"] = "YMD",
    ):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (default 1.0).
            tolerance: Maximum allowed difference between dates (default timedelta(0)).
            granularity: Comparison precision - "year", "month", "day", or "second"
                (default "day"). Coarser granularities ignore finer date components.
            date_order: Disambiguation for numeric dates like "01/02/2025".
                "YMD" (default) treats the first component as year,
                "MDY" treats it as month, "DMY" treats it as day.

        Raises:
            ImportError: If python-dateutil is not installed.
            ValueError: If any configuration parameter is invalid.
        """
        if not DATEUTIL_AVAILABLE:
            raise ImportError(
                "python-dateutil is required for DateComparator. "
                "Install it with: pip install python-dateutil"
            )
        super().__init__(threshold=threshold)

        if not isinstance(tolerance, timedelta):
            raise ValueError("tolerance must be a timedelta instance")
        if tolerance < timedelta(0):
            raise ValueError("tolerance must be non-negative")
        if granularity not in _GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {_GRANULARITIES}, got {granularity!r}"
            )
        if date_order not in _DATE_ORDER_FLAGS:
            raise ValueError(
                f"date_order must be one of {list(_DATE_ORDER_FLAGS)}, got {date_order!r}"
            )

        self.tolerance = tolerance
        self.granularity = granularity
        self.date_order = date_order

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Return non-default configuration parameters for serialization."""
        cfg: Dict[str, Any] = {}
        if self.tolerance != timedelta(0):
            cfg["tolerance"] = self.tolerance
        if self.granularity != "day":
            cfg["granularity"] = self.granularity
        if self.date_order != "YMD":
            cfg["date_order"] = self.date_order
        return cfg or None

    def compare(self, str1: Any, str2: Any) -> float:
        """Compare two date strings.

        Args:
            str1: First date value (string or string-convertible).
            str2: Second date value (string or string-convertible).

        Returns:
            1.0 if the parsed dates are equal within tolerance and granularity,
            0.0 if they differ or either value cannot be parsed.
        """
        if str1 is None and str2 is None:
            return 1.0
        if str1 is None or str2 is None:
            return 0.0

        dt1 = self._parse_date(str(str1))
        dt2 = self._parse_date(str(str2))

        if dt1 is None or dt2 is None:
            return 0.0

        dt1 = self._truncate_to_granularity(dt1)
        dt2 = self._truncate_to_granularity(dt2)

        return 1.0 if abs(dt1 - dt2) <= self.tolerance else 0.0

    def _parse_date(self, value: str) -> Optional[datetime]:
        """Parse a date string into a UTC-naive datetime.

        Args:
            value: Date string to parse.

        Returns:
            Parsed datetime (UTC-naive), or None if parsing fails.
        """
        flags = _DATE_ORDER_FLAGS[self.date_order]
        try:
            dt = dateutil_parser.parse(value, default=_DEFAULT_DATE, **flags)
        except (ValueError, OverflowError):
            return None

        # Normalize timezone-aware datetimes to UTC-naive so comparison works
        # regardless of whether the input had an explicit timezone.
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

        return dt

    def _truncate_to_granularity(self, dt: datetime) -> datetime:
        """Zero out datetime components finer than the configured granularity.

        Args:
            dt: Datetime to truncate.

        Returns:
            Truncated datetime.
        """
        if self.granularity == "year":
            return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif self.granularity == "month":
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif self.granularity == "day":
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        else:  # second
            return dt.replace(microsecond=0)
