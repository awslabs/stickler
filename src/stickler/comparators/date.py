"""Date comparison comparator.

Provides a deterministic, non-LLM comparator that parses two date strings
(or ``datetime`` / ``date`` objects) and compares them as ``datetime`` values.

Built on ``python-dateutil`` for flexible parsing of canonical and
non-canonical formats (ISO 8601, ``"Jan 1, 2025"``, ``"1/1/25"``, etc.)
with no network calls and no LLM usage.

Example:
    ```python
    from stickler.comparators import DateComparator

    # Default: ISO-biased parsing, exact day-level equality
    cmp = DateComparator()
    cmp.compare("2025-01-01", "Jan 1, 2025")         # 1.0
    cmp.compare("2025-01-01", "2025-01-02")          # 0.0

    # Year-level granularity
    cmp = DateComparator(granularity="year")
    cmp.compare("2025-01-01", "2025-12-31")          # 1.0

    # Tolerance window
    from datetime import timedelta
    cmp = DateComparator(tolerance=timedelta(days=1))
    cmp.compare("2025-01-01", "2025-01-02")          # 1.0

    # Disambiguate ambiguous numeric dates. The hint only kicks in for
    # strings where the component order is unclear (like "01/02/2025").
    # ISO-looking and named-month inputs always parse the same way.
    us = DateComparator(date_order="us")
    us.compare("01/02/2025", "2025-01-02")           # 1.0 (Jan 2)

    eu = DateComparator(date_order="european")
    eu.compare("01/02/2025", "2025-02-01")           # 1.0 (Feb 1)
    ```

Note on parse failures:
    Unparseable input (``"not a date"``, garbled strings, empty values)
    returns ``0.0`` — the same value as a valid-but-different date. This
    matches the behavior of other comparators in this package (e.g.
    ``NumericComparator`` treats ``"abc"`` the same way). Callers that
    need to distinguish "garbage in" from "dates differ" should validate
    inputs upstream.

Note on scoring:
    Scoring is currently binary: within tolerance → ``1.0``, outside →
    ``0.0``. A future enhancement could introduce a "ramp" mode that
    awards partial credit between an inner ``tolerance`` and an outer
    ``ramp_tolerance``, decaying linearly (or on another curve) from
    ``1.0`` down to ``threshold``. Not included in v1 — flagged here for
    future work.
"""

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Optional

from stickler.comparators.base import BaseComparator

try:
    from dateutil import parser as _dateutil_parser

    _DATEUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - dateutil is a declared dep
    _DATEUTIL_AVAILABLE = False


# ISO 8601-ish strings lead with a 4-digit year (e.g., "2025-01-01",
# "2025-01-01T00:00:00Z", "2025/01/01"). These should always parse
# year-first regardless of the user's date_order hint — the hint is for
# disambiguating numeric slash-separated dates like "01/02/2025".
_ISO_LEADING_YEAR = re.compile(r"^\s*\d{4}[-/.]")


Granularity = Literal["year", "month", "day", "hour", "minute", "second"]
DateOrder = Literal["iso", "us", "european"]


class DateComparator(BaseComparator):
    """Deterministic date comparator.

    Parses both operands into ``datetime`` objects and compares them.
    Supports configurable tolerance windows, partial granularity (e.g.
    year-only or month-only comparison), and a ``date_order`` hint for
    disambiguating genuinely ambiguous numeric dates like ``"01/02/2025"``.

    The ``date_order`` hint is a **tiebreaker**, not a global mode.
    Unambiguously-formatted input (ISO-looking strings like
    ``"2025-01-01"``, or named-month strings like ``"Jan 1, 2025"``)
    always parses the same way regardless of the hint. The hint only
    matters when the parser can't tell which of two interpretations is
    correct.

    Unparseable input is treated as a comparison failure (returns
    ``0.0``). See the module docstring for the rationale.

    Attributes:
        tolerance: ``timedelta`` window for near-matches, applied at the
            configured granularity. Defaults to ``timedelta(0)`` (exact).
        granularity: Precision of the comparison. One of ``"year"``,
            ``"month"``, ``"day"``, ``"hour"``, ``"minute"``, ``"second"``.
            Defaults to ``"day"``.
        date_order: Tiebreaker for ambiguous numeric dates. One of
            ``"iso"`` (year-first, default), ``"us"`` (month-first), or
            ``"european"`` (day-first). Only affects parsing of dates
            like ``"01/02/2025"`` where the component order is unclear.
    """

    _GRANULARITY_ORDER = ("year", "month", "day", "hour", "minute", "second")
    _VALID_DATE_ORDERS = ("iso", "us", "european")

    def __init__(
        self,
        threshold: float = 1.0,
        tolerance: Optional[timedelta] = None,
        granularity: Granularity = "day",
        date_order: DateOrder = "iso",
    ):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (default 1.0). Kept for
                interface compatibility; this comparator is currently
                binary (0.0 or 1.0).
            tolerance: Optional ``timedelta`` window. Two dates are
                considered equal if their absolute difference is within
                this window. Defaults to ``timedelta(0)``.
            granularity: Precision at which dates are compared. Fields
                below this level are truncated before comparison.
            date_order: Tiebreaker for ambiguous numeric dates. Only
                matters for strings like ``"01/02/2025"`` where the
                component order is unclear.
                ``"iso"`` → year-first (default, matches typical clean
                ground truth).
                ``"us"`` → month-first (e.g. ``"01/02/2025"`` = Jan 2).
                ``"european"`` → day-first (e.g. ``"01/02/2025"`` = Feb 1).
                Has no effect on ISO-formatted or named-month inputs.

        Raises:
            ImportError: If ``python-dateutil`` is not installed.
            ValueError: If ``granularity`` or ``date_order`` is invalid,
                or if ``tolerance`` is negative.
        """
        super().__init__(threshold=threshold)

        if not _DATEUTIL_AVAILABLE:
            raise ImportError(
                "The python-dateutil library is required for DateComparator. "
                "Install it with: pip install python-dateutil"
            )

        if granularity not in self._GRANULARITY_ORDER:
            raise ValueError(
                f"Invalid granularity '{granularity}'. "
                f"Must be one of {self._GRANULARITY_ORDER}."
            )

        if date_order not in self._VALID_DATE_ORDERS:
            raise ValueError(
                f"Invalid date_order '{date_order}'. "
                f"Must be one of {self._VALID_DATE_ORDERS}."
            )

        self.tolerance = tolerance if tolerance is not None else timedelta(0)
        if self.tolerance < timedelta(0):
            raise ValueError("tolerance must be non-negative")

        self.granularity = granularity
        self.date_order = date_order

        # Map date_order to dateutil's dayfirst / yearfirst flags.
        if date_order == "iso":
            self._dayfirst = False
            self._yearfirst = True
        elif date_order == "european":
            self._dayfirst = True
            self._yearfirst = False
        else:  # "us"
            self._dayfirst = False
            self._yearfirst = False

    def compare(self, str1: Any, str2: Any) -> float:
        """Compare two date values.

        Args:
            str1: First value. May be ``str``, ``datetime``, ``date``, or ``None``.
            str2: Second value. Same accepted types as ``str1``.

        Returns:
            ``1.0`` if the parsed dates are equal at the configured
            granularity (optionally within ``tolerance``), else ``0.0``.
            Also returns ``0.0`` when either input is unparseable.
        """
        if str1 is None and str2 is None:
            return 1.0
        if str1 is None or str2 is None:
            return 0.0

        dt1 = self._parse(str1)
        dt2 = self._parse(str2)
        if dt1 is None or dt2 is None:
            return 0.0

        dt1, dt2 = self._align_timezones(dt1, dt2)
        dt1 = self._truncate(dt1)
        dt2 = self._truncate(dt2)

        diff = abs(dt1 - dt2)
        return 1.0 if diff <= self.tolerance else 0.0

    @staticmethod
    def _align_timezones(dt1: datetime, dt2: datetime) -> tuple[datetime, datetime]:
        """Make two datetimes tz-comparable before subtraction.

        Python rejects subtracting a tz-aware datetime from a tz-naive one.
        When exactly one side is tz-aware, we assume the naive side is in
        the same implicit zone and attach the aware side's tzinfo. When
        both are aware, we normalize to UTC to get an unambiguous diff.
        """
        aware1 = dt1.tzinfo is not None
        aware2 = dt2.tzinfo is not None

        if aware1 and aware2:
            return dt1.astimezone(timezone.utc), dt2.astimezone(timezone.utc)
        if aware1 and not aware2:
            return dt1, dt2.replace(tzinfo=dt1.tzinfo)
        if aware2 and not aware1:
            return dt1.replace(tzinfo=dt2.tzinfo), dt2
        return dt1, dt2

    def _parse(self, value: Any) -> Optional[datetime]:
        """Parse a value into a ``datetime``.

        Returns ``None`` for unparseable input so the caller can treat it
        as a comparison failure.

        Uses the ``date_order`` hint only when the input is genuinely
        ambiguous. Strings that start with a 4-digit year (ISO 8601-ish)
        always parse year-first, so mixed-format inputs — common when
        predictions come from LLM extractions — are handled correctly
        without requiring the user to configure anything per-side.
        """
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day)

        if not isinstance(value, str):
            value = str(value)

        stripped = value.strip()
        if not stripped:
            return None

        # ISO-leading strings are unambiguous; force year-first parsing
        # so the date_order hint doesn't accidentally flip them.
        if _ISO_LEADING_YEAR.match(stripped):
            dayfirst, yearfirst = False, True
        else:
            dayfirst, yearfirst = self._dayfirst, self._yearfirst

        try:
            return _dateutil_parser.parse(
                stripped, dayfirst=dayfirst, yearfirst=yearfirst
            )
        except (ValueError, OverflowError, TypeError):
            return None

    def _truncate(self, dt: datetime) -> datetime:
        """Truncate a ``datetime`` to the configured granularity."""
        if self.granularity == "year":
            return dt.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
        if self.granularity == "month":
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if self.granularity == "day":
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.granularity == "hour":
            return dt.replace(minute=0, second=0, microsecond=0)
        if self.granularity == "minute":
            return dt.replace(second=0, microsecond=0)
        # second
        return dt.replace(microsecond=0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold}, "
            f"tolerance={self.tolerance!r}, "
            f"granularity='{self.granularity}', "
            f"date_order='{self.date_order}')"
        )
