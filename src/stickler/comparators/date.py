"""Date comparison comparator.

Deterministic, non-LLM date comparator. Parses both sides into ``datetime``
(or a date range) and scores per the tier system documented in
``docs/docs/Guides/Comparators/date-comparator.md``.

Scoring tiers:

* Tier 1: same calendar day (after surface-form normalization) → 1.0
* Tier 2: both sides year-less, same month/day → 1.0
* Tier 3: one side year-less, m/d match (only when ``allow_partial_year=True``)
  → ``_PARTIAL_YEAR_MULTIPLIER`` (0.7), else 0.0
* Tier 4 / 4b: range comparisons, behavior controlled by ``range_mode``:
    - ``"strict"``  – range-vs-single = 0; range-vs-range = endpoints must
      match exactly, else 0.
    - ``"reject"``  – any range input = 0 regardless of the other side;
      single-day ranges are NOT collapsed.
    - ``"contains"`` – range-vs-single = 1 if single is inside range, else
      0; range-vs-range = endpoints exact, else 0.
    - ``"graded"`` (default) – range-vs-single = 0.5 if inside, else 0;
      range-vs-range = Jaccard (overlap days / union days).
* Tier 5: anything else → 0.0

Year-presence mismatch in range comparisons multiplies the base range score
by ``_PARTIAL_YEAR_MULTIPLIER`` when ``allow_partial_year=True``, or by
``0.0`` (i.e. the comparison is rejected) otherwise.

Higher-level threshold/clip behavior is the caller's job — this comparator
returns the honest similarity, ``ComparableField`` decides what counts as
a match.

Built on ``python-dateutil`` for parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Optional, Tuple, Union

from stickler.comparators.base import BaseComparator

try:
    from dateutil import parser as _dateutil_parser

    _DATEUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - dateutil is a declared dep
    _DATEUTIL_AVAILABLE = False


# Score constants (baked in; tune via ``ComparableField.threshold`` rather
# than per-comparator knobs).
_PARTIAL_YEAR_MULTIPLIER = 0.7
_RANGE_CONTAINS_GRADED_SCORE = 0.5

# Range delimiters in priority order. Spaces are required around the bare
# ``-`` so we don't shred ISO dates like ``2025-01-01``.
_RANGE_DELIMS = (" to ", " through ", " - ")


RangeMode = Literal["strict", "reject", "contains", "graded"]
_VALID_RANGE_MODES: Tuple[RangeMode, ...] = (
    "strict",
    "reject",
    "contains",
    "graded",
)


@dataclass(frozen=True)
class _ParsedSingle:
    """A successfully-parsed single date plus whether the input claimed a year."""

    dt: datetime
    has_year: bool


@dataclass(frozen=True)
class _ParsedRange:
    """A successfully-parsed date range (start <= end)."""

    start: _ParsedSingle
    end: _ParsedSingle


_ParseResult = Union[_ParsedSingle, _ParsedRange, None]


class DateComparator(BaseComparator):
    """Deterministic date comparator with year/range awareness.

    See ``docs/docs/Guides/Comparators/date-comparator.md`` for the full
    behavior reference, configuration matrix, and corner cases.

    Args:
        threshold: Forwarded to :class:`BaseComparator`.
        tolerance: Optional ``timedelta`` window for Tier 1 same-day
            comparisons only. Range and partial-year branches ignore it.
            Defaults to ``timedelta(0)``.
        dayfirst: How to interpret ambiguous numeric dates like
            ``"01/02/2025"``. ``None`` (default) tries both
            interpretations and takes the better-matching score; ``True``
            forces day-first; ``False`` forces month-first.
        allow_partial_year: If ``True``, year-less ↔ year-bearing pairs
            with matching month/day score ``0.7``. Default ``False``.
        range_mode: How range comparisons are scored. One of
            ``"strict"``, ``"reject"``, ``"contains"``, ``"graded"``
            (default).
    """

    def __init__(
        self,
        threshold: float = 1.0,
        tolerance: Optional[Union[timedelta, int, float]] = None,
        dayfirst: Optional[bool] = None,
        allow_partial_year: bool = False,
        range_mode: RangeMode = "graded",
    ):
        super().__init__(threshold=threshold)

        if not _DATEUTIL_AVAILABLE:
            raise ImportError(
                "The python-dateutil library is required for DateComparator. "
                "Install it with: pip install python-dateutil"
            )

        if dayfirst not in (None, True, False):
            raise ValueError(
                f"dayfirst must be None, True, or False; got {dayfirst!r}"
            )

        if range_mode not in _VALID_RANGE_MODES:
            raise ValueError(
                f"range_mode must be one of {_VALID_RANGE_MODES}; "
                f"got {range_mode!r}"
            )

        # Tolerance accepts ``timedelta``, ``int``, or ``float``. Numeric
        # inputs are interpreted as days — friendlier for JSON-schema
        # configs where a literal ``timedelta(days=N)`` isn't expressible.
        if tolerance is None:
            self.tolerance = timedelta(0)
        elif isinstance(tolerance, timedelta):
            self.tolerance = tolerance
        elif isinstance(tolerance, bool):
            # bool is a subclass of int; reject it explicitly so True/False
            # don't silently become 1-day / 0-day windows.
            raise ValueError(
                "tolerance must be a timedelta or a numeric value in days; "
                f"got bool {tolerance!r}"
            )
        elif isinstance(tolerance, (int, float)):
            self.tolerance = timedelta(days=tolerance)
        else:
            raise ValueError(
                "tolerance must be a timedelta or a numeric value in days; "
                f"got {type(tolerance).__name__}"
            )

        if self.tolerance < timedelta(0):
            raise ValueError("tolerance must be non-negative")

        self.dayfirst = dayfirst
        self.allow_partial_year = allow_partial_year
        self.range_mode = range_mode

    @property
    def config(self) -> dict:
        """Round-trippable config for JSON-schema export.

        Tolerance is exported as days (an int when the timedelta is a
        whole number of days, otherwise a float) so it can survive a
        JSON round-trip.
        """
        cfg: dict = {
            "dayfirst": self.dayfirst,
            "allow_partial_year": self.allow_partial_year,
            "range_mode": self.range_mode,
        }
        if self.tolerance != timedelta(0):
            seconds = self.tolerance.total_seconds()
            days = seconds / 86400
            cfg["tolerance"] = int(days) if days.is_integer() else days
        return cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self, str1: Any, str2: Any) -> float:
        """Score two date values per the tier system documented above."""
        if str1 is None and str2 is None:
            return 1.0
        if str1 is None or str2 is None:
            return 0.0

        # Resolve dayfirst pairwise. ``None`` means "try both
        # interpretations and take the best score" — that way a string
        # whose layout is genuinely ambiguous in isolation can still
        # match if one consistent interpretation lines up.
        if self.dayfirst is not None:
            return self._compare_with_dayfirst(str1, str2, self.dayfirst)

        return max(
            self._compare_with_dayfirst(str1, str2, False),
            self._compare_with_dayfirst(str1, str2, True),
        )

    def _compare_with_dayfirst(
        self, str1: Any, str2: Any, dayfirst: bool
    ) -> float:
        """Run the tier dispatch with ``dayfirst`` pinned to one value."""
        a = self._parse(str1, dayfirst=dayfirst)
        b = self._parse(str2, dayfirst=dayfirst)
        if a is None or b is None:
            return 0.0

        a_is_range = isinstance(a, _ParsedRange)
        b_is_range = isinstance(b, _ParsedRange)

        # ``reject`` mode: any range input zeros out the comparison.
        if self.range_mode == "reject" and (a_is_range or b_is_range):
            return 0.0

        # Tier 4b: range vs range
        if a_is_range and b_is_range:
            return self._compare_range_range(a, b)

        # Tier 4: range vs single
        if a_is_range or b_is_range:
            single = b if a_is_range else a  # type: ignore[assignment]
            rng = a if a_is_range else b  # type: ignore[assignment]
            return self._compare_range_single(rng, single)

        # Both singles
        return self._compare_singles(a, b)

    # ------------------------------------------------------------------
    # Tier dispatch
    # ------------------------------------------------------------------

    def _compare_range_range(
        self, a: _ParsedRange, b: _ParsedRange
    ) -> float:
        """Tier 4b: range vs range under the configured range_mode."""
        # Year-presence consistency on both endpoints of both sides.
        # If endpoints disagree on year-presence within a side it's
        # malformed; we treat that as a 0.0 rather than try to repair.
        if a.start.has_year != a.end.has_year:
            return 0.0
        if b.start.has_year != b.end.has_year:
            return 0.0

        year_match = a.start.has_year == b.start.has_year
        partial_year_multiplier = self._partial_year_multiplier(year_match)
        if partial_year_multiplier == 0.0:
            return 0.0

        if self.range_mode in ("strict", "contains"):
            if (
                self._dates_equal_day(a.start.dt, b.start.dt)
                and self._dates_equal_day(a.end.dt, b.end.dt)
            ):
                return 1.0 * partial_year_multiplier
            return 0.0

        # graded → Jaccard
        # (reject mode is handled before we get here)
        return self._jaccard(a, b) * partial_year_multiplier

    def _compare_range_single(
        self, rng: _ParsedRange, single: _ParsedSingle
    ) -> float:
        """Tier 4: range-vs-single under the configured range_mode."""
        if self.range_mode == "strict":
            return 0.0

        # Year-presence consistency: the range's endpoints must agree
        # internally, and we compare against the single's claim.
        if rng.start.has_year != rng.end.has_year:
            return 0.0
        year_match = rng.start.has_year == single.has_year
        partial_year_multiplier = self._partial_year_multiplier(year_match)
        if partial_year_multiplier == 0.0:
            return 0.0

        # Containment: when both sides agree on year-presence we compare
        # the full datetimes; when they disagree (only possible under
        # allow_partial_year=True) the year on the year-less side is a
        # fictional 1900 placeholder, so we compare on (month, day) only.
        if year_match:
            s = self._truncate_day(single.dt)
            lo = self._truncate_day(rng.start.dt)
            hi = self._truncate_day(rng.end.dt)
            inside = lo <= s <= hi
        else:
            inside = self._md_in_md_range(
                (single.dt.month, single.dt.day),
                (rng.start.dt.month, rng.start.dt.day),
                (rng.end.dt.month, rng.end.dt.day),
            )

        if inside:
            base = (
                1.0 if self.range_mode == "contains" else _RANGE_CONTAINS_GRADED_SCORE
            )
            return base * partial_year_multiplier
        return 0.0

    @staticmethod
    def _md_in_md_range(
        target: Tuple[int, int],
        lo: Tuple[int, int],
        hi: Tuple[int, int],
    ) -> bool:
        """Check (month, day) containment in a (month, day) range.

        Treats the range as a same-year span. If the range crosses year
        boundary in m/d terms (e.g. Dec 20 to Jan 5), we'd need wrap-around
        logic; for the year-mismatch partial-year case the range is always
        year-bearing internally so this only matters when its m/d span
        wraps mid-comparison. In practice ranges in our data are
        same-year, so we keep this simple and treat the m/d ordering as
        non-wrapping (lo <= target <= hi).
        """
        return lo <= target <= hi

    def _compare_singles(
        self, a: _ParsedSingle, b: _ParsedSingle
    ) -> float:
        """Tiers 1/2/3/5 for two single dates."""
        if not a.has_year and not b.has_year:
            # Tier 2: both year-less. Match on m/d alone.
            return 1.0 if (a.dt.month, a.dt.day) == (b.dt.month, b.dt.day) else 0.0

        if a.has_year != b.has_year:
            # Tier 3: only one side claims a year.
            if not self.allow_partial_year:
                return 0.0
            if (a.dt.month, a.dt.day) == (b.dt.month, b.dt.day):
                return _PARTIAL_YEAR_MULTIPLIER
            return 0.0

        # Tier 1/5: both have years. Standard comparison with tolerance.
        a_dt, b_dt = self._align_timezones(a.dt, b.dt)
        a_dt = self._truncate_day(a_dt)
        b_dt = self._truncate_day(b_dt)
        return 1.0 if abs(a_dt - b_dt) <= self.tolerance else 0.0

    def _partial_year_multiplier(self, year_match: bool) -> float:
        """Multiplier applied to range scores when year-presence (mis)matches.

        Returns 1.0 when both sides agree on year-presence. Returns
        ``_PARTIAL_YEAR_MULTIPLIER`` (0.7) when ``allow_partial_year=True``
        and they disagree. Returns 0.0 otherwise.
        """
        if year_match:
            return 1.0
        return _PARTIAL_YEAR_MULTIPLIER if self.allow_partial_year else 0.0

    def _jaccard(self, a: _ParsedRange, b: _ParsedRange) -> float:
        """Jaccard overlap between two date ranges, day-level."""
        a_lo = self._truncate_day(a.start.dt)
        a_hi = self._truncate_day(a.end.dt)
        b_lo = self._truncate_day(b.start.dt)
        b_hi = self._truncate_day(b.end.dt)

        # Inclusive day count.
        intersect_lo = max(a_lo, b_lo)
        intersect_hi = min(a_hi, b_hi)
        if intersect_hi < intersect_lo:
            return 0.0

        intersect_days = (intersect_hi - intersect_lo).days + 1
        union_lo = min(a_lo, b_lo)
        union_hi = max(a_hi, b_hi)
        union_days = (union_hi - union_lo).days + 1
        return intersect_days / union_days

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, value: Any, dayfirst: bool) -> _ParseResult:
        """Parse input into a single date or range, or ``None`` on failure.

        Single-day ranges (``X to X``) are normally collapsed to a
        ``_ParsedSingle`` so they compare consistently with the bare
        single-date form. Under ``range_mode="reject"`` we skip the
        collapse so that the original range shape is preserved and the
        comparison surfaces it as a structural mismatch.
        """
        if isinstance(value, datetime):
            return _ParsedSingle(dt=value, has_year=True)
        if isinstance(value, date):
            return _ParsedSingle(
                dt=datetime(value.year, value.month, value.day), has_year=True
            )

        if not isinstance(value, str):
            value = str(value)

        s = value.strip()
        if not s:
            return None

        rng = self._try_parse_range(s, dayfirst=dayfirst)
        if rng is not None:
            # Collapse degenerate single-day ranges to singles, EXCEPT
            # under reject mode where we want the range shape preserved.
            if (
                self.range_mode != "reject"
                and self._dates_equal_day(rng.start.dt, rng.end.dt)
                and rng.start.has_year == rng.end.has_year
            ):
                return rng.start
            return rng

        return self._try_parse_single(s, dayfirst=dayfirst)

    def _try_parse_range(
        self, s: str, dayfirst: bool
    ) -> Optional[_ParsedRange]:
        """Detect a range by splitting on configured delimiters."""
        for delim in _RANGE_DELIMS:
            if delim not in s:
                continue
            left, _, right = s.partition(delim)
            left_p = self._try_parse_single(left.strip(), dayfirst=dayfirst)
            right_p = self._try_parse_single(right.strip(), dayfirst=dayfirst)
            if left_p is None or right_p is None:
                continue
            if left_p.dt > right_p.dt:
                return None
            return _ParsedRange(start=left_p, end=right_p)
        return None

    def _try_parse_single(
        self, s: str, dayfirst: bool
    ) -> Optional[_ParsedSingle]:
        """Parse one side as a single date (or ``None`` on failure)."""
        if not s:
            return None

        try:
            dt_lo = _dateutil_parser.parse(
                s, default=datetime(1900, 1, 1), dayfirst=dayfirst
            )
            dt_hi = _dateutil_parser.parse(
                s, default=datetime(2099, 1, 1), dayfirst=dayfirst
            )
        except (ValueError, OverflowError, TypeError):
            return None

        has_year = dt_lo.year == dt_hi.year

        # Reject time-only inputs ('12:30 PM', '10/45AM' etc.). When
        # neither year nor month/day is fully specified, both parses
        # land on the default's Jan 1 with a non-zero time component.
        if not has_year:
            if (dt_lo.month, dt_lo.day) != (dt_hi.month, dt_hi.day):
                return None
            time_present = (
                dt_lo.hour != 0
                or dt_lo.minute != 0
                or dt_lo.second != 0
                or dt_lo.microsecond != 0
            )
            if time_present and (dt_lo.month, dt_lo.day) == (1, 1):
                return None

        return _ParsedSingle(dt=dt_lo, has_year=has_year)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_timezones(
        dt1: datetime, dt2: datetime
    ) -> tuple[datetime, datetime]:
        """Make two datetimes tz-comparable for subtraction."""
        aware1 = dt1.tzinfo is not None
        aware2 = dt2.tzinfo is not None
        if aware1 and aware2:
            return dt1.astimezone(timezone.utc), dt2.astimezone(timezone.utc)
        if aware1 and not aware2:
            return dt1, dt2.replace(tzinfo=dt1.tzinfo)
        if aware2 and not aware1:
            return dt1.replace(tzinfo=dt2.tzinfo), dt2
        return dt1, dt2

    @staticmethod
    def _truncate_day(dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    @classmethod
    def _dates_equal_day(cls, dt1: datetime, dt2: datetime) -> bool:
        a, b = cls._align_timezones(dt1, dt2)
        return cls._truncate_day(a) == cls._truncate_day(b)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold}, "
            f"tolerance={self.tolerance!r}, "
            f"dayfirst={self.dayfirst!r}, "
            f"allow_partial_year={self.allow_partial_year}, "
            f"range_mode={self.range_mode!r})"
        )
