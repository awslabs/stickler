"""Date comparison comparator.

Deterministic, non-LLM date comparator. Parses both sides into ``datetime``
(or a date range) and scores per the tier system documented in
``docs/docs/Guides/Comparators/date-comparator.md``.

Scoring tiers:

* Tier 1: same calendar day (after surface-form normalization) → 1.0
* Tier 2: both sides year-less, same month/day → 1.0
* Tier 3: one side year-less, m/d match (only when ``allow_partial_year=True``)
  → ``_PARTIAL_YEAR_MULTIPLIER`` (0.7), else 0.0

Two orthogonal "missing field" axes govern partial-precision comparisons:

* **Year presence** (``allow_partial_year``) — one side omits the year while
  pinning month/day exactly (``'11/03'`` vs ``'11/03/2012'``).
* **Month/day resolution** (``precision_mode``) — one side is coarser than the
  other in its *low-order* fields (``'Jan 2024'`` vs ``'Jan 1, 2024'``,
  ``'2024'`` vs ``'2024-01-01'``). ``"exact"`` (default) requires matching
  resolution; ``"gt_loose"`` lets the prediction be finer than the ground
  truth but not coarser; ``"overlap"`` accepts either side being coarser.
  The two axes compose: year-presence is judged separately from month/day
  resolution.
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

import re
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

# Inputs longer than this are rejected before parsing. The most verbose
# realistic date string ("Wednesday, the 1st day of January...") is well
# under it; anything larger is malformed and not worth a full dateutil
# scan (a 100k-char string costs hundreds of ms per ``compare``).
_MAX_INPUT_LEN = 256

# Range delimiters in priority order. Spaces are required around the bare
# ``-`` so we don't shred ISO dates like ``2025-01-01``.
_RANGE_DELIMS = (" to ", " through ", " - ")

# A four-digit leading group followed by a separator means the year is
# unambiguously first (ISO / year-first layouts like ``2025-02-01`` or
# ``2025/02/01``), which fixes month-then-day order. ``dayfirst`` must NOT
# be applied to these, or an unambiguous canonical date gets misread as a
# different day (the issue #117 headline case).
_YEAR_FIRST_RE = re.compile(r"^\d{4}[-/.]")


RangeMode = Literal["strict", "reject", "contains", "graded"]
_VALID_RANGE_MODES: Tuple[RangeMode, ...] = (
    "strict",
    "reject",
    "contains",
    "graded",
)


PrecisionMode = Literal["exact", "gt_loose", "overlap"]
_VALID_PRECISION_MODES: Tuple[PrecisionMode, ...] = (
    "exact",
    "gt_loose",
    "overlap",
)


@dataclass(frozen=True)
class _ParsedSingle:
    """A successfully-parsed single date plus which fields the input claimed.

    ``has_year``/``has_month``/``has_day`` record whether each component was
    actually present in the source string (vs. fabricated by the parser's
    default). They drive the year-presence (``allow_partial_year``) and
    month/day resolution (``precision_mode``) axes independently.
    """

    dt: datetime
    has_year: bool
    has_month: bool = True
    has_day: bool = True

    @property
    def md_resolution(self) -> int:
        """Low-order specificity: 0=year-only, 1=month, 2=month+day.

        Independent of year presence — a year-less ``'Oct 24'`` and a
        year-bearing ``'10/24/16'`` are both day-grain (``2``).
        """
        if not self.has_month:
            return 0
        return 2 if self.has_day else 1


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
        tolerance: Optional window for Tier 1 single-vs-single
            comparisons only (range and partial-year branches ignore it).
            Accepts a ``timedelta`` or a numeric value in days. A
            whole-day tolerance floors both sides to the calendar day
            (time ignored); a sub-day tolerance (e.g. ``1.5`` = 36h)
            compares actual timestamps. Defaults to ``None``, which is
            normalized to ``timedelta(0)`` (same calendar day).
        dayfirst: How to interpret ambiguous numeric dates like
            ``"01/02/2025"``. ``None`` (default) tries both
            interpretations and takes the better-matching score; ``True``
            forces day-first; ``False`` forces month-first.
        allow_partial_year: If ``True``, year-less ↔ year-bearing pairs
            with matching month/day score ``0.7``. Default ``False``.
        range_mode: How range comparisons are scored. One of
            ``"strict"``, ``"reject"``, ``"contains"``, ``"graded"``
            (default).
        precision_mode: How month/day *resolution* mismatches are scored
            (``"Jan 2024"`` vs ``"Jan 1, 2024"``). The first argument to
            :meth:`compare` is treated as ground truth.

            - ``"exact"`` (default): both sides must share the same
              resolution; a fabricated or dropped month/day is a miss.
            - ``"gt_loose"``: the prediction may be *finer* than the
              ground truth (extra precision ignored if consistent at the
              ground truth's grain) but not coarser.
            - ``"overlap"``: symmetric — either side may be coarser, as
              long as they agree on every field both sides specify.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        tolerance: Optional[Union[timedelta, int, float]] = None,
        dayfirst: Optional[bool] = None,
        allow_partial_year: bool = False,
        range_mode: RangeMode = "graded",
        precision_mode: PrecisionMode = "exact",
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

        if precision_mode not in _VALID_PRECISION_MODES:
            raise ValueError(
                f"precision_mode must be one of {_VALID_PRECISION_MODES}; "
                f"got {precision_mode!r}"
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
        self.precision_mode = precision_mode

    @property
    def config(self) -> Optional[dict]:
        """Round-trippable config for JSON-schema export.

        Only non-default values are emitted, and an all-default instance
        returns ``None`` — matching ``NumericComparator.config`` and
        keeping a redundant ``x-aws-stickler-comparator-config`` block out
        of every exported schema (the exporter keys off truthiness).

        Tolerance is exported as days (an int when the timedelta is a
        whole number of days, otherwise a float) so it can survive a
        JSON round-trip.
        """
        cfg: dict = {}
        if self.dayfirst is not None:
            cfg["dayfirst"] = self.dayfirst
        if self.allow_partial_year:
            cfg["allow_partial_year"] = self.allow_partial_year
        if self.range_mode != "graded":
            cfg["range_mode"] = self.range_mode
        if self.precision_mode != "exact":
            cfg["precision_mode"] = self.precision_mode
        if self.tolerance != timedelta(0):
            seconds = self.tolerance.total_seconds()
            days = seconds / 86400
            cfg["tolerance"] = int(days) if days.is_integer() else days
        return cfg or None

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
        #
        # A malformed value must never crash an evaluation run, so any
        # datetime-comparison edge (e.g. mixed tz-awareness the alignment
        # helpers didn't anticipate) degrades to 0.0 like every other
        # parse failure. The range/single paths align timezones inline;
        # this is a backstop, not the primary defense.
        try:
            if self.dayfirst is not None:
                return self._compare_with_dayfirst(str1, str2, self.dayfirst)

            return max(
                self._compare_with_dayfirst(str1, str2, False),
                self._compare_with_dayfirst(str1, str2, True),
            )
        except TypeError:
            return 0.0

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

        # Tier 4: range vs single. ``a`` is always the ground truth (first
        # compare() argument); track whether it's the single side so the
        # directional precision gate (gt_loose) orients correctly.
        if a_is_range or b_is_range:
            single = b if a_is_range else a  # type: ignore[assignment]
            rng = a if a_is_range else b  # type: ignore[assignment]
            return self._compare_range_single(
                rng, single, single_is_gt=not a_is_range
            )

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

        # Month/day resolution gate, per endpoint (precision_mode).
        if not self._resolution_ok(a.start, b.start):
            return 0.0
        if not self._resolution_ok(a.end, b.end):
            return 0.0

        year_match = a.start.has_year == b.start.has_year
        partial_year_multiplier = self._partial_year_multiplier(year_match)
        if partial_year_multiplier == 0.0:
            return 0.0

        # When year-presence differs (only reachable under
        # allow_partial_year=True), the year-less side's year is a
        # fictional 1900 placeholder, so endpoint equality and overlap are
        # judged on (month, day) only — mirroring the range-vs-single
        # m/d fallback. Otherwise compare full dates.
        if self.range_mode in ("strict", "contains"):
            if year_match:
                endpoints_match = self._dates_equal_day(
                    a.start.dt, b.start.dt
                ) and self._dates_equal_day(a.end.dt, b.end.dt)
            else:
                endpoints_match = self._md_equal(
                    a.start.dt, b.start.dt
                ) and self._md_equal(a.end.dt, b.end.dt)
            if endpoints_match:
                return 1.0 * partial_year_multiplier
            return 0.0

        # graded → Jaccard (reject mode is handled before we get here)
        jaccard = self._jaccard(a, b) if year_match else self._md_jaccard(a, b)
        return jaccard * partial_year_multiplier

    def _compare_range_single(
        self, rng: _ParsedRange, single: _ParsedSingle, single_is_gt: bool
    ) -> float:
        """Tier 4: range-vs-single under the configured range_mode.

        ``single_is_gt`` records whether the single side was the ground
        truth (the first :meth:`compare` argument), so the directional
        precision gate (``gt_loose``) is oriented the same way it is in the
        single-vs-single and range-vs-range paths.
        """
        if self.range_mode == "strict":
            return 0.0

        # Month/day resolution gate (precision_mode), applied per endpoint
        # with ground truth in the correct position — mirroring
        # _compare_range_range. Without this a reduced-precision single
        # (e.g. 'Jan 2024', whose day is fabricated to the 1st) would land
        # inside a day-grain range and score credit even under the default
        # 'exact' mode, the same score-inflating fabrication the gate
        # exists to refuse on the single-vs-single path.
        if single_is_gt:
            resolution_ok = self._resolution_ok(
                single, rng.start
            ) and self._resolution_ok(single, rng.end)
        else:
            resolution_ok = self._resolution_ok(
                rng.start, single
            ) and self._resolution_ok(rng.end, single)
        if not resolution_ok:
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
            # Normalize all three to comparable naive days: endpoints and
            # the single may differ in tz-awareness.
            s = self._normalize_day(single.dt)
            lo = self._normalize_day(rng.start.dt)
            hi = self._normalize_day(rng.end.dt)
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

        When ``lo <= hi`` the span is the ordinary closed interval. When
        ``lo > hi`` the range wraps the year boundary in m/d space (e.g.
        Dec 20 → Jan 5, common for fiscal/holiday ranges in IDP data), so
        membership is the union of the two ends:
        ``target >= lo`` (late-year tail) or ``target <= hi`` (early-year
        head). Endpoints are inclusive either way.
        """
        if lo <= hi:
            return lo <= target <= hi
        return target >= lo or target <= hi

    def _compare_singles(
        self, a: _ParsedSingle, b: _ParsedSingle
    ) -> float:
        """Score two single dates across the year-presence and resolution axes.

        ``a`` is the ground truth (first :meth:`compare` argument). Two
        independent gates run before any value comparison:

        * month/day resolution (``precision_mode``), and
        * year presence (``allow_partial_year``, via
          :meth:`_partial_year_multiplier`).

        If both gates pass, the fields that *both* sides specify must
        agree; year-bearing day-grain pairs additionally honor
        ``tolerance``.
        """
        # Axis 1 — month/day resolution.
        if not self._resolution_ok(a, b):
            return 0.0

        # Axis 2 — year presence (carries the 0.7 partial-year credit).
        year_multiplier = self._partial_year_multiplier(a.has_year == b.has_year)
        if year_multiplier == 0.0:
            return 0.0

        if not self._single_values_agree(a, b):
            return 0.0

        return year_multiplier

    def _resolution_ok(self, a: _ParsedSingle, b: _ParsedSingle) -> bool:
        """Whether a month/day resolution mismatch is permitted.

        ``a`` is ground truth. Equal resolution is always fine; otherwise
        ``precision_mode`` decides:

        * ``"exact"`` — never (resolutions must match);
        * ``"gt_loose"`` — only if the prediction ``b`` is *finer* than
          the ground truth ``a`` (``b`` may add precision, not drop it);
        * ``"overlap"`` — either side may be coarser.
        """
        if a.md_resolution == b.md_resolution:
            return True
        if self.precision_mode == "exact":
            return False
        if self.precision_mode == "overlap":
            return True
        # gt_loose
        return b.md_resolution >= a.md_resolution

    def _single_values_agree(
        self, a: _ParsedSingle, b: _ParsedSingle
    ) -> bool:
        """Whether two singles agree on every field both sides specify.

        Year-bearing day-grain pairs go through the ``tolerance`` window
        (which spans month/year boundaries); every other pairing is exact
        on the fields present at the common (coarser) grain.
        """
        if a.has_year and b.has_year:
            if a.md_resolution == 2 and b.md_resolution == 2:
                # Both full dates: tolerance-aware comparison.
                a_dt, b_dt = self._align_timezones(a.dt, b.dt)
                if not self._has_subday_tolerance():
                    # Whole-day (or zero) tolerance keeps same-calendar-day
                    # semantics: floor to midnight so intra-day times are
                    # ignored and the window counts whole days.
                    a_dt = self._truncate_day(a_dt)
                    b_dt = self._truncate_day(b_dt)
                # A sub-day tolerance (e.g. 1.5 days = 36h) means the caller
                # cares about real elapsed time, so compare actual
                # timestamps without flooring.
                return abs(a_dt - b_dt) <= self.tolerance
            if a.dt.year != b.dt.year:
                return False
        if a.has_month and b.has_month and a.dt.month != b.dt.month:
            return False
        if a.has_day and b.has_day and a.dt.day != b.dt.day:
            return False
        return True

    def _has_subday_tolerance(self) -> bool:
        """Whether ``tolerance`` carries a sub-day (hours/minutes) component."""
        return self.tolerance.total_seconds() % 86400 != 0

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
        # Normalize to comparable naive days so endpoints from different
        # ranges (possibly mixed tz-awareness) can be min/max'd together.
        a_lo = self._normalize_day(a.start.dt)
        a_hi = self._normalize_day(a.end.dt)
        b_lo = self._normalize_day(b.start.dt)
        b_hi = self._normalize_day(b.end.dt)

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

    @staticmethod
    def _md_equal(dt1: datetime, dt2: datetime) -> bool:
        """Whether two datetimes share a (month, day) — ignores year/time."""
        return (dt1.month, dt1.day) == (dt2.month, dt2.day)

    @classmethod
    def _md_jaccard(cls, a: _ParsedRange, b: _ParsedRange) -> float:
        """Jaccard overlap of two ranges in (month, day) space.

        Used when year-presence differs: the year-less side's year is a
        1900 placeholder, so overlap is measured over the set of
        ``(month, day)`` pairs each range spans rather than absolute days.
        """
        a_days = cls._md_set(a)
        b_days = cls._md_set(b)
        union = a_days | b_days
        if not union:
            return 0.0
        return len(a_days & b_days) / len(union)

    @staticmethod
    def _md_set(rng: _ParsedRange) -> set:
        """The set of (month, day) pairs a range covers, inclusive.

        Walks day by day from start to end. Bounded by a one-year cap so
        a malformed multi-year span can't run away.
        """
        start = rng.start.dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = rng.end.dt.replace(hour=0, minute=0, second=0, microsecond=0)
        days = (end - start).days
        # Year-less ranges can't wrap (parser enforces start <= end and
        # both default to 1900); cap the walk at a full year defensively.
        days = min(days, 366)
        out = set()
        cur = start
        for _ in range(days + 1):
            out.add((cur.month, cur.day))
            cur += timedelta(days=1)
        return out

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

        # Reject pathologically long input before any parsing. A real date
        # string is well under this; a huge value is malformed and would
        # otherwise cost dateutil a full scan (and could split into two
        # huge range halves). Over-length degrades to 0.0 like any other
        # parse failure.
        if len(s) > _MAX_INPUT_LEN:
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

        # A string that carries a range-delimiter signal but didn't parse
        # as a valid range is a malformed/truncated range, not a single
        # date. Falling through to a single parse here would let dateutil
        # silently swallow a dangling dash (``'- 10/24/16'``) and score it
        # as a clean date.
        if self._has_range_delim_signal(s):
            return None

        return self._try_parse_single(s, dayfirst=dayfirst)

    @staticmethod
    def _has_range_delim_signal(s: str) -> bool:
        """Whether ``s`` looks like it was meant to be a range.

        Catches both the configured delimiters appearing internally and a
        dangling bare dash at either edge. Legitimate single dates put
        their dashes *between* digits (``2025-01-01``, ``10-24-2016``), so
        an edge dash only shows up on truncated range input.
        """
        if any(delim in s for delim in _RANGE_DELIMS):
            return True
        return s.startswith("-") or s.endswith("-")

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
            # Align before the ordering check: endpoints may differ in
            # tz-awareness (one ISO-with-offset, one naive), which would
            # otherwise raise TypeError on the comparison.
            left_dt, right_dt = self._align_timezones(left_p.dt, right_p.dt)
            if left_dt > right_dt:
                return None
            return _ParsedRange(start=left_p, end=right_p)
        return None

    def _try_parse_single(
        self, s: str, dayfirst: bool
    ) -> Optional[_ParsedSingle]:
        """Parse one side as a single date (or ``None`` on failure).

        Year/month/day presence is detected by parsing twice with default
        dates that differ in *all three* components: any field the parser
        had to borrow from the default reveals itself by disagreeing
        between the two parses. This is what lets reduced-resolution
        inputs (``'Jan 2024'``, ``'2024'``) be told apart from full dates
        rather than silently fabricating the missing fields.
        """
        if not s:
            return None

        # Year-first layouts (ISO and ``YYYY/MM/DD``) fix month-then-day
        # order, so the day-first interpretation would corrupt them. Pin
        # those to month-first regardless of the requested ``dayfirst``.
        if _YEAR_FIRST_RE.match(s):
            dayfirst = False

        try:
            dt_lo = _dateutil_parser.parse(
                s, default=datetime(1900, 1, 1), dayfirst=dayfirst
            )
            # Default differs in year, month, AND day so each component's
            # presence can be probed independently.
            dt_hi = _dateutil_parser.parse(
                s, default=datetime(2099, 6, 15), dayfirst=dayfirst
            )
        except (ValueError, OverflowError, TypeError):
            return None

        has_year = dt_lo.year == dt_hi.year
        has_month = dt_lo.month == dt_hi.month
        has_day = dt_lo.day == dt_hi.day

        # Reject time-only inputs ('12:30 PM', '10/45AM' etc.): no date
        # component at all was specified, so everything came from the
        # default and only the time survives.
        if not (has_year or has_month or has_day):
            return None

        return _ParsedSingle(
            dt=dt_lo, has_year=has_year, has_month=has_month, has_day=has_day
        )

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

    @staticmethod
    def _normalize_day(dt: datetime) -> datetime:
        """Collapse to a tz-naive midnight so any two dates are comparable.

        Aware datetimes are converted to UTC first; the result is always
        naive, so values that started with differing tz-awareness can be
        ordered against each other without raising ``TypeError``.
        """
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
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
            f"range_mode={self.range_mode!r}, "
            f"precision_mode={self.precision_mode!r})"
        )
