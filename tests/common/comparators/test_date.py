"""Tests for DateComparator.

Organized by tier. Tier 1 (surface form) and Tier 5 (must-not-match) are
covered by parametrized format-zoo and unparseable-input tests. Tiers
2/3/4/4b each have a dedicated class.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from stickler.comparators import DateComparator

# ---------------------------------------------------------------------------
# Basics — Tier 1 sanity
# ---------------------------------------------------------------------------


class TestDateComparatorBasics:
    def setup_method(self):
        self.comparator = DateComparator()

    def test_identical_iso_strings(self):
        assert self.comparator.compare("2025-01-01", "2025-01-01") == 1.0

    def test_canonical_vs_non_canonical(self):
        assert self.comparator.compare("2025-01-01", "Jan 1, 2025") == 1.0
        assert self.comparator.compare("2025-01-01", "January 1, 2025") == 1.0
        assert self.comparator.compare("2025-01-01", "2025-01-01T00:00:00Z") == 1.0
        assert self.comparator.compare("2025-01-01", "1 Jan 2025") == 1.0

    def test_different_dates(self):
        assert self.comparator.compare("2025-01-01", "2025-01-02") == 0.0
        assert self.comparator.compare("2025-01-01", "2026-01-01") == 0.0
        assert self.comparator.compare("2025-01-01", "2025-02-01") == 0.0

    def test_none_values(self):
        assert self.comparator.compare(None, None) == 1.0
        assert self.comparator.compare("2025-01-01", None) == 0.0
        assert self.comparator.compare(None, "2025-01-01") == 0.0

    def test_datetime_objects(self):
        dt = datetime(2025, 1, 1, 12, 0, 0)
        assert self.comparator.compare(dt, "2025-01-01") == 1.0
        assert self.comparator.compare(dt, dt) == 1.0

    def test_date_objects(self):
        d = date(2025, 1, 1)
        assert self.comparator.compare(d, "2025-01-01") == 1.0
        assert self.comparator.compare(d, date(2025, 1, 1)) == 1.0


# ---------------------------------------------------------------------------
# Tier 1 — surface-form normalization (parametrized)
# ---------------------------------------------------------------------------


EQUIVALENT_JAN_1_2025 = [
    "2025-01-01",
    "2025-1-1",
    "2025/01/01",
    "Jan 1, 2025",
    "Jan 1 2025",
    "January 1, 2025",
    "1 January 2025",
    "1 Jan 2025",
    "Jan. 1, 2025",
    "2025-01-01T00:00:00",
    "2025-01-01T12:00:00",
    "2025-01-01 12:00:00",
    "2025-01-01T00:00:00Z",
    "2025-01-01T00:00:00+00:00",
    "Wed Jan 1 2025",
    "Wednesday, January 1, 2025",
    "Jan 1st, 2025",
    "1st Jan 2025",
    "  2025-01-01  ",
    "\t2025-01-01\n",
    "20250101",
]


class TestTier1FormatZoo:
    @pytest.mark.parametrize("variant", EQUIVALENT_JAN_1_2025)
    def test_equivalent_to_iso(self, variant):
        cmp = DateComparator()
        assert cmp.compare(variant, "2025-01-01") == 1.0, (
            f"Expected {variant!r} to equal 2025-01-01"
        )

    @pytest.mark.parametrize("case", ["jan 1, 2025", "JAN 1, 2025", "Jan 1, 2025"])
    def test_case_insensitivity_in_month_names(self, case):
        cmp = DateComparator()
        assert cmp.compare(case, "2025-01-01") == 1.0


class TestTier1ZeroPadding:
    """Spec Tier 1 #3: zero-padding independent on month and day."""

    def test_unpadded_vs_padded_same_year_format(self):
        cmp = DateComparator()
        assert cmp.compare("2/1/2016", "02/01/2016") == 1.0

    def test_unpadded_with_two_digit_year_vs_padded_four_digit(self):
        cmp = DateComparator()
        assert cmp.compare("2/1/2016", "02/01/16") == 1.0

    def test_unpadded_iso(self):
        cmp = DateComparator()
        assert cmp.compare("2016-2-1", "2016-02-01") == 1.0


class TestTier1WeekdayPrefix:
    """Spec Tier 1 #8: leading weekday tokens are stripped."""

    def test_short_weekday_prefix(self):
        cmp = DateComparator()
        assert cmp.compare("Mon 10/24/16", "10/24/16") == 1.0

    def test_long_weekday_prefix(self):
        cmp = DateComparator()
        assert cmp.compare("Monday October 24, 2016", "10/24/16") == 1.0


class TestTier1TwoDigitYear:
    """Spec Tier 1 #4: two-digit ↔ four-digit year. Pivot is dateutil's default."""

    def test_two_digit_modern(self):
        cmp = DateComparator()
        assert cmp.compare("10/24/2016", "10/24/16") == 1.0

    def test_two_digit_low_maps_to_2000s(self):
        cmp = DateComparator()
        # NOTE: dateutil's two-digit pivot is a sliding 50-year window
        # centered on the current year, NOT a fixed cutoff. '25' resolves
        # to 2025 today; this assertion is stable for ~25 more years but
        # will flip once 2025 falls outside the window (~2050s). It's not
        # hermetic — if it fails in the future, that's the pivot sliding.
        assert cmp.compare("1/1/25", "2025-01-01") == 1.0


# ---------------------------------------------------------------------------
# Tier 2 — both year-less, same m/d
# ---------------------------------------------------------------------------


class TestTier2BothYearLess:
    """Both inputs lack a year and share m/d → 1.0 regardless of options."""

    def test_named_vs_numeric(self):
        cmp = DateComparator()
        assert cmp.compare("Oct 24", "10/24") == 1.0

    def test_numeric_vs_numeric(self):
        cmp = DateComparator()
        # Both numeric m/d, both unambiguous (no year on either side).
        # Use a day > 12 so the layout is unambiguous either direction.
        assert cmp.compare("10/24", "10/24") == 1.0
        assert cmp.compare("10/24", "10/25") == 0.0

    def test_partial_year_flag_does_not_change_tier_2(self):
        """Tier 2 score is 1.0 whether or not allow_partial_year is set."""
        strict = DateComparator()
        permissive = DateComparator(allow_partial_year=True)
        assert strict.compare("Oct 24", "10/24") == 1.0
        assert permissive.compare("Oct 24", "10/24") == 1.0

    def test_different_md_year_less(self):
        cmp = DateComparator()
        assert cmp.compare("Oct 24", "10/25") == 0.0


# ---------------------------------------------------------------------------
# Tier 3 — one side year-less, m/d match
# ---------------------------------------------------------------------------


class TestTier3PartialYear:
    """One side lacks a year. Score depends on allow_partial_year."""

    def test_default_returns_zero(self):
        cmp = DateComparator()
        assert cmp.compare("11/4", "Nov 4 2016") == 0.0
        assert cmp.compare("Oct 24", "10/24/16") == 0.0

    def test_permissive_returns_partial(self):
        cmp = DateComparator(allow_partial_year=True)
        assert cmp.compare("11/4", "Nov 4 2016") == 0.7
        assert cmp.compare("Oct 24", "10/24/16") == 0.7
        assert cmp.compare("11/04", "11/4/2016") == 0.7

    def test_permissive_md_mismatch_still_zero(self):
        cmp = DateComparator(allow_partial_year=True)
        # Same year-less side, but the year-bearing side has different m/d.
        assert cmp.compare("Oct 24", "10/25/16") == 0.0

    def test_year_hallucination_default_strict(self):
        """Real failure mode: GT '11/03' vs pred '11/03/2012' (179× in eval)."""
        strict = DateComparator()
        permissive = DateComparator(allow_partial_year=True)
        assert strict.compare("11/03", "11/03/2012") == 0.0
        assert permissive.compare("11/03", "11/03/2012") == 0.7


# ---------------------------------------------------------------------------
# Tier 4 — single date contained in range
# ---------------------------------------------------------------------------


class TestTier4SingleInRange:
    """Default range_mode is 'graded'."""

    def test_single_inside_range(self):
        cmp = DateComparator()
        assert cmp.compare("10/28/16", "10/24/16 to 10/30/16") == 0.5

    def test_single_at_range_start_inclusive(self):
        cmp = DateComparator()
        assert cmp.compare("10/24/16", "10/24/16 to 10/30/16") == 0.5

    def test_single_at_range_end_inclusive(self):
        cmp = DateComparator()
        assert cmp.compare("10/30/16", "10/24/16 to 10/30/16") == 0.5

    def test_single_outside_range(self):
        cmp = DateComparator()
        assert cmp.compare("11/15/16", "10/24/16 to 10/30/16") == 0.0

    def test_range_first_arg(self):
        cmp = DateComparator()
        assert cmp.compare("10/24/16 to 10/30/16", "10/28/16") == 0.5

    def test_through_delimiter(self):
        cmp = DateComparator()
        assert cmp.compare("10/28/16", "10/24/16 through 10/30/16") == 0.5

    def test_dash_delimiter_with_spaces(self):
        cmp = DateComparator()
        assert cmp.compare("10/28/16", "10/24/16 - 10/30/16") == 0.5

    def test_iso_not_misread_as_range(self):
        cmp = DateComparator()
        # Identity check, plus a discriminating one: if the internal dash
        # were treated as a range delimiter these distinct days would not
        # compare as a clean single-vs-single 0.0.
        assert cmp.compare("2025-01-01", "2025-01-01") == 1.0
        assert cmp.compare("2025-01-01", "2025-01-02") == 0.0


class TestTier4bRangeVsRange:
    """Default range_mode is 'graded' (Jaccard)."""

    def test_endpoints_match_full_score(self):
        cmp = DateComparator()
        assert (
            cmp.compare(
                "10/24/16 to 10/30/16", "10/24/2016 - 10/30/2016"
            )
            == 1.0
        )

    def test_endpoints_match_with_dash(self):
        cmp = DateComparator()
        assert (
            cmp.compare(
                "09-12-16 to 09-15-16", "09-12-2016 to 09-15-2016"
            )
            == 1.0
        )

    def test_partial_overlap_uses_jaccard(self):
        cmp = DateComparator()
        # Oct 24-30 (7 days) vs Oct 24-31 (8 days) → 7/8.
        result = cmp.compare("10/24/16 to 10/30/16", "10/24/16 to 10/31/16")
        assert result == pytest.approx(7 / 8)

    def test_no_overlap_zero(self):
        cmp = DateComparator()
        assert (
            cmp.compare(
                "10/24/16 to 10/30/16", "12/01/16 to 12/05/16"
            )
            == 0.0
        )


# ---------------------------------------------------------------------------
# range_mode coverage
# ---------------------------------------------------------------------------


class TestRangeModeStrict:
    def setup_method(self):
        self.cmp = DateComparator(range_mode="strict")

    def test_single_in_range_zero(self):
        assert self.cmp.compare("10/28/16", "10/24/16 to 10/30/16") == 0.0

    def test_range_endpoints_exact(self):
        assert (
            self.cmp.compare(
                "10/24/16 to 10/30/16", "10/24/2016 - 10/30/2016"
            )
            == 1.0
        )

    def test_range_partial_overlap_zero(self):
        assert (
            self.cmp.compare(
                "10/24/16 to 10/30/16", "10/24/16 to 10/31/16"
            )
            == 0.0
        )

    def test_singles_unaffected(self):
        assert self.cmp.compare("10/24/16", "10/24/16") == 1.0


class TestRangeModeReject:
    def setup_method(self):
        self.cmp = DateComparator(range_mode="reject")

    def test_single_in_range_zero(self):
        assert self.cmp.compare("10/28/16", "10/24/16 to 10/30/16") == 0.0

    def test_range_vs_range_zero_even_when_endpoints_match(self):
        assert (
            self.cmp.compare(
                "10/24/16 to 10/30/16", "10/24/2016 - 10/30/2016"
            )
            == 0.0
        )

    def test_singles_unaffected(self):
        assert self.cmp.compare("10/24/16", "10/24/16") == 1.0

    def test_degenerate_range_not_collapsed(self):
        """X to X parses as a range under reject; doesn't collapse to single."""
        assert self.cmp.compare("10/28/16 to 10/28/16", "10/28/16") == 0.0


class TestRangeModeContains:
    def setup_method(self):
        self.cmp = DateComparator(range_mode="contains")

    def test_single_inside_range_full(self):
        assert self.cmp.compare("10/28/16", "10/24/16 to 10/30/16") == 1.0

    def test_single_at_endpoint_full(self):
        assert self.cmp.compare("10/24/16", "10/24/16 to 10/30/16") == 1.0
        assert self.cmp.compare("10/30/16", "10/24/16 to 10/30/16") == 1.0

    def test_single_outside_zero(self):
        assert self.cmp.compare("11/15/16", "10/24/16 to 10/30/16") == 0.0

    def test_range_endpoints_exact(self):
        assert (
            self.cmp.compare(
                "10/24/16 to 10/30/16", "10/24/2016 - 10/30/2016"
            )
            == 1.0
        )

    def test_range_partial_overlap_zero(self):
        assert (
            self.cmp.compare(
                "10/24/16 to 10/30/16", "10/24/16 to 10/31/16"
            )
            == 0.0
        )


class TestRangeModeGradedJaccard:
    def setup_method(self):
        self.cmp = DateComparator(range_mode="graded")

    def test_identical_ranges_full(self):
        assert (
            self.cmp.compare(
                "10/24/16 to 10/30/16", "10/24/16 to 10/30/16"
            )
            == 1.0
        )

    def test_off_by_one_endpoint(self):
        # 7-day range vs 8-day range, 7-day overlap, 8-day union.
        assert (
            self.cmp.compare(
                "10/24/16 to 10/30/16", "10/24/16 to 10/31/16"
            )
            == pytest.approx(7 / 8)
        )

    def test_shifted_overlapping(self):
        # Oct 1-10 (10) vs Oct 6-15 (10); overlap Oct 6-10 (5);
        # union Oct 1-15 (15) → 5/15 = 1/3.
        # Pin dayfirst=False so the dayfirst=None max-of-both semantics
        # doesn't interact with Jaccard math.
        cmp = DateComparator(range_mode="graded", dayfirst=False)
        assert (
            cmp.compare(
                "10/01/2016 to 10/10/2016", "10/06/2016 to 10/15/2016"
            )
            == pytest.approx(5 / 15)
        )

    def test_no_overlap_zero(self):
        assert (
            self.cmp.compare(
                "10/24/16 to 10/30/16", "12/01/16 to 12/05/16"
            )
            == 0.0
        )


# ---------------------------------------------------------------------------
# Single-day range collapse
# ---------------------------------------------------------------------------


class TestSingleDayRangeCollapse:
    """``X to X`` collapses to a single date for non-reject modes so the
    degenerate range and the bare single compare consistently."""

    @pytest.mark.parametrize(
        "mode", ["strict", "contains", "graded"]
    )
    def test_collapse_against_single(self, mode):
        cmp = DateComparator(range_mode=mode)
        assert cmp.compare("10/28/16 to 10/28/16", "10/28/16") == 1.0

    @pytest.mark.parametrize(
        "mode", ["strict", "contains", "graded"]
    )
    def test_collapse_against_collapse(self, mode):
        cmp = DateComparator(range_mode=mode)
        assert (
            cmp.compare("10/28/16 to 10/28/16", "10/28/16 to 10/28/16") == 1.0
        )

    def test_no_collapse_under_reject(self):
        cmp = DateComparator(range_mode="reject")
        assert cmp.compare("10/28/16 to 10/28/16", "10/28/16") == 0.0


# ---------------------------------------------------------------------------
# Malformed range delimiters (stray/dangling dash)
# ---------------------------------------------------------------------------


class TestMalformedRangeDelimiter:
    """A stray range delimiter must not slip through as a single date.

    `' - 10/24/16'` previously fell through to a single parse, dateutil
    quietly ignored the dangling dash, and it scored `1.0` against
    `'10/24/16'` (#141 review). A string carrying a range delimiter is a
    malformed range, not a single date.
    """

    @pytest.mark.parametrize(
        "garbage",
        [
            " - 10/24/16",
            "10/24/16 - ",
            "-10/24/16",
            "10/24/16-",
            "10/24/16 to ",
            " to 10/24/16",
            "10/24/16 through ",
        ],
    )
    def test_dangling_delimiter_scores_zero_graded(self, garbage):
        cmp = DateComparator()
        assert cmp.compare(garbage, "10/24/16") == 0.0
        assert cmp.compare("10/24/16", garbage) == 0.0

    @pytest.mark.parametrize("garbage", [" - 10/24/16", "10/24/16 - "])
    def test_dangling_delimiter_scores_zero_reject(self, garbage):
        """reject mode's 'any range input = 0' contract must hold here too."""
        cmp = DateComparator(range_mode="reject")
        assert cmp.compare(garbage, "10/24/16") == 0.0

    def test_legit_iso_with_internal_dash_unaffected(self):
        cmp = DateComparator()
        assert cmp.compare("2025-01-01", "2025-01-01") == 1.0
        assert cmp.compare("2025-01-01", "2025-01-02") == 0.0

    def test_legit_dash_separated_date_unaffected(self):
        cmp = DateComparator()
        assert cmp.compare("10-24-2016", "10/24/16") == 1.0

    def test_legit_dash_range_unaffected(self):
        cmp = DateComparator()
        assert (
            cmp.compare("09-12-16 to 09-15-16", "09-12-2016 to 09-15-2016")
            == 1.0
        )

    def test_legit_spaced_dash_range_unaffected(self):
        cmp = DateComparator()
        assert cmp.compare("10/28/16", "10/24/16 - 10/30/16") == 0.5


# ---------------------------------------------------------------------------
# Year-presence multiplier interactions with ranges
# ---------------------------------------------------------------------------


class TestYearPresenceMultiplierInRanges:
    """allow_partial_year scales range scores when year-presence differs."""

    def test_yearless_single_in_year_range_default_zero(self):
        cmp = DateComparator()
        assert cmp.compare("Oct 28", "10/24/16 to 10/30/16") == 0.0

    def test_yearless_single_in_year_range_partial_year_graded(self):
        cmp = DateComparator(allow_partial_year=True)
        # graded base 0.5 × 0.7 = 0.35
        assert cmp.compare("Oct 28", "10/24/16 to 10/30/16") == pytest.approx(
            0.5 * 0.7
        )

    def test_yearless_single_in_year_range_partial_year_contains(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="contains")
        # contains base 1.0 × 0.7 = 0.7
        assert cmp.compare("Oct 28", "10/24/16 to 10/30/16") == pytest.approx(
            0.7
        )

    def test_yearless_single_in_year_range_partial_year_strict(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="strict")
        # strict refuses range-vs-single regardless of year alignment
        assert cmp.compare("Oct 28", "10/24/16 to 10/30/16") == 0.0

    def test_year_range_vs_year_range_partial_year_irrelevant(self):
        """Both sides have year → multiplier is 1.0."""
        cmp = DateComparator(allow_partial_year=True)
        assert (
            cmp.compare(
                "10/24/16 to 10/30/16", "10/24/2016 - 10/30/2016"
            )
            == 1.0
        )


class TestYearMismatchRangeVsRange:
    """Year-less range vs year-bearing range under allow_partial_year.

    When year-presence differs, range-vs-range must fall back to (month,
    day) comparison just like range-vs-single does. Without it the 1900
    placeholder zeroed every overlap, making the documented `0.7` path
    unreachable (#141 review).
    """

    def test_graded_exact_md_overlap(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="graded")
        # Same m/d span → Jaccard 1.0 × partial-year 0.7.
        assert cmp.compare(
            "Oct 24 to Oct 30", "10/24/16 to 10/30/16"
        ) == pytest.approx(0.7)

    def test_graded_partial_md_overlap(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="graded")
        # 7-day vs 8-day, 7 overlap, 8 union → 7/8 × 0.7.
        assert cmp.compare(
            "Oct 24 to Oct 30", "10/24/16 to 10/31/16"
        ) == pytest.approx((7 / 8) * 0.7)

    def test_contains_exact_endpoints(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="contains")
        # Endpoints equal in m/d → 1.0 × 0.7.
        assert cmp.compare(
            "Oct 24 to Oct 30", "10/24/16 to 10/30/16"
        ) == pytest.approx(0.7)

    def test_contains_endpoints_differ(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="contains")
        assert cmp.compare("Oct 24 to Oct 30", "10/24/16 to 10/31/16") == 0.0

    def test_no_md_overlap_is_zero(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="graded")
        assert cmp.compare("Oct 24 to Oct 30", "12/01/16 to 12/05/16") == 0.0

    def test_default_partial_off_is_zero(self):
        cmp = DateComparator(range_mode="graded")
        # allow_partial_year defaults False → year mismatch zeros out.
        assert cmp.compare("Oct 24 to Oct 30", "10/24/16 to 10/30/16") == 0.0

    def test_both_year_bearing_unaffected(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="graded")
        assert (
            cmp.compare("10/24/16 to 10/30/16", "10/24/2016 - 10/30/2016")
            == 1.0
        )
        assert cmp.compare(
            "10/24/16 to 10/30/16", "10/24/16 to 10/31/16"
        ) == pytest.approx(7 / 8)


class TestYearEndWrapMonthDayRange:
    """Year-less single vs a year-bearing range that wraps year-end.

    When year-presence differs the comparison drops to (month, day)
    space; a range like Dec 20 → Jan 5 wraps there, so containment must
    treat lo > hi as a wrap-around span (#141 review).
    """

    def test_single_inside_wrap_before_boundary(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="contains")
        assert cmp.compare(
            "Dec 25", "Dec 20, 2024 to Jan 5, 2025"
        ) == pytest.approx(0.7)

    def test_single_inside_wrap_after_boundary(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="contains")
        assert cmp.compare(
            "Jan 2", "Dec 20, 2024 to Jan 5, 2025"
        ) == pytest.approx(0.7)

    def test_single_on_wrap_endpoints_inclusive(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="contains")
        assert cmp.compare(
            "Dec 20", "Dec 20, 2024 to Jan 5, 2025"
        ) == pytest.approx(0.7)
        assert cmp.compare(
            "Jan 5", "Dec 20, 2024 to Jan 5, 2025"
        ) == pytest.approx(0.7)

    def test_single_outside_wrap_is_zero(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="contains")
        # Jul 4 is in the excluded middle of the wrap span.
        assert cmp.compare("Jul 4", "Dec 20, 2024 to Jan 5, 2025") == 0.0

    def test_wrap_graded_mode(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="graded")
        # inside → graded base 0.5 × partial-year 0.7 = 0.35
        assert cmp.compare(
            "Dec 25", "Dec 20, 2024 to Jan 5, 2025"
        ) == pytest.approx(0.35)

    def test_non_wrap_md_range_unaffected(self):
        cmp = DateComparator(allow_partial_year=True, range_mode="contains")
        assert cmp.compare(
            "Oct 28", "10/24/16 to 10/30/16"
        ) == pytest.approx(0.7)
        assert cmp.compare("Nov 15", "10/24/16 to 10/30/16") == 0.0


# ---------------------------------------------------------------------------
# Reduced-resolution dates (precision_mode)
# ---------------------------------------------------------------------------


class TestPrecisionModeExactDefault:
    """Default precision_mode='exact': resolutions must match.

    A reduced-precision input (missing month or day) must NOT score 1.0
    against a fuller date — the missing component is fabricated by the
    parser, and for an eval library that's a score-inflating false
    positive (issue #141 review, blocking).
    """

    def test_month_grain_vs_day_grain_is_zero(self):
        cmp = DateComparator()
        # GT 'Jan 2024' lacks a day; pred 'Jan 1, 2024' fabricates day=1.
        assert cmp.compare("Jan 2024", "Jan 1, 2024") == 0.0

    def test_year_grain_vs_day_grain_is_zero(self):
        cmp = DateComparator()
        # GT '2024' lacks month+day; pred '2024-01-01' fabricates both.
        assert cmp.compare("2024", "2024-01-01") == 0.0

    def test_reduced_precision_compounds_in_ranges_is_zero(self):
        cmp = DateComparator()
        assert (
            cmp.compare("Jan 2024 to Mar 2024", "1/1/2024 to 3/1/2024") == 0.0
        )

    def test_same_resolution_year_grain_still_matches(self):
        cmp = DateComparator()
        assert cmp.compare("2024", "2024") == 1.0

    def test_same_resolution_month_grain_still_matches(self):
        cmp = DateComparator()
        assert cmp.compare("Jan 2024", "Jan 2024") == 1.0
        assert cmp.compare("Jan 2024", "2024-01") == 1.0

    def test_same_resolution_month_grain_value_mismatch_is_zero(self):
        cmp = DateComparator()
        assert cmp.compare("Jan 2024", "Feb 2024") == 0.0

    def test_same_resolution_year_grain_value_mismatch_is_zero(self):
        cmp = DateComparator()
        assert cmp.compare("2024", "2025") == 0.0


class TestPrecisionModeGtLoose:
    """precision_mode='gt_loose': pred may be FINER than GT, not coarser.

    GT anchors the required resolution. A prediction that is more precise
    than GT (and consistent at GT's grain) gets full credit; a prediction
    that is less precise than GT under-specifies the truth and misses.
    """

    def test_pred_finer_than_gt_matches(self):
        cmp = DateComparator(precision_mode="gt_loose")
        # GT month-grain, pred day-grain, consistent at month → 1.0
        assert cmp.compare("Jan 2024", "Jan 1, 2024") == 1.0
        assert cmp.compare("2024", "2024-06-15") == 1.0

    def test_pred_coarser_than_gt_is_zero(self):
        cmp = DateComparator(precision_mode="gt_loose")
        # GT day-grain, pred month-grain → pred under-specifies → 0.0
        assert cmp.compare("Jan 1, 2024", "Jan 2024") == 0.0
        assert cmp.compare("2024-01-01", "2024") == 0.0

    def test_finer_pred_inconsistent_at_gt_grain_is_zero(self):
        cmp = DateComparator(precision_mode="gt_loose")
        # GT 'Jan 2024'; pred 'Feb 1, 2024' disagrees on month → 0.0
        assert cmp.compare("Jan 2024", "Feb 1, 2024") == 0.0

    def test_same_resolution_unaffected(self):
        cmp = DateComparator(precision_mode="gt_loose")
        assert cmp.compare("Jan 2024", "Jan 2024") == 1.0
        assert cmp.compare("2024-01-01", "Jan 1, 2024") == 1.0


class TestPrecisionModeOverlap:
    """precision_mode='overlap': either side may be coarser (symmetric)."""

    def test_pred_finer_than_gt_matches(self):
        cmp = DateComparator(precision_mode="overlap")
        assert cmp.compare("Jan 2024", "Jan 1, 2024") == 1.0

    def test_pred_coarser_than_gt_matches(self):
        cmp = DateComparator(precision_mode="overlap")
        assert cmp.compare("Jan 1, 2024", "Jan 2024") == 1.0
        assert cmp.compare("2024-01-01", "2024") == 1.0

    def test_inconsistent_at_coarser_grain_is_zero(self):
        cmp = DateComparator(precision_mode="overlap")
        assert cmp.compare("Jan 2024", "Feb 1, 2024") == 0.0
        assert cmp.compare("2024", "2025-01-01") == 0.0


class TestPrecisionModeValidation:
    @pytest.mark.parametrize(
        "bad", ["", "EXACT", "loose", "gtloose", "yes", 1, None]
    )
    def test_invalid_precision_mode_rejected(self, bad):
        with pytest.raises((ValueError, TypeError), match="precision_mode"):
            DateComparator(precision_mode=bad)  # type: ignore[arg-type]

    def test_error_message_lists_valid_values(self):
        """The error echoes the bad value and all valid options."""
        with pytest.raises(ValueError) as exc:
            DateComparator(precision_mode="loose")  # type: ignore[arg-type]
        msg = str(exc.value)
        assert "loose" in msg
        for valid in ("exact", "gt_loose", "overlap"):
            assert valid in msg

    def test_default_is_exact(self):
        assert DateComparator().precision_mode == "exact"


# ---------------------------------------------------------------------------
# Tier 5 — must-not-match
# ---------------------------------------------------------------------------


class TestTier5DifferentDates:
    def test_off_by_one_day(self):
        cmp = DateComparator()
        assert cmp.compare("10/09/12", "10/10/12") == 0.0

    def test_different_year(self):
        cmp = DateComparator()
        assert cmp.compare("10/24/15", "10/24/16") == 0.0


class TestTier5Ambiguity:
    """dayfirst=None refuses; True/False parses."""

    def test_ambiguous_pair_refused_by_default(self):
        """Both sides ambiguous (m/d both ≤ 12, swapped) → 0.0."""
        cmp = DateComparator()
        # gt="10/03/16" vs pred="03/10/16": dayfirst-flip ambiguous on each.
        assert cmp.compare("10/03/16", "03/10/16") == 0.0

    def test_dayfirst_true_parses_both(self):
        cmp = DateComparator(dayfirst=True)
        # Both interpreted day-first → "10/03/16" = 10 Mar 2016,
        # "03/10/16" = 3 Oct 2016. Different dates, score 0.0.
        assert cmp.compare("10/03/16", "03/10/16") == 0.0

    def test_dayfirst_true_self_match(self):
        cmp = DateComparator(dayfirst=True)
        assert cmp.compare("10/03/16", "10/03/16") == 1.0

    def test_dayfirst_false_parses_both(self):
        cmp = DateComparator(dayfirst=False)
        # Both month-first → "10/03/16" = Oct 3 2016,
        # "03/10/16" = Mar 10 2016. Different dates, score 0.0.
        assert cmp.compare("10/03/16", "03/10/16") == 0.0

    def test_unambiguous_layout_unaffected_by_dayfirst_none(self):
        """When at least one field is >12, the layout is unambiguous."""
        cmp = DateComparator()
        # Day=24 > 12 → unambiguous in both halves.
        assert cmp.compare("10/24/16", "10/24/16") == 1.0
        assert cmp.compare("24/10/2016", "10/24/16") == 1.0


class TestIssue117HeadlineIsoVsAmbiguous:
    """ISO/year-first layout must not be corrupted by a day-first pass.

    #117's headline case: canonical ISO ground truth vs a non-canonical
    numeric prediction. The year unambiguously leads in `2025-02-01`, so
    month-then-day order is fixed; a day-first interpretation that reads
    it as Jan 2 is wrong and must not be applied.
    """

    def test_iso_gt_vs_ambiguous_numeric_default(self):
        cmp = DateComparator()
        # 2025-02-01 is Feb 1; 01/02/2025 read month-first is also Feb 1.
        assert cmp.compare("2025-02-01", "01/02/2025") == 1.0

    def test_iso_gt_vs_ambiguous_numeric_dayfirst_true(self):
        # Even forcing EU, the ISO side stays Feb 1, and 01/02/2025
        # day-first is also Feb 1 → match.
        cmp = DateComparator(dayfirst=True)
        assert cmp.compare("2025-02-01", "01/02/2025") == 1.0

    def test_iso_gt_vs_ambiguous_numeric_dayfirst_false(self):
        # US: 01/02/2025 is Jan 2, ISO side Feb 1 → genuinely differ.
        cmp = DateComparator(dayfirst=False)
        assert cmp.compare("2025-02-01", "01/02/2025") == 0.0

    def test_slash_year_first_also_pinned(self):
        cmp = DateComparator()
        assert cmp.compare("2025/02/01", "01/02/2025") == 1.0

    def test_year_first_self_consistent_distinct_dates_still_zero(self):
        cmp = DateComparator()
        assert cmp.compare("2025-02-01", "2025-02-02") == 0.0

    def test_eu_disambiguation_still_works(self):
        """Non-year-first ambiguous layouts still resolve via dayfirst."""
        cmp = DateComparator()
        # 24/10/2016 (day=24) is unambiguous EU; matches 10/24/16.
        assert cmp.compare("24/10/2016", "10/24/16") == 1.0

    def test_invalid_dayfirst(self):
        with pytest.raises(ValueError, match="dayfirst"):
            DateComparator(dayfirst="auto")  # type: ignore[arg-type]


class TestTier5NonDateTokens:
    """Time-only and other non-date strings must not score as dates."""

    @pytest.mark.parametrize("garbage", ["10/45AM", "9/5AM", "12:30 PM"])
    def test_time_only_against_real_date(self, garbage):
        cmp = DateComparator()
        assert cmp.compare(garbage, "10/24/2016") == 0.0
        assert cmp.compare("10/24/2016", garbage) == 0.0


class TestTier5Unparseable:
    @pytest.mark.parametrize(
        "garbage",
        [
            "not a date",
            "xyz",
            "abc123",
            "",
            "   ",
            "\t\n",
            "!@#$%",
            "the quick brown fox",
            "yesterday",
            "last tuesday",
            "13 months ago",
            "07/17/ 6",  # corrupted GT (embedded whitespace)
            "11/0316",  # corrupted GT (missing separator)
        ],
    )
    def test_garbage_against_valid_date(self, garbage):
        cmp = DateComparator()
        assert cmp.compare(garbage, "2025-01-01") == 0.0
        assert cmp.compare("2025-01-01", garbage) == 0.0

    def test_both_unparseable(self):
        cmp = DateComparator()
        assert cmp.compare("garbage", "nonsense") == 0.0
        assert cmp.compare("", "") == 0.0


# ---------------------------------------------------------------------------
# Tolerance — Tier 1 only
# ---------------------------------------------------------------------------


class TestTolerance:
    def test_one_day_tolerance(self):
        cmp = DateComparator(tolerance=timedelta(days=1))
        assert cmp.compare("2025-01-01", "2025-01-02") == 1.0
        assert cmp.compare("2025-01-01", "2024-12-31") == 1.0
        assert cmp.compare("2025-01-01", "2025-01-03") == 0.0

    def test_zero_tolerance_default(self):
        cmp = DateComparator()
        assert cmp.compare("2025-01-01", "2025-01-02") == 0.0

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            DateComparator(tolerance=timedelta(days=-1))

    def test_tolerance_accepts_int_days(self):
        cmp = DateComparator(tolerance=1)
        assert cmp.tolerance == timedelta(days=1)
        assert cmp.compare("2025-01-01", "2025-01-02") == 1.0
        assert cmp.compare("2025-01-01", "2025-01-03") == 0.0

    def test_tolerance_accepts_float_days(self):
        cmp = DateComparator(tolerance=1.5)
        assert cmp.tolerance == timedelta(days=1, hours=12)

    def test_subday_tolerance_honors_actual_times(self):
        """A sub-day tolerance compares real timestamps, not day-floors."""
        cmp = DateComparator(tolerance=0.5)  # 12 hours
        # 1 hour apart across midnight → within 12h → match.
        assert cmp.compare("2025-01-01 23:30", "2025-01-02 00:30") == 1.0
        # 17 hours apart, same calendar day → beyond 12h → miss.
        assert cmp.compare("2025-01-01 06:00", "2025-01-01 23:00") == 0.0

    def test_subday_tolerance_36_hours(self):
        """The documented `tolerance=1.5  # 36 hours` actually means 36h."""
        cmp = DateComparator(tolerance=1.5)
        assert cmp.compare("2025-01-01 00:00", "2025-01-02 12:00") == 1.0  # 36h
        assert cmp.compare("2025-01-01 00:00", "2025-01-02 13:00") == 0.0  # 37h

    def test_subday_tolerance_boundary_inclusive(self):
        cmp = DateComparator(tolerance=0.5)  # exactly 12h
        assert cmp.compare("2025-01-01 00:00", "2025-01-01 12:00") == 1.0
        assert cmp.compare("2025-01-01 00:00", "2025-01-01 12:01") == 0.0

    def test_whole_day_tolerance_keeps_day_semantics(self):
        """Whole-day tolerance still floors to calendar days (time ignored)."""
        cmp = DateComparator(tolerance=1)
        # 47 hours apart in real time, but only 1 calendar day apart →
        # day-floored → match. (An actual-time reading would be 0.0.)
        assert cmp.compare("2025-01-01 00:00", "2025-01-02 23:00") == 1.0
        # 2 calendar days apart → beyond a 1-day tolerance → miss.
        assert cmp.compare("2025-01-01", "2025-01-03") == 0.0

    def test_zero_tolerance_is_same_calendar_day(self):
        """Default zero tolerance: same calendar day regardless of time."""
        cmp = DateComparator()
        assert cmp.compare("2025-01-01 06:00", "2025-01-01 23:00") == 1.0
        assert cmp.compare("2025-01-01 23:00", "2025-01-02 01:00") == 0.0

    def test_tolerance_rejects_bool(self):
        # bool subclasses int but should not be silently accepted
        with pytest.raises(ValueError, match="bool"):
            DateComparator(tolerance=True)

    def test_tolerance_rejects_string(self):
        with pytest.raises(ValueError, match="timedelta"):
            DateComparator(tolerance="1 day")  # type: ignore[arg-type]

    def test_tolerance_does_not_apply_to_partial_year(self):
        """Tier 3 is m/d-strict; tolerance is irrelevant there."""
        cmp = DateComparator(allow_partial_year=True, tolerance=timedelta(days=2))
        # m/d differ; tolerance doesn't bridge the gap because Tier 3
        # doesn't use it.
        assert cmp.compare("Oct 24", "10/26/16") == 0.0


# ---------------------------------------------------------------------------
# Timezones
# ---------------------------------------------------------------------------


class TestTimezones:
    def test_iso_with_z_suffix_vs_naive(self):
        cmp = DateComparator()
        assert cmp.compare("2025-01-01", "2025-01-01T00:00:00Z") == 1.0

    def test_aware_and_naive_same_date(self):
        cmp = DateComparator()
        assert cmp.compare(
            "2025-01-01", datetime(2025, 1, 1, tzinfo=timezone.utc)
        ) == 1.0


class TestMixedTimezoneRangesDoNotCrash:
    """A mixed tz-aware/naive range endpoint must not raise.

    Comparisons in the range paths previously skipped timezone
    alignment, so one malformed field value (#141 review, blocking)
    raised TypeError and killed the whole evaluation run instead of
    degrading to a score.
    """

    def test_range_vs_single_mixed_tz_graded(self):
        cmp = DateComparator()
        # Must not raise; 2025-02-01 is inside Jan 1–Mar 1 → graded 0.5.
        result = cmp.compare(
            "2025-01-01T00:00:00Z to 2025-03-01", "2025-02-01"
        )
        assert result == pytest.approx(0.5)

    def test_range_vs_single_mixed_tz_contains(self):
        cmp = DateComparator(range_mode="contains")
        result = cmp.compare(
            "2025-01-01T00:00:00Z to 2025-03-01", "2025-02-01"
        )
        assert result == 1.0

    def test_range_vs_range_mixed_tz_jaccard(self):
        cmp = DateComparator()
        # Overlap Jan 15–Feb 15 inside Jan 1–Mar 1; must not raise.
        result = cmp.compare(
            "2025-01-01T00:00:00Z to 2025-03-01",
            "2025-01-15 to 2025-02-15",
        )
        assert 0.0 < result <= 1.0

    def test_range_parse_order_check_mixed_tz_no_crash(self):
        """The start<=end ordering check must tolerate mixed awareness."""
        cmp = DateComparator()
        # Aware start, naive end, correctly ordered → parses as a range.
        result = cmp.compare(
            "2025-01-01T00:00:00Z to 2025-03-01",
            "2025-01-01 to 2025-03-01",
        )
        assert result == 1.0

    def test_compare_never_raises_typeerror_is_zero(self):
        """Even an unforeseen tz edge degrades to 0.0, never raises."""
        cmp = DateComparator()
        # Reversed mixed-tz endpoints (aware after naive) — whatever the
        # parse outcome, compare() must return a float, not raise.
        result = cmp.compare(
            "2025-03-01 to 2025-01-01T00:00:00Z", "2025-02-01"
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Constructor / config defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Pin the spec defaults so future changes can't drift silently."""

    def test_defaults_match_spec(self):
        cmp = DateComparator()
        assert cmp.threshold == 1.0
        assert cmp.tolerance == timedelta(0)
        assert cmp.dayfirst is None
        assert cmp.allow_partial_year is False
        assert cmp.range_mode == "graded"

    def test_repr_contains_config(self):
        cmp = DateComparator(
            tolerance=timedelta(days=1),
            dayfirst=True,
            allow_partial_year=True,
            range_mode="strict",
        )
        r = repr(cmp)
        assert "DateComparator" in r
        assert "tolerance" in r
        assert "dayfirst" in r
        assert "allow_partial_year=True" in r
        assert "range_mode='strict'" in r

    @pytest.mark.parametrize("bad", ["", "GRADED", "permissive", "yes", 1, None])
    def test_invalid_range_mode(self, bad):
        with pytest.raises((ValueError, TypeError)):
            DateComparator(range_mode=bad)


# ---------------------------------------------------------------------------
# BaseComparator interface
# ---------------------------------------------------------------------------


class TestDateComparatorBinaryCompare:
    def test_match_returns_tp(self):
        cmp = DateComparator()
        assert cmp.binary_compare("2025-01-01", "Jan 1, 2025") == (1, 0)

    def test_mismatch_returns_fp(self):
        cmp = DateComparator()
        assert cmp.binary_compare("2025-01-01", "2025-01-02") == (0, 1)

    def test_unparseable_returns_fp(self):
        cmp = DateComparator()
        assert cmp.binary_compare("garbage", "2025-01-01") == (0, 1)


class TestCallable:
    def test_call_syntax(self):
        cmp = DateComparator()
        assert cmp("2025-01-01", "Jan 1, 2025") == 1.0


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered(self):
        from stickler.structured_object_evaluator.models.comparator_registry import (
            get_global_registry,
        )

        assert get_global_registry().is_registered("DateComparator")

    def test_create_from_registry_with_config(self):
        from stickler.structured_object_evaluator.models.comparator_registry import (
            create_comparator,
        )

        cmp = create_comparator(
            "DateComparator",
            {"allow_partial_year": True, "range_mode": "reject"},
        )
        assert isinstance(cmp, DateComparator)
        assert cmp.allow_partial_year is True
        assert cmp.range_mode == "reject"

    def test_create_from_registry_no_config(self):
        from stickler.structured_object_evaluator.models.comparator_registry import (
            create_comparator,
        )

        cmp = create_comparator("DateComparator")
        assert isinstance(cmp, DateComparator)

    def test_create_from_registry_with_int_tolerance(self):
        """JSON-schema configs can express tolerance as an integer (days)."""
        from stickler.structured_object_evaluator.models.comparator_registry import (
            create_comparator,
        )

        cmp = create_comparator(
            "DateComparator", {"tolerance": 2, "range_mode": "contains"}
        )
        assert cmp.tolerance == timedelta(days=2)
        assert cmp.range_mode == "contains"

    def test_config_roundtrip(self):
        """Exported config should reconstruct an equivalent instance."""
        from stickler.structured_object_evaluator.models.comparator_registry import (
            create_comparator,
        )

        original = DateComparator(
            tolerance=timedelta(days=3),
            dayfirst=True,
            allow_partial_year=True,
            range_mode="strict",
            precision_mode="gt_loose",
        )
        config = original.config
        rebuilt = create_comparator("DateComparator", config)
        assert rebuilt.tolerance == original.tolerance
        assert rebuilt.dayfirst == original.dayfirst
        assert rebuilt.allow_partial_year == original.allow_partial_year
        assert rebuilt.range_mode == original.range_mode
        assert rebuilt.precision_mode == original.precision_mode

    def test_config_omits_default_tolerance(self):
        """A zero tolerance shouldn't clutter the config dict."""
        cmp = DateComparator(allow_partial_year=True)
        assert "tolerance" not in cmp.config

    def test_default_config_is_falsy(self):
        """A fully-default instance emits no config block.

        The schema exporter keys off truthiness, and the sibling
        NumericComparator returns None at defaults; a default
        DateComparator must not write a redundant config block into every
        exported schema (#141 review).
        """
        assert not DateComparator().config

    def test_config_omits_default_keys(self):
        """Each non-default instance exports only the keys that changed."""
        assert DateComparator(allow_partial_year=True).config == {
            "allow_partial_year": True
        }
        assert DateComparator(range_mode="contains").config == {
            "range_mode": "contains"
        }
        assert DateComparator(precision_mode="gt_loose").config == {
            "precision_mode": "gt_loose"
        }
        assert DateComparator(dayfirst=True).config == {"dayfirst": True}
        assert DateComparator(tolerance=2).config == {"tolerance": 2}

    def test_config_combines_only_nondefaults(self):
        cmp = DateComparator(allow_partial_year=True, precision_mode="overlap")
        assert cmp.config == {
            "allow_partial_year": True,
            "precision_mode": "overlap",
        }

    def test_default_config_roundtrips(self):
        """A default instance round-trips even though its config is falsy."""
        from stickler.structured_object_evaluator.models.comparator_registry import (
            create_comparator,
        )

        rebuilt = create_comparator("DateComparator", DateComparator().config)
        assert isinstance(rebuilt, DateComparator)
        assert rebuilt.range_mode == "graded"
        assert rebuilt.precision_mode == "exact"

    def test_config_serializable_to_json(self):
        """The exported config must round-trip through JSON."""
        import json

        cmp = DateComparator(
            tolerance=2,
            dayfirst=False,
            allow_partial_year=True,
            range_mode="reject",
        )
        json.dumps(cmp.config)  # must not raise


# ---------------------------------------------------------------------------
# StructuredModel integration
# ---------------------------------------------------------------------------


class TestInStructuredModel:
    def test_field_level_match(self):
        from stickler.structured_object_evaluator.models.comparable_field import (
            ComparableField,
        )
        from stickler.structured_object_evaluator.models.structured_model import (
            StructuredModel,
        )

        class Event(StructuredModel):
            event_date: str = ComparableField(
                comparator=DateComparator(), threshold=1.0, weight=1.0
            )

        gt = Event(event_date="2025-01-01")
        pred = Event(event_date="Jan 1, 2025")
        result = gt.compare_with(pred)
        assert result["overall_score"] == 1.0
        assert result["all_fields_matched"] is True

    def test_partial_year_via_threshold(self):
        """Surface Tier 3 (0.7) through a ComparableField with threshold<=0.7."""
        from stickler.structured_object_evaluator.models.comparable_field import (
            ComparableField,
        )
        from stickler.structured_object_evaluator.models.structured_model import (
            StructuredModel,
        )

        class Event(StructuredModel):
            event_date: str = ComparableField(
                comparator=DateComparator(allow_partial_year=True),
                threshold=0.7,
                weight=1.0,
            )

        gt = Event(event_date="11/03")
        pred = Event(event_date="11/03/2012")
        result = gt.compare_with(pred)
        # Threshold 0.7 + clip means 0.7 should percolate up cleanly.
        assert result["overall_score"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Performance sanity
# ---------------------------------------------------------------------------


class TestPerformanceSanity:
    def test_large_batch_completes(self):
        cmp = DateComparator()
        for _ in range(1000):
            assert cmp.compare("2025-01-01", "Jan 1, 2025") == 1.0


class TestInputLengthCap:
    """Pathologically long input is rejected cheaply rather than parsed.

    A multi-kilobyte garbage string already scored 0.0, but dateutil
    chewed through the whole thing first (~ms-to-100s-of-ms per call).
    The length cap short-circuits before parsing (#141 review hardening).
    """

    def test_over_length_input_scores_zero(self):
        cmp = DateComparator()
        garbage = "a" * 100_000
        assert cmp.compare(garbage, "2025-01-01") == 0.0
        assert cmp.compare("2025-01-01", garbage) == 0.0

    def test_over_length_input_with_delimiter_not_split(self):
        cmp = DateComparator()
        # A huge string containing ' to ' must not be split into two
        # huge range halves and parsed — the cap rejects it first.
        garbage = ("x" * 50_000) + " to " + ("y" * 50_000)
        assert cmp.compare(garbage, "2025-01-01") == 0.0

    def test_realistic_long_date_unaffected(self):
        cmp = DateComparator()
        # The most verbose realistic form is well under the cap.
        assert (
            cmp.compare("Wednesday, January 1, 2025", "2025-01-01") == 1.0
        )

    def test_cap_completes_quickly(self):
        import time

        cmp = DateComparator()
        big = "z" * 200_000
        start = time.perf_counter()
        cmp.compare(big, "2025-01-01")
        elapsed = time.perf_counter() - start
        # Generous bound: capped rejection is sub-millisecond; an
        # uncapped dateutil parse of 200k chars is tens of ms or more.
        assert elapsed < 0.01
