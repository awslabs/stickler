"""Tests for DateComparator.

Organized by behavior category. Intentionally exhaustive on edge cases
since dates are notoriously full of traps.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from stickler.comparators import DateComparator


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestDateComparatorBasics:
    """Core parsing and equality behavior."""

    def setup_method(self):
        self.comparator = DateComparator()

    def test_identical_iso_strings(self):
        assert self.comparator.compare("2025-01-01", "2025-01-01") == 1.0

    def test_canonical_vs_non_canonical(self):
        """Central use case from the issue: different formats, same date."""
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

    def test_mixed_date_types(self):
        """date on one side, datetime on the other, at day granularity."""
        cmp = DateComparator()
        assert cmp.compare(date(2025, 1, 1), datetime(2025, 1, 1, 23, 59)) == 1.0


# ---------------------------------------------------------------------------
# Format variations (parametrized)
# ---------------------------------------------------------------------------


EQUIVALENT_JAN_1_2025 = [
    "2025-01-01",
    "2025-1-1",
    "2025/01/01",
    "2025.01.01",
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
    "2025-01-01T05:00:00-05:00",  # same UTC day
    "Wed Jan 1 2025",
    "Wednesday, January 1, 2025",
    "Jan 1st, 2025",
    "1st Jan 2025",
    "  2025-01-01  ",  # padded whitespace
    "\t2025-01-01\n",  # weird whitespace
    "2025-01-01T00:00:00.000",  # milliseconds
    "2025-01-01T00:00:00.123456",  # microseconds
    "20250101",  # compact ISO
]


class TestDateComparatorFormatZoo:
    """Parametrized: every equivalent rendering of Jan 1, 2025 must match ISO."""

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


# ---------------------------------------------------------------------------
# Unparseable / garbage input
# ---------------------------------------------------------------------------


class TestDateComparatorUnparseable:
    """Unparseable input → silent 0.0. See module docstring for rationale."""

    def setup_method(self):
        self.comparator = DateComparator()

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
            "yesterday",  # dateutil does not support relative dates
            "last tuesday",
            "13 months ago",
        ],
    )
    def test_garbage_against_valid_date(self, garbage):
        assert self.comparator.compare(garbage, "2025-01-01") == 0.0
        assert self.comparator.compare("2025-01-01", garbage) == 0.0

    def test_both_unparseable(self):
        # Even if both fail, it's still a failure — not silently 1.0.
        assert self.comparator.compare("garbage", "nonsense") == 0.0
        assert self.comparator.compare("", "") == 0.0

    def test_very_long_garbage(self):
        long_garbage = "x" * 10_000
        assert self.comparator.compare(long_garbage, "2025-01-01") == 0.0

    def test_garbage_with_date_substring(self):
        """dateutil is fuzzy; a number buried in garbage may still parse.

        We document current behavior: dateutil's default is strict enough
        that it rejects most of these, but we don't guarantee it.
        """
        # Pure garbage surrounding a number — dateutil should reject.
        assert self.comparator.compare("abc 2025 xyz", "2025-01-01") == 0.0


# ---------------------------------------------------------------------------
# Granularity
# ---------------------------------------------------------------------------


class TestDateComparatorGranularity:
    """Partial granularity comparison."""

    def test_year_granularity(self):
        cmp = DateComparator(granularity="year")
        assert cmp.compare("2025-01-01", "2025-12-31") == 1.0
        assert cmp.compare("2025-06-15", "2025-01-01") == 1.0
        assert cmp.compare("2025-12-31", "2026-01-01") == 0.0
        # Jan 1 of year N-1 vs Dec 31 of year N — different years, no match.
        assert cmp.compare("2024-12-31T23:59:59", "2025-01-01T00:00:00") == 0.0

    def test_month_granularity(self):
        cmp = DateComparator(granularity="month")
        assert cmp.compare("2025-01-01", "2025-01-31") == 1.0
        assert cmp.compare("2025-01-31", "2025-02-01") == 0.0
        # Year boundary at month granularity.
        assert cmp.compare("2024-12-31", "2025-01-01") == 0.0

    def test_day_granularity_default(self):
        cmp = DateComparator()
        assert cmp.compare("2025-01-01T08:00:00", "2025-01-01T20:00:00") == 1.0
        # Last microsecond of the day vs first microsecond of the next.
        assert cmp.compare("2025-01-01T23:59:59.999999", "2025-01-02T00:00:00") == 0.0

    def test_hour_granularity(self):
        cmp = DateComparator(granularity="hour")
        assert cmp.compare("2025-01-01T08:15:00", "2025-01-01T08:45:00") == 1.0
        assert cmp.compare("2025-01-01T08:59:59", "2025-01-01T09:00:00") == 0.0

    def test_minute_granularity(self):
        cmp = DateComparator(granularity="minute")
        assert cmp.compare("2025-01-01T08:15:10", "2025-01-01T08:15:50") == 1.0
        assert cmp.compare("2025-01-01T08:15:59", "2025-01-01T08:16:00") == 0.0

    def test_second_granularity(self):
        cmp = DateComparator(granularity="second")
        assert cmp.compare("2025-01-01T08:15:10", "2025-01-01T08:15:10") == 1.0
        # Microsecond differences ignored at second granularity.
        assert cmp.compare("2025-01-01T08:15:10.000001", "2025-01-01T08:15:10.999999") == 1.0
        assert cmp.compare("2025-01-01T08:15:10", "2025-01-01T08:15:11") == 0.0


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------


class TestDateComparatorTolerance:
    """Tolerance window for near-matches."""

    def test_one_day_tolerance(self):
        cmp = DateComparator(tolerance=timedelta(days=1))
        assert cmp.compare("2025-01-01", "2025-01-02") == 1.0
        assert cmp.compare("2025-01-01", "2024-12-31") == 1.0
        assert cmp.compare("2025-01-01", "2025-01-03") == 0.0

    def test_tolerance_inclusive(self):
        """Tolerance boundary is inclusive (<=)."""
        cmp = DateComparator(tolerance=timedelta(days=1))
        # Exactly 1 day apart → match.
        assert cmp.compare("2025-01-01", "2025-01-02") == 1.0

    def test_one_week_tolerance(self):
        cmp = DateComparator(tolerance=timedelta(days=7))
        assert cmp.compare("2025-01-01", "2025-01-08") == 1.0
        assert cmp.compare("2025-01-01", "2025-01-09") == 0.0

    def test_zero_tolerance_default(self):
        cmp = DateComparator()
        assert cmp.compare("2025-01-01", "2025-01-02") == 0.0

    def test_tolerance_spans_year_boundary(self):
        cmp = DateComparator(tolerance=timedelta(days=2))
        assert cmp.compare("2024-12-31", "2025-01-02") == 1.0
        assert cmp.compare("2024-12-30", "2025-01-02") == 0.0

    def test_tolerance_spans_leap_day(self):
        cmp = DateComparator(tolerance=timedelta(days=1))
        # Feb 28 -> Feb 29 (leap year).
        assert cmp.compare("2024-02-28", "2024-02-29") == 1.0
        # Feb 29 -> Mar 1 (leap year).
        assert cmp.compare("2024-02-29", "2024-03-01") == 1.0

    def test_sub_day_tolerance_at_day_granularity(self):
        """Tolerance smaller than granularity-truncation is effectively ignored.

        With `granularity='day'`, both operands are truncated to midnight
        first. A 12-hour tolerance doesn't help if truncation already
        snapped them to the same or adjacent midnight.
        """
        cmp = DateComparator(tolerance=timedelta(hours=12), granularity="day")
        # Same day after truncation → equal regardless.
        assert cmp.compare("2025-01-01T08:00", "2025-01-01T20:00") == 1.0
        # Adjacent days → exactly 1 day diff after truncation, tolerance is 12h, so 0.0.
        assert cmp.compare("2025-01-01T23:00", "2025-01-02T01:00") == 0.0

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            DateComparator(tolerance=timedelta(days=-1))

    def test_huge_tolerance(self):
        cmp = DateComparator(tolerance=timedelta(days=3650))  # ~10 years
        assert cmp.compare("2020-01-01", "2025-01-01") == 1.0


# ---------------------------------------------------------------------------
# date_order tiebreaker
# ---------------------------------------------------------------------------


class TestDateComparatorDateOrder:
    """date_order is a tiebreaker for genuinely ambiguous numeric dates."""

    def test_iso_default(self):
        cmp = DateComparator()
        assert cmp.compare("2025-01-02", "2025-01-02") == 1.0

    def test_us_format_month_first(self):
        """US format: 01/02/2025 is Jan 2."""
        cmp = DateComparator(date_order="us")
        assert cmp.compare("01/02/2025", "2025-01-02") == 1.0
        assert cmp.compare("01/02/2025", "2025-02-01") == 0.0

    def test_european_format_day_first(self):
        """European format: 01/02/2025 is Feb 1."""
        cmp = DateComparator(date_order="european")
        assert cmp.compare("2025-02-01", "01/02/2025") == 1.0
        assert cmp.compare("2025-01-02", "01/02/2025") == 0.0

    def test_named_month_unaffected_by_hint(self):
        """Named-month strings are unambiguous."""
        for order in ("iso", "us", "european"):
            cmp = DateComparator(date_order=order)
            assert cmp.compare("2025-01-02", "Jan 2, 2025") == 1.0

    def test_iso_unaffected_by_hint(self):
        """ISO-leading strings always parse year-first."""
        eu = DateComparator(date_order="european")
        assert eu.compare("2025-02-01", "Feb 1, 2025") == 1.0

    def test_mixed_format_predictions(self):
        """LLM predictions may mix formats within one run."""
        cmp = DateComparator(date_order="european")
        assert cmp.compare("2025-02-01", "2025-02-01") == 1.0
        assert cmp.compare("2025-02-01", "01/02/2025") == 1.0
        assert cmp.compare("2025-02-01", "1 Feb 2025") == 1.0

    def test_unambiguous_dmy_under_us_hint(self):
        """13/01/2025 has no valid month-13, so dateutil falls back.

        Documents dateutil's behavior: when a numeric date is impossible
        under the primary interpretation, dateutil tries the other way.
        """
        us = DateComparator(date_order="us")
        # 13/01/2025 → month 13 is invalid → dateutil parses as day-first.
        assert us.compare("13/01/2025", "2025-01-13") == 1.0

    def test_two_digit_year(self):
        """dateutil's heuristic: 2-digit years near current century.

        Documents behavior rather than enforcing a specific mapping.
        """
        cmp = DateComparator(date_order="us")
        # "1/1/25" — dateutil maps 25 to 2025.
        assert cmp.compare("1/1/25", "2025-01-01") == 1.0

    def test_invalid_date_order(self):
        with pytest.raises(ValueError, match="date_order"):
            DateComparator(date_order="YMD")

        with pytest.raises(ValueError, match="date_order"):
            DateComparator(date_order="martian")

        with pytest.raises(ValueError, match="date_order"):
            DateComparator(date_order="")


# ---------------------------------------------------------------------------
# Timezones
# ---------------------------------------------------------------------------


class TestDateComparatorTimezones:
    """Mixed tz-aware / tz-naive datetimes must compare without errors."""

    def test_iso_with_z_suffix_vs_naive(self):
        cmp = DateComparator()
        assert cmp.compare("2025-01-01", "2025-01-01T00:00:00Z") == 1.0

    def test_both_aware_different_zones_same_instant(self):
        cmp = DateComparator(granularity="second")
        # 12:00 UTC == 07:00 EST
        assert cmp.compare(
            "2025-01-01T12:00:00+00:00", "2025-01-01T07:00:00-05:00"
        ) == 1.0

    def test_both_aware_different_zones_different_day_at_day_granularity(self):
        """23:00 UTC on Jan 1 == 03:00 UTC+04 on Jan 2 — different days at day granularity."""
        cmp = DateComparator(granularity="day")
        # Same instant expressed in two zones, straddling midnight.
        assert cmp.compare(
            "2025-01-01T23:00:00+00:00", "2025-01-02T03:00:00+04:00"
        ) == 1.0  # Same UTC instant → same day after UTC normalization
        # but local time in +04:00 is "Jan 2" — so the user may be
        # surprised. This documents the chosen semantics: we normalize to
        # UTC before truncation. If you have local-time semantics in
        # mind, strip the tz upstream.

    def test_both_naive(self):
        cmp = DateComparator()
        assert cmp.compare("2025-01-01", "2025-01-01") == 1.0

    def test_aware_and_naive_same_date(self):
        cmp = DateComparator()
        assert cmp.compare(
            "2025-01-01", datetime(2025, 1, 1, tzinfo=timezone.utc)
        ) == 1.0

    def test_utc_offset_named_formats(self):
        cmp = DateComparator(granularity="second")
        assert cmp.compare("2025-01-01T12:00:00 UTC", "2025-01-01T12:00:00Z") == 1.0


# ---------------------------------------------------------------------------
# Boundary dates
# ---------------------------------------------------------------------------


class TestDateComparatorBoundaries:
    """Leap years, year transitions, far past/future."""

    def test_leap_day_valid(self):
        cmp = DateComparator()
        assert cmp.compare("2024-02-29", "Feb 29, 2024") == 1.0

    def test_leap_day_invalid_year(self):
        """Feb 29 in a non-leap year should fail to parse."""
        cmp = DateComparator()
        assert cmp.compare("2025-02-29", "2025-02-28") == 0.0  # unparseable LHS

    def test_year_end_transition_zero_tolerance(self):
        cmp = DateComparator()
        assert cmp.compare("2024-12-31", "2025-01-01") == 0.0

    def test_year_end_transition_with_tolerance(self):
        cmp = DateComparator(tolerance=timedelta(days=1))
        assert cmp.compare("2024-12-31", "2025-01-01") == 1.0

    def test_epoch_date(self):
        cmp = DateComparator()
        assert cmp.compare("1970-01-01", "Jan 1, 1970") == 1.0

    def test_very_old_date(self):
        cmp = DateComparator()
        assert cmp.compare("1900-01-01", "Jan 1, 1900") == 1.0

    def test_far_future_date(self):
        cmp = DateComparator()
        assert cmp.compare("9999-12-31", "Dec 31, 9999") == 1.0

    def test_year_only_string(self):
        """dateutil parses '2025' as Jan 1, 2025 (with today's month/day context)."""
        cmp = DateComparator(granularity="year")
        assert cmp.compare("2025", "2025-06-15") == 1.0

    def test_month_year_only(self):
        cmp = DateComparator(granularity="month")
        assert cmp.compare("Jan 2025", "2025-01-15") == 1.0


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestDateComparatorValidation:
    """Constructor input validation."""

    @pytest.mark.parametrize(
        "bad",
        ["decade", "century", "milliseconds", "ms", "DAY", "", None, 5],
    )
    def test_invalid_granularity(self, bad):
        with pytest.raises((ValueError, TypeError)):
            DateComparator(granularity=bad)

    @pytest.mark.parametrize(
        "good", ["year", "month", "day", "hour", "minute", "second"]
    )
    def test_valid_granularities(self, good):
        DateComparator(granularity=good)

    def test_default_construction(self):
        cmp = DateComparator()
        assert cmp.threshold == 1.0
        assert cmp.tolerance == timedelta(0)
        assert cmp.granularity == "day"
        assert cmp.date_order == "iso"

    def test_repr_contains_config(self):
        cmp = DateComparator(
            tolerance=timedelta(days=1), granularity="month", date_order="us"
        )
        r = repr(cmp)
        assert "DateComparator" in r
        assert "tolerance" in r
        assert "month" in r
        assert "us" in r


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


class TestDateComparatorCallable:
    def test_call_syntax(self):
        cmp = DateComparator()
        assert cmp("2025-01-01", "Jan 1, 2025") == 1.0


class TestDateComparatorThreshold:
    """Threshold is vestigial for v1 (binary output) but must not break."""

    def test_custom_threshold_does_not_change_binary_behavior(self):
        cmp = DateComparator(threshold=0.5)
        assert cmp.binary_compare("2025-01-01", "Jan 1, 2025") == (1, 0)
        assert cmp.binary_compare("2025-01-01", "2025-01-02") == (0, 1)

    def test_threshold_above_one_still_passes_match(self):
        """Sanity: a high threshold just means 'matches must be perfect'."""
        cmp = DateComparator(threshold=1.0)
        assert cmp.binary_compare("2025-01-01", "2025-01-01") == (1, 0)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestDateComparatorRegistry:
    def test_registered(self):
        from stickler.structured_object_evaluator.models.comparator_registry import (
            get_global_registry,
        )

        assert get_global_registry().is_registered("DateComparator")

    def test_create_from_registry_with_config(self):
        from stickler.structured_object_evaluator.models.comparator_registry import (
            create_comparator,
        )

        cmp = create_comparator("DateComparator", {"granularity": "month"})
        assert isinstance(cmp, DateComparator)
        assert cmp.granularity == "month"

    def test_create_from_registry_no_config(self):
        from stickler.structured_object_evaluator.models.comparator_registry import (
            create_comparator,
        )

        cmp = create_comparator("DateComparator")
        assert isinstance(cmp, DateComparator)
        assert cmp.granularity == "day"

    def test_create_from_registry_with_tolerance(self):
        from stickler.structured_object_evaluator.models.comparator_registry import (
            create_comparator,
        )

        cmp = create_comparator(
            "DateComparator", {"tolerance": timedelta(days=1), "date_order": "us"}
        )
        assert cmp.tolerance == timedelta(days=1)
        assert cmp.date_order == "us"


# ---------------------------------------------------------------------------
# StructuredModel integration
# ---------------------------------------------------------------------------


class TestDateComparatorInStructuredModel:
    """End-to-end: use DateComparator inside a StructuredModel."""

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

    def test_field_level_mismatch(self):
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
        pred = Event(event_date="2025-01-05")

        result = gt.compare_with(pred)
        assert result["overall_score"] == 0.0

    def test_list_of_events_hungarian_matching(self):
        """Date comparator plays nicely with list-level Hungarian matching."""
        from typing import List

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

        class Calendar(StructuredModel):
            events: List[Event] = ComparableField(weight=1.0)

        gt = Calendar(
            events=[
                Event(event_date="2025-01-01"),
                Event(event_date="2025-06-15"),
            ]
        )
        # Predictions in different order + different format.
        pred = Calendar(
            events=[
                Event(event_date="June 15, 2025"),
                Event(event_date="Jan 1, 2025"),
            ]
        )
        result = gt.compare_with(pred, include_confusion_matrix=True)
        assert result["overall_score"] == 1.0


# ---------------------------------------------------------------------------
# Ambiguity traps from the issue
# ---------------------------------------------------------------------------


class TestDateComparatorAmbiguityTraps:
    """Explicit coverage of the traps called out in the feature request."""

    def test_01_02_2025_us_is_jan_2(self):
        cmp = DateComparator(date_order="us")
        assert cmp.compare("01/02/2025", "Jan 2, 2025") == 1.0

    def test_01_02_2025_european_is_feb_1(self):
        cmp = DateComparator(date_order="european")
        assert cmp.compare("01/02/2025", "Feb 1, 2025") == 1.0

    def test_us_vs_european_disagree_on_ambiguous_date(self):
        """Same input, different hints → different parsed date."""
        us = DateComparator(date_order="us")
        eu = DateComparator(date_order="european")
        assert us.compare("01/02/2025", "Jan 2, 2025") == 1.0
        assert eu.compare("01/02/2025", "Jan 2, 2025") == 0.0


# ---------------------------------------------------------------------------
# Performance sanity (not a benchmark)
# ---------------------------------------------------------------------------


class TestDateComparatorPerformanceSanity:
    """Make sure basic throughput is reasonable and we don't hang."""

    def test_large_batch_completes(self):
        cmp = DateComparator()
        for _ in range(1000):
            assert cmp.compare("2025-01-01", "Jan 1, 2025") == 1.0
