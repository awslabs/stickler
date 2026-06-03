"""Tests for DateComparator."""

from datetime import timedelta

import pytest

from stickler.comparators.date import DateComparator


class TestDateComparatorBasic:
    """Basic matching and non-matching tests."""

    def setup_method(self):
        self.comparator = DateComparator()

    def test_exact_iso_match(self):
        assert self.comparator.compare("2025-01-01", "2025-01-01") == 1.0

    def test_canonical_vs_long_form(self):
        assert self.comparator.compare("2025-01-01", "January 1, 2025") == 1.0

    def test_canonical_vs_short_month(self):
        assert self.comparator.compare("2025-01-01", "Jan 1, 2025") == 1.0

    def test_iso_with_timezone_vs_plain_iso(self):
        # "2025-01-01T00:00:00Z" should equal "2025-01-01" after UTC normalization.
        assert self.comparator.compare("2025-01-01T00:00:00Z", "2025-01-01") == 1.0

    def test_different_dates(self):
        assert self.comparator.compare("2025-01-01", "2025-01-02") == 0.0

    def test_different_years(self):
        assert self.comparator.compare("2024-06-15", "2025-06-15") == 0.0

    def test_callable_interface(self):
        assert self.comparator("2025-03-10", "March 10, 2025") == 1.0


class TestDateComparatorNoneHandling:
    """None value handling."""

    def setup_method(self):
        self.comparator = DateComparator()

    def test_both_none(self):
        assert self.comparator.compare(None, None) == 1.0

    def test_first_none(self):
        assert self.comparator.compare(None, "2025-01-01") == 0.0

    def test_second_none(self):
        assert self.comparator.compare("2025-01-01", None) == 0.0


class TestDateComparatorParseFailure:
    """Unparseable inputs should score 0.0."""

    def setup_method(self):
        self.comparator = DateComparator()

    def test_unparseable_first(self):
        assert self.comparator.compare("not a date", "2025-01-01") == 0.0

    def test_unparseable_second(self):
        assert self.comparator.compare("2025-01-01", "xyz") == 0.0

    def test_both_unparseable(self):
        assert self.comparator.compare("foo", "bar") == 0.0


class TestDateComparatorTolerance:
    """Tolerance parameter tests."""

    def test_no_tolerance_adjacent_days_fail(self):
        comparator = DateComparator(tolerance=timedelta(0))
        assert comparator.compare("2025-01-01", "2025-01-02") == 0.0

    def test_one_day_tolerance_adjacent_days_pass(self):
        comparator = DateComparator(tolerance=timedelta(days=1))
        assert comparator.compare("2025-01-01", "2025-01-02") == 1.0

    def test_one_day_tolerance_two_day_gap_fails(self):
        comparator = DateComparator(tolerance=timedelta(days=1))
        assert comparator.compare("2025-01-01", "2025-01-03") == 0.0

    def test_exact_match_zero_tolerance(self):
        comparator = DateComparator(tolerance=timedelta(0))
        assert comparator.compare("2025-06-15", "2025-06-15") == 1.0


class TestDateComparatorGranularity:
    """Granularity parameter tests."""

    def test_year_granularity_same_year(self):
        comparator = DateComparator(granularity="year")
        assert comparator.compare("2025-01-01", "2025-12-31") == 1.0

    def test_year_granularity_different_years(self):
        comparator = DateComparator(granularity="year")
        assert comparator.compare("2024-12-31", "2025-01-01") == 0.0

    def test_month_granularity_same_month(self):
        comparator = DateComparator(granularity="month")
        assert comparator.compare("2025-06-01", "2025-06-30") == 1.0

    def test_month_granularity_different_months(self):
        comparator = DateComparator(granularity="month")
        assert comparator.compare("2025-06-30", "2025-07-01") == 0.0

    def test_day_granularity_same_day_different_time(self):
        comparator = DateComparator(granularity="day")
        assert comparator.compare("2025-01-01T08:00:00", "2025-01-01T20:00:00") == 1.0

    def test_second_granularity_same_second(self):
        comparator = DateComparator(granularity="second")
        assert (
            comparator.compare("2025-01-01T12:00:00.123", "2025-01-01T12:00:00.456")
            == 1.0
        )

    def test_second_granularity_different_seconds(self):
        comparator = DateComparator(granularity="second")
        assert (
            comparator.compare("2025-01-01T12:00:00", "2025-01-01T12:00:01") == 0.0
        )


class TestDateComparatorDateOrder:
    """date_order disambiguation tests."""

    def test_mdy_parses_month_first(self):
        # "01/02/2025" under MDY is January 2, 2025
        comparator = DateComparator(date_order="MDY")
        assert comparator.compare("01/02/2025", "January 2, 2025") == 1.0

    def test_dmy_parses_day_first(self):
        # "01/02/2025" under DMY is February 1, 2025
        comparator = DateComparator(date_order="DMY")
        assert comparator.compare("01/02/2025", "February 1, 2025") == 1.0

    def test_mdy_and_dmy_produce_different_results(self):
        mdy = DateComparator(date_order="MDY")
        dmy = DateComparator(date_order="DMY")
        assert mdy.compare("01/02/2025", "January 2, 2025") == 1.0
        assert dmy.compare("01/02/2025", "January 2, 2025") == 0.0


class TestDateComparatorBinaryCompare:
    """binary_compare method tests."""

    def setup_method(self):
        self.comparator = DateComparator()

    def test_match_returns_tp(self):
        assert self.comparator.binary_compare("2025-01-01", "Jan 1, 2025") == (1, 0)

    def test_no_match_returns_fp(self):
        assert self.comparator.binary_compare("2025-01-01", "2025-01-02") == (0, 1)

    def test_parse_failure_returns_fp(self):
        assert self.comparator.binary_compare("not a date", "2025-01-01") == (0, 1)


class TestDateComparatorValidation:
    """Invalid configuration should raise ValueError."""

    def test_invalid_granularity(self):
        with pytest.raises(ValueError, match="granularity"):
            DateComparator(granularity="hour")  # type: ignore[arg-type]

    def test_invalid_date_order(self):
        with pytest.raises(ValueError, match="date_order"):
            DateComparator(date_order="YDM")  # type: ignore[arg-type]

    def test_negative_tolerance(self):
        with pytest.raises(ValueError, match="non-negative"):
            DateComparator(tolerance=timedelta(days=-1))

    def test_non_timedelta_tolerance(self):
        with pytest.raises(ValueError, match="timedelta"):
            DateComparator(tolerance=1)  # type: ignore[arg-type]
