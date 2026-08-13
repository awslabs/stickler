"""PhoneComparator compares dialled numbers, not strings.

An extraction pipeline emits the same number many ways, so the comparator parses
both sides to E164 before comparing. These tests pin that, and pin the reason it
cannot be done with a string comparator.

See https://github.com/awslabs/stickler/issues/242
"""

import pytest

from stickler.comparators.phone import PhoneComparator


class TestFormattingIsNotMeaning:
    @pytest.mark.parametrize(
        "gt, pred",
        [
            ("555-123-4567", "(555) 123-4567"),
            ("555-123-4567", "555.123.4567"),
            ("555-123-4567", "555 123 4567"),
            ("555-123-4567", "5551234567"),
            ("+1-555-123-4567", "5551234567"),
            ("+1 (555) 123-4567", "555-123-4567"),
            ("  555-123-4567  ", "5551234567"),
        ],
    )
    def test_same_number_written_differently_matches(self, gt, pred):
        assert PhoneComparator().compare(gt, pred) == 1.0

    def test_extensions_are_reconciled(self):
        assert (
            PhoneComparator().compare("+1 (555) 123-4567 ext. 89", "+15551234567x89")
            == 1.0
        )

    @pytest.mark.parametrize(
        "gt, pred",
        [
            ("555-123-4567", "555-123-4568"),
            ("555-123-4567", "555-124-4567"),
            ("+1-555-123-4567", "+1-555-123-4568"),
        ],
    )
    def test_a_different_number_does_not_match(self, gt, pred):
        assert PhoneComparator().compare(gt, pred) == 0.0


class TestRegion:
    def test_national_format_needs_the_right_region(self):
        """A number with no international prefix is region-dependent."""
        uk = PhoneComparator(region="GB")

        assert uk.compare("+44 20 7183 8750", "02071838750") == 1.0

    def test_the_wrong_region_does_not_silently_match(self):
        """Parsing a GB national number as US must not produce a false match."""
        us = PhoneComparator()

        assert us.compare("+44 20 7183 8750", "02071838750") == 0.0

    def test_e164_is_region_independent(self):
        """Both sides in E164 carry their own country code."""
        assert (
            PhoneComparator(region="GB").compare("+15551234567", "+1 555 123 4567")
            == 1.0
        )


class TestUnparseableInput:
    """Sentinel strings are not phone numbers that matched.

    Genuinely absent values never reach here -- BaseComparator resolves None
    first, and the comparison layer treats None/"" on both sides as a true
    negative. What is left is extraction noise, and reporting it as a true
    positive would inflate the metric.
    """

    @pytest.mark.parametrize("value", ["N/A", "n/a", "unknown", "not a phone", "-", ""])
    def test_identical_unparseable_values_do_not_match(self, value):
        assert PhoneComparator().compare(value, value) == 0.0

    def test_one_side_unparseable_does_not_match(self):
        assert PhoneComparator().compare("555-123-4567", "N/A") == 0.0
        assert PhoneComparator().compare("N/A", "555-123-4567") == 0.0

    def test_a_valid_number_is_unaffected(self):
        assert PhoneComparator().compare("555-123-4567", "5551234567") == 1.0


class TestNoStringComparatorCanDoThis:
    """The measurement that justifies a dedicated comparator."""

    def test_edit_distance_ranks_the_two_cases_backwards(self):
        from stickler.comparators.levenshtein import LevenshteinComparator

        same_number = LevenshteinComparator().compare(
            "555-123-4567", "(555) 123-4567"
        )
        different_number = LevenshteinComparator().compare(
            "555-123-4567", "555-123-4568"
        )

        # A different number scores HIGHER than the same number reformatted, so
        # no threshold separates them.
        assert different_number > same_number

        # PhoneComparator gets both right.
        phone = PhoneComparator()
        assert phone.compare("555-123-4567", "(555) 123-4567") == 1.0
        assert phone.compare("555-123-4567", "555-123-4568") == 0.0

    def test_numeric_and_exact_both_fail_the_formatting_case(self):
        from stickler.comparators.exact import ExactComparator
        from stickler.comparators.numeric import NumericComparator

        assert ExactComparator().compare("555-123-4567", "(555) 123-4567") == 0.0
        assert NumericComparator().compare("555-123-4567", "(555) 123-4567") == 0.0
        assert PhoneComparator().compare("555-123-4567", "(555) 123-4567") == 1.0


class TestNonePolicyAndSerialization:
    def test_none_policy_comes_from_the_base_class(self):
        phone = PhoneComparator()

        assert phone.compare(None, None) == 1.0
        assert phone.compare(None, "555-123-4567") == 0.0
        assert phone.compare("555-123-4567", None) == 0.0

    def test_default_config_serializes_to_nothing(self):
        assert PhoneComparator().config is None

    def test_non_default_region_is_serialized(self):
        assert PhoneComparator(region="GB").config == {"region": "GB"}

    def test_round_trips_through_the_registry(self):
        from stickler.structured_object_evaluator.models.comparator_registry import (
            ComparatorRegistry,
        )

        rebuilt = ComparatorRegistry().create_instance("PhoneComparator", {"region": "GB"})

        assert isinstance(rebuilt, PhoneComparator)
        assert rebuilt.region == "GB"
        assert rebuilt.compare("+44 20 7183 8750", "02071838750") == 1.0

    def test_repr_names_a_non_default_region(self):
        assert "GB" in repr(PhoneComparator(region="GB"))
        assert "region" not in repr(PhoneComparator())
