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
            ("206-555-0100", "(206) 555-0100"),
            ("206-555-0100", "206.555.0100"),
            ("206-555-0100", "206 555 0100"),
            ("206-555-0100", "2065550100"),
            ("+1-206-555-0100", "2065550100"),
            ("+1 (206) 555-0100", "206-555-0100"),
            ("  206-555-0100  ", "2065550100"),
        ],
    )
    def test_same_number_written_differently_matches(self, gt, pred):
        assert PhoneComparator().compare(gt, pred) == 1.0

    def test_extensions_are_reconciled(self):
        assert (
            PhoneComparator().compare("+1 (206) 555-0100 ext. 89", "+12065550100x89")
            == 1.0
        )

    @pytest.mark.parametrize(
        "gt, pred",
        [
            ("206-555-0100", "206-555-0101"),
            ("206-555-0100", "206-556-0100"),
            ("+1-206-555-0100", "+1-206-555-0101"),
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
            PhoneComparator(region="GB").compare("+12065550100", "+1 206 555 0100")
            == 1.0
        )


class TestUnparseableInput:
    """When neither side is a usable number, string equality decides.

    This inverts what 0.7.0rc asserted. Rejecting *every* unparseable pair also
    rejected values that were extracted perfectly -- a UK national number under
    region="US", an extension fragment, a 555-area-code documentation number --
    and reported two identical strings as maximally different, deflating
    precision and recall with no signal why. The fallback is equality, so a
    placeholder pair still never canonicalizes into a fake number match.

    Genuinely absent values never reach here -- BaseComparator resolves None
    first, and the comparison layer treats None/"" on both sides as a true
    negative.

    See https://github.com/awslabs/stickler/issues/258
    """

    @pytest.mark.parametrize("value", ["N/A", "n/a", "unknown", "not a phone", "-", ""])
    def test_identical_unparseable_values_match(self, value):
        assert PhoneComparator().compare(value, value) == 1.0

    @pytest.mark.parametrize(
        "value",
        [
            "555-123-4567",  # 555 area code: never assigned, so invalid
            "020 7183 8750",  # UK national format, read under region="US"
            "ext 4021",  # not a number in any region
        ],
    )
    def test_a_correctly_extracted_but_unusable_number_matches_itself(self, value):
        """The three shapes issue #258 reported, each scored 0.0 before."""
        assert PhoneComparator().compare(value, value) == 1.0

    def test_different_unparseable_values_do_not_match(self):
        """The fallback is equality, not leniency.

        Two values that both failed to extract are not a match just because
        neither is a phone number.
        """
        assert PhoneComparator().compare("N/A", "unknown") == 0.0
        assert PhoneComparator().compare("ext 4021", "ext 4022") == 0.0

    def test_one_side_unparseable_does_not_match(self):
        assert PhoneComparator().compare("206-555-0100", "N/A") == 0.0
        assert PhoneComparator().compare("N/A", "206-555-0100") == 0.0

    def test_one_side_unparseable_does_not_match_in_either_order(self):
        """Both sides are canonicalized before branching.

        The pre-#258 body returned early on the first side and never inspected
        the second, so the two argument orders took different code paths.
        """
        phone = PhoneComparator()

        for usable in ("206-555-0100", "+12065550100x89"):
            for unusable in ("N/A", "020 7183 8750", ""):
                assert phone.compare(usable, unusable) == 0.0
                assert phone.compare(unusable, usable) == 0.0

    def test_two_empty_strings_match(self):
        """Recorded explicitly rather than left to fall out of the fallback.

        `BaseComparator.compare` special-cases None but not "", so an empty
        string reaches `_compare`. It scores 1.0 here, and end-to-end metrics
        are unaffected either way: the comparison layer resolves
        empty-on-both-sides as a true negative before scoring. A carve-out
        would reintroduce the "two identical values score 0.0" branch this
        change exists to remove.
        """
        assert PhoneComparator().compare("", "") == 1.0

    def test_a_valid_number_is_unaffected(self):
        assert PhoneComparator().compare("206-555-0100", "2065550100") == 1.0


class TestNoStringComparatorCanDoThis:
    """The measurement that justifies a dedicated comparator."""

    def test_edit_distance_ranks_the_two_cases_backwards(self):
        from stickler.comparators.levenshtein import LevenshteinComparator

        same_number = LevenshteinComparator().compare(
            "206-555-0100", "(206) 555-0100"
        )
        different_number = LevenshteinComparator().compare(
            "206-555-0100", "206-555-0101"
        )

        # A different number scores HIGHER than the same number reformatted, so
        # no threshold separates them.
        assert different_number > same_number

        # PhoneComparator gets both right.
        phone = PhoneComparator()
        assert phone.compare("206-555-0100", "(206) 555-0100") == 1.0
        assert phone.compare("206-555-0100", "206-555-0101") == 0.0

    def test_numeric_and_exact_both_fail_the_formatting_case(self):
        from stickler.comparators.exact import ExactComparator
        from stickler.comparators.numeric import NumericComparator

        assert ExactComparator().compare("206-555-0100", "(206) 555-0100") == 0.0
        assert NumericComparator().compare("206-555-0100", "(206) 555-0100") == 0.0
        assert PhoneComparator().compare("206-555-0100", "(206) 555-0100") == 1.0


class TestNonePolicyAndSerialization:
    def test_none_policy_comes_from_the_base_class(self):
        phone = PhoneComparator()

        assert phone.compare(None, None) == 1.0
        assert phone.compare(None, "206-555-0100") == 0.0
        assert phone.compare("206-555-0100", None) == 0.0

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

class TestValidityNotJustParseability:
    """libphonenumber parses strings that are not real numbers.

    `parse` accepts "0000000000" and formats it as E164. Validity is still
    checked with `is_valid_number`, so such a value is never treated as a
    *number*: a placeholder pair matches itself as a string, via the #258
    fallback, and a placeholder against a real number still scores 0.0. Raised
    in review of #243.
    """

    @pytest.mark.parametrize(
        "sentinel",
        ["0000000000", "1234567", "1111111111", "5555555555", "999-999-9999"],
    )
    def test_numeric_placeholders_match_themselves_as_strings(self, sentinel):
        """1.0 by string equality, not by canonicalizing into a fake E164.

        The distinction is load-bearing rather than pedantic: it is why
        `is_valid_number` stays. A parse-only check would score a placeholder
        pair 1.0 as a canonical *phone match*, and would also match two
        *different* placeholders that happen to canonicalize alike.
        """
        assert PhoneComparator().compare(sentinel, sentinel) == 1.0

    def test_a_placeholder_against_a_real_number_still_fails(self):
        """Validity still decides which side counts as a number."""
        assert PhoneComparator().compare("0000000000", "206-555-0100") == 0.0
        assert PhoneComparator().compare("206-555-0100", "0000000000") == 0.0

    def test_two_different_placeholders_do_not_match(self):
        assert PhoneComparator().compare("0000000000", "1111111111") == 0.0

    def test_those_sentinels_really_do_parse(self):
        """Guard the guard: if they stopped parsing, the test above is vacuous."""
        import phonenumbers

        parsed = phonenumbers.parse("0000000000", "US")

        # Parses and formats cleanly...
        assert phonenumbers.format_number(
            parsed, phonenumbers.PhoneNumberFormat.E164
        ) == "+10000000000"
        # ...but is not a real number, which is what the guard checks.
        assert phonenumbers.is_valid_number(parsed) is False

    def test_555_in_the_area_code_position_is_not_a_dialable_number(self):
        """Why examples use 206-555-0100 rather than 555-123-4567.

        "555-123-4567" puts 555 where the area code goes, and 555 has never
        been assigned as an area code -- which is why documentation uses it.
        libphonenumber is right to reject it.

        A real area code with the 555 *exchange* is fictional by convention and
        structurally valid, so it is what fixtures should use.

        Both self-compares score 1.0, by different routes: the valid number
        matches canonically, the documentation number matches as a string
        (#258). What the invalidity still buys is the line below -- 555-123-4567
        never canonicalizes, so it cannot collide with a real number.
        """
        phone = PhoneComparator()

        assert phone.compare("555-123-4567", "555-123-4567") == 1.0
        assert phone.compare("206-555-0100", "206-555-0100") == 1.0
        assert phone.compare("555-123-4567", "206-555-0100") == 0.0


class TestExtensionsAreSignificant:
    """E164 omits extensions, so comparing E164 alone loses them.

    Two extensions behind one switchboard reach different people, so they are
    compared separately. Raised in review of #243.
    """

    def test_different_extensions_do_not_match(self):
        assert (
            PhoneComparator().compare("+12065550100x89", "+12065550100x90") == 0.0
        )

    def test_same_extension_matches(self):
        assert (
            PhoneComparator().compare("+12065550100x89", "+12065550100x89") == 1.0
        )

    def test_extension_versus_no_extension_does_not_match(self):
        assert PhoneComparator().compare("+12065550100x89", "+12065550100") == 0.0

    def test_extension_formatting_still_normalizes(self):
        """The extension is compared, but how it is written is not."""
        assert (
            PhoneComparator().compare(
                "+1 (206) 555-0100 ext. 89", "+12065550100x89"
            )
            == 1.0
        )

    def test_e164_alone_would_have_missed_this(self):
        """Guard the guard: E164 really does drop the extension."""
        import phonenumbers

        with_ext = phonenumbers.parse("+12065550100x89", "US")

        assert phonenumbers.format_number(
            with_ext, phonenumbers.PhoneNumberFormat.E164
        ) == "+12065550100"
        assert with_ext.extension == "89"


class TestRegionIsValidated:
    """A plausible typo must fail loudly, not silently zero everything.

    `region="UK"` (the ISO code is "GB") made every national-format number fail
    to parse and score 0.0, which reads as total extraction failure. E164 inputs
    kept working, hiding it further. Raised in review of #243.
    """

    @pytest.mark.parametrize("bad", ["UK", "EN", "usa", "", "ZZ"])
    def test_an_unknown_region_raises(self, bad):
        with pytest.raises(ValueError, match="region"):
            PhoneComparator(region=bad)

    def test_the_message_names_the_gb_confusion(self):
        with pytest.raises(ValueError) as excinfo:
            PhoneComparator(region="UK")

        assert "GB" in str(excinfo.value)

    @pytest.mark.parametrize("good", ["US", "GB", "NL", "BR", "JP"])
    def test_known_regions_are_accepted(self, good):
        assert PhoneComparator(region=good).region == good
