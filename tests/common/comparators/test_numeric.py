"""Tests for NumericComparator."""

import random
import re
import sys
import time
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators import NumericComparator


class TestNumericComparator:
    """Test the NumericComparator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Default comparator with exact matching
        self.comparator = NumericComparator()

        # Comparator with 10% relative tolerance
        self.relative_comparator = NumericComparator(relative_tolerance=0.1)

        # Comparator with 5 absolute tolerance
        self.absolute_comparator = NumericComparator(absolute_tolerance=5)

        # Comparator with both tolerances
        self.combined_comparator = NumericComparator(
            relative_tolerance=0.1, absolute_tolerance=5
        )

    def test_exact_match(self):
        """Test that exact matches return 1.0."""
        assert self.comparator.compare("123", "123") == 1.0
        assert self.comparator.compare(456, 456) == 1.0
        assert self.comparator.compare("789.0", 789) == 1.0
        assert self.comparator.compare("0", "0") == 1.0

    def test_numeric_formatting(self):
        """Test that different numeric formats are handled correctly."""
        assert self.comparator.compare("123", "123.0") == 1.0
        assert self.comparator.compare("$123", "123") == 1.0
        assert self.comparator.compare("123,456", "123456") == 1.0
        assert self.comparator.compare("-123", "-123") == 1.0

    def test_non_matching(self):
        """Test that non-matching numbers return 0.0."""
        assert self.comparator.compare("123", "456") == 0.0
        assert self.comparator.compare(789, 987) == 0.0
        assert self.comparator.compare("1.23", "1.24") == 0.0

    def test_non_numeric_values(self):
        """Test that non-numeric values are handled correctly."""
        assert self.comparator.compare("abc", "123") == 0.0
        assert self.comparator.compare("123", "abc") == 0.0
        assert self.comparator.compare("abc", "def") == 0.0

    def test_none_values(self):
        """Test that None values are handled correctly."""
        assert self.comparator.compare(None, None) == 1.0
        assert self.comparator.compare("123", None) == 0.0
        assert self.comparator.compare(None, "456") == 0.0

    def test_relative_tolerance(self):
        """Test that relative tolerance works correctly."""
        # Within 10% tolerance
        assert self.relative_comparator.compare(100, 109) == 1.0
        assert self.relative_comparator.compare(100, 91) == 1.0

        # Outside 10% tolerance - 11% difference
        assert self.relative_comparator.compare(100, 111) == 0.0
        assert self.relative_comparator.compare(100, 89) == 0.0

        # Edge cases
        assert self.relative_comparator.compare(0, 0) == 1.0
        assert self.relative_comparator.compare(0, 0.05) == 1.0  # Special case for zero
        assert self.relative_comparator.compare(1000000, 1090000) == 1.0

    def test_absolute_tolerance(self):
        """Test that absolute tolerance works correctly."""
        # Within tolerance of 5
        assert self.absolute_comparator.compare(100, 104) == 1.0
        assert self.absolute_comparator.compare(100, 96) == 1.0

        # Outside tolerance of 5
        assert self.absolute_comparator.compare(100, 106) == 0.0
        assert self.absolute_comparator.compare(100, 94) == 0.0

        # Edge cases
        assert self.absolute_comparator.compare(0, 4) == 1.0
        assert self.absolute_comparator.compare(0, -4) == 1.0

    def test_combined_tolerance(self):
        """Test that combined tolerances work correctly."""
        # Passes absolute tolerance (within 5)
        assert self.combined_comparator.compare(10, 14) == 1.0

        # Passes relative tolerance (within 10%)
        assert self.combined_comparator.compare(100, 109) == 1.0

        # Fails both tolerances
        assert self.combined_comparator.compare(100, 116) == 0.0

    def test_binary_compare_match(self):
        """Test that binary_compare returns (1, 0) for match."""
        assert self.comparator.binary_compare("123", "123") == (1, 0)
        assert self.relative_comparator.binary_compare(100, 109) == (1, 0)

    def test_binary_compare_no_match(self):
        """Test that binary_compare returns (0, 1) for no match."""
        assert self.comparator.binary_compare("123", "456") == (0, 1)
        # Use 112 which is 12% different, clearly outside the 10% tolerance
        assert self.relative_comparator.binary_compare(100, 112) == (0, 1)

    def test_custom_threshold(self):
        """Test that the threshold is applied correctly."""
        # This shouldn't affect the behavior for NumericComparator
        # since it only returns 0.0 or 1.0
        comparator = NumericComparator(threshold=0.5)
        assert comparator.binary_compare("123", "123") == (1, 0)
        assert comparator.binary_compare("123", "456") == (0, 1)


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("other", [0, 1, "0", "1", True, False])
def test_boolean_is_not_numeric(value, other):
    """Booleans must not become numeric matches in either operand position."""
    comparator = NumericComparator(absolute_tolerance=5)
    assert comparator.compare(value, other) == 0.0
    assert comparator.compare(other, value) == 0.0


@pytest.mark.parametrize("value, number", [(True, 1), (False, 0)])
def test_boolean_in_any_field(value, number):
    """A boolean in an uncoerced model field must not abort evaluation."""

    class Model(StructuredModel):
        flag: Any = ComparableField(comparator=NumericComparator())

    boolean_model = Model(flag=value)
    numeric_model = Model(flag=number)
    assert boolean_model.compare_with(numeric_model)["overall_score"] == 0.0
    assert numeric_model.compare_with(boolean_model)["overall_score"] == 0.0


def _reading_before_559(value):
    """``NumericComparator._extract_number`` as it was before #559: the
    reference for every value the fix must leave unchanged."""
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    value = str(value)
    is_negative = value.startswith("(") and value.endswith(")")
    if is_negative:
        value = value[1:-1]
    value = re.sub(r"[^0-9.-]", "", value)
    if not value:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return -number if is_negative else number


class TestNumericComparatorNonAsciiDigits:
    """Decimal digits outside ASCII (#559). A value whose digits are all outside
    ASCII is read when it holds one unambiguous number, and otherwise reads as no
    number, which is how it read before. A value with ASCII digits reads as
    before, except that a number switching script reads as no number and the
    minus signs and parentheses of other scripts mark a negative."""

    def setup_method(self):
        self.comparator = NumericComparator()

    def reads_as(self, value):
        return self.comparator._extract_number(value)

    def test_arabic_indic_digits_match_ascii(self):
        """The #559 report: the same number in two scripts matches."""
        assert self.comparator.compare("1250", "١٢٥٠") == 1.0
        assert self.comparator.compare("١٢٥٠", "1250") == 1.0
        assert self.comparator.compare(1250, "١٢٥٠") == 1.0

    @pytest.mark.parametrize(
        "digits",
        [
            "۱۲۵۰",  # Persian
            "१२५०",  # Devanagari
            "১২৫০",  # Bengali
            "௧௨௫௦",  # Tamil
            "๑๒๕๐",  # Thai
            "၁၂၅၀",  # Myanmar
            "１２５０",  # fullwidth
        ],
    )
    def test_other_scripts_match_ascii(self, digits):
        assert self.comparator.compare("1250", digits) == 1.0

    def test_every_decimal_digit_script(self):
        """Every script of decimal digits known to this Python reads 1250."""
        zeros = [
            code
            for code in range(sys.maxunicode + 1)
            if unicodedata.decimal(chr(code), None) == 0
        ]
        assert len(zeros) > 50
        for zero in zeros:
            digits = "".join(chr(zero + d) for d in (1, 2, 5, 0))
            assert self.reads_as(digits) == Decimal("1250"), hex(zero)

    def test_arabic_separators(self):
        """U+066C groups thousands and U+066B is the decimal point."""
        assert self.comparator.compare("1,250.00", "١٬٢٥٠٫٠٠") == 1.0
        assert self.comparator.compare("AED 1,250.50", "١٬٢٥٠٫٥٠ درهم") == 1.0
        assert self.comparator.compare("1250.5", "١٢٥٠٫٥") == 1.0

    def test_cldr_arabic_currency_formats(self):
        """The dots of a currency symbol such as د.إ. are not decimal points."""
        sar = "\u200f١٬٢٥٠٫٥٠\u00a0ر.س.\u200f"  # CLDR ar_SA
        aed = "(\u061c١٬٢٥٠٫٥٠\u00a0د.إ.\u200f)"  # CLDR ar_AE accounting
        assert self.reads_as(sar) == Decimal("1250.50")
        assert self.reads_as(aed) == Decimal("-1250.50")
        assert self.reads_as("ر.س ١٢٥") == Decimal("125")
        assert self.reads_as("د.إ.١٢٥٠") == Decimal("1250")

    @pytest.mark.parametrize(
        "value, reading",
        [
            # CLDR currency symbols with no currency sign character.
            ("-\u200eریال\u00a0۱٬۲۵۰", "-1250"),  # CLDR fa
            ("\u200e(ریال\u00a0۱٬۲۵۰)", "-1250"),
            # One-letter symbols, such as the kyat's "K", are left out: they are
            # as likely to be a label.
            ("-K\u00a0၁,၂၅၀", None),  # CLDR my
            ("(P ١٢)", None),
            ("- R ١٢٥٠", None),
            ("（P 12）", "12"),  # ASCII digits, as before
            ("-नेरू\u00a0१,२५०.५०", "-1250.50"),  # CLDR ne
            ("(Rs.௧,௨௫௦.௫௦)", "-1250.50"),  # CLDR ta
            ("(ر.س\u00a0١٢٥٠)", "-1250"),  # typed without the final dot
            ("-ر.س\u00a0١٢٥٠", "-1250"),
            ("(SAR\u00a0١٢٥٠)", "-1250"),
            ("($ ١٢٥٠)", "-1250"),  # a currency sign, after any space
            # A symbol of ASCII letters only may be a unit: "KM" is the
            # convertible mark and kilometres, "Fdj" is left out with it.
            ("(\u061c١٬٢٥٠Fdj)", None),
            ("(١٢ KM)", None),
            ("(١٢KM)", None),
            ("（KM 125）", "125"),  # ASCII digits, as before
            # A currency word after a typed space may be a label.
            ("(SAR ١٢٥٠)", None),
            ("(ALL ١٢٥٠)", None),
            ("-AED ١٬٢٥٠", None),
            ("(١٢٥٠ SAR)", None),
            ("(١٢٥٠ ر.س.)", None),
            ("(١٢٥٠\u00a0SAR)", "-1250"),
            # Only CLDR's own spelling counts: a dot is added for an abbreviation
            # in another script ("ر.س"), never to make an ASCII word a symbol.
            ("(Gs\u00a0١٢)", None),
            ("-Nu\u00a0١٢٥٠", None),
            ("(ރ\u00a0١٢)", None),  # one letter is not the abbreviation "ރ."
            ("- KM ١٢", None),
            # Words that are not currencies: a label, a note, a unit, an
            # abbreviation, a year, joined by a space or a no-break space.
            ("（税込\u00a0１，２５０）", None),
            ("(Total\u00a0١٢٥٠)", None),
            ("- Total:\u00a0١٢٥٠", None),
            ("(١٢٥٠\u00a0kg)", None),
            ("(ص.ب ١٢٥٠)", None),  # P.O. Box
            ("- ص.ب. ١٢٣٤", None),
            ("(۱۴۰۲ ه.ش.)", None),  # a Solar Hijri year
            ("(e.g. ١٢٥٠)", None),
            ("(XXX ١٢٥٠)", None),  # an ISO code that is no currency
            ("(SAR\u00a0١٢٥٠\u00a0SAR)", None),  # two currencies
            ("(US$-١٢٥٠)", None),  # a minus inside accounting parentheses
            ("(US$(١٢٥٠)", None),
            ("（합계\u00a01,250）", "1250"),  # ASCII digits, as before
        ],
    )
    def test_currency_words(self, value, reading):
        expected = None if reading is None else Decimal(reading)
        assert self.reads_as(value) == expected

    def test_cfa_franc_symbol(self):
        """CLDR writes the CFA franc as "F\u202fCFA", one symbol."""
        assert self.reads_as("-F\u202fCFA\u00a0߁,߂߅߀") == Decimal("-1250")
        assert self.reads_as("(F\u202fCFA\u00a0߁,߂߅߁)") == Decimal("-1251")

    @pytest.mark.parametrize(
        "value, reading",
        [
            ("१२,३४,५६७.८९", "1234567.89"),  # Indian lakh grouping
            ("৳১২,৩৪,৫৬৭", "1234567"),
            ("١,٢٣٤,٥٦٧", "1234567"),
            ("١٢٣٤", "1234"),
        ],
    )
    def test_grouping(self, value, reading):
        assert self.reads_as(value) == Decimal(reading)

    def test_different_values_still_differ(self):
        assert self.comparator.compare("1250", "١٢٥١") == 0.0
        assert self.comparator.compare("1,250.00", "١٬٢٥٠٫٠١") == 0.0

    def test_tolerance_applies_to_non_ascii_digits(self):
        comparator = NumericComparator(relative_tolerance=0.1)
        assert comparator.compare("100", "١٠٩") == 1.0
        assert comparator.compare("100", "١١١") == 0.0

    @pytest.mark.parametrize(
        "value",
        [
            "(١٢٥٠)",
            " (١٢٥٠) ",
            "\u00a0(١٢٥٠)\u00a0",
            "（１２５０）",  # fullwidth, as in ICU's ja accounting format
            "（AED１，２５０）",
            "﹙١٢٥٠﹚",  # small parentheses
        ],
    )
    def test_accounting_negative(self, value):
        assert self.reads_as(value) == Decimal("-1250")

    @pytest.mark.parametrize(
        "value",
        [
            "-١٢٥٠",
            "−١٢٥٠",  # MINUS SIGN
            "–١٢٥٠",  # EN DASH
            "‒١٢٥٠",  # FIGURE DASH
            "‐١٢٥٠",  # HYPHEN
            "‑١٢٥٠",  # NON-BREAKING HYPHEN
            "﹣١٢٥٠",  # SMALL HYPHEN-MINUS
            "－１２５０",  # FULLWIDTH HYPHEN-MINUS
            "➖١٢٥٠",  # HEAVY MINUS SIGN
            "▲１，２５０",  # Japanese accounting negative marks, fullwidth digits
            "△１，２５０",
            "\u061c-١٢٥٠",  # CLDR ar: ARABIC LETTER MARK, hyphen-minus
            "-AED\u00a0١٬٢٥٠",  # CLDR: the minus before the currency
            "₹-१२५०",  # or after a currency symbol
        ],
    )
    def test_minus_signs(self, value):
        assert self.reads_as(value) == Decimal("-1250")
        assert self.comparator.compare("1250", value) == 0.0

    @pytest.mark.parametrize(
        "value, reading",
        [
            # A dash after a word is a label or an ID, not a sign.
            ("المجموع – ١٢٥٠", None),  # "Total – 1250"
            ("INV-١٢٥٠", None),
            ("المجموع: -١٢٥٠", None),
            # A dash before a word that is not a currency may be a list bullet.
            ("- المبلغ الإجمالي ١٢٥٠", None),  # "- total amount 1250"
            ("- Total: ١٢٥٠", None),
            ("- المجموع: ١٢٥٠", None),
            ("€ – 1250", "1250"),  # a dash between a symbol and the number
            # A hyphen inside an earlier word is not a sign.
            ("Sub-total: ৳১,২৫০", "1250"),
            # Parentheses around a note are not accounting parentheses.
            ("（内消費税 １１３円）", None),  # "(includes 113 yen tax)"
            ("（税込 １，２５０円）", None),
            ("（税込１，２５０）", None),  # "(tax included 1250)"
            ("（内税 １１３）", None),
            ("(VAT ١٨٧٫٥٠)", None),  # VAT is not an ISO 4217 code
            ("(Note ٣)", None),
            # A currency inside accounting parentheses, joined as CLDR joins it.
            ("(SAR\u00a0١٢٥٠)", "-1250"),
            ("(ر.س.\u00a0١٢٥٠)", "-1250"),
            # Dashes after the number: a Japanese tamper-proof dash, a label.
            ("￥１，２５０－", "1250"),
            ("١٢٥٠ ريال – مدفوع", "1250"),  # "1250 riyal – paid"
            ("١٢٥٠-", None),  # a trailing minus, as "1250-" reads
            ("۱٬۲۵۰−", None),  # MINUS SIGN after the number, as RTL text extracts
            ("١٬٢٥٠٫٠٠−", None),
            ("١٢٥٠–", None),
            ("▲١٢٥٠", None),  # ▲ outside Japanese text may mark a rise
            ("▼１，２５０", None),  # ▼ and ▽ mark a fall, or a negative
            ("▽١٢٥٠", None),
        ],
    )
    def test_dashes_and_parentheses_that_are_not_signs(self, value, reading):
        expected = None if reading is None else Decimal(reading)
        assert self.reads_as(value) == expected

    def test_icu_persian_formats(self):
        """ICU's fa formats carry an invisible LEFT-TO-RIGHT MARK (U+200E)."""
        assert self.reads_as("\u200e−۱٬۲۵۰٫۵") == Decimal("-1250.5")
        assert self.reads_as("\u200e($ ۱٬۲۵۰٫۵۰)") == Decimal("-1250.50")

    def test_fullwidth_and_small_forms(self):
        assert self.reads_as("１２５０．５０") == Decimal("1250.50")
        assert self.reads_as("１，２５０．５０") == Decimal("1250.50")
        assert self.reads_as("١٢٥٠﹒٥٠") == Decimal("1250.50")

    @pytest.mark.parametrize(
        "mark", ["\ufeff", "\u200f", "\u061c", "\u200d", "\u2066", "\u2069"]
    )
    def test_format_characters_are_ignored(self, mark):
        """Byte order marks, direction marks and isolates (category Cf) beside a
        separator or a sign."""
        assert self.reads_as(f"{mark}١٬{mark}٢٥٠{mark}") == Decimal("1250")
        assert self.reads_as(f"{mark}-{mark}١٢٥٠") == Decimal("-1250")

    @pytest.mark.parametrize(
        "value",
        [
            "\u06dd١\u06dd٢",  # ARABIC END OF AYAH before each number
            "\u0601١٤٤٥\u0603١٢",  # ARABIC SIGN SANAH, ARABIC SIGN SAFHA
            "๑๒\u200b๓๔",  # ZERO WIDTH SPACE
            "١٢\u200f٣٤",  # a direction mark between two digits
        ],
    )
    def test_format_characters_that_separate_numbers(self, value):
        assert self.reads_as(value) is None

    def test_soft_hyphen_is_ignored(self):
        """SOFT HYPHEN is a line-break hint, not a separator."""
        assert self.reads_as("١\u00ad٢٥٠") == Decimal("1250")
        assert self.reads_as("1\u00ad250") == Decimal("1250")

    @pytest.mark.parametrize(
        "value, reading",
        [
            ("٫٥", "0.5"),  # a leading decimal point
            (".٥", "0.5"),
            ("-.٥", "-0.5"),
            ("(.٥)", "-0.5"),
            ("السعر: .٥", "0.5"),
            ("Amount:.٥", "0.5"),
            ("Price=.٥٠", "0.50"),
            ("合計：．５", "0.5"),
            ("$.٥٠", "0.50"),
            ("₹.५०", "0.50"),
            ("ر.س٫٥", "0.5"),  # U+066B is a decimal point even after a letter
            ("المبلغ٫٥", "0.5"),
            ("रु.५००", "500"),  # the "." ends the abbreviation Rs.
            ("Ｎｏ．５", "5"),
            ("Total.......١٢٥٠", "1250"),  # dot leaders
            (".٥ ريال", "0.5"),
            ("कुल......१२५०", "1250"),
            ("合計．．．．１２５０", "1250"),
            ("رقم ١٢٥٠، شكرا", "1250"),  # punctuation after the number
            ("المبلغ ١٬٢٥٠, شكرا", "1250"),
            ("١٢٥٠.", "1250"),
            ("\U0001e951⹁\U0001e952\U0001e955\U0001e950", "1250"),  # Adlam
        ],
    )
    def test_separators_around_the_number(self, value, reading):
        assert self.reads_as(value) == Decimal(reading)

    @pytest.mark.parametrize(
        "value",
        [
            # A "." after a word and a space is a dot leader or a decimal point.
            "Total .١٢٥٠",
            "AED .٥٠",
            "المبلغ .٥٠",
            "合計 ．５",
        ],
    )
    def test_a_dot_after_a_word_and_a_space_is_ambiguous(self, value):
        assert self.reads_as(value) is None

    def test_dot_leaders_do_not_make_a_fraction(self):
        assert self.comparator.compare("Total.......١٠٠٠", "Total.......١٠٠") == 0.0

    def test_devanagari_zero_as_abbreviation_mark(self):
        """ "रु०" (Rs.) writes the abbreviation mark as a Devanagari zero."""
        assert self.reads_as("रु०1250") == Decimal("1250")
        assert self.reads_as("रु०१२५०") == Decimal("1250")

    @pytest.mark.parametrize(
        "value, reading",
        [
            # A CJK numeral that does not touch the number is a word.
            ("一式 １，２５０円", "1250"),  # "one set"
            ("千葉県 １２５０円", "1250"),
            ("参考価格 １，２５０円", "1250"),
            ("北京 １２５０元", "1250"),  # 京 has a numeric value from Unicode 15.1
            ("東京本社 ３，０００円", "3000"),
            ("١٢٠ م²", "120"),
            ("１２万円", None),  # 万 (ten thousand) touches the number
            ("١٢٥٠½", None),
            ("١٠²", None),
            ("①١٢٥٠", None),
            ("一１２５０円", None),
            # 京 (capital) touching the number is a word on every Unicode version.
            ("東京１２５０円", "1250"),
        ],
    )
    def test_numerals_that_are_not_digits(self, value, reading):
        expected = None if reading is None else Decimal(reading)
        assert self.reads_as(value) == expected

    def test_arabic_and_persian_digits_may_mix(self):
        """Persian and Urdu text mixes Arabic-Indic and Persian digits."""
        assert self.reads_as("۱٢٥٠") == Decimal("1250")

    def test_compact_forms_read_the_number_shown(self):
        """As "1.3K" reads 1.3, the compact form "١٫٣ ألف" reads 1.3."""
        assert self.reads_as("١٫٣ ألف") == Decimal("1.3")
        assert self.reads_as("1.3K") == Decimal("1.3")

    @pytest.mark.parametrize(
        "value",
        [
            # "," that is not a thousands group, or a second decimal point.
            # U+060C is a decimal point in Urdu and a group separator in Sindhi.
            "٠،٠٥",
            "١,٢٥٠،٥",
            "۱،۲۵۰",
            "١٢٥٠،٥٠٠",
            "١٢٣٤٬٥٦٧",
            "١٬٢٥",
            "٠٬١٢٥",
            "١,٢٣٤,٥٦,٧٨٩",
            "١٢٣,٤٥,٦٧٨",
            "١٫٢٥٠٬٥",
            "١٫٢٥٠٫٥",
            # "." or "," with digits whose locales use "," as the decimal point
            # (Lao, Khmer, Mongolian, Javanese, Extended Arabic-Indic in az_Arab)
            # or that no locale writes natively (Sundanese, Balinese, Kawi, Cham).
            "១.២៥០,៥",
            "០,៥",
            "៧,០០$",
            "១.២៥០",
            "₭໑.໒໕໑",
            "꧐,꧕",
            "꧑.꧒꧕꧐",
            "᠓,᠑᠔᠒",
            "-᠐,᠑᠒᠕",
            "۱۲۵۰,۵",
            "۱,۲۵۰",
            "᮱.᮲᮵᮰",
            "᭑.᭒᭕᭐",
            "꩑.꩒꩕꩐",
            # Two numbers: a Persian slash decimal, dates, units, exponents.
            "۲/۵ درصد",
            "۱/۵",
            "۱۴۰۲/۰۵/۱۴",
            "٢٠٢٤-٠٥-١٤",
            "١٢٥٠ م٢",
            "١٫٢٥١أس٣",
            "۷×۱۰^۰",
            "فاتورة ٣ المبلغ ١٢٥٠",
        ],
    )
    def test_ambiguous_values_read_as_no_number(self, value):
        assert self.reads_as(value) is None
        assert self.comparator.compare(value, value) == 0.0

    @pytest.mark.parametrize("ascii_first", [True, False])
    @pytest.mark.parametrize("cut", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize(
        "native",
        [
            "٠١٢٣٤٥٦٧٨٩",
            "۰۱۲۳۴۵۶۷۸۹",
            "०१२३४५६७८९",
            "０１２３４５６７８９",
        ],
    )
    def test_a_number_switching_script_reads_as_no_number(
        self, ascii_first, cut, native
    ):
        """The #559 false match: one number whose digits switch script, directly
        or at any of its separators, reads as no number."""
        to_native = str.maketrans("0123456789", native)
        for number in ["1250.50", "1,250.50", "12,34,567.89"]:
            for point, comma in [
                (".", ","),
                ("٫", "٬"),
                ("．", "，"),
                ("﹒", "﹐"),
            ]:
                text = number.replace(",", "\0").replace(".", point)
                text = text.replace("\0", comma)
                if cut >= len(text):
                    continue
                head, tail = text[:cut], text[cut:]
                if ascii_first:
                    value = head + tail.translate(to_native)
                else:
                    value = head.translate(to_native) + tail
                if value.isascii() or not any(
                    c.isascii() and c.isdigit() for c in value
                ):
                    continue
                assert self.reads_as(value) is None, repr(value)

    @pytest.mark.parametrize(
        "value, reading",
        [
            ("1,250.00 (١٬٢٥٠٫٠٠)", "1250.00"),
            ("١٬٢٥٠٫٠٠ (1,250.00)", "1250.00"),
            ("1250 / ١٢٥٠", "1250"),
            ("-1,250.50 (\u200e−۱٬۲۵۰٫۵۰)", "-1250.50"),
            ("9,999.99 (٩٬٩٩٩٫٩٩)", "9999.99"),
            ("1,250.50\u200f١٬٢٥٠٫٥٠ درهم", "1250.50"),  # split only by an RLM
            # A second decimal point, or a comma not starting a thousands
            # group, does not join two numbers into one.
            ("1,250.00．١٬٢٥٠٫٠٠", "1250.00"),
            ("1250,١٢٥٠", "1250"),
            ("1,250.00．١٢٥١", "1250.00"),
            ("1,250.00١٬٢٥٠", "1250.00"),  # 1250.00 and 1250 are one amount
            ("1250,١٢٥١", "1250"),
            ("1250.١٢٥٠", "1250"),  # ASCII copy first: read as before
            ("SAR 125.00,١٢٥٫٠٠", "125.00"),
            # The same amount written twice with nothing between the copies.
            ("1,250.50١٬٢٥٠٫٥٠", "1250.50"),
            ("١٬٢٥٠٫٥٠1,250.50", "1250.50"),
            ("1250١٢٥٠", "1250"),
            ("1,250٫١٬٢٥٠", "1250"),
            ("فاتورة ٣ total 100", "100"),
            # Any ASCII digit keeps the old reading, a VAT rate included.
            ("١٬٢٥٠٫٠٠ درهم (شامل ضريبة 5%)", "5"),
        ],
    )
    def test_values_with_ascii_digits_read_as_before(self, value, reading):
        """Bilingual amounts keep the reading they had: digits in another
        script beside ASCII digits are ignored."""
        assert self.reads_as(value) == Decimal(reading)
        assert self.reads_as(value) == _reading_before_559(value)

    @pytest.mark.parametrize(
        "value, reading",
        [
            ("−1250", "-1250"),  # MINUS SIGN, which the old reading dropped
            ("\u200e−1,250.5", "-1250.5"),  # ICU fa with ASCII digits
            ("▲1,250", "1250"),  # ▲ marks a rise in Korean and Chinese tables
            ("（1250）", "-1250"),
            (" (1250) ", "-1250"),  # spaces around accounting parentheses
            ("\u200f(1,250.00)", "-1250.00"),  # or a format character
            ("\u00a0(1250)\u00a0", "-1250"),
            ("Total − 1250", "1250"),  # a label, as before
            ("INV−1250", "1250"),
            ("（内消費税 113円）", "113"),  # a note, as before
            ("（税込1,250）", "1250"),
            ("（内税113）", "113"),
            ("-SAR 1250", "-1250"),  # read negative once, not twice
            ("−AED 1250", "-1250"),  # a leading minus sign reads as "-" does
            ("− 1250", "-1250"),
            # A leading dash before a word is a bullet, as before.
            ("– Total: 1,250.00", "1250.00"),
            ("– Item 3", "3"),
            ("– Total due: 1,250.00", "1250.00"),
            # A minus sign right after a currency symbol is a sign.
            ("₹−1250", "-1250"),
            ("$−1250", "-1250"),
            ("€−1,250.50", "-1250.50"),
            ("‐ Total: 1250", "1250"),
            ("－ 合計 1,250円", "1250"),
            ("\u200e−AED 1,250.50", "-1250.50"),
            ("￥1,250－", "1250"),  # a tamper-proof dash, as before
        ],
    )
    def test_negatives_with_ascii_digits(self, value, reading):
        """The minus signs and parentheses read in other scripts mark a negative
        with ASCII digits too, so "−1250" no longer matches "١٢٥٠"."""
        assert self.reads_as(value) == Decimal(reading)

    def test_same_notation_reads_the_same_sign(self):
        assert self.comparator.compare("−1250", "−١٢٥٠") == 1.0
        assert self.comparator.compare("−1250", "١٢٥٠") == 0.0
        assert self.comparator.compare(" (1250) ", "١٢٥٠") == 0.0
        assert self.comparator.compare("\u200e−1,250.5", "۱٬۲۵۰٫۵") == 0.0

    def test_values_without_other_script_digits_read_as_before(self):
        """A value with no digit outside ASCII reads as before, except that a
        negative the old reading missed now reads negative."""
        values = [
            "￥1,250－",
            "（内消費税 113円）",
            "請求額は1,250.50円．",
            "1250．50",
            "Rs. 1,250",
            "د.إ. 1,250.00",
            "Total − 1250",
            "²",
            "①",
            "",
            "abc",
        ]
        alphabet = list("0123456789.,-() $%/:") + list(
            "−–－▲．，（）٫٬،\u200e\u200f\u061c\ufeff\u200b\u00a0²½①万درهمر.س合計"
        )
        rng = random.Random(559)
        values += [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
            for _ in range(5000)
        ]
        assert sum(not value.isascii() for value in values) > 4000
        negated = 0
        for value in values:
            before, now = _reading_before_559(value), self.reads_as(value)
            if now != before:
                assert before is not None and before > 0, repr(value)
                assert now == -before, repr(value)
                negated += 1
        assert 0 < negated < len(values) // 5

    def test_bilingual_values(self):
        """An ASCII amount beside the same amount in Arabic-Indic or Persian
        digits reads as the amount, in either order and with any of these
        separators between them. A decimal point between them would join them
        into one number."""
        rng = random.Random(559)
        scripts = [
            str.maketrans("0123456789,.", "٠١٢٣٤٥٦٧٨٩٬٫"),
            str.maketrans("0123456789,.", "۰۱۲۳۴۵۶۷۸۹٬٫"),
        ]
        separators = [
            " ",
            "  ",
            "\n",
            "\t",
            " / ",
            " | ",
            "; ",
            "، ",
            " (",
            "\u200f",
            "\u200e",
            "\u061c",
            "\u200b",
            "\u2068",
            "\u2067",
        ]
        for _ in range(3000):
            number = f"{rng.randint(0, 99999):,}"
            if rng.random() < 0.5:
                number += f".{rng.randint(0, 99):02d}"
            negative = rng.random() < 0.3
            amount = ("-" if negative else "") + number
            other = ("\u200e−" if negative else "") + number.translate(
                rng.choice(scripts)
            )
            separator = rng.choice(separators)
            close = {" (": ")", "\u2068": "\u2069", "\u2067": "\u2069"}.get(
                separator, ""
            )
            pair = [amount, other]
            rng.shuffle(pair)
            value = f"{pair[0]}{separator}{pair[1]}{close}"
            expected = Decimal(amount.replace(",", ""))
            assert self.reads_as(value) == expected, repr(value)

    @pytest.mark.parametrize("value", ["١٢٥٠.1250", "١٬٢٥٠.1,250"])
    def test_an_ascii_point_joins_even_the_same_amount(self, value):
        """The old reading takes an ASCII "." as a decimal point, so the #559
        false match ("١٢٥٠.1250" read 0.1250) stays closed."""
        assert self.reads_as(value) is None

    @pytest.mark.parametrize(
        "value", ["1.2.3٤", "14.05.202٤", "192.168.1.١", "١.٢.٣4", "١٢.٠٥.2024"]
    )
    def test_dotted_runs_switching_script_read_as_no_number(self, value):
        """A version, an IP address or a date with one digit in another script."""
        assert self.reads_as(value) is None
        assert self.comparator.compare("1250", value) == 0.0

    def test_digits_switching_script_twice(self):
        """Digits switching script twice in one run are never one clean number."""
        assert self.reads_as("1250.00．١٢٥٠٫٥0") is None

    @pytest.mark.parametrize("unit", ["1.١.", "1,١١١١,", "١٢50 ", "1٬٢"])
    def test_long_values_read_in_linear_time(self, unit):
        """The script-switch check scans each run of digits once."""
        value = unit * 20000
        started = time.perf_counter()
        reading = self.reads_as(value)
        assert time.perf_counter() - started < 2.0
        assert reading is None

    def test_non_decimal_digit_characters_are_not_numbers(self):
        """Superscripts and circled digits are digits but not decimal digits."""
        assert self.comparator.compare("2", "²") == 0.0
        assert self.comparator.compare("1", "①") == 0.0
