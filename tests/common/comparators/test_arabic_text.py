"""The text, date and phone comparators on Arabic and Persian text (#591).

These tests pin what each comparator scores today, so that a change to any of
them is deliberate. Which of these results are gaps rather than intended
behaviour is an open question in #591. ``NumericComparator`` is covered in
``test_numeric.py``.

Invisible characters, and sequences that render the same as another sequence,
are written as escapes: the precomposed and decomposed hamza, the presentation
forms, and the invisible marks.
"""

import pytest

from stickler.comparators import (
    ANLSStarComparator,
    DateComparator,
    ExactComparator,
    FuzzyComparator,
    LevenshteinComparator,
    NormalizedComparator,
    PhoneComparator,
)

TEXT_COMPARATORS = [
    ExactComparator,
    NormalizedComparator,
    LevenshteinComparator,
    FuzzyComparator,
    ANLSStarComparator,
]

# The same word or value written two ways, each seen in Arabic or Persian
# documents, keyed by the difference.
VARIANTS = {
    "hamza-on-alef": ("أحمد", "احمد"),
    "ta-marbuta-and-ha": ("مدرسة", "مدرسه"),
    "alef-maqsura-and-ya": ("مصطفى", "مصطفي"),
    "diacritics": ("مُحَمَّد", "محمد"),
    "tatweel": ("محـــمد", "محمد"),
    "persian-and-arabic-letter-forms": ("یک کتاب", "يك كتاب"),
    "arabic-indic-digits-in-an-id": ("INV-١٢٥٠", "INV-1250"),
}

# Alef with hamza above, precomposed (U+0623) and as alef plus a combining
# hamza (U+0627 U+0654). The two usually render identically.
COMPOSED_HAMZA = "\u0623\u062d\u0645\u062f"
DECOMPOSED_HAMZA = "\u0627\u0654\u062d\u0645\u062f"

RIGHT_TO_LEFT_MARK = "\u200f"
ZERO_WIDTH_NON_JOINER = "\u200c"

# Arabic presentation forms, which text extracted from some PDFs contains,
# next to the ordinary letters they display as: positional forms of a word,
# and the lam-alef ligature.
PRESENTATION_FORMS = {
    "positional-forms": ("\ufee3\ufea4\ufee4\ufeaa", "محمد"),
    "lam-alef-ligature": ("\ufefb", "لا"),
}


@pytest.mark.parametrize("comparator_class", TEXT_COMPARATORS)
def test_identical_arabic_text_matches(comparator_class):
    text = "شركة الإمارات للتجارة"
    assert comparator_class().compare(text, text) == 1.0


@pytest.mark.parametrize("difference", list(VARIANTS))
@pytest.mark.parametrize("comparator_class", [ExactComparator, NormalizedComparator])
def test_exact_and_normalized_score_written_variants_as_different(
    comparator_class, difference
):
    first, second = VARIANTS[difference]
    assert comparator_class().compare(first, second) == 0.0


# Partial credit today for each variant, per comparator. On plain strings
# ANLS* scores like Levenshtein, except that a score below its leaf threshold
# (0.5 by default) becomes 0.0; see the test after this one.
PARTIAL_SCORES = {
    LevenshteinComparator: {
        "hamza-on-alef": 3 / 4,
        "ta-marbuta-and-ha": 4 / 5,
        "alef-maqsura-and-ya": 4 / 5,
        "diacritics": 1 / 2,
        "tatweel": 4 / 7,
        "persian-and-arabic-letter-forms": 4 / 7,
        "arabic-indic-digits-in-an-id": 1 / 2,
    },
    FuzzyComparator: {
        "hamza-on-alef": 3 / 4,
        "ta-marbuta-and-ha": 4 / 5,
        "alef-maqsura-and-ya": 4 / 5,
        "diacritics": 2 / 3,
        "tatweel": 8 / 11,
        "persian-and-arabic-letter-forms": 4 / 7,
        "arabic-indic-digits-in-an-id": 1 / 2,
    },
    ANLSStarComparator: {
        "hamza-on-alef": 3 / 4,
        "ta-marbuta-and-ha": 4 / 5,
        "alef-maqsura-and-ya": 4 / 5,
        "diacritics": 1 / 2,
        "tatweel": 4 / 7,
        "persian-and-arabic-letter-forms": 4 / 7,
        "arabic-indic-digits-in-an-id": 1 / 2,
    },
}


@pytest.mark.parametrize(
    ("comparator_class", "difference"),
    [
        (comparator_class, difference)
        for comparator_class, scores in PARTIAL_SCORES.items()
        for difference in scores
    ],
)
def test_similarity_comparators_give_partial_credit_to_written_variants(
    comparator_class, difference
):
    """Every variant gets partial credit. Diacritics and tatweel are separate
    code points, so each one costs an edit."""
    first, second = VARIANTS[difference]
    score = comparator_class().compare(first, second)
    assert score == pytest.approx(PARTIAL_SCORES[comparator_class][difference])


def test_anls_star_scores_zero_once_similarity_falls_below_its_leaf_threshold():
    """One more diacritic (tanween) takes Levenshtein below 0.5, and ANLS*
    then scores 0.0 instead of partial credit. The diacritics case in the
    partial-credit test scores exactly 0.5, so that test catches a change from
    ``<`` to ``<=``."""
    first, second = "مُحَمَّدٌ", "محمد"
    assert LevenshteinComparator().compare(first, second) == pytest.approx(4 / 9)
    assert ANLSStarComparator().compare(first, second) == 0.0


def test_normalized_ignores_arabic_punctuation_and_all_whitespace():
    comparator = NormalizedComparator()
    assert comparator.compare("مرحبا، كيف حالك؟", "مرحبا كيف حالك") == 1.0
    assert comparator.compare("دبي  مول", "دبي مول") == 1.0
    assert comparator.compare("دبيمول", "دبي مول") == 1.0


def test_exact_does_not_ignore_arabic_punctuation_or_whitespace():
    comparator = ExactComparator()
    assert comparator.compare("مرحبا، كيف حالك؟", "مرحبا كيف حالك") == 0.0
    assert comparator.compare("دبي  مول", "دبي مول") == 0.0


HAMZA_SPELLING_SCORES = {
    NormalizedComparator: 1.0,
    ExactComparator: 0.0,
    LevenshteinComparator: 3 / 5,
    FuzzyComparator: 2 / 3,
    ANLSStarComparator: 3 / 5,
}


@pytest.mark.parametrize("comparator_class", list(HAMZA_SPELLING_SCORES))
def test_only_normalized_matches_composed_and_decomposed_hamza(comparator_class):
    """Normalized applies NFC first; the other text comparators apply no NFC or
    NFKC normalization."""
    score = comparator_class().compare(COMPOSED_HAMZA, DECOMPOSED_HAMZA)
    assert score == pytest.approx(HAMZA_SPELLING_SCORES[comparator_class])


RIGHT_TO_LEFT_MARK_SCORES = {
    ExactComparator: 0.0,
    NormalizedComparator: 0.0,
    LevenshteinComparator: 3 / 4,
    FuzzyComparator: 6 / 7,
    ANLSStarComparator: 3 / 4,
}


@pytest.mark.parametrize("comparator_class", list(RIGHT_TO_LEFT_MARK_SCORES))
def test_a_right_to_left_mark_changes_the_score(comparator_class):
    """The mark is a format character (Cf), and Normalized removes only
    punctuation and whitespace."""
    marked = RIGHT_TO_LEFT_MARK + "دبي"
    assert comparator_class().compare(marked, "دبي") == pytest.approx(
        RIGHT_TO_LEFT_MARK_SCORES[comparator_class]
    )


@pytest.mark.parametrize("pair", list(PRESENTATION_FORMS))
@pytest.mark.parametrize("comparator_class", TEXT_COMPARATORS)
def test_presentation_forms_do_not_match_the_letters_they_display(
    comparator_class, pair
):
    """No comparator applies NFKC, which would map these to ordinary letters."""
    first, second = PRESENTATION_FORMS[pair]
    assert comparator_class().compare(first, second) == 0.0


ZERO_WIDTH_NON_JOINER_SCORES = {
    ExactComparator: 0.0,
    NormalizedComparator: 0.0,
    LevenshteinComparator: 7 / 8,
    FuzzyComparator: 14 / 15,
    ANLSStarComparator: 7 / 8,
}


@pytest.mark.parametrize("comparator_class", list(ZERO_WIDTH_NON_JOINER_SCORES))
def test_a_zero_width_non_joiner_changes_the_score(comparator_class):
    """Persian writes the non-joiner inside words; it is a format character
    (Cf), like the right-to-left mark."""
    joined = "می" + ZERO_WIDTH_NON_JOINER + "خواهم"
    score = comparator_class().compare(joined, "میخواهم")
    assert score == pytest.approx(ZERO_WIDTH_NON_JOINER_SCORES[comparator_class])


# The date tests depend on python-dateutil reading these digits, and on
# DateComparator's year-first check matching them through ``\d``. The phone
# tests depend on phonenumberslite, and the Fuzzy scores above on rapidfuzz.


@pytest.mark.parametrize(
    "value",
    ["١٤/٠٥/٢٠٢٤", "٢٠٢٤-٠٥-١٤", "۱۴/۰۵/۲۰۲۴"],
    ids=["arabic-indic", "arabic-indic-iso", "persian"],
)
def test_dates_in_arabic_indic_and_persian_digits_match(value):
    assert DateComparator().compare("2024-05-14", value) == 1.0


def test_a_different_date_in_arabic_indic_digits_does_not_match():
    assert DateComparator().compare("2024-05-14", "١٥/٠٥/٢٠٢٤") == 0.0
    # The same value matches its own date, so the 0.0 above is about the day.
    assert DateComparator().compare("2024-05-15", "١٥/٠٥/٢٠٢٤") == 1.0


@pytest.mark.parametrize(
    "value", ["٢٠٢٤-٠٣-٠٤", "۲۰۲۴-۰۳-۰۴"], ids=["arabic-indic", "persian"]
)
def test_a_year_first_date_in_non_ascii_digits_is_read_year_month_day(value):
    """Day 4 and month 3 could be read either way; the year-first check keeps
    the order."""
    assert DateComparator().compare("2024-03-04", value) == 1.0
    assert DateComparator().compare("2024-04-03", value) == 0.0


@pytest.mark.parametrize(
    "value",
    ["14 مايو 2024", "١٤ مايو ٢٠٢٤"],
    ids=["ascii-digits", "arabic-indic"],
)
def test_a_date_with_an_arabic_month_name_does_not_match(value):
    assert DateComparator().compare("2024-05-14", value) == 0.0


@pytest.mark.parametrize(
    "value",
    ["14 May 2024", "١٤ May ٢٠٢٤"],
    ids=["ascii-digits", "arabic-indic"],
)
def test_the_same_date_with_an_english_month_name_matches(value):
    """So the 0.0 above is about the Arabic month name."""
    assert DateComparator().compare("2024-05-14", value) == 1.0


@pytest.mark.parametrize(
    "value",
    [
        "١٤" + RIGHT_TO_LEFT_MARK + "/٥" + RIGHT_TO_LEFT_MARK + "/٢٠٢٤",
        "14" + RIGHT_TO_LEFT_MARK + "/5" + RIGHT_TO_LEFT_MARK + "/2024",
    ],
    ids=["ar-eg", "ar-ae"],
)
def test_the_cldr_arabic_short_date_does_not_match(value):
    """CLDR's short date for these locales puts a right-to-left mark after the
    day and the month. Without the marks the same date matches."""
    assert DateComparator().compare("2024-05-14", value) == 0.0
    unmarked = value.replace(RIGHT_TO_LEFT_MARK, "")
    assert DateComparator().compare("2024-05-14", unmarked) == 1.0


def test_a_solar_hijri_date_does_not_match_the_same_gregorian_date():
    """1403/2/25 in the Solar Hijri calendar, CLDR's short date for fa-IR, is
    14 May 2024."""
    assert DateComparator().compare("2024-05-14", "۱۴۰۳/۲/۲۵") == 0.0


@pytest.mark.parametrize(
    "value",
    ["+٩٧١ ٥٠ ١٢٣ ٤٥٦٧", "+۹۷۱ ۵۰ ۱۲۳ ۴۵۶۷"],
    ids=["arabic-indic", "persian"],
)
def test_phone_numbers_in_arabic_indic_and_persian_digits_match(value):
    assert PhoneComparator().compare("+971 50 123 4567", value) == 1.0


def test_a_different_phone_number_in_arabic_indic_digits_does_not_match():
    assert PhoneComparator().compare("+971 50 123 4567", "+٩٧١ ٥٠ ١٢٣ ٤٥٦٨") == 0.0
