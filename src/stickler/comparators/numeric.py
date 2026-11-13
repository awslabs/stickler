"""Numeric comparison comparator."""

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Union

from stickler.comparators.base import BaseComparator

# Reading numbers written in other scripts (#559). A string whose digits are all
# outside ASCII read as no number before, so here None is never a regression and
# is the answer whenever the string does not hold one unambiguous number.

# U+066B and U+066C are the Arabic decimal and group separators, and U+2E41 is the
# Adlam group separator: each means one thing. "." and "," (and their small and
# fullwidth forms) mean opposite things in different locales.
_ARABIC_DECIMAL_POINT = "\u066b"
_POINTS = ".\ufe52\uff0e"
_COMMAS = ",\ufe50\uff0c"
_ALL_POINTS = _ARABIC_DECIMAL_POINT + _POINTS
_ALL_COMMAS = "\u066c\u2e41" + _COMMAS
# Scripts for which every CLDR and ICU locale that writes them uses "." as the
# decimal point, identified by the code point of their digit zero. Fullwidth
# digits follow the Japanese and Chinese conventions. A number with "." or ","
# in any other script is ambiguous: Lao, Khmer and Javanese locales use "," as
# the decimal point, as ICU 78 (Node 22) formats Mongolian digits and Extended
# Arabic-Indic digits for az_Arab, and other scripts have no locale that writes
# them natively.
_POINT_DECIMAL_SCRIPTS = frozenset(
    {
        0x0660,  # Arabic-Indic
        0x07C0,  # N'Ko
        0x0966,  # Devanagari
        0x09E6,  # Bengali
        0x0A66,  # Gurmukhi
        0x0AE6,  # Gujarati
        0x0B66,  # Oriya
        0x0BE6,  # Tamil
        0x0C66,  # Telugu
        0x0CE6,  # Kannada
        0x0D66,  # Malayalam
        0x0E50,  # Thai
        0x0F20,  # Tibetan
        0x1040,  # Myanmar
        0x1C50,  # Ol Chiki
        0xA620,  # Vai
        0xABF0,  # Meetei Mayek
        0xFF10,  # fullwidth
        0x11136,  # Chakma
        0x1E140,  # Nyiakeng Puachue Hmong
        0x1E950,  # Adlam
    }
)
# Characters written as a minus sign: U+2212 MINUS SIGN, the hyphens and dashes
# typed for it, and their small, fullwidth, superscript and subscript forms.
_NON_ASCII_MINUS_SIGNS = (
    "\u02d7\u2010\u2011\u2012\u2013\u2052\u207b\u208b\u2212\u2796\ufe63\uff0d"
)
_MINUS_SIGNS = "-" + _NON_ASCII_MINUS_SIGNS
# ▲ and △ mark a negative in Japanese accounting, and a rise in Korean and
# Chinese market tables. They are a minus sign with fullwidth digits; with other
# digits outside ASCII the value reads as no number.
_JAPANESE_MINUS_SIGNS = "\u25b2\u25b3"
# The Japanese tamper-proof dash written after an amount ("￥１，２５０－").
_TAMPER_PROOF_DASH = "\uff0d"
# ▼ and ▽ mark a fall, or in some Japanese reports a negative: with digits
# outside ASCII, a value holding one reads as no number.
_FALL_MARKS = "\u25bc\u25bd"
_OPEN_PARENS = "(\u207d\u208d\ufe59\uff08"
_CLOSE_PARENS = ")\u207e\u208e\ufe5a\uff09"
# Numerals that are not decimal digits, such as "½", "²", "Ⅻ" and "万". One that
# touches a number makes the number part of a larger numeral. Categories No and
# Nl are stable; the CJK numeral ideographs are category Lo and are listed, as
# unicodedata.numeric gave values to more of them in Unicode 15.1.
_CJK_NUMERALS = frozenset("〇一二三四五六七八九十百千万萬億亿兆拾佰仟廿卅")
_NUMBER = re.compile(r"[0-9]+(?:[.,][0-9]+)*")
_SEPARATOR_CHARS = _ALL_POINTS + _ALL_COMMAS
_DIGIT_RUN = re.compile(r"\d+(?:[" + re.escape(_SEPARATOR_CHARS) + r"]\d+)*")
# ISO 4217 codes of the currencies that are legal tender in some territory, as
# CLDR 47 lists them.
_CURRENCY_CODES = frozenset(
    (
        "AED AFN ALL AMD AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BRL "
        "BSD BTN BWP BYN BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP DZD "
        "EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR "
        "ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP "
        "LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MYR MZN NAD "
        "NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR "
        "SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SYP SZL THB TJS TMT TND TOP TRY "
        "TTD TWD TZS UAH UGX USD UYU UZS VES VND VUV WST XAF XCD XCG XOF XPF YER ZAR "
        "ZMW ZWG "
    ).split()
)
# Other currency symbols of two or more characters that CLDR 47 writes, with
# format characters removed as _prepare removes them. One-letter symbols such
# as "K" and "R" are left out: they are as likely to be a label, as in "(P ١٢)".
_CURRENCY_SYMBOLS = frozenset(
    (
        "Afl. Ar B/. Br Bs Bs. Bs.S CF CFP CFPF Cg. DA DT Db Dkr ECU Esc. FB FBu FC "
        "FCFA FCFP FG Fdj Ft F\u202fCFA Gs. Ikr KM Ksh Kz Kč Kčs LS Le Lekë Ls MK MTn "
        "NAf. NIS Nfk Nkr Nu. RF RM Rf Rp Rs Rs. S/ SR TSh UM UP USh VT Vẽj. den din. "
        "kn kr kr. mk soʻm zł öS Δρχ КМ Кч ЛСЛ ТМТ Тл ден. дин. зл крб. лв. р. сом "
        "сўм щ.д. ל״י أ.م. ج.س. ج.م. د.أ. د.إ. د.ب. د.ت. د.ج. د.س. د.ع. د.ك. د.ل. "
        "د.م. ر.إ. ر.س. ر.ع. ر.ق. ر.ي. ریال فرانک\u202fCFA ل.س. ل.ل. ރ. नेरू රු. "
        "සිෆ්එ ብር"
    ).split(" ")
)
# No-break spaces join the parts of one currency symbol, as in "F\u202fCFA".
_NO_BREAK_SPACES = "\u00a0\u2007\u202f"


def _is_ascii_digit(char: str) -> bool:
    return "0" <= char <= "9"


def _is_other_script_digit(char: str) -> bool:
    return char.isdecimal() and not char.isascii()


def _digit_zero(char: str) -> int:
    """Code point of the digit zero of ``char``'s script."""
    return ord(char) - unicodedata.decimal(char)


def _digit_script(char: str) -> int:
    """``_digit_zero``, with Arabic-Indic and Extended Arabic-Indic counted as one
    script: Persian and Urdu text mixes them."""
    zero = _digit_zero(char)
    return 0x0660 if zero == 0x06F0 else zero


def _is_numeral(char: str) -> bool:
    return unicodedata.category(char) in ("No", "Nl") or char in _CJK_NUMERALS


def _prepare(text: str) -> str:
    """Drop format characters (category Cf), such as the direction marks CLDR
    puts beside a sign, except between two digits, where one separates two
    numbers ("١٢\u200b٣٤", "\u06dd١\u06dd٢") and becomes a space. A Devanagari
    zero after a Devanagari letter is the abbreviation mark typed in "रु०" (Rs.)
    and becomes U+0970, the abbreviation sign."""
    chars = []
    format_character_skipped = False
    for char in text:
        if char == "\u00ad":  # SOFT HYPHEN, a line-break hint inside a word
            continue
        if unicodedata.category(char) == "Cf":
            format_character_skipped = True
            continue
        if format_character_skipped and chars and chars[-1].isdecimal():
            if char.isdecimal():
                chars.append(" ")
        format_character_skipped = False
        if (
            char == "\u0966"
            and chars
            and "\u0900" <= chars[-1] <= "\u097f"
            and unicodedata.category(chars[-1])[0] in "LM"
        ):
            char = "\u0970"
        chars.append(char)
    return "".join(chars)


def _amount(text: str) -> str:
    """Digits and decimal point of a run of digits and separators, in ASCII."""
    return "".join(
        str(unicodedata.decimal(char))
        if char.isdecimal()
        else "."
        if char in _ALL_POINTS
        else ""
        for char in text
    ).strip(".")


def _same_amount(first: str, second: str) -> bool:
    """Whether two runs of digits are the same amount. A run with two decimal
    points ("1.2.3") is no amount."""
    first, second = _amount(first), _amount(second)
    if first.count(".") > 1 or second.count(".") > 1:
        return False
    return Decimal(first or "0") == Decimal(second or "0")


def _switches_digit_script(text: str) -> bool:
    """True when one number's digits switch script: directly ("١٢50"), or across
    a separator the digits after it continue ("١٢٥٠.50", "1,٢٥٠٫٥٠"). Two numbers
    split by a space or by a second decimal point ("1250.00．١٢٥١"), and the same
    amount written twice ("1250١٢٥٠", "1,250.00．١٬٢٥٠٫٠٠"), do not."""
    for match in _DIGIT_RUN.finditer(text):
        run = match.group()
        digits = [index for index, char in enumerate(run) if char.isdecimal()]
        switches = [
            (previous, index)
            for previous, index in zip(digits, digits[1:])
            if _digit_script(run[previous]) != _digit_script(run[index])
        ]
        if not switches:
            continue
        if len(switches) > 1:
            return True
        previous, index = switches[0]
        # The same amount twice is two copies, unless an ASCII "." follows the
        # other-script copy: the old reading takes that "." as a decimal point
        # ("١٢٥٠.1250" read 0.1250).
        if _same_amount(run[: previous + 1], run[index:]) and not (
            run[previous + 1] == "." and not run[previous].isascii()
        ):
            continue
        if previous + 1 == index:
            return True
        separator = run[previous + 1]
        if separator in _ALL_POINTS:
            if sum(char in _ALL_POINTS for char in run) == 1:
                return True
        else:
            group = index
            while group < len(run) and run[group].isdecimal():
                group += 1
            if group - index in (2, 3):
                return True
    return False


def _is_currency_word(text: str) -> bool:
    """A currency written without a currency sign, as CLDR writes one: an ISO
    4217 code ("SAR") or another CLDR symbol ("ر.س.", "ریال", "Rs.",
    "F\u202fCFA"), also without its final dot when it is not in ASCII ("ر.س").
    Symbols of ASCII letters only, such as "KM" and "Le", are left out: they are
    as likely to be a unit or a word."""
    text = text.strip()
    if not _is_token(text):
        return False
    if text in _CURRENCY_CODES:
        return True
    if text.isascii():
        return text in _CURRENCY_SYMBOLS and not text.isalpha()
    return text in _CURRENCY_SYMBOLS or (
        len(text) > 1 and text + "." in _CURRENCY_SYMBOLS
    )


def _is_currency_beside_number(word: str, gap: str) -> bool:
    """Whether ``word``, with ``gap`` between it and the number, is a currency.
    A symbol with a currency sign ("$ ١٢٥٠") always is. A currency word must be
    joined to the number as CLDR always joins it, by nothing or a no-break
    space: after a typed space ("(SAR ١٢٥٠)", "(KM ١٢)") it may be a label."""
    if any(unicodedata.category(char) == "Sc" for char in word):
        return _is_token(word)
    return _is_currency_word(word) and all(char in _NO_BREAK_SPACES for char in gap)


def _is_token(text: str) -> bool:
    """One word with no digit, sign or parenthesis, such as a currency."""
    return bool(text) and not any(
        (char.isspace() and char not in _NO_BREAK_SPACES)
        or char.isdecimal()
        or char in "-()"
        for char in text
    )


def _sign(before: str) -> Optional[int]:
    """Sign given by the text before a number, in which every minus form is "-".

    CLDR writes the minus first, before a currency, or right after a currency
    symbol, so "-١٢٥٠", "-AED\xa0١٢٥٠" and "₹-١٢٥٠" are negative. A minus after a
    word may be a label or an ID ("INV-١٢٥٠", "المجموع – ١٢٥٠"), and a minus
    before a word that is not a currency may be a bullet ("- Total: ١٢٥٠"):
    both give None. A minus inside an earlier word ("Sub-total: ৳١٢٥٠") is
    ignored."""
    index = before.rfind("-")
    if index < 0:
        return 1
    lead, between = before[:index].strip(), before[index + 1 :].strip()
    if not lead:
        if not between:
            return -1
        gap = before[index + 1 :].lstrip()[len(between) :]
        return -1 if _is_currency_beside_number(between, gap) else None
    if all(unicodedata.category(char)[0] == "S" for char in lead):
        return -1 if before[index + 1 :] == "" else None
    if not between or _is_token(between):
        return None
    return 1


def _wrapped_in_parentheses(text: str, start: int, end: int) -> Optional[bool]:
    """Whether ``text`` is an accounting negative: wrapped in parentheses holding
    only the number at ``text[start:end]`` and at most one currency (see
    ``_is_currency_beside_number``). Parentheses that hold anything else, a note
    such as "（税込１，２５０）" (tax included), give None."""
    opening = len(text) - len(text.lstrip())
    closing = len(text.rstrip()) - 1
    if not (
        opening < start
        and end <= closing
        and text[opening] in _OPEN_PARENS
        and text[closing] in _CLOSE_PARENS
    ):
        return False
    before, after = text[opening + 1 : start], text[end:closing]
    words = [
        (word, gap)
        for word, gap in (
            (before.strip(), before[len(before.rstrip()) :]),
            (after.strip(), after[: len(after) - len(after.lstrip())]),
        )
        if word
    ]
    if len(words) > 1:
        return None
    if all(_is_currency_beside_number(word, gap) for word, gap in words):
        return True
    return None


def _read_other_script_number(text: str) -> Optional[Decimal]:
    """Read a prepared value whose digits are all outside ASCII, or None unless it
    holds exactly one unambiguous number.

    The digits are read through ``unicodedata``, so the scripts covered are those
    of the running Python's Unicode version. No number is read when the value
    holds two numbers (a date, "۲/۵", an exponent, "١٢٥٠ م٢", "۱،۲۵۰"), when a
    "," is not a thousands group, when "." or "," is used with a script outside
    ``_POINT_DECIMAL_SCRIPTS``, when a numeral such as "½" or "万" touches the
    number, or when the sign is ambiguous.
    """
    zeros = {_digit_zero(char) for char in text if char.isdecimal()}
    if any(char in _FALL_MARKS for char in text) or (
        zeros != {0xFF10} and any(char in _JAPANESE_MINUS_SIGNS for char in text)
    ):
        return None
    chars = []
    for char in text:
        if char.isdecimal():
            chars.append(str(unicodedata.decimal(char)))
        elif char in _ALL_POINTS:
            chars.append(".")
        elif char in _ALL_COMMAS:
            chars.append(",")
        elif char in _MINUS_SIGNS + _JAPANESE_MINUS_SIGNS:
            chars.append("-")
        elif char in _OPEN_PARENS:
            chars.append("(")
        elif char in _CLOSE_PARENS:
            chars.append(")")
        else:
            chars.append(char)
    normalized = "".join(chars)

    numbers = list(_NUMBER.finditer(normalized))
    if len(numbers) != 1:
        return None
    start, end = numbers[0].span()
    # A "." before the digits is a decimal point (".5", "Amount:.5", "-.5"),
    # unless a letter or another "." comes right before it: then it ends an
    # abbreviation ("रु.५००") or a dot leader ("Total......١٢٥٠"). After a word
    # and a space it may be either ("AED .٥٠", "Total .١٢٥٠"), so the value
    # reads as no number. U+066B is always the decimal point.
    if normalized[start - 1 : start] == ".":
        previous = normalized[start - 2 : start - 1]
        earlier = normalized[: start - 1].rstrip()[-1:]
        if text[start - 1] == _ARABIC_DECIMAL_POINT or not (
            _ends_word(previous) or (previous.isspace() and _ends_word(earlier))
        ):
            start -= 1
        elif previous.isspace():
            return None
    if (start > 0 and _is_numeral(text[start - 1])) or (
        end < len(text) and _is_numeral(text[end])
    ):
        return None
    if any(char in _POINTS + _COMMAS for char in text[start:end]) and not (
        zeros <= _POINT_DECIMAL_SCRIPTS
    ):
        return None

    number = normalized[start:end]
    integer, _, fraction = number.partition(".")
    if "." in fraction or "," in fraction:
        return None
    if "," in integer and not _is_grouped(integer):
        return None
    # A minus right after the number may be a trailing negative sign, so the
    # value reads as no number, as "1250-" did. The Japanese tamper-proof dash
    # is not a sign.
    if normalized[end : end + 1] == "-" and text[end] != _TAMPER_PROOF_DASH:
        return None

    wrapped = _wrapped_in_parentheses(normalized, start, end)
    if wrapped is None:
        return None
    sign = -1 if wrapped else _sign(normalized[:start])
    if sign is None:
        return None
    digits = integer.replace(",", "") + ("." + fraction if fraction else "")
    return Decimal("-" + digits if sign < 0 else digits)


def _ends_word(char: str) -> bool:
    """Whether ``char`` ends a word or a dot leader, so a "." after it is not a
    decimal point."""
    return char == "." or (bool(char) and unicodedata.category(char)[0] in "LM")


def _is_grouped(integer: str) -> bool:
    """Whether the comma groups are thousands ("1,234,567") or the Indian lakh
    form ("12,34,567")."""
    first, *groups = integer.split(",")
    if first.startswith("0") or groups[-1:] and len(groups[-1]) != 3:
        return False
    if 1 <= len(first) <= 3 and all(len(group) == 3 for group in groups):
        return True
    return 1 <= len(first) <= 2 and all(len(group) == 2 for group in groups[:-1])


def _ascii_reading_is_negative(text: str) -> bool:
    """For a prepared value read by the ASCII path, whether a minus sign or
    fullwidth parentheses mark it negative as they do in
    ``_read_other_script_number``. ASCII "(...)" is left to that path, and a
    value it already reads with a leading "-" is not negated twice."""
    digits = [index for index, char in enumerate(text) if _is_ascii_digit(char)]
    if not digits:
        return False
    start, end = digits[0], digits[-1] + 1
    stripped = text.strip()
    if stripped[:1] == "\uff08" and stripped[-1:] == "\uff09":
        if _wrapped_in_parentheses(text, start, end):
            return True
    before = "".join("-" if char in _MINUS_SIGNS else char for char in text[:start])
    # A minus sign that opens the value is a sign when only a currency, if
    # anything, comes between it and the number ("−AED 1250", as "-AED 1250"
    # reads). Before a word it may be a bullet ("– Total: 1250"); elsewhere it
    # needs a sign position ("₹−1250").
    if before.lstrip()[:1] == "-" and stripped[:1] in _NON_ASCII_MINUS_SIGNS:
        between = before.lstrip()[1:].strip()
        if not between or _is_currency_beside_number(between, ""):
            return True
    return _sign(before) == -1


class NumericComparator(BaseComparator):
    """Comparator for numeric values with configurable tolerance.

    This comparator extracts and compares numeric values from strings or numbers.
    It supports relative and absolute tolerance for comparison.
    Booleans are non-numeric inputs and score 0.0.

    String values may use decimal digits in other scripts (Arabic-Indic,
    Persian, Devanagari, Thai, fullwidth, ...). A string whose digits are all
    outside ASCII is read when it holds one unambiguous number, and otherwise
    reads as no number. A string with ASCII digits is read as before, so a
    bilingual value such as ``"1,250.00 (١٬٢٥٠٫٠٠)"`` ignores its other-script
    digits, except that a number whose digits switch script (``"١٢٥٠.50"``)
    reads as no number, Unicode minus signs and dashes right before the
    number or a currency, or right after a currency symbol, mark a negative
    (``"−1250"``), as do fullwidth accounting parentheses, and spaces and
    format characters around accounting parentheses are ignored. A ``float``, ``int`` or
    ``Decimal`` field rejects ``"١٬٢٥٠٫٥٠"`` before the comparator runs;
    declare the field ``str``.

    Example:
        ```python
        # Default exact matching
        exact = NumericComparator()
        exact.compare("123", "123.0")  # Returns 1.0
        exact.compare("123", "124")    # Returns 0.0

        # With tolerance
        approx = NumericComparator(relative_tolerance=0.1)  # 10% tolerance
        approx.compare("100", "109")   # Returns 1.0 (within 10%)
        approx.compare("100", "111")   # Returns 0.0 (beyond 10%)

        # Arabic-Indic digits and separators
        exact.compare("1,250.00", "١٬٢٥٠٫٠٠")  # Returns 1.0
        ```
    """

    DEFAULT_THRESHOLD = 1.0

    def __init__(
        self,
        threshold: Optional[float] = None,
        relative_tolerance: float = 0.0,
        absolute_tolerance: float = 0.0,
        tolerance: Optional[float] = None,
    ):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (default 1.0)
            relative_tolerance: Relative tolerance for comparison (default 0.0)
            absolute_tolerance: Absolute tolerance for comparison (default 0.0)
            tolerance: Alias for absolute_tolerance (for backward compatibility)
        """
        super().__init__(threshold=threshold)
        self.relative_tolerance = relative_tolerance

        # Handle tolerance alias for backward compatibility
        if tolerance is not None:
            if absolute_tolerance != 0.0:
                raise ValueError(
                    "Cannot specify both 'tolerance' and 'absolute_tolerance'. Use 'absolute_tolerance'."
                )
            self.absolute_tolerance = tolerance
        else:
            self.absolute_tolerance = absolute_tolerance

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Return configuration parameters for serialization."""
        config = {}
        if self.relative_tolerance != 0.0:
            config["relative_tolerance"] = self.relative_tolerance
        if self.absolute_tolerance != 0.0:
            config["absolute_tolerance"] = self.absolute_tolerance
        return config or None

    def _compare(self, str1: Any, str2: Any) -> float:
        """Compare two values numerically.

        Args:
            str1: First value
            str2: Second value

        Returns:
            1.0 if the numbers match within tolerance, 0.0 otherwise
        """
        # Extract numeric values
        num1 = self._extract_number(str1)
        num2 = self._extract_number(str2)

        if num1 is None or num2 is None:
            return 0.0

        # Check equality with tolerance
        if self._numbers_equal(num1, num2):
            return 1.0

        return 0.0

    def _extract_number(self, value: Any) -> Union[Decimal, None]:
        """Extract a numeric value from a string or number.

        Args:
            value: Value to extract a number from

        Returns:
            Decimal value or None if no valid number could be extracted
        """
        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return Decimal(str(value))

        if not isinstance(value, str):
            value = str(value)

        marked_negative = False
        if not value.isascii():
            value = _prepare(value)
            if any(map(_is_other_script_digit, value)):
                if _switches_digit_script(value):
                    return None
                if not any(map(_is_ascii_digit, value)):
                    return _read_other_script_number(value)
            marked_negative = _ascii_reading_is_negative(value)

        # Check for accounting notation: (123) means -123
        value = value.strip()
        is_negative = False
        if value.startswith("(") and value.endswith(")"):
            value = value[1:-1]  # Remove the parentheses
            is_negative = True

        # Remove common currency symbols and other non-numeric characters
        value = re.sub(r"[^0-9.-]", "", value)
        if marked_negative and not is_negative and not value.startswith("-"):
            is_negative = True

        # Handle empty string
        if not value:
            return None

        # Try to convert to Decimal
        try:
            decimal_value = Decimal(value)
            # Apply negative sign if accounting notation was used
            if is_negative:
                decimal_value = -decimal_value
            return decimal_value
        except InvalidOperation:
            return None

    def _numbers_equal(self, num1: Decimal, num2: Decimal) -> bool:
        """Check if two numbers are equal within tolerance.

        Args:
            num1: First number
            num2: Second number

        Returns:
            True if numbers are equal within tolerance, False otherwise
        """
        # Guard non-finite values before tolerance arithmetic: Decimal
        # NaN/Infinity subtraction raises InvalidOperation. A float NaN never
        # gets here -- ``compare()`` reads it as missing (#367) -- so this only
        # sees a Decimal NaN, which equals nothing. Infinite values match only
        # exactly.
        if num1.is_nan() or num2.is_nan():
            return False
        if num1.is_infinite() or num2.is_infinite():
            return num1 == num2

        if num1 == num2:
            return True

        # Check with relative tolerance
        if self.relative_tolerance > 0:
            # Handle zero case
            if num1 == 0:
                return abs(num2) <= self.relative_tolerance

            # Calculate relative difference using num1 as base
            relative_diff = abs(num1 - num2) / abs(num1)
            if relative_diff <= self.relative_tolerance:
                return True

        # Check with absolute tolerance
        if self.absolute_tolerance > 0:
            if abs(num1 - num2) <= self.absolute_tolerance:
                return True

        return False


# Class alias for backward compatibility with NumericExactC() in evaluation metrics
NumericExactC = NumericComparator
