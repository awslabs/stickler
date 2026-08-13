"""Phone number comparison comparator."""

from typing import Any, Dict, Optional, Tuple

import phonenumbers

from stickler.comparators.base import BaseComparator

#: Region used to interpret numbers written without an international prefix.
#: libphonenumber needs a region to resolve "206-555-0100"; an E164 number
#: ("+1206...") carries its own country code and ignores this.
DEFAULT_REGION = "US"


class PhoneComparator(BaseComparator):
    """Comparator that treats two phone numbers as equal if they dial the same.

    Formatting is not meaning. ``"206-555-0100"``, ``"(206) 555-0100"`` and
    ``"+1-206-555-0100"`` are one number written three ways, and an extraction
    pipeline will produce all three from the same document. Both sides are
    parsed and compared in E164 form, so punctuation, spacing and country-code
    prefixes resolve before the comparison. Extensions are compared separately,
    because E164 does not carry them.

    This exists because no string comparator can do the job. ``ExactComparator``
    scores a reformatted number ``0.0``. ``NumericComparator`` strips non-digits
    and also reports ``0.0``. Edit distance ranks the two cases the wrong way
    round: a *different* number scores higher than the same number reformatted,
    so no threshold separates them.

    Both sides must be **valid** numbers, not merely parseable. libphonenumber
    parses ``"0000000000"``, ``"1234567"`` and ``"1111111111"`` and renders each
    as E164, so a parse-only check scores those ``1.0`` against themselves -- a
    placeholder on both sides reported as a successful match. Validity is
    checked with ``is_valid_number``, which rejects them.

    Note that ``is_valid_number`` also rejects ``555`` as an *area code*
    (``"555-123-4567"``), because NANP reserves it to dial nothing. Fictional
    numbers for documentation and tests need a real area code with the ``555``
    exchange, for example ``"206-555-0100"``.

    Unparseable or invalid input scores ``0.0``, including when both sides are
    identical. ``"N/A"`` on both sides is not a phone number that matched, it is
    a field that was not extracted, and reporting it as a true positive inflates
    the metric. Genuinely absent values never reach here: the shared ``None``
    policy in :class:`BaseComparator` resolves those first, and the comparison
    layer treats ``None``/``""`` on both sides as a true negative.

    Example:
        ```python
        comparator = PhoneComparator()
        comparator.compare("206-555-0100", "(206) 555-0100")     # 1.0
        comparator.compare("+1-206-555-0100", "2065550100")       # 1.0
        comparator.compare("206-555-0100", "206-555-0101")        # 0.0
        comparator.compare("N/A", "N/A")                          # 0.0
        comparator.compare("0000000000", "0000000000")            # 0.0 (not valid)

        # Extensions are significant: they reach a different person
        comparator.compare("+12065550100x89", "+12065550100x89")  # 1.0
        comparator.compare("+12065550100x89", "+12065550100x90")  # 0.0
        comparator.compare("+12065550100x89", "+12065550100")     # 0.0

        # Numbers written without an international prefix need a region
        uk = PhoneComparator(region="GB")
        uk.compare("+44 20 7183 8750", "02071838750")             # 1.0
        ```

    Args:
        threshold: Similarity threshold (default 1.0). This comparator returns
            only 0.0 or 1.0, so values below 1.0 accept any valid match.
        region: Two-letter ISO 3166-1 region code used to interpret numbers with
            no international prefix (default ``"US"``). Numbers written in E164
            form carry their own country code and are unaffected.

    Raises:
        ValueError: If ``region`` is not a region libphonenumber recognises. A
            plausible typo such as ``"UK"`` (the ISO code is ``"GB"``) would
            otherwise make every national-format number fail to parse and score
            ``0.0``, which reads as total extraction failure rather than a
            configuration mistake -- and E164 inputs would keep working, hiding
            it further.

    .. versionadded:: 0.7.0
        Added so zero-config evaluation stops scoring formatting-only phone
        differences as complete mismatches (issue #242).
    """

    def __init__(self, threshold: float = 1.0, region: str = DEFAULT_REGION):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (default 1.0)
            region: Default region for numbers without an international prefix

        Raises:
            ValueError: If ``region`` is not recognised by libphonenumber.
        """
        super().__init__(threshold=threshold)

        if region not in phonenumbers.SUPPORTED_REGIONS:
            raise ValueError(
                f"Unknown region {region!r}. Expected an ISO 3166-1 alpha-2 code "
                f"such as 'US' or 'GB' (the code for the United Kingdom is 'GB', "
                f"not 'UK'). An unrecognised region would make every "
                f"national-format number score 0.0 with no error raised."
            )

        self.region = region

    @property
    def name(self) -> str:
        """Return the name of the comparator."""
        return "phone"

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Return configuration parameters for serialization.

        Only includes non-default values to keep serialized output minimal.
        """
        cfg: Dict[str, Any] = {}
        if self.region != DEFAULT_REGION:
            cfg["region"] = self.region
        return cfg if cfg else None

    def _canonical(self, value: Any) -> Optional[Tuple[str, Optional[str]]]:
        """Canonical ``(E164, extension)`` pair, or None if not a valid number.

        The extension is returned separately because
        ``PhoneNumberFormat.E164`` omits it, so comparing E164 alone would treat
        two different destinations behind one switchboard as the same number.
        """
        try:
            parsed = phonenumbers.parse(str(value), self.region)
        except phonenumbers.NumberParseException:
            return None

        # Parseable is not the same as real: "0000000000" parses and formats.
        if not phonenumbers.is_valid_number(parsed):
            return None

        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        return e164, (parsed.extension or None)

    def _compare(self, str1: Any, str2: Any) -> float:
        """Compare two phone numbers by their canonical form.

        Args:
            str1: First value, never None
            str2: Second value, never None

        Returns:
            1.0 if both are valid numbers with the same E164 form and the same
            extension, 0.0 otherwise
        """
        first = self._canonical(str1)
        if first is None:
            return 0.0

        second = self._canonical(str2)
        if second is None:
            return 0.0

        return 1.0 if first == second else 0.0

    def __repr__(self) -> str:
        """Detailed string representation."""
        parts = [f"threshold={self.threshold}"]
        if self.region != DEFAULT_REGION:
            parts.append(f"region={self.region!r}")
        return f"PhoneComparator({', '.join(parts)})"
