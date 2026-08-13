"""Phone number comparison comparator."""

from typing import Any, Dict, Optional

from stickler.comparators.base import BaseComparator

#: Region used to interpret numbers written without an international prefix.
#: libphonenumber needs a region to resolve "555-123-4567"; an E164 number
#: ("+1555...") carries its own country code and ignores this.
DEFAULT_REGION = "US"


class PhoneComparator(BaseComparator):
    """Comparator that treats two phone numbers as equal if they dial the same.

    Formatting is not meaning. ``"555-123-4567"``, ``"(555) 123-4567"`` and
    ``"+1-555-123-4567"`` are one number written three ways, and an extraction
    pipeline will produce all three from the same document. Both sides are
    parsed and compared in E164 form, so punctuation, spacing, country-code
    prefixes and extensions resolve before the comparison.

    This exists because no string comparator can do the job. ``ExactComparator``
    scores a reformatted number ``0.0``. ``NumericComparator`` strips non-digits
    and also reports ``0.0``. Edit distance ranks the two cases the wrong way
    round: ``"555-123-4567"`` against ``"555-123-4568"`` (a different number)
    scores ``0.917``, while the same number reformatted scores ``0.786`` -- so no
    threshold separates them.

    Unparseable input scores ``0.0``, including when both sides are equally
    unparseable. ``"N/A"`` on both sides is not a phone number that matched, it
    is a field that was not extracted, and reporting it as a true positive
    inflates the metric. Genuinely absent values never reach here: the shared
    ``None`` policy in :class:`BaseComparator` resolves those first, and the
    comparison layer treats ``None``/``""`` on both sides as a true negative.

    Example:
        ```python
        comparator = PhoneComparator()
        comparator.compare("555-123-4567", "(555) 123-4567")      # 1.0
        comparator.compare("+1-555-123-4567", "5551234567")        # 1.0
        comparator.compare("+1 (555) 123-4567 ext. 89", "+15551234567x89")  # 1.0
        comparator.compare("555-123-4567", "555-123-4568")         # 0.0
        comparator.compare("N/A", "N/A")                           # 0.0

        # Numbers written without an international prefix need a region
        uk = PhoneComparator(region="GB")
        uk.compare("+44 20 7183 8750", "02071838750")              # 1.0
        ```

    Args:
        threshold: Similarity threshold (default 1.0). This comparator returns
            only 0.0 or 1.0, so values below 1.0 accept any parsed match.
        region: Two-letter region code used to interpret numbers with no
            international prefix (default ``"US"``). Numbers written in E164
            form carry their own country code and are unaffected.

    .. versionadded:: 0.7.0
        Added so zero-config evaluation stops scoring formatting-only phone
        differences as complete mismatches (issue #242).
    """

    def __init__(self, threshold: float = 1.0, region: str = DEFAULT_REGION):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (default 1.0)
            region: Default region for numbers without an international prefix
        """
        super().__init__(threshold=threshold)
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

    def _to_e164(self, value: Any) -> Optional[str]:
        """Canonical E164 form, or None when the value is not a phone number."""
        import phonenumbers

        try:
            parsed = phonenumbers.parse(str(value), self.region)
        except phonenumbers.NumberParseException:
            return None

        return phonenumbers.format_number(
            parsed, phonenumbers.PhoneNumberFormat.E164
        )

    def _compare(self, str1: Any, str2: Any) -> float:
        """Compare two phone numbers by their canonical form.

        Args:
            str1: First value, never None
            str2: Second value, never None

        Returns:
            1.0 if both parse to the same E164 number, 0.0 otherwise
        """
        first = self._to_e164(str1)
        if first is None:
            return 0.0

        second = self._to_e164(str2)
        if second is None:
            return 0.0

        return 1.0 if first == second else 0.0

    def __repr__(self) -> str:
        """Detailed string representation."""
        parts = [f"threshold={self.threshold}"]
        if self.region != DEFAULT_REGION:
            parts.append(f"region={self.region!r}")
        return f"PhoneComparator({', '.join(parts)})"
