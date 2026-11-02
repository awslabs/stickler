"""Helper class for null value checking and validation.

This module provides utility methods for checking various null conditions
used throughout the comparison process.
"""

import math
from typing import Any, Iterable, List, Optional, Sequence


class NullHelper:
    """Helper class for null value checking and validation."""

    @staticmethod
    def is_absent_element(item: Any) -> bool:
        """Whether a list element is an absence rather than a value.

        A ``None`` element, or a float NaN (see :meth:`is_nan`), is left out of
        a list before matching: it is never paired, counted or scored, and a
        list holding only absent elements is an empty list. Counting it as an
        element credited two ``None``s as a true positive and scored a missed
        row as a false discovery (#386). An empty string is still a value.
        Elements keep their original indices in reports. See "``None`` elements
        in a list" in ``docs/docs/Advanced/classification-logic.md``.
        """
        return item is None or NullHelper.is_nan(item)

    @staticmethod
    def present(items: Optional[Iterable[Any]]) -> List[Any]:
        """The elements of ``items`` that are not absent, in order."""
        return [item for item in items or () if not NullHelper.is_absent_element(item)]

    @staticmethod
    def present_indices(items: Optional[Sequence[Any]]) -> List[int]:
        """The original indices of the elements of ``items`` that are not absent."""
        return [
            i
            for i, item in enumerate(items or ())
            if not NullHelper.is_absent_element(item)
        ]

    @staticmethod
    def first_present(*sequences: Optional[Iterable[Any]]) -> Any:
        """The first element that is not absent, across ``sequences`` in order.

        What a list holds is decided from one representative element, and an
        absent element cannot be it: a ``List[Optional[Model]]`` that starts
        with ``None`` otherwise reads as a list of primitives, skipping the
        class gate and field-by-field scoring (#386), and takes its Hungarian
        ``match_threshold`` from the parent instead of the element class
        (#322). Pass the ground truth first, then the prediction, so a list that
        is all ``None`` on one side is still read from the other.

        Returns ``None`` if no sequence has a present element.
        """
        for sequence in sequences:
            for item in sequence or ():
                if not NullHelper.is_absent_element(item):
                    return item
        return None

    @staticmethod
    def is_truly_null(val: Any) -> bool:
        """Check if a value is truly null (None).

        Args:
            val: Value to check

        Returns:
            True if the value is None, False otherwise
        """
        return val is None

    @staticmethod
    def is_nan(val: Any) -> bool:
        """Check if a value is a float NaN.

        NaN is read as absent, the same as ``None``: it is how pandas and numpy
        spell a missing number, and it is not valid JSON, so it is never a value
        an extractor was asked to produce. Reading it as a value scored two
        identical NaNs 0.0, because NaN equals nothing under IEEE 754
        (`#367 <https://github.com/awslabs/stickler/issues/367>`_).

        Args:
            val: Value to check

        Returns:
            True if the value is a float (including ``numpy.float64``) NaN
        """
        return isinstance(val, float) and math.isnan(val)

    @staticmethod
    def is_effectively_null_for_lists(val: Any) -> bool:
        """Check if a list value is effectively null.

        A list whose elements are all absent (see :meth:`is_absent_element`)
        is an empty list, so ``[None]`` is null the same as ``[]``.

        Args:
            val: Value to check

        Returns:
            True if the value is None or a list with no present element
        """
        return val is None or (
            isinstance(val, list)
            and all(NullHelper.is_absent_element(item) for item in val)
        )

    @staticmethod
    def is_effectively_null_for_primitives(val: Any) -> bool:
        """Check if a non-list value is effectively null.

        Treats ``None``, a float NaN (see :meth:`is_nan`), an empty string and
        an empty dict as equivalent, which is the rule the docs state: "Empty strings (``""``), empty lists
        (``[]``), and empty objects (``{}``) are treated as null." Empty lists
        are the one case this does *not* answer, because list-ness is decided
        from the annotation rather than the value -- see
        :meth:`is_effectively_null_for_lists`, which ``ComparisonDispatcher``
        reaches first for any field ``_is_list_field`` recognises.

        ``{}`` was missing here, and its absence was visible in the metrics
        rather than benign: two **identical** objects each holding ``{}`` in a
        dict field classified as a false discovery, the same contradiction
        #233 reports for lists. It was also unreachable to score, because no
        comparator accepts a dict -- ``LevenshteinComparator`` raises
        ``TypeError`` -- so a populated dict field made the pair uncomparable
        instead of merely mismatched. Reading ``{}`` as absent gives the three
        empty/null combinations a true negative and turns the fourth
        (``{}`` against populated) into a false alarm rather than a crash.

        Args:
            val: Value to check

        Returns:
            True if the value is None, a float NaN, an empty string or an
            empty dict
        """
        return (
            val is None
            or NullHelper.is_nan(val)
            or (isinstance(val, (str, dict)) and len(val) == 0)
        )
