"""Base class for comparators."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from stickler.utils.deprecation import warn_once


class BaseComparator(ABC):
    """Base class for all comparators.

    This class defines the interface that all comparators must implement.
    Comparators are used to compare two values and return a similarity score
    between 0.0 and 1.0, where 1.0 means the values are identical.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Warn when a constructible subclass's ``compare`` shadows the ``None`` policy.

        :meth:`compare` is the template method that owns the ``None`` policy.
        A subclass supplying its own ``compare`` -- directly or through a
        mixin -- wins over it by plain Python MRO, so ``None`` reaches user
        code and can be scored as a present value. That shadowing cannot be
        prevented, only reported.

        This only reports the subset that has no other signal. A subclass that
        never supplies ``_compare`` leaves the abstract slot unfilled, so
        ``ABCMeta`` already raises ``TypeError`` at construction naming
        ``_compare``; warning as well would just add noise ahead of a hard
        failure. The gap is the subclass of a *concrete* comparator, which
        inherits a working ``_compare``, constructs fine, and silently skips
        the policy. That is the one warned about here.

        The report is a warning rather than a ``TypeError`` because the
        condition is not decidable at class-definition time: an override that
        forwards both values to ``super().compare()`` unchanged keeps the
        policy intact and is perfectly correct, and only the override itself
        knows whether it does. Refusing the class would reject those along
        with the broken ones.

        Owning ``_compare`` is the escape hatch. A class defining both is
        customising the policy deliberately rather than sitting on the
        pre-1.0 interface, and stays quiet -- as do its subclasses.
        """
        super().__init_subclass__(**kwargs)

        if "_compare" in cls.__dict__:
            return

        # Abstract slot unfilled: ABCMeta raises TypeError at construction,
        # which is a louder and more precise signal than a warning. Checked
        # via the resolved attribute because ``cls.__abstractmethods__`` is
        # not populated until after ``__init_subclass__`` returns.
        if getattr(getattr(cls, "_compare", None), "__isabstractmethod__", False):
            return

        # The class that actually supplies ``compare`` -- which may be a mixin
        # rather than ``cls`` itself. Keying on the owner rather than on
        # ``cls.__dict__`` catches ``class M(LegacyMixin, SomeComparator)``,
        # and keeps ordinary subclasses of a deliberate override quiet.
        owner = next(
            (k for k in cls.__mro__ if "compare" in k.__dict__), BaseComparator
        )
        if owner is BaseComparator or "_compare" in owner.__dict__:
            return

        source = (
            "defines compare()"
            if owner is cls
            else f"inherits compare() from {owner.__name__}"
        )
        warn_once(
            "comparator-compare-shadows-none-policy",
            cls.__qualname__,
            f"{cls.__name__} {source}, which shadows BaseComparator.compare() "
            f"and the shared None policy. Unless that compare() forwards both "
            f"values to super().compare() unchanged, None will reach it and can "
            f"be scored as a present value. Implement _compare() instead: it is "
            f"only called when both values are present.",
            category=UserWarning,
            # warn_once(1) -> __init_subclass__(2) -> ABCMeta.__new__(3) ->
            # the user's ``class`` statement(4). ABCMeta.__new__ is a Python
            # frame, so the default stacklevel=3 blames ``<frozen abc>``.
            stacklevel=4,
        )

    #: The threshold this comparator uses when the caller does not name one.
    #: Overridden per subclass. Read through ``self`` so a subclass's value
    #: wins, which is what lets ``__init__`` below resolve ``None`` without
    #: every subclass repeating its own default in two places.
    DEFAULT_THRESHOLD: float = 0.7

    def __init__(self, threshold: Optional[float] = None):
        """Initialize the comparator.

        ``threshold`` defaults to ``None`` rather than to a number so that a
        threshold the caller named stays distinguishable from one nobody
        asked for. The two are not interchangeable: a named threshold is a
        statement about the field being compared and is adopted as the
        verdict threshold, while a class default was only ever read by
        ``binary_compare()`` and has not been audited as a verdict threshold.
        ``DateComparator``'s default of ``1.0`` would clip its own
        ``allow_partial_year`` partial credit of ``0.7`` to zero, so adopting
        defaults is not safe.

        Recording it here is the only place that can. Each subclass resolves
        its own default before calling ``super().__init__``, so by the time a
        concrete number arrives, comparing it against the signature default
        cannot tell ``DateComparator()`` from ``DateComparator(threshold=1.0)``
        -- both hold ``1.0``. Passing ``None`` through preserves the
        distinction, and it survives a subclass that forwards ``**kwargs``,
        which signature inspection does not.

        Args:
            threshold: Similarity threshold (0.0-1.0), or None to use
                :attr:`DEFAULT_THRESHOLD`.
        """
        #: Whether the caller named a threshold. Consumed by
        #: ``stickler.structured_object_evaluator.models.comparable_field``.
        self.threshold_was_set = threshold is not None
        self.threshold = self.DEFAULT_THRESHOLD if threshold is None else threshold

    def compare(self, str1: Any, str2: Any) -> float:
        """Compare two values and return a similarity score.

        Applies the shared ``None`` policy -- two missing values are an
        exact match, a missing value against a present one is a non-match,
        regardless of what "present" means for a particular comparator --
        then delegates to :meth:`_compare`.

        Args:
            str1: First value
            str2: Second value

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if str1 is None and str2 is None:
            return 1.0

        if str1 is None or str2 is None:
            return 0.0

        return self._compare(str1, str2)

    @abstractmethod
    def _compare(self, str1: Any, str2: Any) -> float:
        """Compare two present values and return a similarity score.

        Implemented by each comparator. Both arguments are guaranteed not to
        be ``None`` -- :meth:`compare` handles that before delegating here.

        Args:
            str1: First value, never None
            str2: Second value, never None

        Returns:
            Similarity score between 0.0 and 1.0
        """
        pass

    def __call__(self, str1: Any, str2: Any) -> float:
        """Make the comparator callable.

        Args:
            str1: First value
            str2: Second value

        Returns:
            Similarity score between 0.0 and 1.0
        """
        return self.compare(str1, str2)

    def binary_compare(self, str1: Any, str2: Any) -> Tuple[int, int]:
        """Compare two values and return a binary result as (tp, fp) tuple.

        This method converts the continuous similarity score to a binary decision
        based on the threshold. If the similarity is greater than or equal to the
        threshold, it returns (1, 0) indicating true positive. Otherwise, it returns
        (0, 1) indicating false positive.

        Args:
            str1: First value
            str2: Second value

        Returns:
            Tuple of (tp, fp) where tp is 1 if similar, 0 otherwise,
            and fp is the opposite
        """
        score = self.compare(str1, str2)
        if score >= self.threshold:
            return (1, 0)  # True positive
        else:
            return (0, 1)  # False positive

    def __str__(self) -> str:
        """String representation for serialization."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(threshold={self.threshold})"
