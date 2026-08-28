"""Base class for comparators."""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from stickler.utils.deprecation import warn_once


class BaseComparator(ABC):
    """Base class for all comparators.

    This class defines the interface that all comparators must implement.
    Comparators are used to compare two values and return a similarity score
    between 0.0 and 1.0, where 1.0 means the values are identical.
    """

    def __init__(self, threshold: float = 0.7):
        """Initialize the comparator.

        Args:
            threshold: Similarity threshold (0.0-1.0)
        """
        self.threshold = threshold

    #: Whether this comparator can score a mapping. False for every comparator
    #: that treats its inputs as scalars, which is all of them except ANLS*.
    #: The dispatcher reads this to decide whether handing a dict to the
    #: configured comparator is meaningful, rather than discovering it via a
    #: TypeError partway through a corpus.
    handles_mappings: bool = False

    def __init_subclass__(cls, **kwargs):
        """Keep comparators written against the pre-rename interface working.

        Before the shared ``None`` policy existed, comparators implemented
        ``compare`` directly. Such a subclass is left exactly as written: its
        ``compare`` still overrides the template method, so its behavior is
        unchanged and it does not get the policy. All this does is fill the
        ``_compare`` slot when nothing in the MRO provides one, so the class
        is not abstract, and warn that the rename is needed.

        Deliberately does not rewire ``compare`` to ``_compare``. Doing that
        breaks any subclass whose body calls ``super().compare(...)``: the
        parent would no longer have a real ``compare``, so the call lands on
        this template, which dispatches straight back to the function that
        made it.

        Temporary. Removed in 1.0 (see issue #215), after which an
        un-migrated comparator becomes abstract again.
        """
        super().__init_subclass__(**kwargs)

        if "_compare" in cls.__dict__:
            return

        legacy_compare = cls.__dict__.get("compare")
        if legacy_compare is None:
            # Also catch a compare() reached through a non-BaseComparator
            # mixin: still the old interface, just not defined on this class.
            candidate = getattr(cls, "compare", None)
            if candidate is None or candidate is BaseComparator.compare:
                return
            legacy_compare = candidate

        if getattr(legacy_compare, "__isabstractmethod__", False):
            return

        inherited = getattr(cls, "_compare", None)
        if inherited is None or getattr(inherited, "__isabstractmethod__", False):
            # Nothing in the MRO implements _compare, so the class would be
            # abstract. Their compare() overrides the template and is what
            # actually runs, so this stub is only reached by an explicit
            # super().compare() from a direct BaseComparator subclass.
            def _unmigrated(self, str1: Any, str2: Any) -> float:
                raise NotImplementedError(
                    f"{type(self).__name__} implements compare() but not "
                    f"_compare(); rename compare() to _compare()."
                )

            cls._compare = _unmigrated

        warn_once(
            "comparator-compare-rename",
            cls.__qualname__,
            f"{cls.__name__} implements compare(), which is no longer the "
            f"extension point and bypasses the shared None policy. Rename "
            f"compare() to _compare() and delete any None handling it "
            f"contains, since _compare() never receives None. Support for "
            f"implementing compare() directly will be removed in 1.0.",
            stacklevel=4,
        )

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
