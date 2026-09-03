"""The shared ``None`` policy, exercised across every built-in comparator.

Regression coverage for #200 (inconsistent ``None`` handling across
comparators). ``BaseComparator.compare`` is a template method holding the
single shared check -- two ``None`` values are an exact match (1.0), one
``None`` against a present value is a non-match (0.0) -- and delegates to
each comparator's ``_compare`` only for values that are present. No
comparator implements the policy itself, so the truth table is asserted
once here across all of them rather than duplicated per comparator module.

Kept in one file because it maps 1:1 to the issue's acceptance criteria:
the truth table, the originally-reported Levenshtein bug, and the template
method contract that makes the policy impossible to bypass.
"""

from unittest.mock import MagicMock, patch

import pytest

from stickler.comparators import (
    BERT_AVAILABLE,
    RAPIDFUZZ_AVAILABLE,
    BaseComparator,
    BBoxIoUComparator,
    DateComparator,
    ExactComparator,
    LevenshteinComparator,
    NumericComparator,
    SemanticComparator,
    StructuredModelComparator,
)
from stickler.comparators.fuzzy import FuzzyComparator

# Imported from its own module, not from `stickler.comparators`, because the
# package namespace only exposes the name when the `llm` extra is installed
# (#259). The module itself always imports, degrading to
# STRANDS_AVAILABLE=False and raising at instantiation -- and instantiation
# here goes through a mocked strands (see the fixture below), so the None
# policy is testable without the extra.
from stickler.comparators.llm import LLMComparator

# Comparators needing more than a bare constructor get a factory. Everything
# else is just the class, which is already a zero-argument callable.


def _semantic() -> SemanticComparator:
    """Pin a dummy model id so no real embedding model is resolved."""
    return SemanticComparator(model_id="test-model")


def _llm() -> LLMComparator:
    """Build an LLMComparator without a real strands-agents install.

    ``STRANDS_AVAILABLE`` is evaluated once when ``stickler.comparators.llm``
    is first imported, so patching it here (rather than relying on the
    ``sys.modules`` mock in ``conftest.py``) keeps construction working
    regardless of import order. Only ``__init__`` needs to survive: the
    ``None`` policy short-circuits in ``compare`` before the agent is used.
    """
    with (
        patch("stickler.comparators.llm.STRANDS_AVAILABLE", True),
        patch("stickler.comparators.llm.Agent", MagicMock()),
    ):
        return LLMComparator(model="test-model")


def _bert():
    """Import lazily -- ``comparators.bert`` imports ``evaluate`` at module level."""
    from stickler.comparators.bert import BERTComparator

    return BERTComparator()


# Comparators behind an optional extra carry a skip mark instead of being
# filtered out, so an uninstalled extra shows up as SKIPPED in the report
# rather than silently disappearing from the sweep.
BUILTIN_COMPARATORS = [
    pytest.param(ExactComparator, id="Exact"),
    pytest.param(LevenshteinComparator, id="Levenshtein"),
    pytest.param(NumericComparator, id="Numeric"),
    pytest.param(DateComparator, id="Date"),
    pytest.param(BBoxIoUComparator, id="BBox"),
    pytest.param(StructuredModelComparator, id="Structured"),
    pytest.param(_semantic, id="Semantic"),
    pytest.param(_llm, id="LLM"),
    pytest.param(
        FuzzyComparator,
        id="Fuzzy",
        marks=pytest.mark.skipif(
            not RAPIDFUZZ_AVAILABLE, reason="rapidfuzz is not installed"
        ),
    ),
    pytest.param(
        _bert,
        id="BERT",
        marks=pytest.mark.skipif(
            not BERT_AVAILABLE, reason="the 'bert' extra is not installed"
        ),
    ),
]


@pytest.fixture
def comparator(request):
    """The comparator under test, built from its parametrized factory.

    Constructed per test rather than at import time, so a constructor
    failure fails that comparator's tests instead of breaking collection
    for the whole module.
    """
    return request.param()


@pytest.mark.parametrize("comparator", BUILTIN_COMPARATORS, indirect=True)
class TestNonePolicy:
    """The full truth table, asserted identically for every comparator.

    ``""`` is deliberately included: ``None`` is a *missing* value and ``""``
    is a *present but empty* one, and collapsing the two is the exact bug
    #200 was filed for. Both argument orderings are covered so the policy
    can't hold in one direction only.
    """

    def test_both_none_is_a_match(self, comparator):
        assert comparator.compare(None, None) == 1.0

    def test_none_against_a_value_is_not_a_match(self, comparator):
        assert comparator.compare(None, "some-value") == 0.0

    def test_value_against_none_is_not_a_match(self, comparator):
        assert comparator.compare("some-value", None) == 0.0

    def test_none_against_empty_string_is_not_a_match(self, comparator):
        assert comparator.compare(None, "") == 0.0

    def test_empty_string_against_none_is_not_a_match(self, comparator):
        assert comparator.compare("", None) == 0.0

    def test_does_not_override_the_template_method(self, comparator):
        """Overriding ``compare`` would bypass the policy entirely.

        Comparators extend ``_compare``; ``compare`` must stay the base
        class's template method for the guarantees above to hold.
        """
        assert type(comparator).compare is BaseComparator.compare


def test_levenshtein_none_vs_empty_string_is_not_a_match():
    """Named regression for the originally-reported #200 bug.

    The truth table above covers ``(None, "")`` for every comparator; this
    pins the specific case that was reported -- Levenshtein coerced ``None``
    to ``""`` before any check, so ``(None, "")`` fell into the "both empty"
    branch and scored 1.0 -- and guards against over-correcting, since a
    genuine empty-vs-empty comparison must still be a match.
    """
    comparator = LevenshteinComparator()

    assert comparator.compare(None, "") == 0.0
    assert comparator.compare("", None) == 0.0
    assert comparator.compare("", "") == 1.0


class TestTemplateMethodContract:
    """What makes the policy inheritable, and hard to bypass by accident."""

    def test_subclass_inherits_policy_without_implementing_anything(self):
        """A new comparator gets the policy for free -- the point of #200."""

        class Minimal(BaseComparator):
            def _compare(self, str1, str2):
                return 0.5

        comparator = Minimal()

        assert comparator.compare(None, None) == 1.0
        assert comparator.compare(None, "x") == 0.0
        assert comparator.compare("x", "y") == 0.5

    def test_subclass_that_does_not_override_compare_inherits_parents(self):
        class NoOverride(LevenshteinComparator):
            pass

        comparator = NoOverride()

        assert comparator.compare(None, None) == 1.0
        assert comparator.compare(None, "x") == 0.0

    def test_two_level_override_chain_applies_the_policy_once(self):
        """Mirrors the real ``CaseInsensitiveComparator(LevenshteinComparator)``
        pattern used elsewhere in the suite: a subclass overrides ``_compare``
        and delegates to ``super()._compare()``. The policy is applied once,
        by ``BaseComparator.compare``, before either level runs.
        """

        class CaseInsensitive(LevenshteinComparator):
            def _compare(self, a, b):
                return super()._compare(str(a).lower(), str(b).lower())

        comparator = CaseInsensitive()

        assert comparator.compare(None, None) == 1.0
        assert comparator.compare(None, "x") == 0.0
        assert comparator.compare("HELLO", "hello") == 1.0

    def test_migrated_subclass_gets_the_policy(self):
        """None policy is enforced by BaseComparator.compare even when a
        subclass overrides _compare -- _compare never receives None."""

        class Migrated(LevenshteinComparator):
            def _compare(self, str1, str2):
                return super()._compare(str(str1).lower(), str(str2).lower())

        comparator = Migrated()

        assert comparator.compare(None, None) == 1.0
        assert comparator.compare(None, "") == 0.0
        assert comparator.compare("HELLO", "hello") == 1.0


class TestPreRenameSubclassRejected:
    """The deprecation shim was removed in 1.0 (issue #215).

    Two different signals replaced it, and which one fires depends on whether
    the subclass ends up constructible:

    * No ``_compare`` anywhere -- extending ``BaseComparator`` directly, or via
      a mixin over it -- leaves the abstract slot unfilled, so ``ABCMeta``
      raises ``TypeError`` at construction naming ``_compare``. That is the
      hard break, and it fires without a warning: the exception is already the
      louder and more precise signal.
    * ``_compare`` inherited from a *concrete* comparator makes the class
      construct fine while its ``compare()`` shadows the template method and
      skips the ``None`` policy. Nothing else flags that, so
      ``__init_subclass__`` warns at class-definition time.

    The second signal is a ``UserWarning`` rather than a ``TypeError`` because
    the condition is not decidable at class-definition time: an override that
    forwards both values to ``super().compare()`` unchanged keeps the policy
    intact and is correct code, and nothing at class creation can tell it apart
    from one that mangles the values first. Refusing the class would reject the
    correct shape along with the broken one, so both warn and neither is
    blocked. Defining ``_compare`` opts out.

    So the MRO shadowing itself is unavoidable, but going *undetected* is not:
    a deliberate ``compare()`` override is left as the user's choice, and the
    warning is what makes it a choice rather than an accident.
    """

    def test_subclassing_basecomparator_with_only_compare_raises(self):
        """Extending ``BaseComparator`` directly with only ``compare()`` leaves
        ``_compare`` abstract, so it raises ``TypeError`` at construction."""

        class PreRename(BaseComparator):
            def compare(self, str1, str2):  # pre-rename interface
                if str1 is None or str2 is None:
                    return 0.0
                return 1.0 if str(str1) == str(str2) else 0.0

        with pytest.raises(TypeError, match="_compare"):
            PreRename()

    def test_pre_rename_subclass_of_concrete_comparator_warns_and_constructs(self):
        """A pre-rename subclass of a *concrete* comparator constructs without
        error because it inherits ``_compare`` from the parent -- and warns at
        class definition, because that is the shape with no other signal."""

        with pytest.warns(UserWarning, match="shadows BaseComparator.compare"):

            class PreRenameCaseInsensitive(LevenshteinComparator):
                def compare(self, str1, str2):
                    return super().compare(str(str1).lower(), str(str2).lower())

        comparator = PreRenameCaseInsensitive()

        assert comparator.compare("HELLO", "hello") == 1.0
        assert comparator.compare("kitten", "sitting") == pytest.approx(
            0.5714, abs=1e-4
        )

    def test_pre_rename_subclass_via_mixin_is_abstract(self):
        """A class inheriting ``compare()`` from a non-BaseComparator mixin
        and extending ``BaseComparator`` directly has no ``_compare``
        implementation, so it is abstract and raises ``TypeError``."""

        class LegacyMixin:
            def compare(self, str1, str2):
                return 1.0 if str1 == str2 else 0.0

        class ViaMixin(LegacyMixin, BaseComparator):
            pass

        with pytest.raises(TypeError, match="_compare"):
            ViaMixin()

    def test_pre_rename_subclass_of_concrete_bypasses_none_policy(self):
        """A pre-rename subclass of a concrete comparator still constructs,
        and its ``compare()`` override bypasses the template method and the
        shared ``None`` policy because it does not delegate to
        ``super().compare()``.

        This is the residual risk the shim used to carry for *all* pre-rename
        subclasses.  After removing the shim it survives only for subclasses of
        concrete comparators, and it is no longer silent: the warning asserted
        here is the signal that replaced the shim's ``DeprecationWarning``.
        Removing the shim does not fix the bypass -- nothing can, it is plain
        MRO -- but a deliberate ``compare()`` override is now the user's
        informed choice rather than an accident.
        """

        with pytest.warns(UserWarning, match="_compare"):

            class Legacy(LevenshteinComparator):
                def compare(self, a, b):  # pre-rename interface
                    return 0.42

        comparator = Legacy()

        assert comparator.compare(None, "") == 0.42  # policy bypassed
        assert type(comparator).compare is not BaseComparator.compare

    def test_the_old_none_coercion_is_gone_from_the_algorithm(self):
        """The pre-fix `(None, "") -> 1.0` result cannot be inherited.

        The coercion was deleted from Levenshtein's ``_compare``, not merely
        guarded, so a pre-rename subclass that delegates upward via
        ``super().compare()`` gets the corrected answer through the template
        method's ``None`` policy.

        This shape is *correct* -- it forwards both values unchanged, so the
        policy runs -- and it still warns, because class creation cannot see
        what an override does with its arguments. That over-warning is the
        deliberate price of not hard-failing: a false alarm on correct code
        costs a suppressible message, whereas a ``TypeError`` here would make
        the shape impossible to write at all.
        """
        assert LevenshteinComparator()._compare(None, "") == 0.0

        with pytest.warns(UserWarning, match="_compare"):

            class LegacyDelegating(LevenshteinComparator):
                def compare(self, a, b):
                    return super().compare(a, b)

        assert LegacyDelegating().compare(None, "") == 0.0
