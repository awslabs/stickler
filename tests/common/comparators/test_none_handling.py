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
    LLMComparator,
    NumericComparator,
    SemanticComparator,
    StructuredModelComparator,
)
from stickler.comparators.fuzzy import FuzzyComparator

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

    def test_subclassing_basecomparator_with_only_compare_is_abstract(self):
        """Extending ``BaseComparator`` directly and implementing only
        ``compare()`` fails at construction -- ``_compare`` is unimplemented,
        so the class stays abstract and the mistake surfaces immediately."""

        class WrongMethod(BaseComparator):
            def compare(self, str1, str2):  # not the extension point
                return 1.0

        with pytest.raises(TypeError, match="_compare"):
            WrongMethod()


class TestPreRenameSubclassMigration:
    """How subclasses written against the old ``compare()`` interface behave.

    The rename is a breaking change, and it does not fail loudly in every
    shape. These tests pin what actually happens per shape so the migration
    surface is documented and testable rather than asserted in a changelog.
    """

    def test_subclassing_a_concrete_comparator_does_not_fail_loudly(self):
        """The one shape that survives construction: ``_compare`` is
        inherited from the concrete parent, so the class is not abstract and
        the legacy ``compare()`` shadows the template, bypassing the policy.

        This is the case that needs the changelog to say "rename", because
        nothing here raises.
        """

        class Legacy(LevenshteinComparator):
            def compare(self, a, b):  # pre-rename interface
                return 0.42

        comparator = Legacy()  # constructs -- no TypeError

        assert comparator.compare(None, "") == 0.42  # policy bypassed
        assert type(comparator).compare is not BaseComparator.compare

    def test_the_old_none_coercion_is_gone_from_the_algorithm(self):
        """The pre-fix `(None, "") -> 1.0` result cannot be inherited.

        The coercion was deleted from Levenshtein's algorithm, not merely
        guarded, so a legacy subclass that delegates upward gets the
        corrected answer. Only a subclass that reimplemented the coercion in
        its own ``compare()`` can still produce the old score.
        """
        assert LevenshteinComparator()._compare(None, "") == 0.0

        class LegacyDelegating(LevenshteinComparator):
            def compare(self, a, b):
                return super().compare(a, b)

        assert LegacyDelegating().compare(None, "") == 0.0
