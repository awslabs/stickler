"""Registry-naming tests for the comparator registry.

Guards against regressions where a comparator is registered under a name
that does not match its canonical class name. A mismatch makes the import
inside ``_register_builtin_comparators`` raise ``ImportError``, which is
silently swallowed by the surrounding ``try/except ImportError: pass`` -- so
the comparator never registers and ``create_comparator`` fails with
``KeyError`` even when the optional extra is installed.

The BERT comparator is the canary here: it lives behind the optional ``bert``
extra (torch, bert-score, evaluate), so its registration is gated. The test
covers both paths -- BERT available and BERT not available -- so it locks in
the canonical name without requiring the heavy extra in the default test env.
"""

from __future__ import annotations

import importlib

import pytest

from stickler.comparators.base import BaseComparator
from stickler.structured_object_evaluator.models.comparator_registry import (
    ComparatorRegistry,
    get_comparator_class,
    get_global_registry,
)

# Detect whether the optional BERT extra is installed. Prefer the package-level
# availability flag if present; otherwise fall back to attempting the import.
try:
    from stickler.comparators import BERT_AVAILABLE
except ImportError:  # pragma: no cover - defensive, flag should always exist
    try:
        from stickler.comparators.bert import BERTComparator  # noqa: F401

        BERT_AVAILABLE = True
    except ImportError:
        BERT_AVAILABLE = False


class TestBERTRegistryNaming:
    """BERT registers under its canonical class name -- or not at all."""

    def test_old_misspelled_key_never_registered(self):
        """The historical broken key must never be present in the registry."""
        assert not get_global_registry().is_registered("BertComparator")

    def test_bert_registered_under_canonical_name(self):
        """When the bert extra is installed, BERT is registered as ``BERTComparator``.

        When it is not installed, neither the canonical nor the old key is
        present -- the try/except correctly skipped it.
        """
        registry = get_global_registry()

        if BERT_AVAILABLE:
            from stickler.comparators.bert import BERTComparator

            assert registry.is_registered("BERTComparator")
            assert "BERTComparator" in registry.list_available()
            assert get_comparator_class("BERTComparator") is BERTComparator
        else:
            assert not registry.is_registered("BERTComparator")
            assert not registry.is_registered("BertComparator")


class TestAFailedImportLeavesTheBuiltinRegistered:
    """A failed lookup must not deregister a built-in.

    ``_resolve`` used to pop the pending entry *before* attempting the import,
    so an ``ImportError`` returned None with the entry already gone. A
    broken-but-installed extra -- a version-skewed transitive dependency
    raising plain ImportError, which is exactly the case the method docstring
    says it handles -- was deregistered permanently by the first failed lookup.

    Nothing about a failed import means the name stopped being a built-in, so
    these pin that the registry's reported state does not depend on prior call
    history.

    See https://github.com/awslabs/stickler/issues/260
    """

    # A built-in with no optional extra, so the failure under test comes from
    # the patched import rather than from a missing dependency.
    NAME = "LevenshteinComparator"

    @pytest.fixture
    def registry_with_a_broken_builtin(self, monkeypatch):
        """A fresh registry whose NAME built-in cannot be imported."""
        registry = ComparatorRegistry()
        module_path = ComparatorRegistry._BUILTINS[self.NAME][0]

        def explode(name, *args, **kwargs):
            if name == module_path:
                raise ImportError(f"simulated broken extra: {name}")
            return importlib.import_module(name)

        monkeypatch.setattr(
            "stickler.structured_object_evaluator.models."
            "comparator_registry.importlib.import_module",
            explode,
        )

        # Guard the guard: the patch must actually break this lookup, or every
        # assertion below is vacuous.
        with pytest.raises(KeyError):
            registry.get(self.NAME)

        return registry

    def test_is_registered_is_unchanged_by_a_failed_get(self, monkeypatch):
        registry = ComparatorRegistry()
        module_path = ComparatorRegistry._BUILTINS[self.NAME][0]

        before = registry.is_registered(self.NAME)
        assert before is True

        monkeypatch.setattr(
            "stickler.structured_object_evaluator.models."
            "comparator_registry.importlib.import_module",
            lambda name, *a, **k: (_ for _ in ()).throw(ImportError(name)),
        )
        with pytest.raises(KeyError):
            registry.get(self.NAME)

        assert registry.is_registered(self.NAME) == before, (
            f"{self.NAME} was deregistered by a failed lookup"
        )
        assert module_path  # the name really does have a module to import

    def test_list_available_is_unchanged_by_a_failed_get(self, monkeypatch):
        registry = ComparatorRegistry()
        before = sorted(registry.list_available())

        monkeypatch.setattr(
            "stickler.structured_object_evaluator.models."
            "comparator_registry.importlib.import_module",
            lambda name, *a, **k: (_ for _ in ()).throw(ImportError(name)),
        )
        with pytest.raises(KeyError):
            registry.get(self.NAME)

        assert sorted(registry.list_available()) == before, (
            "list_available() shrank as a side effect of a failed lookup"
        )

    def test_two_failed_gets_behave_identically(self, registry_with_a_broken_builtin):
        """The second call must not take a different path from the first."""
        registry = registry_with_a_broken_builtin

        with pytest.raises(KeyError) as first:
            registry.get(self.NAME)
        with pytest.raises(KeyError) as second:
            registry.get(self.NAME)

        assert str(first.value) == str(second.value)

    def test_a_broken_extra_raises_keyerror_not_importerror(
        self, registry_with_a_broken_builtin
    ):
        """The unavailable-not-propagated contract, pinned alongside the fix.

        Only the bookkeeping changed; a broken extra is still reported as an
        absent comparator rather than surfacing the raw ImportError.
        """
        with pytest.raises(KeyError):
            registry_with_a_broken_builtin.get(self.NAME)

    def test_the_name_stays_reserved_against_registration(
        self, registry_with_a_broken_builtin
    ):
        """The quietest consequence: a failed resolve freed the name.

        ``register`` rejects a name only when it is in ``_registry`` or
        ``_pending``, so popping the entry early let a caller silently shadow a
        built-in.
        """

        class Custom(BaseComparator):
            def _compare(self, a, b):
                return 1.0

        with pytest.raises(ValueError, match="already registered"):
            registry_with_a_broken_builtin.register(self.NAME, Custom)


class TestASuccessfulResolveStillCachesOnce:
    """The fix must not break the path that was working."""

    NAME = "ExactComparator"

    def test_the_entry_moves_from_pending_to_registry(self):
        registry = ComparatorRegistry()

        assert self.NAME in registry._pending
        assert self.NAME not in registry._registry

        resolved = registry.get(self.NAME)

        assert self.NAME not in registry._pending
        assert registry._registry[self.NAME] is resolved

    def test_a_second_get_returns_the_cached_class(self):
        registry = ComparatorRegistry()

        assert registry.get(self.NAME) is registry.get(self.NAME)

    def test_the_name_is_listed_exactly_once_after_resolving(self):
        """`list_available` concatenates `_registry` and `_pending` keys.

        Reading without popping would list a resolved comparator twice.
        """
        registry = ComparatorRegistry()
        registry.get(self.NAME)

        assert registry.list_available().count(self.NAME) == 1

    def test_is_registered_holds_before_and_after(self):
        registry = ComparatorRegistry()

        assert registry.is_registered(self.NAME)
        registry.get(self.NAME)
        assert registry.is_registered(self.NAME)
