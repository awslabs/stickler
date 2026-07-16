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

from stickler.structured_object_evaluator.models.comparator_registry import (
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
