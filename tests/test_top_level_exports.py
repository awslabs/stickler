"""Smoke tests for top-level stickler package exports.

Verifies that re-exported names are the canonical classes from their
source modules, not accidental rebindings.
"""

import importlib
import sys

import pytest

import stickler

# Mapping of public name -> canonical source module
ALWAYS_AVAILABLE_EXPORTS = {
    "BaseComparator": "stickler.comparators.base",
    "ExactComparator": "stickler.comparators.exact",
    "FuzzyComparator": "stickler.comparators.fuzzy",
    "LevenshteinComparator": "stickler.comparators.levenshtein",
    "NumericComparator": "stickler.comparators.numeric",
    "SemanticComparator": "stickler.comparators.semantic",
    "StructuredModelComparator": "stickler.comparators.structured",
}

MODEL_AND_EVALUATION_EXPORTS = {
    "StructuredModel": "stickler.structured_object_evaluator",
    "ComparableField": "stickler.structured_object_evaluator",
    "NonMatchField": "stickler.structured_object_evaluator",
    "NonMatchType": "stickler.structured_object_evaluator",
    "compare_structured_models": "stickler.structured_object_evaluator",
    "anls_score": "stickler.structured_object_evaluator",
    "compare_json": "stickler.structured_object_evaluator",
    "aggregate_from_comparisons": "stickler.structured_object_evaluator",
}


class TestReexportIdentity:
    """Verify re-exports are identical to canonical source objects."""

    @pytest.mark.parametrize(
        "name,module", ALWAYS_AVAILABLE_EXPORTS.items()
    )
    def test_comparator_is_canonical_class(self, name, module):
        """Test that stickler.X is stickler.comparators.x.X."""
        canonical = getattr(importlib.import_module(module), name)
        exported = getattr(stickler, name)
        assert exported is canonical

    @pytest.mark.parametrize(
        "name,module", MODEL_AND_EVALUATION_EXPORTS.items()
    )
    def test_model_export_is_canonical(self, name, module):
        """Test that model/evaluation exports are canonical objects."""
        canonical = getattr(importlib.import_module(module), name)
        exported = getattr(stickler, name)
        assert exported is canonical


class TestAllConsistency:
    """Verify __all__ is consistent with actual module attributes."""

    def test_all_entries_are_importable(self):
        """Every name in __all__ must exist as an attribute."""
        for name in stickler.__all__:
            assert hasattr(stickler, name), (
                f"{name!r} is in __all__ but not an attribute of stickler"
            )

    def test_star_import_matches_all(self):
        """from stickler import * should yield exactly __all__."""
        ns = {}
        exec("from stickler import *", ns)  # noqa: S102
        exported_names = set(ns) - {"__builtins__"}
        assert exported_names == set(stickler.__all__)

    def test_always_available_in_all(self):
        """All always-available comparators must be in __all__."""
        for name in ALWAYS_AVAILABLE_EXPORTS:
            assert name in stickler.__all__

    def test_numeric_exact_c_not_in_all(self):
        """NumericExactC is a compat alias - not re-exported at top level."""
        assert "NumericExactC" not in stickler.__all__
        assert not hasattr(stickler, "NumericExactC")


class TestOptionalGating:
    """Verify optional comparators are gated correctly in __all__."""

    def test_llm_in_all_iff_strands_available(self):
        """LLMComparator in __all__ only when strands-agents is installed."""
        from stickler.comparators.llm import STRANDS_AVAILABLE

        if STRANDS_AVAILABLE:
            assert "LLMComparator" in stickler.__all__
        else:
            assert "LLMComparator" not in stickler.__all__

    def test_bert_in_all_iff_evaluate_available(self):
        """BERTComparator in __all__ only when evaluate is installed."""
        try:
            import evaluate  # noqa: F401

            assert "BERTComparator" in stickler.__all__
        except ModuleNotFoundError:
            assert "BERTComparator" not in stickler.__all__

    def test_llm_import_gated(self):
        """LLMComparator is only importable from stickler when strands is installed."""
        from stickler.comparators.llm import STRANDS_AVAILABLE

        if STRANDS_AVAILABLE:
            from stickler import LLMComparator
            from stickler.comparators.llm import LLMComparator as Canonical

            assert LLMComparator is Canonical
        else:
            assert not hasattr(stickler, "LLMComparator")

    def test_llm_not_exported_when_strands_missing(self, monkeypatch):
        """Simulate missing strands-agents and verify LLMComparator is excluded."""
        # Block strands from importing
        monkeypatch.setitem(sys.modules, "strands", None)
        monkeypatch.setitem(sys.modules, "strands.agent", None)

        # Clear cached modules so reload picks up the block
        modules_to_clear = [
            k for k in sys.modules if k.startswith("stickler")
        ]
        for mod in modules_to_clear:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        # Re-import with strands blocked
        import stickler as reloaded

        importlib.reload(reloaded)

        assert "LLMComparator" not in reloaded.__all__
        assert not hasattr(reloaded, "LLMComparator")
