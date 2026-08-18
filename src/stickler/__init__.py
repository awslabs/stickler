"""
stickler: Structured object comparison and evaluation library.

This library provides tools for comparing complex structured objects
with configurable comparison strategies and detailed evaluation metrics.
"""

import importlib as _il
import importlib.util as _ilu
import sys as _sys

# Zero-config evaluation of vanilla pydantic models (no StructuredModel
# subclass or schema required). See stickler/auto/README.md.
from .auto import EvalResult, EvalSpec, eval_for, evaluate

# Always-available comparators (core deps)
from .comparators.base import BaseComparator
from .comparators.bbox import BBoxIoUComparator
from .comparators.date import DateComparator
from .comparators.exact import ExactComparator
from .comparators.fuzzy import FuzzyComparator
from .comparators.levenshtein import LevenshteinComparator
from .comparators.numeric import NumericComparator
from .comparators.phone import PhoneComparator
from .comparators.semantic import SemanticComparator
from .comparators.structured import StructuredModelComparator
from .structured_object_evaluator import (
    ComparableField,
    NonMatchField,
    NonMatchType,
    StructuredModel,
    aggregate_from_comparisons,
    anls_score,
    compare_json,
    compare_structured_models,
)

# Optional comparators are PROBED, not imported.
#
# Importing them here would put their dependencies on the `import stickler`
# path: strands-agents and boto3 for LLMComparator, and torch, transformers,
# datasets, and pandas for BERTComparator. That cost lands on every caller who
# installed an extra, even one who never touches the comparator, and it is
# seconds rather than milliseconds.
#
# `find_spec` answers "is the dependency installed?" without executing the
# module, so `__all__` stays accurate while the import is deferred to first
# attribute access via `__getattr__` below.

def _dependency_available(name: str) -> bool:
    """Whether an optional dependency can be imported, without importing it.

    Checks ``sys.modules`` before the filesystem so a test that injects a mock
    (see tests/common/comparators/conftest.py) is treated as available, which
    ``importlib.util.find_spec`` alone would not do.
    """
    module = _sys.modules.get(name, False)
    if module is not None and module is not False:
        # Present in sys.modules, including a test-injected mock.
        return True
    if module is None:
        # Explicitly blocked (sys.modules[name] = None), which is how tests
        # simulate a missing dependency.
        return False
    try:
        return _ilu.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# `_HAS_LLM`/`LLM_AVAILABLE` mean "strands-agents is installed", which is what
# gates both __getattr__ below and __all__, and what
# tests/test_top_level_exports.py asserts. The flag rather than a trial import
# is the gate because the MODULE stickler.comparators.llm imports fine without
# strands: it degrades to STRANDS_AVAILABLE=False and raises at instantiation,
# so a successful import would prove nothing about availability.
_HAS_LLM = _dependency_available("strands")
_HAS_BERT = _dependency_available("evaluate")

# Optional comparator name -> (module, owning extra, whether it is available).
# Only available entries are resolvable, so `hasattr(stickler, "LLMComparator")`
# stays False when the extra is not installed, matching the behavior of the
# previous eager try/except gating.
_LAZY_COMPARATORS = {
    "LLMComparator": (".comparators.llm", "llm", _HAS_LLM),
    "BERTComparator": (".comparators.bert", "bert", _HAS_BERT),
}


def __getattr__(name: str):
    """Import optional comparators on first access (PEP 562).

    `stickler.LLMComparator` and `from stickler import BERTComparator` both
    route here, so the heavy import happens when the comparator is actually
    used rather than at package import.

    Raises:
        AttributeError: If the name is unknown, or if it is an optional
            comparator whose extra is not installed. Raising AttributeError
            rather than ImportError keeps `hasattr()` False for a missing
            extra, which is what the eager gating used to provide.
    """
    entry = _LAZY_COMPARATORS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, extra, available = entry
    if not available:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}: it requires the "
            f"'{extra}' extra. Install it with: "
            f'pip install "stickler-eval[{extra}]"'
        )

    try:
        # `module_path` is a literal from `_LAZY_COMPARATORS`, not the caller's
        # `name`: an unrecognized `name` raises AttributeError above and never
        # reaches here. Semgrep matches a non-literal first argument and does not
        # follow the dict lookup.
        # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
        module = _il.import_module(module_path, __name__)
    except ImportError as exc:
        # Installed but broken (a version-skewed transitive dependency raising
        # plain ImportError). Surface it here, at the point of use, rather than
        # taking down `import stickler`.
        raise ImportError(
            f"{name} requires the '{extra}' extra, which is installed but "
            f"failed to import. Original error: {exc}"
        ) from exc
    value = getattr(module, name)
    # Cache on the module so subsequent lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list:
    """Include lazily-available comparators in `dir(stickler)`."""
    return sorted(set(globals()) | set(__all__))

__version__ = "0.7.0"
__all__ = [
    # Models and evaluation
    "StructuredModel",
    "ComparableField",
    "NonMatchField",
    "NonMatchType",
    "compare_structured_models",
    "anls_score",
    "compare_json",
    "aggregate_from_comparisons",
    # Zero-config evaluation (vanilla pydantic -> scored eval)
    "evaluate",
    "eval_for",
    "EvalResult",
    "EvalSpec",
    # Comparators (always available)
    "BaseComparator",
    "ExactComparator",
    "PhoneComparator",
    "DateComparator",
    "BBoxIoUComparator",
    "FuzzyComparator",
    "LevenshteinComparator",
    "NumericComparator",
    "SemanticComparator",
    "StructuredModelComparator",
]

# Conditionally add optional comparators to __all__
if _HAS_LLM:
    __all__.append("LLMComparator")

if _HAS_BERT:
    __all__.append("BERTComparator")
