"""Common comparators for key information evaluation.

This package contains comparators that are shared between the traditional
and ANLS Star evaluation systems. These comparators implement a unified
interface that works with both systems.
"""

import importlib as _il
import importlib.util as _ilu
import sys as _sys

from stickler.comparators.base import BaseComparator
from stickler.comparators.bbox import BBoxIoUComparator
from stickler.comparators.date import DateComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator, NumericExactC
from stickler.comparators.semantic import SemanticComparator
from stickler.comparators.structured import StructuredModelComparator
from stickler.comparators.utils import generate_bedrock_embedding

# LLMComparator and BERTComparator are probed rather than imported: importing
# them pulls strands-agents/boto3 and torch/transformers/datasets respectively,
# which would land on the `import stickler` path. See the module __getattr__
# below and the same pattern in stickler/__init__.py.

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
# gates __all__ and what tests/test_top_level_exports.py asserts. Note that the
# MODULE stickler.comparators.llm imports fine without strands (it degrades to
# STRANDS_AVAILABLE=False and raises at instantiation), so __getattr__ below
# does not gate on this flag for LLMComparator; only __all__ does.
LLM_AVAILABLE = _dependency_available("strands")
BERT_AVAILABLE = _dependency_available("evaluate")

_LAZY_COMPARATORS = {
    "LLMComparator": ("stickler.comparators.llm", "llm", True),
    "BERTComparator": ("stickler.comparators.bert", "bert", BERT_AVAILABLE),
}


def __getattr__(name: str):
    """Import the optional comparators on first access (PEP 562).

    Raises AttributeError (not ImportError) for a missing extra, so
    ``hasattr()`` stays False exactly as the previous eager gating provided.
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
        module = _il.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"{name} requires the '{extra}' extra, which is installed but "
            f"failed to import. Original error: {exc}"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))

# Import FuzzyComparator and Fuzz alias only if rapidfuzz is available
try:
    from stickler.comparators.fuzzy import (  # noqa: F401
        RAPIDFUZZ_AVAILABLE,
        Fuzz,
        FuzzyComparator,
    )
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


__all__ = [
    "BaseComparator",
    "BBoxIoUComparator",
    "LevenshteinComparator",
    "NumericComparator",
    "NumericExactC",
    "ExactComparator",
    "DateComparator",
    "StructuredModelComparator",
    "SemanticComparator",
    "generate_bedrock_embedding",
]

# Add LLMComparator to __all__ if available
if LLM_AVAILABLE:
    __all__.append("LLMComparator")

# Add BERTComparator to __all__ if available
if BERT_AVAILABLE:
    __all__.append("BERTComparator")

# Add FuzzyComparator and Fuzz to __all__ if available
if RAPIDFUZZ_AVAILABLE:
    __all__.append("FuzzyComparator")
    __all__.append("Fuzz")
