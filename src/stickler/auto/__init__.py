"""Zero-config evaluation of vanilla pydantic models.

Turn any ``pydantic.BaseModel`` (e.g. a Strands agent ``response_model``) into a
scored stickler evaluation with a single call — no ``StructuredModel`` subclass,
no JSON schema, no per-field configuration:

    >>> import stickler
    >>> result = stickler.evaluate(ground_truth, prediction)
    >>> result.f1, result.field_scores

The comparison config (comparator, threshold, weight per field) is inferred
from each field's python type and name. See ``auto/README.md`` for the
inference rules and precedence.
"""

from .facade import EvalResult, EvalSpec, eval_for, evaluate
from .inference import InferredSpec, infer_field_config

__all__ = [
    "evaluate",
    "eval_for",
    "EvalResult",
    "EvalSpec",
    "InferredSpec",
    "infer_field_config",
]
