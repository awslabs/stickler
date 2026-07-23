"""Public zero-config evaluation surface.

The dead-simple entry point for evaluating structured output from a Strands
agent (or any pydantic-producing system):

    >>> import stickler
    >>> pred = agent.structured_output(Invoice, "Extract the invoice: ...")
    >>> result = stickler.evaluate(ground_truth, pred)
    >>> print(result.f1, result.recall, result.field_scores)

No ``StructuredModel`` subclass, no JSON schema, no ``x-aws-stickler-*``
annotations. Both arguments are ordinary pydantic instances; the comparison
config is inferred from their class (see :mod:`.inference`).

For a batch loop, compile once with :func:`eval_for` and reuse the returned
:class:`EvalSpec`.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import BaseModel

from .builder import specs_for, structured_model_for


class EvalResult:
    """Flat, friendly view over a stickler comparison result.

    Wraps the nested dict returned by ``StructuredModel.compare_with`` and
    exposes the metrics users actually reach for. The full raw dict is always
    available via :attr:`raw`.
    """

    def __init__(self, raw: Dict[str, Any], spec: "EvalSpec"):
        self.raw = raw
        self._spec = spec
        cm = raw.get("confusion_matrix", {}) or {}
        derived = (cm.get("overall", {}) or {}).get("derived", {}) or {}
        self.overall_score: float = raw.get("overall_score", 0.0)
        self.field_scores: Dict[str, float] = raw.get("field_scores", {})
        self.precision: float = derived.get("cm_precision", 0.0)
        self.recall: float = derived.get("cm_recall", 0.0)
        self.f1: float = derived.get("cm_f1", 0.0)
        self.accuracy: float = derived.get("cm_accuracy", 0.0)
        self.confusion_matrix: Dict[str, Any] = cm

    def explain(self) -> Dict[str, Dict[str, Any]]:
        """Per-field inferred config + provenance (see :meth:`EvalSpec.explain`)."""
        return self._spec.explain()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"EvalResult(overall_score={self.overall_score:.3f}, "
            f"precision={self.precision:.3f}, recall={self.recall:.3f}, "
            f"f1={self.f1:.3f})"
        )


class EvalSpec:
    """A compiled, reusable evaluator for one pydantic class.

    Build once with :func:`eval_for`, then call :meth:`evaluate` per pair. The
    inferred shadow ``StructuredModel`` is cached, so this is the efficient path
    for evaluating a dataset.
    """

    def __init__(
        self,
        source_cls: Type[BaseModel],
        eval_model: Type,
        *,
        weight_hints: bool,
    ):
        self.source_cls = source_cls
        self.eval_model = eval_model
        self._weight_hints = weight_hints

    def evaluate(self, ground_truth: BaseModel, prediction: BaseModel) -> EvalResult:
        """Score a single ground-truth / prediction pair."""
        gt = self.eval_model.from_json(_dump(ground_truth))
        pred = self.eval_model.from_json(_dump(prediction))
        raw = gt.compare_with(
            pred, include_confusion_matrix=True, add_derived_metrics=True
        )
        return EvalResult(raw, self)

    def explain(self) -> Dict[str, Dict[str, Any]]:
        """Return ``{field: {comparator, threshold, weight, source, why}}``.

        Makes every inferred choice auditable. ``why`` is the ordered provenance
        trail; ``source`` is a coarse label (``type`` / ``name-token`` /
        ``degrade``).
        """
        out: Dict[str, Dict[str, Any]] = {}
        for name, spec in specs_for(
            self.source_cls,
            weight_hints=self._weight_hints,
        ).items():
            out[name] = {
                "comparator": spec.comparator_name,
                "threshold": spec.threshold,
                "weight": spec.weight,
                "clip_under_threshold": spec.clip_under_threshold,
                "source": spec.source,
                "why": spec.provenance,
            }
        return out


def eval_for(
    cls: Type[BaseModel],
    *,
    weight_hints: bool = False,
    match_threshold: float = 0.7,
) -> EvalSpec:
    """Compile a reusable :class:`EvalSpec` for a pydantic class.

    Args:
        cls: The pydantic ``BaseModel`` subclass to evaluate instances of.
        weight_hints: Apply name-token weight heuristics (default off, so
            weights stay uniform and precision/recall are not skewed by guessed
            business-criticality).
        match_threshold: Overall match threshold for the generated model.
    """
    eval_model = structured_model_for(
        cls,
        weight_hints=weight_hints,
        match_threshold=match_threshold,
    )
    return EvalSpec(cls, eval_model, weight_hints=weight_hints)


def evaluate(
    ground_truth: BaseModel,
    prediction: BaseModel,
    *,
    weight_hints: bool = False,
    match_threshold: float = 0.7,
) -> EvalResult:
    """Evaluate a prediction against ground truth with zero configuration.

    ``ground_truth`` and ``prediction`` must be pydantic instances of the same
    class (or a compatible superset, so extra/missing fields are tolerated). The
    comparison config is inferred from their class.

    Args:
        ground_truth: The reference instance.
        prediction: The instance to score (e.g. a Strands ``response_model``).
        weight_hints: Enable name-token weight heuristics (default off).
        match_threshold: Overall match threshold.

    Returns:
        An :class:`EvalResult` with ``overall_score``, ``precision``,
        ``recall``, ``f1``, ``accuracy``, ``field_scores`` and ``.explain()``.
    """
    cls = _shared_class(ground_truth, prediction)
    spec = eval_for(
        cls,
        weight_hints=weight_hints,
        match_threshold=match_threshold,
    )
    return spec.evaluate(ground_truth, prediction)


def _dump(instance: BaseModel) -> Dict[str, Any]:
    """Normalize a pydantic instance to its JSON wire form.

    ``mode="json"`` is required: it turns ``date``/``datetime`` into ISO
    strings and enums into their values, matching the wire types the inferred
    comparators expect.
    """
    if not isinstance(instance, BaseModel):
        raise TypeError(
            f"expected a pydantic BaseModel instance, got {type(instance)!r}"
        )
    return instance.model_dump(mode="json")


def _shared_class(gt: BaseModel, pred: BaseModel) -> Type[BaseModel]:
    """Pick the class to infer from; require gt/pred to be related."""
    gt_cls, pred_cls = type(gt), type(pred)
    if gt_cls is pred_cls:
        return gt_cls
    if issubclass(pred_cls, gt_cls):
        return gt_cls
    if issubclass(gt_cls, pred_cls):
        return pred_cls
    raise TypeError(
        f"ground_truth ({gt_cls.__name__}) and prediction ({pred_cls.__name__}) "
        f"must be instances of the same (or a related) pydantic model."
    )
