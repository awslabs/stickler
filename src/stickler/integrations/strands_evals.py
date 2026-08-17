"""Field-level structured-output evaluation for Strands Evals.

Strands Evals' deterministic evaluators compare structured output with
``Equals``: whole-object ``==``, scoring 0.0 or 1.0. An extraction that gets
nine of ten fields right is indistinguishable from one that gets none right,
and a reordered list counts as wrong. This evaluator compares field by field
with type-aware comparators, order-independent list matching and per-field
thresholds, so the score reflects how wrong the output is and names which
field to fix.

Requires the ``strands-evals`` extra::

    pip install "stickler-eval[strands-evals]"

Usage::

    from strands_evals import Case, Experiment
    from stickler.integrations.strands_evals import StructuredOutputEvaluator

    evaluator = StructuredOutputEvaluator(Invoice)
    report = Experiment(cases=cases, evaluators=[evaluator]).run_evaluations(task)

    report.overall_score          # weighted mean across the dataset
    report.scores                 # one weighted score per case
    evaluator.per_case()          # per-document field scores
    evaluator.metrics()           # per-field confusion matrix across the dataset

The design choices, and the two upstream gaps this works around, are written up
in ``docs/docs/Guides/Integrations/strands-evals.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Type

from pydantic import BaseModel

from .. import aggregate_from_comparisons, eval_for

try:
    from strands_evals.evaluators import Evaluator
    from strands_evals.types.evaluation import EvaluationData, EvaluationOutput

    STRANDS_EVALS_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - exercised by the extras gate
    STRANDS_EVALS_AVAILABLE = False
    _IMPORT_ERROR = exc

    class Evaluator:  # type: ignore[no-redef]
        """Placeholder so the module imports without the extra installed."""


@dataclass(frozen=True)
class _CaseResult:
    """What one comparison produced, kept for reading after the run.

    ``EvaluationOutput`` carries four scalars, so it cannot express a per-field
    breakdown. Keeping the comparison here instead means :meth:`per_case` can
    return the real numbers rather than a reconstruction.
    """

    name: Optional[str]
    model_cls: type
    overall_score: float
    matched: bool
    field_scores: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def _require_strands_evals() -> None:
    if not STRANDS_EVALS_AVAILABLE:
        raise ImportError(
            "StructuredOutputEvaluator requires the 'strands-agents-evals' "
            'package. Install it with: pip install "stickler-eval[strands-evals]"'
        ) from _IMPORT_ERROR


class StructuredOutputEvaluator(Evaluator):
    """Score structured output against ground truth, field by field.

    Deterministic and offline: no LLM judge, no credentials, no per-call cost.
    Comparison configuration is inferred from the Pydantic model itself, the
    same model the agent already passes as ``structured_output_model``, so no
    schema or annotation work is required. Every inferred decision is
    inspectable via :meth:`explain`.

    Returns one ``EvaluationOutput`` per case, carrying stickler's weighted
    ``overall_score``. Field-level detail is read from the evaluator rather than
    squeezed through that type, which has only four scalar fields:
    :meth:`per_case` for per-document field scores, :meth:`metrics` for the
    dataset-wide confusion matrix. Both read comparisons already performed, so
    neither runs anything twice.

    Args:
        model_cls: The Pydantic model the agent emits. Optional. When given,
            every case is coerced to it and anything that will not validate
            raises, which is what a single-schema suite wants. When omitted,
            the class is inferred per case and :meth:`metrics` partitions its
            rollup by class, so a suite mixing output types stays readable.
        match_threshold: Similarity at or above which an object counts as a
            match. Drives ``test_pass`` and the per-element matching of
            ``List[Model]`` fields.
        weight_hints: When True, weight fields by name-based importance
            heuristics (ids and amounts count for more). Off by default so
            weights stay uniform and metrics are not skewed by guessed
            business criticality.
        name: Optional evaluator name, forwarded to ``Evaluator``.

    Raises:
        ImportError: If the ``strands-evals`` extra is not installed.
    """

    def __init__(
        self,
        model_cls: Optional[Type[BaseModel]] = None,
        *,
        match_threshold: float = 0.7,
        weight_hints: bool = False,
        name: Optional[str] = None,
    ) -> None:
        _require_strands_evals()
        super().__init__(name=name)
        self.declared_cls = model_cls
        self.match_threshold = match_threshold
        self.weight_hints = weight_hints

        self._specs: Dict[type, Any] = {}
        # One _CaseResult per evaluated case. Appended to and never read during a
        # run. `list.append` is atomic under the GIL, so this needs no lock even
        # though the harness calls evaluate() from up to `max_workers` threads
        # via asyncio.to_thread (default 10). Everything is read afterwards, on
        # one thread, by metrics() and per_case().
        self._results: List[_CaseResult] = []

    # ---------------------------------------------------------------- per case

    def evaluate(self, evaluation_case: "EvaluationData") -> List["EvaluationOutput"]:
        """Compare one case's actual output against its expected output."""
        cls = self._resolve_cls(evaluation_case)
        spec = self._spec_for(cls)

        expected = self._coerce(evaluation_case.expected_output, cls, "expected_output")
        actual = self._coerce(evaluation_case.actual_output, cls, "actual_output")

        result = spec.evaluate(expected, actual)

        # `prediction_raw` is dropped: only the confidence accumulators consume
        # it, and they need `field_comparisons` alongside it. Keeping it without
        # them makes aggregate_from_comparisons warn on every call, and it is the
        # bulkiest part of the result. field_metrics is identical without it.
        self._results.append(
            _CaseResult(
                name=evaluation_case.name,
                model_cls=cls,
                overall_score=result.overall_score,
                matched=result.matched,
                field_scores=dict(result.field_scores),
                raw={k: v for k, v in result.raw.items() if k != "prediction_raw"},
            )
        )

        return [
            EvaluationOutput(
                score=result.overall_score,
                test_pass=result.matched,
                reason=self._weakest(result.field_scores),
                label=cls.__name__,
            )
        ]

    # ------------------------------------------------------------- cross case

    def metrics(self) -> Dict[str, Any]:
        """Per-field metrics across every case evaluated so far.

        Keyed by model class name, so a suite mixing output types gets one
        rollup per type. Feeding two schemas into a single rollup is accepted
        silently by the bulk evaluator and unions their field paths, which makes
        a field present in half the documents read as missed in the rest.

        Each value is a ``ProcessEvaluation``. Its ``field_metrics`` is keyed by
        dotted path (``line_items.sku``) and carries the five-category counts
        (tp/tn/fn/fa/fd) plus precision, recall, F1 and accuracy.

        Read after the run. A nested path's counts only cover documents whose
        parent pair scored at or above ``match_threshold``: below that,
        threshold gating treats the pair as atomic and emits no field breakdown,
        so a nested field has a smaller denominator than the document count.
        """
        by_cls: Dict[type, List[Dict[str, Any]]] = {}
        for case in list(self._results):
            by_cls.setdefault(case.model_cls, []).append(case.raw)
        return {
            cls.__name__: aggregate_from_comparisons(raws)
            for cls, raws in by_cls.items()
        }

    def per_case(self) -> List[Mapping[str, Any]]:
        """Per-document field scores, in the order the cases completed.

        ``EvaluationOutput`` has four scalar fields, so the harness's
        ``report.detailed_results`` can only ever echo what ``evaluate()``
        returned. Reading from the retained comparison instead means the real
        per-field numbers are available without squeezing them through that
        shape, and without a second comparison pass.

        Each entry has ``case``, ``model``, ``overall_score``, ``matched`` and
        ``field_scores``. Nested list children are absent from ``field_scores``
        because stickler emits no per-leaf score for them; use :meth:`metrics`
        for those, which reports their counts and precision/recall/F1.
        """
        return [
            {
                "case": case.name,
                "model": case.model_cls.__name__,
                "overall_score": case.overall_score,
                "matched": case.matched,
                "field_scores": dict(case.field_scores),
            }
            for case in list(self._results)
        ]

    def reset(self) -> None:
        """Drop accumulated results.

        Evaluator instances are shared across cases and may be reused across
        experiments, so a stateful evaluator needs an explicit way to clear.
        """
        self._results.clear()

    def explain(self) -> Dict[str, Dict[str, Any]]:
        """Per-field comparison config and why it was chosen.

        Keyed by dotted path, so nested decisions are auditable too. Needs
        either a declared ``model_cls`` or at least one evaluated case, since
        the config comes from the model.
        """
        if self.declared_cls is not None:
            return self._spec_for(self.declared_cls).explain()
        if not self._specs:
            raise RuntimeError(
                "explain() needs a model: pass model_cls to the constructor, or "
                "evaluate at least one case first."
            )
        if len(self._specs) > 1:
            names = ", ".join(sorted(c.__name__ for c in self._specs))
            raise RuntimeError(
                f"explain() is ambiguous across {len(self._specs)} inferred "
                f"schemas ({names}). Pass model_cls to select one."
            )
        return next(iter(self._specs.values())).explain()

    # ---------------------------------------------------------------- internal

    def _resolve_cls(self, evaluation_case: "EvaluationData") -> Type[BaseModel]:
        if self.declared_cls is not None:
            return self.declared_cls
        for value in (evaluation_case.actual_output, evaluation_case.expected_output):
            if isinstance(value, BaseModel):
                return type(value)
        raise TypeError(
            f"{type(self).__name__} could not infer a model class from this case. "
            f"Pass model_cls to the constructor, or supply outputs as Pydantic "
            f"model instances rather than dicts or JSON strings."
        )

    def _spec_for(self, cls: Type[BaseModel]) -> Any:
        spec = self._specs.get(cls)
        if spec is None:
            spec = eval_for(
                cls,
                match_threshold=self.match_threshold,
                weight_hints=self.weight_hints,
            )
            self._specs[cls] = spec
        return spec

    def _coerce(self, value: Any, cls: Type[BaseModel], which: str) -> BaseModel:
        """Accept a model instance, a dict, or a JSON string."""
        if isinstance(value, cls):
            return value
        if isinstance(value, BaseModel):
            # A different model class carrying the same fields.
            return cls.model_validate(value.model_dump())
        if isinstance(value, dict):
            return cls.model_validate(value)
        if isinstance(value, str):
            return cls.model_validate_json(value)
        raise TypeError(
            f"{type(self).__name__} needs {which} as a {cls.__name__} instance, "
            f"dict, or JSON string; got {type(value).__name__}"
        )

    @staticmethod
    def _weakest(field_scores: Mapping[str, float], limit: int = 4) -> str:
        """Name the weakest fields so a low case score is actionable."""
        imperfect = sorted(
            ((name, score) for name, score in field_scores.items() if score < 1.0),
            key=lambda pair: pair[1],
        )
        if not imperfect:
            return "all fields matched"
        listed = "; ".join(f"{name}={score:.2f}" for name, score in imperfect[:limit])
        if len(imperfect) > limit:
            listed += f"; (+{len(imperfect) - limit} more)"
        return f"weakest fields: {listed}"
