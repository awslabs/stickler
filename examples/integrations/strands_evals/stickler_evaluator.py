"""Field-level structured-output evaluation for Strands Evals, via stickler.

Reference implementation of the integration requested in
strands-agents/evals#310. Written in the shape it would take inside
``strands-agents/evals`` (``src/strands_evals/evaluators/stickler.py``), behind
an optional extra:

    [project.optional-dependencies]
    stickler = ["stickler-eval>=0.5.0"]

Why: Strands Evals' deterministic evaluators compare structured output with
``Equals`` (whole-object ``==``, scoring 0.0 or 1.0). An extraction that gets
nine of ten fields right is indistinguishable from one that gets none right,
and a reordered list counts as wrong. Stickler compares field by field with
type-aware comparators, order-independent list matching, and per-field
thresholds, so the score reflects how wrong the output actually is and names
which fields to look at.

Usage:

    from strands_evals import Case, Experiment

    experiment = Experiment(
        cases=[Case(name="doc-1", input=doc, expected_output=labeled_invoice)],
        evaluators=[StructuredOutputEvaluator(Invoice)],
    )
    report = experiment.run_evaluations(
        lambda case: agent(case.input, structured_output_model=Invoice).structured_output
    )
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Type

from pydantic import BaseModel
from strands_evals.evaluators import Evaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput

try:
    import stickler

    STICKLER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the extras gate
    STICKLER_AVAILABLE = False


class StructuredOutputEvaluator(Evaluator):
    """Score structured output against ground truth, field by field.

    Deterministic and offline: no LLM judge, no credentials, no per-call cost.
    Comparison configuration is inferred from the Pydantic model itself (the
    same model the agent already uses for structured output), so no schema or
    annotation work is required. Every inferred decision is inspectable via
    :meth:`explain`.

    Args:
        model_cls: The Pydantic model the agent emits, e.g. the class passed as
            ``structured_output_model``.
        match_threshold: Similarity at or above which an object counts as a
            match. Drives ``test_pass`` and the per-element matching of
            ``List[Model]`` fields.
        weight_hints: When True, weight fields by name-based importance
            heuristics (ids and amounts count for more). Off by default so
            weights stay uniform and metrics are not skewed by guessed
            business criticality.
        name: Optional evaluator name, forwarded to ``Evaluator``.

    Maps stickler's result onto ``EvaluationOutput``:

    - ``score``: weighted mean of per-field scores (``overall_score``)
    - ``test_pass``: every field met its threshold (``matched``)
    - ``reason``: the lowest-scoring fields, so a failure is actionable
    - ``label``: the model name, to group results by output type
    """

    def __init__(
        self,
        model_cls: Type[BaseModel],
        *,
        match_threshold: float = 0.7,
        weight_hints: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if not STICKLER_AVAILABLE:
            raise ImportError(
                "StructuredOutputEvaluator requires the 'stickler-eval' package. "
                'Install it with: pip install "strands-agents-evals[stickler]"'
            )
        super().__init__(name=name)
        self.model_cls = model_cls
        self.spec = stickler.eval_for(
            model_cls,
            match_threshold=match_threshold,
            weight_hints=weight_hints,
        )

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        """Compare one case's actual output against its expected output."""
        expected = self._coerce(evaluation_case.expected_output, "expected_output")
        actual = self._coerce(evaluation_case.actual_output, "actual_output")
        result = self.spec.evaluate(expected, actual)

        return [
            EvaluationOutput(
                score=result.overall_score,
                test_pass=result.matched,
                reason=self._reason(result),
                label=self.model_cls.__name__,
            )
        ]

    def explain(self) -> Dict[str, Dict[str, Any]]:
        """Per-field comparison config and why it was chosen.

        Keyed by dotted path (``line_items.sku``), so nested decisions are
        auditable too. Useful for justifying scores in a report, and for
        deciding which fields to configure explicitly.
        """
        return self.spec.explain()

    def aggregate(
        self, pairs: Iterable[tuple[Any, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Roll up per-field scores across a dataset of (expected, actual) pairs.

        Strands Evals has no post-evaluation aggregation hook, and ``EvaluationOutput``
        carries only four scalar fields, so the field-level rollup that is the whole
        point of this integration cannot cross the harness boundary. This method keeps
        that rollup inside the evaluator instead of forcing callers to re-loop over
        results, returning per field: ``mean``, ``worst``, ``perfect`` (count scoring
        1.0), ``count``, ``below_threshold`` (count under the field's threshold), and
        the chosen ``comparator``.

        TODO: replace this bespoke loop with stickler's stateful bulk path
        (``stickler.structured_object_evaluator.BulkStructuredModelEvaluator`` /
        ``aggregate_from_comparisons``), which accumulates a confusion matrix
        incrementally instead of holding every per-field score in memory. Kept inline
        for now because that path returns confusion-matrix metrics rather than the
        mean/worst/perfect view this demo reports.
        """
        config = self.spec.explain()
        scores: Dict[str, list] = defaultdict(list)
        below_threshold: Dict[str, int] = defaultdict(int)

        for expected, actual in pairs:
            result = self.spec.evaluate(
                self._coerce(expected, "expected_output"),
                self._coerce(actual, "actual_output"),
            )
            for field, score in result.field_scores.items():
                scores[field].append(score)
            for field, entry in result.explain().items():
                if "score" in entry and entry["score"] < entry["threshold"]:
                    below_threshold[field] += 1

        rollup = {}
        for field, field_scores in scores.items():
            rollup[field] = {
                "mean": statistics.mean(field_scores),
                "worst": min(field_scores),
                "perfect": sum(1 for s in field_scores if s == 1.0),
                "count": len(field_scores),
                "below_threshold": below_threshold[field],
                "comparator": config.get(field, {}).get("comparator", "unknown"),
            }
        # Sort worst mean first: the fields an engineer should look at.
        return dict(sorted(rollup.items(), key=lambda kv: kv[1]["mean"]))

    def _coerce(self, value: Any, which: str) -> BaseModel:
        """Accept a model instance, a dict, or a JSON string."""
        if isinstance(value, self.model_cls):
            return value
        if isinstance(value, BaseModel):
            # A different model class that carries the same fields.
            return self.model_cls.model_validate(value.model_dump())
        if isinstance(value, dict):
            return self.model_cls.model_validate(value)
        if isinstance(value, str):
            return self.model_cls.model_validate_json(value)
        raise TypeError(
            f"{type(self).__name__} needs {which} as a {self.model_cls.__name__} "
            f"instance, dict, or JSON string; got {type(value).__name__}"
        )

    @staticmethod
    def _reason(result: Any, limit: int = 4) -> str:
        """Name the weakest fields so a low score is actionable."""
        imperfect = sorted(
            (
                (field, score)
                for field, score in result.field_scores.items()
                if score < 1.0
            ),
            key=lambda pair: pair[1],
        )
        if not imperfect:
            return "all fields matched"
        listed = "; ".join(f"{field}={score:.2f}" for field, score in imperfect[:limit])
        if len(imperfect) > limit:
            listed += f"; (+{len(imperfect) - limit} more)"
        return f"weakest fields: {listed}"
