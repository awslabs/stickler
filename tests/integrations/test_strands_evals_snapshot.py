"""Golden snapshot of what Strands Evals users see for typical structured output.

Any change to this snapshot changes what Strands Evals users see: their scores,
their `test_pass` verdicts, or their `StructuredOutputReport.metrics()`. A PR
that changes it needs a CHANGELOG entry and a check against strands-agents-evals
(run tests/integrations/test_strands_evals_e2e.py with it installed) before
merging.

For each typical case in `strands_corpus.SNAPSHOT_CORPUS` it records the scalars,
`field_scores`, the overall confusion counts and the 1.6.0 `test_pass`; per
model, the aggregate `metrics`, `field_metrics` and the `explain()` choices.
Floats are rounded to 6 places, then compared exactly.

Regenerate after an intended change, then review the JSON diff:

    STICKLER_UPDATE_STRANDS_SNAPSHOT=1 pytest tests/integrations/test_strands_evals_snapshot.py
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from stickler import aggregate_from_comparisons, eval_for
from tests.integrations.strands_corpus import (
    SNAPSHOT_CORPUS,
    STRANDS_DEFAULT_THRESHOLD,
    stored_comparison,
    strands_test_pass,
)

SNAPSHOT_PATH = Path(__file__).with_name("strands_evals_snapshot.json")
UPDATE_ENV = "STICKLER_UPDATE_STRANDS_SNAPSHOT"
EXPLAIN_FIELDS = ("comparator", "threshold", "weight", "clip_under_threshold", "source")


def _round(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(k): _round(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def build_snapshot() -> Dict[str, Any]:
    specs: Dict[type, Any] = {}
    stored: Dict[str, List[Dict[str, Any]]] = {}
    cases: Dict[str, Any] = {}
    for case in SNAPSHOT_CORPUS:
        spec = specs.setdefault(
            case.model,
            eval_for(
                case.model,
                match_threshold=STRANDS_DEFAULT_THRESHOLD,
                weight_hints=False,
            ),
        )
        result = spec.evaluate(case.expected, case.actual)
        overall = result.raw["confusion_matrix"]["overall"]
        cases[case.id] = _round(
            {
                "overall_score": result.overall_score,
                "matched": result.matched,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "field_scores": result.field_scores,
                "confusion_matrix": {
                    k: overall[k] for k in ("tp", "fa", "fd", "fn", "tn")
                },
                "test_pass": strands_test_pass(result, STRANDS_DEFAULT_THRESHOLD),
            }
        )
        stored.setdefault(case.model.__name__, []).append(stored_comparison(result))

    models: Dict[str, Any] = {}
    for model, spec in specs.items():
        evaluation = aggregate_from_comparisons(stored[model.__name__])
        models[model.__name__] = _round(
            {
                "document_count": evaluation.document_count,
                "metrics": evaluation.metrics,
                "field_metrics": evaluation.field_metrics,
                "explain": {
                    path: {k: entry[k] for k in EXPLAIN_FIELDS}
                    for path, entry in spec.explain().items()
                },
            }
        )
    return {
        "match_threshold": STRANDS_DEFAULT_THRESHOLD,
        "cases": cases,
        "models": models,
    }


def _diff(expected: Any, actual: Any, path: str = "") -> List[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        lines: List[str] = []
        for key in sorted(set(expected) | set(actual)):
            where = f"{path}.{key}" if path else key
            if key not in actual:
                lines.append(f"{where}: missing (expected {expected[key]!r})")
            elif key not in expected:
                lines.append(f"{where}: unexpected {actual[key]!r}")
            else:
                lines.extend(_diff(expected[key], actual[key], where))
        return lines
    return [] if expected == actual else [f"{path}: {expected!r} -> {actual!r}"]


def test_snapshot_matches():
    actual = json.loads(json.dumps(build_snapshot()))
    if os.environ.get(UPDATE_ENV):
        SNAPSHOT_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"rewrote {SNAPSHOT_PATH.name}; review the diff")
    expected = json.loads(SNAPSHOT_PATH.read_text())
    differences = _diff(expected, actual)
    assert not differences, (
        "What Strands Evals users see has changed. If intended, add a CHANGELOG "
        f"entry, check strands-agents-evals, and rerun with {UPDATE_ENV}=1.\n"
        + "\n".join(differences[:60])
    )


def test_snapshot_covers_the_corpus():
    expected = json.loads(SNAPSHOT_PATH.read_text())
    assert set(expected["cases"]) == {case.id for case in SNAPSHOT_CORPUS}
