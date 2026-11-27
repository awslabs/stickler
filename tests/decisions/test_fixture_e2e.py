"""The System One fixture, end to end: questions -> model -> bulk -> metrics.

Reads examples/notebooks/fixtures/systemone/. The committed responses are
synthetic placeholders, so these tests check consistency, not model quality.
"""

import json
from pathlib import Path

import pytest

from stickler.decisions import accept_or_escalate, confidence_intervals
from stickler.integrations.systemone import from_systemone, model_from_questions
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)

FIXTURE = Path(__file__).resolve().parents[2] / "examples" / "notebooks" / "fixtures" / "systemone"


def _jsonl(name):
    return [json.loads(line) for line in (FIXTURE / name).read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def run():
    questions = json.loads((FIXTURE / "questions.json").read_text())
    tickets = {t["id"]: t for t in _jsonl("tickets.jsonl")}
    responses = {r["id"]: r["response"] for r in _jsonl("responses.jsonl")}
    Model = model_from_questions(questions)
    ev = BulkStructuredModelEvaluator(target_schema=Model)
    for tid, ticket in tickets.items():
        ev.update(Model(**ticket["gold"]), Model.from_json(from_systemone(responses[tid])), doc_id=tid)
    return questions, tickets, responses, ev.compute()


def test_every_question_has_metrics(run):
    questions, _, _, res = run
    assert set(res.decision_metrics["fields"]) == set(questions)


def test_counts_match_the_fixture(run):
    questions, tickets, responses, res = run
    for q in questions:
        asked = [t for t in tickets.values() if q in t["asked"] and t["gold"].get(q) is not None]
        answered = [t for t in asked if q in responses[t["id"]]["answers"]]
        cov = res.decision_metrics["fields"][q]["correctness"]["coverage"]
        assert cov["ok"] + cov["missing"] == len(asked)
        assert cov["ok"] == len(answered)


def test_accept_or_escalate_and_intervals_run(run):
    *_, res = run
    rep = accept_or_escalate(res, taus=[0.5, 0.8], target_accuracy=0.9)
    assert set(rep.fields) == set(res.decision_metrics["fields"])
    ci = confidence_intervals(res, n_boot=50, seed=0)
    q = next(iter(res.decision_metrics["fields"]))
    lo, hi = ci[q]["correctness"]["accuracy"]["low"], ci[q]["correctness"]["accuracy"]["high"]
    assert lo <= res.decision_metrics["fields"][q]["correctness"]["accuracy"] <= hi


def test_provenance_is_honest():
    prov = json.loads((FIXTURE / "provenance.json").read_text())
    models = {r["response"].get("model") for r in _jsonl("responses.jsonl")}
    if prov.get("synthetic") or prov.get("hand_authored"):
        assert models == {"synthetic-placeholder"}
    else:
        assert "synthetic-placeholder" not in models
