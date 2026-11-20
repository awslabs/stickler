"""Regression tests for the maintainer review of #540 (Diego, Harry, Parmida)."""

import json
import warnings
from typing import List, Optional

import numpy as np
import pytest

from stickler import StructuredModel
from stickler.decisions import (
    Choice,
    Noul,
    Score,
    accept_or_escalate,
    confidence_intervals,
)
from stickler.decisions.accumulator import MIN_CALIBRATION_N, DecisionAccumulator
from stickler.decisions.arrays import DecisionArrays
from stickler.decisions.metrics import bootstrap_ci, calibration
from stickler.decisions.normalize import AnswerError, normalize_answer
from stickler.decisions.primitives import decision_paths, spec_from_question
from stickler.integrations.systemone import from_systemone, model_from_questions
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.post_comparison_accumulator import (
    PostComparisonAccumulator,
)

pytestmark = pytest.mark.filterwarnings("ignore:Decision field .* answered decisions")


def noul(p):
    return {"type": "noul", "noul": p}


class YesNo(StructuredModel):
    q: Optional[bool] = Noul()


class Queue(StructuredModel):
    queue: Optional[str] = Choice(["billing", "tech"])


def evaluate(pairs, **kwargs):
    ev = BulkStructuredModelEvaluator(target_schema=YesNo, **kwargs)
    for i, (gold, p) in enumerate(pairs):
        ev.update(YesNo(q=gold), YesNo.from_json({"q": None if p is None else noul(p)}), doc_id=f"d{i}")
    return ev.compute()


# Parmida 1: a "no" at 1 - c must tie with a "yes" at c.


@pytest.mark.parametrize("c", [round(0.51 + i / 100, 2) for i in range(49)])
def test_yes_and_no_at_the_same_confidence_tie(c):
    spec = spec_from_question("noul")
    rows = [
        {"doc": 0, "gold": 1, "probs": list(normalize_answer(spec, noul(c)).probs)},
        {"doc": 1, "gold": 1, "probs": list(normalize_answer(spec, noul(round(1 - c, 2))).probs)},
    ]
    a = DecisionArrays.from_records(spec, rows)
    assert a.conf[0] == a.conf[1]


def test_tied_confidence_auroc_is_one_half():
    res = evaluate([(True, 0.57), (True, 0.43)])  # yes at 0.57 right, no at 0.57 wrong
    assert res.decision_metrics["fields"]["q"]["calibration"]["auroc"] == 0.5


# Harry must-fix 1, Parmida 2: refusing must not improve AURC.


def test_missing_answers_do_not_improve_aurc():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 400)
    gold = rng.uniform(size=400) < p
    full = accept_or_escalate(evaluate(list(zip(gold, p)))).fields["q"]
    half = accept_or_escalate(evaluate([(g, x if i % 2 else None) for i, (g, x) in enumerate(zip(gold, p))])).fields["q"]
    assert full["aurc"] == pytest.approx(full["aurc_answered"])
    assert half["aurc"] > full["aurc"]
    assert half["aurc"] > half["aurc_answered"]


def test_aurc_is_the_mean_selective_risk():
    # Confidences 0.9 (right), 0.8 (wrong), 0.7 (right); one missing answer.
    res = evaluate([(True, 0.9), (True, 0.2), (True, 0.7), (True, None)])
    f = accept_or_escalate(res).fields["q"]
    # Answered: risks 0, 1/2, 1/3 at coverage 1/3, 2/3, 1.
    assert f["aurc_answered"] == pytest.approx((0 + 1 / 2 + 1 / 3) / 3)
    # All asked: the missing answer is accepted last, as an error: risk 2/4.
    assert f["aurc"] == pytest.approx((0 + 1 / 2 + 1 / 3 + 2 / 4) / 4)


# Parmida 3: the interval is for the same ece_mass as the point estimate.


def test_confidence_intervals_use_the_accumulators_bins():
    ev = BulkStructuredModelEvaluator(target_schema=YesNo, accumulators=[DecisionAccumulator(YesNo, n_bins=3)])
    rng = np.random.default_rng(1)
    for i, p in enumerate(rng.uniform(0, 1, 60)):
        ev.update(YesNo(q=bool(rng.uniform() < p)), YesNo.from_json({"q": noul(float(p))}), doc_id=i)
    res = ev.compute()
    ci = confidence_intervals(res, n_boot=50)
    assert ci["settings"]["n_bins"] == 3
    a = DecisionArrays.from_records(spec_from_question("noul"), res.decision_metrics["fields"]["q"]["records"])
    expected = bootstrap_ci(a, lambda x: calibration(x, n_bins=3), n_boot=50, keep=["ece_mass"])
    assert ci["q"]["calibration"]["ece_mass"] == expected["ece_mass"]


# Parmida 4: a reused doc_id is not a new document.


def test_reused_doc_id_is_rejected():
    ev = BulkStructuredModelEvaluator(target_schema=YesNo)
    ev.update(YesNo(q=True), YesNo.from_json({"q": noul(0.9)}), doc_id="same")
    ev.update(YesNo(q=True), YesNo.from_json({"q": noul(0.8)}), doc_id="same")
    res = ev.compute()
    assert res.decision_metrics["document_count"] == 1
    assert len(res.decision_metrics["fields"]["q"]["records"]) == 1
    assert res.accumulator_errors == {"decision_metrics": 1}
    assert "already evaluated" in res.errors[0]["error"]


def test_repeated_doc_ids_can_be_grouped():
    ev = BulkStructuredModelEvaluator(
        target_schema=YesNo, accumulators=[DecisionAccumulator(YesNo, repeated_doc_ids="group")]
    )
    for doc_id in ("a", "b", "a"):
        ev.update(YesNo(q=True), YesNo.from_json({"q": noul(0.9)}), doc_id=doc_id)
    dm = ev.compute().decision_metrics
    assert dm["document_count"] == 2
    assert [r["doc"] for r in dm["fields"]["q"]["records"]] == [0, 1, 0]


def test_generated_doc_ids_are_not_identities():
    shards = [BulkStructuredModelEvaluator(target_schema=YesNo) for _ in range(2)]
    for ev in shards:
        ev.update(YesNo(q=True), YesNo.from_json({"q": noul(0.9)}))  # no doc_id
    shards[0].merge_state(shards[1].get_state())
    assert shards[0].compute().decision_metrics["document_count"] == 2


def test_merging_shards_that_share_a_doc_id_raises():
    shards = [BulkStructuredModelEvaluator(target_schema=YesNo) for _ in range(2)]
    for ev in shards:
        ev.update(YesNo(q=True), YesNo.from_json({"q": noul(0.9)}), doc_id="x")
    with pytest.raises(ValueError, match="appear in both states"):
        shards[0].merge_state(shards[1].get_state())


def test_accumulator_without_the_hook_still_runs():
    class DuckTyped:
        name = "duck"

        def __init__(self):
            self.n = 0

        def reset(self):
            self.n = 0

        def accumulate(self, comparison_result, prediction_raw):
            self.n += 1

        def compute(self):
            return {"n": self.n}

    ev = BulkStructuredModelEvaluator(target_schema=Queue, accumulators=[DuckTyped()])
    ev.update(Queue(queue="billing"), Queue(queue=None))
    assert ev.compute().accumulator_metrics["duck"] == {"n": 1}
    assert PostComparisonAccumulator.start_document(DuckTyped(), "x") is None


# Diego 2, Harry must-fix 2: a bare value has no probabilities to score.


def test_bare_value_prediction_is_rejected_not_certain():
    ev = BulkStructuredModelEvaluator(target_schema=Queue)
    ev.update(Queue(queue="billing"), Queue.from_json({"queue": "billing"}), doc_id="a")
    res = ev.compute()
    assert res.decision_metrics is None
    assert res.accumulator_errors == {"decision_metrics": 1}
    assert "bare value" in res.errors[0]["error"]


@pytest.mark.parametrize("response", [{"answers": [1, 2]}, {"answers": "x"}, {"model": "m"}, object()])
def test_from_systemone_rejects_malformed_responses(response):
    with pytest.raises(AnswerError):
        from_systemone(response)


# Harry must-fix 3: the exported schema loads back as decision fields.


class Item(StructuredModel):
    intent: Optional[str] = Choice(["ask", "complain"])


class Ticket(StructuredModel):
    queue: Optional[str] = Choice({"billing": "Money", "tech": None}, weight=2.0, description="Queue")
    severity: Optional[int] = Score(["low", {"name": "high", "what": "Down"}])
    urgent: Optional[bool] = Noul("Act today?")
    items: Optional[List[Item]] = None


def test_schema_round_trip_keeps_decision_fields():
    schema = Ticket.to_json_schema()
    assert schema["properties"]["queue"]["x-aws-stickler-decision"] == {
        "type": "choice", "criteria": {"billing": "Money", "tech": None}
    }
    rebuilt = StructuredModel.from_json_schema(schema)
    original, again = decision_paths(Ticket), decision_paths(rebuilt)
    assert {p: (s.kind, s.labels, s.criteria) for p, s in again.items()} == {
        p: (s.kind, s.labels, s.criteria) for p, s in original.items()
    }
    assert rebuilt.to_json_schema()["properties"]["queue"] == schema["properties"]["queue"]
    pred = rebuilt.from_json({"queue": {"type": "choice", "probabilities": {"billing": 0.3, "tech": 0.7}}})
    assert pred.queue == "tech"


def test_schema_decision_field_with_another_comparator_is_rejected():
    schema = Ticket.to_json_schema()
    schema["properties"]["queue"]["x-aws-stickler-comparator"] = "LevenshteinComparator"
    with pytest.raises(ValueError, match="exact match at threshold 1.0"):
        StructuredModel.from_json_schema(schema)


# Diego 3, Harry: level numbers and level names never disagree.


def test_score_level_named_after_another_number_is_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        Score(["2", "1", "0"])
    Score(["0", "1", "2"])  # each name is its own position: fine


@pytest.mark.parametrize("key", ["²", "٢"])
def test_non_ascii_digit_keys_are_not_level_numbers(key):
    spec = spec_from_question("score", ["low", "mid", "high"])
    with pytest.raises(AnswerError, match="not a level number or level name"):
        normalize_answer(spec, {"type": "score", "probabilities": {key: 1.0, "0": 0.0, "1": 0.0}})


# Diego 4 and 5, Harry: saved files and accumulator names.


def test_saved_metrics_unchanged_without_decision_fields(tmp_path):
    class Plain(StructuredModel):
        name: Optional[str] = None

    ev = BulkStructuredModelEvaluator(target_schema=Plain)
    ev.update(Plain(name="a"), Plain(name="a"))
    ev.save_metrics(str(tmp_path / "m.json"))
    assert "decision_metrics" not in json.loads((tmp_path / "m.json").read_text())


def test_saved_metrics_hold_one_copy_of_the_records(tmp_path):
    ev = BulkStructuredModelEvaluator(target_schema=YesNo)
    ev.update(YesNo(q=True), YesNo.from_json({"q": noul(0.9)}), doc_id="a")
    ev.save_metrics(str(tmp_path / "m.json"))
    text = (tmp_path / "m.json").read_text()
    assert text.count('"records"') == 1


def test_user_accumulator_named_decision_metrics_is_not_read_as_ours():
    class Mine(PostComparisonAccumulator):
        name = "decision_metrics"

        def reset(self):
            pass

        def accumulate(self, comparison_result, prediction_raw):
            pass

        def compute(self):
            return {"mine": True}

        def get_state(self):
            return {}

        def load_state(self, state):
            pass

        def merge_state(self, other_state):
            pass

    class Plain(StructuredModel):
        name: Optional[str] = None

    ev = BulkStructuredModelEvaluator(target_schema=Plain, accumulators=[Mine()])
    ev.update(Plain(name="a"), Plain(name="a"))
    res = ev.compute()
    assert res.decision_metrics is None
    assert res.accumulator_metrics["decision_metrics"] == {"mine": True}
    with pytest.raises(ValueError, match="duplicate accumulator names"):
        BulkStructuredModelEvaluator(target_schema=YesNo, accumulators=[Mine()])


# Harry: undefined is not 0.0; Parmida 6: balanced accuracy and recall of "no".


def test_noul_precision_and_recall_are_undefined_without_positives():
    c = evaluate([(False, 0.2), (False, 0.3)]).decision_metrics["fields"]["q"]["correctness"]
    assert (c["precision"], c["recall"], c["f1"]) == (None, None, None)
    assert c["recall_no"] == 1.0


def test_balanced_accuracy_counts_missing_as_wrong():
    # Gold yes x3 (right, right, missing), gold no x1 (wrong).
    c = evaluate([(True, 0.9), (True, 0.8), (True, None), (False, 0.7)]).decision_metrics["fields"]["q"]["correctness"]
    assert c["balanced_accuracy"] == pytest.approx((2 / 3 + 0) / 2)
    assert c["recall_no"] == 0.0


def test_log_loss_median_resists_one_confident_mistake():
    cal = evaluate([(True, 0.9)] * 9 + [(True, 1e-9)]).decision_metrics["fields"]["q"]["calibration"]
    assert cal["log_loss"] > 2.0
    assert cal["log_loss_median"] == pytest.approx(-np.log(0.9))


# Harry, Parmida 9: intervals on rates only.


def test_confidence_intervals_skip_counts_and_bins():
    ev = BulkStructuredModelEvaluator(target_schema=Queue)
    rng = np.random.default_rng(2)
    for i in range(40):
        a = float(rng.uniform())
        ev.update(Queue(queue=["billing", "tech"][i % 2]),
                  Queue.from_json({"queue": {"type": "choice", "probabilities": {"billing": a, "tech": 1 - a}}}),
                  doc_id=i)
    ci = confidence_intervals(ev.compute(), n_boot=20)["queue"]

    def keys(d):
        for k, v in d.items():
            yield k
            if isinstance(v, dict) and "low" not in v:
                yield from keys(v)

    found = set(keys(ci["correctness"])) | set(keys(ci["calibration"]))
    assert not found & {"support", "confusion", "ece_mass_bins", "ok", "missing", "threshold"}
    assert {"accuracy", "macro_f1", "auroc", "ece_mass", "nll"} <= found
    only = confidence_intervals(ev.compute(), n_boot=20, metrics=["accuracy"])["queue"]
    assert set(only["correctness"]) == {"accuracy"}


# Parmida 10: per-field thresholds.


def test_per_field_taus():
    Two = model_from_questions(
        {"q": {"type": "noul", "instructions": "?"}, "r": {"type": "noul", "instructions": "?"}}, model_name="Two"
    )
    ev = BulkStructuredModelEvaluator(target_schema=Two)
    ev.update(Two(q=True, r=True), Two.from_json({"q": noul(0.9), "r": noul(0.9)}), doc_id="a")
    rep = accept_or_escalate(ev.compute(), taus={"q": [0.5]})
    assert [r["tau"] for r in rep.fields["q"]["rows"]] == [0.5]
    assert len(rep.fields["r"]["rows"]) == 6  # DEFAULT_TAUS
    with pytest.raises(ValueError, match="unknown decision fields"):
        accept_or_escalate(ev.compute(), taus={"nope": [0.5]})
    with pytest.raises(ValueError, match="cannot be pooled"):
        accept_or_escalate(ev.compute(), taus={"q": [0.5]}, pool=True)


# Harry: small samples.


def test_small_sample_warns():
    ev = BulkStructuredModelEvaluator(target_schema=YesNo)
    ev.update(YesNo(q=True), YesNo.from_json({"q": noul(0.9)}), doc_id="a")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev.compute()
    assert any(f"below {MIN_CALIBRATION_N}" in str(w.message) for w in caught)
