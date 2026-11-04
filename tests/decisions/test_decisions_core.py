"""Decision fields end to end: from_json, compare_with, the bulk evaluator,
accept-or-escalate and confidence intervals.

The four tickets are the ones in the #469 API sketch; their expected numbers
were computed independently with sklearn and numpy.
"""

import json
import math
import warnings
from typing import Optional

import pytest

from stickler import StructuredModel
from stickler.decisions import (
    AnswerError,
    Choice,
    DecisionAccumulator,
    Noul,
    Score,
    accept_or_escalate,
    confidence_intervals,
    decision_fields,
)
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)


class Triage(StructuredModel):
    queue: Optional[str] = Choice(criteria={"billing": None, "sales": None, "retail": None}, weight=2.0)
    urgent: Optional[bool] = Noul()
    severity: Optional[int] = Score(criteria=["low", "medium", "high"])


def choice(b, s, r):
    return {"type": "choice", "probabilities": {"billing": b, "sales": s, "retail": r}}


def noul(p):
    return {"type": "noul", "noul": p}


def score(*ps):
    return {"type": "score", "probabilities": {str(i): p for i, p in enumerate(ps)}}


TICKETS = [
    ({"queue": "billing", "urgent": True, "severity": 1},
     {"queue": choice(0.77, 0.12, 0.11), "urgent": noul(0.86), "severity": score(0.20, 0.57, 0.23)}),
    ({"queue": "sales", "urgent": False, "severity": 0},
     {"queue": choice(0.60, 0.30, 0.10), "urgent": noul(0.30), "severity": score(0.70, 0.25, 0.05)}),
    ({"queue": "retail", "urgent": True, "severity": 2},
     {"queue": choice(0.05, 0.05, 0.90), "urgent": noul(0.40), "severity": score(0.05, 0.55, 0.40)}),
    ({"queue": "billing", "urgent": False, "severity": 1},
     {"urgent": noul(0.10), "severity": score(0.10, 0.30, 0.60)}),  # queue answer missing
]


def run(tickets=TICKETS):
    ev = BulkStructuredModelEvaluator(target_schema=Triage)
    for i, (gold, answers) in enumerate(tickets):
        ev.update(Triage(**gold), Triage.from_json(answers), doc_id=f"t{i + 1}")
    return ev.compute()


approx = pytest.approx


class TestFields:
    def test_specs(self):
        specs = decision_fields(Triage)
        assert {k: v.kind for k, v in specs.items()} == {"queue": "choice", "urgent": "noul", "severity": "score"}
        assert specs["severity"].labels == ("low", "medium", "high")

    def test_schema_records_the_question(self):
        props = Triage.model_json_schema()["properties"]
        assert props["severity"]["x-stickler-decision"] == {"type": "score", "criteria": ["low", "medium", "high"]}

    def test_from_json_reads_answers(self):
        pred = Triage.from_json(TICKETS[1][1])
        assert (pred.queue, pred.urgent, pred.severity) == ("billing", False, 0)
        assert pred.get_all_confidences() == approx({"queue": 0.6, "urgent": 0.7, "severity": 0.7})

    def test_score_keys_by_level_name(self):
        pred = Triage.from_json({"severity": {"type": "score", "probabilities": {"low": 0.1, "medium": 0.2, "high": 0.7}}})
        assert pred.severity == 2

    @pytest.mark.parametrize(
        "answer",
        [
            {"type": "noul", "noul": 0.5},  # wrong type for a Choice field
            {"type": "choice", "probabilities": {"billing": 0.5, "sales": 0.5}},  # missing option
            {"type": "choice", "probabilities": {"billing": 0.5, "sales": 0.3, "retail": 0.1, "other": 0.1}},
            {"type": "choice", "probabilities": {"billing": 0.5, "sales": 0.3, "retail": 0.1}},  # sums to 0.9
        ],
    )
    def test_bad_answers_raise(self, answer):
        with pytest.raises(AnswerError):
            Triage.from_json({"queue": answer})

    def test_answer_confidence_field_is_ignored(self):
        answer = {**choice(0.77, 0.12, 0.11), "choice": "sales", "confidence": 0.66}
        pred = Triage.from_json({"queue": answer})
        assert pred.queue == "billing"
        assert pred.get_all_confidences()["queue"] == approx(0.77)


class TestCompareWith:
    def test_one_ticket(self):
        gold, answers = TICKETS[1]
        r = Triage(**gold).compare_with(Triage.from_json(answers))
        assert r["field_scores"] == {"queue": 0.0, "urgent": 1.0, "severity": 1.0}
        assert r["overall_score"] == approx(0.5)

    def test_unasked_question_carries_no_weight(self):
        gold = Triage(queue="sales", urgent=True, severity=None)  # severity not asked
        r = gold.compare_with(Triage.from_json({"queue": choice(0.6, 0.3, 0.1)}))
        # queue wrong (weight 2), urgent missing (weight 1); severity excluded, else 0.25
        assert r["overall_score"] == approx(0.0)

    def test_extraction_fields_keep_absent_on_both(self):
        from stickler import ComparableField, ExactComparator

        class Mixed(StructuredModel):
            name: Optional[str] = ComparableField(comparator=ExactComparator())
            queue: Optional[str] = Choice(criteria=["billing", "sales"])

        r = Mixed(name=None, queue="sales").compare_with(
            Mixed.from_json({"queue": {"type": "choice", "probabilities": {"billing": 0.9, "sales": 0.1}}})
        )
        # name is None on both sides: an extraction field still scores 1.0 and counts
        assert r["overall_score"] == approx(0.5)


class TestBulk:
    def test_accumulator_installed_and_result_shape(self):
        res = run()
        dm = res.decision_metrics
        assert dm is res.accumulator_metrics["decision_metrics"] or dm == res.accumulator_metrics["decision_metrics"]
        assert set(dm["fields"]) == {"queue", "urgent", "severity"}
        json.dumps(dm)  # JSON-serialisable

    def test_choice_numbers(self):
        q = run().decision_metrics["fields"]["queue"]
        c, k = q["correctness"], q["calibration"]
        assert c["accuracy"] == approx(0.5)
        assert c["answered_accuracy"] == approx(2 / 3)
        assert c["coverage"] == {"ratio": approx(0.75), "ok": 3, "missing": 1}
        assert c["macro_f1"] == approx(0.5)
        assert c["confusion"]["billing"] == {"billing": 1, "(no answer)": 1}
        assert k["auroc"] == approx(1.0)
        assert k["brier_multiclass"] == approx(0.31813, abs=1e-5)
        assert k["nll"] == approx(0.52357, abs=1e-5)
        assert k["ece_mass"] == approx(0.31)

    def test_noul_numbers(self):
        u = run().decision_metrics["fields"]["urgent"]
        c, k = u["correctness"], u["calibration"]
        assert (c["accuracy"], c["precision"], c["recall"]) == (approx(0.75), approx(1.0), approx(0.5))
        assert k["brier"] == approx(0.1199)
        assert k["log_loss"] == approx(0.38229, abs=1e-5)
        assert k["auroc"] == approx(1.0) and k["auroc_label"] == approx(1.0)

    def test_score_numbers(self):
        s = run().decision_metrics["fields"]["severity"]
        c, k = s["correctness"], s["calibration"]
        assert (c["qwk"], c["mae_argmax"], c["mae_expected"]) == (approx(0.5), approx(0.5), approx(0.3825))
        assert k["rps"] == approx(0.11474, abs=1e-5)
        assert k["auroc"] == approx(0.75)

    def test_unasked_is_not_recorded(self):
        tickets = [(dict(TICKETS[0][0], severity=None), {k: v for k, v in TICKETS[0][1].items() if k != "severity"})]
        tickets += TICKETS[1:]
        s = run(tickets).decision_metrics["fields"]["severity"]
        assert s["correctness"]["coverage"]["ok"] + s["correctness"]["coverage"]["missing"] == 3

    def test_existing_confidence_metrics_see_decision_confidence(self):
        res = run()
        assert res.confidence_metrics is not None

    def test_merge_state_offsets_documents(self):
        a, b = (BulkStructuredModelEvaluator(target_schema=Triage) for _ in range(2))
        for i, (gold, answers) in enumerate(TICKETS):
            (a if i < 2 else b).update(Triage(**gold), Triage.from_json(answers))
        a.merge_state(b.get_state())
        merged = a.compute().decision_metrics["fields"]["queue"]
        assert merged["correctness"] == run().decision_metrics["fields"]["queue"]["correctness"]
        assert sorted({r["doc"] for r in merged["records"]}) == [0, 1, 2, 3]

    def test_caller_supplied_accumulator_not_duplicated(self):
        ev = BulkStructuredModelEvaluator(target_schema=Triage, accumulators=[DecisionAccumulator(Triage, n_bins=5)])
        assert [acc.name for acc in ev._accumulators] == ["decision_metrics"]


class TestAcceptOrEscalate:
    def test_per_field(self):
        q = accept_or_escalate(run(), taus=[0.6, 0.7]).fields["queue"]
        r6, r7 = q["rows"]
        assert (r6["coverage"], r6["accuracy_accepted"], r6["errors_escalated"]) == (approx(0.75), approx(2 / 3), 1)
        assert (r7["coverage"], r7["accuracy_accepted"], r7["errors_escalated"]) == (approx(0.5), approx(1.0), 2)

    def test_pooled_matches_the_sketch(self):
        rep = accept_or_escalate(run(), taus=[0.6, 0.7], target_accuracy=0.95, pool=True)
        f = rep.fields["all"]
        r6, r7 = f["rows"]
        assert (r6["coverage"], r6["accuracy_accepted"], r6["errors_escalated"], r6["errors_total"]) == (
            approx(0.75), approx(2 / 3), 2, 5)
        assert (r7["coverage"], r7["accuracy_accepted"], r7["errors_escalated"]) == (approx(0.5), approx(1.0), 5)
        assert f["coverage_at_target"] == approx(0.5)
        assert f["tau_at_target"] == approx(0.70)
        assert "coverage at accuracy" in rep.table()

    def test_requires_decision_metrics(self):
        with pytest.raises(ValueError):
            accept_or_escalate({})


class TestConfidenceIntervals:
    def test_deterministic_and_brackets(self):
        res = run()
        a = confidence_intervals(res, n_boot=200, seed=1)
        b = confidence_intervals(res, n_boot=200, seed=1)
        assert a == b
        acc = a["queue"]["correctness"]["accuracy"]
        point = res.decision_metrics["fields"]["queue"]["correctness"]["accuracy"]
        assert acc["low"] <= point <= acc["high"]
        assert a["settings"]["unit"] == "document"
        assert not any(isinstance(v, float) and math.isnan(v) for v in json.loads(json.dumps(a, default=str)).values())


def test_no_warnings_from_decision_answers():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Triage.from_json(TICKETS[0][1])


class TestNested:
    """Decision fields inside nested models and lists of models."""

    def _models(self):
        from typing import List

        from stickler import ComparableField

        class Msg(StructuredModel):
            text: Optional[str] = ComparableField()
            intent: Optional[str] = Choice(criteria=["ask", "complain"])
            rude: Optional[bool] = Noul()

        class Thread(StructuredModel):
            first: Optional[Msg] = None
            msgs: Optional[List[Msg]] = None

        return Msg, Thread

    @staticmethod
    def c(a):
        return {"type": "choice", "probabilities": {"ask": a, "complain": 1 - a}}

    def test_paths(self):
        from stickler.decisions.primitives import decision_paths

        _, Thread = self._models()
        assert sorted(decision_paths(Thread)) == ["first.intent", "first.rude", "msgs[].intent", "msgs[].rude"]

    def test_records_matched_unmatched_and_missing_items(self):
        Msg, Thread = self._models()
        gold = Thread(
            first=Msg(text="hi", intent="ask", rude=None),
            msgs=[Msg(text="a", intent="ask", rude=False), Msg(text="b", intent="complain", rude=True)],
        )
        pred = Thread.from_json({
            "first": {"text": "hi", "intent": self.c(0.8)},
            "msgs": [
                {"text": "b", "intent": self.c(0.1), "rude": noul(0.7)},
                {"text": "a", "intent": self.c(0.4), "rude": noul(0.2)},
            ],
        })
        assert pred.get_all_confidences()["msgs[1].intent"] == approx(0.6)
        ev = BulkStructuredModelEvaluator(target_schema=Thread)
        ev.update(gold, pred)
        ev.update(gold, Thread.from_json({"first": {"text": "hi", "intent": self.c(0.3)}}))  # no msgs
        fields = ev.compute().decision_metrics["fields"]
        assert "first.rude" not in fields  # never asked
        assert fields["first.intent"]["correctness"]["accuracy"] == approx(0.5)
        intent = fields["msgs[].intent"]
        # Hungarian pairs gold "a" with predicted item 1 and gold "b" with item 0.
        assert [(r["gold"], r["probs"]) for r in intent["records"][:2]] == [
            (0, approx([0.4, 0.6])), (1, approx([0.1, 0.9]))]
        assert intent["correctness"]["coverage"] == {"ratio": approx(0.5), "ok": 2, "missing": 2}

    def test_nested_unasked_has_no_weight(self):
        Msg, Thread = self._models()
        gold = Thread(first=Msg(text="hi", intent="ask", rude=None))
        r = gold.compare_with(Thread.from_json({"first": {"text": "hi", "intent": self.c(0.9), "rude": noul(0.9)}}))
        assert r["field_scores"]["first"] == approx(1.0)


def test_rounding_tolerance_grows_with_options():
    from stickler.decisions.normalize import sum_tolerance

    assert sum_tolerance(3) == approx(0.01)
    assert sum_tolerance(255) == approx(0.1275)
    Triage.from_json({"queue": choice(0.773, 0.121, 0.105)})  # sums to 0.999, as Decider's README prints


class TestValidationAndReporting:
    @pytest.mark.parametrize("gold", [{"queue": "retail2"}, {"severity": 5}, {"severity": -1}])
    def test_invalid_gold_raises_on_construction(self, gold):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Triage(**gold)

    def test_unasked_field_not_in_field_scores(self):
        r = Triage(queue="sales", urgent=True, severity=None).compare_with(
            Triage.from_json({"queue": choice(0.6, 0.3, 0.1)})
        )
        assert "severity" not in r["field_scores"]

    def test_macro_f1_over_seen_options(self):
        from stickler.decisions import DecisionArrays
        from stickler.decisions.metrics import correctness
        from stickler.decisions.primitives import spec_from_question

        spec = spec_from_question("choice", ["a", "b", "c"])
        a = DecisionArrays.from_records(spec, [{"doc": i, "gold": 0, "probs": [0.9, 0.05, 0.05]} for i in range(3)])
        assert correctness(a)["macro_f1"] == approx(1.0)

    def test_exact_sums_are_not_renormalised(self):
        pred = Triage.from_json(TICKETS[1][1])
        assert pred.get_all_confidences()["queue"] == 0.6
