"""Regression tests for the independent review of the decision-model branch."""

import gc
import json
import weakref
from typing import List, Optional

import pytest
from pydantic import create_model

from stickler import ComparableField, StructuredModel
from stickler.decisions import Choice, Noul, confidence_intervals
from stickler.decisions.primitives import decision_paths, has_decision_fields
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)


def noul(p):
    return {"type": "noul", "noul": p}


def choice(a):
    return {"type": "choice", "probabilities": {"ask": a, "complain": round(1 - a, 6)}}


class Msg(StructuredModel):
    text: Optional[str] = ComparableField()
    intent: Optional[str] = Choice(criteria=["ask", "complain"])


class Cust(StructuredModel):
    vip: Optional[bool] = Noul()
    msgs: Optional[List[Msg]] = None


class Doc(StructuredModel):
    cust: Optional[Cust] = None
    msgs: Optional[List[Msg]] = None


class Flat(StructuredModel):
    urgent: Optional[bool] = Noul()
    intent: Optional[str] = Choice(criteria=["ask", "complain"])


def test_forward_reference_is_not_cached_empty():
    class Outer(StructuredModel):
        inner: Optional["Inner"] = None

    has_decision_fields(Outer)  # asked before Inner exists

    class Inner(StructuredModel):
        vip: Optional[bool] = Noul()

    Outer.model_rebuild()
    assert decision_paths(Outer) == {"inner.vip": decision_paths(Inner)["vip"]}
    pred = Outer.from_json({"inner": {"vip": noul(0.2)}})
    assert pred.inner.vip is False
    ev = BulkStructuredModelEvaluator(target_schema=Outer)
    assert "decision_metrics" in [a.name for a in ev._accumulators]


def test_path_cache_does_not_keep_classes_alive():
    refs = []
    for i in range(20):
        cls = create_model(f"M{i}", __base__=StructuredModel, a=(Optional[str], ComparableField()))
        cls.from_json({"a": "x"})
        refs.append(weakref.ref(cls))
        del cls
    gc.collect()
    assert sum(r() is not None for r in refs) == 0


def test_document_with_nothing_asked_scores_one():
    r = Flat().compare_with(Flat.from_json({}))
    assert r["overall_score"] == pytest.approx(1.0)
    assert Flat().compare(Flat.from_json({})) == pytest.approx(1.0)


def test_unasked_adds_no_counts_and_no_confidence_pair():
    gold = Flat(intent="ask")  # urgent not asked
    pred = Flat.from_json({"intent": choice(0.9), "urgent": noul(0.9)})
    r = gold.compare_with(pred, include_confusion_matrix=True, document_field_comparisons=True)
    overall = r["confusion_matrix"]["overall"]
    assert (overall["tp"], overall["fa"], overall["fp"], overall["tn"]) == (1, 0, 0, 0)
    assert [row["expected_key"] for row in r["field_comparisons"]] == ["intent"]


def test_list_inside_missing_nested_object_counts_as_missing():
    gold = Doc(cust=Cust(vip=True, msgs=[Msg(text="a", intent="ask"), Msg(text="b", intent="complain")]))
    ev = BulkStructuredModelEvaluator(target_schema=Doc)
    ev.update(gold, Doc.from_json({}))
    fields = ev.compute().decision_metrics["fields"]
    assert fields["cust.vip"]["correctness"]["coverage"]["missing"] == 1
    assert fields["cust.msgs[].intent"]["correctness"]["coverage"] == {"ratio": 0.0, "ok": 0, "missing": 2}


def test_alias_is_rejected():
    with pytest.raises(TypeError, match="alias"):
        Choice(criteria=["a", "b"], alias="queue.v2")


def test_non_optional_decision_field_is_rejected():
    with pytest.raises(TypeError, match="must be Optional"):

        class Bad(StructuredModel):
            urgent: bool = Noul()


@pytest.mark.parametrize("pred", [{}, {"msgs": [{"text": "a", "intent": choice(0.8)}]}])
def test_jsonl_round_trip_keeps_missing_decisions(tmp_path, pred):
    gold = Doc(msgs=[Msg(text="a", intent="ask"), Msg(text="b", intent="complain")])
    path = tmp_path / "results.jsonl"
    live = BulkStructuredModelEvaluator(target_schema=Doc, individual_results_jsonl=str(path))
    live.update(gold, Doc.from_json(pred), doc_id="d1")
    expected = live.compute().decision_metrics["fields"]["msgs[].intent"]["records"]
    replay = BulkStructuredModelEvaluator(target_schema=Doc)
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        replay.update_from_comparison_result(rec["comparison_result"], doc_id=rec["doc_id"])
    assert replay.compute().decision_metrics["fields"]["msgs[].intent"]["records"] == expected


def test_jsonl_unchanged_for_models_without_decisions(tmp_path):
    class Plain(StructuredModel):
        name: Optional[str] = ComparableField()

    path = tmp_path / "plain.jsonl"
    ev = BulkStructuredModelEvaluator(target_schema=Plain, individual_results_jsonl=str(path))
    assert ev._has_decisions is False
    assert ev._jsonl_default(Plain(name="x")) == str(Plain(name="x"))


def test_confidence_intervals_unknown_field():
    ev = BulkStructuredModelEvaluator(target_schema=Flat)
    ev.update(Flat(intent="ask"), Flat.from_json({"intent": choice(0.9)}))
    with pytest.raises(ValueError, match="Unknown decision fields"):
        confidence_intervals(ev.compute(), fields=["nope"])
