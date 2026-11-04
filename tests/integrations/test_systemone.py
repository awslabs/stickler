"""System One integration: model_from_questions, from_systemone, and their contract.

The contract tests pin the answer shapes Strands Decider and Jev publish, so a
change on either side that Stickler cannot read fails here first. Vendored from:

- strands-labs/strands-decider @ 63abfbc0e8df12c90688f6617ec94cf9ae1fb070
  (README.md last changed in a8f61a6, src/strands_decider/schema.py in d942e55).
  The server example is copied verbatim. The CLI examples print rounded values,
  rebuilt here into schema.py's ChoiceAnswer / ScoreAnswer wire fields.
- docs.typesafe.ai/primitives/{choice,score,noul}.md, fetched 2026-11-04 (jev-1.13.0).

The fixture tests check examples/notebooks/fixtures/systemone/ is internally
consistent, whether its responses are synthetic or recorded.
"""

import json
from pathlib import Path
from typing import Optional

import pytest
from pydantic import BaseModel

from stickler.decisions.normalize import AnswerError, normalize_answer
from stickler.decisions.primitives import decision_fields, spec_from_question
from stickler.integrations.systemone import from_systemone, model_from_questions
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "notebooks"
    / "fixtures"
    / "systemone"
)

# Answer fields per type, from schema.py's NoulAnswer, ChoiceAnswer and ScoreAnswer.
DECIDER_ANSWER_FIELDS = {
    "noul": {"type", "noul"},
    "choice": {"type", "choice", "probabilities", "confidence"},
    "score": {"type", "score", "legend", "probabilities", "confidence"},
}

PUBLISHED_EXAMPLES = json.loads(
    r"""
[
  {
    "source": "strands-decider README, Running as a server (verbatim response)",
    "decider": true,
    "question": {"type": "noul", "instructions": "Does this convey urgency?"},
    "answer": {"type": "noul", "noul": 0.8581}
  },
  {
    "source": "strands-decider README, Choice question (v1-2610, --device cpu)",
    "decider": true,
    "question": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": null, "sales": null, "retail": null}},
    "answer": {"type": "choice", "choice": "billing",
               "probabilities": {"billing": 0.773, "sales": 0.121, "retail": 0.106},
               "confidence": 0.659}
  },
  {
    "source": "strands-decider README, Score question (v1-2610, --device cpu)",
    "decider": true,
    "question": {"type": "score", "instructions": "How frustrated is the writer?",
                 "criteria": ["calm", "frustrated", "depressed"]},
    "answer": {"type": "score", "score": 1.03,
               "legend": {"0": "calm", "1": "frustrated", "2": "depressed"},
               "probabilities": {"0": 0.196, "1": 0.574, "2": 0.230},
               "confidence": 0.509}
  },
  {
    "source": "strands-decider README, Gemma 4 install example (E4B, --device cpu)",
    "decider": true,
    "printed_sum_off": true,
    "question": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": null, "sales": null, "retail": null}},
    "answer": {"type": "choice", "choice": "billing",
               "probabilities": {"billing": 0.967, "sales": 0.019, "retail": 0.013},
               "confidence": 0.951}
  },
  {
    "source": "strands-decider README, Gemma 4 install example (E4B, --device cpu)",
    "decider": true,
    "question": {"type": "score", "instructions": "How frustrated is the writer?",
                 "criteria": ["calm", "frustrated", "depressed"]},
    "answer": {"type": "score", "score": 0.97,
               "legend": {"0": "calm", "1": "frustrated", "2": "depressed"},
               "probabilities": {"0": 0.102, "1": 0.823, "2": 0.075},
               "confidence": 0.848}
  },
  {
    "source": "docs.typesafe.ai/primitives/choice.md, response structure",
    "decider": false,
    "question": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"returns": "Exchanges, wrong or damaged items",
                              "shipping": "Delivery status, delays, lost packages",
                              "billing": "Charges, invoices, payment problems"}},
    "answer": {"type": "choice", "choice": "returns", "confidence": 1.0,
               "probabilities": {"shipping": 0.0, "returns": 1.0, "billing": 0.0}}
  },
  {
    "source": "docs.typesafe.ai/primitives/score.md, response structure",
    "decider": false,
    "question": {"type": "score", "instructions": "How severe is the reported issue?",
                 "criteria": ["Cosmetic; no impact to functionality",
                              "Broken or degraded feature, but workaround exists",
                              "Blocking issue; no workaround exists"]},
    "answer": {"type": "score", "score": 1.43, "confidence": 0.35,
               "legend": {"0": "Cosmetic; no impact to functionality",
                          "1": "Broken or degraded feature, but workaround exists",
                          "2": "Blocking issue; no workaround exists"},
               "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43}}
  },
  {
    "source": "docs.typesafe.ai/primitives/noul.md, response structure",
    "decider": false,
    "question": {"type": "noul", "instructions": "Has the customer contacted support about this before?",
                 "criteria": {"true": "Mentions a prior attempt, ticket, or that they have asked before",
                              "false": "No sign of any previous contact"}},
    "answer": {"type": "noul", "noul": 0.93}
  }
]
"""
)

DECIDER_SERVER_RESPONSE = {
    "model": "strands-decider-2B-qwen3.5-v1-2610",
    "answers": {"is_urgent": {"type": "noul", "noul": 0.8581}},
    "usage": {"input_tokens": 86, "output_tokens": 1},
    "latency_ms": 489.01,
}

TRIAGE_QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": None, "sales": None, "retail": None},
    },
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "severity": {
        "type": "score",
        "instructions": "How severe?",
        "criteria": ["low", "medium", "high"],
    },
}


def _example_id(example):
    return f"{example['answer']['type']}:{example['source'][:40]}"


# The CLI prints three decimals, so one rebuilt example sums to 0.999. Rounded
# answers like that must still normalise.
def _normalisable(examples):
    return [pytest.param(e, id=_example_id(e)) for e in examples]


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- model_from_questions -------------------------------------------------


class TestModelFromQuestions:
    def test_dict_questions_build_decision_fields(self):
        Triage = model_from_questions(
            TRIAGE_QUESTIONS, weights={"queue": 2.0}, model_name="Triage"
        )

        assert issubclass(Triage, StructuredModel)
        assert Triage.__name__ == "Triage"
        specs = decision_fields(Triage)
        assert list(specs) == ["queue", "urgent", "severity"]
        assert specs["queue"].kind == "choice"
        assert specs["queue"].labels == ("billing", "sales", "retail")
        assert specs["urgent"].kind == "noul"
        assert specs["urgent"].values == (False, True)
        assert specs["severity"].kind == "score"
        assert specs["severity"].labels == ("low", "medium", "high")
        assert specs["severity"].values == (0, 1, 2)

    def test_field_types_defaults_and_weights(self):
        Triage = model_from_questions(TRIAGE_QUESTIONS, weights={"queue": 2.0})
        fields = Triage.model_fields

        assert fields["queue"].annotation == Optional[str]
        assert fields["severity"].annotation == Optional[int]
        assert fields["urgent"].annotation == Optional[bool]
        assert fields["queue"].description == "Which team should handle this?"
        assert Triage._get_comparison_info("queue").weight == 2.0
        assert Triage._get_comparison_info("urgent").weight == 1.0

        empty = Triage()
        assert (empty.queue, empty.urgent, empty.severity) == (None, None, None)
        gold = Triage(queue="sales", urgent=True, severity=2)
        assert (gold.queue, gold.urgent, gold.severity) == ("sales", True, 2)

    def test_sdk_shaped_questions_match_by_class_name(self):
        class NoulCriteria(BaseModel):
            true: str
            false: str

        class _Question:
            def __init__(self, instructions, criteria=None):
                self.instructions = instructions
                self.criteria = criteria

        class Choice(_Question):
            pass

        class Score(_Question):
            pass

        class Noul(_Question):
            pass

        questions = {
            "queue": Choice("Which team?", {"billing": "Charges", "sales": "Buying"}),
            "severity": Score("How severe?", ["low", "high"]),
            "urgent": Noul(
                "Urgent?", NoulCriteria(true="needs action today", false="can wait")
            ),
        }
        specs = decision_fields(model_from_questions(questions))

        assert {k: s.kind for k, s in specs.items()} == {
            "queue": "choice",
            "severity": "score",
            "urgent": "noul",
        }
        assert specs["queue"].labels == ("billing", "sales")
        assert specs["severity"].labels == ("low", "high")
        assert specs["urgent"].criteria == {
            "true": "needs action today",
            "false": "can wait",
        }

    def test_object_type_attribute_wins_over_class_name(self):
        class ChoiceQuestion(BaseModel):  # Decider's own request model has this shape
            type: str = "choice"
            instructions: str
            criteria: dict

        specs = decision_fields(
            model_from_questions(
                {
                    "team": ChoiceQuestion(
                        instructions="Team?", criteria={"a": None, "b": None}
                    )
                }
            )
        )
        assert specs["team"].kind == "choice"

    def test_same_specs_as_hand_declared_model(self):
        from stickler.decisions import Choice, Noul, Score

        class Hand(StructuredModel):
            queue: Optional[str] = Choice(
                criteria={"billing": None, "sales": None, "retail": None}
            )
            urgent: Optional[bool] = Noul()
            severity: Optional[int] = Score(criteria=["low", "medium", "high"])

        assert decision_fields(
            model_from_questions(TRIAGE_QUESTIONS)
        ) == decision_fields(Hand)

    @pytest.mark.parametrize(
        "questions, match",
        [
            (
                {"where": {"type": "box", "instructions": "Where?"}},
                "unknown type 'box'",
            ),
            ({"q": {"instructions": "No type"}}, "has no 'type'"),
            (
                {
                    "q": {
                        "type": "choice",
                        "instructions": "One option",
                        "criteria": {"a": None},
                    }
                },
                "at least two",
            ),
            ({"q": {"type": "score", "instructions": "No levels"}}, "Question 'q'"),
            (
                {"my-q": {"type": "noul", "instructions": "?"}},
                "not a valid Python identifier",
            ),
            (
                {"2fa": {"type": "noul", "instructions": "?"}},
                "not a valid Python identifier",
            ),
            (
                {"class": {"type": "noul", "instructions": "?"}},
                "not a valid Python identifier",
            ),
            ({"_hidden": {"type": "noul", "instructions": "?"}}, "underscore"),
            ({"extra_fields": {"type": "noul", "instructions": "?"}}, "collides"),
            ({"compare": {"type": "noul", "instructions": "?"}}, "collides"),
            ({}, "non-empty"),
        ],
    )
    def test_invalid_questions_raise(self, questions, match):
        with pytest.raises(ValueError, match=match):
            model_from_questions(questions)

    def test_weight_for_unknown_question_raises(self):
        with pytest.raises(
            ValueError, match=r"weights name questions that do not exist: \['quuee'\]"
        ):
            model_from_questions(TRIAGE_QUESTIONS, weights={"quuee": 2.0})


# --- from_systemone -------------------------------------------------------


class TestFromSystemOne:
    def test_dict_response_returns_answers_unchanged(self):
        answers = from_systemone(DECIDER_SERVER_RESPONSE)
        assert answers is DECIDER_SERVER_RESPONSE["answers"]

    def test_object_with_answers_attribute(self):
        class Response:
            answers = {"is_urgent": {"type": "noul", "noul": 0.8581}}

        assert from_systemone(Response()) == {
            "is_urgent": {"type": "noul", "noul": 0.8581}
        }

    def test_pydantic_response_is_dumped(self):
        class NoulAnswer(BaseModel):
            type: str = "noul"
            noul: float

        class SystemOneResponse(BaseModel):
            model: str
            answers: dict[str, NoulAnswer]

        response = SystemOneResponse(
            model="m", answers={"is_urgent": NoulAnswer(noul=0.8581)}
        )
        assert from_systemone(response) == {
            "is_urgent": {"type": "noul", "noul": 0.8581}
        }

    @pytest.mark.parametrize("response", [{"model": "m"}, object()])
    def test_response_without_answers_raises(self, response):
        with pytest.raises(ValueError, match="no 'answers'"):
            from_systemone(response)


# --- contract: published answer shapes ------------------------------------


class TestPublishedShapes:
    @pytest.mark.parametrize("example", _normalisable(PUBLISHED_EXAMPLES))
    def test_every_published_example_normalises(self, example):
        question, answer = example["question"], example["answer"]
        spec = spec_from_question(question["type"], question.get("criteria"))
        dist = normalize_answer(spec, answer)

        assert dist.probs.sum() == pytest.approx(1.0)
        if spec.kind == "choice":
            assert spec.labels[dist.top] == answer["choice"]
        elif spec.kind == "score":
            assert dist.top == int(
                max(answer["probabilities"], key=answer["probabilities"].get)
            )
        else:
            assert dist.probs[1] == pytest.approx(answer["noul"])

    @pytest.mark.parametrize(
        "example", [e for e in PUBLISHED_EXAMPLES if e["decider"]], ids=_example_id
    )
    def test_decider_examples_carry_schema_fields(self, example):
        answer = example["answer"]
        assert set(answer) == DECIDER_ANSWER_FIELDS[answer["type"]]

    def test_server_response_round_trips_through_from_systemone(self):
        spec = spec_from_question("noul")
        dist = normalize_answer(
            spec, from_systemone(DECIDER_SERVER_RESPONSE)["is_urgent"]
        )
        assert dist.top == 1
        assert dist.p_top == pytest.approx(0.8581)

    def test_unknown_extra_field_still_normalises(self):
        spec = spec_from_question(
            "choice", {"billing": None, "sales": None, "retail": None}
        )
        answer = {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.773, "sales": 0.121, "retail": 0.106},
            "confidence": 0.659,
            "rationale_tokens": 12,
            "calibrated": True,
        }
        assert normalize_answer(spec, answer).p_top == pytest.approx(0.773)

    def test_score_keys_may_be_level_text(self):
        spec = spec_from_question("score", ["calm", "frustrated", "depressed"])
        answer = {
            "type": "score",
            "probabilities": {"calm": 0.196, "frustrated": 0.574, "depressed": 0.230},
        }
        assert normalize_answer(spec, answer).top == 1

    def test_wrong_type_raises(self):
        spec = spec_from_question("choice", {"billing": None, "sales": None})
        with pytest.raises(AnswerError, match="answer's type is 'noul'"):
            normalize_answer(spec, {"type": "noul", "noul": 0.9})

    def test_missing_option_raises(self):
        spec = spec_from_question(
            "choice", {"billing": None, "sales": None, "retail": None}
        )
        answer = {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "sales": 0.1},
        }
        with pytest.raises(AnswerError, match="missing for \\['retail'\\]"):
            normalize_answer(spec, answer)

    def test_unknown_option_raises(self):
        spec = spec_from_question("choice", {"billing": None, "sales": None})
        answer = {
            "type": "choice",
            "choice": "retail",
            "probabilities": {"billing": 0.1, "sales": 0.1, "retail": 0.8},
        }
        with pytest.raises(AnswerError, match="'retail' is not an option"):
            normalize_answer(spec, answer)

    def test_unknown_score_level_raises(self):
        spec = spec_from_question("score", ["calm", "frustrated", "depressed"])
        answer = {
            "type": "score",
            "probabilities": {"0": 0.2, "1": 0.3, "2": 0.4, "3": 0.1},
        }
        with pytest.raises(AnswerError, match="outside 0..2"):
            normalize_answer(spec, answer)


# --- fixture validity -----------------------------------------------------


@pytest.fixture(scope="module")
def fixture_data():
    questions = json.loads((FIXTURES / "questions.json").read_text())
    tickets = _read_jsonl(FIXTURES / "tickets.jsonl")
    responses = _read_jsonl(FIXTURES / "responses.jsonl")
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    return questions, tickets, responses, provenance


class TestFixture:
    def test_questions_build_a_model(self, fixture_data):
        questions, *_ = fixture_data
        specs = decision_fields(model_from_questions(questions))
        assert {k: s.kind for k, s in specs.items()} == {
            "queue": "choice",
            "priority": "choice",
            "ticket_type": "choice",
            "frustration": "score",
            "needs_refund": "noul",
            "urgent": "noul",
        }
        assert specs["frustration"].k == 4

    def test_every_ticket_has_one_response(self, fixture_data):
        _, tickets, responses, _ = fixture_data
        ticket_ids = [t["id"] for t in tickets]
        assert len(set(ticket_ids)) == len(ticket_ids)
        assert [r["id"] for r in responses] == ticket_ids

    def test_gold_is_a_valid_label_and_null_only_when_unasked(self, fixture_data):
        questions, tickets, _, _ = fixture_data
        specs = decision_fields(model_from_questions(questions))
        Model = model_from_questions(questions)
        for t in tickets:
            assert set(t["gold"]) == set(questions), t["id"]
            assert set(t["asked"]) <= set(questions), t["id"]
            for qid, gold in t["gold"].items():
                if qid in t["asked"]:
                    specs[qid].index_of_value(gold)  # raises on an invalid label
                else:
                    assert gold is None, (
                        f"{t['id']}.{qid} is unasked but has gold {gold!r}"
                    )
            Model(**t["gold"])

    def test_some_tickets_leave_questions_unasked(self, fixture_data):
        questions, tickets, _, _ = fixture_data
        assert sum(len(t["asked"]) < len(questions) for t in tickets) >= 5

    def test_only_asked_questions_are_answered(self, fixture_data):
        _, tickets, responses, _ = fixture_data
        for t, r in zip(tickets, responses):
            answered = set(from_systemone(r["response"]))
            assert answered <= set(t["asked"]), (
                f"{t['id']} answers unasked {answered - set(t['asked'])}"
            )

    def test_every_answer_normalises_against_its_question(self, fixture_data):
        questions, tickets, responses, _ = fixture_data
        specs = decision_fields(model_from_questions(questions))
        for r in responses:
            for qid, answer in from_systemone(r["response"]).items():
                normalize_answer(specs[qid], answer, field=f"{r['id']}.{qid}")

    def test_provenance_matches_responses(self, fixture_data):
        _, _, responses, provenance = fixture_data
        models = {r["response"]["model"] for r in responses}
        assert models == {provenance["response_model"]}
        if provenance["hand_authored"]:
            assert provenance["synthetic"] is True
            assert provenance["recorded"] is None
            assert models == {"synthetic-placeholder"}
        else:
            assert provenance["recorded"]
            assert "synthetic-placeholder" not in models
