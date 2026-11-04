"""System One integration: evaluate Strands Decider, OpenJev and Jev answers.

The questions sent to a System One model are the evaluation schema.
``model_from_questions`` turns them into a StructuredModel with one decision
field per question, and ``from_systemone`` hands back a response's answers for
that model to read::

    Triage = model_from_questions(questions, weights={"queue": 2.0})
    pred = Triage.from_json(from_systemone(response))

Questions are matched by shape, so plain wire dicts and ``typesafe_sdk``
objects both work. Nothing here imports ``typesafe_sdk`` or ``strands_decider``.
"""

import keyword
from typing import Any, Dict, Mapping, Optional, Type

from pydantic import create_model

from stickler.decisions.primitives import Choice, Noul, Score, spec_from_question
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

# Question type -> (field annotation, field constructor).
_FIELD_KINDS = {
    "choice": (Optional[str], Choice),
    "score": (Optional[int], Score),
    "noul": (Optional[bool], Noul),
}


def _plain(value: Any) -> Any:
    """Dump SDK (pydantic) objects to plain data, recursively."""
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _plain(value.model_dump())
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _get(question: Any, name: str) -> Any:
    if isinstance(question, Mapping):
        return question.get(name)
    return getattr(question, name, None)


def _question_type(qid: str, question: Any) -> str:
    """Read a question's type: the dict's ``type``, an object's ``type``, or its class name."""
    kind = _get(question, "type")
    if kind is None and not isinstance(question, Mapping):
        kind = type(question).__name__
    kind = getattr(kind, "value", kind)  # a str Enum
    if not isinstance(kind, str):
        raise ValueError(
            f"Question {qid!r} has no 'type'; expected choice, score or noul"
        )
    kind = kind.lower()
    if kind not in _FIELD_KINDS:
        raise ValueError(
            f"Question {qid!r} has unknown type {kind!r}; expected choice, score or noul"
        )
    return kind


def _check_id(qid: Any) -> None:
    if not isinstance(qid, str) or not qid.isidentifier() or keyword.iskeyword(qid):
        raise ValueError(
            f"Question id {qid!r} is not a valid Python identifier, so it cannot be a field name"
        )
    if qid.startswith("_"):
        raise ValueError(
            f"Question id {qid!r} starts with an underscore, which pydantic treats as private"
        )
    if qid == "extra_fields" or hasattr(StructuredModel, qid):
        raise ValueError(
            f"Question id {qid!r} collides with a StructuredModel attribute; rename the question"
        )


def model_from_questions(
    questions: Mapping[str, Any],
    *,
    weights: Optional[Mapping[str, float]] = None,
    model_name: str = "SystemOneModel",
) -> Type[StructuredModel]:
    """Build a StructuredModel from a System One ``questions`` dict.

    Each question becomes one field named by its id: a Choice is a ``str``
    field, a Score an ``int`` (the level number) and a Noul a ``bool``. Every
    field is optional and defaults to None, so a question that was not asked
    for a document can be left out of its gold.

    Args:
        questions: Question id to question, as passed to ``system_one(...)``.
            A question may be a wire dict (``{"type", "instructions",
            "criteria"}``) or an object with ``instructions`` and ``criteria``
            attributes, such as a ``typesafe_sdk`` Choice, Score or Noul. An
            object's type is its ``type`` attribute, else its class name.
        weights: Optional weight per question id, used in ``overall_score``.
            Questions not named weigh 1.0.
        model_name: Name of the generated class.

    Returns:
        A StructuredModel subclass with one decision field per question.

    Raises:
        ValueError: For an unknown question type, invalid criteria, an id that
            is not a valid field name or collides with a StructuredModel
            attribute (such as ``extra_fields``), or a weight for a question
            that does not exist.
    """
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError(
            "questions must be a non-empty dict of question id to question"
        )
    weights = dict(weights or {})
    unknown = sorted(set(weights) - set(questions), key=str)
    if unknown:
        raise ValueError(
            f"weights name questions that do not exist: {unknown}; questions are {list(questions)}"
        )

    fields: Dict[str, Any] = {}
    for qid, question in questions.items():
        _check_id(qid)
        kind = _question_type(qid, question)
        criteria = _plain(_get(question, "criteria"))
        try:
            spec_from_question(kind, criteria)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Question {qid!r}: {exc}") from exc

        field_kwargs: Dict[str, Any] = {
            "default": None,
            "weight": float(weights.get(qid, 1.0)),
        }
        instructions = _get(question, "instructions")
        if isinstance(instructions, str):
            field_kwargs["description"] = instructions
        annotation, make_field = _FIELD_KINDS[kind]
        if kind == "noul":
            fields[qid] = (annotation, make_field(criteria, **field_kwargs))
        else:
            fields[qid] = (annotation, make_field(criteria=criteria, **field_kwargs))

    return create_model(
        model_name,
        __base__=StructuredModel,
        __doc__=f"Built by model_from_questions from {len(fields)} System One question(s).",
        **fields,
    )


def from_systemone(response: Any) -> Dict[str, Any]:
    """Return the ``answers`` of a System One response, untouched.

    This is a pass-through, not a translation: each answer keeps its wire
    shape, and the decision fields read it against the question.

    Args:
        response: A response dict (``{"model", "answers", "usage", ...}``),
            or an object with an ``answers`` attribute such as an SDK response.
            Pydantic answers are dumped with ``model_dump()``.

    Returns:
        Question id to answer. For a dict response, ``response["answers"]``
        itself.

    Raises:
        ValueError: If the response has no answers.
    """
    if isinstance(response, Mapping):
        if "answers" not in response:
            raise ValueError("System One response has no 'answers'")
        return response["answers"]
    if not hasattr(response, "answers"):
        raise ValueError(
            f"System One response {type(response).__name__} has no 'answers'"
        )
    answers = response.answers
    if hasattr(answers, "model_dump"):
        answers = answers.model_dump()
    if not isinstance(answers, Mapping):
        raise ValueError(
            f"System One 'answers' must be a dict, got {type(answers).__name__}"
        )
    return {
        qid: a.model_dump() if hasattr(a, "model_dump") else a
        for qid, a in answers.items()
    }


__all__ = ["model_from_questions", "from_systemone"]
