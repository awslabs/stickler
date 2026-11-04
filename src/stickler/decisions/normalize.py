"""Read a System One answer into one probability distribution.

Every decision metric reads the same normalised form, so the confidence rule
lives in one place: confidence is the probability of the predicted answer.
An answer's own ``confidence`` field measures how peaked the distribution is,
not P(correct), so it is never read. Neither are ``choice``, ``score`` or
``legend``: the predicted answer is the argmax of ``probabilities``, and level
order comes from the question.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from .primitives import DecisionSpec

# Probabilities that sum to within this of 1 are renormalised silently. Servers
# round each probability (Decider to 4 decimals, its README to 3), so the
# allowance grows with the number of options. Anything further off is an
# error: it means a malformed answer, not rounding.
SUM_TOLERANCE = 0.01
SUM_TOLERANCE_PER_OPTION = 5e-4  # worst case of rounding to 3 decimals


def sum_tolerance(k: int) -> float:
    """How far from 1 the probabilities of a k-option answer may sum."""
    return max(SUM_TOLERANCE, k * SUM_TOLERANCE_PER_OPTION)

ANSWER_TYPES = ("choice", "score", "noul")


class AnswerError(ValueError):
    """A System One answer that cannot be scored against its question."""


@dataclass(frozen=True)
class Distribution:
    """A normalised answer.

    Attributes:
        probs: Probabilities in the question's label order, summing to 1.
        top: Index of the predicted answer (the argmax; ties go to the
            earliest label in question order).
        p_top: Probability of the predicted answer: the confidence every
            calibration metric uses.
    """

    probs: np.ndarray
    top: int
    p_top: float


def is_systemone_answer(value: Any) -> bool:
    """Whether a value looks like a System One answer object."""
    return isinstance(value, Mapping) and value.get("type") in ANSWER_TYPES


def _probability(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnswerError(f"{where}: probability must be a number, got {value!r}")
    p = float(value)
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise AnswerError(f"{where}: probability must be between 0 and 1, got {value!r}")
    return p


def _score_key_index(key: str, spec: DecisionSpec, where: str) -> int:
    if key.isdigit():
        i = int(key)
        if i >= spec.k:
            raise AnswerError(f"{where}: level {key!r} is outside 0..{spec.k - 1}")
        return i
    matches = [i for i, name in enumerate(spec.labels) if name == key]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise AnswerError(
            f"{where}: {key!r} is not a level number or level name; levels are {list(spec.labels)}"
        )
    raise AnswerError(f"{where}: level name {key!r} is ambiguous")


def normalize_answer(spec: DecisionSpec, answer: Mapping[str, Any], field: str = "answer") -> Distribution:
    """Normalise a System One answer against its question.

    Args:
        spec: The question the answer replies to.
        answer: The System One answer object, as returned.
        field: The field name, used in error messages.

    Returns:
        The answer's Distribution in the question's label order.

    Raises:
        AnswerError: If the answer's ``type`` disagrees with the question, an
            option key is missing or unknown, a probability is out of range, or
            the probabilities do not sum to 1.
    """
    where = f"field {field!r}"
    kind = answer.get("type") if isinstance(answer, Mapping) else None
    if kind != spec.kind:
        raise AnswerError(f"{where}: question is {spec.kind!r} but the answer's type is {kind!r}")

    if spec.kind == "noul":
        if "noul" not in answer:
            raise AnswerError(f"{where}: a noul answer needs a 'noul' value")
        p_yes = _probability(answer["noul"], where)
        probs = np.array([1.0 - p_yes, p_yes])
    else:
        raw = answer.get("probabilities")
        if not isinstance(raw, Mapping) or not raw:
            raise AnswerError(f"{where}: a {spec.kind} answer needs a 'probabilities' object")
        probs = np.full(spec.k, np.nan)
        for key, value in raw.items():
            if spec.kind == "choice":
                if key not in spec.labels:
                    raise AnswerError(
                        f"{where}: {key!r} is not an option of this question; options are {list(spec.labels)}"
                    )
                i = spec.labels.index(key)
            else:
                i = _score_key_index(str(key), spec, where)
            if not np.isnan(probs[i]):
                raise AnswerError(f"{where}: {key!r} gives a probability for the same answer twice")
            probs[i] = _probability(value, where)
        missing = [spec.labels[i] for i in np.flatnonzero(np.isnan(probs))]
        if missing:
            raise AnswerError(f"{where}: probabilities are missing for {missing}")

    total = float(probs.sum())
    if abs(total - 1.0) > sum_tolerance(len(probs)):
        raise AnswerError(f"{where}: probabilities sum to {total:.6f}, not 1")
    if abs(total - 1.0) > 1e-9:  # leave float noise in an exact answer alone
        probs = probs / total
    top = noul_or_argmax(spec.kind, probs)
    return Distribution(probs=probs, top=top, p_top=float(probs[top]))


def noul_or_argmax(kind: str, probs: np.ndarray) -> int:
    """The predicted label index: yes when P(yes) >= 0.5 for Noul, else the argmax.

    Ties in the argmax go to the earliest label in question order.
    """
    if kind == "noul":
        return int(probs[1] >= 0.5)
    return int(np.argmax(probs))


def answer_to_rich_value(spec: DecisionSpec, answer: Mapping[str, Any], field: str) -> Dict[str, Any]:
    """Turn an answer into a rich value: the predicted value plus its confidence."""
    dist = normalize_answer(spec, answer, field)
    return {"_value": spec.values[dist.top], "_confidence": dist.p_top}


def answers_to_rich_values(model_cls: Any, json_data: Any, _path: str = "") -> Any:
    """Replace each System One answer on a decision field with its rich value.

    Walks nested models and lists of models, so a decision field at any depth
    is read. Every other value is returned as given; the input is not modified.

    Raises:
        AnswerError: If an answer does not fit its question.
    """
    from .primitives import _model_shape, field_decision_spec, has_decision_fields

    if not isinstance(json_data, Mapping) or not has_decision_fields(model_cls):
        return json_data
    out = dict(json_data)
    for name, info in model_cls.model_fields.items():
        if name not in out:
            continue
        value, path = out[name], f"{_path}{name}"
        spec = field_decision_spec(info)
        if spec is not None:
            if is_systemone_answer(value):
                out[name] = answer_to_rich_value(spec, value, path)
            continue
        child, is_list = _model_shape(info.annotation)
        if child is None:
            continue
        if is_list and isinstance(value, list):
            out[name] = [answers_to_rich_values(child, v, f"{path}[{i}].") for i, v in enumerate(value)]
        elif not is_list and isinstance(value, Mapping):
            out[name] = answers_to_rich_values(child, value, f"{path}.")
    return out
