"""Decision fields: Choice, Score and Noul.

Each is a ``ComparableField`` that scores the predicted answer by exact match
and carries a ``DecisionSpec``: the question's type and, for Choice and Score,
its options or ordered levels. The spec comes from the question, never from
the answer, so the model under test cannot change how it is scored.
"""

import weakref
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import AfterValidator

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField

DecisionKind = Literal["choice", "score", "noul"]

# Attribute on a field's json_schema_extra callable that holds its DecisionSpec.
_SPEC_ATTR = "_decision_spec"
# Key written into the field's JSON schema, so an exported model records it.
SCHEMA_KEY = "x-stickler-decision"


@dataclass(frozen=True)
class DecisionSpec:
    """What a decision field asks.

    Attributes:
        kind: ``"choice"``, ``"score"`` or ``"noul"``.
        labels: The answer labels in scoring order. Choice: the option keys.
            Score: the level names, lowest first. Noul: ``("no", "yes")``.
        values: The Python value each label stands for, in the same order.
            Choice: the option keys. Score: the level numbers ``0..k-1``.
            Noul: ``(False, True)``.
        criteria: The ``criteria`` as given, kept for export and reports.
    """

    kind: DecisionKind
    labels: Tuple[str, ...]
    values: Tuple[Any, ...]
    criteria: Any = None

    @property
    def k(self) -> int:
        """Number of answer labels."""
        return len(self.labels)

    def index_of_value(self, value: Any) -> int:
        """Return the label index for a gold or predicted Python value.

        Raises:
            ValueError: If ``value`` is not one of this question's answers.
        """
        if self.kind == "noul":
            if isinstance(value, bool):
                return int(value)
            raise ValueError(f"A Noul answer must be True or False, got {value!r}")
        if self.kind == "score":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"A Score gold value must be a level number, got {value!r}")
            if not 0 <= value < self.k:
                raise ValueError(f"Level {value} is outside 0..{self.k - 1}")
            return value
        try:
            return self.values.index(value)
        except ValueError:
            raise ValueError(
                f"{value!r} is not an option of this Choice question; options are {list(self.labels)}"
            ) from None

    def to_schema(self) -> Dict[str, Any]:
        """The JSON-serialisable form written under ``x-stickler-decision``."""
        out: Dict[str, Any] = {"type": self.kind}
        if self.criteria is not None:
            out["criteria"] = self.criteria
        return out


def _choice_spec(criteria: Union[Dict[str, Any], Sequence[str]]) -> DecisionSpec:
    if isinstance(criteria, dict):
        keys = list(criteria)
    elif isinstance(criteria, (list, tuple)):
        keys = list(criteria)
    else:
        raise TypeError("Choice criteria must be a dict of option -> description, or a list of options")
    if len(keys) < 2:
        raise ValueError("A Choice question needs at least two options")
    if not all(isinstance(k, str) for k in keys):
        raise TypeError("Choice options must be strings")
    if len(set(keys)) != len(keys):
        raise ValueError(f"Choice options must be unique, got {keys}")
    return DecisionSpec("choice", tuple(keys), tuple(keys), criteria)


def _level_name(level: Any, index: int) -> str:
    if isinstance(level, str):
        return level
    if isinstance(level, dict):
        for key in ("name", "label", "what"):
            if isinstance(level.get(key), str):
                return level[key]
    return str(index)


def _score_spec(criteria: Sequence[Any]) -> DecisionSpec:
    if not isinstance(criteria, (list, tuple)):
        raise TypeError("Score criteria must be a list of levels, lowest first")
    if len(criteria) < 2:
        raise ValueError("A Score question needs at least two levels")
    names = tuple(_level_name(level, i) for i, level in enumerate(criteria))
    return DecisionSpec("score", names, tuple(range(len(criteria))), list(criteria))


def _noul_spec(criteria: Any = None) -> DecisionSpec:
    return DecisionSpec("noul", ("no", "yes"), (False, True), criteria)


def spec_from_question(kind: str, criteria: Any = None) -> DecisionSpec:
    """Build a DecisionSpec from a System One question's ``type`` and ``criteria``."""
    if kind == "choice":
        return _choice_spec(criteria)
    if kind == "score":
        return _score_spec(criteria)
    if kind == "noul":
        return _noul_spec(criteria)
    raise ValueError(f"Unknown System One question type {kind!r}; expected choice, score or noul")


def _decision_field(spec: DecisionSpec, weight: float, field_kwargs: Dict[str, Any]):
    if field_kwargs.get("alias") is not None:
        # Answers and records are matched by field name throughout; an alias
        # would silently break both. Name the field after the question id.
        raise TypeError("Decision fields do not support alias; name the field after the question id")
    extra = dict(field_kwargs.pop("json_schema_extra", None) or {})
    extra[SCHEMA_KEY] = spec.to_schema()
    field = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=weight,
        json_schema_extra=extra,
        **field_kwargs,
    )
    setattr(field.json_schema_extra, _SPEC_ATTR, spec)

    def _valid_answer(value: Any) -> Any:
        # A gold or predicted value that is not one of the question's answers
        # is rejected when the model is built, not deep inside a bulk run.
        if value is not None:
            spec.index_of_value(value)
        return value

    field.metadata.append(AfterValidator(_valid_answer))
    return field


def Choice(criteria: Union[Dict[str, Any], Sequence[str]], *, weight: float = 1.0, **field_kwargs):
    """A Choice decision field: one option out of a fixed, unordered set.

    Args:
        criteria: The options, as in a System One Choice question: a dict of
            option key to description (descriptions may be None), or a list
            of option keys.
        weight: The field's weight in ``overall_score``.
        **field_kwargs: Passed to ``ComparableField`` (``description``, ...). ``alias`` is not supported.

    Gold is an option key. The prediction is a System One Choice answer.
    """
    return _decision_field(_choice_spec(criteria), weight, field_kwargs)


def Score(criteria: Sequence[Any], *, weight: float = 1.0, **field_kwargs):
    """A Score decision field: a position on ordered levels.

    Args:
        criteria: The levels, lowest first, as in a System One Score question.
            A level's number is its position.
        weight: The field's weight in ``overall_score``.
        **field_kwargs: Passed to ``ComparableField``.

    Gold is a level number (``int``). The prediction is a System One Score answer.
    """
    return _decision_field(_score_spec(criteria), weight, field_kwargs)


def Noul(criteria: Any = None, *, weight: float = 1.0, **field_kwargs):
    """A Noul decision field: yes or no.

    Args:
        criteria: Optional description of what yes and no mean. Not used for scoring.
        weight: The field's weight in ``overall_score``.
        **field_kwargs: Passed to ``ComparableField``.

    Gold is ``True`` or ``False``. The prediction is a System One Noul answer.
    """
    return _decision_field(_noul_spec(criteria), weight, field_kwargs)


def field_decision_spec(field_info: Any) -> Optional[DecisionSpec]:
    """Return the DecisionSpec of a pydantic FieldInfo, or None for other fields."""
    extra = getattr(field_info, "json_schema_extra", None)
    return getattr(extra, _SPEC_ATTR, None) if extra is not None else None


def decision_fields(model_cls: Type[Any]) -> Dict[str, DecisionSpec]:
    """Map each top-level decision field of a StructuredModel class to its spec."""
    fields = getattr(model_cls, "model_fields", {}) or {}
    out: Dict[str, DecisionSpec] = {}
    for name, info in fields.items():
        spec = field_decision_spec(info)
        if spec is not None:
            out[name] = spec
    return out


def _model_shape(annotation: Any) -> Tuple[Optional[Type[Any]], bool]:
    """``(StructuredModel subclass, is_list)`` for an annotation, or ``(None, False)``.

    Unwraps ``Optional``/``Annotated``; reads ``List[M]`` and ``M``.
    """
    from stickler.structured_object_evaluator.models.optional_annotation import (
        unwrap_annotated,
        unwrap_optional,
    )
    from stickler.structured_object_evaluator.models.structured_model import (
        StructuredModel,
    )

    ann = unwrap_optional(unwrap_annotated(annotation))[0]
    origin = get_origin(ann)
    if origin in (list, List):
        args = get_args(ann)
        inner = unwrap_optional(unwrap_annotated(args[0]))[0] if args else None
        if isinstance(inner, type) and issubclass(inner, StructuredModel):
            return inner, True
        return None, False
    if isinstance(ann, type) and issubclass(ann, StructuredModel):
        return ann, False
    return None, False


def decision_paths(model_cls: Optional[Type[Any]]) -> Dict[str, DecisionSpec]:
    """Map every decision field, at any depth, to its spec.

    Paths use dots for nested models and ``[]`` for list items, matching
    ``field_comparisons`` keys with their indexes removed: ``queue``,
    ``customer.tier``, ``messages[].intent``. Cached per class, because
    ``from_json`` asks on every call.
    """
    if model_cls is None:
        return {}
    return dict(_cached_paths(model_cls))


# Weak keys, so a dynamically built model class can still be garbage collected.
_PATH_CACHE: "weakref.WeakKeyDictionary[Any, Dict[str, DecisionSpec]]" = weakref.WeakKeyDictionary()


def _cached_paths(model_cls: Type[Any]) -> Dict[str, DecisionSpec]:
    try:
        return _PATH_CACHE[model_cls]
    except (KeyError, TypeError):
        pass
    if not getattr(model_cls, "__pydantic_complete__", True):
        # Resolve forward references now, as pydantic would on first
        # validation; a class whose references are still undefined stays as is.
        rebuild = getattr(model_cls, "model_rebuild", None)
        if callable(rebuild):
            rebuild(raise_errors=False)
    paths = _walk_paths(model_cls, "", ())
    # A class with unresolved forward references reads as having no nested
    # models; cache only once pydantic has finished building it.
    if getattr(model_cls, "__pydantic_complete__", False):
        try:
            _PATH_CACHE[model_cls] = paths
        except TypeError:
            pass
    return paths


def _walk_paths(model_cls: Type[Any], _prefix: str, _seen: Tuple[Any, ...]) -> Dict[str, DecisionSpec]:
    if model_cls in _seen:
        return {}
    out: Dict[str, DecisionSpec] = {}
    for name, info in (getattr(model_cls, "model_fields", {}) or {}).items():
        path = f"{_prefix}{name}"
        spec = field_decision_spec(info)
        if spec is not None:
            out[path] = spec
            continue
        child, is_list = _model_shape(info.annotation)
        if child is not None:
            out.update(_walk_paths(child, f"{path}[]." if is_list else f"{path}.", _seen + (model_cls,)))
    return out


def has_decision_fields(model_cls: Optional[Type[Any]]) -> bool:
    """Whether a StructuredModel class declares a decision field at any depth."""
    return model_cls is not None and bool(_cached_paths(model_cls))


__all__: List[str] = [
    "Choice",
    "Score",
    "Noul",
    "DecisionSpec",
    "decision_fields",
    "decision_paths",
    "field_decision_spec",
    "has_decision_fields",
    "spec_from_question",
]
