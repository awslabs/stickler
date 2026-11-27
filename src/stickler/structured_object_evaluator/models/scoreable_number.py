"""Keep a non-numeric string in a numeric field scoreable.

A prediction that writes ``"N/A"``, ``""`` or ``"see attached"`` into a field
typed as a number is an ordinary miss: the field should score 0.0, not abort
the run at model construction. ``from_json_schema()`` and ``model_from_json()``
build their numeric fields as ``float`` or ``int``, so pydantic used to reject
those strings with a ``ValidationError``.

The annotation is left as it is, because readers across the package choose a
comparator, decide nullability and export the field from it. Instead the
generated class gets a wrap validator on its numeric fields: a value pydantic
accepts is stored exactly as before (``"1234.50"`` still becomes ``1234.5``),
and a string it rejects is stored as written, so ``NumericComparator`` reads it.
That is how ``"١٢٥٠"`` reaches the comparator, which reads it as 1250.

Such a field serializes a number with its own numeric serializer and a kept
string as a string, both in pydantic-core, so ``model_dump()`` returns a kept
string without warning that it expected a number. ``FieldInfo.annotation`` is
unchanged; the serializer is ``Annotated`` metadata in ``FieldInfo.metadata``.
A JSON Schema generator reads that serializer as no serializer at all, so
``model_json_schema()`` describes the field as before in both modes.

Only strings fall back. A list or a dict in a numeric field still raises, as
does any value in a numeric field of a class written by hand or built with
``ModelFactory.create_model_from_fields()`` without ``scoreable_numbers=True``:
those classes are the caller's own, and their validation is the caller's to
choose.
"""

from typing import Annotated, Any, Dict, List

from pydantic import ValidationError, field_validator
from pydantic_core import core_schema

from .optional_annotation import is_optional_union, unwrap_annotated, unwrap_optional

__all__ = [
    "numeric_field_names",
    "scoreable_number_validators",
    "serialize_by_value",
]

_NUMERIC_TYPES = (int, float)


def _numeric_type(annotation: Any) -> Any:
    """``int`` or ``float`` when ``annotation`` is one, optionally ``Optional``."""
    annotation = unwrap_annotated(annotation)
    if is_optional_union(annotation):
        annotation, _ = unwrap_optional(annotation)
        annotation = unwrap_annotated(annotation)
    return next((numeric for numeric in _NUMERIC_TYPES if annotation is numeric), None)


def _is_numeric_annotation(annotation: Any) -> bool:
    """Whether ``annotation`` is ``int`` or ``float``, optionally ``Optional``."""
    return _numeric_type(annotation) is not None


def numeric_field_names(field_definitions: Dict[str, tuple]) -> List[str]:
    """The fields of a ``create_model`` definition typed as a number."""
    return [
        name
        for name, definition in field_definitions.items()
        if isinstance(definition, tuple)
        and definition
        and _is_numeric_annotation(definition[0])
    ]


def _keep_unparseable_string(cls, value: Any, handler: Any) -> Any:
    try:
        return handler(value)
    except ValidationError:
        if isinstance(value, str):
            return value
        raise


class _SerializeByValue:
    """``Annotated`` metadata: serialize a number as before, a kept string as one.

    The serializer is a union of the field's own schema and ``str``, so a
    validated number is serialized by the field's numeric serializer (``inf``
    and ``nan`` included, in every mode), and a kept string by pydantic-core's
    string serializer. A by-value serializer is not enough:
    ``model_dump(mode="json")`` would turn ``inf`` into ``None``, which the
    numeric serializer keeps.

    A ``float`` field gets a third arm, last, for a value the first two do not
    serialize: in practice an ``int`` or ``bool`` that reached the field without
    validation, through attribute assignment, ``model_copy(update=)`` or
    ``model_construct()``. The ``float`` arm rejects those, and the plain
    ``float`` serializer converts them in JSON modes, so the arm does the same:
    ``100`` dumps as ``100.0`` there and as ``100`` in Python mode, as before.
    The arm's serializer is a plain function, which pydantic-core calls on any
    value that reaches it, so it must come after ``str``: ahead of it, it would
    turn a kept ``"١٢٥٠"`` into ``1250.0``.
    """

    def __repr__(self) -> str:
        # Shown in FieldInfo.metadata and rebuild_annotation().
        return "_SerializeByValue()"

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        schema = handler(source)
        # Order matters: pydantic-core does not type-check a plain function
        # serializer, so the int arm would take any value that reached it.
        # Behind `str`, it sees only what neither arm before it serializes.
        choices = [dict(schema), core_schema.str_schema()]
        if _numeric_type(source) is float:
            choices.append(
                core_schema.int_schema(
                    serialization=core_schema.plain_serializer_function_ser_schema(
                        float,
                        # Documentation only: the JSON Schema generator does
                        # not read a union serializer, so this changes no schema.
                        return_schema=core_schema.float_schema(),
                        when_used="json",
                    )
                )
            )
        serialization = core_schema.union_schema(choices)
        return {**schema, "serialization": serialization}


_SERIALIZE_BY_VALUE = _SerializeByValue()


def serialize_by_value(annotation: Any) -> Any:
    """``annotation`` with the metadata above, which pydantic stores beside it.

    ``create_model`` strips an outer ``Annotated`` into the field's metadata, so
    ``FieldInfo.annotation`` is ``annotation`` itself, unchanged.
    """
    return Annotated[annotation, _SERIALIZE_BY_VALUE]


def scoreable_number_validators(names: List[str]) -> Dict[str, Any]:
    """The ``__validators__`` that keep a non-numeric string in ``names``."""
    return {
        "_keep_unparseable_numeric_string": field_validator(*names, mode="wrap")(
            _keep_unparseable_string
        ),
    }
