"""A non-numeric string in a numeric field scores 0.0 instead of raising (#310).

``from_json_schema()`` and ``model_from_json()`` build a numeric field as
``float`` or ``int``. A prediction holding ``"N/A"`` there used to fail model
construction, so a bulk run died on that document. These classes now keep a
string the numeric type rejects, as written, and ``NumericComparator`` scores
it. A value the numeric type accepts is stored exactly as before.

A class written by hand, or built with ``ModelFactory.create_model_from_fields()``
without ``scoreable_numbers=True``, keeps pydantic's validation: those are the
caller's own classes.
"""

import json
import warnings
from typing import Optional

import pytest
from pydantic import ValidationError

from stickler import StructuredModel
from stickler.comparators import NumericComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.model_factory import ModelFactory

UNPARSEABLE = ["N/A", "", "see attached", "unknown"]


def schema_model(json_type, *, required=True):
    return StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Invoice",
            "properties": {"amount": {"type": json_type}},
            "required": ["amount"] if required else [],
        }
    )


def config_model(type_name, *, required=True):
    return StructuredModel.model_from_json(
        {
            "model_name": "Invoice",
            "fields": {
                "amount": {
                    "type": type_name,
                    "comparator": "NumericComparator",
                    "required": required,
                }
            },
        }
    )


BUILDERS = {
    "schema-number": lambda **kw: schema_model("number", **kw),
    "schema-integer": lambda **kw: schema_model("integer", **kw),
    "config-float": lambda **kw: config_model("float", **kw),
    "config-int": lambda **kw: config_model("int", **kw),
}


def strict_reference(path, *, required=True):
    """The same field as a class that keeps pydantic's validation."""
    annotation = int if path.endswith(("integer", "int")) else float
    if not required:
        annotation = Optional[annotation]
    field = (
        ComparableField(comparator=NumericComparator())
        if required
        else ComparableField(comparator=NumericComparator(), default=None)
    )
    return ModelFactory.create_model_from_fields(
        model_name="Reference", field_definitions={"amount": (annotation, field)}
    )


@pytest.mark.parametrize("path", list(BUILDERS))
@pytest.mark.parametrize("value", UNPARSEABLE)
def test_a_non_numeric_string_is_kept_and_scores_zero(path, value):
    Model = BUILDERS[path]()
    prediction = Model(amount=value)
    assert prediction.amount == value

    ground_truth = Model(amount=1234)
    result = ground_truth.compare_with(prediction)
    assert result["field_scores"]["amount"] == 0.0
    assert result["overall_score"] == 0.0


@pytest.mark.parametrize("path", list(BUILDERS))
@pytest.mark.parametrize("value", ["1234", "1234.00", 1234, 1234.0, True])
def test_a_value_the_numeric_type_accepts_is_stored_as_before(path, value):
    stored = BUILDERS[path]().model_validate({"amount": value}).amount
    expected = strict_reference(path).model_validate({"amount": value}).amount
    assert stored == expected
    assert type(stored) is type(expected)


@pytest.mark.parametrize("path", ["schema-number", "config-float"])
def test_a_decimal_string_is_still_coerced_to_float(path):
    assert BUILDERS[path]()(amount="1234.50").amount == 1234.5


@pytest.mark.parametrize("path", list(BUILDERS))
def test_arabic_indic_digits_reach_the_comparator(path):
    Model = BUILDERS[path]()
    result = Model(amount=1250).compare_with(Model(amount="١٢٥٠"))
    assert result["field_scores"]["amount"] == 1.0


@pytest.mark.parametrize("path", list(BUILDERS))
@pytest.mark.parametrize(
    "value", [["1"], {"a": 1}, object(), b"N/A", bytearray(b"N/A")]
)
def test_a_value_that_is_not_a_string_still_raises(path, value):
    with pytest.raises(ValidationError):
        BUILDERS[path]()(amount=value)


@pytest.mark.parametrize("path", list(BUILDERS))
def test_an_optional_field_keeps_none_and_the_string(path):
    Model = BUILDERS[path](required=False)
    assert Model(amount=None).amount is None
    assert Model().amount is None
    assert Model(amount="N/A").amount == "N/A"


@pytest.mark.parametrize("path", list(BUILDERS))
def test_model_dump_round_trips_the_kept_string_without_warning(path):
    Model = BUILDERS[path]()
    prediction = Model(amount="N/A")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dumped = prediction.model_dump()
        dumped_json = prediction.model_dump_json()
    assert dumped == {"amount": "N/A"}
    assert '"amount":"N/A"' in dumped_json
    assert Model.model_validate(dumped).amount == "N/A"
    assert Model.model_validate_json(dumped_json).amount == "N/A"


@pytest.mark.parametrize("path", list(BUILDERS))
def test_model_dump_of_a_number_is_unchanged(path):
    dumped = BUILDERS[path]()(amount=12).model_dump()
    assert dumped == strict_reference(path)(amount=12).model_dump()


def test_a_nested_schema_object_keeps_the_string():
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Order",
            "properties": {
                "total": {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                    "required": ["amount"],
                }
            },
            "required": ["total"],
        }
    )
    assert Model(total={"amount": "N/A"}).total.amount == "N/A"


def test_a_nested_config_object_keeps_the_string():
    Model = StructuredModel.model_from_json(
        {
            "model_name": "Order",
            "fields": {
                "total": {
                    "type": "structured_model",
                    "fields": {
                        "amount": {"type": "float", "comparator": "NumericComparator"}
                    },
                }
            },
        }
    )
    assert Model(total={"amount": "N/A"}).total.amount == "N/A"


def test_a_schema_list_of_models_keeps_the_string():
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Statement",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"amount": {"type": "number"}},
                        "required": ["amount"],
                    },
                }
            },
            "required": ["rows"],
        }
    )
    assert Model(rows=[{"amount": "N/A"}]).rows[0].amount == "N/A"


def test_a_hand_written_numeric_field_still_raises():
    class Invoice(StructuredModel):
        amount: float = ComparableField(comparator=NumericComparator())

    with pytest.raises(ValidationError):
        Invoice(amount="N/A")


def test_create_model_from_fields_keeps_validation_by_default():
    with pytest.raises(ValidationError):
        strict_reference("schema-number")(amount="N/A")


def test_create_model_from_fields_can_opt_in():
    Model = ModelFactory.create_model_from_fields(
        model_name="Invoice",
        field_definitions={
            "amount": (float, ComparableField(comparator=NumericComparator()))
        },
        scoreable_numbers=True,
    )
    assert Model(amount="N/A").amount == "N/A"


def test_a_string_field_is_not_affected():
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Invoice",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    )
    with pytest.raises(ValidationError):
        Model(name=["not", "a", "string"])


def test_a_union_with_a_number_arm_is_unchanged():
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Invoice",
            "properties": {
                "value": {"oneOf": [{"type": "integer"}, {"type": "string"}]}
            },
            "required": ["value"],
        }
    )
    assert Model(value=3).value == 3
    assert Model(value="three").value == "three"


def test_the_field_keeps_its_numeric_annotation_and_comparator():
    Model = schema_model("number")
    assert Model.model_fields["amount"].annotation is float
    schema = Model.to_json_schema()
    assert schema["properties"]["amount"]["type"] == "number"


@pytest.mark.parametrize(
    "prop",
    [
        {"type": ["number", "null"]},
        {"anyOf": [{"type": "number"}, {"type": "null"}]},
        {"type": "number", "default": None},
    ],
    ids=["type-list", "anyOf-null", "default-null"],
)
def test_every_nullable_number_spelling_keeps_the_string(prop):
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Invoice",
            "properties": {"amount": prop},
            "required": ["amount"],
        }
    )
    assert Model(amount="N/A").amount == "N/A"
    assert Model(amount=None).amount is None


def test_an_element_of_a_numeric_array_still_raises():
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Readings",
            "properties": {"values": {"type": "array", "items": {"type": "number"}}},
            "required": ["values"],
        }
    )
    assert Model(values=[1, "2"]).values == [1.0, 2.0]
    with pytest.raises(ValidationError):
        Model(values=["N/A"])


def test_a_union_of_a_number_and_a_non_string_type_still_raises():
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Invoice",
            "properties": {
                "value": {"oneOf": [{"type": "integer"}, {"type": "boolean"}]}
            },
            "required": ["value"],
        }
    )
    with pytest.raises(ValidationError):
        Model(value="N/A")


def test_from_pydantic_keeps_pydantic_validation():
    from pydantic import BaseModel

    class Product(BaseModel):
        price: float

    Model = StructuredModel.from_pydantic(Product)
    with pytest.raises(ValidationError):
        Model(price="N/A")


def test_a_constrained_number_keeps_the_string():
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Invoice",
            "properties": {"amount": {"type": "number", "minimum": 0}},
            "required": ["amount"],
        }
    )
    assert Model(amount="N/A").amount == "N/A"
    # #285 already lets a constraint violation reach the comparator.
    assert Model(amount=-5).amount == -5


def test_an_annotated_numeric_field_can_opt_in():
    from typing import Annotated, Optional

    Model = ModelFactory.create_model_from_fields(
        model_name="Invoice",
        field_definitions={
            "a": (
                Annotated[float, "meta"],
                ComparableField(comparator=NumericComparator()),
            ),
            "b": (
                Optional[Annotated[int, "meta"]],
                ComparableField(comparator=NumericComparator(), default=None),
            ),
        },
        scoreable_numbers=True,
    )
    instance = Model(a="N/A", b="n/a")
    assert (instance.a, instance.b) == ("N/A", "n/a")


def test_a_config_list_of_models_keeps_the_string():
    Model = StructuredModel.model_from_json(
        {
            "model_name": "Statement",
            "fields": {
                "rows": {
                    "type": "list_structured_model",
                    "fields": {
                        "amount": {"type": "float", "comparator": "NumericComparator"}
                    },
                }
            },
        }
    )
    assert Model(rows=[{"amount": "N/A"}]).rows[0].amount == "N/A"


@pytest.mark.parametrize("path", list(BUILDERS))
@pytest.mark.parametrize("required", [True, False], ids=["required", "optional"])
@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_model_json_schema_describes_the_field_as_a_strict_class_does(
    path, required, mode
):
    """A serializer typed `-> Any` once exported every numeric field as `{}` in
    serialization mode; the schema must not see the kept-string handling."""
    Model = BUILDERS[path](required=required)
    annotation = int if path.endswith(("integer", "int")) else float
    Reference = ModelFactory.create_model_from_fields(
        model_name="Reference",
        field_definitions={
            "amount": (
                annotation if required else Optional[annotation],
                Model.model_fields["amount"],
            )
        },
    )
    got = Model.model_json_schema(mode=mode)["properties"]["amount"]
    want = Reference.model_json_schema(mode=mode)["properties"]["amount"]
    assert got == want
    assert got == Model.model_json_schema(mode="validation")["properties"]["amount"]


FLOAT_VALUES = [12, 1.5, -0.0, 1e300, float("inf"), float("-inf"), float("nan")]
INT_VALUES = [12, 0, -7, 10**30]
DUMP_CASES = [
    (path, value)
    for path in BUILDERS
    for value in (INT_VALUES if path.endswith(("integer", "int")) else FLOAT_VALUES)
]


@pytest.mark.parametrize(("path", "value"), DUMP_CASES)
def test_model_dump_of_a_number_matches_a_strict_class(path, value):
    got = BUILDERS[path]()(amount=value)
    expected = strict_reference(path)(amount=value)
    assert got.model_dump_json() == expected.model_dump_json()
    for mode in ("python", "json"):
        dumped = got.model_dump(mode=mode)["amount"]
        want = expected.model_dump(mode=mode)["amount"]
        assert type(dumped) is type(want), mode
        assert dumped == want or (dumped != dumped and want != want), mode


def test_a_union_of_two_numeric_types_still_raises():
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Invoice",
            "properties": {"value": {"type": ["number", "integer"]}},
            "required": ["value"],
        }
    )
    assert Model(value=3).value == 3
    with pytest.raises(ValidationError):
        Model(value="N/A")


@pytest.mark.parametrize("value", ["1e3", "1.50", "inf"])
def test_an_integer_field_keeps_a_non_integer_string(value):
    Integer, Number = schema_model("integer"), schema_model("number")
    assert Integer(amount=value).amount == value
    assert Number(amount=value).amount == float(value)


def test_an_exponent_scores_only_in_a_number_field():
    Integer, Number = schema_model("integer"), schema_model("number")
    assert (
        Integer(amount=1000).compare_with(Integer(amount="1e3"))["field_scores"][
            "amount"
        ]
        == 0.0
    )
    assert (
        Number(amount=1000).compare_with(Number(amount="1e3"))["field_scores"]["amount"]
        == 1.0
    )


@pytest.mark.parametrize("prediction", ["N/A", "1234", 1234, None])
def test_a_ground_truth_holding_an_unreadable_string_scores_zero(prediction):
    Model = schema_model("number", required=False)
    result = Model(amount="N/A").compare_with(Model(amount=prediction))
    assert result["field_scores"]["amount"] == 0.0


@pytest.mark.parametrize("prediction", ["", None])
def test_an_empty_ground_truth_is_absent_as_in_any_field(prediction):
    Model = schema_model("number", required=False)
    result = Model(amount="").compare_with(Model(amount=prediction))
    assert result["field_scores"]["amount"] == 1.0


@pytest.mark.parametrize("path", ["schema-number", "config-float"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_evaluate_scores_inf_and_nan_as_a_strict_class_does(path, value):
    """`evaluate()` dumps with `mode="json"`; a by-value serializer turned `inf`
    into `None` there, so the pair failed to rebuild or scored differently."""
    from stickler import evaluate

    Model, Reference = BUILDERS[path](), strict_reference(path)
    got = evaluate(Model(amount=value), Model(amount=value))
    want = evaluate(Reference(amount=value), Reference(amount=value))
    assert repr(got) == repr(want)


def test_a_boolean_field_is_not_affected():
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Flag",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
    )
    with pytest.raises(ValidationError):
        Model(ok="N/A")


def test_strict_validation_keeps_a_numeric_string_as_a_string():
    """Strict mode rejects "1234.5" for a number, so it is kept as written."""
    Model = schema_model("number")
    assert Model.model_validate({"amount": "1234.5"}, strict=True).amount == "1234.5"
    with pytest.raises(ValidationError):
        Model.model_validate({"amount": [1]}, strict=True)


def test_an_integer_field_still_rejects_a_non_integer_number():
    Model = schema_model("integer")
    with pytest.raises(ValidationError):
        Model(amount=12.5)
    with pytest.raises(ValidationError):
        Model.model_validate_json('{"amount": 12.5}')
    assert Model(amount="12.5").amount == "12.5"


def _set_without_validation(Model, how, value):
    """Store `value` the three ways pydantic does not validate it."""
    if how == "assign":
        instance = Model.model_construct(amount=0)
        instance.amount = value
        return instance
    if how == "copy":
        return Model.model_construct(amount=0).model_copy(update={"amount": value})
    return Model.model_construct(amount=value)


def _all_dumps(instance):
    return (
        instance.model_dump(),
        instance.model_dump(mode="json"),
        instance.model_dump_json(),
    )


UNVALIDATED_CASES = [
    (path, required, value)
    for path in BUILDERS
    for required in (True, False)
    for value in (
        [100, True, 10**30, 2**70, 1.5, float("inf"), -0.0]
        if not path.endswith(("integer", "int"))
        else [100, True, 10**30, -7]
    )
]


@pytest.mark.parametrize("how", ["assign", "copy", "construct"])
@pytest.mark.parametrize(("path", "required", "value"), UNVALIDATED_CASES)
def test_a_number_set_without_validation_dumps_as_a_strict_class_does(
    path, required, value, how
):
    """An int in a float field, set by assignment, `model_copy(update=)` or
    `model_construct()`, was rejected by the float arm of the serializer, so it
    dumped as an int and warned in every mode. An `Optional[float]` field needs
    the same int arm as a `float` one."""
    Model = BUILDERS[path](required=required)
    Reference = strict_reference(path, required=required)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _all_dumps(_set_without_validation(Model, how, value))
        want = _all_dumps(_set_without_validation(Reference, how, value))
    assert got == want
    assert [type(dump["amount"]) for dump in got[:2]] == [
        type(dump["amount"]) for dump in want[:2]
    ]


@pytest.mark.parametrize("how", ["assign", "copy", "construct"])
@pytest.mark.parametrize("path", list(BUILDERS))
def test_a_string_set_without_validation_dumps_without_warning(path, how):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dumps = _all_dumps(_set_without_validation(BUILDERS[path](), how, "N/A"))
    assert dumps == ({"amount": "N/A"}, {"amount": "N/A"}, '{"amount":"N/A"}')


@pytest.mark.parametrize("path", ["schema-number", "config-float"])
def test_a_decimal_set_without_validation_differs_only_in_json_mode(path):
    """A residual difference: pydantic's float serializer writes a Decimal that
    skipped validation as its string in `mode="json"`, with a warning. Here the
    float and str arms do not serialize it, so it reaches the last arm, whose
    `float()` converts it. The other two modes match."""
    from decimal import Decimal

    with warnings.catch_warnings():
        # Python mode warns that a Decimal is not a float, on both classes.
        warnings.simplefilter("ignore")
        got = _all_dumps(BUILDERS[path]().model_construct(amount=Decimal("1.5")))
        want = _all_dumps(strict_reference(path).model_construct(amount=Decimal("1.5")))
    assert (got[0], got[2]) == (want[0], want[2])
    assert (got[1]["amount"], want[1]["amount"]) == (1.5, "1.5")


def test_strict_json_validation_keeps_a_numeric_string_as_a_string():
    assert (
        schema_model("number")
        .model_validate_json('{"amount": "1.5"}', strict=True)
        .amount
        == "1.5"
    )
    assert (
        schema_model("integer")
        .model_validate_json('{"amount": "2"}', strict=True)
        .amount
        == "2"
    )


KEPT_STRINGS = [
    ("١٢٥٠", False),
    ("１٢", False),
    ("N/A", False),
    ("1234.5", True),
    ("12", True),
    (" 12 ", True),
    ("1_000.5", True),
    ("nan", True),
    ("infinity", True),
]


@pytest.mark.parametrize("path", ["schema-number", "config-float", "schema-integer"])
@pytest.mark.parametrize(("value", "strict"), KEPT_STRINGS)
def test_a_kept_string_dumps_as_the_string_in_every_mode(path, value, strict):
    """The float field's int arm is a plain `float` call, which pydantic-core
    applies to any value that reaches it, and `float()` reads Unicode digits:
    ahead of `str`, a kept "١٢٥٠" dumped as 1250.0 in JSON modes."""
    Model = BUILDERS[path]()
    instance = Model.model_validate({"amount": value}, strict=strict)
    assert instance.amount == value
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert instance.model_dump()["amount"] == value
        assert instance.model_dump(mode="json")["amount"] == value
        dumped = instance.model_dump_json()
    assert dumped == json.dumps({"amount": value}, ensure_ascii=False).replace(
        ", ", ","
    ).replace(": ", ":")
    assert Model.model_validate_json(dumped, strict=strict).amount == value


@pytest.mark.parametrize("path", ["schema-number", "config-float"])
def test_an_int_too_large_for_a_float_set_without_validation(path):
    """The second residual difference: pydantic's float serializer raises
    OverflowError on such an int under `model_dump(mode="json")`; the int arm's
    `float()` fails too, so the union falls back to the value, with a warning."""
    huge = 10**400
    got = BUILDERS[path]().model_construct(amount=huge)
    want = strict_reference(path).model_construct(amount=huge)
    assert got.model_dump() == want.model_dump()
    with pytest.raises(OverflowError):
        want.model_dump(mode="json")
    with pytest.warns(UserWarning):
        assert got.model_dump(mode="json")["amount"] == huge


def test_a_mixed_numeric_enum_still_raises():
    Model = StructuredModel.from_json_schema(
        {
            "type": "object",
            "title": "Invoice",
            "properties": {"value": {"type": "number", "enum": [1, 2.5]}},
            "required": ["value"],
        }
    )
    with pytest.raises(ValidationError):
        Model(value="N/A")


@pytest.mark.parametrize(
    ("ground_truth", "prediction"),
    [("$1,250", 1250), ("١٢٥٠", 1250), ("1,250", "1,250")],
)
def test_a_ground_truth_the_comparator_can_read_still_matches(ground_truth, prediction):
    Model = schema_model("number")
    result = Model(amount=ground_truth).compare_with(Model(amount=prediction))
    assert result["field_scores"]["amount"] == 1.0
