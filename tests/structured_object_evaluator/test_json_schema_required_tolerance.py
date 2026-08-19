"""``from_json_schema(..., tolerate_missing_fields=True)``.

``ComparableField`` gives every field ``default=None`` so that construction
tolerates partial predictions -- the engine builds instances from prediction
JSON that omits fields, and a prediction that omits a field is the ordinary case
it exists to score. ``from_json_schema`` gave a required field ``default=...``
(pydantic's "no default") instead, so a rebuilt model raised ``ValidationError``
where the hand-written source model would have scored a miss.

Rather than change that unconditionally, the tolerance is opt-in. Two audiences
want different things from the same call: an evaluation model wants to accept
whatever the extraction system produced, while a caller using a schema's
``required`` list as a contract wants it enforced. The default stays strict --
in particular ``required`` plus nullable still means "must be present, may be
null", which
``test_json_schema_field_converter.py::TestNullableTypeListForm`` and friends
deliberately pin.

With the flag on, a required field gets ``default=None`` while keeping a
non-Optional annotation, which reproduces the source model's field shape
exactly. Keeping the annotation bare is load-bearing:
``_AnnotationDrivenJsonSchema.field_is_required`` reads exactly that sentinel
(non-nullable annotation + ``default=None`` means required), so widening to
``Optional[T]`` would report ``required: None`` and defeat the Strands tool
spec. ``TestRequirednessSurvivesTheRebuild`` pins that invariant.

Group 1 is characterisation: ``widen_to_optional`` is where #105, #127, #149 and
#198 all converge, so those behaviors are pinned first and must not move.
"""

from typing import List, Optional

import pytest
from pydantic import ValidationError

from stickler import ComparableField, StructuredModel
from stickler.structured_object_evaluator.models.json_schema_field_converter import (
    JsonSchemaFieldConverter,
)


def _fields(properties, required):
    """Field definitions as JsonSchemaFieldConverter produces them."""
    schema = {"type": "object", "properties": properties, "required": required}
    return JsonSchemaFieldConverter(schema).convert_properties_to_fields(
        properties, required
    )


def _annotation(properties, required, name):
    return _fields(properties, required)[name][0]


def _field_info(properties, required, name):
    return _fields(properties, required)[name][1]


# ---------------------------------------------------------------------------
# Group 1 -- characterisation. These pass before and after the change.
# ---------------------------------------------------------------------------


class TestWidenToOptionalCharacterisation:
    """The four cases converging on the widening decision. None may move."""

    def test_not_required_field_is_widened_and_defaults_to_none(self):
        """Issue #149."""
        props = {"a": {"type": "string"}}
        assert _annotation(props, [], "a") == Optional[str]
        assert _field_info(props, [], "a").default is None

    def test_required_field_with_explicit_null_default_is_widened(self):
        """An explicit ``"default": null`` genuinely means nullable."""
        props = {"a": {"type": "string", "default": None}}
        assert _annotation(props, ["a"], "a") == Optional[str]
        assert _field_info(props, ["a"], "a").default is None

    def test_explicitly_nullable_type_list_is_widened(self):
        """Issue #127: ``type: ["string", "null"]``."""
        props = {"a": {"type": ["string", "null"]}}
        assert _annotation(props, ["a"], "a") == Optional[str]

    def test_two_branch_nullable_anyof_is_widened(self):
        """Issues #105 / #198."""
        props = {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}}
        assert _annotation(props, ["a"], "a") == Optional[str]

    def test_required_field_with_a_non_null_default_keeps_it(self):
        """A stated default must survive, and must not widen the annotation."""
        props = {"a": {"type": "string", "default": "fallback"}}
        assert _annotation(props, ["a"], "a") is str
        assert _field_info(props, ["a"], "a").default == "fallback"

    def test_required_field_annotation_stays_bare(self):
        """The constraint that forbids the widen-to-Optional shortcut."""
        props = {"a": {"type": "string"}}
        assert _annotation(props, ["a"], "a") is str


# ---------------------------------------------------------------------------
# Group 2 -- the defect.
# ---------------------------------------------------------------------------


class _Source(StructuredModel):
    a: str = ComparableField()
    b: Optional[str] = ComparableField()


class _AllRequired(StructuredModel):
    a: str = ComparableField()
    c: int = ComparableField()


class _Nested(StructuredModel):
    city: str = ComparableField()


class _WithNested(StructuredModel):
    obj: _Nested = ComparableField()
    items: List[str] = ComparableField()
    scalar: str = ComparableField()


class TestRebuiltModelToleratesPartialPredictions:
    def test_a_required_field_may_be_omitted(self):
        rebuilt = StructuredModel.from_json_schema(
            _Source.model_json_schema(), tolerate_missing_fields=True
        )
        instance = rebuilt(b="y")
        assert instance.a is None

    def test_the_source_model_already_tolerated_it(self):
        """Establishes that the rebuilt model was the odd one out."""
        assert _Source(b="y").a is None

    def test_a_model_with_no_optional_field_at_all(self):
        """The shape the CHANGELOG called benign is the one that was broken.

        No ``Optional`` field is involved, so the nullable ``anyOf`` gap #198
        addresses has nothing to do with it.
        """
        rebuilt = StructuredModel.from_json_schema(
            _AllRequired.model_json_schema(), tolerate_missing_fields=True
        )
        instance = rebuilt(a="x")
        assert instance.c is None

    def test_empty_input_constructs(self):
        rebuilt = StructuredModel.from_json_schema(
            _Source.model_json_schema(), tolerate_missing_fields=True
        )
        instance = rebuilt()
        assert instance.a is None
        assert instance.b is None

    def test_the_rebuilt_model_is_not_stricter_than_the_hand_written_one(self):
        rebuilt = StructuredModel.from_json_schema(
            _Source.model_json_schema(), tolerate_missing_fields=True
        )
        for name in ("a", "b"):
            source_field = _Source.model_fields[name]
            rebuilt_field = rebuilt.model_fields[name]
            assert rebuilt_field.is_required() == source_field.is_required(), name
            assert rebuilt_field.default == source_field.default, name


class TestRequirednessSurvivesTheRebuild:
    """Tolerance is a construction concession, not a claim of optionality."""

    def test_a_required_field_is_still_reported_required(self):
        rebuilt = StructuredModel.from_json_schema(
            _Source.model_json_schema(), tolerate_missing_fields=True
        )
        assert "a" in rebuilt.model_json_schema()["required"]

    def test_an_optional_field_is_still_not_reported_required(self):
        rebuilt = StructuredModel.from_json_schema(
            _Source.model_json_schema(), tolerate_missing_fields=True
        )
        assert "b" not in (rebuilt.model_json_schema().get("required") or [])

    def test_a_required_field_is_not_nullable_in_the_config_export(self):
        rebuilt = StructuredModel.from_json_schema(
            _Source.model_json_schema(), tolerate_missing_fields=True
        )
        assert rebuilt.to_json_schema()["properties"]["a"]["type"] == "string"

    def test_requiredness_matches_the_source_model(self):
        rebuilt = StructuredModel.from_json_schema(
            _Source.model_json_schema(), tolerate_missing_fields=True
        )
        assert rebuilt.model_json_schema()["required"] == (
            _Source.model_json_schema()["required"]
        )


class TestToleranceIsUniformAcrossFieldShapes:
    def test_a_required_nested_object_may_be_omitted(self):
        rebuilt = StructuredModel.from_json_schema(
            _WithNested.model_json_schema(), tolerate_missing_fields=True
        )
        instance = rebuilt(scalar="x")
        assert instance.obj is None

    def test_a_required_array_may_be_omitted(self):
        rebuilt = StructuredModel.from_json_schema(
            _WithNested.model_json_schema(), tolerate_missing_fields=True
        )
        instance = rebuilt(scalar="x")
        assert instance.items is None

    def test_a_required_field_inside_a_nested_object_may_be_omitted(self):
        rebuilt = StructuredModel.from_json_schema(
            _WithNested.model_json_schema(), tolerate_missing_fields=True
        )
        instance = rebuilt(obj={})
        assert instance.obj is not None

    def test_every_field_omitted_across_all_shapes(self):
        rebuilt = StructuredModel.from_json_schema(
            _WithNested.model_json_schema(), tolerate_missing_fields=True
        )
        instance = rebuilt()
        assert instance.obj is None
        assert instance.items is None
        assert instance.scalar is None


class TestExplicitDefaultsStillWin:
    def test_a_non_null_default_on_a_required_field_is_used(self):
        props = {"a": {"type": "string", "default": "fallback"}}
        schema = {"type": "object", "properties": props, "required": ["a"]}
        rebuilt = StructuredModel.from_json_schema(schema)
        assert rebuilt().a == "fallback"

    def test_an_explicit_null_default_makes_the_field_nullable(self):
        props = {"a": {"type": "string", "default": None}}
        schema = {"type": "object", "properties": props, "required": ["a"]}
        rebuilt = StructuredModel.from_json_schema(schema)
        assert rebuilt(a=None).a is None


class TestTheSupportedRoundTripIsUnchanged:
    """``to_json_schema()`` marks nothing required, so it already tolerated
    omission. Its field shapes must not move.
    """

    def test_config_preserving_round_trip_still_constructs(self):
        rebuilt = StructuredModel.from_json_schema(_AllRequired.to_json_schema())
        assert rebuilt(a="x").c is None

    def test_config_preserving_round_trip_field_shapes(self):
        rebuilt = StructuredModel.from_json_schema(_AllRequired.to_json_schema())
        for name in ("a", "c"):
            assert rebuilt.model_fields[name].is_required() is False
            assert rebuilt.model_fields[name].default is None


class TestValidationErrorIsNoLongerRaised:
    def test_the_exact_reproduction_from_the_review(self):
        rebuilt = StructuredModel.from_json_schema(
            _Source.model_json_schema(), tolerate_missing_fields=True
        )
        try:
            rebuilt(b="y")
        except ValidationError as exc:  # pragma: no cover - the defect
            pytest.fail(f"rebuilt model rejected a partial prediction: {exc}")


class TestTheDefaultRemainsStrict:
    """Opting out is the default, so nothing existing changes shape."""

    def test_omitting_a_required_field_still_raises_by_default(self):
        rebuilt = StructuredModel.from_json_schema(_Source.model_json_schema())
        with pytest.raises(ValidationError):
            rebuilt(b="y")

    def test_required_and_nullable_still_means_present_but_may_be_null(self):
        """The distinction the flag must not erase when it is off."""
        schema = {
            "type": "object",
            "properties": {"description": {"type": ["string", "null"]}},
            "required": ["description"],
        }
        strict = StructuredModel.from_json_schema(schema)
        assert strict(description=None).description is None
        with pytest.raises(ValidationError):
            strict()

    def test_the_same_schema_tolerates_omission_when_asked(self):
        schema = {
            "type": "object",
            "properties": {"description": {"type": ["string", "null"]}},
            "required": ["description"],
        }
        tolerant = StructuredModel.from_json_schema(
            schema, tolerate_missing_fields=True
        )
        assert tolerant().description is None

    def test_default_field_shapes_are_unchanged(self):
        rebuilt = StructuredModel.from_json_schema(_Source.model_json_schema())
        assert rebuilt.model_fields["a"].is_required() is True
        assert rebuilt.model_fields["b"].is_required() is False

    def test_round_trip_idempotence_is_preserved_by_default(self):
        """``to_json_schema()`` derives ``required`` from ``is_required()``, so
        flipping it unconditionally would have degraded this round trip.
        """

        class Product(StructuredModel):
            name: str = ComparableField(default=...)
            in_stock: bool = ComparableField(default=...)

        first = StructuredModel.from_json_schema(Product.to_json_schema())
        second = StructuredModel.from_json_schema(first.to_json_schema())
        assert first.to_json_schema() == second.to_json_schema()

    def test_nested_and_array_shapes_are_strict_by_default(self):
        rebuilt = StructuredModel.from_json_schema(_WithNested.model_json_schema())
        with pytest.raises(ValidationError):
            rebuilt(scalar="x")


class TestToleranceReachesNestedModels:
    """The flag must propagate through nested objects and array items."""

    def test_a_nested_models_own_required_field_may_be_omitted(self):
        rebuilt = StructuredModel.from_json_schema(
            _WithNested.model_json_schema(), tolerate_missing_fields=True
        )
        instance = rebuilt(obj={})
        assert instance.obj is not None
        assert instance.obj.city is None

    def test_an_array_items_required_field_may_be_omitted(self):
        class Item(StructuredModel):
            code: str = ComparableField()
            label: str = ComparableField()

        class Doc(StructuredModel):
            items: List[Item] = ComparableField()

        rebuilt = StructuredModel.from_json_schema(
            Doc.model_json_schema(), tolerate_missing_fields=True
        )
        instance = rebuilt(items=[{"code": "A"}])
        assert instance.items[0].label is None
