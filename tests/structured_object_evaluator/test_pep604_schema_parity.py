"""How an optional field is spelled must not change what the library does.

``Optional[T]``, ``Union[T, None]`` and ``T | None`` are the same type. Ten
sites tested ``get_origin(x) is Union``, which is False for the PEP 604
spelling, so a ``T | None`` field took a different path from an identical
``Optional[T]`` field.

Export was the visible half: the nested model, its fields, and its whole
comparison configuration were replaced by ``{"type": "string"}`` -- with no
error and no warning -- so ``from_json_schema(M.to_json_schema())`` rebuilt a
structurally different model. A nested *list* of models collapsed the same way.

Assertions compare the spellings to each other rather than to literal schemas,
except where the absolute value is the claim (a nested field being ``"object"``
and not ``"string"``). Two independent literals can drift apart while both stay
green, which is how this survived #268.
"""

from typing import List, Optional, Union

import pytest

from stickler import ComparableField, StructuredModel


class Nested(StructuredModel):
    a: str = ComparableField(threshold=0.9)


class TypingOptional(StructuredModel):
    n: Optional[Nested] = ComparableField()


class TypingUnion(StructuredModel):
    n: Union[Nested, None] = ComparableField()


class Pep604(StructuredModel):
    n: Nested | None = ComparableField()


class TypingOptionalList(StructuredModel):
    items: Optional[List[Nested]] = ComparableField()


class Pep604List(StructuredModel):
    items: list[Nested] | None = ComparableField()


class TypingOptionalScalar(StructuredModel):
    s: Optional[str] = ComparableField()


class Pep604Scalar(StructuredModel):
    s: str | None = ComparableField()


class TestNestedModelExportParity:
    def test_all_three_spellings_export_equal_schemas(self):
        typing_optional = TypingOptional.to_json_schema()["properties"]["n"]
        typing_union = TypingUnion.to_json_schema()["properties"]["n"]
        pep604 = Pep604.to_json_schema()["properties"]["n"]
        assert typing_optional == typing_union
        assert typing_optional == pep604

    def test_a_nested_model_is_not_exported_as_a_string(self):
        """The defect: the model, its fields and its config became "string"."""
        prop = Pep604.to_json_schema()["properties"]["n"]
        assert prop["type"] == "object"
        assert prop["type"] != "string"

    def test_the_nested_models_own_configuration_survives(self):
        prop = Pep604.to_json_schema()["properties"]["n"]
        assert prop["x-aws-stickler-model-name"] == "Nested"
        assert prop["properties"]["a"]["x-aws-stickler-threshold"] == 0.9


class TestNestedListExportParity:
    def test_optional_list_of_models_exports_equal_schemas(self):
        assert (
            TypingOptionalList.to_json_schema()["properties"]["items"]
            == Pep604List.to_json_schema()["properties"]["items"]
        )

    def test_an_optional_list_of_models_is_not_exported_as_a_string(self):
        """A whole array of models, its item schema and its match threshold."""
        prop = Pep604List.to_json_schema()["properties"]["items"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "object"
        assert prop["items"]["x-aws-stickler-model-name"] == "Nested"


class TestScalarNullabilityParity:
    def test_optional_scalar_exports_equal_schemas(self):
        assert (
            TypingOptionalScalar.to_json_schema()["properties"]["s"]
            == Pep604Scalar.to_json_schema()["properties"]["s"]
        )

    def test_nullability_survives_the_pep604_spelling(self):
        """``is_nullable`` was derived from the same failing unwrap, so a
        ``str | None`` field exported as ``"string"`` rather than
        ``["string", "null"]``.
        """
        prop = Pep604Scalar.to_json_schema()["properties"]["s"]
        assert prop["type"] == ["string", "null"]


class TestRoundTripParity:
    def test_pep604_nested_model_rebuilds_as_a_nested_model(self):
        """It previously rebuilt as ``Optional[str]`` -- a different model."""
        rebuilt = StructuredModel.from_json_schema(Pep604.to_json_schema())
        annotation = rebuilt.model_fields["n"].annotation
        assert annotation is not str
        assert annotation != Optional[str]

    def test_both_spellings_rebuild_to_the_same_field_shape(self):
        from_typing = StructuredModel.from_json_schema(TypingOptional.to_json_schema())
        from_pep604 = StructuredModel.from_json_schema(Pep604.to_json_schema())
        assert set(from_typing.model_fields) == set(from_pep604.model_fields)
        for name in from_typing.model_fields:
            assert str(from_typing.model_fields[name].annotation) == str(
                from_pep604.model_fields[name].annotation
            ), name

    def test_rebuilt_nested_model_keeps_its_nested_field(self):
        rebuilt = StructuredModel.from_json_schema(Pep604.to_json_schema())
        nested_annotation = rebuilt.model_fields["n"].annotation
        # Unwrap the Optional to reach the nested model class.
        from stickler.structured_object_evaluator.models.optional_annotation import (
            unwrap_optional,
        )

        inner, was_optional = unwrap_optional(nested_annotation)
        assert was_optional
        assert issubclass(inner, StructuredModel)
        assert "a" in inner.model_fields


class TestANestedModelIsNeverExportedAsAScalar:
    """The silent `"string"` fallback is replaced by a raise.

    `PYTHON_TYPE_TO_JSON_TYPE.get(field_type, "string")` treated "I don't know
    this type" as "it's a string". For a nested model that yields a schema which
    rebuilds into a different model, so it fails loudly instead.

    Unreachable through `to_json_schema()` now that `_unwrap_optional` handles
    every spelling -- this is a guard against regression, exercised directly.
    """

    def test_passing_a_model_type_to_field_to_property_raises(self):
        from stickler.structured_object_evaluator.models.json_schema_field_converter import (
            JsonSchemaFieldConverter,
        )

        converter = JsonSchemaFieldConverter({})
        field_info = Pep604.model_fields["n"]
        with pytest.raises(ValueError, match="StructuredModel") as excinfo:
            converter.field_to_property(Nested, field_info)
        assert "Nested" in str(excinfo.value)
        assert "to_json_schema" in str(excinfo.value)

    def test_unmodelled_scalar_types_still_export_as_string(self):
        """The raise must not widen to every dict miss.

        `Decimal`, `UUID` and enums legitimately fall back to `"string"`.
        """
        from decimal import Decimal
        from uuid import UUID

        from stickler.structured_object_evaluator.models.json_schema_field_converter import (
            JsonSchemaFieldConverter,
        )

        converter = JsonSchemaFieldConverter({})
        field_info = Pep604Scalar.model_fields["s"]
        for scalar_type in (Decimal, UUID):
            prop = converter.field_to_property(scalar_type, field_info)
            assert prop["type"] == "string"


class TestMultiArmUnionsAreLeftAlone:
    def test_a_two_model_union_is_not_unwrapped_to_an_arbitrary_arm(self):
        """Both spellings must agree, and neither may pick an arm."""
        from stickler.structured_object_evaluator.models.optional_annotation import (
            unwrap_optional,
        )

        class Other(StructuredModel):
            b: str = ComparableField()

        typing_form = Union[Nested, Other, None]
        pep604_form = Nested | Other | None
        assert unwrap_optional(typing_form) == (typing_form, False)
        assert unwrap_optional(pep604_form) == (pep604_form, False)
