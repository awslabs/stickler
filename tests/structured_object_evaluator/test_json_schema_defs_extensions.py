"""Stickler extensions on a `$defs` / `definitions` entry are read and checked.

A `$ref` stands for the schema it points at, so a field key written on a
definition applies to every property that references it, as if it had been
written on that property. A key written beside the `$ref` (or beside the
`anyOf` that wraps it) is the more specific statement and wins for that property.

Before #340 the extension walk entered each definition with no scope, so a typo
there imported cleanly, and the schema library carried a definition's keys only
into the generated class's config, where nothing read the field-level ones.

See https://github.com/awslabs/stickler/issues/340
"""

import typing
import warnings

import pytest

from stickler import StructuredModel

INNER = {
    "type": "object",
    "properties": {"b": {"type": "string"}, "c": {"type": "string"}},
}


def _with_defs(prop, defs, namespace="$defs"):
    return {
        "type": "object",
        "title": "Parent",
        "properties": {"inner": prop},
        namespace: defs,
    }


def _child_class(model, field="inner"):
    annotation = model.model_fields[field].annotation
    while True:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if not args:
            return annotation
        annotation = args[0]


def _threshold(schema, field="inner"):
    model = StructuredModel.from_json_schema(schema)
    return model._get_comparison_info(field).threshold


DECLARED = dict(INNER, **{"x-aws-stickler-threshold": 0.88})


class TestEverySpellingOfAReferenceReadsIt:
    @pytest.mark.parametrize(
        "schema",
        (
            _with_defs({"$ref": "#/definitions/I"}, {"I": DECLARED}, "definitions"),
            # What pydantic writes for `Optional[Model]`.
            _with_defs(
                {"anyOf": [{"$ref": "#/$defs/I"}, {"type": "null"}]}, {"I": DECLARED}
            ),
            # How draft-07, which ignores keywords beside `$ref`, gives it siblings.
            _with_defs({"allOf": [{"$ref": "#/$defs/I"}]}, {"I": DECLARED}),
            _with_defs(
                {"$ref": "#/$defs/Outer/$defs/I"},
                {"Outer": {"type": "object", "$defs": {"I": DECLARED}}},
            ),
            _with_defs(
                {"$ref": "#/$defs/Alias"},
                {"Alias": {"$ref": "#/$defs/I"}, "I": DECLARED},
            ),
        ),
        ids=("definitions", "nullable-anyOf", "allOf", "nested-defs", "ref-chain"),
    )
    def test_the_declared_threshold_is_read(self, schema):
        assert _threshold(schema) == pytest.approx(0.88)

    def test_a_reference_inside_a_definition_is_read(self):
        """A `$ref` written in a definition's own properties is a property too."""
        model = StructuredModel.from_json_schema(
            _with_defs(
                {"$ref": "#/$defs/Mid"},
                {
                    "Mid": {
                        "type": "object",
                        "properties": {"deep": {"$ref": "#/$defs/I"}},
                    },
                    "I": DECLARED,
                },
            )
        )
        mid = _child_class(model)
        assert mid._get_comparison_info("deep").threshold == pytest.approx(0.88)


class TestTheDefinitionsKeysTakeEffect:
    def test_every_field_key_is_read(self):
        definition = dict(
            INNER,
            **{
                "x-aws-stickler-threshold": 0.88,
                "x-aws-stickler-weight": 3.0,
                "x-aws-stickler-clip-under-threshold": False,
            },
        )
        model = StructuredModel.from_json_schema(
            _with_defs({"$ref": "#/$defs/I"}, {"I": definition})
        )
        info = model._get_comparison_info("inner")
        assert (info.threshold, info.weight, info.clip_under_threshold) == (
            0.88,
            3.0,
            False,
        )

    def test_the_threshold_changes_the_score(self):
        """Four leaves, one wrong, score about 0.78: above 0.7, below a declared 0.8."""
        wide = {
            "type": "object",
            "properties": {f"c{index}": {"type": "string"} for index in range(4)},
            "x-aws-stickler-threshold": 0.8,
        }
        model = StructuredModel.from_json_schema(
            _with_defs({"$ref": "#/$defs/Wide"}, {"Wide": wide})
        )
        child = _child_class(model)
        right = {f"c{index}": "same" for index in range(4)}
        wrong = dict(right, c3="DIFFERENT")
        score = model(inner=child(**right)).compare_with(model(inner=child(**wrong)))[
            "field_scores"
        ]["inner"]
        assert score == pytest.approx(0.0)

    def test_a_scalar_definition_is_read_too(self):
        """The library resolves this `$ref` to a plain `str` and keeps no keys."""
        model = StructuredModel.from_json_schema(
            _with_defs(
                {"$ref": "#/$defs/Code"},
                {
                    "Code": {
                        "type": "string",
                        "x-aws-stickler-comparator": "ExactComparator",
                        "x-aws-stickler-threshold": 0.9,
                    }
                },
            )
        )
        info = model._get_comparison_info("inner")
        assert type(info.comparator).__name__ == "ExactComparator"
        assert info.threshold == pytest.approx(0.9)

    def test_the_class_keys_still_reach_the_class(self):
        """`match_threshold` was already read through `$ref`; it must stay so."""
        definition = dict(
            DECLARED,
            **{
                "x-aws-stickler-match-threshold": 0.9,
                "x-aws-stickler-model-name": "Renamed",
            },
        )
        model = StructuredModel.from_json_schema(
            _with_defs({"$ref": "#/$defs/I"}, {"I": definition})
        )
        child = _child_class(model)
        assert (child.__name__, child.match_threshold) == ("Renamed", 0.9)
        assert model._get_comparison_info("inner").threshold == pytest.approx(0.88)

    def test_it_survives_a_round_trip(self):
        model = StructuredModel.from_json_schema(
            _with_defs({"$ref": "#/$defs/I"}, {"I": DECLARED})
        )
        rebuilt = StructuredModel.from_json_schema(model.to_json_schema())
        assert rebuilt._get_comparison_info("inner").threshold == pytest.approx(0.88)


class TestTheReferencingPropertyWins:
    def test_a_key_beside_the_ref_overrides_the_definition_for_that_property(self):
        schema = {
            "type": "object",
            "title": "Parent",
            "properties": {
                "plain": {"$ref": "#/$defs/I"},
                "tuned": {"$ref": "#/$defs/I", "x-aws-stickler-threshold": 0.6},
                "optional": {
                    "anyOf": [{"$ref": "#/$defs/I"}, {"type": "null"}],
                    "x-aws-stickler-threshold": 0.65,
                },
            },
            "$defs": {"I": DECLARED},
        }
        model = StructuredModel.from_json_schema(schema)
        assert [
            model._get_comparison_info(name).threshold
            for name in ("plain", "tuned", "optional")
        ] == [pytest.approx(0.88), pytest.approx(0.6), pytest.approx(0.65)]

    def test_each_key_is_taken_separately(self):
        """A property that names a weight still inherits the definition's threshold."""
        model = StructuredModel.from_json_schema(
            _with_defs(
                {"$ref": "#/$defs/I", "x-aws-stickler-weight": 2.0},
                {"I": dict(DECLARED, **{"x-aws-stickler-weight": 5.0})},
            )
        )
        info = model._get_comparison_info("inner")
        assert (info.threshold, info.weight) == (pytest.approx(0.88), 2.0)

    def test_in_a_chain_the_nearer_definition_wins(self):
        schema = _with_defs(
            {"$ref": "#/$defs/Alias"},
            {
                "Alias": {"$ref": "#/$defs/I", "x-aws-stickler-threshold": 0.55},
                "I": DECLARED,
            },
        )
        assert _threshold(schema) == pytest.approx(0.55)


NUMERIC = {
    "type": "number",
    "x-aws-stickler-comparator": "NumericComparator",
    "x-aws-stickler-comparator-config": {"absolute_tolerance": 0.5},
}


class TestAComparatorKeepsItsOwnConfig:
    """A config belongs to the comparator it was written for.

    Carrying the definition's `NumericComparator` config onto a property that
    names `ExactComparator` warned that Exact does not accept it, about a config
    nobody wrote at that property.
    """

    @staticmethod
    def _import(prop, definition=NUMERIC):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = StructuredModel.from_json_schema(
                _with_defs(prop, {"N": definition})
            )
        return model._get_comparison_info("inner").comparator, [
            str(w.message) for w in caught
        ]

    def test_the_config_is_carried_with_the_comparator(self):
        comparator, messages = self._import({"$ref": "#/$defs/N"})
        assert type(comparator).__name__ == "NumericComparator"
        assert comparator.absolute_tolerance == 0.5
        assert not messages

    def test_a_property_naming_its_own_comparator_does_not_take_the_config(self):
        comparator, messages = self._import(
            {"$ref": "#/$defs/N", "x-aws-stickler-comparator": "ExactComparator"}
        )
        assert type(comparator).__name__ == "ExactComparator"
        assert not [m for m in messages if "does not accept" in m]

    def test_a_property_stating_only_a_config_keeps_the_definitions_comparator(self):
        """The site's config replaces the definition's, for the same comparator.

        A string definition, so the definition's `ExactComparator` differs from
        the type default (`LevenshteinComparator`) the config would otherwise
        land on.
        """
        comparator, messages = self._import(
            {
                "$ref": "#/$defs/N",
                "x-aws-stickler-comparator-config": {"case_sensitive": False},
            },
            definition={
                "type": "string",
                "x-aws-stickler-comparator": "ExactComparator",
                "x-aws-stickler-comparator-config": {"case_sensitive": True},
            },
        )
        assert type(comparator).__name__ == "ExactComparator"
        assert comparator.case_sensitive is False
        assert not messages


class TestTyposInsideDefsAreRejected:
    @pytest.mark.parametrize(
        "schema, path",
        (
            (
                _with_defs(
                    {"$ref": "#/definitions/I"},
                    {"I": dict(INNER, **{"x-aws-stickler-thresold": 0.88})},
                    "definitions",
                ),
                "definitions.I",
            ),
            (
                _with_defs(
                    {"$ref": "#/$defs/Outer/$defs/I"},
                    {
                        "Outer": {
                            "type": "object",
                            "$defs": {
                                "I": dict(INNER, **{"x-aws-stickler-thresold": 0.88})
                            },
                        }
                    },
                ),
                "$defs.Outer.$defs.I",
            ),
            (
                _with_defs(
                    {"$ref": "#/$defs/Code"},
                    {"Code": {"type": "string", "x-aws-stickler-thresold": 0.9}},
                ),
                "$defs.Code",
            ),
            (
                _with_defs(
                    {"type": "string"},
                    {"Unused": dict(INNER, **{"x-aws-stickler-thresold": 0.88})},
                ),
                "$defs.Unused",
            ),
        ),
        ids=("definitions", "nested-defs", "scalar-definition", "unreferenced"),
    )
    def test_a_misspelled_key_raises_naming_the_definition(self, schema, path):
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(schema)
        message = str(exc.value)
        assert "Unrecognized Stickler extension 'x-aws-stickler-thresold'" in message
        assert f"'{path}'" in message
        assert "Did you mean 'x-aws-stickler-threshold'" in message

    def test_a_class_key_on_a_scalar_definition_raises(self):
        """Nothing reads it there, the same as on a scalar property."""
        with pytest.raises(ValueError, match="belongs on the object"):
            StructuredModel.from_json_schema(
                _with_defs(
                    {"$ref": "#/$defs/Code"},
                    {"Code": {"type": "string", "x-aws-stickler-match-threshold": 0.9}},
                )
            )

    @pytest.mark.parametrize(
        "extension, error",
        (
            ({"x-aws-stickler-threshold": 5}, "between 0.0 and 1.0"),
            ({"x-aws-stickler-comparator": "NoSuch"}, "Unknown comparator"),
            ({"x-aws-stickler-comparator": "auto"}, "no comparator to infer"),
        ),
        ids=("threshold", "unknown-comparator", "auto-on-object"),
    )
    def test_a_bad_value_raises_the_same_error_as_inline(self, extension, error):
        with pytest.raises(ValueError, match=error):
            StructuredModel.from_json_schema(
                _with_defs({"$ref": "#/$defs/I"}, {"I": dict(INNER, **extension)})
            )


class TestWhatDoesNotChange:
    def test_an_items_reference_takes_only_the_class_keys(self):
        """An array's `items` is not a property, so no field key is carried there.

        The array is the field, and a field threshold on an array of models is
        ignored anyway. Pinned so the definition's `0.88` cannot start leaking
        onto the array, or into the "has no effect" warning, unnoticed.
        """
        definition = dict(DECLARED, **{"x-aws-stickler-match-threshold": 0.9})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = StructuredModel.from_json_schema(
                _with_defs(
                    {"type": "array", "items": {"$ref": "#/$defs/I"}},
                    {"I": definition},
                )
            )
        assert _child_class(model).match_threshold == 0.9
        # The array-of-models resolution, the same as with no `$defs` at all.
        assert model._get_comparison_info("inner").threshold == pytest.approx(0.5)
        assert not [w for w in caught if "has no effect" in str(w.message)]

    @pytest.mark.parametrize(
        "defs",
        (
            {
                "N": {
                    "type": "object",
                    "x-aws-stickler-threshold": 0.88,
                    "properties": {
                        "v": {"type": "string"},
                        "child": {"$ref": "#/$defs/N"},
                    },
                }
            },
            {
                "N": {"$ref": "#/$defs/B", "x-aws-stickler-weight": 2.0},
                "B": {"$ref": "#/$defs/N"},
            },
        ),
        ids=("self-reference", "cyclic-chain"),
    )
    def test_recursion_still_gets_the_recursion_error(self, defs):
        with pytest.raises(ValueError, match="Recursive JSON Schema models"):
            StructuredModel.from_json_schema(_with_defs({"$ref": "#/$defs/N"}, defs))
