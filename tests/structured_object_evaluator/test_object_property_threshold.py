"""`x-aws-stickler-threshold` is honoured on an object-typed property.

The importer read `weight` and `clip-under-threshold` from an object-typed
property's extensions and used a hardcoded literal for `threshold`, so the same
node honoured two field-level keys and discarded the third, with no error and no
warning.

The key is NOT inert in that position, which is why carrying it is the fix rather
than rejecting it the way #312 rejects a genuinely misplaced key. A nested-model
field's threshold gates the subtree mean, so it changes scores. Set through a
`StructuredModel` class it worked; set in a schema it did not, and the two
configuration paths disagreeing about what is configurable is the divergence #210
and #211 are about.

The same key one position over -- on an array-of-objects property -- is IGNORED,
because array pairing reads the element class's own `match_threshold`. Refusing it
was the first attempt and would have stopped previously exported schemas from
importing: every released `to_json_schema()` emits the key there.

The warning is gated on the VALUE, not on presence. Export writes `0.5` and can
write nothing else, since `__init_subclass__` refuses a named threshold on a
`List[StructuredModel]` field, so `0.5` carries no authorial intent and passes
silently; any other value is the author's and warns. Warning on presence fired on
stickler's own export, and its advice named the discarded value as the one to
write, which would overwrite the element class's real gate.

See https://github.com/awslabs/stickler/issues/317
"""

import warnings
from typing import List, Optional

import pytest

from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)

CHILD = {
    "type": "object",
    "properties": {"b": {"type": "string"}, "c": {"type": "string"}},
}


def _schema(**extensions):
    return {
        "type": "object",
        "title": "Parent",
        "properties": {"inner": dict(CHILD, **extensions)},
    }


def _child_class(model):
    """The nested model class behind an Optional[Model] annotation."""
    annotation = model.model_fields["inner"].annotation
    args = getattr(annotation, "__args__", None)
    return args[0] if args else annotation


class TestTheDeclaredThresholdIsRead:
    def test_it_reaches_the_field_config(self):
        model = StructuredModel.from_json_schema(
            _schema(**{"x-aws-stickler-threshold": 0.88})
        )
        assert model._get_comparison_info("inner").threshold == 0.88

    def test_the_neighbouring_keys_were_always_read(self):
        """The asymmetry that identified this as a defect, pinned.

        One node, three field-level keys, and only one of them was dropped.
        """
        model = StructuredModel.from_json_schema(
            _schema(
                **{
                    "x-aws-stickler-threshold": 0.88,
                    "x-aws-stickler-weight": 3.0,
                    "x-aws-stickler-clip-under-threshold": False,
                }
            )
        )
        info = model._get_comparison_info("inner")
        assert (info.threshold, info.weight, info.clip_under_threshold) == (
            0.88,
            3.0,
            False,
        )

    def test_an_unstated_threshold_keeps_the_object_grade_default(self):
        model = StructuredModel.from_json_schema(_schema())
        assert model._get_comparison_info("inner").threshold == 0.7


class TestTheThresholdChangesTheScore:
    """Asserting the stored number alone would pass even if nothing read it.

    That is exactly the bug: the value was being stored somewhere and never
    consulted for this position.
    """

    @staticmethod
    def _score(threshold):
        model = StructuredModel.from_json_schema(
            _schema(
                **{
                    "x-aws-stickler-threshold": threshold,
                    "x-aws-stickler-clip-under-threshold": True,
                }
            )
        )
        child = _child_class(model)
        # One of two leaves differs, so the subtree mean is 0.5.
        return model(inner=child(b="x", c="y")).compare_with(
            model(inner=child(b="x", c="ZZ"))
        )["field_scores"]["inner"]

    def test_below_the_threshold_the_subtree_is_clipped(self):
        assert self._score(0.6) == pytest.approx(0.0)

    def test_above_it_the_partial_score_survives(self):
        assert self._score(0.0) == pytest.approx(0.5)


class TestItSurvivesARoundTrip:
    """The regression guard the issue asks for.

    Export a model whose nested-model field states a threshold, reimport, and
    assert the value comes back. Without the fix the reimport silently produced
    `0.7`, so a model could be exported and not restored.
    """

    def test_a_python_declared_nested_threshold_comes_back(self):
        class Kid(StructuredModel):
            b: Optional[str] = ComparableField(default=None)

        class Doc(StructuredModel):
            kid: Optional[Kid] = ComparableField(
                threshold=0.83, weight=2.5, default=None
            )

        schema = Doc.to_json_schema()
        assert schema["properties"]["kid"]["x-aws-stickler-threshold"] == 0.83

        rebuilt = StructuredModel.from_json_schema(schema)
        info = rebuilt._get_comparison_info("kid")
        assert (info.threshold, info.weight) == (0.83, 2.5)


class TestEveryPositionAnswersTheSameWay:
    """The gap was found by probing one position; the others need pinning too."""

    @pytest.mark.parametrize(
        "prop",
        (
            {"type": "string"},
            CHILD,
            {"type": "array", "items": {"type": "string"}},
        ),
        ids=("scalar", "object", "array-of-scalars"),
    )
    def test_a_declared_threshold_is_honoured(self, prop):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "properties": {"f": dict(prop, **{"x-aws-stickler-threshold": 0.88})},
            }
        )
        assert model._get_comparison_info("f").threshold == 0.88


class TestAnArrayOfObjectsIgnoresItWithAWarning:
    """The same key one position over, where it genuinely has no effect.

    Array pairing reads the element class's `match_threshold`, so a field
    threshold there does nothing. It is dropped with a warning rather than
    refused: every `to_json_schema()` on a released version emitted this key on an
    array-of-model property, so refusing it would stop previously exported schemas
    from importing in order to flag a key whose only cost is being ignored.

    The warning is gated on the VALUE, not on presence -- see
    `TestOurOwnExportDoesNotWarn` for why.
    """

    ARRAY_OF_OBJECTS = {
        "type": "object",
        "title": "T",
        "properties": {
            "f": {
                "type": "array",
                "items": CHILD,
                "x-aws-stickler-threshold": 0.88,
            }
        },
    }

    def test_the_schema_still_imports(self):
        model = StructuredModel.from_json_schema(self.ARRAY_OF_OBJECTS)
        assert "f" in model.model_fields

    def test_it_warns_naming_the_key_the_author_should_write(self):
        with pytest.warns(UserWarning, match="x-aws-stickler-match-threshold"):
            StructuredModel.from_json_schema(self.ARRAY_OF_OBJECTS)

    def test_the_warning_echoes_the_declared_value(self):
        with pytest.warns(UserWarning, match="0.88"):
            StructuredModel.from_json_schema(self.ARRAY_OF_OBJECTS)

    def test_it_does_not_offer_the_discarded_value_as_the_fix(self):
        """Naming it as the value to write would overwrite a working gate.

        The element class's `match_threshold` is a different number, so
        "put 'x-aws-stickler-match-threshold': 0.88 inside items" told the reader
        to replace their real gate with the one being thrown away.
        """
        with pytest.warns(UserWarning) as caught:
            StructuredModel.from_json_schema(self.ARRAY_OF_OBJECTS)
        message = str(caught[0].message)
        assert "x-aws-stickler-match-threshold" in message
        assert "'x-aws-stickler-match-threshold': 0.88" not in message

    def test_following_the_advice_works(self):
        """A warning recommending something unusable is not an improvement."""
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "properties": {
                    "f": {
                        "type": "array",
                        "items": dict(
                            CHILD, **{"x-aws-stickler-match-threshold": 0.88}
                        ),
                    }
                },
            }
        )
        element = model.model_fields["f"].annotation
        while getattr(element, "__args__", None):
            element = element.__args__[0]
        assert element.match_threshold == 0.88

    def test_the_python_path_still_raises_with_its_own_advice(self):
        """Ignoring is for a schema key. A Python declaration is still refused."""

        class Line(StructuredModel):
            sku: Optional[str] = ComparableField(default=None)

        with pytest.raises(ValueError, match="ComparableField"):

            class Doc(StructuredModel):
                items: Optional[List[Line]] = ComparableField(
                    threshold=0.88, default=None
                )


class TestOurOwnExportDoesNotWarn:
    """The warning must not fire on a schema stickler itself produced.

    Originally this pinned a sentinel: `to_json_schema()` emitted
    `x-aws-stickler-threshold: 0.5` on every `List[StructuredModel]` property and
    could emit nothing else, since `__init_subclass__` refuses a named threshold on
    that shape. Warning on the key's mere presence fired on the library's own output
    for every model holding a list of models, advising the author to change
    something they never wrote.

    #246 then stopped emitting the key there at all, so the silence below now has a
    stronger cause than a value comparison: there is nothing on the property to
    ignore. The legacy path still matters, because every schema already written to
    disk carries the key, and `TestAPreviouslyExportedSchemaStillImports` covers
    that side.
    """

    class Line(StructuredModel):
        match_threshold = 0.85
        sku: Optional[str] = ComparableField(default=None)

    @staticmethod
    def _doc():
        class Doc(StructuredModel):
            lines: Optional[List["TestOurOwnExportDoesNotWarn.Line"]] = ComparableField(
                default=None
            )

        return Doc

    def test_export_no_longer_writes_a_threshold_there(self):
        """The premise, restated after #246 stopped emitting the key.

        This asserted the sentinel `0.5` was present, and said in its own docstring
        that the class could go if export ever stopped writing it. Export did stop,
        so the assertion is inverted rather than deleted: an emitted key here would
        be a regression, since a named threshold on this shape is refused at class
        definition and the value could only ever be a placeholder.
        """
        exported = self._doc().to_json_schema()["properties"]["lines"]
        assert "x-aws-stickler-threshold" not in exported

    def test_a_round_trip_of_our_own_export_is_silent(self):
        schema = self._doc().to_json_schema()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StructuredModel.from_json_schema(schema)
        assert [str(w.message) for w in caught] == []

    def test_the_element_gate_survives_that_round_trip(self):
        """Silence is only correct if the ignored value really was inert."""
        rebuilt = StructuredModel.from_json_schema(self._doc().to_json_schema())
        element = rebuilt.model_fields["lines"].annotation
        while getattr(element, "__args__", None):
            element = element.__args__[0]
        assert element.match_threshold == 0.85

    def test_a_value_other_than_the_sentinel_still_warns(self):
        """Gating on the value must not swallow a number the author chose."""
        schema = self._doc().to_json_schema()
        schema["properties"]["lines"]["x-aws-stickler-threshold"] = 0.8
        with pytest.warns(UserWarning, match="has no effect on array property"):
            StructuredModel.from_json_schema(schema)


class TestASecondSchemaIsNotSilenced:
    """Two schemas naming the same property must both be told.

    `warn_once` memoises on `(id, context)` for the life of the process, so with
    the field path as the context the second schema declaring an array property
    called `f` imported silently. Its author never heard that their key was dead,
    which is exactly the silent drop this module exists to prevent. `warn_once` is
    right for a per-document deprecation, where the alternative is one warning per
    row of a corpus; a schema import happens once per call.
    """

    @staticmethod
    def _schema(threshold):
        return {
            "type": "object",
            "title": "T",
            "properties": {
                "f": {
                    "type": "array",
                    "items": CHILD,
                    "x-aws-stickler-threshold": threshold,
                }
            },
        }

    def test_the_same_path_warns_again_for_a_second_schema(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StructuredModel.from_json_schema(self._schema(0.8))
            StructuredModel.from_json_schema(self._schema(0.8))
        assert len(caught) == 2

    def test_a_different_value_at_the_same_path_also_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StructuredModel.from_json_schema(self._schema(0.8))
            StructuredModel.from_json_schema(self._schema(0.9))
        assert len(caught) == 2
        assert "0.8" in str(caught[0].message)
        assert "0.9" in str(caught[1].message)

    def test_the_sentinel_is_still_silent_across_repeats(self):
        """Gating on the value must not be undone by dropping `warn_once`."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StructuredModel.from_json_schema(self._schema(0.5))
            StructuredModel.from_json_schema(self._schema(0.5))
        assert [str(w.message) for w in caught] == []
