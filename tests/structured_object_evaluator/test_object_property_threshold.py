"""`x-aws-stickler-threshold` is honoured on an object-typed property.

The importer read `weight` and `clip-under-threshold` from an object-typed
property's extensions and used a hardcoded literal for `threshold`, so the same
node honoured two of the field-level keys it was given and discarded this one,
with no error and no warning. `x-aws-stickler-comparator` on that node is ALSO
discarded and is deliberately NOT fixed here, because it is score-inert for a
nested model; `TestTheComparatorOnTheSameNodeIsStillDropped` pins it so this
paragraph cannot drift into claiming more than it does.

The key is NOT inert in that position, which is why carrying it is the fix rather
than rejecting it the way #312 rejects a genuinely misplaced key. A nested-model
field's threshold gates the subtree mean, so it changes scores. Set through a
`StructuredModel` class it worked; set in a schema it did not, and the two
configuration paths disagreeing about what is configurable is the divergence #210
and #211 are about.

The same key one position over -- on an array-of-MODELS property, meaning an array
whose `items` declare `properties` -- is IGNORED, because array pairing reads the
element class's own `match_threshold`. Refusing it was the first attempt and would
have stopped previously exported schemas from importing: every released
`to_json_schema()` emitted the key there.

"Models", not "objects". An array of free-form `{"type": "object"}` items becomes
`List[dict]`, which READS the declared threshold and warns about nothing. See
`TestAnArrayOfFreeFormObjectsReadsIt`.

The warning is gated on the VALUE, not on presence. Every released
`to_json_schema()` wrote `0.5` and could write nothing else, since
`__init_subclass__` refuses a named threshold on a `List[StructuredModel]` field,
so `0.5` carries no authorial intent and passes silently; any other value is the
author's and warns. This release stops emitting the key on that shape at all, so
only legacy artifacts reach the sentinel. Warning on presence fired on stickler's
own export, and its advice named the discarded value as the one to write, which
would overwrite the element class's real gate.

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

        One node, and the threshold was the only one of these three dropped.
        `x-aws-stickler-comparator` on the same node is ALSO dropped and is not
        fixed here; see `TestTheComparatorOnTheSameNodeIsStillDropped`, which
        keeps this docstring from reading as "and now everything is honoured".
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

    def test_an_unstated_threshold_falls_back_to_the_class_match_threshold(self):
        """`0.7` here is `StructuredModel.match_threshold`, not `ComparableField`'s.

        Named for the right mechanism. `ComparableField`'s own default is `0.5`
        (`comparable_field.py`, `_LEGACY_DEFAULT_THRESHOLD`) and that is what a
        SCALAR property falls back to; a nested model falls back to the class
        gate instead. Calling this "the object-grade default" pointed at 0.5 and
        asserted 0.7.
        """
        model = StructuredModel.from_json_schema(_schema())
        assert model._get_comparison_info("inner").threshold == 0.7
        assert _child_class(model).match_threshold == 0.7

    def test_a_scalar_property_falls_back_to_a_different_number(self):
        """The contrast that makes the line above mean something."""
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "Parent",
                "properties": {"inner": {"type": "string"}},
            }
        )
        assert model._get_comparison_info("inner").threshold == 0.5


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
        """Does NOT discriminate on its own -- see the test below, which does.

        A two-leaf subtree scores `0.5`, and `0.5` is under the old hardcoded
        `0.7` as well as under the declared `0.6`, so this assertion passed
        before the fix too. Kept because it states the intended behaviour, but it
        is not the regression guard.
        """
        assert self._score(0.6) == pytest.approx(0.0)

    def test_above_it_the_partial_score_survives(self):
        assert self._score(0.0) == pytest.approx(0.5)

    def test_a_declared_threshold_ABOVE_the_old_default_is_what_discriminates(self):
        """The real guard: a subtree that clears `0.7` but not the declared value.

        Four leaves with one wrong scores `0.75`. Under the old hardcoded `0.7`
        that survives; under a declared `0.8` it is clipped to `0.0`. Measured on
        `dev` it is `0.75`, here it is `0.0`, so this test fails if the literal
        ever comes back -- which the two tests above do not.
        """
        wide = {
            "type": "object",
            "title": "Wide",
            "properties": {f"c{index}": {"type": "string"} for index in range(4)},
        }
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "Parent",
                "properties": {
                    "inner": dict(
                        wide,
                        **{
                            "x-aws-stickler-threshold": 0.8,
                            "x-aws-stickler-clip-under-threshold": True,
                        },
                    )
                },
            }
        )
        child = _child_class(model)
        right = {f"c{index}": "same" for index in range(4)}
        wrong = dict(right, c3="DIFFERENT")
        assert model._get_comparison_info("inner").threshold == pytest.approx(0.8)
        assert model(inner=child(**right)).compare_with(model(inner=child(**wrong)))[
            "field_scores"
        ]["inner"] == pytest.approx(0.0)


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


class TestAnArrayOfFreeFormObjectsReadsIt:
    """"Array of MODELS" ignores it; "array of objects" is not the same thing.

    The distinction is `items` declaring `properties`. Without them the array
    becomes `List[dict]`, which READS the declared threshold and warns about
    nothing -- so the docs sentence had to say "models", and this pins the shape
    that would have made it false.
    """

    @staticmethod
    def _import(items):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "Parent",
                    "properties": {
                        "rows": {
                            "type": "array",
                            "x-aws-stickler-threshold": 0.88,
                            "items": items,
                        }
                    },
                }
            )
        return model, [str(w.message) for w in caught]

    def test_free_form_object_items_honour_it_and_do_not_warn(self):
        model, messages = self._import({"type": "object"})
        assert model._get_comparison_info("rows").threshold == pytest.approx(0.88)
        assert not [m for m in messages if "has no effect on array property" in m]

    def test_items_with_properties_ignore_it_and_do_warn(self):
        model, messages = self._import(
            {"type": "object", "title": "Row", "properties": {"c": {"type": "string"}}}
        )
        assert model._get_comparison_info("rows").threshold != pytest.approx(0.88)
        assert [m for m in messages if "has no effect on array property" in m]


class TestTheComparatorOnTheSameNodeIsStillDropped:
    """The sibling key this PR does NOT fix, pinned so the claim stays honest.

    The entry for this change argues from "one node honoured two field-level keys
    and discarded the third". A reviewer checking that sentence finds a FOURTH
    key on the same node which is also discarded, so the sentence has to be
    precise and something has to hold it precise.

    `x-aws-stickler-comparator` is parsed and validated -- a bogus name still
    raises -- then dropped without being stored or exported, because the
    object-typed branch passes a fixed `comparator_name` rather than the node's
    extensions. Score-inert for a nested model, which is compared by recursion
    rather than by the field's comparator, which is why it is recorded here
    rather than carried. Identical on `dev`.
    """

    def test_a_declared_comparator_is_dropped_on_an_object_property(self):
        model = StructuredModel.from_json_schema(
            _schema(**{"x-aws-stickler-comparator": "ExactComparator"})
        )
        resolved = model._get_comparison_info("inner").comparator
        assert type(resolved).__name__ == "LevenshteinComparator"

    def test_a_scalar_property_honours_the_same_key(self):
        """The contrast, so this reads as an asymmetry rather than a global."""
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "Parent",
                "properties": {
                    "inner": {
                        "type": "string",
                        "x-aws-stickler-comparator": "ExactComparator",
                    }
                },
            }
        )
        resolved = model._get_comparison_info("inner").comparator
        assert type(resolved).__name__ == "ExactComparator"

    def test_a_bogus_name_still_raises_there(self):
        """Parsed and validated, then dropped -- not ignored wholesale."""
        with pytest.raises(Exception):
            StructuredModel.from_json_schema(
                _schema(**{"x-aws-stickler-comparator": "NoSuchComparator"})
            )


class TestTheKeyIsStillDroppedInsideDefs:
    """The declared gap: `$ref` into `$defs` does not see the extension.

    `$ref` is how a generated schema expresses a reusable nested object, so this
    is the spelling many real schemas use. The extension walk does not descend
    into `$defs`, which is pre-existing on `dev` for every key -- but this change
    is what makes the loss score-relevant for the threshold.

    Asserts CURRENT behaviour. When the walk is extended, these fail, and that
    failure is the reminder to delete the "Known gap" paragraph from the
    CHANGELOG in the same commit.
    """

    WIDE = {
        "type": "object",
        "title": "Inner",
        "properties": {"b": {"type": "string"}, "c": {"type": "string"}},
    }

    def _via_defs(self, child):
        return StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "Parent",
                "properties": {"inner": {"$ref": "#/$defs/Inner"}},
                "$defs": {"Inner": child},
            }
        )

    def test_inline_honours_it(self):
        """The control: the same node, the other spelling."""
        model = StructuredModel.from_json_schema(
            _schema(**{"x-aws-stickler-threshold": 0.88})
        )
        assert model._get_comparison_info("inner").threshold == pytest.approx(0.88)

    def test_via_defs_silently_falls_back(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = self._via_defs(
                dict(self.WIDE, **{"x-aws-stickler-threshold": 0.88})
            )
        assert model._get_comparison_info("inner").threshold == pytest.approx(0.7)
        assert not [str(w.message) for w in caught if "threshold" in str(w.message)]

    def test_a_typo_inline_is_rejected(self):
        """#312's guard works in the position this change touches."""
        with pytest.raises(ValueError, match="Unrecognized Stickler extension"):
            StructuredModel.from_json_schema(
                _schema(**{"x-aws-stickler-thresold": 0.88})
            )

    def test_the_same_typo_inside_defs_is_not(self):
        """The half that makes the gap a validation hole, not just a default."""
        model = self._via_defs(dict(self.WIDE, **{"x-aws-stickler-thresold": 0.88}))
        assert model._get_comparison_info("inner").threshold == pytest.approx(0.7)


class TestAnUndeclaredSchemaIsUnaffectedEndToEnd:
    """The blast-radius bound the entry claims, measured.

    Scores move only where the key was DECLARED. A schema that declares nothing
    exports `0.7` on a nested object property and re-imports at `0.7`, scoring
    identically on `dev` and here -- so the export/import cycle does not silently
    move anyone who never touched this key.
    """

    def test_export_then_reimport_keeps_the_same_score(self):
        model = StructuredModel.from_json_schema(_schema())
        child = _child_class(model)
        right, wrong = child(b="x", c="y"), child(b="x", c="ZZ")
        before = model(inner=right).compare_with(model(inner=wrong))["field_scores"][
            "inner"
        ]

        rebuilt = StructuredModel.from_json_schema(model.to_json_schema())
        rebuilt_child = _child_class(rebuilt)
        after = rebuilt(inner=rebuilt_child(b="x", c="y")).compare_with(
            rebuilt(inner=rebuilt_child(b="x", c="ZZ"))
        )["field_scores"]["inner"]

        assert before == pytest.approx(after)
        assert rebuilt._get_comparison_info("inner").threshold == pytest.approx(0.7)
