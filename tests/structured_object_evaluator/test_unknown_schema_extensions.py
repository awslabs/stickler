"""An extension key Stickler does not honour is rejected, not dropped.

A dropped key left a model that built, ran, and produced numbers wrong in the
direction of over-reporting accuracy. The sharpest shape was a typo alongside a
correctly-spelled sibling, which produced a configuration nobody would choose:

    x-aws-stickler-comparitor: ExactComparator   (typo, dropped)
    x-aws-stickler-threshold:  1.0               (honoured)
    -> LevenshteinComparator at threshold 1.0

Raising is safe here in a way it is not elsewhere in the codebase: a schema is
authored once and read deterministically, so there is no risk of failing on
document N of a corpus after succeeding on N-1. The author is present and the
mistake is in a file they can edit.

See https://github.com/awslabs/stickler/issues/210
"""

import typing

import pytest

from stickler import StructuredModel


def build(props: dict):
    return StructuredModel.from_json_schema(
        {"type": "object", "title": "T", "properties": props}
    )


class TestTypoAndWrongPrefixAreRejected:
    def test_a_typo_raises_and_suggests_the_right_key(self):
        with pytest.raises(ValueError) as exc:
            build(
                {
                    "invoice_id": {
                        "type": "string",
                        "x-aws-stickler-comparitor": "ExactComparator",
                        "x-aws-stickler-threshold": 1.0,
                    }
                }
            )
        message = str(exc.value)
        assert "x-aws-stickler-comparitor" in message, "names the offending key"
        assert "invoice_id" in message, "names the field"
        assert "x-aws-stickler-comparator" in message, "suggests the correction"

    def test_the_wrong_prefix_raises_too(self):
        """`x-stickler-*` appears in our own README, so it is a mistake we taught.

        The suggestion has to work across a four-character prefix difference,
        which plain string similarity does not catch, so it is matched on the
        suffix as well.
        """
        with pytest.raises(ValueError, match="x-aws-stickler-comparator"):
            build({"name": {"type": "string", "x-stickler-comparator": "fuzzy"}})

    @pytest.mark.parametrize(
        "key",
        [
            "x-aws-stickler-thresold",
            "x-aws-stickler-weightt",
            "x-aws-stickler-clip-under-treshold",
            "x-aws-stickler-nonsense",
            "x-stickler-threshold",
        ],
    )
    def test_every_extension_shaped_key_is_checked(self, key):
        with pytest.raises(ValueError, match="Unrecognized Stickler extension"):
            build({"a": {"type": "string", key: 0.9}})

    def test_the_suggestion_is_omitted_rather_than_wrong(self):
        """A key with no plausible correction still raises, without inventing one."""
        with pytest.raises(ValueError) as exc:
            build({"a": {"type": "string", "x-aws-stickler-zzzzzzzz": 1}})
        assert "Did you mean" not in str(exc.value)


class TestWhatMustKeepWorking:
    def test_correct_keys_are_honoured(self):
        model = build(
            {
                "a": {
                    "type": "string",
                    "x-aws-stickler-comparator": "ExactComparator",
                    "x-aws-stickler-threshold": 1.0,
                    "x-aws-stickler-weight": 2.0,
                    "x-aws-stickler-clip-under-threshold": False,
                }
            }
        )
        info = model._get_comparison_info("a")
        assert type(info.comparator).__name__ == "ExactComparator"
        assert info.threshold == 1.0
        assert info.weight == 2.0
        assert info.clip_under_threshold is False

    def test_a_schema_exported_by_an_older_version_still_imports(self):
        """`x-aws-stickler-aggregate` was removed in #226 and is accepted, not
        advertised. Rejecting it would break every schema file already on disk."""
        model = build(
            {
                "a": {
                    "type": "string",
                    "x-aws-stickler-comparator": "ExactComparator",
                    "x-aws-stickler-aggregate": True,
                }
            }
        )
        assert type(model._get_comparison_info("a").comparator).__name__ == (
            "ExactComparator"
        )

    def test_an_unrelated_x_extension_is_left_alone(self):
        """Other tooling puts `x-*` keys in schemas. Only Stickler-shaped ones
        are our business."""
        model = build({"a": {"type": "string", "x-my-own-tool": {"anything": 1}}})
        assert model is not None

    def test_a_bad_value_still_raises_its_own_error(self):
        """The pre-existing value check must not be shadowed by the key check."""
        with pytest.raises(ValueError, match="Invalid x-aws-stickler-comparator"):
            build({"a": {"type": "string", "x-aws-stickler-comparator": "Nonsense"}})

    def test_the_advertised_key_list_omits_internal_and_removed_keys(self):
        """Suggesting `aggregate` (removed) or `internal-examples` (ours) would
        send an author somewhere useless."""
        with pytest.raises(ValueError) as exc:
            build({"a": {"type": "string", "x-aws-stickler-nonsense": 1}})
        message = str(exc.value)
        assert "x-aws-stickler-aggregate" not in message
        assert "x-aws-stickler-internal-examples" not in message
        assert "x-aws-stickler-comparator" in message


class TestNestedFields:
    def test_a_typo_inside_a_nested_object_names_its_path(self):
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "Outer",
                    "properties": {
                        "inner": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "string", "x-aws-stickler-thresold": 0.9}
                            },
                        }
                    },
                }
            )
        assert "x-aws-stickler-thresold" in str(exc.value)


def _unwrap(annotation):
    """The class behind `Optional[X]` / `List[X]` / `Optional[List[X]]`."""
    while True:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if not args:
            return annotation
        annotation = args[0]


class TestTheCheckReachesEveryPosition:
    """A field-level check only reached positions that produce a field.

    The root object, `items`, and a list-form `["object", "null"]` node all
    carried extension keys that were dropped in silence, while the same typo one
    level down raised. The guarantee is worth little if it has holes, so the walk
    over the raw schema is what enforces it now.
    """

    def test_a_typo_on_the_root_object_raises(self):
        with pytest.raises(ValueError, match="x-aws-stickler-match-treshold"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "T",
                    "x-aws-stickler-match-treshold": 0.95,
                    "properties": {"a": {"type": "string"}},
                }
            )

    def test_a_model_name_typo_on_the_root_raises(self):
        with pytest.raises(ValueError, match="x-aws-stickler-model-nmae"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "T",
                    "x-aws-stickler-model-nmae": "Foo",
                    "properties": {"a": {"type": "string"}},
                }
            )

    def test_the_wrong_prefix_on_the_root_raises(self):
        with pytest.raises(ValueError, match="x-stickler-match-threshold"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "T",
                    "x-stickler-match-threshold": 0.9,
                    "properties": {"a": {"type": "string"}},
                }
            )

    def test_a_typo_on_array_items_raises(self):
        with pytest.raises(ValueError, match="x-aws-stickler-thresold"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "T",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "x-aws-stickler-thresold": 0.9,
                            },
                        }
                    },
                }
            )

    @pytest.mark.parametrize("declared", (["object", "null"], ["array", "null"]))
    def test_the_wrong_prefix_on_a_list_form_type_raises(self, declared):
        """The check runs before list-form types are rewritten to `anyOf`.

        Running after normalization was how these escaped: the node the author
        wrote no longer existed in the shape being inspected.
        """
        node = {"type": declared, "x-stickler-comparator": "fuzzy"}
        if "object" in declared:
            node["properties"] = {"z": {"type": "string"}}
        else:
            node["items"] = {"type": "string"}

        with pytest.raises(ValueError, match="x-stickler-comparator"):
            StructuredModel.from_json_schema(
                {"type": "object", "title": "T", "properties": {"o": node}}
            )


class TestPositionIsPartOfValidity:
    """A valid key in a position that does not read it is still a silent drop.

    Allowlisting both key sets everywhere made the check pass on keys that do
    nothing, and worse, let the suggester answer a field-position typo of
    `-match-threshold` with "did you mean `x-aws-stickler-match-threshold`" when
    applying that advice imported cleanly and changed nothing.
    """

    @pytest.mark.parametrize(
        "key, value",
        (
            ("x-aws-stickler-match-threshold", 0.9),
            ("x-aws-stickler-model-name", "Foo"),
        ),
    )
    def test_an_object_level_key_on_a_scalar_field_raises(self, key, value):
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "T",
                    "properties": {"a": {"type": "string", key: value}},
                }
            )
        message = str(exc.value)
        assert key in message
        assert "belongs on the object" in message

    def test_the_suggester_never_routes_to_a_key_that_is_not_read_here(self):
        """The failure this closes: a suggestion that reintroduces the bug."""
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "T",
                    "properties": {
                        "a": {"type": "string", "x-aws-stickler-match-treshold": 0.9}
                    },
                }
            )
        message = str(exc.value)
        assert "x-aws-stickler-match-threshold" not in message.split("Valid keys")[0]


class TestObjectPropertiesReadBothKeySets:
    """An object-typed property is a field of its parent AND its own class.

    Both key sets are honoured there, so scoping the check must not turn a
    working schema into an error. These pin the values, not just that it imports.
    """

    @staticmethod
    def _object_property(**extensions):
        return {
            "type": "object",
            "title": "T",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"b": {"type": "string"}},
                    **extensions,
                }
            },
        }

    def test_match_threshold_on_a_nested_object_is_honoured(self):
        model = StructuredModel.from_json_schema(
            self._object_property(**{"x-aws-stickler-match-threshold": 0.93})
        )
        assert _unwrap(model.model_fields["inner"].annotation).match_threshold == 0.93

    def test_model_name_on_a_nested_object_is_honoured(self):
        model = StructuredModel.from_json_schema(
            self._object_property(**{"x-aws-stickler-model-name": "Renamed"})
        )
        assert _unwrap(model.model_fields["inner"].annotation).__name__ == "Renamed"

    def test_weight_on_a_nested_object_is_honoured(self):
        model = StructuredModel.from_json_schema(
            self._object_property(**{"x-aws-stickler-weight": 3.0})
        )
        assert model._get_comparison_info("inner").weight == 3.0

    def test_match_threshold_on_array_items_is_honoured(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "x-aws-stickler-match-threshold": 0.91,
                            "properties": {"b": {"type": "string"}},
                        },
                    }
                },
            }
        )
        assert _unwrap(model.model_fields["rows"].annotation).match_threshold == 0.91

    def test_the_root_still_reads_both_of_its_own_keys(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "x-aws-stickler-model-name": "Invoice",
                "x-aws-stickler-match-threshold": 0.9,
                "properties": {"a": {"type": "string"}},
            }
        )
        assert model.__name__ == "Invoice"
        assert model.match_threshold == 0.9


class TestTheDocumentedValuesWork:
    """The README's own values must import, since copying them is the point.

    Correcting the prefix without correcting the values would have turned a
    double no-op into a hard error for anyone following the page.
    """

    @pytest.mark.parametrize(
        "comparator_name",
        (
            "ExactComparator",
            "FuzzyComparator",
            "LevenshteinComparator",
        ),
    )
    def test_each_documented_comparator_class_name_is_accepted(self, comparator_name):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "properties": {
                    "a": {
                        "type": "string",
                        "x-aws-stickler-comparator": comparator_name,
                    }
                },
            }
        )
        installed = model._get_comparison_info("a").comparator
        assert type(installed).__name__ == comparator_name

    @pytest.mark.parametrize("alias", ("exact", "fuzzy", "levenshtein", "semantic"))
    def test_the_lowercase_aliases_the_readme_used_to_list_still_raise(self, alias):
        """Recorded so the corrected README cannot quietly regress.

        These were documented for a prefix that was itself wrong, so the example
        applied none of its settings and raised nothing. Now the prefix is right,
        a lowercase value is a hard error, which is why both halves had to move
        together.
        """
        with pytest.raises(ValueError, match="Invalid x-aws-stickler-comparator"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "T",
                    "properties": {
                        "a": {"type": "string", "x-aws-stickler-comparator": alias}
                    },
                }
            )

    def test_the_readme_example_applies_every_setting_it_documents(self):
        """Verbatim from `structured_object_evaluator/README.md`."""
        document_schema = {
            "type": "object",
            "title": "Document",
            "properties": {
                "title": {
                    "type": "string",
                    "x-aws-stickler-comparator": "FuzzyComparator",
                    "x-aws-stickler-threshold": 0.8,
                },
                "priority": {"type": "integer", "x-aws-stickler-weight": 2.0},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "priority"],
        }

        model = StructuredModel.from_json_schema(document_schema)

        title = model._get_comparison_info("title")
        assert type(title.comparator).__name__ == "FuzzyComparator"
        assert title.threshold == 0.8
        assert model._get_comparison_info("priority").weight == 2.0


_OBJECT_SPELLINGS = pytest.mark.parametrize(
    "declared", ("object", ["object", "null"]), ids=("object", "object-or-null")
)


class TestANullableObjectReadsItsModelKeys:
    """`["object", "null"]` configures its class exactly as `"object"` does.

    The list form is rewritten to `anyOf` before parsing, and the model keys
    stayed beside the `anyOf` instead of on the object arm that becomes the
    class. The node built `DynamicModel` at 0.7 while its field keys, read off
    the property, were honoured, so the loss was partial and invisible.

    See https://github.com/awslabs/stickler/issues/349 (part A).
    """

    @staticmethod
    def _child(declared, **extensions):
        return StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "properties": {
                    "child": {
                        "type": declared,
                        "properties": {
                            "a": {"type": "string"},
                            "n": {"type": "number"},
                        },
                        "required": ["a", "n"],
                        **extensions,
                    }
                },
                "required": ["child"],
            }
        )

    @_OBJECT_SPELLINGS
    def test_name_and_match_threshold_are_honoured(self, declared):
        model = self._child(
            declared,
            **{
                "x-aws-stickler-model-name": "ChildRenamed",
                "x-aws-stickler-match-threshold": 0.93,
                "x-aws-stickler-weight": 3.0,
            },
        )
        child = _unwrap(model.model_fields["child"].annotation)
        assert child.__name__ == "ChildRenamed"
        assert child.match_threshold == 0.93
        assert model._get_comparison_info("child").weight == 3.0

    @_OBJECT_SPELLINGS
    def test_infer_unspecified_is_honoured(self, declared):
        model = self._child(declared, **{"x-aws-stickler-infer-unspecified": True})
        info = _unwrap(model.model_fields["child"].annotation)._get_comparison_info("n")
        assert type(info.comparator).__name__ == "NumericComparator"
        assert info.threshold == 0.95

    @pytest.mark.parametrize(
        "key, value, read",
        (
            ("x-aws-stickler-model-name", "Renamed", lambda c: c.__name__),
            ("x-aws-stickler-match-threshold", 0.93, lambda c: c.match_threshold),
            (
                "x-aws-stickler-infer-unspecified",
                True,
                lambda c: c._get_comparison_info("n").threshold,
            ),
        ),
    )
    def test_every_model_key_the_check_accepts_changes_the_model(
        self, key, value, read
    ):
        """No key may pass `_reject_unknown_extensions` here and then do nothing."""
        from stickler.structured_object_evaluator.models.json_schema_importer import (
            _KNOWN_MODEL_EXTENSIONS,
        )

        assert key in _KNOWN_MODEL_EXTENSIONS
        plain = read(
            _unwrap(self._child(["object", "null"]).model_fields["child"].annotation)
        )
        declared = read(
            _unwrap(
                self._child(["object", "null"], **{key: value})
                .model_fields["child"]
                .annotation
            )
        )
        assert declared != plain

    def test_the_parametrisation_covers_every_model_key(self):
        from stickler.structured_object_evaluator.models.json_schema_importer import (
            _KNOWN_MODEL_EXTENSIONS,
        )

        assert _KNOWN_MODEL_EXTENSIONS == {
            "x-aws-stickler-model-name",
            "x-aws-stickler-match-threshold",
            "x-aws-stickler-infer-unspecified",
        }

    @_OBJECT_SPELLINGS
    def test_a_near_miss_element_is_a_false_discovery(self, declared):
        """The element scores 0.9, between the declared 0.93 and the default 0.7."""
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": declared,
                            "x-aws-stickler-model-name": "Row",
                            "x-aws-stickler-match-threshold": 0.93,
                            "properties": {
                                "a": {"type": "string"},
                                "b": {"type": "string"},
                            },
                            "required": ["a", "b"],
                        },
                    }
                },
                "required": ["rows"],
            }
        )
        gt = model(rows=[{"a": "alpha", "b": "bravo"}])
        pred = model(rows=[{"a": "alpha", "b": "brava"}])
        overall = gt.compare_with(pred, include_confusion_matrix=True)[
            "confusion_matrix"
        ]["fields"]["rows"]["overall"]
        assert (overall["tp"], overall["fd"], overall["fp"]) == (0, 1, 1)


class TestDefinitionsAreChecked:
    """A `$defs` body is checked like an inline node, and reads only model keys.

    The walk used to enter a definition with no scope, so a typo there imported
    silently while the same typo inline raised. A definition is a type that
    several positions can `$ref`, and `items` does not read field keys, so a
    field key inside one would work at one reference and vanish at the next.
    It goes beside the `$ref` instead, where it is read.

    See https://github.com/awslabs/stickler/issues/340
    """

    INNER = {
        "type": "object",
        "title": "Inner",
        "properties": {"b": {"type": "string"}, "c": {"type": "string"}},
    }

    @staticmethod
    def _via(namespace, body, **ref_extensions):
        return StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "Parent",
                "properties": {
                    "inner": {"$ref": f"#/{namespace}/Inner", **ref_extensions}
                },
                namespace: {"Inner": body},
            }
        )

    @pytest.mark.parametrize("namespace", ("$defs", "definitions"))
    @pytest.mark.parametrize(
        "key", ("x-aws-stickler-thresold", "x-aws-stickler-match-treshold")
    )
    def test_a_typo_raises_as_it_does_inline(self, namespace, key):
        with pytest.raises(ValueError, match="Unrecognized Stickler extension") as exc:
            self._via(namespace, dict(self.INNER, **{key: 0.88}))
        assert key in str(exc.value)
        assert f"{namespace}.Inner" in str(exc.value)

    @pytest.mark.parametrize(
        "key, value",
        (
            ("x-aws-stickler-threshold", 0.88),
            ("x-aws-stickler-weight", 3.0),
            ("x-aws-stickler-comparator", "ExactComparator"),
        ),
    )
    def test_a_field_key_raises_and_names_where_it_belongs(self, key, value):
        with pytest.raises(ValueError) as exc:
            self._via("$defs", dict(self.INNER, **{key: value}))
        message = str(exc.value)
        assert key in message
        assert "belongs on the property that uses this definition" in message

    def test_a_field_key_on_a_scalar_definition_raises(self):
        with pytest.raises(ValueError, match="x-aws-stickler-threshold"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"total": {"$ref": "#/$defs/Money"}},
                    "$defs": {
                        "Money": {"type": "number", "x-aws-stickler-threshold": 0.95}
                    },
                }
            )

    @pytest.mark.parametrize(
        "body",
        (
            {"type": "number"},
            {"type": "array", "items": INNER},
            {"anyOf": [INNER, {"type": "null"}]},
            {"$ref": "#/$defs/Other"},
        ),
        ids=("scalar", "array", "anyOf", "alias"),
    )
    def test_a_model_key_on_a_non_object_definition_raises(self, body):
        """Only an object definition generates a class to configure.

        Accepted here, the key passed the check and the class it was meant for
        was built at the default `0.7`.
        """
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"total": {"$ref": "#/$defs/Value"}},
                    "$defs": {
                        "Value": dict(body, **{"x-aws-stickler-match-threshold": 0.9}),
                        "Other": self.INNER,
                    },
                }
            )
        message = str(exc.value)
        assert "not read on definition '$defs.Value'" in message
        assert "belongs on the object" in message

    def test_the_error_names_the_ref(self):
        with pytest.raises(ValueError, match=r"beside its '\$ref'"):
            self._via("$defs", dict(self.INNER, **{"x-aws-stickler-weight": 3.0}))

    def test_a_field_key_beside_the_ref_is_honoured(self):
        """The documented spelling, so the error's advice works."""
        model = self._via(
            "$defs",
            self.INNER,
            **{"x-aws-stickler-threshold": 0.88, "x-aws-stickler-weight": 3.0},
        )
        info = model._get_comparison_info("inner")
        assert info.threshold == 0.88
        assert info.weight == 3.0

    def test_model_keys_in_a_definition_are_honoured(self):
        model = self._via(
            "$defs",
            dict(
                self.INNER,
                **{
                    "x-aws-stickler-model-name": "Renamed",
                    "x-aws-stickler-match-threshold": 0.91,
                },
            ),
        )
        inner = _unwrap(model.model_fields["inner"].annotation)
        assert inner.__name__ == "Renamed"
        assert inner.match_threshold == 0.91

    def test_a_nullable_definition_reads_its_model_keys(self):
        """Both fixes meet here: a `$defs` body typed `["object", "null"]`."""
        model = self._via(
            "$defs",
            dict(
                self.INNER,
                type=["object", "null"],
                **{"x-aws-stickler-match-threshold": 0.91},
            ),
        )
        assert _unwrap(model.model_fields["inner"].annotation).match_threshold == 0.91

    def test_a_typo_on_a_property_inside_a_definition_names_its_path(self):
        with pytest.raises(ValueError, match=r"\$defs\.Inner\.b"):
            self._via(
                "$defs",
                dict(
                    self.INNER,
                    properties={
                        "b": {"type": "string", "x-aws-stickler-thresold": 0.8}
                    },
                ),
            )

    _ITEMS = (
        {"$ref": "#/$defs/Inner"},
        {"anyOf": [{"$ref": "#/$defs/Inner"}, {"type": "null"}]},
    )

    @pytest.mark.parametrize("items", _ITEMS, ids=("ref", "anyOf-arm"))
    @pytest.mark.parametrize(
        "key, value",
        (("x-aws-stickler-threshold", 0.9), ("x-aws-stickler-weight", 3.0)),
    )
    def test_a_field_key_used_from_items_names_the_array_property(
        self, items, key, value
    ):
        """`items` does not read field keys, so beside the `$ref` is wrong there,
        and an array of objects reads only the weight: the threshold's place is
        taken by the definition's match threshold."""
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"rows": {"type": "array", "items": items}},
                    "$defs": {"Inner": dict(self.INNER, **{key: value})},
                }
            )
        message = str(exc.value)
        assert (
            "When the definition is an array's 'items' or an arm of them, the array "
            "property reads only 'x-aws-stickler-weight', and the elements pair by "
            "this definition's 'x-aws-stickler-match-threshold'"
        ) in message

    @pytest.mark.parametrize("items", _ITEMS, ids=("ref", "anyOf-arm"))
    def test_the_array_property_reads_it(self, items):
        """Where the error sends the author for `items`."""
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": items,
                        "x-aws-stickler-weight": 3.0,
                    }
                },
                "$defs": {"Inner": self.INNER},
            }
        )
        assert model._get_comparison_info("rows").weight == 3.0

    @pytest.mark.parametrize(
        "body, combiner, offered",
        (
            (INNER, "anyOf", "'anyOf', 'oneOf' or 'allOf'"),
            (INNER, "oneOf", "'anyOf', 'oneOf' or 'allOf'"),
            (INNER, "allOf", "'anyOf', 'oneOf' or 'allOf'"),
            ({"type": "string"}, "anyOf", "'anyOf' or 'oneOf'"),
            ({"type": "string"}, "oneOf", "'anyOf' or 'oneOf'"),
            ({"type": "string"}, "allOf", "'anyOf' or 'oneOf'"),
        ),
        ids=(
            "object-anyOf",
            "object-oneOf",
            "object-allOf",
            "scalar-anyOf",
            "scalar-oneOf",
            "scalar-allOf",
        ),
    )
    def test_a_field_key_used_from_a_combiner_arm_names_the_combiner(
        self, body, combiner, offered
    ):
        """Beside the arm's `$ref` raises again, so the error names the combiner.

        A one-arm `allOf` of a scalar fails to import whatever its keys, so a
        scalar definition is not sent there.
        """
        arms = [{"$ref": "#/$defs/Inner"}]
        if combiner != "allOf":
            arms.append({"type": "null"})
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"a": {combiner: arms}},
                    "$defs": {"Inner": dict(body, **{"x-aws-stickler-threshold": 0.9})},
                }
            )
        assert f"beside the {offered} when the '$ref' is one of its arms" in str(
            exc.value
        )

    @pytest.mark.parametrize(
        "body, combiner",
        (
            (INNER, "anyOf"),
            (INNER, "oneOf"),
            (INNER, "allOf"),
            ({"type": "string"}, "anyOf"),
            ({"type": "string"}, "oneOf"),
        ),
        ids=(
            "object-anyOf",
            "object-oneOf",
            "object-allOf",
            "scalar-anyOf",
            "scalar-oneOf",
        ),
    )
    def test_the_property_beside_the_combiner_reads_it(self, body, combiner):
        """Where the error sends the author for a combiner arm."""
        arms = [{"$ref": "#/$defs/Inner"}]
        if combiner != "allOf":
            arms.append({"type": "null"})
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "a": {
                        combiner: arms,
                        "x-aws-stickler-threshold": 0.9,
                        "x-aws-stickler-weight": 2.0,
                    }
                },
                "$defs": {"Inner": body},
            }
        )
        info = model._get_comparison_info("a")
        assert (info.threshold, info.weight) == (0.9, 2.0)

    @staticmethod
    def _alias(end, alias_keys=None, **prop):
        """`a` reaches `end` through the alias `$defs.Alias`."""
        return StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {"a": prop or {"$ref": "#/$defs/Alias"}},
                "$defs": {
                    "Alias": {"$ref": "#/$defs/End", **(alias_keys or {})},
                    "End": end,
                },
            }
        )

    _OF_ALIAS = (
        "the elements pair by the 'x-aws-stickler-match-threshold' of the "
        "object it aliases"
    )
    # A one-arm `allOf` of these fails with `field_definitions must contain at
    # least one field`, directly or through an alias.
    _LIST_TYPED = (["object", "null"], ["null", "object"], ["object"])

    @pytest.mark.parametrize(
        "end, offered, pairs",
        (
            (INNER, "'anyOf', 'oneOf' or 'allOf'", _OF_ALIAS),
            (
                {"properties": INNER["properties"]},
                "'anyOf', 'oneOf' or 'allOf'",
                _OF_ALIAS,
            ),
            (dict(INNER, type=_LIST_TYPED[0]), "'anyOf' or 'oneOf'", _OF_ALIAS),
            (dict(INNER, type=_LIST_TYPED[1]), "'anyOf' or 'oneOf'", _OF_ALIAS),
            (dict(INNER, type=_LIST_TYPED[2]), "'anyOf' or 'oneOf'", _OF_ALIAS),
            ({"type": "string"}, "'anyOf' or 'oneOf'", "on the array property"),
        ),
        ids=(
            "object",
            "untyped",
            "object-null",
            "null-object",
            "one-element-list",
            "scalar",
        ),
    )
    def test_a_field_key_on_an_alias_offers_what_its_end_accepts(
        self, end, offered, pairs
    ):
        """An alias of an object is used as the object is, `allOf` included when
        a one-arm `allOf` of the object imports."""
        with pytest.raises(ValueError) as exc:
            self._alias(end, {"x-aws-stickler-threshold": 0.9})
        message = str(exc.value)
        assert "not read on definition '$defs.Alias'" in message
        assert f"beside the {offered} when the '$ref' is one of its arms" in message
        assert pairs in message

    @pytest.mark.parametrize("declared", _LIST_TYPED, ids=lambda t: "-".join(t))
    def test_a_field_key_on_a_list_typed_definition_offers_no_allof(self, declared):
        with pytest.raises(ValueError) as exc:
            self._via(
                "$defs",
                dict(self.INNER, type=declared, **{"x-aws-stickler-weight": 2.0}),
            )
        message = str(exc.value)
        assert "beside the 'anyOf' or 'oneOf' when the '$ref' is one of its arms" in (
            message
        )
        assert "this definition's 'x-aws-stickler-match-threshold'" in message

    def test_an_alias_typed_as_a_list_offers_no_allof(self):
        """Each hop counts: the alias's own list form breaks the `allOf` too."""
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/$defs/Alias"}},
                    "$defs": {
                        "Alias": {
                            "type": ["object", "null"],
                            "$ref": "#/$defs/End",
                            "x-aws-stickler-weight": 2.0,
                        },
                        "End": self.INNER,
                    },
                }
            )
        assert "beside the 'anyOf' or 'oneOf' when" in str(exc.value)

    @pytest.mark.parametrize("declared", _LIST_TYPED, ids=lambda t: "-".join(t))
    @pytest.mark.parametrize("alias", (False, True), ids=("direct", "alias"))
    def test_a_one_arm_allof_of_a_list_typed_definition_fails(self, declared, alias):
        """Why the advice leaves the `allOf` out. If this starts importing, the
        advice can offer it again."""
        with pytest.raises(ValueError, match="at least one field"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "a": {
                            "allOf": [{"$ref": f"#/$defs/{'A' if alias else 'N'}"}],
                            "x-aws-stickler-threshold": 0.9,
                            "x-aws-stickler-weight": 2.0,
                        }
                    },
                    "$defs": {
                        "N": dict(self.INNER, type=declared),
                        "A": {"$ref": "#/$defs/N"},
                    },
                }
            )

    def test_the_allof_beside_an_alias_of_an_object_reads_it(self):
        """Where the alias error sends the author for an `allOf`."""
        model = self._alias(
            self.INNER,
            allOf=[{"$ref": "#/$defs/Alias"}],
            **{"x-aws-stickler-threshold": 0.9, "x-aws-stickler-weight": 2.0},
        )
        info = model._get_comparison_info("a")
        assert (info.threshold, info.weight) == (0.9, 2.0)

    def test_a_definition_used_as_the_root_says_it_reads_no_field_keys(self):
        """The root is no field, so no position beside a root `$ref` reads one."""
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "$ref": "#/$defs/Inner",
                    "$defs": {
                        "Inner": dict(self.INNER, **{"x-aws-stickler-threshold": 0.9})
                    },
                }
            )
        assert "A definition used as the root reads no field keys" in str(exc.value)

    @pytest.mark.parametrize(
        "body",
        (
            {"type": "object", "additionalProperties": {"type": "string"}},
            {"type": "object"},
        ),
        ids=("mapping", "bare-object"),
    )
    def test_a_model_key_on_a_definition_without_properties_raises(self, body):
        """Such a body becomes a `dict` field and generates no class to configure."""
        with pytest.raises(ValueError, match="not read on definition '\\$defs.Inner'"):
            self._via("$defs", dict(body, **{"x-aws-stickler-match-threshold": 0.9}))

    @pytest.mark.parametrize(
        "body", (INNER, {"type": "string"}), ids=("object", "scalar")
    )
    def test_the_removed_aggregate_key_still_imports(self, body):
        """Accepted and ignored, as on a field, so a schema that imported before
        definitions were checked still imports."""
        model = self._via("$defs", dict(body, **{"x-aws-stickler-aggregate": True}))
        assert "inner" in model.model_fields

    def test_the_removed_aggregate_key_is_not_advertised(self):
        with pytest.raises(ValueError) as exc:
            self._via("$defs", dict(self.INNER, **{"x-aws-stickler-nonsense": 1}))
        assert "x-aws-stickler-aggregate" not in str(exc.value)


_KEYED = dict(
    TestDefinitionsAreChecked.INNER,
    **{
        "x-aws-stickler-model-name": "Renamed",
        "x-aws-stickler-match-threshold": 0.93,
    },
)
_PLAIN = {"type": "object", "properties": {"z": {"type": "string"}}}
_TYPO = {"type": "string", "x-aws-stickler-thresold": 0.9}


def _unused(body, namespace="$defs"):
    """A schema whose only definition nothing references."""
    return {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        namespace: {"Unused": body},
    }


class TestModelKeysWhereNoClassIsBuiltFromThem:
    """Model keys raise where they cannot reach the class they name.

    An `allOf` arm is merged into one class at the default name and match
    threshold, and a root `$ref` builds the root class from the root's own keys.
    Model keys on such an arm, or on a definition reached that way, passed the
    check and were dropped: a declared `0.93` built at `0.7`. See #340.
    """

    @staticmethod
    def _import(schema):
        return StructuredModel.from_json_schema(schema)

    @pytest.mark.parametrize(
        "schema, reached",
        (
            (
                {
                    "type": "object",
                    "properties": {"a": {"allOf": [{"$ref": "#/$defs/Addr"}]}},
                    "$defs": {"Addr": _KEYED},
                },
                "'allOf' at 'a'",
            ),
            (
                {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "array",
                            "items": {"allOf": [{"$ref": "#/$defs/Addr"}]},
                        }
                    },
                    "$defs": {"Addr": _KEYED},
                },
                "'allOf' at 'a[]'",
            ),
            (
                {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/$defs/Comp"}},
                    "$defs": {
                        "Addr": _KEYED,
                        "Comp": {"allOf": [{"$ref": "#/$defs/Addr"}]},
                    },
                },
                "'allOf' at '$defs.Comp'",
            ),
            (
                {
                    "type": "object",
                    "properties": {"a": {"allOf": [{"$ref": "#/definitions/Alias"}]}},
                    "definitions": {
                        "Addr": _KEYED,
                        "Alias": {"$ref": "#/definitions/Addr"},
                    },
                },
                "'allOf' at 'a'",
            ),
            (
                {"allOf": [{"$ref": "#/$defs/Addr"}], "$defs": {"Addr": _KEYED}},
                "'allOf' at '(root)'",
            ),
            (
                {"$ref": "#/$defs/Addr", "$defs": {"Addr": _KEYED}},
                "the root '$ref'",
            ),
        ),
        ids=("property", "items", "definition", "alias", "root-allOf", "root-ref"),
    )
    def test_a_definition_reached_there_raises(self, schema, reached):
        with pytest.raises(ValueError) as exc:
            self._import(schema)
        message = str(exc.value)
        assert "x-aws-stickler-match-threshold" in message
        assert "Addr'" in message
        assert f"reached through {reached}" in message

    def test_an_inline_allof_arm_raises(self):
        with pytest.raises(ValueError, match=r"object 'a\.allOf\[0\]'"):
            self._import({"type": "object", "properties": {"a": {"allOf": [_KEYED]}}})

    def test_the_root_reads_keys_beside_its_ref(self):
        """Where the root `$ref` error sends the author."""
        model = self._import(
            {
                "$ref": "#/$defs/Addr",
                "x-aws-stickler-model-name": "Root",
                "x-aws-stickler-match-threshold": 0.93,
                "$defs": {"Addr": _PLAIN},
            }
        )
        assert (model.__name__, model.match_threshold) == ("Root", 0.93)

    @pytest.mark.parametrize(
        "prop",
        (
            {"$ref": "#/$defs/Addr"},
            {"type": "array", "items": {"$ref": "#/$defs/Addr"}},
            {"anyOf": [{"$ref": "#/$defs/Addr"}, {"type": "null"}]},
            {"anyOf": [{"$ref": "#/$defs/Addr"}, {"$ref": "#/$defs/Plain"}]},
        ),
        ids=("property", "items", "nullable", "union"),
    )
    def test_the_positions_that_build_it_still_read_them(self, prop):
        model = self._import(
            {
                "type": "object",
                "properties": {"a": prop},
                "$defs": {"Addr": _KEYED, "Plain": _PLAIN},
            }
        )

        def classes(annotation):
            yield annotation
            for arg in typing.get_args(annotation):
                yield from classes(arg)

        built = {
            (c.__name__, c.match_threshold)
            for c in classes(model.model_fields["a"].annotation)
            if isinstance(c, type) and issubclass(c, StructuredModel)
        }
        assert ("Renamed", 0.93) in built

    def test_an_unused_definition_is_left_alone(self):
        self._import(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "$defs": {"Addr": _KEYED},
            }
        )

    _INSIDE = (
        ({"allOf": [{"$ref": "#/components/schemas/Addr"}]}, "reached through 'allOf'"),
        (
            {
                "type": "array",
                "items": {
                    "$ref": "#/components/schemas/Row",
                    "x-aws-stickler-match-threshold": 0.93,
                },
            },
            "beside the '$ref' at 'components.schemas.Holder.w[]'",
        ),
        (
            {
                "type": "object",
                "$ref": "#/components/schemas/Row",
                "x-aws-stickler-match-threshold": 0.93,
            },
            "beside the '$ref' at 'components.schemas.Holder.w'",
        ),
    )

    @staticmethod
    def _components(inner, reached):
        props = {"a": {"type": "string"}}
        if reached:
            props["c"] = {"$ref": "#/components/schemas/Holder"}
        holder = {"type": "object", "properties": {"w": inner}}
        return {
            "type": "object",
            "properties": props,
            "components": {
                "schemas": {"Holder": holder, "Addr": _KEYED, "Row": _PLAIN}
            },
        }

    @pytest.mark.parametrize(
        "inner, error", _INSIDE, ids=("allOf-arm", "beside-items-ref", "typed-ref")
    )
    def test_inside_a_components_target_a_class_is_built_from(self, inner, error):
        with pytest.raises(ValueError, match="x-aws-stickler-match-threshold") as exc:
            self._import(self._components(inner, reached=True))
        assert error in str(exc.value)

    @pytest.mark.parametrize(
        "inner", [inner for inner, _ in _INSIDE], ids=("allOf", "items", "typed")
    )
    def test_inside_a_components_target_nothing_references_is_left_alone(self, inner):
        self._import(self._components(inner, reached=False))

    def test_a_root_ref_to_a_components_target_raises(self):
        with pytest.raises(ValueError) as exc:
            self._import(
                {
                    "$ref": "#/components/schemas/Addr",
                    "components": {"schemas": {"Addr": _KEYED}},
                }
            )
        message = str(exc.value)
        assert "on object 'components.schemas.Addr'" in message
        assert "reached through the root '$ref'" in message

    def test_a_root_ref_to_a_components_target_reads_the_root_keys(self):
        model = self._import(
            {
                "$ref": "#/components/schemas/Addr",
                "x-aws-stickler-match-threshold": 0.93,
                "components": {"schemas": {"Addr": _PLAIN}},
            }
        )
        assert model.match_threshold == 0.93


class TestTheRootMustBeOneObject:
    """A union root crashed with `AttributeError: model_fields`."""

    @pytest.mark.parametrize(
        "root",
        (
            {"type": ["object", "null"]},
            {"type": ["null", "object"]},
            {"anyOf": [_PLAIN, {"type": "null"}]},
            {"oneOf": [_PLAIN, TestDefinitionsAreChecked.INNER]},
        ),
        ids=("object-null", "null-object", "anyOf", "oneOf"),
    )
    def test_a_union_root_raises_a_value_error(self, root):
        schema = dict(root)
        if "type" in schema:
            schema["properties"] = {"c": {"type": "string"}}
        with pytest.raises(ValueError, match="root of a JSON Schema must be a single"):
            StructuredModel.from_json_schema(schema)

    @pytest.mark.parametrize(
        "body",
        (dict(_PLAIN, type=["object", "null"]), {"anyOf": [_PLAIN, {"type": "null"}]}),
        ids=("object-null", "anyOf"),
    )
    def test_a_root_ref_to_a_union_names_the_ref(self, body):
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {"$ref": "#/$defs/N", "$defs": {"N": body}}
            )
        message = str(exc.value)
        assert "root of a JSON Schema must be a single" in message
        assert "or a root '$ref' to either" in message

    def test_a_single_element_list_still_imports(self):
        model = StructuredModel.from_json_schema(
            {"type": ["object"], "properties": {"c": {"type": "string"}}}
        )
        assert "c" in model.model_fields


class TestOnlyReachedDefinitionsAreChecked:
    """A `$defs` entry is checked as a definition once a `$ref` reaches it.

    A dereferenced schema inlines every `$ref` and keeps `$defs`, so the
    originals carry keys that are read on their inlined copies. Checking every
    entry raised on such a schema, which imported and scored correctly before.
    An entry no `$ref` reaches is still walked for its contents, as before, so
    a typo in one of its properties raises.
    """

    def test_a_dereferenced_schema_imports_and_reads_the_inlined_keys(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "billTo": {
                        "type": "object",
                        "properties": {
                            "state": {
                                "type": "string",
                                "x-aws-stickler-comparator": "ExactComparator",
                            }
                        },
                        "x-aws-stickler-weight": 3.0,
                    }
                },
                "$defs": {
                    "Address": {
                        "type": "object",
                        "properties": {"state": {"$ref": "#/$defs/State"}},
                        "x-aws-stickler-weight": 3.0,
                    },
                    "State": {
                        "type": "string",
                        "x-aws-stickler-comparator": "ExactComparator",
                    },
                },
            }
        )
        assert model._get_comparison_info("billTo").weight == 3.0
        bill_to = _unwrap(model.model_fields["billTo"].annotation)
        state = bill_to._get_comparison_info("state")
        assert type(state.comparator).__name__ == "ExactComparator"

    @pytest.mark.parametrize(
        "body",
        (
            dict(TestDefinitionsAreChecked.INNER, **{"x-aws-stickler-thresold": 0.9}),
            dict(TestDefinitionsAreChecked.INNER, **{"x-aws-stickler-weight": 3.0}),
            {"type": "number", "x-aws-stickler-match-threshold": 0.9},
        ),
        ids=("typo", "field-key", "model-key-on-scalar"),
    )
    def test_the_own_keys_of_an_unreferenced_definition_are_not_checked(self, body):
        StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "$defs": {"Unused": body},
            }
        )

    @pytest.mark.parametrize(
        "schema, at",
        (
            (
                _unused({"type": "object", "properties": {"p": _TYPO}}, "definitions"),
                "field 'definitions.Unused.p'",
            ),
            (
                _unused({"type": "object", "properties": {"p": _TYPO}}),
                "field '$defs.Unused.p'",
            ),
            (
                _unused(
                    {
                        "type": "object",
                        "properties": {
                            "p": {
                                "type": "string",
                                "x-aws-stickler-match-threshold": 0.9,
                            }
                        },
                    }
                ),
                "field '$defs.Unused.p'",
            ),
            (
                _unused({"type": "object", "patternProperties": {"^x": _PLAIN}}),
                "at '$defs.Unused'",
            ),
            (
                _unused({"anyOf": [_PLAIN], "type": "object"}),
                "for field '$defs.Unused'",
            ),
            (
                _unused({"allOf": [dict(_PLAIN, **{"x-aws-stickler-weight": 2.0})]}),
                "object '$defs.Unused[0]'",
            ),
            (
                _unused(
                    dict(
                        _PLAIN,
                        **{
                            "$defs": {
                                "N": {"type": "object", "properties": {"p": _TYPO}}
                            }
                        },
                    )
                ),
                "field '$defs.Unused.$defs.N.p'",
            ),
            (
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "$defs": {
                        "Outer": {
                            "type": "object",
                            "properties": {"i": {"$ref": "#/$defs/In"}},
                        },
                        "In": {"type": "object", "properties": {"p": _TYPO}},
                    },
                },
                "field '$defs.In.p'",
            ),
            (
                {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/$defs/Outer"}},
                    "$defs": {
                        "Outer": dict(
                            _PLAIN,
                            **{
                                "$defs": {
                                    "N": {"type": "object", "properties": {"p": _TYPO}}
                                }
                            },
                        )
                    },
                },
                "field '$defs.Outer.$defs.N.p'",
            ),
        ),
        ids=(
            "definitions",
            "typo",
            "misplaced-key",
            "patternProperties",
            "combiner-sibling",
            "allOf-arm",
            "nested",
            "reached-only-from-an-unused-entry",
            "nested-in-a-reached-entry",
        ),
    )
    def test_the_contents_of_an_unreferenced_definition_are_checked(self, schema, at):
        """As before definitions were checked: only the entry's own keys are new."""
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(schema)
        assert at in str(exc.value)

    @pytest.mark.parametrize(
        "body",
        (
            {"allOf": [_KEYED]},
            {
                "type": "object",
                "properties": {
                    "r": {
                        "type": "array",
                        "items": {
                            "$ref": "#/$defs/Row",
                            "x-aws-stickler-match-threshold": 0.9,
                        },
                    }
                },
                "$defs": {"Row": _PLAIN},
            },
        ),
        ids=("model-key-on-an-allOf-arm", "model-key-beside-a-ref"),
    )
    def test_an_unreferenced_definition_builds_nothing_to_drop(self, body):
        """These raise only where a class is built, and nothing builds it."""
        StructuredModel.from_json_schema(_unused(body))

    @pytest.mark.parametrize("keyword", ("default", "examples"))
    def test_definitions_inside_data_are_not_checked(self, keyword):
        data = {"a": "x", "$defs": {"X": {"x-aws-stickler-weight": 2.0}}}
        value = [data] if keyword == "examples" else data
        StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "cfg": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                        keyword: value,
                    }
                },
            }
        )

    @pytest.mark.parametrize("keyword", ("default", "x-vendor"))
    def test_a_typo_in_a_definition_inside_data_still_raises(self, keyword):
        """Data and `x-*` payloads are not walked, but a `$defs` entry inside one
        was checked for its contents before, so it still is."""
        with pytest.raises(ValueError, match=rf"field '{keyword}\.\$defs\.U\.p'"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    keyword: {
                        "$defs": {"U": {"type": "object", "properties": {"p": _TYPO}}}
                    },
                }
            )

    def test_a_definition_reached_through_another_is_checked(self):
        with pytest.raises(ValueError, match=r"thresold.*definition '\$defs\.Inner'"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"o": {"$ref": "#/$defs/Outer"}},
                    "$defs": {
                        "Outer": {
                            "type": "object",
                            "properties": {"i": {"$ref": "#/$defs/Inner"}},
                        },
                        "Inner": dict(
                            TestDefinitionsAreChecked.INNER,
                            **{"x-aws-stickler-thresold": 0.9},
                        ),
                    },
                }
            )


class TestValuesAreNotWalkedAsSchemas:
    """An `x-*` payload or a data keyword holds a value, not a schema.

    Walking one as a schema raised for keys that nothing reads as Stickler
    configuration: a vendor's own schema copy, or a comparator config whose
    data happens to look like a schema.
    """

    _SHAPES = {
        "typo": {"properties": {"p": _TYPO}},
        "misplaced-key": {
            "properties": {
                "p": {"type": "string", "x-aws-stickler-match-threshold": 0.9}
            }
        },
        "patternProperties": {"patternProperties": {"^x": {"type": "string"}}},
        "combiner-sibling": {"anyOf": [_PLAIN], "type": "object"},
    }

    @pytest.mark.parametrize("shape", sorted(_SHAPES))
    @pytest.mark.parametrize(
        "keyword, on_root",
        (
            ("x-vendor", True),
            ("x-vendor", False),
            ("default", False),
            ("examples", False),
        ),
        ids=("root-x-vendor", "property-x-vendor", "default", "examples"),
    )
    def test_a_schema_shaped_value_imports(self, keyword, on_root, shape):
        value = self._SHAPES[shape]
        if keyword == "examples":
            value = [value]
        schema = {"type": "object", "properties": {"a": dict(_PLAIN)}}
        (schema if on_root else schema["properties"]["a"])[keyword] = value
        StructuredModel.from_json_schema(schema)

    @pytest.mark.parametrize("keyword", ("enum", "const"))
    def test_a_schema_shaped_enum_or_const_imports(self, keyword):
        value = {"properties": {"p": _TYPO}}
        StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {"a": {keyword: [value] if keyword == "enum" else value}},
            }
        )

    def test_a_comparator_config_shaped_like_allof_imports(self):
        """The case from review: a config whose data is an `allOf` of model keys.

        The comparator ignores the unknown parameter, with a warning.
        """
        prop = {
            "type": "string",
            "x-aws-stickler-comparator": "ExactComparator",
            "x-aws-stickler-comparator-config": {
                "allOf": [{"x-aws-stickler-model-name": "X"}]
            },
        }
        with pytest.warns(UserWarning, match="does not accept 'allOf'"):
            model = StructuredModel.from_json_schema(
                {"type": "object", "properties": {"p": prop}}
            )
        assert "p" in model.model_fields

    @pytest.mark.parametrize(
        "config, ignored",
        (
            ({"properties": {"p": _TYPO}}, "'properties'"),
            ({"patternProperties": {"^x": {"type": "string"}}}, "'patternProperties'"),
        ),
        ids=("typo", "patternProperties"),
    )
    def test_a_comparator_config_shaped_like_a_schema_imports(self, config, ignored):
        """Walked as a schema, this config raised for the typo and for
        `patternProperties`, though the comparator only ignores the keys."""
        prop = {
            "type": "string",
            "x-aws-stickler-comparator": "ExactComparator",
            "x-aws-stickler-comparator-config": config,
        }
        with pytest.warns(UserWarning, match=f"does not accept {ignored}"):
            model = StructuredModel.from_json_schema(
                {"type": "object", "properties": {"p": prop}}
            )
        assert "p" in model.model_fields

    def test_an_x_payload_a_ref_reaches_is_checked(self):
        """Once a `$ref` builds from it, the payload is a schema."""
        with pytest.raises(ValueError, match=r"thresold.*'x-models\.Team\.p'"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"t": {"$ref": "#/x-models/Team"}},
                    "x-models": {
                        "Team": {"type": "object", "properties": {"p": _TYPO}}
                    },
                }
            )

    @pytest.mark.parametrize("holder", ("$defs", "default"))
    def test_an_x_payload_only_an_unused_definition_references_is_checked(self, holder):
        """An unused definition builds nothing, but its contents are checked as
        before, and that followed its `$ref`s: the typo raised before and imported
        once the queue they went to was no longer drained."""
        outer = {"type": "object", "properties": {"i": {"$ref": "#/x-models/Team"}}}
        definitions = {"$defs": {"Outer": outer}}
        with pytest.raises(ValueError, match=r"thresold.*'x-models\.Team\.p'"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "x-models": {
                        "Team": {"type": "object", "properties": {"p": _TYPO}}
                    },
                    **(definitions if holder == "$defs" else {holder: definitions}),
                }
            )

    def test_an_x_payload_only_an_unused_definition_references_keeps_its_own_keys(
        self,
    ):
        """Only its contents are checked: its own keys build nothing."""
        StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "x-models": {"Team": dict(_PLAIN, **{"x-aws-stickler-thresold": 0.9})},
                "$defs": {
                    "Outer": {
                        "type": "object",
                        "properties": {"i": {"$ref": "#/x-models/Team"}},
                    }
                },
            }
        )


class TestAllOfRaisesOnlyWhereNothingReadsTheKeys:
    """An `allOf` arm drops its model keys, but the object it names may not.

    Raising whenever an `allOf` arm reached a keyed object took away classes
    that another position built from the same keys, and raised for an `allOf`
    that builds nothing.
    """

    def test_a_definition_also_referenced_directly_is_built_from_them(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "home": {"$ref": "#/$defs/Addr"},
                    "work": {"allOf": [{"$ref": "#/$defs/Addr"}]},
                },
                "$defs": {"Addr": _KEYED},
            }
        )
        home = _unwrap(model.model_fields["home"].annotation)
        assert (home.__name__, home.match_threshold) == ("Renamed", 0.93)

    def test_a_keyed_property_referenced_from_allof_is_built_from_them(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "home": _KEYED,
                    "work": {"allOf": [{"$ref": "#/properties/home"}]},
                },
            }
        )
        home = _unwrap(model.model_fields["home"].annotation)
        assert (home.__name__, home.match_threshold) == ("Renamed", 0.93)

    @pytest.mark.parametrize(
        "schema",
        (
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "$defs": {
                    "Addr": _KEYED,
                    "Employee": {"allOf": [{"$ref": "#/$defs/Addr"}]},
                },
            },
            {
                "type": "object",
                "properties": {
                    "p": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                        "not": {"allOf": [{"$ref": "#/$defs/Addr"}]},
                    }
                },
                "$defs": {"Addr": _KEYED},
            },
            {
                "type": "object",
                "properties": {
                    "p": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                        "not": {"allOf": [_KEYED]},
                    }
                },
            },
        ),
        ids=("unused-definition", "not-ref", "not-inline"),
    )
    def test_an_allof_that_builds_nothing_is_left_alone(self, schema):
        StructuredModel.from_json_schema(schema)

    @staticmethod
    def _team_schema(where, lead):
        """`team` builds `Team` from a `$ref` outside `$defs`; `employee` reaches
        `Addr` only through `allOf`."""
        if where == "components":
            team_ref, addr_ref = (
                "#/components/schemas/Team",
                "#/components/schemas/Addr",
            )
            holder = {"components": {"schemas": {"Addr": _KEYED}}}
            teams = holder["components"]["schemas"]
        else:
            team_ref, addr_ref = "#/x-models/Team", "#/$defs/Addr"
            holder = {"x-models": {}, "$defs": {"Addr": _KEYED}}
            teams = holder["x-models"]
        teams["Team"] = {
            "type": "object",
            "properties": {"lead": lead(addr_ref)},
        }
        return {
            "type": "object",
            "properties": {
                "team": {"$ref": team_ref},
                "employee": {"allOf": [{"$ref": addr_ref}]},
            },
            **holder,
        }

    @pytest.mark.parametrize("where", ("components", "x-models"))
    def test_a_reader_inside_a_ref_target_outside_defs_counts(self, where):
        """`team.lead` builds `Addr` from its keys, so the `allOf` drops nothing.

        Only `$defs` entries were walked from their references, so a reader
        inside `#/components/...`, or under an `x-*` key, did not count and the
        import raised.
        """
        model = StructuredModel.from_json_schema(
            self._team_schema(where, lambda ref: {"$ref": ref})
        )
        team = _unwrap(model.model_fields["team"].annotation)
        lead = _unwrap(team.model_fields["lead"].annotation)
        assert (lead.__name__, lead.match_threshold) == ("Renamed", 0.93)

    @pytest.mark.parametrize("where", ("components", "x-models"))
    @pytest.mark.parametrize(
        "lead",
        (
            lambda ref: {"type": "string"},
            lambda ref: {"type": "string", "not": {"$ref": ref}},
        ),
        ids=("no-reader", "not"),
    )
    def test_a_ref_target_outside_defs_without_a_reader_still_raises(self, where, lead):
        with pytest.raises(ValueError, match="reached through 'allOf' at 'employee'"):
            StructuredModel.from_json_schema(self._team_schema(where, lead))

    def test_a_ref_target_reached_where_nothing_builds_does_not_count(self):
        schema = self._team_schema("components", lambda ref: {"$ref": ref})
        schema["properties"]["team"] = {
            "type": "string",
            "not": {"$ref": "#/components/schemas/Team"},
        }
        with pytest.raises(ValueError, match="reached through 'allOf' at 'employee'"):
            StructuredModel.from_json_schema(schema)

    @pytest.mark.parametrize(
        "other",
        (
            {"type": "string", "not": {"$ref": "#/$defs/Addr"}},
            {"type": "object", "additionalProperties": {"$ref": "#/$defs/Addr"}},
            {"type": "string", "if": {"$ref": "#/$defs/Addr"}},
        ),
        ids=("not", "additionalProperties", "if"),
    )
    def test_a_reference_that_builds_nothing_does_not_excuse_it(self, other):
        # These positions reach `Addr` but build no class from it, so its model
        # keys are still read nowhere and the `allOf` must raise.
        with pytest.raises(ValueError, match="allOf"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "w": {"allOf": [{"$ref": "#/$defs/Addr"}]},
                        "x": other,
                    },
                    "$defs": {"Addr": _KEYED},
                }
            )


class TestARefIntoADefinition:
    """A `$ref` that points inside a `$defs` entry builds that node, not the entry.

    The whole entry was walked with the reference's flag. Where the reference
    builds a class, every reader in the entry counted, so an `allOf` arm or a
    root `$ref` to `#/$defs/Wrap/properties/inner` built the default class at
    `0.7` without raising. Where the target sat under `not` or an `x-*` key in
    the entry, it was walked as building nothing, so the class it builds did not
    count and an `allOf` elsewhere raised where the keys were read.
    """

    DEFS = {
        "Wrap": {"type": "object", "properties": {"inner": _KEYED}},
        "List": {"type": "array", "items": _KEYED},
    }

    @pytest.mark.parametrize(
        "pointer, target",
        (
            ("#/$defs/Wrap/properties/inner", "$defs.Wrap.properties.inner"),
            ("#/$defs/List/items", "$defs.List.items"),
        ),
        ids=("property", "items"),
    )
    @pytest.mark.parametrize(
        "make, reached",
        (
            (lambda ref: {"$ref": ref}, "the root '$ref'"),
            (
                lambda ref: {
                    "type": "object",
                    "properties": {"a": {"allOf": [{"$ref": ref}]}},
                },
                "'allOf' at 'a'",
            ),
        ),
        ids=("root-ref", "allOf"),
    )
    def test_keys_nothing_reads_raise(self, pointer, target, make, reached):
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                dict(make(pointer), **{"$defs": self.DEFS})
            )
        message = str(exc.value)
        assert f"on object '{target}' is not read" in message
        assert f"reached through {reached}" in message

    @pytest.mark.parametrize("where", ("not", "x-sub", "additionalProperties", "then"))
    def test_a_reader_anywhere_in_the_entry_counts(self, where):
        """`team.lead` builds `Person` from its keys, so the `allOf` drops nothing."""
        team = {"type": "object", "properties": {"lead": {"$ref": "#/$defs/Person"}}}
        holder = {"type": "object", "properties": {"z": {"type": "string"}}}
        if where == "x-sub":
            holder[where] = {"Team": team}
            ref = "#/$defs/Holder/x-sub/Team"
        else:
            holder[where] = team
            ref = f"#/$defs/Holder/{where}"
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "team": {"$ref": ref},
                    "employee": {"allOf": [{"$ref": "#/$defs/Person"}]},
                },
                "$defs": {"Person": _KEYED, "Holder": holder},
            }
        )
        team_class = _unwrap(model.model_fields["team"].annotation)
        lead = _unwrap(team_class.model_fields["lead"].annotation)
        assert (lead.__name__, lead.match_threshold) == ("Renamed", 0.93)

    @pytest.mark.parametrize(
        "ref, fields",
        (("#/$defs/Wrap", ("inner",)), ("#/$defs/Wrap/properties/inner", ())),
        ids=("the-entry", "the-node"),
    )
    def test_a_position_that_builds_the_node_still_reads_it(self, ref, fields):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "w": {"$ref": ref},
                    "a": {"allOf": [{"$ref": "#/$defs/Wrap/properties/inner"}]},
                },
                "$defs": self.DEFS,
            }
        )
        built = _unwrap(model.model_fields["w"].annotation)
        for name in fields:
            built = _unwrap(built.model_fields[name].annotation)
        assert (built.__name__, built.match_threshold) == ("Renamed", 0.93)

    # What the node holds, and the error each raises once it is built.
    _INSIDE = (
        (
            {"allOf": [{"$ref": "#/$defs/Addr"}]},
            "on definition '$defs.Addr' is not read when it is reached through "
            "'allOf' at '{at}.w'",
        ),
        (
            {
                "type": "array",
                "items": {
                    "$ref": "#/$defs/Row",
                    "x-aws-stickler-match-threshold": 0.93,
                },
            },
            "beside the '$ref' at '{at}.w[]' is not read",
        ),
        (
            {
                "type": "object",
                "$ref": "#/$defs/Row",
                "x-aws-stickler-match-threshold": 0.93,
            },
            "beside the '$ref' at '{at}.w' is not read",
        ),
    )

    @staticmethod
    def _holder(where, inner, ref):
        """`Holder` keeps an object holding `w` under `where`; `t` uses `ref`."""
        node = {"type": "object", "properties": {"w": inner}}
        holder = {"type": "object", "properties": {"z": {"type": "string"}}}
        holder[where] = {"T": node} if where == "x-sub" else node
        props = {"a": {"type": "string"}}
        if ref is not None:
            props["t"] = {"$ref": ref}
        return {
            "type": "object",
            "properties": props,
            "$defs": {"Holder": holder, "Addr": _KEYED, "Row": _PLAIN},
        }

    @pytest.mark.parametrize("where", ("not", "then", "additionalProperties", "x-sub"))
    @pytest.mark.parametrize(
        "inner, error", _INSIDE, ids=("allOf-arm", "beside-items-ref", "typed-ref")
    )
    def test_inside_a_node_a_class_is_built_from_the_checks_apply(
        self, where, inner, error
    ):
        """As inside a `#/components/...` target. Dev builds these at the
        defaults (`AllOfModel@0.7`, `Row@0.7`), dropping the keys."""
        pointer = "/x-sub/T" if where == "x-sub" else f"/{where}"
        at = "$defs.Holder" + pointer.replace("/", ".")
        with pytest.raises(ValueError, match="x-aws-stickler-match-threshold") as exc:
            StructuredModel.from_json_schema(
                self._holder(where, inner, f"#/$defs/Holder{pointer}")
            )
        assert error.format(at=at) in str(exc.value)

    @pytest.mark.parametrize(
        "where, ref",
        (
            ("not", None),
            ("then", None),
            ("additionalProperties", None),
            ("x-sub", None),
            ("not", "#/$defs/Holder"),
            ("then", "#/$defs/Holder"),
            ("additionalProperties", "#/$defs/Holder"),
        ),
        # The parser fails a `$ref` under an `x-*` key of an entry it builds
        # (`KeyError`, as on dev), so that case has no whole-entry variant.
        ids=(
            "unreferenced-not",
            "unreferenced-then",
            "unreferenced-additionalProperties",
            "unreferenced-x-sub",
            "whole-entry-not",
            "whole-entry-then",
            "whole-entry-additionalProperties",
        ),
    )
    @pytest.mark.parametrize(
        "inner", [inner for inner, _ in _INSIDE], ids=("allOf", "items", "typed")
    )
    def test_where_nothing_builds_the_node_it_imports(self, where, ref, inner):
        """The entry's walk builds nothing under these keys."""
        StructuredModel.from_json_schema(self._holder(where, inner, ref))

    def test_the_entry_around_the_node_is_still_checked(self):
        with pytest.raises(ValueError, match=r"thresold.*definition '\$defs\.Wrap'"):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/$defs/Wrap/properties/inner"}},
                    "$defs": {
                        "Wrap": dict(
                            self.DEFS["Wrap"], **{"x-aws-stickler-thresold": 0.9}
                        )
                    },
                }
            )


class TestAListFormTypeBesideARef:
    """`["object", "null"]` beside a `$ref` builds one class, as `"object"` does.

    The list form is rewritten to `anyOf`, and the `$ref` was copied onto the
    null arm too, so the node became a union of two classes compared with
    ExactComparator: a one-character difference scored 0.0, not 0.9.
    """

    @staticmethod
    def _model(declared, inner=TestDefinitionsAreChecked.INNER):
        return StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {"child": {"type": declared, "$ref": "#/$defs/Inner"}},
                "$defs": {"Inner": inner},
            }
        )

    @pytest.mark.parametrize(
        "declared",
        ("object", ["object", "null"], ["null", "object"]),
        ids=("object", "object-null", "null-object"),
    )
    def test_it_builds_one_class_and_scores_like_object(self, declared):
        model = self._model(declared)
        annotation = model.model_fields["child"].annotation
        arms = [a for a in typing.get_args(annotation) if a is not type(None)]
        assert len(arms) == 1 and issubclass(arms[0], StructuredModel)
        info = model._get_comparison_info("child")
        assert type(info.comparator).__name__ == "LevenshteinComparator"
        gt = model(child={"b": "alpha", "c": "bravo"})
        pred = model(child={"b": "alpha", "c": "brava"})
        assert gt.compare_with(pred)["overall_score"] == pytest.approx(0.9)

    def test_it_reads_the_definition_model_keys(self):
        child = _unwrap(
            self._model(["object", "null"], _KEYED).model_fields["child"].annotation
        )
        assert (child.__name__, child.match_threshold) == ("Renamed", 0.93)


class TestModelKeysBesideANonRootRef:
    """Model keys beside a `$ref` are read at the root and nowhere else.

    Elsewhere the class is built from the `$ref` target, so the keys passed the
    check and were dropped: `items: {"$ref": Row, match-threshold: 0.93}`
    built `Row` at 0.7.
    """

    @pytest.mark.parametrize(
        "key, value",
        (
            ("x-aws-stickler-model-name", "Renamed"),
            ("x-aws-stickler-match-threshold", 0.93),
            ("x-aws-stickler-infer-unspecified", True),
        ),
    )
    @pytest.mark.parametrize(
        "make, at",
        (
            (
                lambda ext: {"type": "array", "items": {"$ref": "#/$defs/Row", **ext}},
                "a[]",
            ),
            (
                lambda ext: {
                    "anyOf": [{"$ref": "#/$defs/Row", **ext}, {"type": "null"}]
                },
                "a[0]",
            ),
            (
                lambda ext: {
                    "oneOf": [{"$ref": "#/$defs/Row", **ext}, {"type": "null"}]
                },
                "a[0]",
            ),
            (lambda ext: {"type": "object", "$ref": "#/$defs/Row", **ext}, "a"),
            (
                lambda ext: {"type": ["object", "null"], "$ref": "#/$defs/Row", **ext},
                "a",
            ),
        ),
        ids=("items", "anyOf", "oneOf", "object", "object-null"),
    )
    def test_it_raises_and_names_the_target(self, make, at, key, value):
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {"a": make({key: value})},
                    "$defs": {"Row": TestDefinitionsAreChecked.INNER},
                }
            )
        message = str(exc.value)
        assert f"'{key}' beside the '$ref' at '{at}' is not read" in message
        assert "'$defs.Row'" in message

    def test_through_an_alias_it_names_the_end_of_the_chain(self):
        """The alias reads no model keys, so naming it sent the author to a
        position that raises too."""
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "array",
                            "items": {
                                "$ref": "#/$defs/Alias",
                                "x-aws-stickler-match-threshold": 0.93,
                            },
                        }
                    },
                    "$defs": {
                        "Alias": {"$ref": "#/$defs/Row"},
                        "Row": TestDefinitionsAreChecked.INNER,
                    },
                }
            )
        assert "the class is built from definition '$defs.Row'" in str(exc.value)

    # A node inside an entry that carries the keys beside its own `$ref`, and
    # the error path the entry's walk gives it.
    _NODES = {
        "List": (
            {
                "type": "array",
                "items": {
                    "$ref": "#/$defs/Row",
                    "x-aws-stickler-match-threshold": 0.93,
                },
            },
            "#/$defs/List/items",
            "$defs.List[]",
        ),
        "W": (
            {
                "type": "object",
                "properties": {
                    "inner": {
                        "type": "object",
                        "$ref": "#/$defs/Row",
                        "x-aws-stickler-match-threshold": 0.93,
                    }
                },
            },
            "#/$defs/W/properties/inner",
            "$defs.W.inner",
        ),
        "N": (
            {
                "type": "object",
                "properties": {
                    "inner": {
                        "type": ["object", "null"],
                        "$ref": "#/$defs/Row",
                        "x-aws-stickler-match-threshold": 0.93,
                    }
                },
            },
            "#/$defs/N/properties/inner",
            "$defs.N.inner",
        ),
    }

    @classmethod
    def _node_schema(cls, entry, use):
        body, pointer, _ = cls._NODES[entry]
        return {
            "type": "object",
            "properties": {"a": use(pointer)},
            "$defs": {entry: body, "Row": TestDefinitionsAreChecked.INNER},
        }

    @pytest.mark.parametrize(
        "entry, use",
        (
            ("List", lambda ref: {"$ref": ref}),
            ("List", lambda ref: {"type": "array", "items": {"$ref": ref}}),
            ("List", lambda ref: {"anyOf": [{"$ref": ref}, {"type": "null"}]}),
            ("W", lambda ref: {"$ref": ref}),
            ("W", lambda ref: {"type": "array", "items": {"$ref": ref}}),
            ("N", lambda ref: {"$ref": ref}),
            ("N", lambda ref: {"type": "array", "items": {"$ref": ref}}),
            ("List", lambda ref: {"$ref": "#/$defs/List"}),
            ("W", lambda ref: {"$ref": "#/$defs/W"}),
            ("N", lambda ref: {"$ref": "#/$defs/N"}),
        ),
        ids=(
            "items-node-from-property",
            "items-node-from-items",
            "items-node-from-anyOf-arm",
            "property-node-from-property",
            "property-node-from-items",
            "nullable-node-from-property",
            "nullable-node-from-items",
            "whole-List",
            "whole-W",
            "whole-N",
        ),
    )
    def test_on_a_node_inside_a_definition_it_raises(self, entry, use):
        """A `$ref` to such a node built it at 0.7 (`items@0.7`, `inner@0.7`,
        `DynamicModel@0.7`), dropping the keys, while a `$ref` to the whole
        entry raised."""
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(self._node_schema(entry, use))
        at = self._NODES[entry][2]
        message = str(exc.value)
        assert (
            f"'x-aws-stickler-match-threshold' beside the '$ref' at '{at}' is not read"
            in message
        )
        assert "the class is built from definition '$defs.Row'" in message

    @pytest.mark.parametrize("entry", ("List", "W"))
    @pytest.mark.parametrize(
        "where",
        (
            lambda schema, ref: schema["properties"].update(
                a={"type": "string", "not": {"$ref": ref}}
            ),
            lambda schema, ref: schema["$defs"].update(
                U={"type": "object", "properties": {"u": {"$ref": ref}}}
            ),
        ),
        ids=("under-not", "from-an-unused-definition"),
    )
    def test_where_nothing_is_built_from_the_node_it_imports(self, entry, where):
        """A `$ref` that builds no class drops no keys, as on dev."""
        schema = self._node_schema(entry, lambda ref: {"type": "string"})
        where(schema, self._NODES[entry][1])
        StructuredModel.from_json_schema(schema)

    def test_the_advice_works_through_an_alias(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "a": {"type": "array", "items": {"$ref": "#/$defs/Alias"}}
                },
                "$defs": {
                    "Alias": {"$ref": "#/$defs/Row"},
                    "Row": dict(
                        TestDefinitionsAreChecked.INNER,
                        **{"x-aws-stickler-match-threshold": 0.93},
                    ),
                },
            }
        )
        assert _unwrap(model.model_fields["a"].annotation).match_threshold == 0.93
