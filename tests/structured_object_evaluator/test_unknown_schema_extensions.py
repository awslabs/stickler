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

    def test_a_field_key_used_from_items_names_the_array_property(self):
        """`items` does not read field keys, so beside the `$ref` is wrong there."""
        with pytest.raises(ValueError) as exc:
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "rows": {"type": "array", "items": {"$ref": "#/$defs/Inner"}}
                    },
                    "$defs": {
                        "Inner": dict(self.INNER, **{"x-aws-stickler-weight": 3.0})
                    },
                }
            )
        assert "on the array property when the definition is its 'items'" in str(
            exc.value
        )

    def test_the_array_property_reads_it(self):
        """Where the error sends the author for `items`."""
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Inner"},
                        "x-aws-stickler-weight": 3.0,
                    }
                },
                "$defs": {"Inner": self.INNER},
            }
        )
        assert model._get_comparison_info("rows").weight == 3.0

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


_KEYED = dict(
    TestDefinitionsAreChecked.INNER,
    **{
        "x-aws-stickler-model-name": "Renamed",
        "x-aws-stickler-match-threshold": 0.93,
    },
)
_PLAIN = {"type": "object", "properties": {"z": {"type": "string"}}}


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

    def test_a_single_element_list_still_imports(self):
        model = StructuredModel.from_json_schema(
            {"type": ["object"], "properties": {"c": {"type": "string"}}}
        )
        assert "c" in model.model_fields


class TestOnlyReachedDefinitionsAreChecked:
    """A `$defs` entry is checked once a `$ref` reaches it, and not before.

    A dereferenced schema inlines every `$ref` and keeps `$defs`, so the
    originals carry keys that are read on their inlined copies. Checking every
    entry raised on such a schema, which imported and scored correctly before.
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
    def test_an_unreferenced_definition_is_not_checked(self, body):
        StructuredModel.from_json_schema(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "$defs": {"Unused": body},
            }
        )

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
