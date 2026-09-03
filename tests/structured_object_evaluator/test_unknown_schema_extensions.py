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
