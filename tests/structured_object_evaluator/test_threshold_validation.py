"""A threshold must be an int or float from 0.0 to 1.0, on every path.

The rule covers the threshold a field ends up with, its own or one it adopts
from the comparator, and ``match_threshold``. Anything else raises
``ValueError`` when the class is defined or the config or schema is loaded,
never during evaluation, with one message built by ``threshold_rule``. It
quotes the value as the author wrote it, names the model and the field, says
why the value fails and gives the likely fix.

Before, the Python path checked neither ``ComparableField(threshold=85)`` nor
``LevenshteinComparator(threshold=85)``, so an identical pair scored 0.0 FD;
the importers ignored an out-of-range comparator threshold with a warning; and
a threshold of ``True`` loaded as 1 on the importers.
"""

import __future__

import inspect
import json
import math
import warnings
from decimal import Decimal
from fractions import Fraction
from typing import Annotated, List, Optional

import pytest

from stickler.comparators.base import BaseComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models import comparator_registry
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.model_factory import ModelFactory
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)

RULE = "A threshold is the minimum similarity score for a match, from 0.0 to 1.0."
HEAD = "Invalid threshold for field 'vendor' of model 'Invoice': "


def _python(annotation, field, name="Invoice", attr="vendor"):
    return type(
        name,
        (StructuredModel,),
        {"__annotations__": {attr: Optional[annotation]}, attr: field},
    )


def _config(field_config, name="Invoice", attr="vendor", **root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StructuredModel.model_from_json(
            {"model_name": name, "fields": {attr: field_config}, **root}
        )


def _schema(prop, title="Invoice", attr="vendor", **root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StructuredModel.from_json_schema(
            {"type": "object", "title": title, "properties": {attr: prop}, **root}
        )


def _message(build, value):
    with pytest.raises(ValueError) as caught:
        build(value)
    return str(caught.value)


class _Inner(StructuredModel):
    a: str = ComparableField(comparator=ExactComparator(), threshold=1.0)


_LEAF = {"type": "str", "comparator": "ExactComparator", "threshold": 1.0}
_SLEAF = {
    "type": "string",
    "x-aws-stickler-comparator": "ExactComparator",
    "x-aws-stickler-threshold": 1.0,
}

#: How each bad value reads in Python and in JSON, and what is wrong with it.
BAD = {
    "percent": (85, "85", "85", "85 is above 1.0, so no pair would match."),
    "above one": (1.5, "1.5", "1.5", "1.5 is above 1.0, so no pair would match."),
    "negative": (-1, "-1", "-1", "-1 is below 0.0, so every pair would match."),
    "string": ("0.9", "'0.9'", '"0.9"', "is a string, not a number."),
    "bool": (True, "True", "true", "is a boolean, not a number."),
    "nan": (
        float("nan"),
        "float('nan')",
        "NaN",
        "NaN fails every comparison, so no pair would match.",
    ),
    "inf": (
        float("inf"),
        "float('inf')",
        "Infinity",
        "is above 1.0, so no pair would match.",
    ),
}

# path -> (build(value), JSON spelling?, how the message opens for a value).
# One row per place a threshold becomes a field's gate.
PATHS = {
    "python field": (
        lambda v: _python(str, ComparableField(threshold=v)),
        False,
        lambda v: f"ComparableField(threshold={v}) ",
    ),
    "python Annotated field": (
        lambda v: type(
            "Invoice",
            (StructuredModel,),
            {"__annotations__": {"vendor": Annotated[str, ComparableField(threshold=v)]}},
        ),
        False,
        lambda v: f"ComparableField(threshold={v}) ",
    ),
    "python comparator": (
        lambda v: _python(
            str, ComparableField(comparator=LevenshteinComparator(threshold=v))
        ),
        False,
        lambda v: f"the field sets no threshold, so it uses LevenshteinComparator(threshold={v}). ",
    ),
    "python Annotated comparator": (
        lambda v: type(
            "Invoice",
            (StructuredModel,),
            {
                "__annotations__": {
                    "vendor": Annotated[
                        str, ComparableField(comparator=LevenshteinComparator(threshold=v))
                    ]
                }
            },
        ),
        False,
        lambda v: f"the field sets no threshold, so it uses LevenshteinComparator(threshold={v}). ",
    ),
    "python nested-model comparator": (
        lambda v: _python(
            _Inner, ComparableField(comparator=ExactComparator(threshold=v))
        ),
        False,
        lambda v: f"the field sets no threshold, so it uses ExactComparator(threshold={v}). ",
    ),
    "config field": (
        lambda v: _config(
            {"type": "str", "comparator": "LevenshteinComparator", "threshold": v}
        ),
        True,
        lambda v: f'"threshold": {v} ',
    ),
    "config comparator_config": (
        lambda v: _config(
            {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "comparator_config": {"threshold": v},
            }
        ),
        True,
        lambda v: f'the field sets no "threshold", so it uses "comparator_config": {{"threshold": {v}}}. ',
    ),
    "config nested-model comparator_config": (
        lambda v: _config(
            {
                "type": "structured_model",
                "comparator_config": {"threshold": v},
                "fields": {"a": _LEAF},
            }
        ),
        True,
        lambda v: f'the field sets no "threshold", so it uses "comparator_config": {{"threshold": {v}}}. ',
    ),
    "schema field": (
        lambda v: _schema({"type": "string", "x-aws-stickler-threshold": v}),
        True,
        lambda v: f'"x-aws-stickler-threshold": {v} ',
    ),
    "schema comparator-config": (
        lambda v: _schema(
            {
                "type": "string",
                "x-aws-stickler-comparator": "LevenshteinComparator",
                "x-aws-stickler-comparator-config": {"threshold": v},
            }
        ),
        True,
        lambda v: (
            'the field sets no "x-aws-stickler-threshold", so it uses '
            f'"x-aws-stickler-comparator-config": {{"threshold": {v}}}. '
        ),
    ),
    "schema nested-object comparator-config": (
        lambda v: _schema(
            {
                "type": "object",
                "properties": {"a": _SLEAF},
                "x-aws-stickler-comparator-config": {"threshold": v},
            }
        ),
        True,
        lambda v: (
            'the field sets no "x-aws-stickler-threshold", so it uses '
            f'"x-aws-stickler-comparator-config": {{"threshold": {v}}}. '
        ),
    ),
}


@pytest.mark.parametrize("kind", list(BAD))
@pytest.mark.parametrize("path", list(PATHS))
def test_every_path_raises_for_every_bad_kind(path, kind):
    """Model, field, source and value as written, the range, and what is wrong."""
    build, as_json, opens = PATHS[path]
    value, in_python, in_json, wrong = BAD[kind]
    message = _message(build, value)
    assert message.startswith(HEAD + opens(in_json if as_json else in_python))
    assert RULE in message
    assert wrong in message


@pytest.mark.parametrize("path", list(PATHS))
def test_a_valid_threshold_is_adopted(path):
    build, _, _ = PATHS[path]
    assert build(0.9)._get_comparison_info("vendor").threshold == 0.9


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("value", [0, 0.0, 1, 1.0, 0.5])
@pytest.mark.parametrize("path", list(PATHS))
def test_the_bounds_are_inclusive(path, value):
    build, _, _ = PATHS[path]
    assert math.isclose(build(value)._get_comparison_info("vendor").threshold, value)


class TestTheDocumentedMessages:
    """The examples the docs show, verbatim."""

    def test_python_comparator(self):
        line = inspect.currentframe().f_lineno + 2
        with pytest.raises(ValueError) as caught:
            _python(str, ComparableField(comparator=LevenshteinComparator(threshold=85)))
        assert str(caught.value) == (
            "Invalid threshold for field 'vendor' of model 'Invoice': the field "
            "sets no threshold, so it uses LevenshteinComparator(threshold=85). "
            f"{RULE} 85 is above 1.0, so no pair would match. If 85 is a "
            "percentage, write LevenshteinComparator(threshold=0.85), or set "
            "ComparableField(threshold=0.85), which takes precedence. "
            f"(Set at test_threshold_validation.py:{line}.)"
        )

    def test_python_field(self):
        line = inspect.currentframe().f_lineno + 2
        with pytest.raises(ValueError) as caught:
            _python(str, ComparableField(threshold=85))
        assert str(caught.value) == (
            "Invalid threshold for field 'vendor' of model 'Invoice': "
            f"ComparableField(threshold=85) is out of range. {RULE} 85 is above "
            "1.0, so no pair would match. If 85 is a percentage, use 0.85. "
            f"(Set at test_threshold_validation.py:{line}.)"
        )

    def test_config_comparator_config(self):
        message = _message(PATHS["config comparator_config"][0], 85)
        assert message == (
            HEAD + 'the field sets no "threshold", so it uses "comparator_config": '
            f'{{"threshold": 85}}. {RULE} 85 is above 1.0, so no pair would '
            'match. If 85 is a percentage, write 0.85 there, or set "threshold": '
            "0.85 on the field, which takes precedence."
        )

    def test_schema_comparator_config(self):
        message = _message(PATHS["schema comparator-config"][0], 85)
        assert message == (
            HEAD + 'the field sets no "x-aws-stickler-threshold", so it uses '
            '"x-aws-stickler-comparator-config": {"threshold": 85}. '
            f"{RULE} 85 is above 1.0, so no pair would match. If 85 is a "
            'percentage, write 0.85 there, or set "x-aws-stickler-threshold": '
            "0.85 on the property, which takes precedence."
        )

    def test_the_traceback_says_value_error(self):
        with pytest.raises(ValueError) as caught:
            _python(str, ComparableField(threshold=85))
        assert type(caught.value) is ValueError


class TestTheLikelyFix:
    """Each bad kind gets the fix it most likely needs, in the path's spelling."""

    @pytest.mark.parametrize(
        "path, fix",
        [
            ("python field", "If 85 is a percentage, use 0.85."),
            ("config field", "If 85 is a percentage, use 0.85."),
            ("schema field", "If 85 is a percentage, use 0.85."),
            (
                "python comparator",
                "If 85 is a percentage, write LevenshteinComparator(threshold=0.85), "
                "or set ComparableField(threshold=0.85), which takes precedence.",
            ),
        ],
    )
    def test_a_percentage(self, path, fix):
        assert _message(PATHS[path][0], 85).endswith(fix) or fix + " (Set at" in (
            _message(PATHS[path][0], 85)
        )

    @pytest.mark.parametrize(
        "path, weight",
        [
            ("python field", "ComparableField(weight=1.5)"),
            ("config field", '"weight": 1.5'),
            ("schema field", '"x-aws-stickler-weight": 1.5'),
        ],
    )
    def test_just_above_one_is_an_exact_match_or_a_weight(self, path, weight):
        assert (
            "1.5 is above 1.0, so no pair would match. Use 1.0 to require an "
            f"exact match; if you meant the field's weight, set {weight}."
        ) in _message(PATHS[path][0], 1.5)

    def test_a_hair_above_one_is_not_a_percentage_or_a_weight(self):
        message = _message(PATHS["config field"][0], 1.0000001)
        assert message.endswith("Use 1.0 to require an exact match.")

    @pytest.mark.parametrize(
        "value, fix",
        [
            ("0.9", "Write 0.9 without quotes."),
            ("85", "If 85 is a percentage, write 0.85 without quotes."),
            ("85%", "For 85%, use 0.85."),
            ("1.5", "Write a number from 0.0 to 1.0 without quotes."),
        ],
    )
    def test_a_numeric_string(self, value, fix):
        message = _message(PATHS["config field"][0], value)
        assert message == (
            HEAD + f'"threshold": {json.dumps(value)} is a string, not a number. '
            f"{RULE} {fix}"
        )

    @pytest.mark.parametrize("kind", ["negative", "bool", "nan", "inf"])
    def test_what_is_wrong_is_said_plainly(self, kind):
        """With no guess at a number: none is likely."""
        value, _, _, wrong = BAD[kind]
        message = _message(PATHS["schema field"][0], value)
        assert wrong in message
        assert message.endswith(RULE) or message.endswith(wrong)

    @pytest.mark.parametrize("path", ["config field", "schema field"])
    def test_null_on_the_importers(self, path):
        message = _message(PATHS[path][0], None)
        assert "null is not a number." in message
        assert message.endswith(
            "Remove the key to use the default, or write a number such as 0.8."
        )

    def test_an_unset_comparator_config_threshold_is_unset(self):
        """``null`` there means "not given", as ``ComparableField(threshold=None)``."""
        model = PATHS["config comparator_config"][0](None)
        assert model._get_comparison_info("vendor").threshold == 0.5


class TestOtherNumberTypesAskForAFloat:
    """``int`` or ``float`` only: ``numpy.float32`` did not serialize on export."""

    @pytest.mark.parametrize("path", ["python field", "config field", "schema field"])
    @pytest.mark.parametrize("value", [Decimal("0.9"), Fraction(9, 10)])
    def test_in_range(self, path, value):
        message = _message(PATHS[path][0], value)
        assert f"is a {type(value).__name__}, not an int or float." in message
        assert "Pass a float: 0.9." in message

    def test_out_of_range(self):
        message = _message(PATHS["python field"][0], Decimal("85"))
        assert "85 is above 1.0, so no pair would match. If 85 is a percentage, pass the float 0.85." in message

    @pytest.mark.parametrize("path", ["python field", "config field", "schema field"])
    def test_numpy(self, path):
        np = pytest.importorskip("numpy")
        message = _message(PATHS[path][0], np.float32(0.9))
        assert "is a numpy.float32, not an int or float." in message
        assert "Pass a float: 0.9." in message
        assert PATHS[path][0](np.float64(0.85))._get_comparison_info(
            "vendor"
        ).threshold == 0.85


class TestTheFieldThresholdWins:
    """The comparator's threshold is not read when the field sets one.

    ``binary_compare()`` is its only other reader, and the evaluator does not
    call it, so an out-of-range comparator threshold there is not checked.
    """

    @staticmethod
    def _paths():
        return {
            "python": _python(
                str,
                ComparableField(
                    comparator=LevenshteinComparator(threshold=85), threshold=0.9
                ),
            ),
            "config": _config(
                {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "comparator_config": {"threshold": 85},
                    "threshold": 0.9,
                }
            ),
            "schema": _schema(
                {
                    "type": "string",
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                    "x-aws-stickler-comparator-config": {"threshold": 85},
                    "x-aws-stickler-threshold": 0.9,
                }
            ),
            "python nested": _python(
                _Inner,
                ComparableField(comparator=ExactComparator(threshold=85), threshold=0.9),
            ),
            "config nested": _config(
                {
                    "type": "structured_model",
                    "comparator_config": {"threshold": 85},
                    "threshold": 0.9,
                    "fields": {"a": _LEAF},
                }
            ),
            "schema nested": _schema(
                {
                    "type": "object",
                    "properties": {"a": _SLEAF},
                    "x-aws-stickler-comparator-config": {"threshold": 85},
                    "x-aws-stickler-threshold": 0.9,
                }
            ),
        }

    def test_no_path_raises_and_the_field_gates(self):
        paths = self._paths()
        assert {n: m._get_comparison_info("vendor").threshold for n, m in paths.items()} == {
            n: 0.9 for n in paths
        }

    def test_an_identical_pair_is_a_match(self):
        for name in ("python", "config", "schema"):
            model = self._paths()[name]
            result = model(vendor="Acme").compare_with(
                model(vendor="Acme"), include_confusion_matrix=True
            )
            assert result["confusion_matrix"]["overall"]["tp"] == 1, name


class TestANestedModelReadsOnlyTheComparatorsThreshold:
    """Its comparator is never called, so the fix moves the threshold to the field."""

    def test_python(self):
        message = _message(PATHS["python nested-model comparator"][0], 85)
        assert (
            "ExactComparator has no effect on a nested model except for its "
            "threshold. If 85 is a percentage, remove it and set "
            "ComparableField(threshold=0.85)."
        ) in message

    def test_config(self):
        message = _message(PATHS["config nested-model comparator_config"][0], 85)
        assert message.endswith(
            '"comparator_config" has no effect on a nested model except for its '
            'threshold. If 85 is a percentage, remove it and set "threshold": '
            "0.85 on the field."
        )

    def test_schema(self):
        message = _message(PATHS["schema nested-object comparator-config"][0], 85)
        assert message.endswith(
            '"x-aws-stickler-comparator-config" has no effect on an object with '
            "properties except for its threshold. If 85 is a percentage, remove "
            'it and set "x-aws-stickler-threshold": 0.85 on the property.'
        )


class TestAListOfModels:
    """Pairing uses the element class's ``match_threshold``."""

    def test_a_comparator_threshold_is_not_checked_on_any_path(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _python(
                List[_Inner],
                ComparableField(comparator=LevenshteinComparator(threshold=85)),
            )
        _config(
            {
                "type": "list_structured_model",
                "comparator_config": {"threshold": 85},
                "fields": {"a": _LEAF},
            }
        )
        _schema(
            {
                "type": "array",
                "items": {"type": "object", "properties": {"a": _SLEAF}},
                "x-aws-stickler-comparator-config": {"threshold": 85},
            }
        )

    def test_nor_under_postponed_annotations(self):
        """``__annotations__`` holds ``"List[_Inner]"``, a string, there."""
        namespace = {"StructuredModel": StructuredModel, "List": List, "_Inner": _Inner}
        source = (
            "class Order(StructuredModel):\n"
            "    items: List[_Inner] = ComparableField(\n"
            "        comparator=LevenshteinComparator(threshold=85)\n"
            "    )\n"
        )
        namespace.update(
            ComparableField=ComparableField, LevenshteinComparator=LevenshteinComparator
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(
                compile(
                    source,
                    "order_models.py",
                    "exec",
                    flags=__future__.annotations.compiler_flag,
                    dont_inherit=True,
                ),
                namespace,
            )
        assert namespace["Order"]._get_comparison_info("items").threshold == 85

    @pytest.mark.parametrize(
        "build, note, fix",
        [
            (
                lambda: _python(List[_Inner], ComparableField(threshold=85), attr="items"),
                "A List[StructuredModel] field takes no threshold: its items are "
                "paired on the element class's match_threshold.",
                "If 85 is a percentage, set match_threshold = 0.85 on _Inner instead.",
            ),
            (
                lambda: _config(
                    {"type": "list_structured_model", "threshold": 85, "fields": {"a": _LEAF}},
                    attr="items",
                ),
                'A list_structured_model field reads no "threshold": its items are '
                'paired on "match_threshold".',
                'If 85 is a percentage, set "match_threshold": 0.85 on the field instead.',
            ),
            (
                lambda: _schema(
                    {
                        "type": "array",
                        "x-aws-stickler-threshold": 85,
                        "items": {"type": "object", "properties": {"a": _SLEAF}},
                    },
                    attr="items",
                ),
                'An array of objects reads no "x-aws-stickler-threshold": its items '
                'are paired on "x-aws-stickler-match-threshold".',
                'If 85 is a percentage, set "x-aws-stickler-match-threshold": 0.85 '
                'in its "items" instead.',
            ),
        ],
    )
    def test_a_field_threshold_there_points_at_match_threshold(self, build, note, fix):
        with pytest.raises(ValueError) as caught:
            build()
        message = str(caught.value)
        assert message.startswith("Invalid threshold for field 'items' of model 'Invoice': ")
        assert f"85 is above 1.0, so no pair would match. {note} {fix}" in message


class TestTheMessageNamesTheModelAsTheAuthorDoes:
    def test_a_schema_uses_its_model_name_then_its_title(self):
        prop = {"type": "string", "x-aws-stickler-comparator-config": {"threshold": 85}}
        with pytest.raises(ValueError, match="of model 'Receipt'"):
            _schema(prop, title="Ignored", **{"x-aws-stickler-model-name": "Receipt"})
        with pytest.raises(ValueError, match="of model 'Invoice'"):
            _schema(prop, title="Invoice")

    def test_an_unnamed_model_says_how_to_name_it(self):
        prop = {"type": "string", "x-aws-stickler-comparator-config": {"threshold": 85}}
        with pytest.raises(ValueError) as caught:
            StructuredModel.from_json_schema({"type": "object", "properties": {"vendor": prop}})
        assert str(caught.value).startswith(
            "Invalid threshold for field 'vendor' of an unnamed model (set title "
            "or x-aws-stickler-model-name to name it): "
        )
        with pytest.raises(ValueError) as caught:
            StructuredModel.model_from_json(
                {
                    "fields": {
                        "vendor": {
                            "type": "str",
                            "comparator": "LevenshteinComparator",
                            "comparator_config": {"threshold": 85},
                        }
                    }
                }
            )
        assert str(caught.value).startswith(
            "Invalid threshold for field 'vendor' of an unnamed model (set "
            "model_name to name it): "
        )

    @pytest.mark.parametrize(
        "build, field",
        [
            (
                lambda: _config(
                    {
                        "type": "structured_model",
                        "fields": {
                            "city": {
                                "type": "str",
                                "comparator": "ExactComparator",
                                "comparator_config": {"threshold": 85},
                            }
                        },
                    },
                    attr="billing",
                ),
                "billing.city",
            ),
            (
                lambda: _schema(
                    {"type": "object", "properties": {"city": {**_SLEAF, "x-aws-stickler-threshold": 85}}},
                    attr="billing",
                ),
                "billing.city",
            ),
            (
                lambda: _config(
                    {"type": "list_structured_model", "fields": {"sku": {**_LEAF, "threshold": 85}}},
                    attr="items",
                ),
                "items[].sku",
            ),
            (
                lambda: _schema(
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"sku": {**_SLEAF, "x-aws-stickler-threshold": 85}},
                        },
                    },
                    attr="items",
                ),
                "items[].sku",
            ),
        ],
    )
    def test_a_nested_field_is_named_alike_on_both_importers(self, build, field):
        with pytest.raises(ValueError) as caught:
            build()
        assert str(caught.value).startswith(
            f"Invalid threshold for field '{field}' of model 'Invoice': "
        )

    def test_a_ref_leaf_names_its_definition(self):
        with pytest.raises(ValueError) as caught:
            StructuredModel.from_json_schema(
                {
                    "title": "Invoice",
                    "type": "object",
                    "$defs": {
                        "Address": {
                            "type": "object",
                            "properties": {"city": {**_SLEAF, "x-aws-stickler-threshold": 85}},
                        }
                    },
                    "properties": {
                        "shipping": {"$ref": "#/$defs/Address"},
                        "billing": {"$ref": "#/$defs/Address"},
                    },
                }
            )
        assert str(caught.value).startswith(
            "Invalid threshold for field 'shipping.city' (defined at "
            "#/$defs/Address/properties/city) of model 'Invoice': "
        )

    def test_a_class_built_by_create_model_from_fields_is_named(self):
        with pytest.raises(ValueError) as caught:
            ModelFactory.create_model_from_fields(
                "Receipt", {"vendor": (str, ComparableField(threshold=85))}
            )
        assert str(caught.value).startswith(
            "Invalid threshold for field 'vendor' of model 'Receipt': "
        )


class TestTheConfigPathAddsNoPrefix:
    """Whatever the key or the depth, the message starts the way the docs show it."""

    @pytest.mark.parametrize(
        "field_config",
        [
            {"type": "str", "comparator": "ExactComparator", "threshold": 85},
            {
                "type": "str",
                "comparator": "ExactComparator",
                "comparator_config": {"threshold": 85},
            },
            {
                "type": "structured_model",
                "fields": {
                    "inner": {
                        "type": "structured_model",
                        "fields": {"sku": {**_LEAF, "threshold": 85}},
                    }
                },
            },
        ],
    )
    def test_no_wrapper_and_no_chain(self, field_config):
        with pytest.raises(ValueError) as caught:
            _config(field_config)
        assert str(caught.value).startswith("Invalid threshold for field 'vendor")
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None


class _Strict85(BaseComparator):
    """Sets its own threshold, as a domain comparator might."""

    def __init__(self):
        super().__init__(threshold=85)

    def _compare(self, a, b):
        return 1.0 if a == b else 0.0


@pytest.fixture
def strict85():
    name = "_Strict85ForThresholdTests"
    comparator_registry._global_registry._registry[name] = _Strict85
    yield name
    comparator_registry._global_registry._registry.pop(name, None)


class TestAThresholdTheComparatorSetsItself:
    """Named as such, not as a call the author never wrote, and with no guess."""

    def test_every_path(self, strict85):
        python = _message(lambda _: _python(str, ComparableField(comparator=_Strict85())), None)
        assert "the field sets no threshold, so it uses the threshold its comparator _Strict85 sets, 85." in python
        assert "Set ComparableField(threshold=...) to a number from 0.0 to 1.0, which takes precedence." in python
        config = _message(lambda _: _config({"type": "str", "comparator": strict85}), None)
        assert (
            f'the field sets no "threshold", so it uses the threshold its comparator "{strict85}" sets, 85.'
        ) in config
        assert config.endswith(
            'Set "threshold" on the field to a number from 0.0 to 1.0, which takes precedence.'
        )
        schema = _message(
            lambda _: _schema({"type": "string", "x-aws-stickler-comparator": strict85}), None
        )
        assert f'its comparator "{strict85}" sets, 85.' in schema

    def test_on_a_nested_model(self, strict85):
        message = _message(
            lambda _: _config(
                {"type": "structured_model", "comparator": strict85, "fields": {"a": _LEAF}}
            ),
            None,
        )
        assert message.endswith(
            '"comparator" has no effect on a nested model except for its threshold. '
            'Remove it and set "threshold" on the field to a number from 0.0 to 1.0.'
        )


class _Units(BaseComparator):
    """Uses ``threshold`` as a distance in units, not as a similarity."""

    def __init__(self, threshold: Optional[float] = None):
        super().__init__(threshold=threshold)

    def _compare(self, a, b):
        return 1.0 if abs(float(a) - float(b)) <= self.threshold else 0.0


def test_a_comparator_constructor_is_not_checked():
    """Only a value that becomes the field's threshold is."""
    comparator = _Units(threshold=5)
    model = _python(float, ComparableField(comparator=comparator, threshold=1.0))
    assert model._get_comparison_info("vendor").threshold == 1.0
    assert model(vendor=10).compare_with(model(vendor=13))["overall_score"] == 1.0


def test_a_string_in_annotated_raises_at_definition_not_evaluation():
    with pytest.raises(ValueError, match="Write 0.9 without quotes."):
        type(
            "Invoice",
            (StructuredModel,),
            {"__annotations__": {"vendor": Annotated[str, ComparableField(threshold="0.9")]}},
        )


class TestMatchThreshold:
    """Checked by the same rule, with the same message, on every path."""

    MATCH_RULE = (
        "A match threshold is the minimum overall score for two objects to match, "
        "from 0.0 to 1.0."
    )

    @staticmethod
    def _builds():
        leaf = {"vendor": {"type": "str", "comparator": "ExactComparator"}}
        return {
            "python": (
                lambda v: type(
                    "Invoice",
                    (StructuredModel,),
                    {
                        "__annotations__": {"vendor": str},
                        "match_threshold": v,
                        "vendor": ComparableField(),
                    },
                ),
                "match_threshold = {} ",
                False,
                None,
            ),
            "config": (
                lambda v: _config(
                    leaf["vendor"], match_threshold=v
                ),
                '"match_threshold": {} ',
                True,
                None,
            ),
            "config nested": (
                lambda v: _config(
                    {"type": "structured_model", "match_threshold": v, "fields": {"a": _LEAF}},
                    attr="billing",
                ),
                '"match_threshold": {} ',
                True,
                "billing",
            ),
            "config list": (
                lambda v: _config(
                    {"type": "list_structured_model", "match_threshold": v, "fields": {"a": _LEAF}},
                    attr="items",
                ),
                '"match_threshold": {} ',
                True,
                "items",
            ),
            "schema": (
                lambda v: _schema(
                    {"type": "string"}, **{"x-aws-stickler-match-threshold": v}
                ),
                '"x-aws-stickler-match-threshold": {} ',
                True,
                None,
            ),
            "schema nested": (
                lambda v: _schema(
                    {
                        "type": "object",
                        "x-aws-stickler-match-threshold": v,
                        "properties": {"a": _SLEAF},
                    },
                    attr="billing",
                ),
                '"x-aws-stickler-match-threshold": {} ',
                True,
                "billing",
            ),
        }

    @pytest.mark.parametrize("kind", ["percent", "negative", "string", "bool", "nan", "inf"])
    @pytest.mark.parametrize("path", list(_builds.__func__()))
    def test_every_path_raises(self, path, kind):
        build, opens, as_json, field = self._builds()[path]
        value, in_python, in_json, wrong = BAD[kind]
        message = _message(build, value)
        where = f"field '{field}' of model 'Invoice'" if field else "model 'Invoice'"
        assert message.startswith(
            f"Invalid match threshold for {where}: "
            + opens.format(in_json if as_json else in_python)
        )
        assert self.MATCH_RULE in message
        assert wrong.replace("pair", "pair of objects") in message

    def test_the_percentage_fix(self):
        message = _message(self._builds()["config nested"][0], 85)
        assert message == (
            "Invalid match threshold for field 'billing' of model 'Invoice': "
            f'"match_threshold": 85 is out of range. {self.MATCH_RULE} 85 is above '
            "1.0, so no pair of objects would match. If 85 is a percentage, use 0.85."
        )

    @pytest.mark.parametrize("path", list(_builds.__func__()))
    def test_a_valid_value_loads(self, path):
        self._builds()[path][0](0.8)

    def test_none_on_a_python_class_is_unset(self):
        self._builds()["python"][0](None)


class TestExportAndReloadAreUnchanged:
    """A valid model exports and reloads as it did, on both formats."""

    @staticmethod
    def _models():
        return {
            "adopted": _python(
                str, ComparableField(comparator=LevenshteinComparator(threshold=0.9))
            ),
            "field wins": TestTheFieldThresholdWins._paths()["python"],
            "config adopted": PATHS["config comparator_config"][0](0.9),
            "schema adopted": PATHS["schema comparator-config"][0](0.9),
            "nested adopted": PATHS["config nested-model comparator_config"][0](0.9),
        }

    @pytest.mark.parametrize("name", list(_models.__func__()))
    def test_both_exports_reload_with_the_same_threshold(self, name):
        model = self._models()[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = model.to_stickler_config()
            schema = model.to_json_schema()
            from_config = StructuredModel.model_from_json(config)
            from_schema = StructuredModel.from_json_schema(schema)
        threshold = model._get_comparison_info("vendor").threshold
        assert from_config._get_comparison_info("vendor").threshold == threshold
        assert from_schema._get_comparison_info("vendor").threshold == threshold
        assert from_config.to_stickler_config() == config
        assert from_schema.to_json_schema() == schema

    def test_a_list_of_models_with_an_unread_comparator_threshold_reloads(self):
        """Pairing never reads it, so the export leaves an out-of-range one off."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _python(
                List[_Inner],
                ComparableField(comparator=LevenshteinComparator(threshold=85)),
                attr="items",
            )
            config = model.to_stickler_config()
            assert "threshold" not in config["fields"]["items"]
            json.dumps(config)
            from_config = StructuredModel.model_from_json(config)
            from_schema = StructuredModel.from_json_schema(model.to_json_schema())
        gt = {"items": [{"a": "x"}, {"a": "y"}]}
        pred = {"items": [{"a": "x"}, {"a": "z"}]}
        scores = {
            m(**gt).compare_with(m(**pred))["overall_score"]
            for m in (model, from_config, from_schema)
        }
        assert len(scores) == 1

    def test_a_valid_list_of_models_export_is_unchanged(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _python(
                List[_Inner],
                ComparableField(comparator=LevenshteinComparator(threshold=0.9)),
                attr="items",
            )
        assert model.to_stickler_config()["fields"]["items"]["threshold"] == 0.9
