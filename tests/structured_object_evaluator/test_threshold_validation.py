"""A threshold must be a number from 0.0 to 1.0, on the class, config and schema.

The rule covers the threshold a field ends up with, its own or one it adopts
from the comparator, and ``match_threshold``. A ``Decimal``, ``Fraction`` or
numpy number in range is converted with ``float()``. Anything else raises
``ValueError`` when the class is defined or the config or schema is loaded,
also for a field whose annotation is still a forward reference then, with one
message built by ``threshold_rule``. It quotes the value as the author wrote
it, names the model and the field, says why the value fails and gives the
likely fix.

Before, the Python path checked neither ``ComparableField(threshold=85)`` nor
``LevenshteinComparator(threshold=85)``, so an identical pair scored 0.0 FD;
the importers ignored an out-of-range comparator threshold with a warning; and
a threshold of ``True`` loaded as 1 on the importers.
"""

import __future__

import enum
import inspect
import json
import math
import os
import sys
import warnings
from decimal import Decimal
from fractions import Fraction
from typing import Annotated, ClassVar, Dict, List, Optional, Tuple, Union

import pytest
from pydantic import BaseModel

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
#: How the Python path's messages name this file: relative to the working directory.
_HERE = os.path.relpath(__file__)
_HERE = __file__ if _HERE.startswith(os.pardir) else _HERE
ADOPTS = "ComparableField sets no threshold, so the field adopts its comparator's: "


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
    "percent": (85, "85", "85", "85 is above 1.0, so even identical values would not match."),
    "above one": (
        1.5,
        "1.5",
        "1.5",
        "1.5 is above 1.0, so even identical values would not match.",
    ),
    "negative": (
        -1,
        "-1",
        "-1",
        "-1 is below 0.0, so even completely different values would match.",
    ),
    "string": ("0.9", "'0.9'", '"0.9"', "is a string, not a number."),
    "bool": (True, "True", "true", "is a boolean, not a number."),
    "nan": (
        float("nan"),
        "float('nan')",
        "NaN",
        "NaN fails every comparison, so even identical values would not match.",
    ),
    "inf": (
        float("inf"),
        "float('inf')",
        "Infinity",
        "is above 1.0, so even identical values would not match.",
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
        lambda v: f"{ADOPTS}LevenshteinComparator(threshold={v}). ",
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
        lambda v: f"{ADOPTS}LevenshteinComparator(threshold={v}). ",
    ),
    "python nested-model comparator": (
        lambda v: _python(
            _Inner, ComparableField(comparator=ExactComparator(threshold=v))
        ),
        False,
        lambda v: f"{ADOPTS}ExactComparator(threshold={v}). ",
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
        lambda v: (
            'the field sets no "threshold", so it adopts its comparator\'s: '
            f'"comparator_config": {{"threshold": {v}}}. '
        ),
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
        lambda v: (
            'the field sets no "threshold", so it adopts its comparator\'s: '
            f'"comparator_config": {{"threshold": {v}}}. '
        ),
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
            'the field sets no "x-aws-stickler-threshold", so it adopts its '
            f'comparator\'s: "x-aws-stickler-comparator-config": {{"threshold": {v}}}. '
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
            'the field sets no "x-aws-stickler-threshold", so it adopts its '
            f'comparator\'s: "x-aws-stickler-comparator-config": {{"threshold": {v}}}. '
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
            "Invalid threshold for field 'vendor' of model 'Invoice': "
            f"{ADOPTS}LevenshteinComparator(threshold=85). {RULE} 85 is above "
            "1.0, so even identical values would not match. If 85 is a "
            "percentage, write LevenshteinComparator(threshold=0.85), or "
            "ComparableField(threshold=0.85), which wins over the comparator's. "
            f"(Set at {_HERE}:{line}.)"
        )

    def test_python_field(self):
        line = inspect.currentframe().f_lineno + 2
        with pytest.raises(ValueError) as caught:
            _python(str, ComparableField(threshold=85))
        assert str(caught.value) == (
            "Invalid threshold for field 'vendor' of model 'Invoice': "
            f"ComparableField(threshold=85) is out of range. {RULE} 85 is above "
            "1.0, so even identical values would not match. If 85 is a "
            "percentage, write ComparableField(threshold=0.85). "
            f"(Set at {_HERE}:{line}.)"
        )

    def test_config_comparator_config(self):
        message = _message(PATHS["config comparator_config"][0], 85)
        assert message == (
            HEAD + 'the field sets no "threshold", so it adopts its comparator\'s: '
            f'"comparator_config": {{"threshold": 85}}. {RULE} 85 is above 1.0, '
            "so even identical values would not match. If 85 is a percentage, "
            'write "comparator_config": {"threshold": 0.85}, or "threshold": 0.85 '
            "on the field, which wins over the comparator's."
        )

    def test_schema_comparator_config(self):
        message = _message(PATHS["schema comparator-config"][0], 85)
        assert message == (
            HEAD + 'the field sets no "x-aws-stickler-threshold", so it adopts its '
            "comparator's: "
            '"x-aws-stickler-comparator-config": {"threshold": 85}. '
            f"{RULE} 85 is above 1.0, so even identical values would not match. "
            "If 85 is a percentage, write "
            '"x-aws-stickler-comparator-config": {"threshold": 0.85}, or '
            '"x-aws-stickler-threshold": 0.85 on the property, which wins over '
            "the comparator's."
        )

    def test_the_traceback_says_value_error(self):
        with pytest.raises(ValueError) as caught:
            _python(str, ComparableField(threshold=85))
        assert type(caught.value) is ValueError


class TestTheLikelyFix:
    """Each bad kind gets the fix it most likely needs, spelled out in full."""

    @pytest.mark.parametrize(
        "path, fix",
        [
            ("python field", "If 85 is a percentage, write ComparableField(threshold=0.85)."),
            ("config field", 'If 85 is a percentage, write "threshold": 0.85.'),
            (
                "schema field",
                'If 85 is a percentage, write "x-aws-stickler-threshold": 0.85.',
            ),
            (
                "python comparator",
                "If 85 is a percentage, write LevenshteinComparator(threshold=0.85), "
                "or ComparableField(threshold=0.85), which wins over the "
                "comparator's.",
            ),
        ],
    )
    def test_a_percentage(self, path, fix):
        assert _message(PATHS[path][0], 85).endswith(fix) or fix + " (Set at" in (
            _message(PATHS[path][0], 85)
        )

    @pytest.mark.parametrize(
        "path, key, weight",
        [
            ("python field", "ComparableField(threshold=1.0)", "ComparableField(weight=1.5)"),
            ("config field", '"threshold": 1.0', '"weight": 1.5'),
            ("schema field", '"x-aws-stickler-threshold": 1.0', '"x-aws-stickler-weight": 1.5'),
        ],
    )
    def test_just_above_one_is_a_perfect_score_or_a_weight(self, path, key, weight):
        assert (
            "1.5 is above 1.0, so even identical values would not match. "
            f"Write {key} to require a perfect similarity score; if you meant "
            f"the field's weight, write {weight}."
        ) in _message(PATHS[path][0], 1.5)

    def test_a_hair_above_one_is_not_a_percentage_or_a_weight(self):
        message = _message(PATHS["config field"][0], 1.0000001)
        assert message.endswith(
            'Write "threshold": 1.0 to require a perfect similarity score.'
        )

    @pytest.mark.parametrize(
        "value, fix",
        [
            ("0.9", 'Write "threshold": 0.9, without quotes.'),
            ("85", 'If 85 is a percentage, write "threshold": 0.85, without quotes.'),
            ("85%", 'For 85%, write "threshold": 0.85.'),
            ("1.5", "Write a number from 0.0 to 1.0, without quotes."),
        ],
    )
    def test_a_numeric_string(self, value, fix):
        message = _message(PATHS["config field"][0], value)
        assert message == (
            HEAD + f'"threshold": {json.dumps(value)} is a string, not a number. '
            f"{RULE} {fix}"
        )

    @pytest.mark.parametrize("kind", ["negative", "inf"])
    @pytest.mark.parametrize("path", ["config field", "schema field"])
    def test_with_no_likely_number_the_fix_gives_the_range(self, path, kind):
        value, _, _, wrong = BAD[kind]
        message = _message(PATHS[path][0], value)
        assert message.endswith(f"{wrong} Write a number from 0.0 to 1.0 instead.")

    @pytest.mark.parametrize(
        "path, unset",
        [
            ("python field", "leave threshold unset"),
            ("config field", 'remove "threshold"'),
            ("schema field", 'remove "x-aws-stickler-threshold"'),
        ],
    )
    def test_nan_says_where_it_usually_comes_from(self, path, unset):
        message = _message(PATHS[path][0], float("nan"))
        assert " is not a number. " in message
        assert (
            "NaN fails every comparison, so even identical values would not match. "
            "A NaN usually comes from an empty cell in a DataFrame or CSV: fill it "
            f"in, or {unset} to use the default."
        ) in message

    @pytest.mark.parametrize(
        "path, fix",
        [
            (
                "python field",
                "Write ComparableField(threshold=1.0) to require a perfect "
                "similarity score.",
            ),
            (
                "config field",
                'Write "threshold": 1.0 to require a perfect similarity score.',
            ),
            (
                "python comparator",
                "To require a perfect similarity score, write "
                "LevenshteinComparator(threshold=1.0), or "
                "ComparableField(threshold=1.0), which wins over the comparator's.",
            ),
            (
                "config comparator_config",
                'To require a perfect similarity score, write "comparator_config": '
                '{"threshold": 1.0}, or "threshold": 1.0 on the field, which wins '
                "over the comparator's.",
            ),
        ],
    )
    def test_true_is_a_perfect_score(self, path, fix):
        """``true`` loaded as 1 on the importers, so 1.0 keeps that verdict.

        No clip hint: ``clip_under_threshold`` is already on for most fields.
        """
        message = _message(PATHS[path][0], True)
        assert "is a boolean, not a number." in message
        assert message.endswith(fix) or fix + " (Set at" in message

    def test_false_points_at_the_default(self):
        message = _message(PATHS["schema field"][0], False)
        assert message.endswith(
            "is a boolean, not a number. "
            f'{RULE} Remove "x-aws-stickler-threshold" to use the default, or '
            'write "x-aws-stickler-threshold": 0.0 to match every pair.'
        )

    def test_every_message_ends_with_a_fix(self):
        for path, (build, _, _) in PATHS.items():
            for kind, (value, *_rest) in BAD.items():
                message = _message(build, value)
                assert not message.endswith(RULE), (path, kind)

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


class TestWhereTheMessageSaysItWasSet:
    """A path the author can find, or the notebook cell."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("<ipython-input-3-9f2c>", "line 4 of the notebook cell"),
            (
                os.path.join(os.sep, "var", "folders", "ipykernel_4242", "1234567.py"),
                "line 4 of the notebook cell",
            ),
            ("<string>", None),
            (os.path.join("models", "invoice.py"), os.path.join("models", "invoice.py") + ":4"),
            (os.path.abspath(os.path.join("models", "invoice.py")), os.path.join("models", "invoice.py") + ":4"),
        ],
    )
    def test_location(self, filename, expected):
        from stickler.structured_object_evaluator.models.threshold_rule import location

        assert location(filename, 4) == expected

    def test_a_file_outside_the_working_directory_is_given_in_full(self):
        from stickler.structured_object_evaluator.models.threshold_rule import location

        outside = os.path.join(os.path.dirname(os.path.abspath(os.curdir)), "x.py")
        assert location(outside, 4) == f"{outside}:4"

    def test_a_class_match_threshold_names_the_class_line(self):
        source = (
            "import os\n"
            "class Order(StructuredModel):\n"
            "    match_threshold = 85\n"
            "    vendor: str = ComparableField()\n"
        )
        with pytest.raises(ValueError) as caught:
            _exec_models(source)
        assert str(caught.value).endswith(
            "If 85 is a percentage, write match_threshold = 0.85. "
            "(Set at order_models.py:2.)"
        )


class _Fixed(BaseComparator):
    """Scores every pair ``score``."""

    def __init__(self, score: float, threshold: Optional[float] = None):
        super().__init__(threshold=threshold)
        self.score = score

    def _compare(self, a, b):
        return self.score


def _other_reals():
    """In-range thresholds that are real numbers but not ``int`` or ``float``."""
    values = [Decimal("0.9"), Fraction(9, 10)]
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a dependency
        return values
    return values + [
        np.float32(0.9),
        np.float64(0.9),
        np.float16(0.9),
        np.int64(1),
        np.array(0.9),
    ]


class TestOtherRealNumbersAreConvertedToFloat:
    """A ``Decimal``, ``Fraction`` or numpy number means the same threshold.

    The Python path loaded and scored it, so refusing it would break working
    code, and the importers refused it. It is converted with ``float()``, not
    through ``str``, and the field's metadata and ``explain()`` carry that
    float. A numpy float16 or float32 is then compared as that Python float,
    not in its own dtype as before, so a score within its rounding below the
    threshold is an FD where it was a TP.
    """

    @pytest.mark.parametrize("value", _other_reals(), ids=repr)
    @pytest.mark.parametrize("path", list(PATHS))
    def test_in_range_loads_as_a_float(self, path, value):
        threshold = PATHS[path][0](value)._get_comparison_info("vendor").threshold
        assert type(threshold) is float
        assert threshold == float(value)

    @pytest.mark.parametrize("value", _other_reals(), ids=repr)
    @pytest.mark.parametrize("path", ["python", "config", "config nested", "schema"])
    def test_a_match_threshold_loads_as_a_float(self, path, value):
        model = TestMatchThreshold._builds()[path][0](value)
        if path == "config nested":
            model = model.model_fields["billing"].annotation
            model = getattr(model, "__args__", (model,))[0]
        assert type(model.match_threshold) is float
        assert model.match_threshold == float(value)

    def test_an_identical_pair_matches_and_both_exports_reload(self):
        model = PATHS["python comparator"][0](Decimal("0.9"))
        result = model(vendor="Acme").compare_with(
            model(vendor="Acme"), include_confusion_matrix=True
        )
        assert result["confusion_matrix"]["overall"]["tp"] == 1
        config = model.to_stickler_config()
        json.dumps(config)
        assert config["fields"]["vendor"]["threshold"] == 0.9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reloaded = StructuredModel.model_from_json(config)
            from_schema = StructuredModel.from_json_schema(model.to_json_schema())
        assert reloaded._get_comparison_info("vendor").threshold == 0.9
        assert from_schema._get_comparison_info("vendor").threshold == 0.9

    def test_a_numpy_float32_match_threshold_is_its_float(self):
        """``match_threshold`` is 0.8999999761581421, so a 0.899999985 item matches."""
        np = pytest.importorskip("numpy")
        record = type(
            "Rec",
            (StructuredModel,),
            {
                "__annotations__": {"f0": str, "f1": str},
                "match_threshold": np.float32(0.9),
                "f0": ComparableField(comparator=ExactComparator(), weight=1.00000015),
                "f1": ComparableField(comparator=ExactComparator(), weight=8.99999985),
            },
        )
        order = type(
            "Order",
            (StructuredModel,),
            {"__annotations__": {"items": List[record]}, "items": ComparableField()},
        )
        result = order(items=[record(f0="a", f1="b")]).compare_with(
            order(items=[record(f0="X", f1="b")]), include_confusion_matrix=True
        )
        cell = result["confusion_matrix"]["fields"]["items"]["overall"]
        assert (cell["tp"], cell["fd"], cell["fp"]) == (1, 0, 0)

    @pytest.mark.parametrize(
        "dtype, below",
        [("float32", 0.89999997), ("float16", 0.8998)],
    )
    def test_a_numpy_float16_or_float32_compares_as_a_python_float(self, dtype, below):
        """Its float() value is the threshold, so a score just below it is an FD.

        In the numpy dtype, as dev compared it, ``below`` rounds to the
        threshold and was a TP.
        """
        np = pytest.importorskip("numpy")
        threshold = getattr(np, dtype)(0.9)
        assert below >= threshold  # in the dtype, as before
        assert below < float(threshold)

        def verdict(score):
            model = _python(
                str, ComparableField(comparator=_Fixed(score), threshold=threshold)
            )
            result = model(vendor="a").compare_with(
                model(vendor="b"), include_confusion_matrix=True
            )
            cell = result["confusion_matrix"]["overall"]
            return cell["tp"], cell["fd"]

        assert verdict(below) == (0, 1)
        assert verdict(float(threshold)) == (1, 0)

    def test_the_comparator_keeps_the_value_it_was_given(self):
        """Only the field's threshold is converted, as documented."""
        comparator = LevenshteinComparator(threshold=Decimal("0.9"))
        model = _python(str, ComparableField(comparator=comparator))
        assert model._get_comparison_info("vendor").threshold == 0.9
        assert comparator.threshold == Decimal("0.9")

    @pytest.mark.parametrize(
        "value, python, as_json",
        [
            (Decimal("85"), "Decimal('85')", "85"),
            (Fraction(17, 2), "Fraction(17, 2)", "8.5"),
            (Decimal("-1"), "Decimal('-1')", "-1"),
        ],
    )
    def test_out_of_range_raises_quoting_the_value_as_written(
        self, value, python, as_json
    ):
        """In Python as code; on the importers as the JSON number it means."""
        assert _message(PATHS["python field"][0], value).startswith(
            HEAD + f"ComparableField(threshold={python}) is out of range. "
        )
        assert _message(PATHS["config field"][0], value).startswith(
            HEAD + f'"threshold": {as_json} is out of range. '
        )
        assert _message(PATHS["schema comparator-config"][0], value).startswith(
            HEAD + 'the field sets no "x-aws-stickler-threshold", so it adopts its '
            f'comparator\'s: "x-aws-stickler-comparator-config": {{"threshold": {as_json}}}. '
        )

    def test_a_decimal_nan_is_not_a_number(self):
        assert _message(PATHS["config field"][0], Decimal("NaN")).startswith(
            HEAD + '"threshold": NaN is not a number. '
        )

    def test_out_of_range_gets_the_percentage_fix(self):
        message = _message(PATHS["python field"][0], Decimal("85"))
        assert (
            "85 is above 1.0, so even identical values would not match. If 85 is "
            "a percentage, write ComparableField(threshold=0.85)."
        ) in message

    def test_numpy_out_of_range_reads_as_json(self):
        np = pytest.importorskip("numpy")
        assert _message(PATHS["schema field"][0], np.float32(85.0)).startswith(
            HEAD + '"x-aws-stickler-threshold": 85.0 is out of range. '
        )
        assert _message(PATHS["config comparator_config"][0], np.array(85.0)).startswith(
            HEAD + 'the field sets no "threshold", so it adopts its comparator\'s: '
            '"comparator_config": {"threshold": 85.0}. '
        )

    @pytest.mark.parametrize("path", list(PATHS))
    def test_a_numpy_boolean_is_a_boolean(self, path):
        np = pytest.importorskip("numpy")
        build, as_json, opens = PATHS[path]
        message = _message(build, np.True_)
        assert message.startswith(
            HEAD + opens("true" if as_json else repr(np.True_))
        )
        assert "is a boolean, not a number." in message
        assert "1.0" in message and "perfect similarity score" in message

    def test_an_array_with_more_than_one_value_is_not_a_number(self):
        np = pytest.importorskip("numpy")
        assert "is a numpy.ndarray, not a number." in _message(
            PATHS["python field"][0], np.array([0.9])
        )

    @pytest.mark.parametrize("value", ["0.9", True])
    def test_bool_and_str_still_raise(self, value):
        for path in PATHS:
            _message(PATHS[path][0], value)


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
            "_Inner is scored field by field, so ExactComparator is never called; "
            "the field takes only its threshold. If 85 is a percentage, write "
            "ComparableField(threshold=0.85) and drop the comparator."
        ) in message
        assert "ExactComparator(threshold=0.85)" not in message

    def test_config(self):
        message = _message(PATHS["config nested-model comparator_config"][0], 85)
        assert message.endswith(
            "On a nested model, which is scored field by field, only the "
            'threshold in "comparator_config" is read. Delete "comparator_config", '
            'and write "threshold": 0.85 on the field if 85 is a percentage.'
        )

    def test_config_with_a_named_comparator_deletes_both(self):
        """Deleting only ``comparator_config`` would leave a comparator that warns."""
        message = _message(
            lambda v: _config(
                {
                    "type": "structured_model",
                    "comparator": "LevenshteinComparator",
                    "comparator_config": {"threshold": v},
                    "fields": {"a": _LEAF},
                }
            ),
            85,
        )
        assert message.endswith(
            'On a nested model, which is scored field by field, "comparator" is '
            'never called and only the threshold in "comparator_config" is read. '
            'Delete "comparator" and "comparator_config", and write "threshold": '
            "0.85 on the field if 85 is a percentage."
        )

    def test_schema_with_a_named_comparator_deletes_both(self):
        message = _message(
            lambda v: _schema(
                {
                    "type": "object",
                    "properties": {"a": _SLEAF},
                    "x-aws-stickler-comparator": "LevenshteinComparator",
                    "x-aws-stickler-comparator-config": {"threshold": v},
                }
            ),
            -1,
        )
        assert message.endswith(
            "On an object with properties, which is scored field by field, "
            '"x-aws-stickler-comparator" is never called and only the threshold '
            'in "x-aws-stickler-comparator-config" is read. Delete '
            '"x-aws-stickler-comparator" and "x-aws-stickler-comparator-config", '
            'and write "x-aws-stickler-threshold" on the property with a number '
            "from 0.0 to 1.0."
        )

    def test_schema(self):
        message = _message(PATHS["schema nested-object comparator-config"][0], 85)
        assert message.endswith(
            "On an object with properties, which is scored field by field, only "
            'the threshold in "x-aws-stickler-comparator-config" is read. Delete '
            '"x-aws-stickler-comparator-config", and write '
            '"x-aws-stickler-threshold": 0.85 on the property if 85 is a '
            "percentage."
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
        """``__annotations__`` holds ``"List[_Inner]"``, a string, there.

        ``__name__`` makes this module the class's, as in a real module, so
        pydantic resolves ``_Inner`` when the class is defined.
        """
        namespace = {
            "__name__": __name__,
            "StructuredModel": StructuredModel,
            "List": List,
            "_Inner": _Inner,
        }
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
        "build, body",
        [
            (
                lambda: _python(List[_Inner], ComparableField(threshold=85), attr="items"),
                "ComparableField(threshold=85) has no effect on a List[_Inner] field: "
                "its items are matched up using _Inner.match_threshold. Remove "
                "threshold=85. To change how items match, add match_threshold = 0.85 "
                "to the body of class _Inner (it must be from 0.0 to 1.0, so 85 would "
                "fail there too).",
            ),
            (
                lambda: _config(
                    {"type": "list_structured_model", "threshold": 85, "fields": {"a": _LEAF}},
                    attr="items",
                ),
                '"threshold": 85 has no effect on a list_structured_model field: its '
                'items are matched up using "match_threshold". Remove "threshold". To '
                'change how items match, write "match_threshold": 0.85 on the field '
                "(it must be from 0.0 to 1.0, so 85 would fail there too).",
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
                '"x-aws-stickler-threshold": 85 has no effect on an array of objects: '
                'its items are matched up using "x-aws-stickler-match-threshold" in '
                'its "items". Remove "x-aws-stickler-threshold". To change how items '
                'match, write "x-aws-stickler-match-threshold": 0.85 in its "items" '
                "(it must be from 0.0 to 1.0, so 85 would fail there too).",
            ),
        ],
    )
    def test_a_field_threshold_there_says_to_remove_it(self, build, body):
        """It has no effect whatever its value, so the range is not taught first.

        Headed ``Unused`` for any value, as for a valid one.
        """
        with pytest.raises(ValueError) as caught:
            build()
        message = str(caught.value)
        assert message.startswith(
            "Unused threshold for field 'items' of model 'Invoice': " + body
        )
        assert RULE not in message

    def test_the_comparator_threshold_warning_names_the_element_and_the_caller(self):
        field = ComparableField(comparator=LevenshteinComparator(threshold=85))
        line = inspect.currentframe().f_lineno + 3
        with pytest.warns(UserWarning) as caught:
            warnings.simplefilter("always")
            type("WarnsOrder", (StructuredModel,), {"__annotations__": {"items": List[_Inner]}, "items": field})
        [warning] = [w for w in caught if "WarnsOrder" in str(w.message)]
        assert str(warning.message) == (
            "Field 'items' of model 'WarnsOrder' is a List[_Inner] field, so "
            "LevenshteinComparator(threshold=85) is never used: its items are "
            "matched up using _Inner.match_threshold. Remove the comparator. To "
            "change how items match, add match_threshold = 0.85 to the body of "
            "class _Inner (it must be from 0.0 to 1.0, so 85 would fail there too)."
        )
        assert (os.path.abspath(warning.filename), warning.lineno) == (
            os.path.abspath(__file__),
            line,
        )


class _Sets06(StructuredModel):
    match_threshold = 0.6
    a: str = ComparableField()


class TestAValidThresholdOnAListOfModels:
    """Still refused, as unused: the same template, naming the element class."""

    def test_the_message(self):
        line = inspect.currentframe().f_lineno + 2
        with pytest.raises(ValueError) as caught:
            _python(List[_Inner], ComparableField(threshold=0.9), name="Order", attr="items")
        assert str(caught.value) == (
            "Unused threshold for field 'items' of model 'Order': "
            "ComparableField(threshold=0.9) has no effect on a List[_Inner] "
            "field: its items are matched up using _Inner.match_threshold. Remove "
            "threshold=0.9. To change how items match, add match_threshold = 0.9 "
            f"to the body of class _Inner. (Set at {_HERE}:{line}.)"
        )

    @pytest.mark.parametrize(
        "value, written, moved",
        [(Decimal("0.9"), "Decimal('0.9')", "0.9"), (Fraction(1, 2), "Fraction(1, 2)", "0.5")],
    )
    def test_the_value_is_quoted_as_written(self, value, written, moved):
        message = _message(
            lambda v: _python(List[_Inner], ComparableField(threshold=v), attr="items"),
            value,
        )
        assert f"ComparableField(threshold={written}) has no effect" in message
        assert f"Remove threshold={written}." in message
        assert f"add match_threshold = {moved} to the body of class _Inner." in message

    @pytest.mark.parametrize(
        "threshold, change",
        [
            (85, "To change how items match, set _Sets06.match_threshold, now 0.6, "
             "to 0.85 (it must be from 0.0 to 1.0, so 85 would fail there too)."),
            (0.85, "To change how items match, set _Sets06.match_threshold, now "
             "0.6, to 0.85."),
            (0.6, "_Sets06.match_threshold is already 0.6."),
        ],
    )
    def test_an_element_class_that_sets_match_threshold(self, threshold, change):
        """Its value is shown and changed, not added, under one header."""
        message = _message(
            lambda v: _python(
                List[_Sets06], ComparableField(threshold=v), name="Order", attr="items"
            ),
            threshold,
        )
        assert message.startswith(
            "Unused threshold for field 'items' of model 'Order': "
            f"ComparableField(threshold={threshold}) has no effect on a "
            "List[_Sets06] field: its items are matched up using "
            f"_Sets06.match_threshold. Remove threshold={threshold}. {change} "
        )

    def test_the_comparator_warning_shows_the_value_too(self):
        field = ComparableField(comparator=LevenshteinComparator(threshold=0.85))
        with pytest.warns(UserWarning) as caught:
            warnings.simplefilter("always")
            type(
                "WarnsSets06",
                (StructuredModel,),
                {"__annotations__": {"items": List[_Sets06]}, "items": field},
            )
        [warning] = [w for w in caught if "WarnsSets06" in str(w.message)]
        assert str(warning.message).endswith(
            "Remove the comparator. To change how items match, set "
            "_Sets06.match_threshold, now 0.6, to 0.85."
        )

    def test_zero_is_not_echoed_back(self):
        message = _message(
            lambda v: _python(List[_Inner], ComparableField(threshold=v), attr="items"),
            0.0,
        )
        assert (
            "To change how items match, add a positive match_threshold to the "
            "body of class _Inner: 0.0 would classify every paired object as a "
            "true positive. See "
        ) in message

    @pytest.mark.parametrize(
        "threshold, change",
        [
            (0, "To change how items match, set _Sets06.match_threshold, now 0.6, "
             "to another positive number: 0.0 would classify every paired object "
             "as a true positive. See "),
            (True, "To change how items match, set _Sets06.match_threshold, now "
             "0.6, to another number from 0.0 to 1.0. (Set at "),
            (float("nan"), "To change how items match, set _Sets06.match_threshold, "
             "now 0.6, to another number from 0.0 to 1.0. (Set at "),
        ],
    )
    def test_no_value_to_move_does_not_tell_the_element_class_to_keep_its_own(
        self, threshold, change
    ):
        """0.6 is already positive, and already a number from 0.0 to 1.0."""
        message = _message(
            lambda v: _python(
                List[_Sets06], ComparableField(threshold=v), name="Order", attr="items"
            ),
            threshold,
        )
        assert change in message

    def test_an_element_class_at_zero_is_told_to_change(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # its own zero-threshold warning
            zero = type(
                "_AtZero",
                (StructuredModel,),
                {
                    "__annotations__": {"sku": str},
                    "match_threshold": 0.0,
                    "sku": ComparableField(),
                },
            )
        message = _message(
            lambda v: _python(List[zero], ComparableField(threshold=v), attr="items"),
            0.0,
        )
        assert (
            "Remove threshold=0.0. To change how items match, set "
            "_AtZero.match_threshold, now 0.0, to a positive number: 0.0 "
            "classifies every paired object as a true positive. See "
        ) in message

    @pytest.mark.parametrize(
        "threshold, change",
        [
            (0.0, "set _Sets06.match_threshold, now 0.6, to another positive "
             "number: 0.0 would classify every paired object as a true positive."),
            (True, "set _Sets06.match_threshold, now 0.6, to another number from "
             "0.0 to 1.0."),
        ],
    )
    def test_the_comparator_warning_words_it_the_same(self, threshold, change):
        field = ComparableField(comparator=LevenshteinComparator(threshold=threshold))
        name = f"WarnsSets06{type(threshold).__name__}"
        with pytest.warns(UserWarning) as caught:
            warnings.simplefilter("always")
            type(
                name,
                (StructuredModel,),
                {"__annotations__": {"items": List[_Sets06]}, "items": field},
            )
        [warning] = [
            w for w in caught if f"{name}' is a List" in str(w.message)
        ]
        assert f"Remove the comparator. To change how items match, {change}" in str(
            warning.message
        )


def _exec_models(source, *, postponed=True, names=None):
    """Run ``source`` as a module, with ``from __future__ import annotations``.

    ``names`` adds globals, such as a base class from another module.
    """
    namespace = {
        **(names or {}),
        "Annotated": Annotated,
        "ClassVar": ClassVar,
        "Dict": Dict,
        "Tuple": Tuple,
        "StructuredModel": StructuredModel,
        "ComparableField": ComparableField,
        "LevenshteinComparator": LevenshteinComparator,
        "List": List,
        "Optional": Optional,
        "Union": Union,
        "BaseModel": BaseModel,
        "enum": enum,
    }
    flags = __future__.annotations.compiler_flag if postponed else 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(
            compile(source, "order_models.py", "exec", flags=flags, dont_inherit=True),
            namespace,
        )
    return namespace


_ADDRESS = "class {name}(StructuredModel):\n    city: str = ComparableField()\n"


class TestAForwardReference:
    """Checked when the class is defined, before the reference resolves.

    Nothing checks the field once it does, so an unresolved annotation is read
    as adopting its comparator's threshold, even when it is visibly a list:
    stickler cannot tell yet whether the element is a model, and only a list
    of models does not adopt.
    """

    FIELDS = {
        "addr: Address": ("addr: Address", "Address", True),
        '"Later"': ('later: "Later"', "Later", True),
        'Optional["Address2"]': ('opt: Optional["Address2"]', "Address2", True),
        "addr: Address, not postponed": ('addr: "Address"', "Address", False),
    }

    @staticmethod
    def _source(field, target, threshold):
        return (
            "class Order(StructuredModel):\n"
            f"    {field} = ComparableField(\n"
            f"        comparator=LevenshteinComparator(threshold={threshold})\n"
            "    )\n" + _ADDRESS.format(name=target)
        )

    @pytest.mark.parametrize("kind", ["percent", "string", "negative"])
    @pytest.mark.parametrize("shape", list(FIELDS))
    def test_an_adopted_comparator_threshold_raises(self, shape, kind):
        field, target, postponed = self.FIELDS[shape]
        value, in_python, _, wrong = BAD[kind]
        with pytest.raises(ValueError) as caught:
            _exec_models(self._source(field, target, in_python), postponed=postponed)
        name = field.partition(":")[0]
        message = str(caught.value)
        assert message.startswith(
            f"Invalid threshold for field '{name}' of model 'Order': "
            f"{ADOPTS}LevenshteinComparator(threshold={in_python}). "
        )
        assert wrong in message
        assert message.endswith("(Set at order_models.py:2.)")

    def test_the_documented_example(self):
        with pytest.raises(ValueError) as caught:
            _exec_models(self._source("addr: Address", "Address", 85))
        assert str(caught.value) == (
            "Invalid threshold for field 'addr' of model 'Order': "
            f"{ADOPTS}LevenshteinComparator(threshold=85). {RULE} 85 is above "
            "1.0, so even identical values would not match. If 85 is a "
            "percentage, write LevenshteinComparator(threshold=0.85), or "
            "ComparableField(threshold=0.85), which wins over the comparator's. "
            "(Set at order_models.py:2.)"
        )

    @pytest.mark.parametrize("shape", list(FIELDS))
    def test_a_valid_one_loads_and_gates_once_resolved(self, shape):
        field, target, postponed = self.FIELDS[shape]
        source = self._source(field, target, 0.9) + "Order.model_rebuild()\n"
        order = _exec_models(source, postponed=postponed)["Order"]
        assert order._get_comparison_info(field.partition(":")[0]).threshold == 0.9

    #: (field, what is defined below it, postponed): each is a list whose element
    #: is not a model once resolved, so the field adopts the 85, and defined
    #: above the class each raises.
    NOT_MODELS = {
        "an Enum": ("tags: List[Color]", "class Color(str, enum.Enum):\n    RED = 'red'\n", True),
        "a str subclass": ('codes: List["Code"]', "class Code(str):\n    pass\n", False),
        "a plain BaseModel": ('rows: List["Plain"]', "class Plain(BaseModel):\n    a: str\n", False),
        "a union of models": (
            'items: List[Union["A", "B"]]',
            _ADDRESS.format(name="A") + _ADDRESS.format(name="B"),
            False,
        ),
    }

    @pytest.mark.parametrize("shape", list(NOT_MODELS))
    @pytest.mark.parametrize("defined", ["below", "above"])
    def test_a_list_of_what_is_not_a_model_raises_either_way(self, shape, defined):
        field, target, postponed = self.NOT_MODELS[shape]
        body = (
            "class Order(StructuredModel):\n"
            f"    {field} = ComparableField(\n"
            "        comparator=LevenshteinComparator(threshold=85)\n"
            "    )\n"
        )
        source = body + target if defined == "below" else target + body
        with pytest.raises(ValueError, match="so even identical values would not match"):
            _exec_models(source, postponed=postponed)

    @pytest.mark.parametrize(
        "field, postponed",
        [
            ('items: List["Later"]', False),
            ('items: Optional[List["Later"]]', False),
            ("items: List[Later]", True),
            ("items: Optional[List[Later]]", True),
            ("items: list[Later] | None", True),
        ],
    )
    def test_a_list_of_models_defined_later_raises_and_says_to_remove_the_comparator(
        self, field, postponed
    ):
        """Resolved, it would only warn; unresolved, the element may not be a model."""
        with pytest.raises(ValueError) as caught:
            _exec_models(self._source(field, "Later", 85), postponed=postponed)
        assert str(caught.value) == (
            "Invalid threshold for field 'items' of model 'Order': "
            f"{ADOPTS}LevenshteinComparator(threshold=85). {RULE} 85 is above "
            "1.0, so even identical values would not match. Later is not defined "
            "yet, so stickler cannot tell whether it is a StructuredModel. If it "
            "is, remove the comparator: a list of models matches its items using "
            "Later.match_threshold and never calls it. If it is not, write "
            "LevenshteinComparator(threshold=0.85), or "
            "ComparableField(threshold=0.85), which wins over the comparator's. "
            "(Set at order_models.py:2.)"
        )

    def test_a_valid_comparator_threshold_on_a_list_defined_later_loads(self):
        _exec_models(self._source("items: List[Later]", "Later", 0.9))

    @pytest.mark.parametrize(
        "field, element",
        [
            ("items: List[LaterItem]", "LaterItem"),
            ('items: List["LaterItem"]', "LaterItem"),
            ("items: Optional[List[LaterItem]]", "LaterItem"),
            ("items: list[LaterItem] | None", "LaterItem"),
        ],
    )
    def test_a_field_threshold_on_a_list_says_what_to_do_either_way(self, field, element):
        """A 0.85 moved onto the field would load and decide nothing for a model."""
        source = (
            f"class Order(StructuredModel):\n    {field} = ComparableField(threshold=85)\n"
            + _ADDRESS.format(name="LaterItem")
        )
        with pytest.raises(ValueError) as caught:
            _exec_models(source)
        assert str(caught.value) == (
            "Invalid threshold for field 'items' of model 'Order': "
            f"ComparableField(threshold=85) is out of range. {RULE} 85 is above "
            f"1.0, so even identical values would not match. {element} is not "
            "defined yet, so stickler cannot tell whether it is a StructuredModel. "
            "If it is, remove threshold=85 and add match_threshold = 0.85 to the "
            f"body of class {element}: a threshold on a list of models is ignored. "
            "If it is not, write ComparableField(threshold=0.85). "
            "(Set at order_models.py:2.)"
        )

    @pytest.mark.parametrize(
        "field, name, below",
        [
            ('items: "ItemList"', "ItemList", "ItemList = List[LaterItem]\n"),
            ('vendor: "Vendor"', "Vendor", "class Vendor(str):\n    pass\n"),
            ('rows: "Dict[str, Later]"', "The field's type", ""),
        ],
    )
    def test_a_field_threshold_on_a_bare_name_says_what_to_do_either_way(
        self, field, name, below
    ):
        """It may name a list of models, where 0.85 would load and decide nothing."""
        source = (
            f"class Order(StructuredModel):\n    {field} = ComparableField(threshold=85)\n"
            + _ADDRESS.format(name="LaterItem")
            + below
        )
        with pytest.raises(ValueError) as caught:
            _exec_models(source, postponed=False)
        assert str(caught.value).endswith(
            f"{RULE} 85 is above 1.0, so even identical values would not match. "
            f"{name} is not defined yet, so stickler cannot tell whether it is a "
            "list of models. If it is, remove threshold=85 and add "
            "match_threshold = 0.85 to the element class: a threshold on a list "
            "of models is ignored. If it is not, write "
            "ComparableField(threshold=0.85). (Set at order_models.py:2.)"
        )

    def test_with_no_likely_number(self):
        source = "class Order(StructuredModel):\n    items: List[LaterItem] = ComparableField(threshold=-1)\n"
        with pytest.raises(ValueError) as caught:
            _exec_models(source)
        assert (
            "If it is, remove threshold=-1 and add match_threshold to the body of "
            "class LaterItem, as a number from 0.0 to 1.0: a threshold on a list "
            "of models is ignored. If it is not, write a number from 0.0 to 1.0 "
            "instead."
        ) in str(caught.value)


class TestAnAnnotatedForwardReference:
    """``Annotated[Address, ComparableField(...)]`` with ``Address`` defined below.

    pydantic holds the whole annotation as one string until ``model_rebuild()``,
    so the ``ComparableField`` in it does not exist when the class is defined.
    Its metadata is evaluated then, and checked as adopting. On dev each of
    these loaded and scored an identical pair 0.0, or raised ``TypeError``
    for the string.
    """

    FIELDS = {
        "a model below": "addr: Annotated[Address, {field}]",
        "Optional, defaulting to None": "addr: Annotated[Optional[Address], {field}] = None",
        "an alias of str below": "addr: Annotated[Code, {field}]",
    }
    BELOW = _ADDRESS.format(name="Address") + "Code = str\n"

    @pytest.mark.parametrize("postponed", [True, False], ids=["postponed", "quoted"])
    @pytest.mark.parametrize("shape", list(FIELDS))
    @pytest.mark.parametrize("kind", ["percent", "string", "bool"])
    def test_a_field_threshold_raises_at_class_definition(self, shape, kind, postponed):
        value, in_python, _, wrong = BAD[kind]
        annotation = self.FIELDS[shape].format(field=f"ComparableField(threshold={in_python})")
        if not postponed:
            name, _, rest = annotation.partition(": ")
            rest, _, default = rest.partition(" = ")
            annotation = f"{name}: {rest!r}" + (f" = {default}" if default else "")
        source = f"class Order(StructuredModel):\n    {annotation}\n" + self.BELOW
        with pytest.raises(ValueError) as caught:
            _exec_models(source, postponed=postponed)
        message = str(caught.value)
        assert message.startswith(
            "Invalid threshold for field 'addr' of model 'Order': "
            f"ComparableField(threshold={in_python}) "
        )
        assert wrong in message
        # The class line: the ComparableField call has no line of its own yet.
        assert message.endswith("(Set at order_models.py:1.)")

    def test_an_adopted_comparator_threshold_raises(self):
        source = (
            "class Order(StructuredModel):\n"
            "    addr: Annotated[Address, ComparableField("
            "comparator=LevenshteinComparator(threshold=85))]\n" + self.BELOW
        )
        with pytest.raises(ValueError) as caught:
            _exec_models(source)
        assert str(caught.value) == (
            "Invalid threshold for field 'addr' of model 'Order': "
            f"{ADOPTS}LevenshteinComparator(threshold=85). {RULE} 85 is above "
            "1.0, so even identical values would not match. If 85 is a "
            "percentage, write LevenshteinComparator(threshold=0.85), or "
            "ComparableField(threshold=0.85), which wins over the comparator's. "
            "(Set at order_models.py:1.)"
        )

    def test_a_list_of_models_below_says_to_remove_the_comparator(self):
        source = (
            "class Order(StructuredModel):\n"
            "    items: Annotated[List[Later], ComparableField("
            "comparator=LevenshteinComparator(threshold=85))]\n"
            + _ADDRESS.format(name="Later")
        )
        with pytest.raises(ValueError) as caught:
            _exec_models(source)
        assert (
            "Later is not defined yet, so stickler cannot tell whether it is a "
            "StructuredModel. If it is, remove the comparator"
        ) in str(caught.value)

    def test_a_valid_one_loads_and_gates_once_resolved(self):
        source = (
            "class Order(StructuredModel):\n"
            "    addr: Annotated[Address, ComparableField(threshold=0.6)]\n"
            + self.BELOW
            + "Order.model_rebuild()\n"
        )
        order = _exec_models(source)["Order"]
        assert order._get_comparison_info("addr").threshold == 0.6

    def test_metadata_that_names_a_value_defined_below_is_left_to_pydantic(self):
        source = (
            "class Order(StructuredModel):\n"
            "    addr: Annotated[Address, ComparableField(threshold=LATER)]\n"
            + self.BELOW
            + "LATER = 0.6\nOrder.model_rebuild()\n"
        )
        order = _exec_models(source)["Order"]
        assert order._get_comparison_info("addr").threshold == 0.6

    def test_a_class_defined_in_a_function_reads_its_locals(self):
        comparator = LevenshteinComparator(threshold=85)  # noqa: F841 - read by name

        with pytest.raises(ValueError, match="so even identical values would not match"):
            type(
                "Order",
                (StructuredModel,),
                {
                    "__annotations__": {
                        "addr": "Annotated[Address, ComparableField(comparator=comparator)]"
                    }
                },
            )


    def test_a_class_body_name_wins_over_a_module_global(self):
        """As pydantic resolves the annotation; on dev this loaded and gated at 0.9."""
        source = (
            "THR = 85\n"
            "class Order(StructuredModel):\n"
            "    THR: ClassVar[float] = 0.9\n"
            "    addr: Annotated[Address, ComparableField(threshold=THR)]\n"
            + self.BELOW
            + "Order.model_rebuild()\n"
        )
        order = _exec_models(source)["Order"]
        assert order._get_comparison_info("addr").threshold == 0.9

    def test_an_invalid_class_body_name_raises(self):
        source = (
            "THR = 0.9\n"
            "class Order(StructuredModel):\n"
            "    THR: ClassVar[float] = 85\n"
            "    addr: Annotated[Address, ComparableField(threshold=THR)]\n"
            + self.BELOW
        )
        with pytest.raises(ValueError, match=r"ComparableField\(threshold=85\) is out"):
            _exec_models(source)

    def test_a_subclass_does_not_evaluate_an_inherited_field_again(self):
        """Not in its own module, where LIMIT is 85, and not a second time."""
        base = _exec_models(
            "BUILT = []\n"
            "def built(value):\n    BUILT.append(value)\n    return value\n"
            "LIMIT = 0.9\n"
            "class Base(StructuredModel):\n"
            "    addr: Annotated[Later, ComparableField(threshold=built(LIMIT))]\n"
        )
        assert base["BUILT"] == [0.9]
        child = _exec_models(
            "LIMIT = 85\nclass Child(Base):\n    name: str = ComparableField()\n",
            postponed=False,
            names={"Base": base["Base"], "built": base["built"]},
        )["Child"]
        assert base["BUILT"] == [0.9]
        assert {"addr", "name"} <= set(child.model_fields)


class TestAPep695Alias:
    """Read as the type it names once the type it names is defined, as scoring reads it."""

    @staticmethod
    def _order(threshold):
        from typing_extensions import TypeAliasType

        items = TypeAliasType("Items", List[_Inner])
        return type(
            "Order",
            (StructuredModel,),
            {
                "__annotations__": {"items": items},
                "items": ComparableField(
                    comparator=LevenshteinComparator(threshold=threshold)
                ),
            },
        )

    def test_a_list_of_models_alias_does_not_adopt(self):
        """85 loads, as on dev, and an identical pair is a match."""
        order = self._order(85)
        pair = order(items=[_Inner(a="x")])
        assert pair.compare_with(pair)["overall_score"] == 1.0

    @pytest.mark.skipif(sys.version_info < (3, 12), reason="the type statement")
    @pytest.mark.parametrize("threshold, raises", [(85, True), (0.9, False)])
    def test_an_alias_of_a_list_defined_later_is_checked_as_adopting(
        self, threshold, raises
    ):
        """Before ``Later`` exists the alias cannot be read, so 85 raises; 0.9 loads."""
        source = (
            "type Items = list[Later]\n"
            "class Order(StructuredModel):\n"
            "    items: Items = ComparableField(\n"
            f"        comparator=LevenshteinComparator(threshold={threshold})\n"
            "    )\n" + _ADDRESS.format(name="Later")
        )
        if raises:
            with pytest.raises(
                ValueError, match="so even identical values would not match"
            ):
                _exec_models(source, postponed=False)
        else:
            _exec_models(source, postponed=False)

    def test_an_alias_of_a_scalar_adopts(self):
        from typing_extensions import TypeAliasType

        code = TypeAliasType("Code", str)
        with pytest.raises(ValueError, match="so even identical values would not match"):
            type(
                "Invoice",
                (StructuredModel,),
                {
                    "__annotations__": {"vendor": code},
                    "vendor": ComparableField(
                        comparator=LevenshteinComparator(threshold=85)
                    ),
                },
            )


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


_STRICT85_LINE = inspect.currentframe().f_lineno + 3


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
        assert (
            "ComparableField sets no threshold, so the field adopts the threshold "
            "its comparator _Strict85 sets, 85."
        ) in python
        assert (
            "Write ComparableField(threshold=...) with a number from 0.0 to 1.0, "
            "which wins over the comparator's. (Set at "
        ) in python
        assert python.endswith(f"_Strict85 is defined at {_HERE}:{_STRICT85_LINE}.)")
        config = _message(lambda _: _config({"type": "str", "comparator": strict85}), None)
        assert (
            'the field sets no "threshold", so it adopts the threshold its '
            f'comparator "{strict85}" sets, 85.'
        ) in config
        assert config.endswith(
            'Write "threshold" on the field with a number from 0.0 to 1.0, which '
            "wins over the comparator's."
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
            'On a nested model, which is scored field by field, "comparator" is '
            'never called and only its threshold is read. Delete "comparator", and '
            'write "threshold" on the field with a number from 0.0 to 1.0.'
        )


_FILLS_IN_LINE = inspect.currentframe().f_lineno + 3


class _FillsIn85(BaseComparator):
    """Takes ``threshold``, and sets 85 itself when it is not passed."""

    def __init__(self, threshold: Optional[float] = None):
        super().__init__(threshold=85 if threshold is None else threshold)

    def _compare(self, a, b):
        return 1.0 if a == b else 0.0


class TestACustomComparatorThatTakesAThreshold:
    """Its call is not quoted: the author may not have written ``threshold=85``."""

    def test_the_message_says_the_comparators_threshold(self):
        message = _message(
            lambda _: _python(str, ComparableField(comparator=_FillsIn85())), None
        )
        assert message.startswith(
            HEAD + "ComparableField sets no threshold, so the field adopts its "
            f"comparator's, which for _FillsIn85 is 85. {RULE} 85 is above 1.0, "
            "so even identical values would not match. If 85 is a percentage, "
            "write ComparableField(threshold=0.85), which wins over the "
            "comparator's, or pass threshold=0.85 to _FillsIn85. If _FillsIn85 "
            "sets 85 itself, make that default a number from 0.0 to 1.0. If the "
            "field should not adopt it, pass threshold to super().__init__() "
            "unchanged and put the default in DEFAULT_THRESHOLD; the field then "
            "uses its own default. (Set at "
        )
        assert message.endswith(f"_FillsIn85 is defined at {_HERE}:{_FILLS_IN_LINE}.)")
        assert "_FillsIn85(threshold=" not in message

    def test_a_stickler_comparator_is_quoted(self):
        """Each built-in passes ``threshold`` through, so the call is the author's."""

        class MyLevenshtein(LevenshteinComparator):
            pass

        message = _message(
            lambda v: _python(
                str, ComparableField(comparator=MyLevenshtein(threshold=v))
            ),
            85,
        )
        assert "adopts its comparator's: MyLevenshtein(threshold=85)." in message
        assert "is defined at" not in message

    def test_every_built_in_passes_the_threshold_through(self):
        """What quoting a built-in's call relies on."""
        registry = comparator_registry._global_registry
        checked = 0
        for name in registry.list_available():
            cls = registry.get(name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    bare, named = cls(), cls(threshold=0.3)
                except Exception:
                    continue  # needs an extra or a required argument
            assert bare.threshold_was_set is False, name
            assert (named.threshold, named.threshold_was_set) == (0.3, True), name
            checked += 1
        assert checked >= 8


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
    with pytest.raises(ValueError, match=r"Write ComparableField\(threshold=0.9\), without quotes."):
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
        assert wrong.replace("values", "objects") in message

    def test_the_percentage_fix(self):
        message = _message(self._builds()["config nested"][0], 85)
        assert message == (
            "Invalid match threshold for field 'billing' of model 'Invoice': "
            f'"match_threshold": 85 is out of range. {self.MATCH_RULE} 85 is above '
            "1.0, so even identical objects would not match. If 85 is a "
            'percentage, write "match_threshold": 0.85.'
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


class TestTheRuntimeMatchThresholdIsOutsideTheRule:
    """``evaluate()``, ``eval_for()`` and ``from_pydantic()`` keep their own check.

    For a plain pydantic model they build shadow classes, which the rule would
    check under a generated name (``FlatEval``) the caller never wrote. So the
    runtime value gets the check it always had, with its message, and a
    ``bool`` still counts as 1 or 0. For a ``StructuredModel`` it is not checked.
    """

    @staticmethod
    def _models():
        from pydantic import BaseModel

        class Addr(BaseModel):
            city: str

        class Line(BaseModel):
            sku: str

        class Flat(BaseModel):
            vendor: str
            total: float

        class Nested(BaseModel):
            vendor: str
            addr: Addr

        class Lines(BaseModel):
            items: List[Line]

        return {
            "flat": Flat(vendor="ACME", total=1.0),
            "nested": Nested(vendor="ACME", addr=Addr(city="X")),
            "list": Lines(items=[Line(sku="a")]),
        }

    @pytest.mark.parametrize("value", [85, -1, "0.9", float("nan"), Decimal("0.9")])
    @pytest.mark.parametrize("shape", ["flat", "nested", "list"])
    def test_a_plain_model_raises_the_older_message(self, shape, value):
        import stickler

        model = self._models()[shape]
        with pytest.raises(ValueError) as caught:
            stickler.evaluate(model, model, match_threshold=value)
        assert str(caught.value) == (
            f"match_threshold must be a number between 0.0 and 1.0, got: {value}"
        )
        with pytest.raises(ValueError, match="^match_threshold must be a number"):
            stickler.eval_for(type(model), match_threshold=value)

    @pytest.mark.parametrize("value", [True, False, 1, 0.8])
    @pytest.mark.parametrize("shape", ["flat", "nested", "list"])
    def test_a_plain_model_accepts_what_it_accepted(self, shape, value):
        import stickler

        model = self._models()[shape]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stickler.evaluate(model, model, match_threshold=value)
            stickler.eval_for(type(model), match_threshold=value)
        assert result.overall_score == 1.0
        assert result.matched is True

    @pytest.mark.parametrize("value, warns", [(False, False), (0.0, True)])
    def test_false_does_not_warn_that_the_threshold_is_zero(self, value, warns):
        """``False`` builds with 0.0 but warns as little as it did; 0.0 still warns."""
        import stickler

        name = f"RuntimeZero{type(value).__name__.title()}"
        line = type(f"{name}Line", (BaseModel,), {"__annotations__": {"sku": str}})
        flat = type(
            name,
            (BaseModel,),
            {"__annotations__": {"meta": dict, "lines": List[line]}},
        )
        pair = flat(meta={"a": 1}, lines=[line(sku="a")])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = stickler.evaluate(pair, pair, match_threshold=value)
        zero = [w for w in caught if "=0.0. The threshold test is" in str(w.message)]
        assert bool(zero) is warns
        assert (result.overall_score, result.matched) == (1.0, True)

    def test_a_structured_model_is_not_checked(self):
        import stickler

        model = PATHS["python field"][0](0.9)
        pair = model(vendor="Acme")
        assert stickler.evaluate(pair, pair, match_threshold=85).matched is False
        assert stickler.evaluate(pair, pair, match_threshold=True).matched is True
