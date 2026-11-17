"""A field's threshold must be a number from 0.0 to 1.0, on every path.

The rule covers the threshold the field ends up with: its own, or one it adopts
from the comparator. Anything else raises ``ValueError`` when the class is
defined or the config or schema is loaded, never during evaluation, with one
message naming the model, the field, where the value came from, the value, the
range and the likely fix.

Before, the Python path checked neither ``ComparableField(threshold=85)`` nor
``LevenshteinComparator(threshold=85)``, so an identical pair scored 0.0 FD;
the importers ignored an out-of-range comparator threshold with a warning; and
a field threshold of ``True`` loaded as 1 on the importers.
"""

import math
import re
import warnings
from typing import List, Optional

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

RANGE = "a threshold is a number from 0.0 to 1.0, the fraction of similarity a match needs"

#: Each bad kind, with the likely fix its message must give.
BAD = {
    "percent": (85, "For 85%, use 0.85."),
    "above one": (1.5, "For 1.5%, use 0.015."),
    "negative": (-1, "-1 is negative, so every pair would match."),
    "string": ("0.9", "'0.9' is a string; write 0.9 without quotes."),
    "bool": (True, "True is a boolean, not a number."),
    "nan": (float("nan"), "NaN never compares, so no pair would match."),
    "inf": (float("inf"), "inf is not finite, so no pair would match."),
}


def _shown(value):
    """The value as the message shows it: repr, plus the type if not a number."""
    if isinstance(value, (bool, str)):
        return f"{value!r} ({type(value).__name__})"
    return repr(value)


def _python(annotation, field, name="Invoice", attr="vendor"):
    return type(
        name,
        (StructuredModel,),
        {"__annotations__": {attr: Optional[annotation]}, attr: field},
    )


def _config(field_config, name="Invoice", attr="vendor"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StructuredModel.model_from_json(
            {"model_name": name, "fields": {attr: field_config}}
        )


def _schema(prop, title="Invoice", attr="vendor", **root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StructuredModel.from_json_schema(
            {"type": "object", "title": title, "properties": {attr: prop}, **root}
        )


class _Inner(StructuredModel):
    a: str = ComparableField(comparator=ExactComparator(), threshold=1.0)


_LEAF = {"type": "str", "comparator": "ExactComparator", "threshold": 1.0}
_SLEAF = {
    "type": "string",
    "x-aws-stickler-comparator": "ExactComparator",
    "x-aws-stickler-threshold": 1.0,
}

# path -> (build(value), source as the message names it, key that overrides a
# comparator's value or None). One row per place a threshold becomes a gate.
PATHS = {
    "python field": (
        lambda v: _python(str, ComparableField(threshold=v)),
        lambda v: f"ComparableField(threshold={v!r})",
        None,
    ),
    "python comparator": (
        lambda v: _python(
            str, ComparableField(comparator=LevenshteinComparator(threshold=v))
        ),
        lambda v: f"LevenshteinComparator(threshold={v!r})",
        "ComparableField(threshold=...)",
    ),
    "python nested-model comparator": (
        lambda v: _python(
            _Inner, ComparableField(comparator=ExactComparator(threshold=v))
        ),
        lambda v: f"ExactComparator(threshold={v!r})",
        "ComparableField(threshold=...)",
    ),
    "config field": (
        lambda v: _config(
            {"type": "str", "comparator": "LevenshteinComparator", "threshold": v}
        ),
        lambda v: "threshold",
        None,
    ),
    "config comparator_config": (
        lambda v: _config(
            {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "comparator_config": {"threshold": v},
            }
        ),
        lambda v: "comparator_config.threshold",
        "the field's threshold",
    ),
    "config nested-model comparator_config": (
        lambda v: _config(
            {
                "type": "structured_model",
                "comparator_config": {"threshold": v},
                "fields": {"a": _LEAF},
            }
        ),
        lambda v: "comparator_config.threshold",
        "the field's threshold",
    ),
    "schema field": (
        lambda v: _schema({"type": "string", "x-aws-stickler-threshold": v}),
        lambda v: "x-aws-stickler-threshold",
        None,
    ),
    "schema comparator-config": (
        lambda v: _schema(
            {
                "type": "string",
                "x-aws-stickler-comparator": "LevenshteinComparator",
                "x-aws-stickler-comparator-config": {"threshold": v},
            }
        ),
        lambda v: "x-aws-stickler-comparator-config threshold",
        "x-aws-stickler-threshold",
    ),
    "schema nested-object comparator-config": (
        lambda v: _schema(
            {
                "type": "object",
                "properties": {"a": _SLEAF},
                "x-aws-stickler-comparator-config": {"threshold": v},
            }
        ),
        lambda v: "x-aws-stickler-comparator-config threshold",
        "x-aws-stickler-threshold",
    ),
}


@pytest.mark.parametrize("kind", list(BAD))
@pytest.mark.parametrize("path", list(PATHS))
def test_every_path_raises_for_every_bad_kind(path, kind):
    build, source, field_key = PATHS[path]
    value, fix = BAD[kind]
    with pytest.raises(ValueError) as caught:
        build(value)
    message = str(caught.value)
    assert (
        f"Invalid threshold for field 'vendor' of model 'Invoice': "
        f"{source(value)} gives {_shown(value)}, but {RANGE}. {fix}"
    ) in message
    override = f" Or set {field_key} to override the comparator's value."
    assert message.endswith(override if field_key else fix)


@pytest.mark.parametrize("path", list(PATHS))
def test_a_valid_threshold_is_adopted(path):
    build, _, _ = PATHS[path]
    model = build(0.9)
    assert model._get_comparison_info("vendor").threshold == 0.9


def test_the_message_matches_the_documented_example():
    with pytest.raises(ValueError) as caught:
        _python(str, ComparableField(comparator=LevenshteinComparator(threshold=85)))
    assert str(caught.value) == (
        "Invalid threshold for field 'vendor' of model 'Invoice': "
        "LevenshteinComparator(threshold=85) gives 85, but a threshold is a "
        "number from 0.0 to 1.0, the fraction of similarity a match needs. "
        "For 85%, use 0.85. Or set ComparableField(threshold=...) to override "
        "the comparator's value."
    )


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


class TestAListOfModelsDoesNotCheckAComparatorThreshold:
    """Pairing uses the element class's ``match_threshold``; each path warns."""

    def test_no_path_raises(self):
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

    def test_a_field_threshold_there_is_still_checked(self):
        """It raises anyway; the range error comes first, so its fix is sound."""
        with pytest.raises(ValueError, match=r"ComparableField\(threshold=85\) gives 85"):
            _python(List[_Inner], ComparableField(threshold=85))


class TestTheMessageNamesTheModelAsTheAuthorDoes:
    def test_a_schema_uses_its_model_name_then_its_title(self):
        prop = {"type": "string", "x-aws-stickler-comparator-config": {"threshold": 85}}
        with pytest.raises(ValueError, match="of model 'Receipt'"):
            _schema(prop, title="Ignored", **{"x-aws-stickler-model-name": "Receipt"})
        with pytest.raises(ValueError, match="of model 'Invoice'"):
            _schema(prop, title="Invoice")

    def test_an_unnamed_model_is_dynamic_model_without_its_field_list(self):
        prop = {"type": "string", "x-aws-stickler-comparator-config": {"threshold": 85}}
        for build in (
            lambda: StructuredModel.from_json_schema(
                {"type": "object", "properties": {"vendor": prop}}
            ),
            lambda: StructuredModel.model_from_json(
                {
                    "fields": {
                        "vendor": {
                            "type": "str",
                            "comparator": "LevenshteinComparator",
                            "comparator_config": {"threshold": 85},
                        }
                    }
                }
            ),
        ):
            with pytest.raises(ValueError) as caught:
                build()
            assert "of model 'DynamicModel':" in str(caught.value)
            assert "(fields:" not in str(caught.value)

    def test_a_nested_field_is_named_by_its_path_in_the_root_model(self):
        with pytest.raises(ValueError, match="field 'billing.city' of model 'Invoice'"):
            _config(
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
            )
        with pytest.raises(ValueError, match="field 'billing.city' of model 'Invoice'"):
            _schema(
                {
                    "type": "object",
                    "properties": {
                        "city": {**_SLEAF, "x-aws-stickler-threshold": 85},
                    },
                },
                attr="billing",
            )

    def test_a_class_built_by_create_model_from_fields_is_named(self):
        with pytest.raises(ValueError, match="field 'vendor' of model 'Receipt'"):
            ModelFactory.create_model_from_fields(
                "Receipt", {"vendor": (str, ComparableField(threshold=85))}
            )


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


def test_a_threshold_the_comparator_sets_itself_names_the_comparator(strict85):
    source = re.escape("comparator '_Strict85' (its own threshold) gives 85")
    with pytest.raises(ValueError, match=source):
        _config({"type": "str", "comparator": strict85})
    with pytest.raises(ValueError, match=source):
        _schema({"type": "string", "x-aws-stickler-comparator": strict85})
    with pytest.raises(ValueError, match=source):
        _config({"type": "structured_model", "comparator": strict85, "fields": {"a": _LEAF}})


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


def test_a_numpy_scalar_is_a_number():
    np = pytest.importorskip("numpy")
    model = _python(str, ComparableField(threshold=np.float32(0.5)))
    assert model._get_comparison_info("vendor").threshold == 0.5


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("value", [0, 0.0, 1, 1.0, 0.5])
def test_the_bounds_are_inclusive(value):
    for build, _, _ in PATHS.values():
        assert math.isclose(build(value)._get_comparison_info("vendor").threshold, value)


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
