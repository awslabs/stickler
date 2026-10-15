"""The three entry points resolve threshold and clip the same way.

A field's threshold and clip are resolved by ``ComparableField`` on the Python
path. Both importers used to resolve their own defaults first and pass concrete
values, so every imported field read as "the author stated this", and two
defaults keyed on that marker never ran:

- a threshold named on the comparator was discarded on the config and schema
  paths, so a field gated at 0.9 scored TP there and FD in Python (#334, #348
  item 2);
- the object-grade clip default was suppressed, so a dict field's partial
  credit was zeroed below its threshold on both importers (#348 item 6).

Asserted against each other rather than against literals where the paths are
meant to agree, so they cannot drift apart again. The schema path's free-form
object threshold (1.0) still differs from Python's; that is #348 item 1,
deferred, so the mapping tests compare clip and score rather than threshold.

`x-aws-stickler-comparator` on an object property (#348 item 7) is covered by
``test_object_property_threshold.TestTheComparatorOnTheSameNodeIsRefused``.
"""

from typing import Any, Dict, Optional

import pytest

from stickler.comparators.anls import ANLSStarComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)

# Levenshtein scores this pair 0.7778: a TP at the legacy 0.5, an FD at 0.9.
GT_CODE, PRED_CODE = "ACME-1000", "ACNE-1009"
# ANLS* scores this pair 0.3333, below every threshold in play.
GT_META = {"a": "1", "b": "2", "c": "3"}
PRED_META = {"a": "1", "b": "9", "c": "9"}


def _python(annotation, field):
    return type(
        "Py",
        (StructuredModel,),
        {"__annotations__": {"f": Optional[annotation]}, "f": field},
    )


def _config(field_config):
    return StructuredModel.model_from_json(
        {"model_name": "Cfg", "fields": {"f": field_config}}
    )


def _schema(prop):
    return StructuredModel.from_json_schema(
        {"type": "object", "title": "Sch", "properties": {"f": prop}}
    )


def _info(model):
    return model._get_comparison_info("f")


def _marker(model, name):
    return getattr(model.model_fields["f"].json_schema_extra, name)


def _verdict(model, gt, pred):
    result = model(f=gt).compare_with(model(f=pred), include_confusion_matrix=True)
    overall = result["confusion_matrix"]["overall"]
    return result["field_scores"]["f"], overall["tp"], overall["fd"]


def _code_paths(comparator_threshold, field_threshold=None):
    """One string field, the comparator named with a threshold, on each path."""
    py_kwargs, cfg, sch = {}, {}, {}
    if field_threshold is not None:
        py_kwargs["threshold"] = field_threshold
        cfg["threshold"] = field_threshold
        sch["x-aws-stickler-threshold"] = field_threshold
    return {
        "python": _python(
            str,
            ComparableField(
                comparator=LevenshteinComparator(threshold=comparator_threshold),
                **py_kwargs,
            ),
        ),
        "config": _config(
            {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "comparator_config": {"threshold": comparator_threshold},
                **cfg,
            }
        ),
        "schema": _schema(
            {
                "type": "string",
                "x-aws-stickler-comparator": "LevenshteinComparator",
                "x-aws-stickler-comparator-config": {"threshold": comparator_threshold},
                **sch,
            }
        ),
    }


class TestAComparatorThresholdReachesTheFieldOnEveryPath:
    def test_the_three_paths_resolve_the_same_threshold(self):
        paths = _code_paths(0.9)
        assert {name: _info(m).threshold for name, m in paths.items()} == {
            "python": 0.9,
            "config": 0.9,
            "schema": 0.9,
        }

    def test_the_three_paths_reach_the_same_verdict(self):
        """0.7778 under a 0.9 gate: FD and clipped to 0.0 on all three."""
        verdicts = {
            name: _verdict(m, GT_CODE, PRED_CODE)
            for name, m in _code_paths(0.9).items()
        }
        assert set(verdicts.values()) == {(0.0, 0, 1)}

    def test_a_field_threshold_still_wins_on_every_path(self):
        paths = _code_paths(0.9, field_threshold=0.6)
        assert {_info(m).threshold for m in paths.values()} == {0.6}

    def test_it_is_not_marked_as_a_field_threshold(self):
        """Adopted from the comparator, so the field did not state it."""
        paths = _code_paths(0.9)
        assert {_marker(m, "_threshold_explicit") for m in paths.values()} == {False}

    def test_a_schema_config_without_a_comparator_name_still_reaches_it(self):
        """It configures the type's default comparator. The config path
        requires a comparator name on a primitive, so has no such spelling."""
        sch = _schema(
            {"type": "string", "x-aws-stickler-comparator-config": {"threshold": 0.9}}
        )
        assert _info(sch).threshold == 0.9

    def test_it_outranks_an_inferred_threshold(self):
        """Field, then comparator, then inference: the author wrote it."""
        cfg = _config(
            {
                "type": "str",
                "comparator": "auto",
                "comparator_config": {"threshold": 0.9},
            }
        )
        sch = _schema(
            {
                "type": "string",
                "x-aws-stickler-comparator": "auto",
                "x-aws-stickler-comparator-config": {"threshold": 0.9},
            }
        )
        assert _info(cfg).threshold == _info(sch).threshold == 0.9


class TestAnUnstatedThresholdKeepsEachPathsDefault:
    """Passing `None` must not flatten the importers' own defaults to 0.5."""

    def test_a_bare_config_field_is_the_legacy_default(self):
        model = _config({"type": "str", "comparator": "LevenshteinComparator"})
        assert _info(model).threshold == 0.5

    @pytest.mark.parametrize(
        "prop",
        [
            {"type": "string", "format": "date"},
            {"type": "string", "enum": ["a", "b"]},
        ],
    )
    def test_a_schema_type_default_survives(self, prop):
        assert _info(_schema(prop)).threshold == 1.0

    def test_an_inferred_threshold_survives(self):
        model = _config({"type": "str", "comparator": "auto"})
        assert _info(model).threshold == 0.7

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                lambda: _config({"type": "str", "comparator": "LevenshteinComparator"}),
                id="config",
            ),
            pytest.param(
                lambda: _config({"type": "str", "comparator": "auto"}),
                id="config-inferred",
            ),
            pytest.param(lambda: _schema({"type": "string"}), id="schema"),
            pytest.param(
                lambda: _schema({"type": "string", "format": "date"}),
                id="schema-type-default",
            ),
            pytest.param(
                lambda: _schema(
                    {"type": "string", "x-aws-stickler-comparator": "auto"}
                ),
                id="schema-inferred",
            ),
        ],
    )
    def test_none_of_them_is_marked_as_stated(self, model):
        assert _marker(model(), "_threshold_explicit") is False

    def test_a_stated_one_is(self):
        cfg = _config({"type": "str", "comparator": "auto", "threshold": 0.8})
        sch = _schema({"type": "string", "x-aws-stickler-threshold": 0.8})
        assert _marker(cfg, "_threshold_explicit") is True
        assert _marker(sch, "_threshold_explicit") is True


def _meta_paths(clip=None):
    """One mapping field naming ANLS*, optionally stating clip, on each path."""
    py_kwargs = {} if clip is None else {"clip_under_threshold": clip}
    cfg = {} if clip is None else {"clip_under_threshold": clip}
    sch = {} if clip is None else {"x-aws-stickler-clip-under-threshold": clip}
    return {
        "python": _python(
            Dict[str, Any],
            ComparableField(comparator=ANLSStarComparator(), **py_kwargs),
        ),
        "config": _config(
            {"type": "Dict[str, Any]", "comparator": "ANLSStarComparator", **cfg}
        ),
        "schema": _schema(
            {
                "type": "object",
                "additionalProperties": True,
                "x-aws-stickler-comparator": "ANLSStarComparator",
                **sch,
            }
        ),
    }


class TestAMappingKeepsItsPartialCreditOnEveryPath:
    def test_the_three_paths_turn_clipping_off(self):
        paths = _meta_paths()
        assert {_info(m).clip_under_threshold for m in paths.values()} == {False}

    def test_the_three_paths_report_the_same_score(self):
        scores = {
            name: _verdict(m, GT_META, PRED_META)[0]
            for name, m in _meta_paths().items()
        }
        assert scores["config"] == scores["python"]
        assert scores["schema"] == scores["python"]
        assert scores["python"] == pytest.approx(1 / 3)

    def test_an_explicit_clip_is_still_honoured_on_every_path(self):
        paths = _meta_paths(clip=True)
        assert {_info(m).clip_under_threshold for m in paths.values()} == {True}
        assert {_verdict(m, GT_META, PRED_META)[0] for m in paths.values()} == {0.0}

    def test_a_scalar_field_still_clips_by_default(self):
        """The default moved for object-grade fields only."""
        cfg = _config({"type": "str", "comparator": "LevenshteinComparator"})
        sch = _schema({"type": "string"})
        assert _info(cfg).clip_under_threshold is True
        assert _info(sch).clip_under_threshold is True
