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

Comparator keys on a nested model, a list of them, and an object property
(#348 item 7) are asserted against the Python path below, and the schema
spellings in ``test_object_property_threshold.TestTheComparatorOnTheSameNodeWarns``.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import pytest

import stickler
from stickler.comparators.anls import ANLSStarComparator
from stickler.comparators.base import BaseComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models import comparator_registry
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)
from stickler.utils import deprecation

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

    def test_one_a_custom_comparators_constructor_sets_reaches_it(self, strict95):
        """As in Python, over the importer's 0.5 and a schema type default."""
        enum = {"type": "string", "enum": ["a", "b"]}
        models = {
            "python": _python(str, ComparableField(comparator=_Strict95())),
            "config": _config({"type": "str", "comparator": strict95}),
            "schema": _schema({"type": "string", "x-aws-stickler-comparator": strict95}),
            "schema enum": _schema({**enum, "x-aws-stickler-comparator": strict95}),
        }
        assert {n: _info(m).threshold for n, m in models.items()} == dict.fromkeys(
            models, 0.95
        )

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


def _why(model):
    return stickler.eval_for(model).explain()["f"]


class TestTheTrailRecordsAnAdoptedComparatorThreshold:
    """`explain()` must not contradict its own row.

    A comparator threshold replaces the inferred one, so the trail names it the
    way it names a field `threshold`. It used to report 0.9 under a trail that
    quoted only the inferred `@0.7` / `@0.95`.
    """

    def test_on_the_config_path(self):
        entry = _why(
            _config(
                {
                    "type": "str",
                    "comparator": "auto",
                    "comparator_config": {"threshold": 0.9},
                }
            )
        )
        assert entry["threshold"] == 0.9
        assert entry["why"] == [
            "type:str -> LevenshteinComparator@0.7",
            "config: comparator_config.threshold=0.9 (overrides above)",
        ]

    def test_on_the_schema_path(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "x-aws-stickler-infer-unspecified": True,
                "properties": {
                    "f": {
                        "type": "number",
                        "x-aws-stickler-comparator-config": {"threshold": 0.9},
                    }
                },
            }
        )
        entry = _why(model)
        assert entry["threshold"] == 0.9
        assert entry["why"][-1] == (
            "schema: x-aws-stickler-comparator-config.threshold=0.9 (overrides above)"
        )

    def test_the_schema_path_names_a_field_threshold_too(self):
        """The same gap for `x-aws-stickler-threshold`, closed alongside."""
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "x-aws-stickler-infer-unspecified": True,
                "properties": {
                    "f": {"type": "number", "x-aws-stickler-threshold": 0.6}
                },
            }
        )
        assert _why(model)["why"][-1] == (
            "schema: x-aws-stickler-threshold=0.6 (overrides above)"
        )

    def test_a_field_threshold_that_wins_is_the_only_one_named(self):
        entry = _why(
            _config(
                {
                    "type": "str",
                    "comparator": "auto",
                    "comparator_config": {"threshold": 0.9},
                    "threshold": 0.6,
                }
            )
        )
        assert entry["why"][-1] == "config: threshold=0.6 (overrides above)"


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


class TestAListOfMappingsKeepsItsPartialCreditOnEveryPath:
    """The list form of the mapping case, which the CHANGELOG entry also names."""

    def test_the_three_paths_agree(self):
        paths = {
            "python": _python(
                List[Dict[str, Any]],
                ComparableField(comparator=ANLSStarComparator()),
            ),
            "config": _config(
                {"type": "List[Dict[str, Any]]", "comparator": "ANLSStarComparator"}
            ),
            "schema": _schema(
                {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": True},
                    "x-aws-stickler-comparator": "ANLSStarComparator",
                }
            ),
        }
        assert {_info(m).clip_under_threshold for m in paths.values()} == {False}
        scores = {
            name: _verdict(m, [GT_META], [PRED_META])[0] for name, m in paths.items()
        }
        assert scores["config"] == scores["schema"] == scores["python"]
        assert scores["python"] == pytest.approx(1 / 3)


class TestTheSchemaFreeFormObjectDefault:
    """Exact at 1.0 is unchanged (#348 item 1); only its clip reads false."""

    def test_clip_is_off_and_the_comparator_is_unchanged(self):
        info = _info(_schema({"type": "object", "additionalProperties": True}))
        assert type(info.comparator).__name__ == "ExactComparator"
        assert info.threshold == 1.0
        assert info.clip_under_threshold is False


class TestAZeroThresholdWarningNamesItsSource:
    """Clearing the marker on an importer default must not credit the comparator.

    "Not stated on the field" used to imply "adopted from the comparator", so
    an inferred mapping threshold copied from `match_threshold: 0.0` was
    reported as "sets comparator threshold=0.0", naming a key nobody wrote.
    """

    @pytest.fixture(autouse=True)
    def _fresh_warn_once(self):
        deprecation._warned.clear()
        yield
        deprecation._warned.clear()

    def _messages(self, build):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build()
        return [str(w.message) for w in caught if "=0.0" in str(w.message)]

    def test_an_inferred_zero_is_not_attributed_to_the_comparator(self):
        messages = self._messages(
            lambda: StructuredModel.model_from_json(
                {
                    "model_name": "ZeroInferred",
                    "match_threshold": 0.0,
                    "infer_unspecified_fields": True,
                    "fields": {"meta": {"type": "Dict[str, Any]"}},
                }
            )
        )
        assert any("ZeroInferred.meta sets threshold=0.0" in m for m in messages)
        assert not any("comparator threshold" in m for m in messages)

    def test_a_comparator_zero_still_is(self):
        messages = self._messages(
            lambda: _schema(
                {
                    "type": "string",
                    "x-aws-stickler-comparator-config": {"threshold": 0.0},
                }
            )
        )
        assert any(".f sets comparator threshold=0.0" in m for m in messages)


# A nested model whose two Exact leaves score a half-matching pair 0.5: a TP at
# 0.3 and an FD at 0.9 wherever the field's own threshold gates it.
class _Inner(StructuredModel):
    a: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
    b: str = ComparableField(comparator=ExactComparator(), threshold=1.0)


_LEAF = {"type": "str", "comparator": "ExactComparator", "threshold": 1.0}
_SLEAF = {
    "type": "string",
    "x-aws-stickler-comparator": "ExactComparator",
    "x-aws-stickler-threshold": 1.0,
}
GT_INNER, PRED_INNER = {"a": "x", "b": "y"}, {"a": "x", "b": "z"}


def _nested_verdict(model, *, is_list=False):
    gt, pred = ([GT_INNER], [PRED_INNER]) if is_list else (GT_INNER, PRED_INNER)
    result = model(f=gt).compare_with(model(f=pred), include_confusion_matrix=True)
    overall = result["confusion_matrix"]["fields"]["f"]["overall"]
    return overall["tp"], overall["fd"]


def _imported(build):
    """Build a model, returning it with the warnings it raised."""
    deprecation._warned.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = build()
    return model, [str(w.message) for w in caught if "no effect" in str(w.message)]


def _nested_paths(*, config_keys, schema_keys, python_field, shape="single"):
    """One nested-model field carrying comparator keys, on each path."""
    config_type = {
        "single": "structured_model",
        "optional": "optional_structured_model",
        "list": "list_structured_model",
    }[shape]
    child = {
        "type": "object",
        "properties": {"a": _SLEAF, "b": _SLEAF},
        "required": ["a", "b"],
    }
    if shape == "list":
        prop = {"type": "array", "items": child, **schema_keys}
    else:
        prop = {**child, **schema_keys}
        if shape == "optional":
            prop["type"] = ["object", "null"]
    annotation = List[_Inner] if shape == "list" else _Inner
    return {
        "python": _imported(lambda: _python(annotation, python_field())),
        "config": _imported(
            lambda: _config(
                {"type": config_type, "fields": {"a": _LEAF, "b": _LEAF}, **config_keys}
            )
        ),
        "schema": _imported(lambda: _schema(prop)),
    }


@pytest.mark.parametrize("shape", ["single", "optional"])
class TestANestedModelReadsItsComparatorThresholdOnEveryPath:
    """`ComparableField(comparator=X(threshold=t))` on a nested model gates at t.

    The config path ignored `comparator_config` there in silence, and the
    schema path raised on `x-aws-stickler-comparator-config` (dropped it on
    `dev`), so a threshold the Python path reads was read by neither importer.
    """

    @staticmethod
    def _paths(shape, threshold):
        return _nested_paths(
            config_keys={"comparator_config": {"threshold": threshold}},
            schema_keys={"x-aws-stickler-comparator-config": {"threshold": threshold}},
            python_field=lambda: ComparableField(
                comparator=LevenshteinComparator(threshold=threshold)
            ),
            shape=shape,
        )

    @pytest.mark.parametrize("threshold, verdict", [(0.3, (1, 0)), (0.9, (0, 1))])
    def test_the_three_paths_gate_at_it(self, shape, threshold, verdict):
        paths = self._paths(shape, threshold)
        assert {n: _info(m).threshold for n, (m, _) in paths.items()} == {
            "python": threshold,
            "config": threshold,
            "schema": threshold,
        }
        assert {n: _nested_verdict(m) for n, (m, _) in paths.items()} == {
            "python": verdict,
            "config": verdict,
            "schema": verdict,
        }

    def test_none_of_them_warns(self, shape):
        """Read, so there is nothing to warn about."""
        assert all(not msgs for _, msgs in self._paths(shape, 0.9).values())

    def test_a_field_threshold_still_wins(self, shape):
        paths = _nested_paths(
            config_keys={"comparator_config": {"threshold": 0.9}, "threshold": 0.3},
            schema_keys={
                "x-aws-stickler-comparator-config": {"threshold": 0.9},
                "x-aws-stickler-threshold": 0.3,
            },
            python_field=lambda: ComparableField(
                comparator=LevenshteinComparator(threshold=0.9), threshold=0.3
            ),
            shape=shape,
        )
        assert {_info(m).threshold for m, _ in paths.values()} == {0.3}


class TestANamedComparatorOnANestedModelWarnsOnBothImporters:
    """The comparator is never called; Python accepts and ignores it.

    The config path raised on `comparator` here and the schema path (on this
    branch) on `x-aws-stickler-comparator`. Both now warn, in one wording, and
    still read the threshold the comparator names.
    """

    @pytest.fixture
    def paths(self):
        return _nested_paths(
            config_keys={
                "comparator": "ExactComparator",
                "comparator_config": {"threshold": 0.9},
            },
            schema_keys={
                "x-aws-stickler-comparator": "ExactComparator",
                "x-aws-stickler-comparator-config": {"threshold": 0.9},
            },
            python_field=lambda: ComparableField(
                comparator=ExactComparator(threshold=0.9)
            ),
        )

    def test_the_threshold_it_names_is_read(self, paths):
        assert {_info(m).threshold for m, _ in paths.values()} == {0.9}
        assert {_nested_verdict(m) for m, _ in paths.values()} == {(0, 1)}

    def test_both_importers_warn_in_the_same_words(self, paths):
        """And name the threshold it carries, which removing it would drop."""
        expected = {
            "config": (
                "'comparator' has no effect on field 'f' of model 'Cfg' and is "
                "ignored, except for its threshold 0.9, which this field uses: "
                "it is a structured_model field, scored field by field, so its "
                "comparator is never called. Set comparators on the fields "
                "inside it, and set 'threshold' to 0.9 to keep that threshold "
                "without the comparator."
            ),
            "schema": (
                "'x-aws-stickler-comparator' has no effect on field 'f' of model "
                "'Sch' and is ignored, except for its "
                "threshold 0.9, which this field uses: it is an object with "
                "properties, scored field by field, so its comparator is never "
                "called. Set comparators on the properties inside it, and set "
                "'x-aws-stickler-threshold' to 0.9 to keep that threshold "
                "without the comparator."
            ),
        }
        assert {n: paths[n][1] for n in expected} == {
            n: [m] for n, m in expected.items()
        }

    def test_without_a_threshold_it_names_no_gate(self):
        paths = _nested_paths(
            config_keys={"comparator": "ExactComparator"},
            schema_keys={"x-aws-stickler-comparator": "ExactComparator"},
            python_field=lambda: ComparableField(comparator=ExactComparator()),
        )
        assert paths["config"][1] == [
            "'comparator' has no effect on field 'f' of model 'Cfg' and is "
            "ignored: it is a structured_model field, scored field by field, so "
            "its comparator is never called. Set comparators on the fields "
            "inside it, and this field's threshold with 'threshold'."
        ]
        assert _info(paths["config"][0]).threshold == 0.7
        assert {
            name: stickler.eval_for(model).explain()["f"]["comparator"]
            for name, (model, _) in paths.items()
        } == {name: "ExactComparator" for name in paths}

    def test_explain_reports_it_as_the_python_path_does(self, paths):
        """Not the placeholder: that is a comparator the author never wrote."""
        reported = {
            name: (
                type(_info(model).comparator).__name__,
                stickler.eval_for(model).explain()["f"]["comparator"],
            )
            for name, (model, _) in paths.items()
        }
        assert reported == {
            name: ("ExactComparator", "ExactComparator") for name in reported
        }

    def test_any_other_config_key_is_named(self):
        _, messages = _imported(
            lambda: _config(
                {
                    "type": "structured_model",
                    "comparator_config": {"threshold": 0.9, "case_sensitive": False},
                    "fields": {"a": _LEAF},
                }
            )
        )
        assert messages == [
            "'comparator_config' key 'case_sensitive' has no effect on field 'f' "
            "of model 'Cfg' and is ignored: it is a structured_model field, "
            "scored field by field, so its comparator is never called. Set "
            "comparators on the fields inside it, and this field's threshold "
            "with 'threshold'."
        ]

    def test_a_config_that_is_not_a_mapping_warns_rather_than_raising(self):
        """It loaded before, ignored; refusing it now would be a new raise."""
        model, messages = _imported(
            lambda: _config(
                {
                    "type": "structured_model",
                    "comparator_config": "strict",
                    "fields": {"a": _LEAF},
                }
            )
        )
        assert _info(model).threshold == 0.7
        assert messages and messages[0].startswith("'comparator_config' has no effect")

    def test_auto_is_still_refused(self):
        with pytest.raises(ValueError, match="no comparator to infer"):
            _config(
                {
                    "type": "structured_model",
                    "comparator": "auto",
                    "fields": {"a": _LEAF},
                }
            )


class TestAListOfModelsReadsNoComparatorConfig:
    """Hungarian matching pairs on the element class's `match_threshold`.

    Python warns that a comparator threshold on `List[StructuredModel]` is not
    consulted. The config path dropped `comparator_config` there in silence, and
    the schema path forwarded it, so `explain()` reported it as the explicit
    threshold. Both importers now warn, and the gate is unchanged.
    """

    @pytest.fixture
    def paths(self):
        return _nested_paths(
            config_keys={"comparator_config": {"threshold": 0.3}},
            schema_keys={"x-aws-stickler-comparator-config": {"threshold": 0.3}},
            python_field=lambda: ComparableField(
                comparator=LevenshteinComparator(threshold=0.3)
            ),
            shape="list",
        )

    def test_the_gate_is_the_element_match_threshold_on_every_path(self, paths):
        """0.5 is below the default 0.7, so FD, though the comparator says 0.3."""
        assert {n: _nested_verdict(m, is_list=True) for n, (m, _) in paths.items()} == {
            "python": (0, 1),
            "config": (0, 1),
            "schema": (0, 1),
        }

    def test_neither_importer_reports_it_as_the_threshold(self, paths):
        assert _info(paths["config"][0]).threshold == 0.5
        assert _info(paths["schema"][0]).threshold == 0.5

    def test_both_importers_warn_and_name_the_key_to_use(self, paths):
        assert paths["config"][1] == [
            "'comparator_config' key 'threshold' has no effect on field 'f' of "
            "model 'Cfg' and is ignored: it is a list_structured_model field, "
            "which matches its items using 'match_threshold' on this field and "
            "never calls a comparator. Set comparators on the fields inside each "
            "element, and how items match with 'match_threshold' on this field, "
            "a number from 0.0 to 1.0."
        ]
        assert paths["schema"][1] == [
            "'x-aws-stickler-comparator-config' key 'threshold' has no effect on "
            "field 'f' of model 'Sch' and is ignored: it is an array of "
            "objects, which matches its items using "
            "'x-aws-stickler-match-threshold' in its 'items' and never calls a "
            "comparator. Set comparators on the properties inside each element, "
            "and how items match with 'x-aws-stickler-match-threshold' in its "
            "'items', a number from 0.0 to 1.0."
        ]


def _recorded(build, action="always"):
    """Build under a fresh warn-once set, returning the warnings it raised."""
    deprecation._warned.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(action)
        model = build()
    return model, caught


class _Strict95(BaseComparator):
    """Sets its own threshold, as a domain comparator might."""

    def __init__(self):
        super().__init__(threshold=0.95)

    def _compare(self, a, b):
        return 1.0 if a == b else 0.0


@pytest.fixture
def strict95():
    name = "_Strict95ForParityTests"
    comparator_registry._global_registry._registry[name] = _Strict95
    yield name
    comparator_registry._global_registry._registry.pop(name, None)


class TestANestedModelAsksANamedComparatorForItsThreshold:
    """With or without a config, on both importers, as on the Python path.

    The config path asked only when `comparator_config` was present, so a
    comparator that sets its own threshold gated at 0.95 in Python and on the
    schema path, and at 0.7 from a config.
    """

    def test_the_three_paths_gate_at_it(self, strict95):
        paths = _nested_paths(
            config_keys={"comparator": strict95},
            schema_keys={"x-aws-stickler-comparator": strict95},
            python_field=lambda: ComparableField(comparator=_Strict95()),
        )
        # The child scores 0.5: a TP at 0.7, an FD at 0.95.
        assert {n: _info(m).threshold for n, (m, _) in paths.items()} == {
            "python": 0.95,
            "config": 0.95,
            "schema": 0.95,
        }
        assert {_nested_verdict(m) for m, _ in paths.values()} == {(0, 1)}
        assert "set 'threshold' to 0.95 to keep that threshold" in paths["config"][1][0]


class _Counting(BaseComparator):
    """Counts its calls, to show it is installed and never called."""

    calls = 0

    def __init__(self, threshold: Optional[float] = None):
        super().__init__(threshold=threshold)

    def _compare(self, a, b):
        type(self).calls += 1
        return 1.0 if a == b else 0.0


_UNION_KEYS = [f"k{i}" for i in range(20)]
_UNION_ARM = {
    "type": "object",
    "properties": {k: _SLEAF for k in _UNION_KEYS},
    "required": _UNION_KEYS,
}
_UNION_OTHER = {"type": "object", "properties": {"c": _SLEAF}, "required": ["c"]}


@pytest.mark.parametrize(
    "union",
    [
        {"anyOf": [_UNION_ARM, _UNION_OTHER]},
        {"oneOf": [_UNION_ARM, _UNION_OTHER]},
        {"anyOf": [_UNION_ARM, _UNION_OTHER, {"type": "null"}]},
    ],
    ids=["anyOf", "oneOf", "anyOf-null"],
)
class TestAUnionOfObjectsReadsItsKeysAsPythonDoes:
    """A union of objects is not a nested model, so neither path warns.

    A comparator-config threshold gates it, and a named comparator is installed
    and never called, as for a Python `Union` field.
    """

    @staticmethod
    def _verdict(model):
        """One of twenty fields differs: 0.95, a TP at 0.95, an FD at 1.0."""
        gt = {k: "x" for k in _UNION_KEYS}
        result = model(f=gt).compare_with(
            model(f={**gt, "k0": "y"}), include_confusion_matrix=True
        )
        overall = result["confusion_matrix"]["fields"]["f"]["overall"]
        return overall["tp"], overall["fd"]

    def test_a_comparator_config_threshold_gates_it(self, union):
        model, no_effect = _imported(
            lambda: _schema(
                {**union, "x-aws-stickler-comparator-config": {"threshold": 0.95}}
            )
        )
        assert no_effect == []
        assert _info(model).threshold == 0.95
        assert self._verdict(model) == (1, 0)

    def test_a_named_comparator_is_installed_and_never_called(self, union, monkeypatch):
        monkeypatch.setitem(
            comparator_registry._global_registry._registry,
            "_CountingForUnionTests",
            _Counting,
        )
        monkeypatch.setattr(_Counting, "calls", 0)
        model, no_effect = _imported(
            lambda: _schema(
                {**union, "x-aws-stickler-comparator": "_CountingForUnionTests"}
            )
        )
        assert no_effect == []
        assert isinstance(_info(model).comparator, _Counting)
        self._verdict(model)
        assert _Counting.calls == 0


def _python_arm(keys):
    return type(
        "Arm",
        (StructuredModel,),
        {
            "__annotations__": {k: str for k in keys},
            **{
                k: ComparableField(comparator=ExactComparator(), threshold=1.0)
                for k in keys
            },
        },
    )


def test_a_python_union_field_reads_its_comparator_the_same_way(monkeypatch):
    """The Python side of `TestAUnionOfObjectsReadsItsKeysAsPythonDoes`."""
    monkeypatch.setattr(_Counting, "calls", 0)
    union = Union[_python_arm(_UNION_KEYS), _python_arm(["c"])]
    model, no_effect = _imported(
        lambda: _python(union, ComparableField(comparator=_Counting(threshold=0.95)))
    )
    assert no_effect == []
    assert _info(model).threshold == 0.95
    assert TestAUnionOfObjectsReadsItsKeysAsPythonDoes._verdict(model) == (1, 0)
    assert _Counting.calls == 0


class TestAListOfModelsRebuildsItsComparatorWithTheRestOfItsConfig:
    """Only the config's `threshold` is dropped there.

    Rebuilt from `{}`, a comparator that requires another key raised a message
    naming no field and blaming an argument the author supplied, and any other
    exception escaped `from_json_schema`.
    """

    @staticmethod
    def _schema(**keys):
        return {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"a": _SLEAF}},
                    **keys,
                }
            },
        }

    def test_a_comparator_that_requires_a_key_still_builds(self, monkeypatch):
        class LevenshteinComparator(BaseComparator):
            def __init__(self, pattern, threshold: Optional[float] = None):
                super().__init__(threshold=threshold)
                self.pattern = pattern

            def _compare(self, a, b):
                return 1.0 if a == b else 0.0

        monkeypatch.setitem(
            comparator_registry._global_registry._registry,
            "LevenshteinComparator",
            LevenshteinComparator,
        )
        config = {"pattern": "x", "threshold": 0.9}
        for keys in (
            {"x-aws-stickler-comparator-config": config},
            {
                "x-aws-stickler-comparator": "LevenshteinComparator",
                "x-aws-stickler-comparator-config": config,
            },
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = StructuredModel.from_json_schema(self._schema(**keys))
            info = model._get_comparison_info("rows")
            assert info.comparator.pattern == "x"
            assert info.threshold == 0.5

    def test_a_comparator_that_requires_threshold_still_loads(self, monkeypatch):
        """It cannot be rebuilt without the one key that is dropped.

        It loaded before, so it is built from the full config, and its
        threshold still does not reach the field.
        """

        class LevenshteinComparator(BaseComparator):
            def __init__(self, threshold):
                super().__init__(threshold=threshold)

            def _compare(self, a, b):
                return 1.0 if a == b else 0.0

        monkeypatch.setitem(
            comparator_registry._global_registry._registry,
            "LevenshteinComparator",
            LevenshteinComparator,
        )
        config = {"threshold": 0.9}
        for keys in (
            {"x-aws-stickler-comparator-config": config},
            {
                "x-aws-stickler-comparator": "LevenshteinComparator",
                "x-aws-stickler-comparator-config": config,
            },
        ):
            model, no_effect = _imported(
                lambda: StructuredModel.from_json_schema(self._schema(**keys))
            )
            info = model._get_comparison_info("rows")
            assert isinstance(info.comparator, LevenshteinComparator)
            assert info.threshold == 0.5
            assert stickler.eval_for(model).explain()["rows"]["threshold"] == 0.5
            extra = model.model_fields["rows"].json_schema_extra
            assert extra._threshold_explicit is False
            assert any("key 'threshold' has no effect" in m for m in no_effect)

    def test_any_failure_is_a_value_error_naming_the_field(self, monkeypatch):
        class LevenshteinComparator(BaseComparator):
            def __init__(self, **config):
                raise RuntimeError("cannot build")

            def _compare(self, a, b):
                return 0.0

        monkeypatch.setitem(
            comparator_registry._global_registry._registry,
            "LevenshteinComparator",
            LevenshteinComparator,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="in field 'rows': cannot build"):
                StructuredModel.from_json_schema(
                    self._schema(
                        **{"x-aws-stickler-comparator-config": {"threshold": 0.9}}
                    )
                )


class TestNestedModelWarningsReachEveryModel:
    """Under Python's default filter, which shows a text once per line.

    The text named no model and the warning was attributed to library code, so
    a second model with the same field showed nothing.
    """

    def test_on_the_config_path(self):
        def build():
            for name in ("Invoice", "Receipt"):
                _config_named(name)

        def _config_named(name):
            StructuredModel.model_from_json(
                {
                    "model_name": name,
                    "fields": {
                        "addr": {
                            "type": "structured_model",
                            "comparator": "ExactComparator",
                            "fields": {"a": _LEAF},
                        }
                    },
                }
            )

        _, caught = _recorded(build, action="default")
        assert [("of model 'Invoice'" in str(w.message)) for w in caught] == [
            True,
            False,
        ]
        assert all(w.filename == __file__ for w in caught)

    def test_on_the_schema_path(self):
        def build():
            for name in ("Invoice", "Receipt"):
                StructuredModel.from_json_schema(
                    {
                        "type": "object",
                        "x-aws-stickler-model-name": name,
                        "properties": {
                            "address": {
                                "type": "object",
                                "x-aws-stickler-comparator": "ExactComparator",
                                "properties": {"a": _SLEAF},
                            }
                        },
                    }
                )

        _, caught = _recorded(build, action="default")
        assert len(caught) == 2
        assert all(w.filename == __file__ for w in caught)

    def test_two_levels_down_it_is_attributed_to_the_caller(self):
        _, caught = _recorded(
            lambda: _config(
                {
                    "type": "structured_model",
                    "fields": {
                        "addr": {
                            "type": "structured_model",
                            "comparator": "ExactComparator",
                            "fields": {"a": _LEAF},
                        }
                    },
                }
            )
        )
        assert [w.filename for w in caught] == [__file__]
        assert "field 'f.addr' of model 'Cfg'" in str(caught[0].message)

    def test_two_levels_down_on_the_schema_path_too(self):
        _, caught = _recorded(
            lambda: _schema(
                {
                    "type": "object",
                    "properties": {
                        "addr": {
                            "type": "object",
                            "x-aws-stickler-comparator": "ExactComparator",
                            "properties": {"a": _SLEAF},
                        }
                    },
                }
            )
        )
        assert [w.filename for w in caught] == [__file__]
        assert "field 'f.addr' of model 'Sch'" in str(
            caught[0].message
        )


class TestAnUnnamedModelIsNamedAsTheErrorsNameIt:
    """Not by the `(fields: ...)` identity that keys warn-once.

    An untitled schema's warning said `of model 'DynamicModel(fields:
    address)'`, a name the author never wrote, where its error says `of an
    unnamed model` and how to name it.
    """

    @staticmethod
    def _warning_and_error(build, keys, bad_keys):
        _, caught = _recorded(lambda: build(keys))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as error:
                build(bad_keys)
        (warning,) = [str(w.message) for w in caught]
        return warning, str(error.value)

    def test_on_the_schema_path(self):
        def build(keys):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "object",
                            "properties": {"a": _SLEAF},
                            **keys,
                        }
                    },
                }
            )

        warning, error = self._warning_and_error(
            build,
            {"x-aws-stickler-comparator": "ExactComparator"},
            {"x-aws-stickler-comparator-config": {"threshold": 85}},
        )
        named = (
            "field 'address' of an unnamed model (set title or "
            "x-aws-stickler-model-name to name it)"
        )
        assert named in warning
        assert named in error
        assert "DynamicModel" not in warning

    def test_on_the_config_path(self):
        def build(keys):
            StructuredModel.model_from_json(
                {
                    "fields": {
                        "address": {
                            "type": "structured_model",
                            "fields": {"a": _LEAF},
                            **keys,
                        }
                    }
                }
            )

        warning, error = self._warning_and_error(
            build,
            {"comparator": "ExactComparator"},
            {"comparator_config": {"threshold": 85}},
        )
        named = "field 'address' of an unnamed model (set model_name to name it)"
        assert named in warning
        assert named in error
        assert "DynamicModel" not in warning

    def test_a_title_names_it_in_both(self):
        def build(keys):
            StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "Invoice",
                    "properties": {
                        "address": {
                            "type": "object",
                            "properties": {"a": _SLEAF},
                            **keys,
                        }
                    },
                }
            )

        warning, error = self._warning_and_error(
            build,
            {"x-aws-stickler-comparator": "ExactComparator"},
            {"x-aws-stickler-comparator-config": {"threshold": 85}},
        )
        assert "field 'address' of model 'Invoice'" in warning
        assert "field 'address' of model 'Invoice'" in error
