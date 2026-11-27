"""A field rebuilt from its exported ``x-comparison`` dict scores as the original.

``ConfigurationHelper`` falls back to the ``x-comparison`` dict when a field
carries no ``ComparableField`` runtime data, as a hand-built
``Field(json_schema_extra={"x-comparison": ...})`` does. It used to rebuild the
comparator from a private list of six names, so every other one, including
comparators stickler itself writes there, silently became
``LevenshteinComparator`` (#432). It now resolves through the global
``ComparatorRegistry``.
"""

import warnings
from typing import Any, Dict, List

import pytest
from pydantic import Field

from stickler import ComparableField, StructuredModel
from stickler.comparators import (
    BBoxIoUComparator,
    DateComparator,
    ExactComparator,
    FuzzyComparator,
    LevenshteinComparator,
    NormalizedComparator,
    NumericComparator,
    PhoneComparator,
)
from stickler.comparators.anls import ANLSStarComparator
from stickler.comparators.base import BaseComparator
from stickler.comparators.semantic import SemanticComparator
from stickler.structured_object_evaluator.models.comparator_registry import (
    get_global_registry,
)


def _exported(model: type, field: str = "d") -> Dict[str, Any]:
    schema: Dict[str, Any] = {}
    model.model_fields[field].json_schema_extra(schema)
    return schema["x-comparison"]


def _model(name: str, annotation: Any, field: Any) -> type:
    return type(
        name, (StructuredModel,), {"__annotations__": {"d": annotation}, "d": field}
    )


def _rebuild(original: type, annotation: Any) -> type:
    """The original's field as a plain pydantic Field carrying only x-comparison."""
    return _model(
        "Rebuilt",
        annotation,
        Field(default=None, json_schema_extra={"x-comparison": _exported(original)}),
    )


def _score(model: type, gt: Any, pred: Any) -> float:
    return model(d=gt).compare_with(model(d=pred))["overall_score"]


GT_DICT = {"vendor": "Acme Corporation", "city": "Springfield"}
PRED_DICT = {"vendor": "Acme Corp", "city": "Shelbyville"}

# (id, annotation, ComparableField kwargs, gt, pred, original score)
# Configs are non-default where the comparator takes one, so a rebuild that
# dropped comparator_config would also move the score.
CASES = [
    (
        "levenshtein",
        str,
        {"comparator": LevenshteinComparator(normalize=False)},
        "Hello",
        "hello ",
        None,
    ),
    (
        "exact",
        str,
        {"comparator": ExactComparator(case_sensitive=False)},
        "ABC",
        "abc",
        1.0,
    ),
    (
        "normalized",
        str,
        {"comparator": NormalizedComparator(case_sensitive=True)},
        "Abc",
        "abc",
        0.0,
    ),
    (
        "phone",
        str,
        {"comparator": PhoneComparator(region="GB")},
        "020 7946 0018",
        "+44 20 7946 0018",
        1.0,
    ),
    (
        "numeric",
        float,
        {"comparator": NumericComparator(relative_tolerance=0.01)},
        100.0,
        100.5,
        1.0,
    ),
    ("date", str, {"comparator": DateComparator()}, "2026-10-20", "Oct 20, 2026", 1.0),
    (
        "date-dayfirst",
        str,
        {"comparator": DateComparator(dayfirst=True)},
        "03/04/2026",
        "2026-04-03",
        1.0,
    ),
    (
        "fuzzy",
        str,
        {"comparator": FuzzyComparator(method="token_sort_ratio")},
        "beta alpha",
        "alpha beta",
        1.0,
    ),
    ("anls-dict", Dict[str, Any], {}, GT_DICT, PRED_DICT, None),
    ("anls-list-dict", List[Dict[str, Any]], {}, [GT_DICT], [PRED_DICT], None),
    # The issue's headline: two dicts that share nothing scored 0.80 rebuilt.
    (
        "anls-list-dict-disjoint",
        List[Dict[str, Any]],
        {},
        [{"a": "x"}],
        [{"b": "y"}],
        0.0,
    ),
    (
        "anls-leaf-threshold",
        Dict[str, Any],
        {"comparator": ANLSStarComparator(leaf_threshold=0.9)},
        GT_DICT,
        PRED_DICT,
        None,
    ),
    (
        "bbox",
        List[List[int]],
        {"comparator": BBoxIoUComparator(), "threshold": 0.4},
        [[0, 0, 10, 10]],
        [[0, 0, 10, 5]],
        0.5,
    ),
]


@pytest.mark.parametrize(
    "annotation, field_kwargs, gt, pred, expected",
    [pytest.param(*case[1:], id=case[0]) for case in CASES],
)
def test_rebuilt_field_keeps_comparator_and_score(
    annotation, field_kwargs, gt, pred, expected
):
    original = _model("Original", annotation, ComparableField(**field_kwargs))
    rebuilt = _rebuild(original, annotation)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a silent swap must not become a noisy one
        rebuilt_info = rebuilt._get_comparison_info("d")
    original_info = original._get_comparison_info("d")

    assert type(rebuilt_info.comparator) is type(original_info.comparator)
    assert getattr(rebuilt_info.comparator, "config", None) == getattr(
        original_info.comparator, "config", None
    )
    original_score = _score(original, gt, pred)
    if expected is not None:
        assert original_score == pytest.approx(expected)
    assert _score(rebuilt, gt, pred) == pytest.approx(original_score)


def test_semantic_comparator_is_rebuilt_by_name():
    """Scoring needs Bedrock, so only the rebuilt type is checked."""
    original = _model(
        "Original", str, ComparableField(comparator=SemanticComparator(), threshold=0.8)
    )
    rebuilt = _rebuild(original, str)
    assert isinstance(rebuilt._get_comparison_info("d").comparator, SemanticComparator)


# Built-ins with their own test below rather than a CASES row, and built-ins
# deliberately not exercised: StructuredModelComparator is installed for nested
# models, which never reach this fallback, and BERT downloads a model.
TESTED_SEPARATELY = {"SemanticComparator", "LLMComparator"}
EXEMPT = {"StructuredModelComparator", "BERTComparator"}


def test_every_builtin_has_a_rebuild_case():
    """Drift guard: a built-in added to the registry needs a CASES row.

    Read from what each case's field actually exports, so adding a name here
    without a case cannot satisfy it.
    """
    in_cases = {
        _exported(_model("Original", case[1], ComparableField(**case[2])))[
            "comparator_type"
        ]
        for case in CASES
    }
    assert in_cases.isdisjoint(TESTED_SEPARATELY | EXEMPT)
    builtins = set(get_global_registry()._BUILTINS)
    assert builtins == in_cases | TESTED_SEPARATELY | EXEMPT


def test_registered_custom_comparator_is_rebuilt_with_its_config():
    class _Halving(BaseComparator):
        def __init__(self, factor: float = 1.0, threshold=None):
            super().__init__(threshold=threshold)
            self.factor = factor

        @property
        def config(self):
            return {"factor": self.factor}

        def _compare(self, a, b):
            return self.factor if a == b else 0.0

    registry = get_global_registry()
    registry.register("_Halving432", _Halving)
    try:
        model = _model(
            "M",
            str,
            Field(
                default=None,
                json_schema_extra={
                    "x-comparison": {
                        "comparator_type": "_Halving432",
                        "comparator_config": {"factor": 0.5},
                        "threshold": 0.1,
                    }
                },
            ),
        )
        comparator = model._get_comparison_info("d").comparator
        assert isinstance(comparator, _Halving) and comparator.factor == 0.5
    finally:
        registry._registry.pop("_Halving432", None)


def _hand_built(comparison: Dict[str, Any]) -> type:
    return _model(
        "HandBuilt",
        str,
        Field(default=None, json_schema_extra={"x-comparison": comparison}),
    )


def test_unknown_name_warns_and_falls_back_to_levenshtein():
    model = _hand_built({"comparator_type": "NoSuchComparator432"})
    with pytest.warns(UserWarning, match=r"HandBuilt\.d.*'NoSuchComparator432'"):
        comparator = model._get_comparison_info("d").comparator
    assert isinstance(comparator, LevenshteinComparator)


def test_fallback_warning_points_at_the_callers_code():
    model = _hand_built({"comparator_type": "NoSuchComparator432Location"})
    with pytest.warns(UserWarning, match="NoSuchComparator432Location") as record:
        model(d="a").compare_with(model(d="b"))
    assert [w.filename for w in record if "x-comparison" in str(w.message)] == [
        __file__
    ]


def test_same_named_dynamic_models_each_warn():
    """Every dynamic model is `DynamicModel`; a name-only key warned once."""

    def dynamic(extra_field: str) -> type:
        return type(
            "DynamicModel",
            (StructuredModel,),
            {
                "__annotations__": {"d": str, extra_field: str},
                "d": Field(
                    default=None,
                    json_schema_extra={
                        "x-comparison": {"comparator_type": "NoSuchComparator432Dyn"}
                    },
                ),
                extra_field: None,
            },
        )

    for extra_field in ("first", "second"):
        model = dynamic(extra_field)
        with pytest.warns(UserWarning, match=rf"fields: .*{extra_field}"):
            model._get_comparison_info("d")


def test_comparator_that_cannot_be_built_from_config_warns():
    """LLMComparator needs a model; its x-comparison carries none."""
    if not get_global_registry().is_registered("LLMComparator"):
        pytest.skip("llm extra not installed")
    model = _hand_built({"comparator_type": "LLMComparator"})
    with pytest.warns(UserWarning, match=r"'LLMComparator'.*could not be built"):
        comparator = model._get_comparison_info("d").comparator
    assert isinstance(comparator, LevenshteinComparator)


def test_null_comparator_config_is_treated_as_empty():
    model = _hand_built(
        {"comparator_type": "DateComparator", "comparator_config": None}
    )
    assert isinstance(model._get_comparison_info("d").comparator, DateComparator)


class TestComparatorExplicitSurvivesTheRebuild:
    """An inferred comparator must not come back marked as declared."""

    @pytest.mark.parametrize("annotation", [Any, List[Any], str])
    def test_inferred_stays_inferred(self, annotation):
        original = _model("Original", annotation, ComparableField())
        rebuilt = _rebuild(original, annotation)
        assert original._get_comparison_info("d").comparator_explicit is False
        assert rebuilt._get_comparison_info("d").comparator_explicit is False

    def test_declared_stays_declared(self):
        original = _model("Original", str, ComparableField(comparator=DateComparator()))
        rebuilt = _rebuild(original, str)
        assert rebuilt._get_comparison_info("d").comparator_explicit is True

    def test_object_grade_substitution_matches_runtime(self):
        original = _model("Original", Dict[str, Any], ComparableField())
        rebuilt = _rebuild(original, Dict[str, Any])
        assert (
            rebuilt._get_comparison_info("d").comparator_explicit
            is original._get_comparison_info("d").comparator_explicit
        )

    def test_hand_written_dict_naming_a_comparator_is_declared(self):
        model = _hand_built({"comparator_type": "ExactComparator"})
        assert model._get_comparison_info("d").comparator_explicit is True
