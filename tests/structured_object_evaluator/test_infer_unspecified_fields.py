"""A config-driven model can have its unspecified fields inferred.

A large extraction schema has a handful of fields whose comparison rules matter and
a long tail where anything sensible will do. Before this, the config-driven paths
gave three different answers to "no comparator named", none of them what
`stickler.evaluate()` would choose for the same field:

    model_from_json()      raises
    from_json_schema()     shallow type defaults, e.g. number -> Numeric @ 0.5
    stickler.evaluate()    Numeric @ 0.95

`infer_unspecified_fields` routes the unspecified parameters through the same
inference `stickler.evaluate()` uses, so all the entry points converge.

OFF BY DEFAULT. Enabling it changes reported metrics for any field that named no
comparator -- a `float` previously compared as text starts being compared as a
number -- so nothing moves until it is asked for.

See https://github.com/awslabs/stickler/issues/239
"""

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

import stickler
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)

CONFIG_FIELDS = {
    "invoice_id": {"type": "str", "comparator": "ExactComparator", "threshold": 1.0},
    "total": {"type": "float"},
    "paid": {"type": "bool"},
    "vendor": {"type": "str"},
}

SCHEMA_PROPERTIES = {
    "invoice_id": {"type": "string", "x-aws-stickler-comparator": "ExactComparator"},
    "total": {"type": "number"},
    "paid": {"type": "boolean"},
    "issued": {"type": "string", "format": "date"},
    "vendor": {"type": "string"},
}


def _config(**top_level):
    return dict({"model_name": "Invoice", "fields": CONFIG_FIELDS}, **top_level)


def _schema(**model_extensions):
    return dict(
        {
            "type": "object",
            "x-aws-stickler-model-name": "Invoice",
            "properties": SCHEMA_PROPERTIES,
        },
        **model_extensions,
    )


def _resolved(model, field):
    info = model._get_comparison_info(field)
    return type(info.comparator).__name__, info.threshold


class TestNothingChangesWithoutTheFlag:
    """The guarantee that makes this safe to ship: silence by default."""

    def test_model_from_json_still_refuses_an_unspecified_field(self):
        with pytest.raises(ValueError, match="requires a 'comparator'"):
            StructuredModel.model_from_json(_config())

    def test_the_refusal_names_both_ways_to_opt_in(self):
        """An error that does not say what to do instead is only half an error."""
        with pytest.raises(ValueError) as caught:
            StructuredModel.model_from_json(_config())
        message = str(caught.value)
        assert "infer_unspecified_fields" in message
        assert "'comparator': 'auto'" in message

    def test_from_json_schema_keeps_its_shallow_defaults(self):
        model = StructuredModel.from_json_schema(_schema())
        assert _resolved(model, "total") == ("NumericComparator", 0.5)
        assert _resolved(model, "paid") == ("ExactComparator", 0.5)
        assert _resolved(model, "vendor") == ("LevenshteinComparator", 0.5)
        assert _resolved(model, "issued") == ("DateComparator", 1.0)


class TestTheFlagInfersBothComparatorAndThreshold:
    """Half-inferring would leave the paths still disagreeing on thresholds."""

    @pytest.mark.parametrize(
        "field,expected",
        (
            ("total", ("NumericComparator", 0.95)),
            ("paid", ("ExactComparator", 1.0)),
            ("vendor", ("LevenshteinComparator", 0.85)),
        ),
    )
    def test_the_config_path(self, field, expected):
        model = StructuredModel.model_from_json(_config(infer_unspecified_fields=True))
        assert _resolved(model, field) == expected

    @pytest.mark.parametrize(
        "field,expected",
        (
            ("total", ("NumericComparator", 0.95)),
            ("paid", ("ExactComparator", 1.0)),
            ("issued", ("DateComparator", 0.95)),
            ("vendor", ("LevenshteinComparator", 0.85)),
        ),
    )
    def test_the_schema_path(self, field, expected):
        model = StructuredModel.from_json_schema(
            _schema(**{"x-aws-stickler-infer-unspecified": True})
        )
        assert _resolved(model, field) == expected

    def test_it_agrees_with_stickler_evaluate(self):
        """The convergence this feature exists for, asserted rather than assumed.

        Comparing against `auto`'s own answer rather than against literals, so the
        two cannot drift apart if inference is retuned later.
        """

        class Plain(BaseModel):
            total: Optional[float] = None
            paid: Optional[bool] = None
            vendor: Optional[str] = None

        auto = stickler.eval_for(Plain).explain()
        model = StructuredModel.model_from_json(_config(infer_unspecified_fields=True))
        config = stickler.eval_for(model).explain()

        for field in ("total", "paid", "vendor"):
            assert (
                config[field]["comparator"],
                config[field]["threshold"],
            ) == (auto[field]["comparator"], auto[field]["threshold"]), field


class TestExplicitConfigAlwaysWins:
    def test_a_named_comparator_survives_the_flag(self):
        model = StructuredModel.model_from_json(_config(infer_unspecified_fields=True))
        assert _resolved(model, "invoice_id") == ("ExactComparator", 1.0)

    def test_a_field_can_pin_itself_against_the_flag(self):
        """Naming a comparator opts one field OUT of an inferred model."""
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {
                    "total": {"type": "float", "comparator": "LevenshteinComparator"}
                },
            }
        )
        assert _resolved(model, "total") == ("LevenshteinComparator", 0.5)


class TestTheAutoSentinelOptsInOneField:
    """Precedence runs both ways, which a model-level flag alone cannot express."""

    def test_it_works_with_no_model_flag(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "fields": {
                    "invoice_id": {"type": "str", "comparator": "ExactComparator"},
                    "total": {"type": "float", "comparator": "auto"},
                },
            }
        )
        assert _resolved(model, "total") == ("NumericComparator", 0.95)
        assert _resolved(model, "invoice_id") == ("ExactComparator", 0.5)

    def test_the_schema_path_too(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "properties": {
                    "total": {"type": "number", "x-aws-stickler-comparator": "auto"}
                },
            }
        )
        assert _resolved(model, "total") == ("NumericComparator", 0.95)

    def test_it_is_not_reported_as_an_unknown_comparator(self):
        """`auto` is a request, not a name, so the registry must not see it.

        It was resolved through `create_comparator` first and rejected with a list
        of the built-ins, which reads as a typo rather than a feature.
        """
        try:
            StructuredModel.model_from_json(
                {
                    "model_name": "M",
                    "fields": {"total": {"type": "float", "comparator": "auto"}},
                }
            )
        except ValueError as exc:  # pragma: no cover - guards the regression
            pytest.fail(f"'auto' rejected as a comparator name: {exc}")


class TestPartialConfigIsFilledPerParameter:
    """A field that names one parameter is not thereby "fully configured".

    Treating it that way left `{"type": "float", "threshold": 0.99}` on the
    type-blind Levenshtein default: a threshold tuned against edit distance over
    `"1000.00"` and `"1000.0"`, which is the sharpest form of the original problem.
    """

    def test_a_declared_threshold_keeps_an_inferred_comparator(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {"total": {"type": "float", "threshold": 0.99}},
            }
        )
        assert _resolved(model, "total") == ("NumericComparator", 0.99)

    def test_the_schema_path_too(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "x-aws-stickler-infer-unspecified": True,
                "properties": {
                    "total": {"type": "number", "x-aws-stickler-threshold": 0.99}
                },
            }
        )
        assert _resolved(model, "total") == ("NumericComparator", 0.99)

    def test_a_declared_weight_does_not_suppress_inference(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {"total": {"type": "float", "weight": 3.0}},
            }
        )
        info = model._get_comparison_info("total")
        assert type(info.comparator).__name__ == "NumericComparator"
        assert info.weight == 3.0


class TestInferenceIsAuditable:
    """Inference nobody can inspect is indistinguishable from a wrong default."""

    @staticmethod
    def _explained():
        model = StructuredModel.model_from_json(_config(infer_unspecified_fields=True))
        return stickler.eval_for(model).explain()

    def test_a_configured_field_reports_explicit(self):
        assert self._explained()["invoice_id"]["source"] == "explicit"

    def test_an_inferred_field_does_not(self):
        """It used to claim `explicit` for a value the author never wrote."""
        assert self._explained()["total"]["source"] != "explicit"

    def test_the_reason_is_retrievable(self):
        why = self._explained()["total"]["why"]
        assert any("type:float" in entry for entry in why)

    def test_a_configured_field_carries_no_inference_trail(self):
        why = self._explained()["invoice_id"]["why"]
        assert why == ["explicit: configured on the StructuredModel class"]


class TestNestedModelsInheritTheFlag:
    def test_the_schema_path_descends(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "x-aws-stickler-infer-unspecified": True,
                "properties": {
                    "inner": {
                        "type": "object",
                        "properties": {"amount": {"type": "number"}},
                    }
                },
            }
        )
        inner = model.model_fields["inner"].annotation
        inner = getattr(inner, "__args__", (inner,))[0]
        assert _resolved(inner, "amount") == ("NumericComparator", 0.95)

    def test_the_config_path_descends(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {
                    "inner": {
                        "type": "structured_model",
                        "fields": {"amount": {"type": "float"}},
                    }
                },
            }
        )
        inner = model.model_fields["inner"].annotation
        inner = getattr(inner, "__args__", (inner,))[0]
        assert _resolved(inner, "amount") == ("NumericComparator", 0.95)


class TestTheFlagIsValidated:
    @pytest.mark.parametrize("value", ("yes", 1, None, []))
    def test_a_non_boolean_is_refused_on_the_config_path(self, value):
        with pytest.raises(ValueError, match="must be true or false"):
            StructuredModel.model_from_json(
                {
                    "model_name": "M",
                    "infer_unspecified_fields": value,
                    "fields": {"a": {"type": "str", "comparator": "ExactComparator"}},
                }
            )

    @pytest.mark.parametrize("value", ("yes", 1, []))
    def test_a_non_boolean_is_refused_on_the_schema_path(self, value):
        with pytest.raises(ValueError, match="must be true or false"):
            StructuredModel.from_json_schema(
                _schema(**{"x-aws-stickler-infer-unspecified": value})
            )

    def test_the_schema_key_is_not_reported_as_unrecognized(self):
        """#312 rejects unknown `x-aws-stickler-*` keys; this one must be known."""
        model = StructuredModel.from_json_schema(
            _schema(**{"x-aws-stickler-infer-unspecified": True})
        )
        assert "total" in model.model_fields


class TestAnIncompatibleNameTokenIsNotForced:
    """`model_from_json` has no date type, and inference will not fake one.

    A field named `issued_date` declared as `str` matches the DateComparator name
    token, but a str-typed field cannot support date semantics, so the type default
    stands and `why` records the refusal. Documented, because "a date needs date
    semantics" is one of the cases this feature was requested for.
    """

    @staticmethod
    def _issued_date():
        return StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {"issued_date": {"type": "str"}},
            }
        )

    def test_a_str_named_like_a_date_keeps_the_str_default(self):
        assert _resolved(self._issued_date(), "issued_date") == (
            "LevenshteinComparator",
            0.7,
        )

    def test_the_refusal_is_recorded(self):
        why = stickler.eval_for(self._issued_date()).explain()["issued_date"]["why"]
        assert any("incompatible" in entry for entry in why)

    def test_the_refused_token_is_not_reported_as_the_source(self):
        """`source` must name what produced the comparator, not what matched.

        The comparator here is the plain `str` type default; the name token was
        refused. Reporting `name-token` said the name had been honoured in exactly
        the case where it was not, on this feature's own headline example.
        """
        row = stickler.eval_for(self._issued_date()).explain()["issued_date"]
        assert row["source"] == "type"


class TestComparatorConfigIsMergedNotReplaced:
    """A `comparator_config` beside an inferred comparator is a tweak, not a swap.

    Replacing the inferred dict wholesale dropped the rest of it. Inference gives
    `NumericComparator` a `relative_tolerance` of 0.001; setting any other key
    silently took that to 0.0, turning a tolerance-based numeric comparison into an
    exact one. The author gets a stricter comparison than either they or inference
    asked for.
    """

    @staticmethod
    def _numeric(**field_extra):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {"total": dict({"type": "float"}, **field_extra)},
            }
        )
        return model._get_comparison_info("total").comparator

    def test_the_inferred_config_is_the_baseline(self):
        assert self._numeric().relative_tolerance == pytest.approx(0.001)

    def test_an_unrelated_key_does_not_erase_it(self):
        comparator = self._numeric(comparator_config={"absolute_tolerance": 0.5})
        assert comparator.relative_tolerance == pytest.approx(0.001)
        assert comparator.absolute_tolerance == pytest.approx(0.5)

    def test_the_author_can_still_override_a_key(self):
        assert self._numeric(
            comparator_config={"relative_tolerance": 0.02}
        ).relative_tolerance == pytest.approx(0.02)

    def test_a_named_comparator_keeps_its_own_config_alone(self):
        """No merging when the caller chose the comparator: nothing to merge with."""
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {
                    "total": {
                        "type": "float",
                        "comparator": "NumericComparator",
                        "comparator_config": {"absolute_tolerance": 0.5},
                    }
                },
            }
        )
        comparator = model._get_comparison_info("total").comparator
        assert comparator.absolute_tolerance == pytest.approx(0.5)
        assert comparator.relative_tolerance == pytest.approx(0.0)


class TestTheSchemaPathMergesComparatorConfigToo:
    """The same merge, on the other entry point, from the same input.

    `x-aws-stickler-comparator-config` was read only inside the branch that also
    resolved a NAMED comparator, so for an inferred field it was read into a local
    and dropped. `absolute_tolerance: 0.5` beside `"auto"` built
    rel=0.001/abs=0.0 while the identical Stickler config built abs=0.5 -- one
    input, two comparators, decided by which document format the author chose.
    """

    @staticmethod
    def _numeric(property_extra, **model_extra):
        model = StructuredModel.from_json_schema(
            dict(
                {
                    "type": "object",
                    "title": "T",
                    "properties": {
                        "total": dict({"type": "number"}, **property_extra)
                    },
                },
                **model_extra,
            )
        )
        return model._get_comparison_info("total").comparator

    def test_auto_merges_the_authors_key_over_the_inferred_one(self):
        comparator = self._numeric(
            {
                "x-aws-stickler-comparator": "auto",
                "x-aws-stickler-comparator-config": {"absolute_tolerance": 0.5},
            }
        )
        assert comparator.relative_tolerance == pytest.approx(0.001)
        assert comparator.absolute_tolerance == pytest.approx(0.5)

    def test_the_model_flag_merges_a_bare_config(self):
        comparator = self._numeric(
            {"x-aws-stickler-comparator-config": {"absolute_tolerance": 0.5}},
            **{"x-aws-stickler-infer-unspecified": True},
        )
        assert comparator.relative_tolerance == pytest.approx(0.001)
        assert comparator.absolute_tolerance == pytest.approx(0.5)

    def test_the_author_can_still_override_an_inferred_key(self):
        comparator = self._numeric(
            {
                "x-aws-stickler-comparator": "auto",
                "x-aws-stickler-comparator-config": {"relative_tolerance": 0.02},
            }
        )
        assert comparator.relative_tolerance == pytest.approx(0.02)

    def test_a_bare_config_reaches_the_type_default_without_the_flag(self):
        """Nothing inferred, so nothing to merge -- but still not dropped.

        The Stickler-config path applies a bare `comparator_config` to the
        comparator it falls back to, so dropping it here was the same silent drop
        one branch over.
        """
        comparator = self._numeric(
            {"x-aws-stickler-comparator-config": {"absolute_tolerance": 0.5}}
        )
        assert comparator.absolute_tolerance == pytest.approx(0.5)
        assert comparator.relative_tolerance == pytest.approx(0.0)

    def test_the_two_paths_build_the_same_comparator(self):
        """The convergence, asserted directly rather than via two literal lists."""
        from_schema = self._numeric(
            {
                "x-aws-stickler-comparator": "auto",
                "x-aws-stickler-comparator-config": {"absolute_tolerance": 0.5},
            }
        )
        from_config = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "fields": {
                    "total": {
                        "type": "float",
                        "comparator": "auto",
                        "comparator_config": {"absolute_tolerance": 0.5},
                    }
                },
            }
        )._get_comparison_info("total")
        assert (
            type(from_schema).__name__,
            from_schema.relative_tolerance,
            from_schema.absolute_tolerance,
        ) == (
            type(from_config.comparator).__name__,
            from_config.comparator.relative_tolerance,
            from_config.comparator.absolute_tolerance,
        )


class TestAutoIsRefusedOnAContainer:
    """A container has no comparator, so `"auto"` there cannot be honoured.

    An object is scored recursively and an array of objects by Hungarian matching,
    so inference has nothing to choose. Before this the key was accepted and did
    nothing: the field kept `LevenshteinComparator@0.7` / `@0.5` while the schema
    said it had been inferred. Any OTHER unusable value at the same position
    already raised, so adding `"auto"` turned a loud error into a silent drop --
    the failure mode #210 and #312 exist to remove.
    """

    @staticmethod
    def _object_property(**extra):
        return {
            "type": "object",
            "title": "T",
            "properties": {
                "inner": dict(
                    {
                        "type": "object",
                        "properties": {"amount": {"type": "number"}},
                    },
                    **extra,
                )
            },
        }

    @staticmethod
    def _array_property(**extra):
        return {
            "type": "object",
            "title": "T",
            "properties": {
                "rows": dict(
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"amount": {"type": "number"}},
                        },
                    },
                    **extra,
                )
            },
        }

    def test_an_object_property_raises(self):
        with pytest.raises(ValueError, match="cannot be applied to field 'inner'"):
            StructuredModel.from_json_schema(
                self._object_property(**{"x-aws-stickler-comparator": "auto"})
            )

    def test_an_array_of_objects_raises(self):
        with pytest.raises(ValueError, match="cannot be applied to field 'rows'"):
            StructuredModel.from_json_schema(
                self._array_property(**{"x-aws-stickler-comparator": "auto"})
            )

    def test_the_error_names_the_key_that_does_work_there(self):
        """An error that does not say what to do instead is only half an error."""
        with pytest.raises(ValueError) as caught:
            StructuredModel.from_json_schema(
                self._object_property(**{"x-aws-stickler-comparator": "auto"})
            )
        assert "x-aws-stickler-infer-unspecified" in str(caught.value)

    def test_a_scalar_property_is_untouched(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "properties": {
                    "total": {"type": "number", "x-aws-stickler-comparator": "auto"}
                },
            }
        )
        assert _resolved(model, "total") == ("NumericComparator", 0.95)

    def test_a_primitive_array_is_untouched(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "properties": {
                    "amounts": {
                        "type": "array",
                        "items": {"type": "number"},
                        "x-aws-stickler-comparator": "auto",
                    }
                },
            }
        )
        assert _resolved(model, "amounts") == ("NumericComparator", 0.95)

    def test_the_object_subtree_key_still_works(self):
        """The remedy the error recommends has to actually work."""
        model = StructuredModel.from_json_schema(
            self._object_property(**{"x-aws-stickler-infer-unspecified": True})
        )
        annotation = model.model_fields["inner"].annotation
        inner = getattr(annotation, "__args__", (annotation,))[0]
        assert _resolved(inner, "amount") == ("NumericComparator", 0.95)


class TestMatchThresholdReachesInference:
    """A mapping field takes the OBJECT's threshold, so the object must supply it.

    A dict declines to name its keys, so inference scores it structurally with
    ANLS* and uses `match_threshold` as the field threshold rather than the scalar
    default. Neither config path forwarded it, so `{"type": "dict"}` sat at 0.7
    while `stickler.eval_for(cls, match_threshold=0.9)` gave 0.9 for the same
    field -- the entry-point disagreement this feature exists to end.
    """

    def test_the_config_path_forwards_it(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "match_threshold": 0.9,
                "infer_unspecified_fields": True,
                "fields": {"meta": {"type": "dict"}},
            }
        )
        assert _resolved(model, "meta") == ("ANLSStarComparator", 0.9)

    def test_it_agrees_with_stickler_eval_for(self):
        class Plain(BaseModel):
            meta: Optional[Dict[str, Any]] = None

        auto = stickler.eval_for(Plain, match_threshold=0.9).explain()["meta"]
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "match_threshold": 0.9,
                "infer_unspecified_fields": True,
                "fields": {"meta": {"type": "dict"}},
            }
        )
        assert _resolved(model, "meta") == (auto["comparator"], auto["threshold"])

    def test_the_schema_path_forwards_it(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "x-aws-stickler-match-threshold": 0.9,
                "x-aws-stickler-infer-unspecified": True,
                "properties": {"meta": {"type": "object", "additionalProperties": True}},
            }
        )
        assert _resolved(model, "meta") == ("ANLSStarComparator", 0.9)

    @pytest.mark.parametrize("path", ("config", "schema"))
    def test_the_default_is_unchanged_when_none_is_declared(self, path):
        if path == "config":
            model = StructuredModel.model_from_json(
                {
                    "model_name": "M",
                    "infer_unspecified_fields": True,
                    "fields": {"meta": {"type": "dict"}},
                }
            )
        else:
            model = StructuredModel.from_json_schema(
                {
                    "type": "object",
                    "title": "T",
                    "x-aws-stickler-infer-unspecified": True,
                    "properties": {
                        "meta": {"type": "object", "additionalProperties": True}
                    },
                }
            )
        assert _resolved(model, "meta") == ("ANLSStarComparator", 0.7)

    def test_a_nested_object_supplies_its_own_and_does_not_leak_it(self):
        """Scoped per subtree, like the flag beside it, and restored afterwards."""
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "x-aws-stickler-match-threshold": 0.9,
                "x-aws-stickler-infer-unspecified": True,
                "properties": {
                    "inner": {
                        "type": "object",
                        "x-aws-stickler-match-threshold": 0.4,
                        "properties": {
                            "meta": {"type": "object", "additionalProperties": True}
                        },
                    },
                    "outer_meta": {"type": "object", "additionalProperties": True},
                },
            }
        )
        annotation = model.model_fields["inner"].annotation
        inner = getattr(annotation, "__args__", (annotation,))[0]
        assert _resolved(inner, "meta") == ("ANLSStarComparator", 0.4)
        assert _resolved(model, "outer_meta") == ("ANLSStarComparator", 0.9)


class TestNamingAComparatorPinsTheThreshold:
    """Per-parameter filling stops at the comparator, deliberately.

    A threshold is only meaningful beside the metric that produced the score: 0.85
    means one thing on edit distance and another on numeric tolerance. Inference's
    threshold belongs to the comparator inference would have chosen, so applying it
    to a comparator the caller named instead would be a number lifted from a
    different metric.
    """

    def test_a_named_comparator_gets_the_ordinary_default(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {"total": {"type": "float", "comparator": "ExactComparator"}},
            }
        )
        assert _resolved(model, "total") == ("ExactComparator", 0.5)

    def test_letting_inference_pick_gets_the_matched_threshold(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {"total": {"type": "float"}},
            }
        )
        assert _resolved(model, "total") == ("NumericComparator", 0.95)


class TestListsAreInferredFromTheDeclaredElementType:
    """A list's comparator is inferred from its ELEMENT type, on every path.

    A list field's comparator is applied per element, so that is what inference
    reads. `{"type": "array", "items": {"type": "number"}}`,
    `{"type": "List[float]"}` and a pydantic `List[float]` all declare a float
    element and now all resolve to `NumericComparator@0.95`. The schema path used
    to pass no annotation at all and the config path inferred from the whole
    `List[float]`, landing on the exotic-type branch (whole-list canonical-JSON
    equality) -- two ways to get the wrong answer for the same declaration.

    A declaration that names NO element type is a different declaration, not the
    same one answered differently: see
    `TestAnUndeclaredElementTypeScoresTheWholeList`.
    """

    def test_the_schema_path(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "x-aws-stickler-infer-unspecified": True,
                "properties": {
                    "amounts": {"type": "array", "items": {"type": "number"}}
                },
            }
        )
        assert _resolved(model, "amounts") == ("NumericComparator", 0.95)

    def test_the_config_path_agrees(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {"amounts": {"type": "List[float]"}},
            }
        )
        assert _resolved(model, "amounts") == ("NumericComparator", 0.95)

    def test_and_so_does_stickler_eval_for(self):
        class Plain(BaseModel):
            amounts: Optional[List[float]] = None

        auto = stickler.eval_for(Plain).explain()["amounts"]
        assert (auto["comparator"], auto["threshold"]) == ("NumericComparator", 0.95)

    def test_a_string_element_gets_the_string_answer(self):
        """Not a NumericComparator special case: the element type decides."""
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {"tags": {"type": "List[str]"}},
            }
        )
        assert _resolved(model, "tags") == ("LevenshteinComparator", 0.7)

    def test_the_trail_says_the_spec_is_per_element(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {"amounts": {"type": "List[float]"}},
            }
        )
        why = stickler.eval_for(model).explain()["amounts"]["why"]
        assert why[0] == "list: spec applies to each element"

    def test_the_schema_path_still_defaults_without_the_flag(self):
        model = StructuredModel.from_json_schema(
            {
                "type": "object",
                "title": "T",
                "properties": {
                    "amounts": {"type": "array", "items": {"type": "number"}}
                },
            }
        )
        assert _resolved(model, "amounts") == ("NumericComparator", 0.5)


class TestAnUndeclaredElementTypeComparesElementsExactly:
    """`{"type": "list"}` names no element type, so there is nothing to infer.

    Inference sees a bare `list`, which has no scalar form of its own, and returns
    the exotic-type answer: `ExactComparator@1.0`. The engine still applies that
    per element, so the list keeps positional partial credit -- what it loses is
    the ELEMENT comparator. A float element differing by 1e-7 scores 0.0 where a
    declared `List[float]` scores 1.0, and a string element off by one character
    scores 0.0 where `List[str]` scores 0.83.

    This is a gotcha rather than an entry-point divergence: `stickler.eval_for`
    gives the same answer for a bare `list` annotation, so it is one rule applied
    to the same information, and `{"type": "List[float]"}` states the thing that
    makes an element comparator possible. Documented in
    docs/docs/Advanced/dynamic-models.md.
    """

    @staticmethod
    def _model(type_string, field="amounts"):
        return StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {field: {"type": type_string}},
            }
        )

    def test_a_bare_list_gets_the_exotic_type_answer(self):
        assert _resolved(self._model("list"), "amounts") == ("ExactComparator", 1.0)

    def test_stickler_eval_for_says_the_same_for_a_bare_list(self):
        class Plain(BaseModel):
            amounts: Optional[list] = None

        auto = stickler.eval_for(Plain).explain()["amounts"]
        assert (auto["comparator"], auto["threshold"]) == ("ExactComparator", 1.0)

    def test_positional_partial_credit_survives(self):
        """The exact comparator is applied per element, not to the list as a blob."""
        Bare = self._model("list")
        result = Bare(amounts=[1.0, 2.0, 3.0]).compare_with(
            Bare(amounts=[1.0, 2.0, 99.0])
        )
        assert result["overall_score"] == pytest.approx(2 / 3)

    def test_what_is_lost_is_the_element_comparator(self):
        """A float within the inferred tolerance scores 0.0 without the element type."""
        Bare, Typed = self._model("list"), self._model("List[float]")
        assert Bare(amounts=[1.0]).compare_with(Bare(amounts=[1.0000001]))[
            "overall_score"
        ] == pytest.approx(0.0)
        assert Typed(amounts=[1.0]).compare_with(Typed(amounts=[1.0000001]))[
            "overall_score"
        ] == pytest.approx(1.0)

    def test_the_same_holds_for_string_elements(self):
        Bare = self._model("list", field="tags")
        Typed = self._model("List[str]", field="tags")
        assert Bare(tags=["alpha"]).compare_with(Bare(tags=["alphaa"]))[
            "overall_score"
        ] == pytest.approx(0.0)
        assert Typed(tags=["alpha"]).compare_with(Typed(tags=["alphaa"]))[
            "overall_score"
        ] == pytest.approx(5 / 6)


class TestANestedConfigFieldCanScopeTheFlag:
    """The config path matches the schema path's per-subtree scoping.

    `_convert_nested_model_field` hardcoded the parent's flag into the synthesised
    nested config, so a nested field's own `infer_unspecified_fields` was discarded
    in both directions: a nested `false` still inferred, and a nested `true` was
    answered with an error advising exactly what the author had already written.
    """

    def test_a_nested_field_can_opt_its_subtree_in(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "fields": {
                    "outer": {"type": "str", "comparator": "ExactComparator"},
                    "inner": {
                        "type": "structured_model",
                        "infer_unspecified_fields": True,
                        "fields": {"amount": {"type": "float"}},
                    },
                },
            }
        )
        annotation = model.model_fields["inner"].annotation
        inner = getattr(annotation, "__args__", (annotation,))[0]
        assert _resolved(inner, "amount") == ("NumericComparator", 0.95)
        assert _resolved(model, "outer") == ("ExactComparator", 0.5)

    def test_the_refusal_no_longer_advises_what_the_author_already_did(self):
        """Setting the flag on the nested field must satisfy the requirement.

        It used to raise "Set 'infer_unspecified_fields': true on the model" for a
        config that had done precisely that, one level down.
        """
        StructuredModel.model_from_json(
            {
                "model_name": "M",
                "fields": {
                    "inner": {
                        "type": "structured_model",
                        "infer_unspecified_fields": True,
                        "fields": {"amount": {"type": "float"}},
                    }
                },
            }
        )

    def test_a_nested_field_can_opt_out_of_a_model_flag(self):
        with pytest.raises(ValueError, match="requires a 'comparator'"):
            StructuredModel.model_from_json(
                {
                    "model_name": "M",
                    "infer_unspecified_fields": True,
                    "fields": {
                        "inner": {
                            "type": "structured_model",
                            "infer_unspecified_fields": False,
                            "fields": {"amount": {"type": "float"}},
                        }
                    },
                }
            )

    def test_opting_out_does_not_change_the_enclosing_model(self):
        model = StructuredModel.model_from_json(
            {
                "model_name": "M",
                "infer_unspecified_fields": True,
                "fields": {
                    "outer_amount": {"type": "float"},
                    "inner": {
                        "type": "structured_model",
                        "infer_unspecified_fields": False,
                        "fields": {
                            "amount": {"type": "float", "comparator": "ExactComparator"}
                        },
                    },
                },
            }
        )
        annotation = model.model_fields["inner"].annotation
        inner = getattr(annotation, "__args__", (annotation,))[0]
        assert _resolved(model, "outer_amount") == ("NumericComparator", 0.95)
        assert _resolved(inner, "amount") == ("ExactComparator", 0.5)

    @pytest.mark.parametrize("value", ("yes", 1, None, []))
    def test_a_non_boolean_on_a_nested_field_is_refused(self, value):
        with pytest.raises(ValueError, match="must be true or false on field 'inner'"):
            StructuredModel.model_from_json(
                {
                    "model_name": "M",
                    "fields": {
                        "inner": {
                            "type": "structured_model",
                            "infer_unspecified_fields": value,
                            "fields": {
                                "amount": {
                                    "type": "float",
                                    "comparator": "ExactComparator",
                                }
                            },
                        }
                    },
                }
            )


class TestANestedObjectCanScopeTheFlag:
    """Reading the key only at the root made it a silent no-op elsewhere.

    `x-aws-stickler-infer-unspecified` is a known model-level key, so #312's
    unknown-key check accepts it at any object position. Accepting it and then
    ignoring it is the silent drop #210 exists to remove, reintroduced by a key
    those checks consider valid everywhere.
    """

    @staticmethod
    def _build(root, nested):
        schema = {
            "type": "object",
            "title": "T",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                },
                "outer_amount": {"type": "number"},
            },
        }
        if root is not None:
            schema["x-aws-stickler-infer-unspecified"] = root
        if nested is not None:
            schema["properties"]["inner"]["x-aws-stickler-infer-unspecified"] = nested
        model = StructuredModel.from_json_schema(schema)
        annotation = model.model_fields["inner"].annotation
        inner = getattr(annotation, "__args__", (annotation,))[0]
        return model, inner

    def test_a_nested_object_can_opt_its_subtree_in(self):
        model, inner = self._build(root=None, nested=True)
        assert _resolved(inner, "amount") == ("NumericComparator", 0.95)

    def test_and_does_not_leak_to_the_outer_object(self):
        """A sibling declared after the nested object must be unaffected."""
        model, _ = self._build(root=None, nested=True)
        assert _resolved(model, "outer_amount") == ("NumericComparator", 0.5)

    def test_a_nested_object_can_opt_out_of_a_root_flag(self):
        model, inner = self._build(root=True, nested=False)
        assert _resolved(inner, "amount") == ("NumericComparator", 0.5)

    def test_and_the_outer_setting_is_restored_afterwards(self):
        model, _ = self._build(root=True, nested=False)
        assert _resolved(model, "outer_amount") == ("NumericComparator", 0.95)

    def test_a_non_boolean_on_a_nested_object_is_refused(self):
        with pytest.raises(ValueError, match="must be true or false at 'inner'"):
            self._build(root=None, nested="yes")
