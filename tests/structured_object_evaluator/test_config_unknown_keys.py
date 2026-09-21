"""
Regression test for unrecognized config keys being dropped in silence.

``model_from_json`` read a closed set of keys and ignored everything else, so a
misspelled ``"threshhold"`` built at the fallback 0.5 with no exception and no
warning, and every score below it was wrong with nothing to point at.

See: https://github.com/awslabs/stickler/issues/350
"""

import inspect
import warnings

import pytest

from stickler.structured_object_evaluator.models.structured_model import StructuredModel


def build(config):
    """Build a model, returning it with any warnings raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = StructuredModel.model_from_json(config)
    return model, [str(w.message) for w in caught]


def unknown_key_warnings(messages):
    return [m for m in messages if "does not accept" in m]


BASE_FIELD = {"type": "str", "comparator": "ExactComparator"}


def test_misspelled_field_key_warns():
    """The reported case: 'threshhold' no longer builds in silence."""
    config = {
        "model_name": "Doc",
        "fields": {"name": dict(BASE_FIELD, threshhold=0.99)},
    }
    _, messages = build(config)
    warned = unknown_key_warnings(messages)
    assert len(warned) == 1
    assert "'threshhold'" in warned[0]
    assert "name" in warned[0]


def test_warning_names_the_accepted_keys():
    """A reader has to be able to find the right spelling from the message."""
    config = {"fields": {"name": dict(BASE_FIELD, weigth=3.0)}}
    _, messages = build(config)
    assert "weight" in unknown_key_warnings(messages)[0]


def test_misspelled_top_level_key_warns():
    """Top-level keys were dropped just as silently as field-level ones."""
    config = {"match_threshhold": 0.42, "fields": {"name": BASE_FIELD}}
    _, messages = build(config)
    assert "'match_threshhold'" in unknown_key_warnings(messages)[0]


@pytest.mark.parametrize(
    "key",
    ["clip_under_treshold", "requried", "discription", "comparator_configg"],
)
def test_each_reported_field_typo_warns(key):
    """Every field-level typo listed in the issue."""
    config = {"fields": {"name": dict(BASE_FIELD, **{key: "x"})}}
    _, messages = build(config)
    assert f"'{key}'" in unknown_key_warnings(messages)[0]


def test_schema_extension_key_on_the_config_path_warns():
    """A JSON Schema extension key is not a config key, and was silently dropped."""
    config = {
        "fields": {"name": dict(BASE_FIELD, **{"x-aws-stickler-threshold": 0.99})}
    }
    _, messages = build(config)
    assert "'x-aws-stickler-threshold'" in unknown_key_warnings(messages)[0]


def test_valid_config_is_silent():
    """Every recognized key, including the ones only a nested field uses."""
    config = {
        "model_name": "Doc",
        "match_threshold": 0.8,
        "fields": {
            "name": {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "comparator_config": {},
                "threshold": 0.9,
                "weight": 2.0,
                "clip_under_threshold": False,
                "required": True,
                "description": "d",
                "alias": "n",
                "examples": ["x"],
            },
            "addr": {
                "type": "structured_model",
                "match_threshold": 0.6,
                "fields": {"city": BASE_FIELD},
            },
        },
    }
    _, messages = build(config)
    assert unknown_key_warnings(messages) == []


def test_model_name_on_a_nested_field_is_accepted():
    """A nested model_name is never read, but to_stickler_config() exports it.

    Rejecting it would make the library's own export fail to round-trip, so it
    is accepted and ignored rather than reported.
    """
    config = {
        "fields": {
            "addr": {
                "type": "structured_model",
                "model_name": "Address",
                "fields": {"city": BASE_FIELD},
            }
        }
    }
    _, messages = build(config)
    assert unknown_key_warnings(messages) == []


def test_exported_config_rebuilds_without_warnings():
    """The round-trip that a strict unknown-key check would break."""
    source = {
        "model_name": "Doc",
        "match_threshold": 0.8,
        "fields": {
            "name": dict(BASE_FIELD, threshold=0.9, weight=2.0),
            "addr": {
                "type": "structured_model",
                "model_name": "Address",
                "match_threshold": 0.6,
                "fields": {"city": BASE_FIELD},
            },
        },
    }
    model, _ = build(source)
    rebuilt, messages = build(model.to_stickler_config())
    assert unknown_key_warnings(messages) == []
    assert rebuilt.to_stickler_config()["match_threshold"] == 0.8


def test_the_misspelled_model_still_builds():
    """The key was never applied, so refusing the model would be a new failure."""
    config = {"fields": {"name": dict(BASE_FIELD, threshhold=0.99)}}
    model, _ = build(config)
    assert model.to_stickler_config()["fields"]["name"]["threshold"] == 0.5


# Both models are built inside ONE test deliberately. The autouse fixture in
# this directory's conftest clears the warn-once set between tests, so a
# process-scoped dedup defect is invisible to any test that builds one model --
# which is why the first round of these tests passed while the warning fired
# once per process.
def test_a_later_model_with_the_same_typo_still_warns():
    """The reported symptom, reproduced after the fix and now closed.

    Keyed on the leaf field name, the first model in a process consumed the
    only slot, so a later unrelated model carrying the same typo scored at the
    default in silence.
    """
    build(
        {"model_name": "Earlier", "fields": {"name": dict(BASE_FIELD, threshhold=0.5)}}
    )
    _, messages = build(
        {"model_name": "Invoice", "fields": {"name": dict(BASE_FIELD, threshhold=0.99)}}
    )
    assert len(unknown_key_warnings(messages)) == 1


def test_each_misspelled_path_warns_and_names_itself():
    """Two fields called 'city' are two misconfigurations, not one."""
    config = {
        "model_name": "Order",
        "fields": {
            "billing": {
                "type": "structured_model",
                "fields": {"city": dict(BASE_FIELD, threshhold=0.9)},
            },
            "shipping": {
                "type": "structured_model",
                "fields": {"city": dict(BASE_FIELD, threshhold=0.9)},
            },
        },
    }
    _, messages = build(config)
    warned = unknown_key_warnings(messages)
    assert len(warned) == 2
    assert any("'billing.city'" in m for m in warned)
    assert any("'shipping.city'" in m for m in warned)


def test_warning_points_at_the_callers_line():
    """A warning attributed to the library names a line the reader cannot fix."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        this_line = inspect.currentframe().f_lineno + 1
        StructuredModel.model_from_json(
            {"fields": {"name": dict(BASE_FIELD, nosuchkey=1)}}
        )
    unknown = [w for w in caught if "does not accept" in str(w.message)]
    assert len(unknown) == 1
    assert unknown[0].filename == __file__
    assert unknown[0].lineno == this_line


def test_model_level_warning_points_at_the_callers_line():
    """The model-level warning sits one frame shallower than the field-level one."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        this_line = inspect.currentframe().f_lineno + 1
        StructuredModel.model_from_json(
            {"nosuchkey": 1, "fields": {"name": BASE_FIELD}}
        )
    unknown = [w for w in caught if "does not accept" in str(w.message)]
    assert len(unknown) == 1
    assert unknown[0].filename == __file__
    assert unknown[0].lineno == this_line


@pytest.mark.parametrize("key", [42, None, (1, 2)])
def test_a_non_string_key_warns_rather_than_raising_type_error(key):
    """`','.join(sorted(...))` raised on any key that was not a string.

    `model_from_json` documents `Raises: ValueError`, so a TypeError escaping
    from the warning path is a worse outcome than the silent drop it replaced.
    """
    _, messages = build({"fields": {"name": {**BASE_FIELD, key: "x"}}})
    assert str(key) in unknown_key_warnings(messages)[0]


def test_a_non_string_key_at_model_level_also_warns():
    """The same join, at the other level."""
    _, messages = build({"fields": {"name": BASE_FIELD}, 42: "x"})
    assert "42" in unknown_key_warnings(messages)[0]


def test_a_non_mapping_field_config_is_not_reported_as_unknown_keys():
    """`set()` of a str iterates its characters.

    `{"name": "str"}` reported the field as not accepting 'r', 's', 't' before
    refusing it. The refusal is right and unchanged; the warning was noise
    naming keys nobody wrote, so the assertion is on the warning, not the raise.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="missing required 'type' parameter"):
            StructuredModel.model_from_json({"fields": {"name": "str"}})
    assert unknown_key_warnings([str(w.message) for w in caught]) == []


def test_match_threshold_on_a_primitive_is_refused():
    """It is the object-level threshold; on a leaf it was accepted and dropped.

    The JSON Schema path already refuses the same misplacement, so refusing
    here is what makes the two front doors agree.
    """
    config = {
        "fields": {
            "amount": {
                "type": "float",
                "comparator": "NumericComparator",
                "match_threshold": 0.99,
            }
        }
    }
    with pytest.raises(ValueError, match="'match_threshold' is not read"):
        build(config)


def test_match_threshold_on_a_nested_model_field_is_still_accepted():
    """It is read there, and `to_stickler_config()` exports it there.

    Removing it from the accepted set would refuse the library's own export,
    which is why the refusal above is per-branch rather than per-key.
    """
    config = {
        "fields": {
            "addr": {
                "type": "structured_model",
                "match_threshold": 0.93,
                "fields": {"city": BASE_FIELD},
            }
        }
    }
    model, messages = build(config)
    assert unknown_key_warnings(messages) == []
    assert model.to_stickler_config()["fields"]["addr"]["match_threshold"] == 0.93
