"""
Regression test for unrecognized config keys being dropped in silence.

``model_from_json`` read a closed set of keys and ignored everything else, so a
misspelled ``"threshhold"`` built at the fallback 0.5 with no exception and no
warning, and every score below it was wrong with nothing to point at.

See: https://github.com/awslabs/stickler/issues/350
"""

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
    config = {"fields": {"name": dict(BASE_FIELD, **{"x-aws-stickler-threshold": 0.99})}}
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
