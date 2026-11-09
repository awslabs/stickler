"""Scalar-union lists retain their element types through a Stickler config."""

import json
from typing import Annotated, List, Optional, Union

import pytest

from stickler import ComparableField, ExactComparator, StructuredModel


@pytest.mark.parametrize(
    "annotation, values, type_name",
    [
        (List[Union[int, str]], [1, "01", "blue"], "List[Union[int, str]]"),
        (list[int | str], [1, "01", "blue"], "List[Union[int, str]]"),
        (
            List[Union[int, float, bool]],
            [1, 1.5, True],
            "List[Union[int, float, bool]]",
        ),
        (
            List[Union[str, bytes]],
            ["text", b"bytes"],
            "List[Union[str, bytes]]",
        ),
        (
            List[Optional[Union[int, str]]],
            [1, "01", None],
            "List[Optional[Union[int, str]]]",
        ),
        (
            list[int | str | None],
            [1, "01", None],
            "List[Optional[Union[int, str]]]",
        ),
        (
            List[Annotated[Union[int, str], "element"]],
            [1, "01"],
            "List[Union[int, str]]",
        ),
        (
            List[Union[Annotated[int, "integer"], str]],
            [1, "01"],
            "List[Union[int, str]]",
        ),
    ],
)
def test_scalar_union_list_survives_two_config_round_trips(
    annotation, values, type_name
):
    """Dropping union arms must not reject or coerce previously valid values."""

    class Original(StructuredModel):
        values: annotation = ComparableField(default=..., comparator=ExactComparator())

    current = Original
    for _ in range(2):
        config = json.loads(json.dumps(current.to_stickler_config()))
        rebuilt = StructuredModel.model_from_json(config)
        assert config["fields"]["values"]["type"] == type_name
        actual = rebuilt(values=values).values
        assert actual == values
        assert [type(value) for value in actual] == [type(value) for value in values]
        assert rebuilt.model_fields["values"].is_required()
        current = rebuilt


@pytest.mark.parametrize(
    "annotation, default",
    [
        (List[Union[int, str]], None),
        (List[Union[int, str]], []),
        (List[Union[int, str]], [1, "01"]),
        (Optional[List[Union[int, str]]], None),
    ],
)
def test_scalar_union_list_preserves_defaults_and_comparison_settings(
    annotation, default
):
    """A new type string must keep defaults and the field's comparison contract."""

    class Original(StructuredModel):
        tags: annotation = ComparableField(
            default=default,
            comparator=ExactComparator(),
            threshold=0.9,
            weight=2.0,
            clip_under_threshold=False,
            description="Mixed tags",
            examples=[[1, "01"]],
        )

    config = json.loads(json.dumps(Original.to_stickler_config()))
    rebuilt = StructuredModel.model_from_json(config)
    assert rebuilt().tags == default
    field = config["fields"]["tags"]
    assert field["required"] is False
    assert field["default"] == default
    assert field["comparator"] == "ExactComparator"
    assert field["threshold"] == 0.9
    assert field["weight"] == 2.0
    assert field["clip_under_threshold"] is False
    assert field["description"] == "Mixed tags"
    assert field["examples"] == [[1, "01"]]
    assert rebuilt.to_stickler_config()["fields"]["tags"] == field
    gt, pred = {"tags": [1, "01"]}, {"tags": [1, "different"]}
    for model in (Original, rebuilt):
        result = model(**gt).compare_with(model(**pred), include_confusion_matrix=True)
        assert result["overall_score"] == 0.5
        assert result["confusion_matrix"]["overall"]["tp"] == 1
        assert result["confusion_matrix"]["overall"]["fd"] == 1
