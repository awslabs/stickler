"""Scalar-union lists retain their element types through a Stickler config."""

import json
import re
from datetime import date
from decimal import Decimal
from typing import Annotated, List, Optional, Union

import pytest
from pydantic import Field

from stickler import ComparableField, ExactComparator, StructuredModel


def _normalized(type_name):
    """Sort each ``Union[...]``'s arms.

    ``typing`` caches aliases, so ``List[Union[int, str]]`` can come back as
    ``List[Union[str, int]]`` if that spelling was created first in the process.
    """
    return re.sub(
        r"Union\[([^\[\]]+)\]",
        lambda m: "Union[" + ", ".join(sorted(m.group(1).split(", "))) + "]",
        type_name,
    )


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
            List[Optional[Annotated[Union[int, str], "element"]]],
            [1, "01", None],
            "List[Optional[Union[int, str]]]",
        ),
        (
            list[Annotated[int | str, Field(description="element")] | None],
            [1, "01", None],
            "List[Optional[Union[int, str]]]",
        ),
        (
            List[Annotated[Optional[Union[int, str]], "element"]],
            [1, "01", None],
            "List[Optional[Union[int, str]]]",
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
        assert _normalized(config["fields"]["values"]["type"]) == _normalized(type_name)
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


@pytest.mark.parametrize(
    "annotation",
    [
        List[Union[int, Decimal]],
        list[int | Decimal],
        List[Optional[Union[int, date]]],
    ],
)
def test_unsupported_union_arm_keeps_legacy_list_union(annotation):
    """An arm the config cannot name must not be exported as another type.

    `model_from_json()` cannot read `List[Union]`, so the export warns (#636).
    """

    class Original(StructuredModel):
        values: annotation = ComparableField(default=..., comparator=ExactComparator())

    with pytest.warns(UserWarning, match='written as "List\\[Union\\]"'):
        config = Original.to_stickler_config()
    assert config["fields"]["values"]["type"] == "List[Union]"
