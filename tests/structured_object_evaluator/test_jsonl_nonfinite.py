"""The individual-results JSONL is strict JSON even when a value is NaN or infinite.

``json.dumps`` writes non-finite floats as bare ``NaN``/``Infinity`` tokens,
which JavaScript ``JSON.parse`` and other strict parsers reject. A blank pandas cell is
NaN, so this is the ordinary case. The writer now writes ``null``, except a
confidence or bbox-map value, written as ``"nan"``/``"inf"``/``"-inf"`` so
replay preserves the direct path's invalid-but-present value (replay parity is
tested in ``test_prediction_raw_roundtrip.py``).

Python's ``json.loads`` accepts ``NaN`` by default, so these tests parse with
``parse_constant`` raising, which is what a strict parser does.

See https://github.com/awslabs/stickler/issues/450
"""

import json
import math
import sys
from typing import Optional

import numpy as np
import pytest

from stickler import ComparableField, StructuredModel
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
    _jsonl_line,
    _replace_non_finite,
)

_JSON_DEEP_NESTING = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="json's C encoder and decoder hit the recursion limit on 5000-deep input before 3.12",
)

NAN, INF = float("nan"), float("inf")


def _reject(constant):
    raise ValueError(f"non-JSON constant {constant}")


def strict_loads(text, **kwargs):
    return json.loads(text, parse_constant=_reject, **kwargs)


class Invoice(StructuredModel):
    vendor: str = ComparableField()
    total: Optional[float] = ComparableField()


def test_the_evaluator_writes_strict_json_lines(tmp_path):
    path = tmp_path / "results.jsonl"
    evaluator = BulkStructuredModelEvaluator(
        target_schema=Invoice, individual_results_jsonl=str(path)
    )
    pairs = [
        (Invoice(vendor="Acme", total=1.0), Invoice(vendor="Acme", total=NAN)),
        (Invoice(vendor="Acme", total=INF), Invoice(vendor="Bolt", total=-INF)),
        (Invoice(vendor="Acme", total=2.0), Invoice(vendor="Acme", total=2.0)),
    ]
    for i, (gt, pred) in enumerate(pairs):
        evaluator.update(gt, pred, doc_id=f"doc{i}")
    evaluator.close()

    lines = path.read_text().splitlines()
    assert len(lines) == len(pairs)
    records = [strict_loads(line) for line in lines]
    assert [r["doc_id"] for r in records] == ["doc0", "doc1", "doc2"]
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()


@pytest.mark.parametrize(
    "value, expected",
    [
        (NAN, None),
        (-INF, None),
        ({"a": [1.5, NAN, {"b": (INF, -INF, 2)}]}, {"a": [1.5, None, {"b": [None, None, 2]}]}),
        ([np.float64("nan"), np.float64(0.25)], [None, 0.25]),
        ({NAN: 1, "k": INF}, {"NaN": 1, "k": None}),
    ],
)
def test_non_finite_numbers_become_null_at_any_depth(value, expected):
    assert strict_loads(_jsonl_line(value)) == expected


@_JSON_DEEP_NESTING
def test_deep_nesting_with_one_nan_is_not_limited_by_recursion():
    """The copy walks iteratively, so it reaches as deep as ``json.dumps`` does."""
    depth = 5000
    value = [NAN]
    for _ in range(depth):
        value = [value]

    node = strict_loads(_jsonl_line(value))
    for _ in range(depth):
        (node,) = node
    assert node == [None]


def test_a_nan_key_next_to_a_nan_string_key_writes_both_pairs():
    """Keys are left alone, so the writer drops neither pair. The line then has
    a duplicate ``"NaN"`` key, as on dev; a plain ``json.loads`` keeps the last,
    so both are only visible through ``object_pairs_hook``."""
    line = _jsonl_line({NAN: 1, "NaN": 2, "v": NAN})

    assert strict_loads(line, object_pairs_hook=list) == [("NaN", 1), ("NaN", 2), ("v", None)]


def test_shared_containers_are_copied_once_and_stay_shared():
    shared = {"x": NAN}
    assert strict_loads(_jsonl_line({"a": shared, "b": [shared, shared]})) == {
        "a": {"x": None},
        "b": [{"x": None}, {"x": None}],
    }


def test_the_copy_alone_is_not_limited_by_recursion():
    """No 3.12 skip: this checks the explicit stack, not ``json``'s C code."""
    depth = 5000
    value = {"_confidence": [INF]}
    for _ in range(depth):
        value = [value]

    node = _replace_non_finite(value)
    for _ in range(depth):
        (node,) = node
    assert node == {"_confidence": ["inf"]}


@pytest.mark.parametrize(
    "value, expected",
    [
        ({"_value": NAN, "_confidence": NAN}, {"_value": None, "_confidence": "nan"}),
        ({"_confidence": np.float64("-inf")}, {"_confidence": "-inf"}),
        (
            {"prediction_confidences": {"a": INF, "b.c": 0.5}, "v": INF},
            {"prediction_confidences": {"a": "inf", "b.c": 0.5}, "v": None},
        ),
    ],
)
def test_a_non_finite_confidence_is_written_as_a_string(value, expected):
    """Null would read back as "no confidence"; a string still fails validation."""
    assert strict_loads(_jsonl_line(value)) == expected


def test_a_non_finite_bbox_map_value_is_written_as_a_string():
    """Null would read back as no box; a string remains an invalid box."""
    value = {
        "ground_truth_bboxes": {"total": NAN},
        "prediction_bboxes": {"total": -INF},
        "unrelated": INF,
    }

    assert strict_loads(_jsonl_line(value)) == {
        "ground_truth_bboxes": {"total": "nan"},
        "prediction_bboxes": {"total": "-inf"},
        "unrelated": None,
    }


@pytest.mark.parametrize("member", [1.0, NAN], ids=["finite", "nan"])
def test_a_circular_record_still_raises_value_error(member):
    loop = {"n": member}
    loop["self"] = loop

    with pytest.raises(ValueError, match="Circular reference"):
        _jsonl_line(loop)


@pytest.mark.parametrize(
    "value",
    [
        {"total": 1e308, "tiny": 5e-324, "neg": -0.0, "int": 10**20, "flag": True},
        [0.1, 1 / 3, -2.5e-10, math.pi, None, "nan", "Infinity"],
        {"nested": ({"x": 1.0},), "np": np.int64(3), "obj": object},
    ],
)
def test_finite_records_are_written_exactly_as_before(value):
    assert _jsonl_line(value) == json.dumps(value, default=str)
