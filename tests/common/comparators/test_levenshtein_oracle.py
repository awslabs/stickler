"""Pin edit distance and public scores before replacing the Python DP (#510)."""

import random
from itertools import product
from typing import Any

import pytest
from pydantic import BaseModel

import stickler
from stickler import ANLSStarComparator, LevenshteinComparator


def oracle(s1: str, s2: str) -> int:
    """The original dynamic-programming implementation, independent of RapidFuzz."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def _string_pairs():
    edge_cases = (
        "",
        "a",
        "ab",
        "e\u0301",
        "\u00e9",
        "\U0001f600",
        "\U0001f469\u200d\U0001f4bb",
        "\ud800",
        "\udfff",
        "\x00",
        "a\x00b",
        "\u00df",
        "\u03a3",
        "\u03c2",
    )
    yield from product(edge_cases, repeat=2)

    rng = random.Random(510)
    alphabet = "abcXYZ \u00e9\u0301\u00df\u03a3\U0001f600\u200d\ud800\x00"

    def text(length):
        return "".join(rng.choices(alphabet, k=length))

    # Exercise both sides of RapidFuzz's 64-bit block boundaries, including
    # unequal lengths so the original DP's argument swap is covered.
    lengths = (63, 64, 65, 127, 128, 129, 257)
    for left, right in product(lengths, repeat=2):
        yield text(left), text(right)
    for length in lengths:
        value = text(length)
        yield value, value
        yield value, value[:-1] + "!"
    for _ in range(100):
        yield text(rng.randrange(81)), text(rng.randrange(81))
    for length in (1000, 3000):
        yield text(length), text(length)


@pytest.mark.parametrize("left,right", list(_string_pairs()))
def test_distance_and_similarity_match_original_dp(left, right):
    distance = oracle(left, right)
    assert LevenshteinComparator._levenshtein_distance(left, right) == distance

    length = max(len(left), len(right))
    expected = 1.0 - (float(distance) / float(length)) if length else 1.0
    # Exact equality intentionally catches a different floating-point formula.
    assert LevenshteinComparator(normalize=False).compare(left, right) == expected


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "left,right",
    [
        ("  Acme\tCorporation\n", "ACME Corp"),
        ("\u03a3 \u00df", "\u03c3 SS"),
        (42, 43),
        (1.25, 1.5),
        (float("nan"), float("inf")),
        (True, False),
        (b"abc", b"abd"),
        ([1, 2], [1, 3]),
    ],
)
def test_normalization_and_string_coercion_are_unchanged(left, right, normalize):
    s1, s2 = str(left), str(right)
    if normalize:
        s1 = " ".join(s1.strip().lower().split())
        s2 = " ".join(s2.strip().lower().split())
    expected = 1.0 - (float(oracle(s1, s2)) / float(max(len(s1), len(s2))))
    assert LevenshteinComparator(normalize=normalize).compare(left, right) == expected


def test_documented_anls_example_exact_scores():
    """The guide rounds the metadata score to 0.8542; pin dev's actual float."""

    class Invoice(BaseModel):
        invoice_id: str
        metadata: dict[str, Any] = {}

    truth = Invoice(
        invoice_id="INV-1042",
        metadata={"vendor": "Acme Corporation", "terms": "Net 30", "po": "PO-88231"},
    )
    prediction = Invoice(
        invoice_id="INV-1042",
        metadata={"vendor": "Acme Corp", "terms": "Net 30", "po": "PO-88231"},
    )

    assert ANLSStarComparator().compare(truth.metadata, prediction.metadata) == (
        0.8541666666666666
    )
    result = stickler.evaluate(truth, prediction)
    assert result.field_scores["metadata"] == 0.8541666666666666
    assert result.overall_score == 0.9270833333333333
