"""Oracle tests for LevenshteinComparator._levenshtein_distance."""

import pytest

from stickler.comparators.levenshtein import LevenshteinComparator


def oracle(a: str, b: str) -> int:
    """Copy of the original pure-Python DP, kept as the reference."""
    if len(a) > len(b):
        a, b = b, a
    distances = range(len(a) + 1)
    for i2, c2 in enumerate(b):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(a):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


CORPUS = [
    ("", ""),
    ("a", ""),
    ("", "a"),
    ("kitten", "sitting"),
    ("saturday", "sunday"),
    ("\ud800", "\ud800"),
    ("ß", "SS"),
    ("Σ", "σ"),
    ("a" * 63, "a" * 64),
    ("a" * 64, "a" * 65),
    ("a" * 127, "a" * 128),
    ("a" * 128, "a" * 129),
    ("a" * 257, "b" * 257),
    ("a" * 1000, "b" * 1000),
]


@pytest.mark.parametrize("a,b", CORPUS)
def test_levenshtein_distance_matches_oracle(a, b):
    assert LevenshteinComparator._levenshtein_distance(a, b) == oracle(a, b)


def test_anls_star_example_score_pinned():
    import stickler
    from pydantic import BaseModel

    class Invoice(BaseModel):
        invoice_id: str
        metadata: dict = {}

    truth = Invoice(invoice_id="INV-1042", metadata={"vendor": "Acme Corporation", "terms": "Net 30", "po": "PO-88231"})
    prediction = Invoice(invoice_id="INV-1042", metadata={"vendor": "Acme Corp", "terms": "Net 30", "po": "PO-88231"})

    result = stickler.evaluate(truth, prediction)
    assert result.field_scores["metadata"] == 0.8541666666666666