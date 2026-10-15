"""Scalar fields reach their comparator, in both readers (#383).

``ComparisonDispatcher`` only routed ``str``/``int``/``float`` to the field's
comparator. A field typed ``date``, ``datetime``, ``time``, ``Decimal``,
``UUID`` or a plain ``Enum`` fell through to the mismatched-types case: it
scored 0.0 against an identical value, was counted as a false discovery, and
the comparator was never called, so declaring ``DateComparator`` could not
help. ``stickler.evaluate()`` was unaffected only because its shadow models
turn these values into strings first.

The rule is shared by ``compare_with()`` and ``compare()``: a comparator the
field declared is always called, and an inferred one sees only the scalar
types it can be trusted with. Tuples and sets are out of scope (#391).
"""

import datetime
import decimal
import enum
import uuid
from typing import Any, List, Optional, Set, Tuple, Union

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators import (
    DateComparator,
    ExactComparator,
    LevenshteinComparator,
    NumericComparator,
)
from stickler.comparators.base import BaseComparator


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Size(enum.Enum):
    RED = "red"


# (type, value, a different value of the same type)
CASES = [
    (datetime.date, datetime.date(2026, 9, 14), datetime.date(2027, 3, 1)),
    (
        datetime.datetime,
        datetime.datetime(2026, 9, 14, 9),
        datetime.datetime(2027, 3, 1, 18),
    ),
    (datetime.time, datetime.time(9, 30), datetime.time(18, 0)),
    (datetime.timedelta, datetime.timedelta(seconds=5), datetime.timedelta(hours=9)),
    (decimal.Decimal, decimal.Decimal("12.50"), decimal.Decimal("99.00")),
    (uuid.UUID, uuid.UUID(int=1), uuid.UUID(int=2)),
    (Color, Color.RED, Color.BLUE),
]
IDS = [c[0].__name__ for c in CASES]


def _model(annotation, comparator):
    return type(
        f"M_{getattr(annotation, '__name__', 'field')}",
        (StructuredModel,),
        {
            "__annotations__": {"f": annotation},
            "f": ComparableField(comparator=comparator),
        },
    )


@pytest.mark.parametrize("annotation, value, other", CASES, ids=IDS)
def test_an_identical_value_is_a_true_positive(annotation, value, other):
    M = _model(annotation, ExactComparator())
    result = M(f=value).compare_with(M(f=value), include_confusion_matrix=True)
    assert result["overall_score"] == pytest.approx(1.0)
    counts = result["confusion_matrix"]["overall"]
    assert (counts["tp"], counts["fd"]) == (1, 0)


@pytest.mark.parametrize("annotation, value, other", CASES, ids=IDS)
def test_a_different_value_is_a_false_discovery(annotation, value, other):
    M = _model(annotation, ExactComparator())
    result = M(f=value).compare_with(M(f=other), include_confusion_matrix=True)
    assert result["overall_score"] == pytest.approx(0.0)
    counts = result["confusion_matrix"]["overall"]
    assert (counts["tp"], counts["fd"]) == (0, 1)


@pytest.mark.parametrize("annotation, value, other", CASES, ids=IDS)
def test_the_declared_comparator_is_called(annotation, value, other, monkeypatch):
    calls = []
    original = ExactComparator.compare

    def spy(self, a, b):
        calls.append((a, b))
        return original(self, a, b)

    monkeypatch.setattr(ExactComparator, "compare", spy)
    M = _model(annotation, ExactComparator())
    M(f=value).compare_with(M(f=value))
    assert calls == [(value, value)]


def test_date_comparator_scores_a_date_field():
    """The case from the issue: a perfect match was reported as an FD."""

    class Shipped(StructuredModel):
        ship_date: datetime.date = ComparableField(comparator=DateComparator())

    d = datetime.date(2026, 9, 14)
    result = Shipped(ship_date=d).compare_with(
        Shipped(ship_date=d), include_confusion_matrix=True
    )
    counts = result["confusion_matrix"]["overall"]
    assert (counts["tp"], counts["fd"]) == (1, 0)


def test_numeric_comparator_scores_a_decimal_field():
    class Invoice(StructuredModel):
        total: decimal.Decimal = ComparableField(comparator=NumericComparator())

    result = Invoice(total=decimal.Decimal("12.50")).compare_with(
        Invoice(total=decimal.Decimal("12.5"))
    )
    assert result["overall_score"] == pytest.approx(1.0)


def test_a_date_and_a_datetime_reach_the_comparator():
    """`datetime` subclasses `date`, so the pair is related, not mismatched."""

    class M(StructuredModel):
        when: Union[datetime.datetime, datetime.date] = ComparableField(
            comparator=DateComparator()
        )

    result = M(when=datetime.date(2026, 9, 14)).compare_with(
        M(when=datetime.datetime(2026, 9, 14, 9))
    )
    assert result["overall_score"] == pytest.approx(1.0)


class Blob:
    """A non-pydantic type: its default repr differs only by address."""


class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y


class BytesEqual(BaseComparator):
    def _compare(self, a, b):
        return 1.0 if a == b else 0.0


class PointDistance(BaseComparator):
    def _compare(self, a, b):
        return max(0.0, 1.0 - (abs(a.x - b.x) + abs(a.y - b.y)) / 10)


class Jaccard(BaseComparator):
    """Written for sets: it needs the values as sets, not a stringified form."""

    def _compare(self, a, b):
        return len(a & b) / len(a | b)


def _readers(model, gt, pred):
    """compare_with()'s overall score and compare(), for one-field models.

    ``compare()`` is the raw score and ``compare_with()`` clips a score under
    the field threshold, so tests that compare the two either use scores of
    0.0 and 1.0 or turn clipping off.
    """
    a, b = model(f=gt), model(f=pred)
    return a.compare_with(b)["overall_score"], a.compare(b)


class TestAnInferredComparatorSeesOnlyTrustedScalars:
    """No comparator declared, so the pair must be of a kind it can score."""

    @pytest.mark.parametrize(
        "gt, pred",
        [
            (datetime.date(2026, 9, 14), "2026-09-14"),
            (Color.RED, Size.RED),
            (uuid.UUID(int=1), decimal.Decimal(1)),
            (b"abc", b"abc"),
        ],
        ids=["date-vs-str", "two-enum-classes", "uuid-vs-decimal", "bytes"],
    )
    def test_unrelated_or_untrusted_types_are_a_mismatch(self, gt, pred):
        class M(StructuredModel):
            f: Any = ComparableField()

        result = M(f=gt).compare_with(M(f=pred), include_confusion_matrix=True)
        assert result["overall_score"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1
        assert M(f=gt).compare(M(f=pred)) == pytest.approx(0.0)

    def test_two_distinct_objects_do_not_match_on_their_reprs(self):
        class M(StructuredModel):
            model_config = {"arbitrary_types_allowed": True}
            f: Blob = ComparableField()

        assert _readers(M, Blob(), Blob()) == (0.0, 0.0)

    def test_two_int_enum_classes_are_a_mismatch(self):
        """An IntEnum member is also an int, so Enum is checked first."""

        class X(enum.IntEnum):
            A = 1

        class Y(enum.IntEnum):
            B = 1

        class M(StructuredModel):
            f: Any = ComparableField()

        assert _readers(M, X.A, Y.B) == (0.0, 0.0)
        assert _readers(M, X.A, 1) == (0.0, 0.0)
        assert _readers(M, X.A, X.A) == (1.0, 1.0)

    def test_a_decimal_against_a_float_is_a_number_pair(self):
        class M(StructuredModel):
            f: Union[decimal.Decimal, float] = ComparableField(
                comparator=NumericComparator()
            )

        gt = M.model_construct(f=decimal.Decimal("12.5"))
        pred = M.model_construct(f=12.5)
        assert gt.compare_with(pred)["overall_score"] == pytest.approx(1.0)
        assert gt.compare(pred) == pytest.approx(1.0)


class TestADeclaredComparatorIsAlwaysCalled:
    """The caller chose it for these values; it decides what it can score."""

    def test_a_bytes_field(self):
        class M(StructuredModel):
            f: bytes = ComparableField(comparator=BytesEqual())

        assert _readers(M, b"abc", b"abc") == (1.0, 1.0)
        assert _readers(M, b"abc", b"xyz") == (0.0, 0.0)

    def test_a_custom_class(self):
        class M(StructuredModel):
            model_config = {"arbitrary_types_allowed": True}
            f: Point = ComparableField(comparator=PointDistance())

        compare_with, compare = _readers(M, Point(0, 0), Point(1, 0))
        assert compare_with == pytest.approx(0.9)
        assert compare == pytest.approx(0.9)

    def test_a_date_against_its_iso_string(self):
        class M(StructuredModel):
            f: Union[datetime.date, str] = ComparableField(comparator=DateComparator())

        assert _readers(M, datetime.date(2026, 9, 14), "2026-09-14") == (1.0, 1.0)

    def test_a_set_reaches_a_set_aware_comparator_as_a_set(self):
        """Values arrive as the caller supplied them (#384 review)."""

        class M(StructuredModel):
            f: Set[str] = ComparableField(
                comparator=Jaccard(), clip_under_threshold=False
            )

        compare_with, compare = _readers(M, {"a", "b"}, {"b", "c"})
        assert compare_with == pytest.approx(1 / 3)
        assert compare == pytest.approx(1 / 3)

    def test_the_comparator_receives_the_values_unchanged(self, monkeypatch):
        seen = []
        original = ExactComparator.compare

        def spy(self, a, b):
            seen.append((a, b))
            return original(self, a, b)

        monkeypatch.setattr(ExactComparator, "compare", spy)

        class M(StructuredModel):
            f: Tuple[float, ...] = ComparableField(comparator=ExactComparator())

        M(f=(3.0, 1.0)).compare_with(M(f=(3.0, 1.0)))
        M(f=(3.0, 1.0)).compare(M(f=(3.0, 1.0)))
        assert seen == [((3.0, 1.0), (3.0, 1.0))] * 2


class TestTheTwoReadersAgree:
    """`compare()` feeds the Hungarian cost matrix, so it must agree (#233)."""

    class X(enum.IntEnum):
        A = 1

    class Y(enum.IntEnum):
        B = 1

    PAIRS = [
        (datetime.date(2026, 9, 14), "2026-09-14"),
        (datetime.date(2026, 9, 14), datetime.date(2026, 9, 14)),
        (decimal.Decimal("12.5"), 12.5),
        (X.A, Y.B),
        (X.A, 1),
        (Color.RED, Color.RED),
        (uuid.UUID(int=1), decimal.Decimal(1)),
        (b"abc", b"abc"),
        ("text", 3),
    ]

    @pytest.mark.parametrize("gt, pred", PAIRS)
    @pytest.mark.parametrize(
        "comparator",
        [None, ExactComparator(), LevenshteinComparator()],
        ids=["inferred", "explicit-exact", "explicit-levenshtein"],
    )
    def test_compare_matches_compare_with(self, gt, pred, comparator):
        # Clipping off, so a partial score is compared raw in both readers.
        field = (
            ComparableField(clip_under_threshold=False)
            if comparator is None
            else ComparableField(comparator=comparator, clip_under_threshold=False)
        )

        class M(StructuredModel):
            f: Any = field

        compare_with, compare = _readers(M, gt, pred)
        assert compare == pytest.approx(compare_with)


@pytest.mark.parametrize("annotation, value, other", CASES, ids=IDS)
def test_scalar_and_list_spellings_agree(annotation, value, other):
    """The list path already reached the comparator; the scalar path now does."""
    scalar = _model(annotation, ExactComparator())
    listed = _model(List[annotation], ExactComparator())
    for pred in (value, other):
        assert scalar(f=value).compare_with(scalar(f=pred))[
            "overall_score"
        ] == pytest.approx(
            listed(f=[value]).compare_with(listed(f=[pred]))["overall_score"]
        )


@pytest.mark.parametrize("annotation, value, other", CASES, ids=IDS)
def test_reports_show_the_comparator_verdict(annotation, value, other):
    M = _model(Optional[annotation], ExactComparator())
    result = M(f=value).compare_with(
        M(f=value), document_non_matches=True, document_field_comparisons=True
    )
    assert result["non_matches"] == []
    assert [row["match"] for row in result["field_comparisons"]] == [True]
