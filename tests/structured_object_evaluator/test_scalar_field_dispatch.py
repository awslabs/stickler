"""Scalar, tuple and set fields reach their comparator (#383).

``ComparisonDispatcher`` only routed ``str``/``int``/``float`` to the field's
comparator. A field typed ``date``, ``datetime``, ``time``, ``Decimal``,
``UUID`` or a plain ``Enum`` fell through to the mismatched-types case: it
scored 0.0 against an identical value, was counted as a false discovery, and
the comparator was never called, so declaring ``DateComparator`` could not
help. ``stickler.evaluate()`` was unaffected only because its shadow models
turn these values into strings first.
"""

import datetime
import decimal
import enum
import uuid
from typing import Any, FrozenSet, List, Optional, Set, Tuple, Union

import pytest

from stickler import ComparableField, StructuredModel
from stickler.comparators import (
    DateComparator,
    ExactComparator,
    LevenshteinComparator,
    NumericComparator,
)


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


@pytest.mark.parametrize(
    "gt, pred",
    [
        (datetime.date(2026, 9, 14), "2026-09-14"),
        (Color.RED, Size.RED),
        (uuid.UUID(int=1), decimal.Decimal(1)),
    ],
    ids=["date-vs-str", "two-enum-classes", "uuid-vs-decimal"],
)
def test_unrelated_types_are_still_a_mismatch(gt, pred):
    class M(StructuredModel):
        f: Any = ComparableField(comparator=LevenshteinComparator())

    result = M(f=gt).compare_with(M(f=pred), include_confusion_matrix=True)
    assert result["overall_score"] == pytest.approx(0.0)
    assert result["confusion_matrix"]["overall"]["fd"] == 1


def test_an_arbitrary_object_is_still_a_mismatch():
    """Two distinct objects have near-identical reprs; they must not match."""

    class Blob:
        pass

    class M(StructuredModel):
        model_config = {"arbitrary_types_allowed": True}
        f: Blob = ComparableField(comparator=LevenshteinComparator())

    result = M(f=Blob()).compare_with(M(f=Blob()))
    assert result["overall_score"] == pytest.approx(0.0)


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


class TestTuplesAndSets:
    """A ``Tuple`` or ``Set`` field also fell through to the mismatch case.

    Only ``list`` takes the list path, so these are scored as one value by the
    field's comparator, the way ``stickler.evaluate()`` scores them.
    """

    def _model(self, comparator=None):
        comparator = comparator or ExactComparator()

        class M(StructuredModel):
            t: Optional[Tuple[int, ...]] = ComparableField(comparator=comparator)
            s: Optional[Set[str]] = ComparableField(comparator=comparator)
            fs: Optional[FrozenSet[str]] = ComparableField(comparator=comparator)

        return M

    def test_identical_values_are_true_positives(self):
        M = self._model()
        a = M(t=(1, 2), s={"a", "b"}, fs=frozenset({"x"}))
        result = a.compare_with(
            M(t=(1, 2), s={"b", "a"}, fs=frozenset({"x"})),
            include_confusion_matrix=True,
        )
        assert result["field_scores"] == {"t": 1.0, "s": 1.0, "fs": 1.0}
        assert result["confusion_matrix"]["overall"]["tp"] == 3

    def test_different_values_are_false_discoveries(self):
        M = self._model()
        result = M(t=(1, 2), s={"a"}, fs=frozenset({"x"})).compare_with(
            M(t=(2, 1), s={"b"}, fs=frozenset({"y"})), include_confusion_matrix=True
        )
        assert result["field_scores"] == {"t": 0.0, "s": 0.0, "fs": 0.0}
        assert result["confusion_matrix"]["overall"]["fd"] == 3

    def test_set_order_cannot_change_a_string_comparators_score(self):
        """Levenshtein reads str(value); a set is sorted before it gets there."""
        M = self._model(LevenshteinComparator())
        a = M(s={f"item{i}" for i in range(20)})
        b = M(s={f"item{i}" for i in reversed(range(20))})
        assert a.compare_with(b)["field_scores"]["s"] == pytest.approx(1.0)
        assert a.compare(b) == pytest.approx(1.0)

    def test_empty_tuples_and_sets_are_absent(self):
        M = self._model()
        result = M(t=(), s=set(), fs=None).compare_with(
            M(t=None, s=set(), fs=frozenset()), include_confusion_matrix=True
        )
        counts = result["confusion_matrix"]["overall"]
        assert (counts["tn"], counts["fd"]) == (3, 0)

    def test_empty_against_populated_is_a_false_alarm(self):
        M = self._model()
        result = M(t=()).compare_with(M(t=(1,)), include_confusion_matrix=True)
        assert result["confusion_matrix"]["overall"]["fa"] == 1

    def test_reports_and_compare_agree_with_compare_with(self):
        M = self._model()
        a = M(t=(1, 2), s={"a"}, fs=frozenset({"x"}))
        b = M(t=(1, 2), s={"a"}, fs=frozenset({"x"}))
        result = a.compare_with(
            b, document_non_matches=True, document_field_comparisons=True
        )
        assert result["non_matches"] == []
        assert all(row["match"] for row in result["field_comparisons"])
        assert a.compare(b) == pytest.approx(1.0)

    def test_list_elements_with_tuple_and_set_fields_pair_up(self):
        M = self._model()

        class Doc(StructuredModel):
            items: List[M] = ComparableField()

        x = M(t=(1, 2), s={"a"})
        y = M(t=(9, 9), s={"q"})
        result = Doc(items=[x, y]).compare_with(
            Doc(items=[M(t=(9, 9), s={"q"}), M(t=(1, 2), s={"a"})]),
            include_confusion_matrix=True,
        )
        assert result["overall_score"] == pytest.approx(1.0)
        assert result["confusion_matrix"]["overall"]["tp"] == 2

    def test_a_tuple_and_a_list_are_still_a_mismatch(self):
        class M(StructuredModel):
            f: Any = ComparableField(comparator=ExactComparator())

        assert M(f=(1, 2)).compare_with(M(f={1, 2}))["overall_score"] == 0.0
