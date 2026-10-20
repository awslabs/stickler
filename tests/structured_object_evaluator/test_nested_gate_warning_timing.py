"""The class gate on a nested field warns only about pairs the matcher kept.

`compare()` is the cost function for a list of `StructuredModel` rows. The
Hungarian matcher scores every cell and keeps one per row, so a gate that warned
while the cost matrix was being filled reported mismatches for discarded pairs.
`warn_once` then spent its one message per field on that, and the genuine
mismatch that followed said nothing (#336). The element gate had the same defect
and was fixed in #319; see `TestTheGateDoesNotWarnAboutDiscardedPairs`.
"""

import warnings
from typing import Any, List, Optional, Union

import pytest
from pydantic import BaseModel

from stickler.algorithms.hungarian import HungarianMatcher
from stickler.comparators import StructuredModelComparator
from stickler.comparators.anls import ANLSStarComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.hungarian_helper import (
    HungarianHelper,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)


class Cat(BaseModel):
    name: Optional[str] = None


class Dog(BaseModel):
    name: Optional[str] = None


class Holder(StructuredModel):
    pet: Optional[Union[Cat, Dog]] = ComparableField(
        comparator=ANLSStarComparator(), default=None
    )


class Doc(StructuredModel):
    rows: Optional[List[Holder]] = None


class Row(StructuredModel):
    holder: Optional[Holder] = None


class Deep(StructuredModel):
    rows: Optional[List[Row]] = None


class Loose(StructuredModel):
    rows: Any = ComparableField(comparator=StructuredModelComparator())


class Tagged(StructuredModel):
    meta: Any = None
    key: Optional[str] = None


class TaggedDoc(StructuredModel):
    rows: Optional[List[Tagged]] = None


def _mismatch_warnings(caught):
    return [w for w in caught if "Different classes are scored" in str(w.message)]


def _mapping_warnings(caught):
    return [w for w in caught if "holds a mapping" in str(w.message)]


def _rows():
    return [Holder(pet=Cat(name="a")), Holder(pet=Dog(name="b"))]


def _swapped_class():
    """Two rows where the matcher must select a Cat-vs-Dog pair."""
    return (
        Doc(rows=[Holder(pet=Cat(name="xavier")), Holder(pet=Cat(name="yolanda"))]),
        Doc(rows=[Holder(pet=Dog(name="xavier")), Holder(pet=Cat(name="yolanda"))]),
    )


class TestADiscardedCellDoesNotWarn:
    def test_an_all_correct_list_does_not_report_a_mismatch(self):
        """The measurement in #336: tp=2, fd=0, and it warned anyway."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Doc(rows=_rows()).compare_with(
                Doc(rows=_rows()), include_confusion_matrix=True
            )
        node = result["confusion_matrix"]["fields"]["rows"]["overall"]
        assert (node["tp"], node["fd"]) == (2, 0)
        assert result["field_scores"]["rows"] == pytest.approx(1.0)
        assert not _mismatch_warnings(caught)

    def test_compare_does_not_report_one_either(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = Doc(rows=_rows()).compare(Doc(rows=_rows()))
        assert score == pytest.approx(1.0)
        assert not _mismatch_warnings(caught)

    def test_a_nested_row_one_level_deeper_does_not_either(self):
        """`compare()` reaches a nested model through `compare_with`, inside the matrix."""

        def deep():
            return Deep(rows=[Row(holder=h) for h in _rows()])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = deep().compare_with(deep())
        assert result["field_scores"]["rows"] == pytest.approx(1.0)
        assert not _mismatch_warnings(caught)


class TestASelectedMismatchStillWarns:
    def test_the_probe_does_not_spend_the_one_warning_the_field_gets(self):
        """The sequence from #336: a clean list, then the real Cat-vs-Dog pair.

        Only the second call is recorded. The defect was that the first call
        spent the warning, so the second had none left to give.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            Doc(rows=_rows()).compare_with(Doc(rows=_rows()))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Doc(rows=[Holder(pet=Cat(name="x"))]).compare_with(
                Doc(rows=[Holder(pet=Dog(name="x"))]), include_confusion_matrix=True
            )
        node = result["confusion_matrix"]["fields"]["rows"]["overall"]
        assert (node["tp"], node["fd"]) == (0, 1)
        assert _mismatch_warnings(caught)

    def test_a_mismatch_selected_from_a_full_matrix_warns(self):
        """Two rows go through the matrix rather than the one-pair fast path."""
        gt, pred = _swapped_class()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = gt.compare_with(pred)["field_scores"]["rows"]
        assert score == pytest.approx(0.5)
        assert _mismatch_warnings(caught)

    def test_compare_warns_for_it_too(self):
        gt, pred = _swapped_class()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = gt.compare(pred)
        assert score == pytest.approx(0.5)
        assert _mismatch_warnings(caught)


class TestAPairNothingScoresAgainStillWarns:
    """The flag is set only where the kept pairs are scored again.

    A bare matcher and an `Any` field's list return the cell scores as the
    result, so silencing the matrix there would lose the only warning.
    """

    def test_a_bare_matcher_warns_for_the_pair_it_keeps(self):
        gt, pred = _swapped_class()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = HungarianMatcher(StructuredModelComparator()).calculate_metrics(
                gt.rows, pred.rows
            )
        assert [(i, j) for i, j, _ in metrics["matched_pairs"]] == [(0, 0), (1, 1)]
        assert metrics["matched_pairs"][0][2] == pytest.approx(0.0)
        assert len(_mismatch_warnings(caught)) == 1

    def test_an_any_field_with_a_structured_comparator_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Loose(
                rows=["s", Holder(pet=Cat(name="a")), Holder(pet=Cat(name="b"))]
            ).compare_with(
                Loose(rows=["s", Holder(pet=Cat(name="a")), Holder(pet=Dog(name="b"))])
            )
        assert result["field_scores"]["rows"] == pytest.approx(2 / 3)
        assert len(_mismatch_warnings(caught)) == 1

    def test_a_mapping_the_comparator_cannot_score_warns_once(self):
        """The `can_score_object` half of the gate, through the same matrix."""

        def doc():
            return TaggedDoc(
                rows=[Tagged(meta={"a": 1}, key="x"), Tagged(meta={"b": 2}, key="y")]
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            doc().compare_with(doc())
        assert len(_mapping_warnings(caught)) == 1


class TestTheMatrixFlagIsScoped:
    @staticmethod
    def _helper(scorer):
        helper = HungarianHelper()
        helper.hungarian = HungarianMatcher(scorer)
        return helper

    def test_it_is_set_only_while_cells_are_scored(self):
        from stickler.algorithms.hungarian import filling_cost_matrix

        seen = []

        def scorer(a, b):
            seen.append(filling_cost_matrix())
            return 1.0 if a == b else 0.0

        assert not filling_cost_matrix()
        self._helper(scorer).get_complete_matching_info(["a", "b"], ["a", "b"])
        assert seen and all(seen)
        assert not filling_cost_matrix()

    def test_a_bare_matcher_does_not_set_it(self):
        from stickler.algorithms.hungarian import filling_cost_matrix

        seen = []

        def scorer(a, b):
            seen.append(filling_cost_matrix())
            return 1.0 if a == b else 0.0

        HungarianMatcher(scorer).match(["a", "b"], ["a", "b"])
        assert seen and not any(seen)

    def test_it_is_cleared_when_a_cell_raises(self):
        from stickler.algorithms.hungarian import filling_cost_matrix

        def scorer(a, b):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            self._helper(scorer).get_complete_matching_info(["a", "b"], ["a", "b"])
        assert not filling_cost_matrix()
