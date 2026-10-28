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
from stickler.structured_object_evaluator.models import hungarian_helper
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.hungarian_helper import (
    HungarianHelper,
    defer_pair_warnings,
    deferring_pair_warnings,
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


class Labelled(StructuredModel):
    label: Optional[str] = None
    pet: Optional[Union[Cat, Dog]] = ComparableField(
        comparator=ANLSStarComparator(), default=None
    )


class LabelledDoc(StructuredModel):
    rows: Optional[List[Labelled]] = None


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


def _only_kept_pairs_cross_classes():
    """Rows the matcher pairs by ``label``, where only the kept pairs cross classes.

    The kept pairs are (0, 0) and (1, 1), both Cat-vs-Dog. Every discarded cell
    is Cat-vs-Cat or Dog-vs-Dog, so a class-mismatch warning can only come from
    a kept pair. A fixture with a Cat-vs-Dog discarded cell passes on that
    cell's warning even if the kept pair never warns.
    """
    gt = [
        Labelled(label="xavier", pet=Cat(name="a")),
        Labelled(label="yolanda", pet=Dog(name="b")),
    ]
    pred = [
        Labelled(label="xavier", pet=Dog(name="a")),
        Labelled(label="yolanda", pet=Cat(name="b")),
    ]
    discarded = [(0, 1), (1, 0)]
    assert all(type(gt[i].pet) is type(pred[j].pet) for i, j in discarded)
    return gt, pred


@pytest.fixture(autouse=True)
def _pair_warnings_are_not_deferred_around_each_test():
    assert not deferring_pair_warnings()
    yield
    assert not deferring_pair_warnings()


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
        gt, pred = _only_kept_pairs_cross_classes()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = LabelledDoc(rows=gt).compare_with(LabelledDoc(rows=pred))
        assert result["field_scores"]["rows"] == pytest.approx(0.5)
        assert len(_mismatch_warnings(caught)) == 1

    def test_compare_warns_for_it_too(self):
        gt, pred = _only_kept_pairs_cross_classes()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = LabelledDoc(rows=gt).compare(LabelledDoc(rows=pred))
        assert score == pytest.approx(0.5)
        assert len(_mismatch_warnings(caught)) == 1


class TestAPairNothingScoresAgainStillWarns:
    """The flag is set only where the kept pairs are scored again.

    A bare matcher and an `Any` field's list return the cell scores as the
    result, so silencing the matrix there would lose the only warning.
    """

    def test_a_bare_matcher_warns_for_the_pair_it_keeps(self):
        gt, pred = _only_kept_pairs_cross_classes()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = HungarianMatcher(StructuredModelComparator()).calculate_metrics(
                gt, pred
            )
        assert [(i, j) for i, j, _ in metrics["matched_pairs"]] == [(0, 0), (1, 1)]
        assert [s for _, _, s in metrics["matched_pairs"]] == pytest.approx([0.5, 0.5])
        assert len(_mismatch_warnings(caught)) == 1

    def test_an_any_field_with_a_structured_comparator_warns(self):
        """The list starts with a scalar, so it is not matched by `HungarianHelper`."""
        gt, pred = _only_kept_pairs_cross_classes()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Loose(rows=["s", *gt]).compare_with(Loose(rows=["s", *pred]))
        assert result["field_scores"]["rows"] == pytest.approx(2 / 3)
        assert len(_mismatch_warnings(caught)) == 1

    def test_a_mapping_the_comparator_cannot_score_still_warns(self):
        """Deferring the mapping warning to the kept pairs did not silence it.

        This guards against over-silencing only. It does not reproduce #336: the
        mapping warning is keyed per field, and every cell here holds a mapping,
        so a discarded cell and a kept pair spend the same message. It passes
        with or without the deferral, and fails only if nothing warns.
        """

        def doc():
            return TaggedDoc(
                rows=[Tagged(meta={"a": 1}, key="x"), Tagged(meta={"b": 2}, key="y")]
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            doc().compare_with(doc())
        assert len(_mapping_warnings(caught)) == 1


class TestPairWarningDeferralIsScoped:
    @staticmethod
    def _helper(scorer):
        helper = HungarianHelper()
        helper.hungarian = HungarianMatcher(scorer)
        return helper

    def test_it_is_set_while_the_helper_scores_cells_and_cleared_after(self):
        seen = []

        def scorer(a, b):
            seen.append(deferring_pair_warnings())
            return 1.0 if a == b else 0.0

        assert not deferring_pair_warnings()
        self._helper(scorer).get_complete_matching_info(["a", "b"], ["a", "b"])
        assert seen and all(seen)
        assert not deferring_pair_warnings()

    def test_a_bare_matcher_does_not_set_it(self):
        seen = []

        def scorer(a, b):
            seen.append(deferring_pair_warnings())
            return 1.0 if a == b else 0.0

        HungarianMatcher(scorer).match(["a", "b"], ["a", "b"])
        assert seen and not any(seen)

    @pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
    def test_it_is_cleared_when_a_cell_raises(self, error):
        def scorer(a, b):
            raise error("boom")

        with pytest.raises(error):
            self._helper(scorer).get_complete_matching_info(["a", "b"], ["a", "b"])
        assert not deferring_pair_warnings()

    def test_the_context_manager_restores_the_previous_value(self):
        with defer_pair_warnings():
            with defer_pair_warnings():
                assert deferring_pair_warnings()
            assert deferring_pair_warnings()
        assert not deferring_pair_warnings()

    def test_an_interrupt_raised_by_the_set_itself_is_restored(self, monkeypatch):
        """A pending signal is raised when ``ContextVar.set`` returns, after the
        value has changed but before a token could be stored."""
        real = hungarian_helper._DEFER_PAIR_WARNINGS

        class InterruptedOnSet:
            get = staticmethod(real.get)

            @staticmethod
            def set(value):
                real.set(value)
                if value:
                    raise KeyboardInterrupt

        monkeypatch.setattr(hungarian_helper, "_DEFER_PAIR_WARNINGS", InterruptedOnSet)
        with pytest.raises(KeyboardInterrupt):
            with defer_pair_warnings():
                pass
        assert not real.get()
