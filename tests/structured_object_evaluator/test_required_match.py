"""A required field that fails vetoes its object pair before list matching.

Without the option, one wrong field still lets an object match: with four
fields and one wrong, the object scores 0.75, above the default
``match_threshold`` of 0.7, so it counts as a true positive. A field marked
``required_match=True`` fails when its score is not at or above its threshold,
and the pair is then excluded from the candidates before the assignment. The
two objects it would have joined are left unmatched, one false negative and one
false positive, and the pair is never a false discovery.
"""

from typing import Annotated, Dict, List, Optional

import pytest

from stickler.comparators.base import BaseComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator import ComparableField, StructuredModel
from stickler.structured_object_evaluator.models.hungarian_helper import (
    HungarianHelper,
)


class ProximityComparator(BaseComparator):
    """Scores 1 minus the distance, so a test can place a score at a threshold."""

    def _compare(self, value1, value2) -> float:
        return max(0.0, 1.0 - abs(value1 - value2))


def exact(**kwargs):
    return ComparableField(comparator=ExactComparator(), threshold=1.0, **kwargs)


class Detection(StructuredModel):
    label: str = exact(required_match=True)
    x: Optional[float] = exact(required_match=True)
    y: float = exact(required_match=True)
    note: str = exact()


class LooseDetection(StructuredModel):
    """The same fields with no field required, as every model was before."""

    label: str = exact()
    x: Optional[float] = exact()
    y: float = exact()
    note: str = exact()


class ExplicitlyOptionalDetection(StructuredModel):
    label: str = exact(required_match=False)
    x: Optional[float] = exact(required_match=False)
    y: float = exact(required_match=False)
    note: str = exact(required_match=False)


class Frame(StructuredModel):
    detections: List[Detection] = ComparableField()


class LooseFrame(StructuredModel):
    detections: List[LooseDetection] = ComparableField()


class ExplicitlyOptionalFrame(StructuredModel):
    detections: List[ExplicitlyOptionalDetection] = ComparableField()


class IdentifiedFrame(StructuredModel):
    # the heavy id keeps the frame pair above match_threshold, so the matcher
    # descends into its detections
    frame_id: str = exact(weight=3.0)
    detections: List[Detection] = ComparableField()


class Scene(StructuredModel):
    frames: List[IdentifiedFrame] = ComparableField()


def detection(label="person", x=1.0, y=1.0, note="n"):
    return {"label": label, "x": x, "y": y, "note": note}


def compared(frame_cls, gt, pred):
    """The full compare_with result, with the confusion matrix."""
    return frame_cls.from_json({"detections": gt}).compare_with(
        frame_cls.from_json({"detections": pred}), include_confusion_matrix=True
    )


def list_counts(frame_cls, gt, pred):
    """The object counts of the detections list, from compare_with."""
    matrix = compared(frame_cls, gt, pred)["confusion_matrix"]
    overall = matrix["fields"]["detections"]["overall"]
    return {key: overall[key] for key in ("tp", "fa", "fd", "fp", "fn")}


UNMATCHED_PAIR = {"tp": 0, "fa": 1, "fd": 0, "fp": 1, "fn": 1}
MATCHED_PAIR = {"tp": 1, "fa": 0, "fd": 0, "fp": 0, "fn": 0}


def test_default_off_leaves_counts_and_score_unchanged():
    gt = [detection(), detection(label="car", x=5.0, y=5.0)]
    pred = [detection(y=9.0), detection(label="car", x=5.0, y=5.0)]

    loose = compared(LooseFrame, gt, pred)
    explicit = compared(ExplicitlyOptionalFrame, gt, pred)

    assert list_counts(LooseFrame, gt, pred)["tp"] == 2
    assert loose == explicit


def test_one_failing_axis_is_one_fn_and_one_fa_and_no_fd():
    assert list_counts(LooseFrame, [detection()], [detection(y=9.0)]) == MATCHED_PAIR
    assert list_counts(Frame, [detection()], [detection(y=9.0)]) == UNMATCHED_PAIR


def test_wrong_class_is_one_fn_and_one_fa():
    assert list_counts(Frame, [detection()], [detection(label="car")]) == (
        UNMATCHED_PAIR
    )


def test_wrong_class_among_several_objects_leaves_the_others_matched():
    gt = [detection(), detection(label="car", x=5.0, y=5.0)]
    pred = [detection(label="bike"), detection(label="car", x=5.0, y=5.0)]

    assert list_counts(Frame, gt, pred) == {
        "tp": 1,
        "fa": 1,
        "fd": 0,
        "fp": 1,
        "fn": 1,
    }


def test_unmatched_objects_are_not_fd_when_the_lists_differ_in_length():
    gt = [detection()]
    pred = [detection(y=9.0), detection(label="car", x=5.0, y=5.0)]

    assert list_counts(Frame, gt, pred) == {
        "tp": 0,
        "fa": 2,
        "fd": 0,
        "fp": 2,
        "fn": 1,
    }


def test_all_required_fields_passing_leaves_the_score_unchanged():
    gt = [detection(note="a")]
    pred = [detection(note="b")]

    assert compared(Frame, gt, pred) == compared(LooseFrame, gt, pred)


def test_required_field_null_on_one_side_fails():
    assert list_counts(Frame, [detection(x=None)], [detection()]) == UNMATCHED_PAIR
    assert list_counts(Frame, [detection()], [detection(x=None)]) == UNMATCHED_PAIR


def test_required_field_null_on_both_sides_passes():
    assert list_counts(Frame, [detection(x=None)], [detection(x=None)]) == (
        MATCHED_PAIR
    )


def test_a_threshold_stated_on_the_field_decides_the_veto():
    class Strict(StructuredModel):
        word: str = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.9, required_match=True
        )
        note: str = exact()

    class Lenient(StructuredModel):
        word: str = ComparableField(
            comparator=LevenshteinComparator(), threshold=0.5, required_match=True
        )
        note: str = exact()

    class StrictBatch(StructuredModel):
        detections: List[Strict] = ComparableField()

    class LenientBatch(StructuredModel):
        detections: List[Lenient] = ComparableField()

    gt = [{"word": "abcd", "note": "n"}]
    pred = [{"word": "abxd", "note": "n"}]

    assert list_counts(StrictBatch, gt, pred) == UNMATCHED_PAIR
    assert list_counts(LenientBatch, gt, pred) == MATCHED_PAIR


def test_a_threshold_named_on_the_comparator_decides_the_veto():
    class Named(StructuredModel):
        word: str = ComparableField(
            comparator=LevenshteinComparator(threshold=0.9), required_match=True
        )
        note: str = exact()

    class Batch(StructuredModel):
        detections: List[Named] = ComparableField()

    gt = [{"word": "abcd", "note": "n"}]
    pred = [{"word": "abxd", "note": "n"}]

    assert list_counts(Batch, gt, pred) == UNMATCHED_PAIR


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (0.5, MATCHED_PAIR),  # score is exactly the threshold
        (0.5 + 5e-11, MATCHED_PAIR),  # 5e-11 under it, inside the 1e-10 tolerance
        (0.5 + 1e-9, UNMATCHED_PAIR),  # 1e-9 under it, outside the tolerance
    ],
)
def test_partial_score_at_and_just_under_the_threshold(gap, expected):
    class Reading(StructuredModel):
        value: float = ComparableField(
            comparator=ProximityComparator(), threshold=0.5, required_match=True
        )
        note: str = exact()

    class Readings(StructuredModel):
        detections: List[Reading] = ComparableField()

    gt = [{"value": 1.0, "note": "n"}]
    pred = [{"value": 1.0 + gap, "note": "n"}]

    assert list_counts(Readings, gt, pred) == expected


def test_an_object_under_match_threshold_with_passing_required_fields_is_fd():
    # the required label passes, so there is no veto and the existing rule holds:
    # the object scores 0.25, under the 0.7 match_threshold, and is an fd
    class Wide(StructuredModel):
        label: str = exact(required_match=True)
        a: str = exact()
        b: str = exact()
        c: str = exact()

    class WideBatch(StructuredModel):
        detections: List[Wide] = ComparableField()

    counts = list_counts(
        WideBatch,
        [{"label": "p", "a": "1", "b": "1", "c": "1"}],
        [{"label": "p", "a": "2", "b": "2", "c": "2"}],
    )

    assert counts == {"tp": 0, "fa": 0, "fd": 1, "fp": 1, "fn": 0}


@pytest.mark.parametrize(
    "annotation",
    [List[str], Dict[str, str], Detection, List[Detection], Optional[List[str]]],
)
def test_a_non_primitive_required_field_raises_when_the_class_is_defined(annotation):
    with pytest.raises(ValueError, match="required_match"):

        class Invalid(StructuredModel):
            value: annotation = ComparableField(required_match=True)


def test_nested_list_applies_the_veto_in_the_inner_list():
    scene = Scene.from_json(
        {
            "frames": [
                {"frame_id": "f1", "detections": [detection(), detection(label="car")]}
            ]
        }
    )
    other = Scene.from_json(
        {
            "frames": [
                {
                    "frame_id": "f1",
                    "detections": [detection(y=9.0), detection(label="car")],
                }
            ]
        }
    )

    result = scene.compare_with(other, include_confusion_matrix=True)
    frames = result["confusion_matrix"]["fields"]["frames"]
    inner = frames["fields"]["detections"]["overall"]
    assert frames["overall"]["tp"] == 1

    assert (inner["tp"], inner["fn"], inner["fa"], inner["fd"]) == (1, 1, 1, 0)


def test_empty_lists_are_unchanged():
    assert list_counts(Frame, [], []) == list_counts(LooseFrame, [], [])
    assert list_counts(Frame, [], [detection()]) == list_counts(
        LooseFrame, [], [detection()]
    )
    assert list_counts(Frame, [detection()], []) == list_counts(
        LooseFrame, [detection()], []
    )


class Tagged(StructuredModel):
    # the required weight is under 1 so the vetoed pairs score above the
    # allowed ones, which makes the unrestricted assignment choose them
    kind: str = exact(weight=0.5, required_match=True)
    a: str = exact()
    b: str = exact()
    c: str = exact()
    d: str = exact()


class LooseTagged(StructuredModel):
    kind: str = exact(weight=0.5)
    a: str = exact()
    b: str = exact()
    c: str = exact()
    d: str = exact()


def tagged(kind, d):
    return {"kind": kind, "a": "1", "b": "1", "c": "1", "d": d}


def test_a_better_pairing_is_found_once_the_vetoed_pairs_are_excluded():
    gt = [tagged("K1", "1"), tagged("K2", "2")]
    pred = [tagged("K2", "1"), tagged("K1", "2")]

    required_gt = [Tagged(**item) for item in gt]
    required_pred = [Tagged(**item) for item in pred]
    loose_gt = [LooseTagged(**item) for item in gt]
    loose_pred = [LooseTagged(**item) for item in pred]

    # scores: a vetoed pair 4/4.5 = 0.889, an allowed pair 3.5/4.5 = 0.778
    unrestricted = HungarianHelper().get_complete_matching_info(loose_gt, loose_pred)
    excluded = HungarianHelper().get_complete_matching_info(required_gt, required_pred)

    # the unrestricted assignment takes the two vetoed pairs (sum 1.778 against
    # 1.556), so removing them afterwards would leave both objects unmatched
    assert unrestricted["assignments"] == [(0, 0), (1, 1)]
    # excluding them first lets the allowed pairs through, both above 0.7
    assert sorted(excluded["assignments"]) == [(0, 1), (1, 0)]

    class Batch(StructuredModel):
        detections: List[Tagged] = ComparableField()

    counts = list_counts(Batch, gt, pred)
    assert counts == {"tp": 2, "fa": 0, "fd": 0, "fp": 0, "fn": 0}
    post_hoc_counts = {"tp": 0, "fa": 2, "fd": 0, "fp": 2, "fn": 2}
    assert counts != post_hoc_counts


def test_inherited_required_field_still_applies_in_a_subclass():
    class Extended(Detection):
        extra: str = exact()

    class Batch(StructuredModel):
        detections: List[Extended] = ComparableField()

    gt = [{**detection(), "extra": "e"}]
    pred = [{**detection(y=9.0), "extra": "e"}]

    assert list_counts(Batch, gt, pred) == UNMATCHED_PAIR


@pytest.mark.parametrize(
    "annotation",
    [
        Annotated[str, "meta"],
        Optional[str],
        Optional[Annotated[str, "meta"]],
        Annotated[Optional[str], "meta"],
        Optional[Annotated[float, "meta"]],
    ],
)
def test_a_wrapped_primitive_is_accepted_as_a_required_field(annotation):
    class Valid(StructuredModel):
        value: annotation = ComparableField(required_match=True)

    assert Valid._required_match_fields == ("value",)


def test_a_model_inside_optional_annotated_still_raises():
    with pytest.raises(ValueError, match="required_match"):

        class Invalid(StructuredModel):
            value: Optional[Annotated[Detection, "meta"]] = ComparableField(
                required_match=True
            )


class Placed(StructuredModel):
    # position is the required field, scored by distance against 0.5
    match_threshold = 0.3
    position: float = ComparableField(
        comparator=ProximityComparator(), threshold=0.5, required_match=True
    )
    tag: str = exact()


class LoosePlaced(StructuredModel):
    match_threshold = 0.3
    position: float = ComparableField(comparator=ProximityComparator(), threshold=0.5)
    tag: str = exact()


class Shelf(StructuredModel):
    items: List[Placed] = ComparableField()


def test_a_veto_is_minimised_before_similarity_is_maximised():
    # A1,A2 against B1,B2 with only the pair (A2,B2) vetoed: its position is
    # 0.6 apart, the other three pairs are within 0.4
    gt = [
        {"position": 0.5, "tag": "p"},
        {"position": 0.0, "tag": "q"},
    ]
    pred = [
        {"position": 0.4, "tag": "p"},
        {"position": 0.6, "tag": "q"},
    ]
    loose_gt = [LoosePlaced(**item) for item in gt]
    loose_pred = [LoosePlaced(**item) for item in pred]
    gt_objects = [Placed(**item) for item in gt]
    pred_objects = [Placed(**item) for item in pred]

    info = HungarianHelper().get_complete_matching_info(gt_objects, pred_objects)
    unrestricted = HungarianHelper().get_complete_matching_info(loose_gt, loose_pred)

    # (A1,B1) plus (A2,B2) has the higher similarity sum, but (A2,B2) is
    # vetoed, so the assignment keeps two allowed pairs at lower scores
    assert sorted(unrestricted["assignments"]) == [(0, 0), (1, 1)]
    assert sorted(info["assignments"]) == [(0, 1), (1, 0)]

    matrix = Shelf.from_json({"items": gt}).compare_with(
        Shelf.from_json({"items": pred}), include_confusion_matrix=True
    )["confusion_matrix"]
    overall = matrix["fields"]["items"]["overall"]
    counts = {key: overall[key] for key in ("tp", "fa", "fd", "fp", "fn")}
    # both allowed pairs score above the 0.3 match_threshold, so each is a true
    # positive and nothing is left over; losing the vetoed pair costs no count
    assert counts == {"tp": 2, "fa": 0, "fd": 0, "fp": 0, "fn": 0}
