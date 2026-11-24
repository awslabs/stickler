"""`clip_under_threshold` on a `List[StructuredModel]` field (#415).

A pair below the element class's `match_threshold` is an FD. With
`clip_under_threshold=True` it contributes 0.0 to the list's score; with
`False` it keeps its partial score. The default is `False`, as on dict and
plain-model fields and lists of them, so a list that does not set the flag
scores as it did before. Counts do not move, and the list's own
`raw_similarity_score` stays unclipped at the list level (#437).
"""

from typing import Annotated, Any, List, Optional, Sequence, Union

import pytest
from pydantic import BaseModel

from stickler import ComparableField, StructuredModel, evaluate
from stickler.comparators import ExactComparator, LevenshteinComparator
from stickler.structured_object_evaluator.models.configuration_helper import (
    ConfigurationHelper,
)


class LineItem(StructuredModel):
    match_threshold = 0.8
    sku: str = ComparableField(comparator=ExactComparator())
    amount: float = ComparableField(comparator=ExactComparator())
    qty: int = ComparableField(comparator=ExactComparator())


def _invoice(clip):
    """The repro from the issue, with the vendor and the list set alike."""

    class Invoice(StructuredModel):
        vendor: str = ComparableField(
            comparator=LevenshteinComparator(),
            threshold=0.8,
            clip_under_threshold=clip,
        )
        line_items: List[LineItem] = ComparableField(clip_under_threshold=clip)

    return Invoice


def _repro_pair(cls):
    gt = cls(vendor="Initech", line_items=[LineItem(sku="A-100", amount=40, qty=1)])
    pred = cls(vendor="Initrode", line_items=[LineItem(sku="A-100", amount=45, qty=2)])
    return gt, pred


@pytest.mark.parametrize(
    "clip, vendor, line_items, overall",
    [(True, 0.0, 0.0, 0.0), (False, 0.5, 1 / 3, 0.5 * (0.5 + 1 / 3))],
)
def test_the_issue_repro_honours_the_flag(clip, vendor, line_items, overall):
    gt, pred = _repro_pair(_invoice(clip))
    result = gt.compare_with(pred)

    assert result["field_scores"]["vendor"] == pytest.approx(vendor)
    assert result["field_scores"]["line_items"] == pytest.approx(line_items)
    assert result["overall_score"] == pytest.approx(overall)


class DefaultList(StructuredModel):
    line_items: List[LineItem] = ComparableField()


class BareList(StructuredModel):
    line_items: List[LineItem]


class OptionalList(StructuredModel):
    line_items: Optional[List[LineItem]] = ComparableField(default=None)


class Pep604List(StructuredModel):
    line_items: List[LineItem] | None = ComparableField(default=None)


@pytest.mark.parametrize("cls", [DefaultList, BareList, OptionalList, Pep604List])
def test_a_list_that_does_not_set_the_flag_keeps_partial_credit(cls):
    gt = cls(line_items=[LineItem(sku="A-100", amount=40, qty=1)])
    pred = cls(line_items=[LineItem(sku="A-100", amount=45, qty=2)])

    assert gt.compare_with(pred)["field_scores"]["line_items"] == pytest.approx(1 / 3)
    assert cls._get_comparison_info("line_items").clip_under_threshold is False


class ClipOptional(StructuredModel):
    line_items: Optional[List[LineItem]] = ComparableField(
        default=None, clip_under_threshold=True
    )


def test_an_explicit_true_is_honoured_on_an_optional_list():
    gt = ClipOptional(line_items=[LineItem(sku="A-100", amount=40, qty=1)])
    pred = ClipOptional(line_items=[LineItem(sku="A-100", amount=45, qty=2)])

    assert gt.compare_with(pred)["field_scores"]["line_items"] == 0.0
    assert ClipOptional._get_comparison_info("line_items").clip_under_threshold


class Row(StructuredModel):
    """Seven exact fields, so one wrong field scores 6/7, above 0.8 (a TP)."""

    match_threshold = 0.8
    a: str = ComparableField(comparator=ExactComparator())
    b: str = ComparableField(comparator=ExactComparator())
    c: str = ComparableField(comparator=ExactComparator())
    d: str = ComparableField(comparator=ExactComparator())
    e: str = ComparableField(comparator=ExactComparator())
    f: str = ComparableField(comparator=ExactComparator())
    g: str = ComparableField(comparator=ExactComparator())


def _row(**changes):
    values = dict(a="a", b="b", c="c", d="d", e="e", f="f", g="g")
    values.update(changes)
    return Row(**values)


class Rows(StructuredModel):
    rows: List[Row] = ComparableField(clip_under_threshold=True)


class RowsKeep(StructuredModel):
    rows: List[Row] = ComparableField(clip_under_threshold=False)


def _node(gt, pred):
    result = gt.compare_with(pred, include_confusion_matrix=True)
    return result, result["confusion_matrix"]["fields"]["rows"]


def test_clipping_zeroes_only_the_rejected_pair():
    """One TP at 6/7 and one FD at 4/7 (below 0.8): only the FD is zeroed."""
    gt_rows = [_row(), _row(a="x1")]
    pred_rows = [_row(g="z"), _row(a="y1", b="q", c="q")]

    clipped, clipped_node = _node(Rows(rows=gt_rows), Rows(rows=pred_rows))
    kept, kept_node = _node(RowsKeep(rows=gt_rows), RowsKeep(rows=pred_rows))

    assert kept["field_scores"]["rows"] == pytest.approx((6 / 7 + 4 / 7) / 2)
    assert clipped["field_scores"]["rows"] == pytest.approx((6 / 7 + 0.0) / 2)
    # Counts are settled before the score and do not depend on the flag.
    for node in (clipped_node, kept_node):
        assert node["overall"]["tp"] == 1
        assert node["overall"]["fd"] == 1


def test_a_matched_pair_keeps_its_score_with_clipping_on():
    clipped, _ = _node(Rows(rows=[_row()]), Rows(rows=[_row(g="z")]))

    assert clipped["field_scores"]["rows"] == pytest.approx(6 / 7)


def test_coverage_scaling_still_applies_when_clipping():
    """An unmatched GT row still counts in `max(n_gt, n_pred)`."""
    gt = Rows(
        rows=[_row(), _row(a="x1", b="x2", c="x3", d="x4", e="x5", f="x6", g="x7")]
    )
    pred = Rows(rows=[_row(g="z")])

    score = gt.compare_with(pred)["field_scores"]["rows"]

    assert score == pytest.approx((6 / 7) / 2)


def test_raw_similarity_stays_unclipped_at_the_list_level():
    gt_rows = [_row(a="x1")]
    pred_rows = [_row(a="y1", b="q", c="q")]

    _, clipped_node = _node(Rows(rows=gt_rows), Rows(rows=pred_rows))
    _, kept_node = _node(RowsKeep(rows=gt_rows), RowsKeep(rows=pred_rows))

    assert clipped_node["threshold_applied_score"] == 0.0
    assert clipped_node["raw_similarity_score"] == pytest.approx(4 / 7)
    assert kept_node["raw_similarity_score"] == pytest.approx(4 / 7)


def _schema(clip=None):
    items = {
        "type": "object",
        "x-aws-stickler-match-threshold": 0.8,
        "properties": {
            "sku": {"type": "string", "x-aws-stickler-comparator": "ExactComparator"},
            "amount": {
                "type": "number",
                "x-aws-stickler-comparator": "ExactComparator",
            },
            "qty": {"type": "integer", "x-aws-stickler-comparator": "ExactComparator"},
        },
    }
    line_items = {"type": "array", "items": items}
    if clip is not None:
        line_items["x-aws-stickler-clip-under-threshold"] = clip
    return {
        "type": "object",
        "title": "Inv",
        "properties": {"line_items": line_items},
    }


_ITEMS_GT = {"line_items": [{"sku": "A-100", "amount": 40, "qty": 1}]}
_ITEMS_PRED = {"line_items": [{"sku": "A-100", "amount": 45, "qty": 2}]}


@pytest.mark.parametrize("clip, expected", [(None, 1 / 3), (False, 1 / 3), (True, 0.0)])
def test_json_schema_array_of_objects(clip, expected):
    model = StructuredModel.from_json_schema(_schema(clip))
    score = model(**_ITEMS_GT).compare_with(model(**_ITEMS_PRED))["field_scores"]

    assert score["line_items"] == pytest.approx(expected)


@pytest.mark.parametrize("cls, expected", [(DefaultList, 1 / 3), (ClipOptional, 0.0)])
def test_a_schema_round_trip_keeps_the_score(cls, expected):
    rebuilt = StructuredModel.from_json_schema(cls.to_json_schema())
    gt = rebuilt(line_items=[{"sku": "A-100", "amount": 40, "qty": 1}])
    pred = rebuilt(line_items=[{"sku": "A-100", "amount": 45, "qty": 2}])

    assert gt.compare_with(pred)["field_scores"]["line_items"] == pytest.approx(
        expected
    )


@pytest.mark.parametrize("cls, expected", [(DefaultList, False), (ClipOptional, True)])
def test_the_exported_clip_is_the_resolved_clip(cls, expected):
    prop = cls.to_json_schema()["properties"]["line_items"]

    assert prop["x-aws-stickler-clip-under-threshold"] is expected


def test_a_shared_comparable_field_is_not_rewritten_for_its_sibling():
    """Amending the list field's clip must copy, not mutate, a shared field."""
    shared = ComparableField(weight=2.0)

    class Scalar(StructuredModel):
        name: str = shared

    class Listed(StructuredModel):
        line_items: List[LineItem] = shared

    assert Scalar._get_comparison_info("name").clip_under_threshold is True
    assert Listed._get_comparison_info("line_items").clip_under_threshold is False


class PlainItem(BaseModel):
    sku: str
    amount: float
    qty: int


class PlainInvoice(BaseModel):
    line_items: List[PlainItem]


def test_zero_config_evaluate_is_unchanged():
    gt = PlainInvoice(line_items=[PlainItem(sku="A-100", amount=40, qty=1)])
    pred = PlainInvoice(line_items=[PlainItem(sku="A-100", amount=45, qty=2)])
    before = evaluate(gt, pred)

    assert before.field_scores["line_items"] > 0.0


def test_object_grade_helper_is_unchanged_for_a_model_list():
    """`List[StructuredModel]` is not object-grade: no comparator is substituted."""
    info = DefaultList.model_fields["line_items"]

    assert not ConfigurationHelper.is_object_grade_annotation(info)


class OtherItem(StructuredModel):
    match_threshold = 0.8
    sku: str = ComparableField(comparator=ExactComparator())
    amount: float = ComparableField(comparator=ExactComparator())
    qty: int = ComparableField(comparator=ExactComparator())


class UnionList(StructuredModel):
    line_items: List[Union[LineItem, OtherItem]] = ComparableField()


class UnionListClip(StructuredModel):
    line_items: List[Union[LineItem, OtherItem]] = ComparableField(
        clip_under_threshold=True
    )


class OptionalElementList(StructuredModel):
    line_items: List[Optional[LineItem]] = ComparableField()


class AnyList(StructuredModel):
    line_items: List[Any] = ComparableField()


class AnyListClip(StructuredModel):
    line_items: List[Any] = ComparableField(clip_under_threshold=True)


class SequenceList(StructuredModel):
    line_items: Sequence[LineItem] = ComparableField()


class AnnotatedElementList(StructuredModel):
    line_items: List[Annotated[LineItem, "m"]] = ComparableField()


class OptionalAnnotatedElementList(StructuredModel):
    line_items: List[Optional[Annotated[LineItem, "m"]]] = ComparableField()


class AnnotatedElementOptionalList(StructuredModel):
    line_items: Optional[List[Annotated[LineItem, "m"]]] = ComparableField(default=None)


ANNOTATED_SPELLINGS = [
    AnnotatedElementList,
    OptionalAnnotatedElementList,
    AnnotatedElementOptionalList,
]


@pytest.mark.parametrize(
    "cls, expected",
    [
        (UnionList, 1 / 3),
        (OptionalElementList, 1 / 3),
        (AnyList, 1 / 3),
        (SequenceList, 1 / 3),
        (AnnotatedElementList, 1 / 3),
        (OptionalAnnotatedElementList, 1 / 3),
        (AnnotatedElementOptionalList, 1 / 3),
        (UnionListClip, 0.0),
        (AnyListClip, 0.0),
    ],
)
def test_every_spelling_of_a_list_of_models(cls, expected):
    """Unset keeps partial credit; an explicit True zeroes the FD pair."""
    gt = cls(line_items=[LineItem(sku="A-100", amount=40, qty=1)])
    pred = cls(line_items=[LineItem(sku="A-100", amount=45, qty=2)])

    score = gt.compare_with(pred)["field_scores"]["line_items"]

    assert score == pytest.approx(expected)


@pytest.mark.parametrize("cls", [OptionalElementList, *ANNOTATED_SPELLINGS])
def test_an_optional_element_reports_the_default_it_scores_with(cls):
    assert cls._get_comparison_info("line_items").clip_under_threshold is False


@pytest.mark.parametrize("cls", ANNOTATED_SPELLINGS)
def test_an_annotated_element_exports_and_round_trips_its_default(cls):
    """Export strips `Annotated` from the element, so the clip default must
    recognise it too, or the exported `true` turns clipping on."""
    schema = cls.to_json_schema()
    config = cls.to_stickler_config()

    assert (
        schema["properties"]["line_items"]["x-aws-stickler-clip-under-threshold"]
        is False
    )
    assert config["fields"]["line_items"]["clip_under_threshold"] is False
    assert _imported_scores(schema, config) == pytest.approx([1 / 3, 1 / 3])


def test_a_model_config_list_honours_an_explicit_clip():
    config = {
        "model_name": "Inv",
        "fields": {
            "line_items": {
                "type": "list_structured_model",
                "clip_under_threshold": True,
                "fields": {
                    "sku": {"type": "str", "comparator": "ExactComparator"},
                    "amount": {"type": "float", "comparator": "ExactComparator"},
                    "qty": {"type": "int", "comparator": "ExactComparator"},
                },
                "match_threshold": 0.8,
            }
        },
    }
    model = StructuredModel.model_from_json(config)
    score = model(**_ITEMS_GT).compare_with(model(**_ITEMS_PRED))["field_scores"]

    assert score["line_items"] == 0.0
    config["fields"]["line_items"].pop("clip_under_threshold")
    model = StructuredModel.model_from_json(config)
    score = model(**_ITEMS_GT).compare_with(model(**_ITEMS_PRED))["field_scores"]

    assert score["line_items"] == pytest.approx(1 / 3)


class LateHolder(StructuredModel):
    line_items: List["LateItem"] = ComparableField()


class LateItem(StructuredModel):
    match_threshold = 0.8
    sku: str = ComparableField(comparator=ExactComparator())
    amount: float = ComparableField(comparator=ExactComparator())
    qty: int = ComparableField(comparator=ExactComparator())


LateHolder.model_rebuild()


def test_a_forward_referenced_list_reports_the_default_it_scores_with():
    """The element class is not defined yet when the holder is, so the
    class-definition amendment cannot see the list; the read path must."""
    gt = LateHolder(line_items=[LateItem(sku="A-100", amount=40, qty=1)])
    pred = LateHolder(line_items=[LateItem(sku="A-100", amount=45, qty=2)])

    assert LateHolder._get_comparison_info("line_items").clip_under_threshold is False
    assert gt.compare_with(pred)["field_scores"]["line_items"] == pytest.approx(1 / 3)


def test_a_forward_referenced_list_exports_and_round_trips_its_default():
    """The raw attribute could not be amended at class definition, so export
    must write the resolved clip, or a round trip turns clipping on."""
    schema = LateHolder.to_json_schema()
    config = LateHolder.to_stickler_config()

    assert (
        schema["properties"]["line_items"]["x-aws-stickler-clip-under-threshold"]
        is False
    )
    assert config["fields"]["line_items"]["clip_under_threshold"] is False
    assert _imported_scores(schema, config) == pytest.approx([1 / 3, 1 / 3])


def _exported_by_1_0_x():
    """What 1.0.x wrote for `DefaultList`: `true` on an array of models that
    never set the flag (measured on v1.0.2)."""
    schema = DefaultList.to_json_schema()
    schema["properties"]["line_items"]["x-aws-stickler-clip-under-threshold"] = True
    config = DefaultList.to_stickler_config()
    config["fields"]["line_items"]["clip_under_threshold"] = True
    return schema, config


def _imported_scores(schema, config):
    scores = []
    for model in (
        StructuredModel.from_json_schema(schema),
        StructuredModel.model_from_json(config),
    ):
        gt, pred = model(**_ITEMS_GT), model(**_ITEMS_PRED)
        scores.append(gt.compare_with(pred)["field_scores"]["line_items"])
    return scores


def test_a_1_0_x_export_now_clips():
    """1.0.x exported `true` for an unset list, and that `true` now takes effect."""
    schema, config = _exported_by_1_0_x()

    assert _imported_scores(schema, config) == [0.0, 0.0]


def test_a_1_0_x_export_with_the_key_removed_keeps_its_1_0_x_score():
    schema, config = _exported_by_1_0_x()
    del schema["properties"]["line_items"]["x-aws-stickler-clip-under-threshold"]
    del config["fields"]["line_items"]["clip_under_threshold"]

    assert _imported_scores(schema, config) == pytest.approx([1 / 3, 1 / 3])


def test_a_1_0_x_export_with_false_keeps_its_1_0_x_score():
    schema, config = _exported_by_1_0_x()
    schema["properties"]["line_items"]["x-aws-stickler-clip-under-threshold"] = False
    config["fields"]["line_items"]["clip_under_threshold"] = False

    assert _imported_scores(schema, config) == pytest.approx([1 / 3, 1 / 3])


@pytest.mark.parametrize("cls", [UnionList, AnyList, SequenceList])
def test_another_list_spelling_still_reports_its_unset_clip_as_true(cls):
    """Pre-existing and left alone: the score keeps partial credit, but the
    reported setting is the global default."""
    assert cls._get_comparison_info("line_items").clip_under_threshold is True


class Person(StructuredModel):
    match_threshold = 0.8
    name: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.95)
    id: str = ComparableField(comparator=ExactComparator())


class People(StructuredModel):
    people: List[Person] = ComparableField(clip_under_threshold=True)


def test_a_matched_pair_keeps_a_score_below_the_match_threshold():
    """Zeroing follows the pairing verdict, not the item's own score.

    "Jonathan"/"1" against "Jonathon"/"1" pairs at 0.9375 (a TP at 0.8), but
    the item's own `compare_with` clips `name` (0.875 < 0.95) and scores 0.5.
    With clipping on, the TP keeps that 0.5 rather than being zeroed.
    """
    gt = People(people=[Person(name="Jonathan", id="1"), Person(name="Ann", id="2")])
    pred = People(people=[Person(name="Jonathon", id="1"), Person(name="Zed", id="9")])
    result = gt.compare_with(pred, include_confusion_matrix=True)
    node = result["confusion_matrix"]["fields"]["people"]

    assert result["field_scores"]["people"] == pytest.approx(0.25)
    assert node["overall"]["tp"] == 1
    assert node["overall"]["fd"] == 1


class EarlyHolder(StructuredModel):
    line_items: List["EarlyItem"] = ComparableField()


# Read before `EarlyItem` exists, so the per-class cache first sees the
# unresolved annotation. It must not keep that answer once pydantic resolves it.
_READ_BEFORE_RESOLUTION = EarlyHolder._get_comparison_info("line_items")


class EarlyItem(StructuredModel):
    match_threshold = 0.8
    sku: str = ComparableField(comparator=ExactComparator())
    amount: float = ComparableField(comparator=ExactComparator())
    qty: int = ComparableField(comparator=ExactComparator())


def test_a_read_before_the_element_is_defined_does_not_stick():
    """Instantiating resolves the forward reference implicitly; the setting,
    the export and both round trips then follow the resolved annotation."""
    gt = EarlyHolder(line_items=[{"sku": "A-100", "amount": 40, "qty": 1}])
    pred = EarlyHolder(line_items=[{"sku": "A-100", "amount": 45, "qty": 2}])
    schema = EarlyHolder.to_json_schema()
    config = EarlyHolder.to_stickler_config()

    assert _READ_BEFORE_RESOLUTION.clip_under_threshold is True
    assert EarlyHolder._get_comparison_info("line_items").clip_under_threshold is False
    assert gt.compare_with(pred)["field_scores"]["line_items"] == pytest.approx(1 / 3)
    assert (
        schema["properties"]["line_items"]["x-aws-stickler-clip-under-threshold"]
        is False
    )
    assert config["fields"]["line_items"]["clip_under_threshold"] is False
    assert _imported_scores(schema, config) == pytest.approx([1 / 3, 1 / 3])


def _run_isolated(code):
    """Run `code` in a fresh interpreter, so module-level state such as the
    pending-class registry starts empty and no other test touches it."""
    import subprocess
    import sys
    import textwrap

    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.split()


def test_a_settled_class_is_no_longer_pending():
    """Once the class completes and a recorded field is read, the class leaves
    the registry and the per-read check turns off."""
    output = _run_isolated(
        """
        from typing import List
        from stickler import ComparableField, StructuredModel
        from stickler.structured_object_evaluator.models import configuration_helper as ch

        class Holder(StructuredModel):
            line_items: List["Item"] = ComparableField()

        class Item(StructuredModel):
            sku: str = ComparableField()

        print(Holder in ch._DEFERRED_CLIP_CLASSES, ch._DEFERRED_CLIP_SEEN)
        finalizer = ch._DEFERRED_CLIP_CLASSES[Holder]
        print(finalizer.atexit)
        Holder.model_rebuild()
        Holder._get_comparison_info("line_items")
        print(Holder in ch._DEFERRED_CLIP_CLASSES, ch._DEFERRED_CLIP_SEEN)
        print(Holder._stickler_deferred_clip_fields == frozenset())
        print(finalizer.alive)
        """
    )

    assert output == ["True", "True", "False", "False", "False", "True", "False"]


def test_a_collected_pending_class_stops_the_per_read_check():
    """A class whose forward reference never resolves stays pending until it
    is garbage collected; then nothing is pending and the flag goes off."""
    output = _run_isolated(
        """
        import gc, weakref
        from typing import List
        from stickler import ComparableField, StructuredModel
        from stickler.structured_object_evaluator.models import configuration_helper as ch

        def define():
            class NeverResolved(StructuredModel):
                line_items: List["UndefinedItem"] = ComparableField()
            return NeverResolved

        pending = define()
        ref = weakref.ref(pending)
        print(ch._DEFERRED_CLIP_SEEN)
        del pending
        gc.collect()
        print(ref() is None, len(list(ch._DEFERRED_CLIP_CLASSES.keys())), ch._DEFERRED_CLIP_SEEN)
        """
    )

    assert output == ["True", "True", "0", "False"]


def test_a_collection_while_the_lock_is_held_does_not_deadlock():
    """The collection callback takes the lock, and a collection can run on a
    thread that already holds it (inside recording or settling)."""
    output = _run_isolated(
        """
        import gc, threading
        from typing import List
        from stickler import ComparableField, StructuredModel
        from stickler.structured_object_evaluator.models import configuration_helper as ch

        def define():
            class Garbage(StructuredModel):
                line_items: List["UndefinedItem"] = ComparableField()

        def collect_holding_the_lock():
            with ch._DEFERRED_CLIP_LOCK:
                define()
                gc.collect()

        worker = threading.Thread(target=collect_holding_the_lock, daemon=True)
        worker.start()
        worker.join(timeout=20)
        print(worker.is_alive(), ch._DEFERRED_CLIP_SEEN)
        """
    )

    assert output == ["False", "False"]


def test_concurrent_first_reads_settle_once_and_agree(monkeypatch):
    """Eight threads read a forward-referenced list at once: every read
    reports the default, and the field is amended exactly once."""
    import threading
    import time

    from stickler.structured_object_evaluator.models import structured_model

    class RacingHolder(StructuredModel):
        line_items: List["RacingItem"] = ComparableField()

    class RacingItem(StructuredModel):
        match_threshold = 0.8
        sku: str = ComparableField(comparator=ExactComparator())

    RacingHolder.model_rebuild()
    amend = structured_model._amend_clip_default
    amendments = []

    def slow_amend(*args):
        amendments.append(args)
        time.sleep(0.05)
        amend(*args)

    monkeypatch.setattr(structured_model, "_amend_clip_default", slow_amend)
    results = []
    barrier = threading.Barrier(8)

    def read():
        barrier.wait()
        results.append(
            RacingHolder._get_comparison_info("line_items").clip_under_threshold
        )

    threads = [threading.Thread(target=read) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [False] * 8
    assert len(amendments) == 1
    schema = RacingHolder.to_json_schema()
    assert (
        schema["properties"]["line_items"]["x-aws-stickler-clip-under-threshold"]
        is False
    )


class BareOptionalElement(StructuredModel):
    line_items: List[Optional[LineItem]]


class BareAnnotatedElement(StructuredModel):
    line_items: List[Annotated[LineItem, "m"]]


class BareOptionalList(StructuredModel):
    line_items: Optional[List[LineItem]] = None


@pytest.mark.parametrize(
    "cls, reported",
    [
        (BareList, False),
        (BareOptionalList, False),
        # Not changed here: a bare annotation with an optional or `Annotated`
        # element still reaches the primitive fallback, as on 1.0.x, and
        # reports the global default. It scores the same as the others.
        (BareOptionalElement, True),
        (BareAnnotatedElement, True),
    ],
)
def test_bare_list_of_models_spellings(cls, reported):
    gt = cls(line_items=[LineItem(sku="A-100", amount=40, qty=1)])
    pred = cls(line_items=[LineItem(sku="A-100", amount=45, qty=2)])

    assert cls._get_comparison_info("line_items").clip_under_threshold is reported
    assert gt.compare_with(pred)["field_scores"]["line_items"] == pytest.approx(1 / 3)


class BareAddress(StructuredModel):
    a: str = ComparableField(comparator=ExactComparator())
    b: str = ComparableField(comparator=ExactComparator())


class BareNested(StructuredModel):
    addr: BareAddress


def test_a_bare_single_nested_model_still_clips():
    """Only a list of models takes the off default; a single nested model
    keeps clipping on, as on 1.0.x."""
    gt = BareNested(addr=BareAddress(a="1", b="2"))
    pred = BareNested(addr=BareAddress(a="1", b="x"))

    assert BareNested._get_comparison_info("addr").clip_under_threshold is True
    assert gt.compare_with(pred)["field_scores"]["addr"] == 0.0


class SubHolder(StructuredModel):
    sub: Optional["LateSub"] = ComparableField(default=None)


class LateSub(StructuredModel):
    a: str = ComparableField(comparator=ExactComparator())
    b: str = ComparableField(comparator=ExactComparator())


SubHolder.model_rebuild()


def test_settling_leaves_a_forward_referenced_single_model_alone():
    """The recorded field is a single model, not a list, so settling must not
    give it the list default."""
    gt = SubHolder(sub=LateSub(a="1", b="2"))
    pred = SubHolder(sub=LateSub(a="1", b="x"))
    score = gt.compare_with(pred)["field_scores"]["sub"]

    assert SubHolder._get_comparison_info("sub").clip_under_threshold is True
    assert score == pytest.approx(0.5)


def test_a_bare_list_of_models_exports_no_clip():
    """The export writes the resolved clip only where it already wrote one."""
    schema = BareList.to_json_schema()["properties"]["line_items"]
    config = BareList.to_stickler_config()["fields"]["line_items"]

    assert "x-aws-stickler-clip-under-threshold" not in schema
    assert "clip_under_threshold" not in config


def test_recording_settling_and_collection_take_the_lock():
    output = _run_isolated(
        """
        import gc, threading
        from typing import List
        from stickler import ComparableField, StructuredModel
        from stickler.structured_object_evaluator.models import configuration_helper as ch

        entered = []

        class CountingLock:
            def __init__(self):
                self._lock = threading.RLock()

            def __enter__(self):
                entered.append(1)
                return self._lock.__enter__()

            def __exit__(self, *exc):
                return self._lock.__exit__(*exc)

        ch._DEFERRED_CLIP_LOCK = CountingLock()

        class Holder(StructuredModel):
            line_items: List["Item"] = ComparableField()

        print(len(entered))

        class Item(StructuredModel):
            sku: str = ComparableField()

        Holder.model_rebuild()
        Holder._get_comparison_info("line_items")
        print(len(entered))

        def define():
            class Garbage(StructuredModel):
                line_items: List["UndefinedItem"] = ComparableField()

        define()
        before = len(entered)
        gc.collect()
        print(len(entered) - before >= 1)
        """
    )

    assert output[:2] == ["1", "2"]
    assert output[2] == "True"


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs os.fork")
def test_a_forked_child_gets_a_fresh_lock():
    """A lock held by another thread at the fork would never be released in
    the child; the child replaces it, as `logging` does for its own."""
    output = _run_isolated(
        """
        import os, threading
        from stickler.structured_object_evaluator.models import configuration_helper as ch

        held = threading.Event()
        release = threading.Event()

        def hold():
            with ch._DEFERRED_CLIP_LOCK:
                held.set()
                release.wait()

        threading.Thread(target=hold, daemon=True).start()
        held.wait()
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:
            acquired = ch._DEFERRED_CLIP_LOCK.acquire(timeout=5)
            os.write(write, b"1" if acquired else b"0")
            os._exit(0)
        os.waitpid(pid, 0)
        print(os.read(read, 1).decode())
        release.set()
        """
    )

    assert output == ["1"]
