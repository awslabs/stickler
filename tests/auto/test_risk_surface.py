"""Risk-surface tests for the zero-config evaluation path.

These pin the CONTRACTS the auto/ package must uphold rather than exact
heuristic outputs, so intentional rule tweaks do not churn the suite:

1. Identity invariant: an instance compared with itself scores 1.0, for every
   supported field shape. A comparator that cannot parse its own field's
   values violates this before anything else.
2. Wire contract: any JSON-dumpable field type either evaluates or raises a
   clear TypeError at eval_for time; never a pydantic ValidationError naming
   an internal shadow class mid-evaluate.
3. Explain contract: .explain() covers every field at every depth, and
   reports the comparator that is actually installed.
"""

import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import pytest
from pydantic import BaseModel

import stickler
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"


class Level(IntEnum):
    LOW = 1
    HIGH = 2


class Nested(BaseModel):
    label: str
    value: int


class EveryShape(BaseModel):
    """One field per supported shape; the identity invariant must hold on all."""

    plain_str: str
    plain_int: int
    plain_float: float
    plain_bool: bool
    str_enum: Color
    int_enum: Level
    lit_str: Literal["a", "b"]
    lit_int: Literal[1, 2]
    a_date: datetime.date
    a_datetime: datetime.datetime
    opt_str: Optional[str] = None
    pep604_str: str | None = None
    pep604_float: float | None = None
    a_dict: Dict[str, str] = {}
    a_tuple: Tuple[str, int] = ("x", 0)
    a_set: Set[str] = set()
    an_any: Any = None
    a_union: Union[int, str] = 0
    bare_list: list = []
    str_list: List[str] = []
    opt_list: Optional[List[str]] = None
    opt_elem_list: List[Optional[str]] = []
    nested: Optional[Nested] = None
    pep604_nested: Nested | None = None
    nested_list: List[Nested] = []
    opt_elem_model_list: List[Optional[Nested]] = []


def _full_instance() -> EveryShape:
    return EveryShape(
        plain_str="hello",
        plain_int=7,
        plain_float=3.14,
        plain_bool=True,
        str_enum=Color.RED,
        int_enum=Level.HIGH,
        lit_str="a",
        lit_int=2,
        a_date=datetime.date(2024, 3, 15),
        a_datetime=datetime.datetime(2024, 3, 15, 12, 30, 45),
        opt_str="present",
        pep604_str="also present",
        pep604_float=2.5,
        a_dict={"k1": "v1", "k2": "v2"},
        a_tuple=("y", 9),
        a_set={"s1", "s2"},
        an_any={"free": ["form", 1]},
        a_union="mixed",
        bare_list=[1, "two"],
        str_list=["a", "b"],
        opt_list=["x"],
        opt_elem_list=["e", None],
        nested=Nested(label="n", value=1),
        pep604_nested=Nested(label="p", value=2),
        nested_list=[Nested(label="l1", value=1), Nested(label="l2", value=2)],
        opt_elem_model_list=[Nested(label="m", value=3), None],
    )


class TestIdentityInvariant:
    """gt == pred must score 1.0 on every field, for every shape."""

    def test_every_shape_identical_scores_one(self):
        gt, pred = _full_instance(), _full_instance()
        result = stickler.evaluate(gt, pred)
        not_perfect = {k: v for k, v in result.field_scores.items() if v != 1.0}
        assert not_perfect == {}, f"identical instances mis-scored: {not_perfect}"
        assert result.overall_score == pytest.approx(1.0)

    def test_every_shape_all_none_defaults(self):
        gt = EveryShape(
            plain_str="s",
            plain_int=1,
            plain_float=1.0,
            plain_bool=False,
            str_enum=Color.BLUE,
            int_enum=Level.LOW,
            lit_str="b",
            lit_int=1,
            a_date=datetime.date(2024, 1, 1),
            a_datetime=datetime.datetime(2024, 1, 1),
        )
        pred = EveryShape(
            plain_str="s",
            plain_int=1,
            plain_float=1.0,
            plain_bool=False,
            str_enum=Color.BLUE,
            int_enum=Level.LOW,
            lit_str="b",
            lit_int=1,
            a_date=datetime.date(2024, 1, 1),
            a_datetime=datetime.datetime(2024, 1, 1),
        )
        result = stickler.evaluate(gt, pred)
        assert result.overall_score == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("field_name", "gt_type", "identical_value"),
        [
            # token/type conflicts: the token must NOT install a comparator
            # that zeroes identical values of this type.
            ("has_due_date", bool, True),
            ("due", str, "next week"),
            ("amount", str, "N/A"),
            ("count", str, "several"),
            ("updated_by", str, "Jane Smith"),
            ("days_past_due", int, -5),
            ("fee_applied", bool, True),
            ("phone_num", str, "(555) 123-4567"),
            ("isbn13", str, "978-3-16-148410-0"),
            ("created", bool, True),
        ],
    )
    def test_token_type_conflicts_identity(self, field_name, gt_type, identical_value):
        Model = type(
            "TokenConflict",
            (BaseModel,),
            {"__annotations__": {field_name: gt_type}},
        )
        gt = Model(**{field_name: identical_value})
        pred = Model(**{field_name: identical_value})
        result = stickler.evaluate(gt, pred)
        assert result.field_scores[field_name] == pytest.approx(1.0), (
            f"{field_name}: identical value {identical_value!r} scored "
            f"{result.field_scores[field_name]}"
        )


class TestWireContract:
    """Exotic values are compared in canonical JSON form, not crashed on."""

    def test_dict_key_order_irrelevant(self):
        class M(BaseModel):
            meta: Dict[str, int]

        r = stickler.evaluate(M(meta={"a": 1, "b": 2}), M(meta={"b": 2, "a": 1}))
        assert r.field_scores["meta"] == pytest.approx(1.0)

    def test_set_order_irrelevant(self):
        class M(BaseModel):
            tags: Set[str]

        r = stickler.evaluate(M(tags={"x", "y", "z"}), M(tags={"z", "y", "x"}))
        assert r.field_scores["tags"] == pytest.approx(1.0)

    def test_union_is_not_data_dependent(self):
        class M(BaseModel):
            mixed: Union[int, str]

        assert stickler.evaluate(M(mixed=1), M(mixed=1)).overall_score == 1.0
        assert stickler.evaluate(M(mixed="a"), M(mixed="a")).overall_score == 1.0

    def test_different_exotic_values_do_not_score_one(self):
        class M(BaseModel):
            meta: Dict[str, str]

        r = stickler.evaluate(M(meta={"a": "1"}), M(meta={"a": "TOTALLY DIFFERENT"}))
        # Exact over the canonical JSON string: a substantively different dict
        # scores 0.0, not the ~0.95 edit distance on the JSON blob would give.
        # Asserting the exact score and comparator (rather than just < 1.0) is
        # what makes this able to detect a change to the exotic-type fallback.
        assert r.field_scores["meta"] == pytest.approx(0.0)
        assert r.explain()["meta"]["comparator"] == "ExactComparator"

    def test_set_wire_form_is_sorted(self):
        """Pin the canonical wire string, not just that two set literals agree.

        Two ``set`` literals built in one process iterate in the same order, so
        a score-only assertion passes even without the sort. The documented
        dataset path is ``from_json`` on rows whose array order is whatever the
        producer wrote, so compare differently-ordered JSON lists.
        """
        from stickler import StructuredModel

        class M(BaseModel):
            tags: Set[str]

        MEval = StructuredModel.from_pydantic(M)
        assert MEval.from_json({"tags": ["gamma", "alpha", "beta"]}).tags == (
            '["alpha", "beta", "gamma"]'
        )
        a = MEval.from_json({"tags": ["alpha", "beta", "gamma"]})
        b = MEval.from_json({"tags": ["gamma", "alpha", "beta"]})
        assert a.compare_with(b)["overall_score"] == pytest.approx(1.0)

    def test_any_field_accepts_none(self):
        class M(BaseModel):
            anything: Any = None

        assert stickler.evaluate(M(), M()).overall_score == pytest.approx(1.0)


class TestNullability:
    """Nullability comes from the annotation, not required-ness."""

    def test_required_nullable_scalar(self):
        class M(BaseModel):
            note: Optional[str]

        assert stickler.evaluate(M(note=None), M(note=None)).overall_score == 1.0
        assert stickler.evaluate(M(note="x"), M(note=None)).overall_score < 1.0

    def test_optional_list_none(self):
        class M(BaseModel):
            tags: Optional[List[str]] = None

        assert stickler.evaluate(M(tags=None), M(tags=None)).overall_score == 1.0

    def test_pep604_required_nullable(self):
        class M(BaseModel):
            note: str | None

        assert stickler.evaluate(M(note=None), M(note=None)).overall_score == 1.0


class TestStructuralErrors:
    """Unsupported shapes fail loudly at build time with actionable errors."""

    def test_recursive_model_raises_clear_error(self):
        class Node(BaseModel):
            val: str
            children: List["Node"] = []

        Node.model_rebuild()
        with pytest.raises(TypeError, match="recursive models"):
            stickler.evaluate(Node(val="a"), Node(val="a"))

    def test_empty_model_raises_clear_error(self):
        class Empty(BaseModel):
            pass

        with pytest.raises(TypeError, match="at least one field"):
            stickler.evaluate(Empty(), Empty())

    def test_reserved_field_name_raises_clear_error(self):
        class WithReserved(BaseModel):
            extra_fields: str = "x"

        with pytest.raises(TypeError, match="reserved"):
            stickler.evaluate(WithReserved(), WithReserved())

    def test_non_pydantic_input_raises_type_error(self):
        class M(BaseModel):
            x: str

        with pytest.raises(TypeError):
            stickler.evaluate({"x": "plain dict"}, M(x="a"))


class TestStructuredModelPassthrough:
    """Explicit StructuredModel configuration is never re-inferred over."""

    def test_explicit_comparator_respected(self):
        from stickler.comparators.exact import ExactComparator

        class SM(StructuredModel):
            code: str = ComparableField(comparator=ExactComparator(), threshold=1.0)

        r = stickler.evaluate(SM(code="ABC-123"), SM(code="ABC-124"))
        # ExactComparator: near-identical strings must score 0, where the
        # inferred Levenshtein default would have scored ~0.9.
        assert r.field_scores["code"] == 0.0

    def test_explain_reports_explicit_source(self):
        from stickler.comparators.exact import ExactComparator

        class SM(StructuredModel):
            code: str = ComparableField(comparator=ExactComparator(), threshold=1.0)

        ex = stickler.eval_for(SM).explain()
        assert ex["code"]["source"] == "explicit"

    def test_structured_model_nested_in_basemodel(self):
        from stickler.comparators.exact import ExactComparator

        class SM(StructuredModel):
            code: str = ComparableField(comparator=ExactComparator(), threshold=1.0)

        class Wrap(BaseModel):
            inner: SM
            label: str

        gt = Wrap(inner=SM(code="X-1"), label="a")
        pred = Wrap(inner=SM(code="X-1"), label="a")
        assert stickler.evaluate(gt, pred).overall_score == pytest.approx(1.0)


class TestExplainContract:
    """explain() must describe what is actually installed, at every depth."""

    def test_covers_nested_fields_with_dotted_paths(self):
        class Line(BaseModel):
            sku: str
            price: float

        class Invoice(BaseModel):
            vendor: str
            lines: List[Line]

        ex = stickler.eval_for(Invoice).explain()
        assert {"vendor", "lines", "lines.sku", "lines.price"} <= set(ex)

    def test_primitive_list_reports_element_comparator(self):
        class M(BaseModel):
            prices: List[float]

        ex = stickler.eval_for(M).explain()
        # Must match the element comparator the builder installs (Numeric),
        # not a fallback derived from the list annotation itself.
        assert ex["prices"]["comparator"] == "NumericComparator"

    def test_every_explained_comparator_is_instantiable(self):
        from stickler.structured_object_evaluator.models.comparator_registry import (
            get_global_registry,
        )

        registry = get_global_registry()
        gt = _full_instance()
        ex = stickler.eval_for(type(gt)).explain()
        for path, entry in ex.items():
            name = entry["comparator"]
            if name.startswith("Hungarian") or name == "StructuredModelComparator":
                continue  # structural rows, not registry comparators
            assert registry.is_registered(name), f"{path}: {name} unregistered"

    def test_result_explain_includes_pair_verdict(self):
        class M(BaseModel):
            vendor_name: str

        r = stickler.evaluate(
            M(vendor_name="Acme Corporation"), M(vendor_name="Acme Corp")
        )
        entry = r.explain()["vendor_name"]
        assert entry["score"] == 0.0
        assert 0 < entry["raw_similarity"] < 1
        assert "clipped" in entry["verdict"]


class TestNestedScoringConsistency:
    """A nested object scores the same whether single or in a one-element list."""

    def test_single_vs_listed_partial_credit(self):
        class Child(BaseModel):
            a: str
            b: str
            c: str
            d: str
            e: str

        class Single(BaseModel):
            child: Child

        class Listed(BaseModel):
            children: List[Child]

        gt_c = Child(a="1", b="2", c="3", d="4", e="5")
        pr_c = Child(a="1", b="2", c="3", d="4", e="WRONG")
        single = stickler.evaluate(Single(child=gt_c), Single(child=pr_c)).field_scores[
            "child"
        ]
        listed = stickler.evaluate(
            Listed(children=[gt_c]), Listed(children=[pr_c])
        ).field_scores["children"]
        assert single == pytest.approx(listed)
        assert 0 < single < 1  # partial credit, not clipped to zero

    def test_single_nested_below_threshold_not_clipped(self):
        """A nested object scoring BELOW match_threshold keeps partial credit.

        This engages the clip_under_threshold=False guard directly: with only
        2/5 sub-fields correct the child scores ~0.4 < 0.7, so a clipping
        model would return 0.0. Kills the mutation that flips the flag.
        """

        class Child(BaseModel):
            a: str
            b: str
            c: str
            d: str
            e: str

        class Single(BaseModel):
            child: Child

        gt = Child(a="1", b="2", c="X", d="Y", e="Z")
        pred = Child(a="1", b="2", c="c", d="d", e="e")  # only a,b match -> 0.4
        score = stickler.evaluate(Single(child=gt), Single(child=pred)).field_scores[
            "child"
        ]
        assert 0.0 < score < 0.7  # partial credit below threshold, not clipped


class TestInferenceInternals:
    """Guard behaviors that end-to-end tests don't exercise."""

    def test_phone_token_beats_numeric_token_order(self):
        """A str field named 'phone_num' resolves to Exact, not Numeric.

        'num' and 'phone' both tokenize; the email/phone rule must precede the
        quantity rule. NumericComparator would strip formatting and score
        distinct formatted phone numbers as equal (and identical ones fine but
        for the wrong reason). Kills the rule-reorder mutation.
        """
        from pydantic.fields import FieldInfo

        from stickler.auto.inference import infer_field_config

        spec = infer_field_config("phone_num", FieldInfo(annotation=str))
        assert spec.comparator_name == "ExactComparator"

    def test_gate_degrades_unregistered_comparator(self):
        """_gate falls back and records provenance for an unavailable comparator."""
        from stickler.auto.inference import _gate

        class _EmptyRegistry:
            def is_registered(self, name):
                return False

            def create_instance(self, name, config):
                raise AssertionError("should not be called when unregistered")

        prov = []
        name, _ = _gate("FuzzyComparator", {}, _EmptyRegistry(), prov)
        assert name == "LevenshteinComparator"
        assert any("degrade" in p for p in prov)

    def test_gate_degrades_when_construction_fails(self):
        """A registered comparator whose backend is missing degrades at infer time."""
        from stickler.auto.inference import _gate

        class _BrokenRegistry:
            def is_registered(self, name):
                return True

            def create_instance(self, name, config):
                raise ImportError("optional backend missing")

        prov = []
        name, _ = _gate("FuzzyComparator", {}, _BrokenRegistry(), prov)
        assert name == "LevenshteinComparator"
        assert any("ImportError" in p for p in prov)

    def test_gate_never_auto_selects_llm(self):
        from stickler.auto.inference import _gate
        from stickler.structured_object_evaluator.models.comparator_registry import (
            get_global_registry,
        )

        prov = []
        name, _ = _gate("LLMComparator", {}, get_global_registry(), prov)
        assert name == "LevenshteinComparator"
        assert any("never auto-selected" in p for p in prov)

    def test_gate_date_degrades_to_exact(self):
        from stickler.auto.inference import _gate

        class _EmptyRegistry:
            def is_registered(self, name):
                return False

            def create_instance(self, name, config):
                raise AssertionError("unused")

        name, _ = _gate("DateComparator", {}, _EmptyRegistry(), [])
        assert name == "ExactComparator"


class TestDatasetWorkflow:
    """The compiled-spec path works with the data users actually have."""

    def test_evalspec_accepts_dicts(self):
        class M(BaseModel):
            name: str
            qty: int

        spec = stickler.eval_for(M)
        r = spec.evaluate({"name": "widget", "qty": 5}, {"name": "widget", "qty": 5})
        assert r.overall_score == pytest.approx(1.0)

    def test_evalspec_dict_validation_error_names_source_class(self):
        class Order(BaseModel):
            qty: int

        spec = stickler.eval_for(Order)
        with pytest.raises(Exception, match="Order"):
            spec.evaluate({"qty": "not-a-number-at-all"}, {"qty": 1})

    def test_matched_flag_reflects_threshold(self):
        class M(BaseModel):
            name: str

        same = stickler.evaluate(M(name="a"), M(name="a"))
        diff = stickler.evaluate(M(name="alpha"), M(name="omega"))
        assert same.matched is True
        assert diff.matched is False

    def test_repeated_evaluation_is_deterministic(self):
        gt, pred = _full_instance(), _full_instance()
        pred.plain_str = "changed"
        r1 = stickler.evaluate(gt, pred)
        r2 = stickler.evaluate(gt, pred)
        assert r1.field_scores == r2.field_scores
        assert r1.overall_score == r2.overall_score

    def test_from_pydantic_with_standard_bulk_evaluator(self):
        """from_pydantic output plugs into BulkStructuredModelEvaluator as-is."""
        from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
            BulkStructuredModelEvaluator,
        )

        class M(BaseModel):
            name: str
            qty: int

        MEval = StructuredModel.from_pydantic(M)
        evaluator = BulkStructuredModelEvaluator(target_schema=MEval)
        pairs = [
            (M(name="a", qty=1), M(name="a", qty=1)),
            (M(name="b", qty=2), M(name="b", qty=2)),
            (M(name="c", qty=3), M(name="WRONG", qty=99)),
        ]
        for gt, pred in pairs:
            evaluator.update(
                MEval.from_json(gt.model_dump()),
                MEval.from_json(pred.model_dump()),
            )
        result = evaluator.compute()
        assert result.document_count == 3
        assert 0 < result.metrics["cm_f1"] < 1  # 2 perfect + 1 mismatch


class TestFromPydantic:
    """StructuredModel.from_pydantic is the first-class constructor."""

    def test_returns_structured_model_subclass(self):
        class M(BaseModel):
            name: str

        MEval = StructuredModel.from_pydantic(M)
        assert issubclass(MEval, StructuredModel)

    def test_accepts_native_and_json_dumps(self):
        """Instances validate from plain model_dump() and mode='json' alike."""

        class Color(Enum):
            RED = "red"

        class M(BaseModel):
            when: datetime.date
            color: Color

        MEval = StructuredModel.from_pydantic(M)
        src = M(when=datetime.date(2024, 1, 1), color=Color.RED)
        native = MEval.from_json(src.model_dump())
        json_form = MEval.from_json(src.model_dump(mode="json"))
        assert native.compare_with(json_form)["overall_score"] == pytest.approx(1.0)

    def test_structured_model_passthrough_identity(self):
        class SM(StructuredModel):
            code: str = ComparableField(default=None)

        assert StructuredModel.from_pydantic(SM) is SM

    def test_export_edit_rebuild_roundtrip(self):
        """The inferred model exports to an editable stickler config."""

        class M(BaseModel):
            name: str
            note: str

        MEval = StructuredModel.from_pydantic(M)
        config = MEval.to_stickler_config()
        config["fields"]["note"]["comparator"] = "ExactComparator"
        Edited = StructuredModel.model_from_json(config)
        gt = Edited.from_json({"name": "x", "note": "Hello World"})
        pred = Edited.from_json({"name": "x", "note": "hello, world"})
        # ExactComparator normalizes case/punctuation: still a match.
        assert gt.compare_with(pred)["field_scores"]["note"] == pytest.approx(1.0)

    def test_evaluate_and_from_pydantic_agree(self):
        """The facade and the constructor produce identical scores."""

        class M(BaseModel):
            name: str
            qty: int

        gt, pred = M(name="acme corp", qty=5), M(name="acme corporation", qty=5)
        facade_score = stickler.evaluate(gt, pred).overall_score
        MEval = StructuredModel.from_pydantic(M)
        direct = MEval.from_json(gt.model_dump()).compare_with(
            MEval.from_json(pred.model_dump())
        )["overall_score"]
        assert facade_score == pytest.approx(direct)


class TestDatetimeSensitivity:
    """Datetimes are compared at sub-day resolution; dates at day resolution."""

    def test_hours_apart_do_not_match(self):
        class M(BaseModel):
            ts: datetime.datetime

        gt = M(ts=datetime.datetime(2020, 1, 1, 12, 0, 0))
        pred = M(ts=datetime.datetime(2020, 1, 1, 15, 0, 0))
        assert stickler.evaluate(gt, pred).field_scores["ts"] < 1.0

    def test_seconds_apart_match(self):
        class M(BaseModel):
            ts: datetime.datetime

        gt = M(ts=datetime.datetime(2020, 1, 1, 12, 0, 0))
        pred = M(ts=datetime.datetime(2020, 1, 1, 12, 0, 5))
        assert stickler.evaluate(gt, pred).field_scores["ts"] == pytest.approx(1.0)

    def test_same_date_matches(self):
        class M(BaseModel):
            d: datetime.date

        gt = M(d=datetime.date(2020, 1, 1))
        assert (
            stickler.evaluate(gt, M(d=datetime.date(2020, 1, 1))).field_scores["d"]
            == 1.0
        )


class TestNumericEdgeCases:
    """Numeric fields must not crash on non-finite values or mis-score Decimal."""

    def test_nan_does_not_crash(self):
        class M(BaseModel):
            x: float

        # Must not raise decimal.InvalidOperation. NaN never equals NaN.
        r = stickler.evaluate(M(x=float("nan")), M(x=float("nan")))
        assert r.field_scores["x"] == pytest.approx(0.0)

    def test_infinity_matches_exactly(self):
        class M(BaseModel):
            x: float

        inf = float("inf")
        assert stickler.evaluate(M(x=inf), M(x=inf)).field_scores["x"] == 1.0
        assert stickler.evaluate(M(x=inf), M(x=-inf)).field_scores["x"] == 0.0

    def test_decimal_numeric_equivalence(self):
        from decimal import Decimal

        class M(BaseModel):
            total_amount: Decimal

        # Numerically equal, different representation -> match.
        r = stickler.evaluate(
            M(total_amount=Decimal("100")), M(total_amount=Decimal("100.00"))
        )
        assert r.field_scores["total_amount"] == pytest.approx(1.0)
        # Different value -> no match.
        r2 = stickler.evaluate(
            M(total_amount=Decimal("100")), M(total_amount=Decimal("200"))
        )
        assert r2.field_scores["total_amount"] == pytest.approx(0.0)

    def test_decimal_inferred_as_numeric(self):
        from decimal import Decimal

        class M(BaseModel):
            price: Decimal

        assert stickler.eval_for(M).explain()["price"]["comparator"] == (
            "NumericComparator"
        )

    def test_decimal_type_branch_without_a_name_token(self):
        """Decimal must infer Numeric on its type alone.

        The two tests above use ``total_amount`` and ``price``, which both hit
        the money name-token rule and reinstall NumericComparator on top of
        whatever the type branch returned. A token-free name is what actually
        pins the Decimal type dispatch. Decimal is the money type: comparing it
        as a string would score Decimal('100') vs Decimal('100.00') as 0.0.
        """
        from decimal import Decimal

        class M(BaseModel):
            val: Decimal

        assert stickler.eval_for(M).explain()["val"]["comparator"] == (
            "NumericComparator"
        )
        r = stickler.evaluate(M(val=Decimal("100")), M(val=Decimal("100.00")))
        assert r.field_scores["val"] == pytest.approx(1.0)

    def test_decimal_money_token_applies_relative_tolerance(self):
        """The Decimal -> numeric token-family mapping carries the tolerance.

        Both existing Decimal pairs (100 vs 100.00, 100 vs 200) match or miss
        under any tolerance, so neither observes the money rule's relative
        tolerance. These land inside and outside the band.
        """
        from decimal import Decimal

        class M(BaseModel):
            total_amount: Decimal

        assert stickler.eval_for(M).explain()["total_amount"]["threshold"] == (
            pytest.approx(0.95)
        )
        inside = stickler.evaluate(
            M(total_amount=Decimal("100")), M(total_amount=Decimal("100.05"))
        )
        assert inside.field_scores["total_amount"] == pytest.approx(1.0)
        outside = stickler.evaluate(
            M(total_amount=Decimal("100")), M(total_amount=Decimal("101"))
        )
        assert outside.field_scores["total_amount"] == pytest.approx(0.0)


class TestDumpModeEquivalence:
    """Plain model_dump() and model_dump(mode='json') must score identically."""

    def test_set_field_both_dump_modes(self):
        from stickler import StructuredModel

        class M(BaseModel):
            tags: Set[str]

        MEval = StructuredModel.from_pydantic(M)
        inst = M(tags={"alpha", "beta", "gamma"})
        native = MEval.from_json(inst.model_dump())
        json_form = MEval.from_json(inst.model_dump(mode="json"))
        assert native.compare_with(json_form)["overall_score"] == pytest.approx(1.0)

    def test_dict_with_date_keys_both_dump_modes(self):
        from stickler import StructuredModel

        class M(BaseModel):
            by_day: Dict[datetime.date, int]

        MEval = StructuredModel.from_pydantic(M)
        inst = M(by_day={datetime.date(2024, 1, 1): 5})
        native = MEval.from_json(inst.model_dump())  # would crash pre-fix
        json_form = MEval.from_json(inst.model_dump(mode="json"))
        assert native.compare_with(json_form)["overall_score"] == pytest.approx(1.0)

    def test_nested_exotic_scalars_both_dump_modes(self):
        import uuid
        from decimal import Decimal

        from stickler import StructuredModel

        class M(BaseModel):
            meta: Dict[str, Any]

        MEval = StructuredModel.from_pydantic(M)
        inst = M(
            meta={
                "id": uuid.UUID(int=7),
                "amt": Decimal("1.50"),
                "ts": datetime.datetime(2024, 1, 1, 12),
            }
        )
        native = MEval.from_json(inst.model_dump())
        json_form = MEval.from_json(inst.model_dump(mode="json"))
        assert native.compare_with(json_form)["overall_score"] == pytest.approx(1.0)
