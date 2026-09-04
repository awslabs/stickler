"""A nested plain pydantic `BaseModel` is scored, not treated as a type mismatch.

`ComparisonDispatcher` routed on `isinstance(value, StructuredModel)`. A plain
`BaseModel` is not one, so a nested one fell through to the mismatched-types
branch and scored `0.0` against an **identical** object, reported as a false
discovery: a perfect match that is also a failure, which is the contradiction
#287 removed elsewhere.

The asymmetry is what made it clearly a defect rather than an unsupported shape:

    Optional[Plain]        identical -> 0.0   and fd=1
    Optional[List[Plain]]  identical -> 1.0

because the list branch sends any non-`StructuredModel` element to the primitive
list comparator. The singular and list forms of one shape disagreed.

Both forms now get the same treatment a `dict` field gets: `ANLSStarComparator`
at object grade, judging the model's JSON form key by key. Routing to the dict
path without also adopting its CONFIGURATION was not enough, and was the defect
found in review: the field kept the primitive default of Levenshtein at 0.5,
i.e. edit distance over `str(model)`. Field names are identical on both sides,
so that put a floor under every score and a wholly wrong prediction classified
as a match:

    LineItem(quantity=2, unit_price=10.5, currency='USD')
    vs LineItem(quantity=9, unit_price=99.9, currency='EUR')  ->  0.8293, tp=1

Under ANLS* the same pair is 0.0 and a false discovery.

Two plain models of DIFFERENT classes are a false discovery, whatever their
field names. `ConfigurationHelper.values_are_same_model_class` owns that rule
because four paths ask the question -- `compare_with`, `compare`, and the list
form of each -- and writing it four times is how they drift apart. A correctly
annotated field never reaches it: pydantic refuses a `Dog` for an
`Optional[Cat]` field at construction, so it fires only where the annotation
permitted both (`Union[Cat, Dog]`, `Any`, `object`) or where a subclass was
supplied for its base. It warns rather than raising, because which class arrives
is prediction data and raising would end a corpus run on document N.

See https://github.com/awslabs/stickler/issues/318 and
https://github.com/awslabs/stickler/issues/321
"""

from typing import Any, List, Optional

import pytest
from pydantic import BaseModel

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)


class Plain(BaseModel):
    """A vanilla pydantic model, the shape a user brings from elsewhere."""

    sku: Optional[str] = None
    qty: Optional[int] = None


class Structured(StructuredModel):
    """The same shape, configured, so the two paths can be compared."""

    sku: Optional[str] = ComparableField(default=None)
    qty: Optional[int] = ComparableField(default=None)


class Nested(StructuredModel):
    kid: Optional[Plain] = ComparableField(default=None)


class NestedList(StructuredModel):
    rows: Optional[List[Plain]] = ComparableField(default=None)


class TestIdenticalObjectsAreNotAFailure:
    """The reported bug: `0.0` and `fd=1` on two equal objects."""

    def test_an_identical_nested_plain_model_scores_one(self):
        result = Nested(kid=Plain(sku="x", qty=1)).compare_with(
            Nested(kid=Plain(sku="x", qty=1))
        )
        assert result["field_scores"]["kid"] == pytest.approx(1.0)
        assert result["overall_score"] == pytest.approx(1.0)

    def test_it_is_classified_as_a_true_positive(self):
        """Score and classification have to agree, which is the #287 lesson."""
        matrix = Nested(kid=Plain(sku="x")).compare_with(
            Nested(kid=Plain(sku="x")), include_confusion_matrix=True
        )["confusion_matrix"]

        assert matrix["overall"]["tp"] == 1
        assert matrix["overall"]["fd"] == 0
        assert matrix["overall"]["fp"] == 0

    def test_a_wholly_different_nested_plain_model_scores_zero(self):
        """`< 1.0` is not a real assertion here, and it hid a bug.

        This asserted only `< 1.0` while the field ran Levenshtein over
        `str(model)`. Field-name boilerplate is identical on both sides, so it
        put a floor under the score: three differing values scored 0.8293 and
        classified as a TRUE POSITIVE, and the loose assertion passed. Pinning
        the number is the point.
        """
        result = Nested(kid=Plain(sku="completely-different", qty=99)).compare_with(
            Nested(kid=Plain(sku="x", qty=1)), include_confusion_matrix=True
        )
        assert result["field_scores"]["kid"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1
        assert result["confusion_matrix"]["overall"]["tp"] == 0

    def test_a_partly_wrong_nested_plain_model_gets_partial_credit(self):
        """Object-grade scoring, not edit distance: one of two fields wrong."""
        result = Nested(kid=Plain(sku="x", qty=1)).compare_with(
            Nested(kid=Plain(sku="x", qty=99))
        )
        assert result["field_scores"]["kid"] == pytest.approx(0.5)


class TestTheSingularAndListFormsAgree:
    """The asymmetry that identified the bug must not survive the fix."""

    @pytest.mark.parametrize(
        "ground_truth, prediction",
        (
            (Plain(sku="x", qty=1), Plain(sku="x", qty=1)),
            (Plain(sku="a", qty=1), Plain(sku="b", qty=1)),
            (Plain(sku="a"), Plain(sku="a", qty=7)),
        ),
    )
    def test_one_element_scores_the_same_either_way(self, ground_truth, prediction):
        """Same class on both sides. Differing classes are covered below: this
        parametrisation cannot see a gate that only fires on a class mismatch."""
        singular = Nested(kid=ground_truth).compare_with(Nested(kid=prediction))[
            "field_scores"
        ]["kid"]
        listed = NestedList(rows=[ground_truth]).compare_with(
            NestedList(rows=[prediction])
        )["field_scores"]["rows"]

        assert singular == pytest.approx(listed)

    def test_the_list_form_was_already_correct_and_is_unchanged(self):
        assert NestedList(rows=[Plain(sku="x")]).compare_with(
            NestedList(rows=[Plain(sku="x")])
        )["field_scores"]["rows"] == pytest.approx(1.0)


class TestTheDeclaredComparatorIsUsed:
    """It routes through the field's comparator, not a hardcoded equality.

    This is the same reasoning as the dict branch added for #297: the "primitive"
    path is only primitive in name, and calls whatever the field declares.
    """

    def test_an_exact_comparator_makes_it_all_or_nothing(self):
        class Strict(StructuredModel):
            kid: Optional[Plain] = ComparableField(
                comparator=ExactComparator(), threshold=1.0, default=None
            )

        assert Strict(kid=Plain(sku="x")).compare_with(Strict(kid=Plain(sku="x")))[
            "field_scores"
        ]["kid"] == pytest.approx(1.0)

        # One character apart would earn partial credit on edit distance.
        assert Strict(kid=Plain(sku="a")).compare_with(Strict(kid=Plain(sku="b")))[
            "field_scores"
        ]["kid"] == pytest.approx(0.0)

    def test_the_default_comparator_gives_partial_credit(self):
        """Levenshtein over the canonical string, so a near miss is not zero."""
        score = Nested(kid=Plain(sku="a", qty=1)).compare_with(
            Nested(kid=Plain(sku="b", qty=1))
        )["field_scores"]["kid"]
        assert 0.0 < score < 1.0


class TestTheCanonicalFormIsStable:
    """Comparison is on the rendered model, so its rendering must be order-free.

    If it were keyword-order dependent, two equal objects built differently would
    score below 1.0, which is the class of bug #276 was about for dicts.
    """

    def test_keyword_order_at_construction_does_not_matter(self):
        result = Nested(kid=Plain(sku="x", qty=1)).compare_with(
            Nested(kid=Plain(qty=1, sku="x"))
        )
        assert result["field_scores"]["kid"] == pytest.approx(1.0)

    def test_an_absent_optional_field_is_rendered_consistently(self):
        result = Nested(kid=Plain(sku="x")).compare_with(Nested(kid=Plain(sku="x")))
        assert result["field_scores"]["kid"] == pytest.approx(1.0)


class TestAStructuredModelIsUnaffected:
    """The new branch sits after the StructuredModel one and must not shadow it.

    `StructuredModel` subclasses `BaseModel`, so an `isinstance` check on the
    latter would catch both. Order in the dispatch chain is what keeps them
    apart, and a nested `StructuredModel` must keep its per-field detail.
    """

    def test_a_nested_structured_model_still_reports_per_field_detail(self):
        class Holder(StructuredModel):
            kid: Optional[Structured] = ComparableField(default=None)

        matrix = Holder(kid=Structured(sku="a", qty=1)).compare_with(
            Holder(kid=Structured(sku="b", qty=1)), include_confusion_matrix=True
        )["confusion_matrix"]

        assert matrix["fields"]["kid"]["fields"], "per-field detail was lost"
        assert "sku" in matrix["fields"]["kid"]["fields"]

    def test_a_nested_plain_model_reports_no_per_field_detail(self):
        """Recorded as the deliberate consequence of comparing the whole model.

        A plain `BaseModel` carries no per-field comparison configuration, so
        there is nothing to report per field. Anyone wanting that detail declares
        the nested model as a `StructuredModel`.
        """
        matrix = Nested(kid=Plain(sku="a")).compare_with(
            Nested(kid=Plain(sku="b")), include_confusion_matrix=True
        )["confusion_matrix"]

        assert not matrix["fields"]["kid"].get("fields")


class TestNullHandlingIsUnchanged:
    """The new branch is only reached when both sides are present."""

    def test_both_absent_is_a_true_negative(self):
        matrix = Nested(kid=None).compare_with(
            Nested(kid=None), include_confusion_matrix=True
        )["confusion_matrix"]
        assert matrix["overall"]["tn"] == 1

    def test_prediction_absent_is_a_false_negative(self):
        matrix = Nested(kid=Plain(sku="x")).compare_with(
            Nested(kid=None), include_confusion_matrix=True
        )["confusion_matrix"]
        assert matrix["overall"]["fn"] == 1

    def test_ground_truth_absent_is_a_false_alarm(self):
        matrix = Nested(kid=None).compare_with(
            Nested(kid=Plain(sku="x")), include_confusion_matrix=True
        )["confusion_matrix"]
        assert matrix["overall"]["fa"] == 1


class TestTheModelsReachTheComparatorUnconverted:
    """The dispatcher must not decide coercion on the comparator's behalf.

    Stringifying here looked equivalent, because `LevenshteinComparator` and
    `ExactComparator` coerce internally anyway, so a test using either cannot see
    the difference. It silently defeats any comparator that reads structure, and
    reintroduces exactly the asymmetry this module exists to remove.
    """

    class WithMapping(BaseModel):
        meta: Optional[dict] = None

    def test_a_structural_comparator_scores_the_model_not_its_repr(self):
        from stickler import ANLSStarComparator

        class Single(StructuredModel):
            kid: Optional["TestTheModelsReachTheComparatorUnconverted.WithMapping"] = (
                ComparableField(comparator=ANLSStarComparator(), default=None)
            )

        ground_truth = self.WithMapping(meta={"a": 1, "b": 2})
        prediction = self.WithMapping(meta={"a": 1, "b": 99})

        # One of two leaves wrong, scored structurally.
        assert Single(kid=ground_truth).compare_with(Single(kid=prediction))[
            "field_scores"
        ]["kid"] == pytest.approx(0.5)

    def test_the_structural_score_matches_the_list_form(self):
        """The parity claim, checked with a comparator that can tell.

        Under stringification this read 0.9091 against 0.5000, so the earlier
        parity tests passed only because their comparators stringify too.
        """
        from stickler import ANLSStarComparator

        class Single(StructuredModel):
            kid: Optional["TestTheModelsReachTheComparatorUnconverted.WithMapping"] = (
                ComparableField(comparator=ANLSStarComparator(), default=None)
            )

        class Listed(StructuredModel):
            kids: Optional[
                List["TestTheModelsReachTheComparatorUnconverted.WithMapping"]
            ] = ComparableField(comparator=ANLSStarComparator(), default=None)

        ground_truth = self.WithMapping(meta={"a": 1, "b": 2})
        prediction = self.WithMapping(meta={"a": 1, "b": 99})

        singular = Single(kid=ground_truth).compare_with(Single(kid=prediction))[
            "field_scores"
        ]["kid"]
        listed = Listed(kids=[ground_truth]).compare_with(Listed(kids=[prediction]))[
            "field_scores"
        ]["kids"]

        assert singular == pytest.approx(listed)

    def test_compare_and_compare_with_agree(self):
        """`compare_field_raw` passes the raw models, so this branch must too.

        Its own comment states the invariant: "Same gate the dispatcher uses, so
        compare() and compare_with() agree". Stringifying on one side only broke
        it, and `compare()` is what the Hungarian cost matrix reads.
        """
        from stickler import ANLSStarComparator
        from stickler.structured_object_evaluator.models.comparison_helper import (
            ComparisonHelper,
        )

        class Single(StructuredModel):
            kid: Optional["TestTheModelsReachTheComparatorUnconverted.WithMapping"] = (
                ComparableField(comparator=ANLSStarComparator(), default=None)
            )

        ground_truth = self.WithMapping(meta={"a": 1, "b": 2})
        prediction = self.WithMapping(meta={"a": 1, "b": 99})

        through_compare_with = Single(kid=ground_truth).compare_with(
            Single(kid=prediction)
        )["field_scores"]["kid"]
        through_compare = ComparisonHelper.compare_field_raw(
            Single(kid=ground_truth), "kid", prediction
        )

        assert through_compare_with == pytest.approx(through_compare)


class TestTwoDifferentClassesAreNotAMatch:
    """Pydantic's `__str__` omits the class name, so shape alone is not identity.

    `Cat(name="rex")` and `Dog(name="rex")` both render as `name='rex'`. Without a
    type guard they compared equal: a genuine type mismatch reported as a perfect
    match, which is worse than the `0.0` this branch was added to fix.
    """

    class Cat(BaseModel):
        name: Optional[str] = None

    class Dog(BaseModel):
        name: Optional[str] = None

    def test_unrelated_models_with_equal_fields_are_a_false_discovery(self):
        class Holder(StructuredModel):
            pet: Optional[Any] = ComparableField(default=None)

        result = Holder(pet=self.Cat(name="rex")).compare_with(
            Holder(pet=self.Dog(name="rex")), include_confusion_matrix=True
        )

        assert result["field_scores"]["pet"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1
        assert result["confusion_matrix"]["overall"]["tp"] == 0

    def test_their_renderings_really_are_identical(self):
        """The measurement behind the guard, so its necessity is visible."""
        assert str(self.Cat(name="rex")) == str(self.Dog(name="rex"))

    def test_a_structured_model_against_a_plain_one_is_a_mismatch(self):
        """`StructuredModel` subclasses `BaseModel`, so this pair reaches here."""

        class PlainShape(BaseModel):
            name: Optional[str] = None

        class StructuredShape(StructuredModel):
            name: Optional[str] = ComparableField(default=None)

        class Holder(StructuredModel):
            thing: Optional[Any] = ComparableField(default=None)

        result = Holder(thing=StructuredShape(name="rex")).compare_with(
            Holder(thing=PlainShape(name="rex")), include_confusion_matrix=True
        )
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_the_same_class_on_both_sides_still_scores(self):
        """The guard must not reject the ordinary case."""

        class Holder(StructuredModel):
            pet: Optional[Any] = ComparableField(default=None)

        assert Holder(pet=self.Cat(name="rex")).compare_with(
            Holder(pet=self.Cat(name="rex"))
        )["field_scores"]["pet"] == pytest.approx(1.0)


class TestASubclassIsADifferentShape:
    """Exact class, not `isinstance`, and the reason is measurable.

    A subclass renders its own fields, so even with the extra field unset the two
    renderings differ:

        str(Base(a="x"))   ->  "a='x'"
        str(Sub(a="x"))    ->  "a='x' b=None"

    Allowing the pair would score a schema mismatch by edit distance and report a
    near-match. A clean false discovery says more.
    """

    class Base(BaseModel):
        a: Optional[str] = None

    class Sub(Base):
        b: Optional[str] = None

    @pytest.mark.parametrize("extra", (None, "y"))
    def test_a_base_against_its_subclass_is_a_false_discovery(self, extra):
        class Holder(StructuredModel):
            v: Optional[Any] = ComparableField(default=None)

        result = Holder(v=self.Base(a="x")).compare_with(
            Holder(v=self.Sub(a="x", b=extra)), include_confusion_matrix=True
        )

        assert result["field_scores"]["v"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_the_renderings_differ_even_with_the_extra_field_unset(self):
        """The measurement behind the choice."""
        assert str(self.Base(a="x")) != str(self.Sub(a="x"))

    def test_a_subclass_against_itself_still_scores(self):
        class Holder(StructuredModel):
            v: Optional[Any] = ComparableField(default=None)

        assert Holder(v=self.Sub(a="x", b="y")).compare_with(
            Holder(v=self.Sub(a="x", b="y"))
        )["field_scores"]["v"] == pytest.approx(1.0)


class Cat(BaseModel):
    name: Optional[str] = None


class Dog(BaseModel):
    name: Optional[str] = None


class Base(BaseModel):
    a: Optional[str] = None


class Sub(Base):
    b: Optional[str] = None


class Permissive(StructuredModel):
    """`Any` is one of the few annotations that lets two classes reach a field."""

    pet: Optional[Any] = None


class PermissiveList(StructuredModel):
    pets: Optional[List[Any]] = None


class TestObjectGradeScoring:
    """The field must be judged as an object, not as its `str()`.

    Levenshtein over `str(model)` is not a metric on objects: the field names
    are boilerplate identical on both sides, so the score can never fall far.
    """

    def test_an_entirely_wrong_model_scores_zero(self):
        result = Nested(kid=Plain(sku="aaa", qty=1)).compare_with(
            Nested(kid=Plain(sku="zzz", qty=99)), include_confusion_matrix=True
        )
        assert result["field_scores"]["kid"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_the_field_gets_the_object_grade_comparator(self):
        """Not the primitive Levenshtein default, which cannot score an object."""
        info = Nested._get_comparison_info("kid")
        assert type(info.comparator).__name__ == "ANLSStarComparator"
        assert info.clip_under_threshold is False

    def test_a_list_of_plain_models_gets_it_too(self):
        """Keyed on the ELEMENT, so the list form reads the same config."""
        info = NestedList._get_comparison_info("rows")
        assert type(info.comparator).__name__ == "ANLSStarComparator"

    def test_a_dict_inside_a_plain_model_is_not_key_order_sensitive(self):
        """What the primitive default got wrong, stated directly."""

        class Wrapper(BaseModel):
            payload: Optional[dict] = None

        class Holder(StructuredModel):
            w: Optional[Wrapper] = None

        a = {"alpha": "1", "beta": "2", "gamma": "3"}
        reordered = {"gamma": "3", "alpha": "1", "beta": "2"}
        assert Holder(w=Wrapper(payload=a)).compare_with(
            Holder(w=Wrapper(payload=reordered))
        )["field_scores"]["w"] == pytest.approx(1.0)


class TestADifferentClassIsAFalseDiscovery:
    """A wrong class is one false discovery, whatever its field names carry."""

    def test_unrelated_classes_with_identical_content_do_not_match(self):
        result = Permissive(pet=Cat(name="rex")).compare_with(
            Permissive(pet=Dog(name="rex")), include_confusion_matrix=True
        )
        assert result["field_scores"]["pet"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_a_subclass_against_its_base_does_not_match(self):
        """`Optional[Base]` accepts a `Sub`, so this is reachable normally."""
        result = Permissive(pet=Base(a="x")).compare_with(
            Permissive(pet=Sub(a="x")), include_confusion_matrix=True
        )
        assert result["field_scores"]["pet"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_it_warns_rather_than_raising(self):
        """Which class arrives is prediction data; raising would end a corpus run."""
        with pytest.warns(UserWarning, match="compared a Cat against a Dog"):
            Permissive(pet=Cat(name="rex")).compare_with(
                Permissive(pet=Dog(name="rex"))
            )

    def test_the_same_class_is_unaffected(self):
        assert Permissive(pet=Cat(name="rex")).compare_with(
            Permissive(pet=Cat(name="rex"))
        )["field_scores"]["pet"] == pytest.approx(1.0)


class TestTheClassGateHoldsInEveryPath:
    """Four readers ask the same question and must give the same answer.

    `compare()` feeds the Hungarian cost matrix, so a disagreement here is not
    cosmetic: a list pairs two items at zero cost and then reports the field as
    a mismatch, the contradiction #233 forbids.
    """

    @pytest.mark.parametrize(
        "gt, pred",
        ((Cat(name="rex"), Dog(name="rex")), (Base(a="x"), Sub(a="x"))),
    )
    def test_compare_agrees_with_compare_with(self, gt, pred):
        raw = Permissive(pet=gt).compare(Permissive(pet=pred))
        scored = Permissive(pet=gt).compare_with(Permissive(pet=pred))
        assert raw == pytest.approx(0.0)
        assert scored["field_scores"]["pet"] == pytest.approx(0.0)

    def test_the_list_form_agrees_with_the_singular_form(self):
        singular = Permissive(pet=Cat(name="rex")).compare_with(
            Permissive(pet=Dog(name="rex"))
        )["field_scores"]["pet"]
        listed = PermissiveList(pets=[Cat(name="rex")]).compare_with(
            PermissiveList(pets=[Dog(name="rex")])
        )["field_scores"]["pets"]
        assert singular == pytest.approx(listed) == pytest.approx(0.0)

    def test_one_wrong_element_among_several_is_one_false_discovery(self):
        """Issue #321's shape: it used to report tp=2 and a perfect score."""
        result = PermissiveList(pets=[Cat(name="a"), Cat(name="b")]).compare_with(
            PermissiveList(pets=[Cat(name="a"), Dog(name="b")]),
            include_confusion_matrix=True,
        )
        overall = result["confusion_matrix"]["overall"]
        assert (overall["tp"], overall["fd"]) == (1, 1)
        assert result["field_scores"]["pets"] == pytest.approx(0.5)

    def test_a_heterogeneous_list_of_matching_classes_is_untouched(self):
        """The gate must not penalise a list that is simply mixed and correct."""
        assert PermissiveList(pets=[Cat(name="a"), Dog(name="b")]).compare_with(
            PermissiveList(pets=[Cat(name="a"), Dog(name="b")])
        )["field_scores"]["pets"] == pytest.approx(1.0)
