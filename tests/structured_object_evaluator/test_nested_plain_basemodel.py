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

Two pydantic models of DIFFERENT classes are a false discovery, whatever their
field names, and that includes a `StructuredModel` against a plain `BaseModel`:
`CASE 3` requires BOTH sides to be a `StructuredModel`, so the mixed pair falls
to `CASE 5` and nothing downstream is waiting to dispatch it. A correctly
annotated field never reaches the rule: pydantic refuses a `Dog` for an
`Optional[Cat]` field at construction, so it fires only where the annotation
permitted both (`Union[Cat, Dog]`, `Any`, `object`) or where a subclass was
supplied for its base. It warns rather than raising, because which class arrives
is prediction data and raising would end a corpus run on document N.

That rule and the "can this comparator score an object at all" rule are composed
in ONE place, `ConfigurationHelper.can_compare_object_pair`, because five readers
ask the question -- `ComparisonDispatcher` CASE 4 and CASE 5, both
`compare_field_raw` implementations, and a list element arriving through the
Hungarian cost matrix. Each rule was added to the dispatcher first and to
`compare_field_raw` second, and both times the result was `compare()` returning a
non-zero score for a pair `compare_with()` called a false discovery: the #233
disagreement, deciding Hungarian pairings before `compare_with` overrules them.
`TestTheGateHoldsInEveryPath` is the table that holds both rules to both readers.

The configuration behind all of this is installed at CLASS-DEFINITION time by
`StructuredModel._install_object_grade_comparators`, not at read time, so
`to_json_schema()` reports what the engine actually uses. Read-time-only
substitution is why the exported schema said `LevenshteinComparator` with
clipping on for fields the engine scored with ANLS* and clipping off. All four
object-grade shapes -- a mapping, a plain model, and a list of either -- now take
that one path; see `TestTheExportedConfigurationMatchesTheEngine`.

See https://github.com/awslabs/stickler/issues/318 and
https://github.com/awslabs/stickler/issues/321
"""

from typing import Annotated, Any, Dict, List, Optional, Union

import pytest
from pydantic import BaseModel, Field

from stickler.comparators.anls import ANLSStarComparator
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
        """ANLS* over the object's keys, so one wrong field is not zero.

        Named for the DEFAULT comparator, which for a plain-model field is
        `ANLSStarComparator` at object grade -- not the Levenshtein this docstring
        used to claim. That is the whole point of the substitution: partial credit
        counted in fields, not in characters.
        """
        score = Nested(kid=Plain(sku="a", qty=1)).compare_with(
            Nested(kid=Plain(sku="b", qty=1))
        )["field_scores"]["kid"]
        # Exactly one of the two fields differs, so object-grade scoring gives
        # 0.5. A range assertion here is what let the Levenshtein floor hide: on
        # the rendered string the same pair scored 0.9375, which also satisfies
        # `0.0 < score < 1.0`.
        assert score == pytest.approx(0.5)


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
        """Comparator DECLARED, so the class rule is the only thing measured.

        A bare `Optional[Any]` also scores 0.0, through the `can_score_object`
        refusal, so this test passed with the class rule deleted outright and
        proved nothing about it.
        """

        class Holder(StructuredModel):
            pet: Optional[Any] = ComparableField(
                comparator=ANLSStarComparator(), default=None
            )

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
        """`StructuredModel` subclasses `BaseModel`, so this pair reaches here.

        `CASE 3` requires BOTH sides to be a `StructuredModel`, so a mixed pair
        falls to `CASE 5` and there is no further dispatch waiting for it. An
        earlier version of the gate waved the pair through on the assumption that
        there was, and scored two different classes -- one not even the same KIND
        of model -- 1.0 and a true positive, where `dev` reported a false
        discovery.

        The comparator is DECLARED, and it is the one the `can_score_object`
        warning recommends. With a bare `Any` this test passed on the other
        refusal and proved nothing about the class gate, which is also the route
        a user following that advice takes to get here.
        """

        class PlainShape(BaseModel):
            name: Optional[str] = None

        class StructuredShape(StructuredModel):
            name: Optional[str] = ComparableField(default=None)

        class Holder(StructuredModel):
            thing: Optional[Any] = ComparableField(
                comparator=ANLSStarComparator(), default=None
            )

        pair = (StructuredShape(name="rex"), PlainShape(name="rex"))
        for gt, pred in (pair, pair[::-1]):
            result = Holder(thing=gt).compare_with(
                Holder(thing=pred), include_confusion_matrix=True
            )
            assert result["field_scores"]["thing"] == pytest.approx(0.0)
            assert result["confusion_matrix"]["overall"]["fd"] == 1
            assert result["confusion_matrix"]["overall"]["tp"] == 0

    def test_that_pair_is_refused_by_the_class_gate_and_says_so(self):
        """Names the rule that fired, so the test cannot drift onto the other one."""

        class PlainShape(BaseModel):
            name: Optional[str] = None

        class StructuredShape(StructuredModel):
            name: Optional[str] = ComparableField(default=None)

        class Holder(StructuredModel):
            thing: Optional[Any] = ComparableField(
                comparator=ANLSStarComparator(), default=None
            )

        with pytest.warns(
            UserWarning, match="compared a StructuredShape against a PlainShape"
        ):
            Holder(thing=StructuredShape(name="rex")).compare_with(
                Holder(thing=PlainShape(name="rex"))
            )

    def test_two_structured_models_of_one_class_are_untouched(self):
        """The gate must not reach the structural path it sits beside.

        `CASE 3` takes a same-class `StructuredModel` pair before `CASE 5` can
        see it, and `ComparisonHelper.compare_field_raw` does the same. Asserted
        rather than assumed, because closing the hole above is only safe if this
        holds.
        """

        class StructuredShape(StructuredModel):
            name: Optional[str] = ComparableField(default=None)

        class Holder(StructuredModel):
            thing: Optional[StructuredShape] = ComparableField(default=None)

        result = Holder(thing=StructuredShape(name="rex")).compare_with(
            Holder(thing=StructuredShape(name="rex")), include_confusion_matrix=True
        )
        assert result["field_scores"]["thing"] == pytest.approx(1.0)
        assert result["confusion_matrix"]["fields"]["thing"]["fields"], (
            "a same-class StructuredModel pair must keep its per-field breakdown"
        )

    def test_the_same_class_on_both_sides_still_scores(self):
        """The guard must not reject the ordinary case.

        The comparator is named here on purpose. A bare `Any` field keeps the
        primitive Levenshtein default, which is refused for an object in its own
        right, so this would pass at 0.0 for the wrong reason and prove nothing
        about the class guard.
        """

        class Holder(StructuredModel):
            pet: Optional[Any] = ComparableField(
                comparator=ANLSStarComparator(), default=None
            )

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
        """Comparator declared, so only the class rule can produce the 0.0."""

        class Holder(StructuredModel):
            v: Optional[Any] = ComparableField(
                comparator=ANLSStarComparator(), default=None
            )

        result = Holder(v=self.Base(a="x")).compare_with(
            Holder(v=self.Sub(a="x", b=extra)), include_confusion_matrix=True
        )

        assert result["field_scores"]["v"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_the_renderings_differ_even_with_the_extra_field_unset(self):
        """The measurement behind the choice."""
        assert str(self.Base(a="x")) != str(self.Sub(a="x"))

    def test_a_subclass_against_itself_still_scores(self):
        """Comparator named for the same reason as above: isolate the guard."""

        class Holder(StructuredModel):
            v: Optional[Any] = ComparableField(
                comparator=ANLSStarComparator(), default=None
            )

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
    """Two classes reach one field, and the field can still score an object.

    A bare `Any` would let two classes through but would ALSO keep the primitive
    Levenshtein default, which is refused for an object (see
    `TestAnUnscoreableAnnotationIsRefused`). Both refusals return 0.0, so a bare
    `Any` cannot show which one fired. Declaring the comparator separates them:
    everything that scores 0.0 here does so because of the class gate.
    """

    pet: Optional[Union[Cat, Dog, Base, Sub]] = ComparableField(
        comparator=ANLSStarComparator(), default=None
    )


class PermissiveList(StructuredModel):
    pets: Optional[List[Union[Cat, Dog]]] = ComparableField(
        comparator=ANLSStarComparator(), default=None
    )


class Inherited(StructuredModel):
    """The shape that needs no `Any` at all: a base annotation takes a subclass.

    `Optional[Base]` accepts a `Sub` by ordinary subtyping and gets ANLS* from
    its declared annotation, so the class gate is the only thing standing
    between `Base(a="x")` / `Sub(a="x")` and a reported 1.0. This is why the gate
    is not merely defensive.
    """

    kid: Optional[Base] = None


class InheritedList(StructuredModel):
    kids: Optional[List[Base]] = None


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

    def test_a_declared_base_annotation_taking_a_subclass_is_refused(self):
        """The gate's load-bearing case: no `Any`, no explicit comparator."""
        result = Inherited(kid=Base(a="x")).compare_with(
            Inherited(kid=Sub(a="x")), include_confusion_matrix=True
        )
        assert result["field_scores"]["kid"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_that_same_annotation_still_scores_its_own_class(self):
        assert Inherited(kid=Base(a="x")).compare_with(Inherited(kid=Base(a="x")))[
            "field_scores"
        ]["kid"] == pytest.approx(1.0)

    def test_the_list_form_of_that_annotation_agrees(self):
        assert InheritedList(kids=[Base(a="x")]).compare_with(
            InheritedList(kids=[Sub(a="x")])
        )["field_scores"]["kids"] == pytest.approx(0.0)
        assert InheritedList(kids=[Base(a="x")]).compare_with(
            InheritedList(kids=[Base(a="x")])
        )["field_scores"]["kids"] == pytest.approx(1.0)


class TestTheGateHoldsInEveryPath:
    """Four readers ask the same question and must give the same answer.

    `compare()` feeds the Hungarian cost matrix, so a disagreement here is not
    cosmetic: a list pairs two items at zero cost and then reports the field as
    a mismatch, the contradiction #233 forbids.

    Parametrised over BOTH rules the gate composes, not just the class one. Each
    rule was added to the dispatcher first and to `compare_field_raw` second, and
    each time the gap was invisible because this test only exercised the other
    rule. `ConfigurationHelper.can_compare_object_pair` now composes them in one
    place so a reader cannot hold one and miss the other, and the table below is
    what checks that claim.
    """

    #: Holder whose annotation declares no model type, so nothing can install an
    #: object-grade comparator and the primitive Levenshtein default survives.
    #: The `can_score_object` half of the gate is the only thing refusing here.
    Unscoreable = type(
        "Unscoreable",
        (StructuredModel,),
        {"__annotations__": {"pet": Optional[Any]}, "pet": None},
    )

    @pytest.mark.parametrize(
        "holder, gt, pred",
        (
            # The class half: comparator declared, so the class rule is alone.
            pytest.param(Permissive, Cat(name="rex"), Dog(name="rex"), id="cat-vs-dog"),
            pytest.param(Permissive, Base(a="x"), Sub(a="x"), id="base-vs-subclass"),
            # The comparator half: same class on both sides, so the class rule
            # cannot fire. The identical pair is the sharper of the two -- both
            # readers must call two EQUAL models unscoreable, and `compare()`
            # answered 1.0 here while `compare_with()` reported fd=1.
            pytest.param(
                Unscoreable,
                Plain(sku="a", qty=1),
                Plain(sku="a", qty=1),
                id="identical-but-unscoreable",
            ),
            pytest.param(
                Unscoreable,
                Plain(sku="a", qty=1),
                Plain(sku="z", qty=99),
                id="all-wrong-and-unscoreable",
            ),
        ),
    )
    def test_compare_agrees_with_compare_with(self, holder, gt, pred):
        raw = holder(pet=gt).compare(holder(pet=pred))
        scored = holder(pet=gt).compare_with(
            holder(pet=pred), include_confusion_matrix=True
        )
        assert raw == pytest.approx(0.0)
        assert scored["field_scores"]["pet"] == pytest.approx(0.0)
        assert scored["confusion_matrix"]["overall"]["fd"] == 1

    @pytest.mark.parametrize(
        "holder, gt, pred",
        (
            pytest.param(Permissive, Cat(name="rex"), Dog(name="rex"), id="cat-vs-dog"),
            pytest.param(
                Unscoreable,
                Plain(sku="a", qty=1),
                Plain(sku="a", qty=1),
                id="identical-but-unscoreable",
            ),
        ),
    )
    def test_the_list_element_reader_agrees_with_the_singular_one(
        self, holder, gt, pred
    ):
        """The third reader: an element arriving through the cost matrix.

        A list element never reaches the field's own dispatch. It reaches the
        comparator directly, which is why `_ClassGatedComparator` exists, and it
        is the reader most easily left behind -- the element gate held only the
        class rule for one round, then read only the ground-truth side for another.
        """
        listed = type(
            f"Listed{holder.__name__}",
            (StructuredModel,),
            {
                "__annotations__": {"pets": Optional[List[Any]]},
                "pets": holder.model_fields["pet"].default,
            },
        )
        singular = holder(pet=gt).compare_with(holder(pet=pred))["field_scores"]["pet"]
        element = listed(pets=[gt]).compare_with(listed(pets=[pred]))["field_scores"][
            "pets"
        ]
        assert singular == pytest.approx(0.0)
        assert element == pytest.approx(singular)

    def test_the_mapping_reader_agrees_too(self):
        """The fourth reader: a dict pair, gated by the same call.

        `StructuredModel.compare_field_raw` handles the mapping half one frame
        above `ComparisonHelper.compare_field_raw`, and it was the site the class
        rule reached last. Both now make the one
        `can_compare_object_pair` call, so this is what holds the mapping shape to
        the same agreement the model shape gets.
        """

        class Holder(StructuredModel):
            m: Optional[Any] = ComparableField(default=None)

        pair = ({"a": "1", "b": "2"}, {"a": "1", "b": "2"})
        raw = Holder(m=pair[0]).compare(Holder(m=pair[1]))
        scored = Holder(m=pair[0]).compare_with(
            Holder(m=pair[1]), include_confusion_matrix=True
        )
        # Identical mappings, and both readers must still refuse: the annotation
        # declares no mapping, so nothing could install a structural comparator.
        assert raw == pytest.approx(0.0)
        assert scored["field_scores"]["m"] == pytest.approx(0.0)
        assert scored["confusion_matrix"]["overall"]["fd"] == 1

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


class TestAnUnscoreableAnnotationIsRefused:
    """`Any`, `object` and a multi-arm `Union` are refused, not guessed at.

    The object-grade configuration is keyed on the annotation, so a field that
    declares no model type keeps the primitive `LevenshteinComparator`. On a
    model that default is not merely wrong, it is confidently wrong: edit
    distance over `str(model)` compares field-name boilerplate that is identical
    on both sides, so it never scores low.

        LineItem(quantity=2, unit_price=10.5, currency='USD')
          vs LineItem(quantity=9, unit_price=99.9, currency='EUR')  ->  0.8293
          (measured on a three-field model; the two-field `Plain` below scores
          the same way for the same reason)

    0.8293 clears the default threshold, so every value being wrong was reported
    as a TRUE POSITIVE. Refusing is strictly better than a number that confident
    and that wrong, and it is the treatment a mapping in the same position has
    always had -- `dev` scores two IDENTICAL dicts in an `Any` field 0.0 with
    `fd=1` for exactly this reason. Plain models now agree with mappings.
    """

    ALL_WRONG = (
        Plain(sku="a", qty=1),
        Plain(sku="z", qty=99),
    )

    @pytest.mark.parametrize(
        "annotation",
        (Optional[Any], Optional[object], Optional[Union[Plain, str]]),
    )
    def test_a_wholly_wrong_model_is_not_a_true_positive(self, annotation):
        model = type(
            "Holder",
            (StructuredModel,),
            {"__annotations__": {"f": annotation}, "f": None},
        )
        result = model(f=self.ALL_WRONG[0]).compare_with(
            model(f=self.ALL_WRONG[1]), include_confusion_matrix=True
        )
        assert result["field_scores"]["f"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["tp"] == 0
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_the_list_form_is_refused_too(self):
        """`List[Any]` reaches the comparator through the Hungarian cost matrix.

        A separate path from the singular one, and it silently matched at 0.8293
        after the singular form was already refused.
        """
        model = type(
            "Holder",
            (StructuredModel,),
            {"__annotations__": {"f": Optional[List[Any]]}, "f": None},
        )
        result = model(f=[self.ALL_WRONG[0]]).compare_with(
            model(f=[self.ALL_WRONG[1]]), include_confusion_matrix=True
        )
        assert result["field_scores"]["f"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_compare_refuses_it_too(self):
        """`compare()` is the other reader, and it fills the cost matrix.

        The refusal reached `compare_with` first and `compare_field_raw` two
        rounds later, so for two releases `compare()` scored an identical pair
        1.0 -- and 0.6875 for a wholly wrong one -- while `compare_with()` called
        both a false discovery. A non-zero cost-matrix entry decides pairings
        that `compare_with` then overrules.
        """

        class Holder(StructuredModel):
            f: Optional[Any] = None

        identical = Holder(f=self.ALL_WRONG[0]).compare(Holder(f=self.ALL_WRONG[0]))
        wrong = Holder(f=self.ALL_WRONG[0]).compare(Holder(f=self.ALL_WRONG[1]))
        assert identical == pytest.approx(0.0)
        assert wrong == pytest.approx(0.0)

    def test_the_element_refusal_reads_both_sides(self):
        """Swapping ground truth and prediction must not change the answer.

        The element gate tested only `str1`, so a plain model arriving as the
        PREDICTION skipped it: `gt=[model] pred=[str]` scored 0.0 while the swap
        scored 1.0 -- a perfect match, and a true positive, for a pair the code
        had just decided it could not score. `_holds_a_plain_model` installs the
        wrapper by scanning both lists, so the gate has to read both too.
        """
        model = type(
            "Holder",
            (StructuredModel,),
            {"__annotations__": {"f": Optional[List[Any]]}, "f": None},
        )
        rendered = str(self.ALL_WRONG[0])
        forward = model(f=[self.ALL_WRONG[0]]).compare_with(
            model(f=[rendered]), include_confusion_matrix=True
        )
        backward = model(f=[rendered]).compare_with(
            model(f=[self.ALL_WRONG[0]]), include_confusion_matrix=True
        )
        assert forward["field_scores"]["f"] == pytest.approx(0.0)
        assert backward["field_scores"]["f"] == pytest.approx(0.0)
        assert forward["confusion_matrix"]["overall"]["fd"] == 1
        assert backward["confusion_matrix"]["overall"]["fd"] == 1

    def test_the_rendered_form_really_is_what_flipped_it(self):
        """The measurement behind the test above, so its point is visible.

        The string is the model's own rendering, which is why the old gate scored
        the swap 1.0: Levenshtein saw two equal strings once the model side was
        coerced. Nothing about the pair being unscoreable had changed.
        """
        assert str(self.ALL_WRONG[0]) == "sku='a' qty=1"

    def test_it_says_what_to_do_about_it(self):
        """A refusal with no remedy is just a wrong number with extra steps."""

        class Holder(StructuredModel):
            f: Optional[Any] = None

        with pytest.warns(UserWarning, match="holds a pydantic model"):
            Holder(f=self.ALL_WRONG[0]).compare_with(Holder(f=self.ALL_WRONG[1]))

    def test_naming_the_annotation_is_the_remedy(self):
        """The advice in the warning has to actually work."""

        class Holder(StructuredModel):
            f: Optional[Plain] = None

        assert Holder(f=self.ALL_WRONG[0]).compare_with(Holder(f=self.ALL_WRONG[0]))[
            "field_scores"
        ]["f"] == pytest.approx(1.0)

    def test_naming_the_comparator_is_the_other_remedy(self):
        class Holder(StructuredModel):
            f: Optional[Any] = ComparableField(
                comparator=ANLSStarComparator(), default=None
            )

        assert Holder(f=self.ALL_WRONG[0]).compare_with(Holder(f=self.ALL_WRONG[0]))[
            "field_scores"
        ]["f"] == pytest.approx(1.0)


class TestAnExplicitClipSettingSurvivesTheSubstitution:
    """The object-grade default must not overwrite a decision the user wrote.

    Installing ANLS* also turns clipping off, because a container keeps its
    partial score. That is right as a DEFAULT and wrong as an override: the
    substitution used to set `clip_under_threshold=False` outright, discarding an
    explicit `True`.

    `_install_object_grade_comparators` gates the amendment on `_clip_explicit`,
    so a `dict` field carrying an explicit `True` never reached the line that
    clobbered it, while a plain-model field did. The same declared setting was
    honoured on one shape and dropped on the other, which is the divergence
    between mappings and plain models that #318 exists to remove. Both shapes now
    read the one answer `ConfigurationHelper.object_grade_clip` gives.
    """

    class Addr(BaseModel):
        city: Optional[str] = None
        zip: Optional[str] = None

    def _half_right(self, model):
        return model(kid=self.Addr(city="a", zip="1")).compare_with(
            model(kid=self.Addr(city="a", zip="2"))
        )["field_scores"]["kid"]

    def test_an_explicit_clip_is_honoured(self):
        class Doc(StructuredModel):
            kid: Optional["TestAnExplicitClipSettingSurvivesTheSubstitution.Addr"] = (
                ComparableField(threshold=0.9, clip_under_threshold=True, default=None)
            )

        assert Doc._get_comparison_info("kid").clip_under_threshold is True
        assert self._half_right(Doc) == pytest.approx(0.0)

    def test_an_unstated_clip_still_defaults_to_off(self):
        """The default is the whole reason the substitution touches clip at all."""

        class Doc(StructuredModel):
            kid: Optional["TestAnExplicitClipSettingSurvivesTheSubstitution.Addr"] = (
                ComparableField(threshold=0.9, default=None)
            )

        assert Doc._get_comparison_info("kid").clip_under_threshold is False
        assert self._half_right(Doc) == pytest.approx(0.5)

    def test_a_bare_annotation_defaults_to_off_too(self):
        class Doc(StructuredModel):
            kid: Optional["TestAnExplicitClipSettingSurvivesTheSubstitution.Addr"] = (
                None
            )

        assert Doc._get_comparison_info("kid").clip_under_threshold is False

    def test_a_dict_field_answers_the_same_way(self):
        """The consistency being claimed, asserted rather than assumed."""

        class Doc(StructuredModel):
            explicit: Optional[Dict[str, str]] = ComparableField(
                threshold=0.9, clip_under_threshold=True, default=None
            )
            defaulted: Optional[Dict[str, str]] = ComparableField(
                threshold=0.9, default=None
            )

        assert Doc._get_comparison_info("explicit").clip_under_threshold is True
        assert Doc._get_comparison_info("defaulted").clip_under_threshold is False


class TestTheExportedConfigurationMatchesTheEngine:
    """`to_json_schema()` must report the comparator the engine actually runs.

    The object-grade substitution has two possible homes, and only one of them is
    honest. `StructuredModel._install_object_grade_comparators` writes it into the
    `FieldInfo` while the class is being defined, and that `FieldInfo` is the
    object `json_schema_extra` renders as `x-comparison`, so the exported schema,
    `explain()`, the HTML reports and the engine all read one answer.
    `ConfigurationHelper.get_comparison_info` can also apply it at read time, and
    when that was the ONLY place it happened the exported schema disagreed with
    the engine:

        item   (plain model)   engine ANLS*/clip off   schema Levenshtein/clip on
        items  (list of them)  engine ANLS*/clip off   schema Levenshtein/clip on
        metas  (list of dicts) engine ANLS*/clip off   schema Levenshtein/clip on
        meta   (dict)          engine ANLS*/clip off   schema ANLS*/clip off

    The dict row agreed because the singular mapping was the one shape the
    definition-time path handled. The plain-model rows were introduced by #318;
    the `List[Dict[...]]` row was inherited from before it. All four are now on
    one path, which is what this class pins.
    """

    class Item(BaseModel):
        quantity: Optional[int] = None

    def _model(self):
        Item = self.Item

        class Invoice(StructuredModel):
            item: Optional[Item] = ComparableField(default=None)
            meta: Optional[Dict[str, str]] = ComparableField(default=None)
            items: Optional[List[Item]] = ComparableField(default=None)
            metas: Optional[List[Dict[str, str]]] = ComparableField(default=None)
            kept: Optional[Item] = ComparableField(
                default=None, clip_under_threshold=True
            )
            named: Optional[Item] = ComparableField(
                default=None, comparator=ExactComparator()
            )
            scalar: Optional[str] = ComparableField(default=None)

        return Invoice

    @pytest.mark.parametrize(
        "field", ("item", "meta", "items", "metas", "kept", "named", "scalar")
    )
    def test_the_schema_agrees_with_the_engine(self, field):
        """Whatever the schema states about a field must be what the engine runs.

        `clip_under_threshold` is exported only where a decision was actually made,
        which is #250's rule: the exporter stopped inventing one for a field nobody
        configured. So the assertion is conditional on the key being present, and
        `test_the_clip_key_is_exported_only_where_a_decision_was_made` below pins
        which fields those are. Asserting the key unconditionally would force the
        exporter back to inventing it.
        """
        model = self._model()
        info = model._get_comparison_info(field)
        prop = model.to_json_schema()["properties"][field]
        assert prop["x-aws-stickler-comparator"] == type(info.comparator).__name__
        if "x-aws-stickler-clip-under-threshold" in prop:
            assert (
                prop["x-aws-stickler-clip-under-threshold"] == info.clip_under_threshold
            )

    def test_the_clip_key_is_exported_only_where_a_decision_was_made(self):
        """The other half of the rule above, so neither can drift alone.

        Stickler decides `clip_under_threshold=False` for every object-grade shape,
        and the user decided `True` on `kept`; all of those are exported. A bare
        scalar carries no decision from anyone, so exporting one would state a
        setting nobody chose, and re-importing it would make that invention
        explicit.
        """
        prop = self._model().to_json_schema()["properties"]
        key = "x-aws-stickler-clip-under-threshold"
        for field in ("item", "meta", "items", "metas", "kept"):
            assert key in prop[field], f"{field} carries a decision and must export it"
        assert key not in prop["scalar"], (
            "a bare scalar field has no clip decision to export"
        )

    @pytest.mark.parametrize("field", ("item", "meta", "items", "metas"))
    def test_all_four_object_grade_shapes_get_the_object_grade_default(self, field):
        """The four shapes the substitution covers, named one by one.

        Asserting agreement alone would also pass if every shape were left on
        Levenshtein, since the engine would then agree with the schema about the
        wrong answer.
        """
        info = self._model()._get_comparison_info(field)
        assert type(info.comparator).__name__ == "ANLSStarComparator"
        assert info.clip_under_threshold is False

    def test_an_explicit_clip_survives_into_the_schema(self):
        """The decision the user wrote, visible where they can check it."""
        model = self._model()
        assert model._get_comparison_info("kept").clip_under_threshold is True
        prop = model.to_json_schema()["properties"]["kept"]
        assert prop["x-aws-stickler-clip-under-threshold"] is True
        assert prop["x-aws-stickler-comparator"] == "ANLSStarComparator"

    def test_an_explicit_comparator_survives_into_the_schema(self):
        """Never overridden: an explicit `comparator=` is consent by definition."""
        model = self._model()
        assert type(model._get_comparison_info("named").comparator).__name__ == (
            "ExactComparator"
        )
        assert (
            model.to_json_schema()["properties"]["named"]["x-aws-stickler-comparator"]
            == "ExactComparator"
        )

    def test_a_scalar_field_is_left_alone(self):
        """The control: the substitution must be keyed on the annotation."""
        model = self._model()
        info = model._get_comparison_info("scalar")
        assert type(info.comparator).__name__ == "LevenshteinComparator"
        assert info.clip_under_threshold is True

    def test_a_shared_comparable_field_is_not_rewritten_by_the_other_field(self):
        """One `ComparableField(...)` can be bound to two fields.

        Pydantic does not clone the `json_schema_extra` closure, so substituting
        in place retroactively rewrote the sibling -- order-dependently on which
        class was defined first. Widening the substitution from one annotation to
        four widens this hazard with it, so it is asserted here rather than only
        in the comment that records it.
        """
        Item = self.Item
        shared = ComparableField(threshold=0.8, default=None)

        class Scalar(StructuredModel):
            v: Optional[str] = shared

        class Objectish(StructuredModel):
            v: Optional[Item] = shared

        assert type(Scalar._get_comparison_info("v").comparator).__name__ == (
            "LevenshteinComparator"
        )
        assert Scalar._get_comparison_info("v").clip_under_threshold is True
        assert type(Objectish._get_comparison_info("v").comparator).__name__ == (
            "ANLSStarComparator"
        )
        assert Objectish._get_comparison_info("v").clip_under_threshold is False

    def test_a_deferred_annotation_still_scores_correctly(self):
        """A forward reference resolved after the class is built.

        This is the one shape the definition-time pass cannot see: the annotation
        is a `ForwardRef` while the class is being built, so it is not recognised
        as object-grade, and pydantic resolves it later. The read-time fallback in
        `get_comparison_info` is what covers it, and asserting the SCORE is what
        proves the fallback is live rather than decorative -- an earlier version of
        this test compared two helpers to each other and passed with the fallback
        deleted outright.
        """

        class Doc(StructuredModel):
            kid: Optional["DeferredLeaf"] = ComparableField(default=None)

        class DeferredLeaf(BaseModel):
            a: Optional[str] = None
            b: Optional[str] = None

        Doc.model_rebuild(force=True)
        info = Doc._get_comparison_info("kid")
        assert type(info.comparator).__name__ == "ANLSStarComparator"
        assert info.clip_under_threshold is False
        # Object-grade scoring, not edit distance over `str(model)`: one of two
        # fields wrong is 0.5, and identical is a true positive rather than the
        # false discovery the scalar default would report.
        half = Doc(kid=DeferredLeaf(a="x", b="1")).compare_with(
            Doc(kid=DeferredLeaf(a="x", b="2")), include_confusion_matrix=True
        )
        same = Doc(kid=DeferredLeaf(a="x", b="1")).compare_with(
            Doc(kid=DeferredLeaf(a="x", b="1")), include_confusion_matrix=True
        )
        assert half["field_scores"]["kid"] == pytest.approx(0.5)
        assert same["field_scores"]["kid"] == pytest.approx(1.0)
        assert same["confusion_matrix"]["overall"]["tp"] == 1

    def test_a_deferred_annotation_is_read_the_same_way_twice(self):
        """Reading the configuration BEFORE the rebuild must not fix the answer.

        The object-grade classification is memoised per (class, field), and an
        annotation is not fixed for the life of a class: it is a `ForwardRef` until
        pydantic resolves it. Caching the False computed from the unresolved form
        made scoring depend on whether anything had looked at the class first --
        two structurally identical models, identical data, 1.0/tp=1 or 0.0/fd=1
        according to call order. The cache remembers the annotation it was
        computed from, so resolution invalidates it.
        """

        class Doc(StructuredModel):
            kid: Optional["EarlyReadLeaf"] = ComparableField(default=None)

        # The early read: this is what `explain()` and `to_json_schema()` do.
        before = type(Doc._get_comparison_info("kid").comparator).__name__

        class EarlyReadLeaf(BaseModel):
            a: Optional[str] = None

        Doc.model_rebuild(force=True)
        after = type(Doc._get_comparison_info("kid").comparator).__name__

        assert before == "LevenshteinComparator", (
            "an unresolved annotation cannot be classified, which is the premise"
        )
        assert after == "ANLSStarComparator", (
            "the memo must not outlive the annotation it was computed from"
        )
        result = Doc(kid=EarlyReadLeaf(a="x")).compare_with(
            Doc(kid=EarlyReadLeaf(a="x")), include_confusion_matrix=True
        )
        assert result["field_scores"]["kid"] == pytest.approx(1.0)
        assert result["confusion_matrix"]["overall"]["tp"] == 1

    def test_a_deferred_annotation_is_the_one_shape_the_schema_cannot_report(self):
        """The limitation, pinned so it cannot widen or vanish unnoticed.

        The definition-time pass is what keeps `to_json_schema()` honest, and it
        runs once, while the class is being built. A forward reference is not
        resolved yet at that point, so the field's metadata is never amended and
        the exported schema keeps the scalar default even though the engine scores
        with ANLS*.

        `dev` behaves identically for a deferred MAPPING annotation, so this is
        the pre-existing cost of substituting at definition time rather than
        something this change introduced; what changed is that all four
        object-grade shapes now share it instead of three of them being wrong in
        the resolved case too. Re-running the pass after `model_rebuild()` is the
        fix, and it needs pydantic's private `_parent_namespace_depth` to be
        compensated for the extra stack frame or local forward references stop
        resolving at all. Tracked separately rather than smuggled in here.
        """

        class Doc(StructuredModel):
            kid: Optional["UnexportedLeaf"] = ComparableField(default=None)

        class UnexportedLeaf(BaseModel):
            a: Optional[str] = None

        Doc.model_rebuild(force=True)
        engine = Doc._get_comparison_info("kid")
        prop = Doc.to_json_schema()["properties"]["kid"]
        assert type(engine.comparator).__name__ == "ANLSStarComparator"
        assert prop["x-aws-stickler-comparator"] == "LevenshteinComparator"


class TestTheFieldConfigurationIsLookedUpOnce:
    """`get_comparison_info` is not free, and it runs per cost-matrix cell.

    It builds a fresh `ANLSStarComparator` on every call -- only the annotation
    predicate is memoised -- so a redundant lookup is real work once per field per
    pair. `CASE 5` asked for the same field a THIRD time in one dispatch, on the
    branch whose own comment argues for memoising exactly this lookup, while
    `STEP 1` had already bound the identical object.
    """

    def test_a_plain_model_field_is_not_looked_up_a_third_time(self):
        from stickler.structured_object_evaluator.models import configuration_helper

        class Item(BaseModel):
            quantity: Optional[int] = None

        class Doc(StructuredModel):
            item: Optional[Item] = ComparableField(default=None)

        original = configuration_helper.ConfigurationHelper.get_comparison_info
        seen = []

        def counting(cls, field_name):
            seen.append(field_name)
            return original(cls, field_name)

        configuration_helper.ConfigurationHelper.get_comparison_info = staticmethod(
            counting
        )
        try:
            Doc(item=Item(quantity=2)).compare_with(Doc(item=Item(quantity=2)))
        finally:
            configuration_helper.ConfigurationHelper.get_comparison_info = staticmethod(
                original
            )

        # STEP 1 in the dispatcher, then `compare_primitive_with_scores`. The
        # third call was the one this removes; asserting the exact count rather
        # than a bound is what stops a fourth arriving unnoticed.
        assert seen.count("item") == 2, seen


class TestAnnotatedDoesNotHideTheAnnotation:
    """`Field(description=...)` on an optional field must not change the score.

    Pydantic strips `Annotated` when it wraps a WHOLE annotation but leaves it on
    a union arm, so `Annotated[List[Leaf], Field(...)] | None` is stored as
    `Optional[Annotated[List[Leaf], FieldInfo]]`. `get_origin` on that arm reports
    `Annotated`, not `list`, so every object-grade predicate answered False and
    the field kept the scalar Levenshtein default.

    On `dev` that was a missed substitution and nothing more. Here it was fatal:
    `_holds_a_plain_model` still finds plain models in the list, so the element
    comparator is still wrapped in `_ClassGatedComparator`, which then refuses
    every pair because Levenshtein is on the object denylist. Two IDENTICAL
    elements became two false discoveries -- `dev` scored 1.0 with `tp=2`, this
    branch scored 0.0 with `fd=2`.

    `_annotation_is_list` in `structured_model.py` documents the same trap for the
    same reason. `ConfigurationHelper.strip_annotation_wrappers` is that lesson
    applied to the four object-grade predicates.
    """

    class Leaf(BaseModel):
        a: Optional[str] = None
        b: Optional[str] = None

    def _models(self):
        Leaf = self.Leaf

        class Bare(StructuredModel):
            v: Optional[List[Leaf]] = ComparableField(default=None)

        class Wrapped(StructuredModel):
            v: Optional[Annotated[List[Leaf], Field(description="d")]] = (
                ComparableField(default=None)
            )

        class Pep604(StructuredModel):
            v: Annotated[List[Leaf], Field(description="d")] | None = ComparableField(
                default=None
            )

        class SingularWrapped(StructuredModel):
            v: Optional[Annotated[Leaf, Field(description="d")]] = ComparableField(
                default=None
            )

        class WrappedMapping(StructuredModel):
            v: Optional[Annotated[Dict[str, str], Field(description="d")]] = (
                ComparableField(default=None)
            )

        return Bare, Wrapped, Pep604, SingularWrapped, WrappedMapping

    @pytest.mark.parametrize("index", range(5))
    def test_every_spelling_gets_the_object_grade_comparator(self, index):
        model = self._models()[index]
        info = model._get_comparison_info("v")
        assert type(info.comparator).__name__ == "ANLSStarComparator"
        assert info.clip_under_threshold is False

    def test_an_identical_wrapped_list_is_not_a_false_discovery(self):
        """The regression, on data that cannot be wrong."""
        Bare, Wrapped, Pep604, _, _ = self._models()
        items = [self.Leaf(a="x", b="1"), self.Leaf(a="y", b="2")]
        for model in (Bare, Wrapped, Pep604):
            result = model(v=list(items)).compare_with(
                model(v=list(items)), include_confusion_matrix=True
            )
            overall = result["confusion_matrix"]["overall"]
            assert result["overall_score"] == pytest.approx(1.0), model.__name__
            assert (overall["tp"], overall["fd"]) == (2, 0), model.__name__

    def test_the_wrapped_and_bare_spellings_score_identically(self):
        """One annotation, one answer: the divergence this work removes."""
        Bare, Wrapped, Pep604, _, _ = self._models()
        gt = [self.Leaf(a="x", b="1")]
        pred = [self.Leaf(a="x", b="2")]
        scores = {
            m.__name__: m(v=list(gt)).compare_with(m(v=list(pred)))["field_scores"]["v"]
            for m in (Bare, Wrapped, Pep604)
        }
        assert len(set(round(s, 6) for s in scores.values())) == 1, scores
        assert scores["Bare"] == pytest.approx(0.5)

    def test_the_singular_wrapped_form_agrees_too(self):
        """Broken on `dev` as well as here, so this one is a fix, not a repair."""
        _, _, _, SingularWrapped, _ = self._models()
        result = SingularWrapped(v=self.Leaf(a="x", b="1")).compare_with(
            SingularWrapped(v=self.Leaf(a="x", b="1")), include_confusion_matrix=True
        )
        assert result["field_scores"]["v"] == pytest.approx(1.0)
        assert result["confusion_matrix"]["overall"]["tp"] == 1


class TestAStructuredModelElementIsNotRefused:
    """The comparator refusal is about PLAIN models, and must stay that way.

    `_ClassGatedComparator` wraps the whole list's element comparator as soon as
    ONE element anywhere in either list is a plain model. If the refusal read
    `isinstance(v, BaseModel)`, a `StructuredModel` element -- which passes that
    test -- was refused as collateral: `[Cat(plain), Note(SM), Note(SM)]` against
    an IDENTICAL copy scored 0.0 with `fd=3`, where `dev` scored 1.0 with `tp=3`.

    The refusal exists because a plain model's ANNOTATION is what installs an
    object-grade comparator, and an undeclared annotation cannot. A
    `StructuredModel` is scored by recursion instead, so the field's comparator is
    not what judges it and has no business refusing it.
    """

    class Cat(BaseModel):
        name: Optional[str] = None

    class Note(StructuredModel):
        text: Optional[str] = ComparableField(default=None)

    def _holder(self):
        Cat, Note = self.Cat, self.Note

        class Doc(StructuredModel):
            items: Optional[List[Union[Cat, Note]]] = ComparableField(default=None)

        return Doc

    def test_a_pure_structured_model_list_is_untouched(self):
        """The control: no plain model, so the wrapper is never installed."""
        Doc = self._holder()
        rows = [self.Note(text=t) for t in "abc"]
        result = Doc(items=list(rows)).compare_with(
            Doc(items=list(rows)), include_confusion_matrix=True
        )
        overall = result["confusion_matrix"]["overall"]
        assert (overall["tp"], overall["fd"]) == (3, 0)

    def test_one_plain_model_does_not_condemn_the_structured_elements(self):
        """The regression: the plain element is refused, the others are not.

        `tp=2 fd=1` is the declared policy -- a multi-arm union cannot configure an
        object-grade comparator for the `Cat`, so that element is refused, and the
        two `Note`s are `StructuredModel`s scored by recursion. `fd=3` was the
        defect; `fd=0` would mean the plain element is no longer refused at all.
        """
        Doc = self._holder()

        def rows():
            return [self.Cat(name="rex"), self.Note(text="a"), self.Note(text="b")]

        result = Doc(items=rows()).compare_with(
            Doc(items=rows()), include_confusion_matrix=True
        )
        overall = result["confusion_matrix"]["overall"]
        assert (overall["tp"], overall["fd"]) == (2, 1)
        assert result["field_scores"]["items"] == pytest.approx(2 / 3)

    def test_the_gate_answers_the_two_element_kinds_differently(self):
        """Stated at the gate, so the distinction cannot be read as incidental."""
        from stickler.comparators.levenshtein import LevenshteinComparator
        from stickler.structured_object_evaluator.models.configuration_helper import (
            ConfigurationHelper,
        )

        Doc = self._holder()
        scalar_default = LevenshteinComparator()
        assert (
            ConfigurationHelper.can_compare_object_pair(
                Doc, "items", scalar_default, self.Note(text="x"), self.Note(text="x")
            )
            is True
        )
        assert (
            ConfigurationHelper.can_compare_object_pair(
                Doc, "items", scalar_default, self.Cat(name="x"), self.Cat(name="x")
            )
            is False
        )


class TestAModelAgainstSomethingElseIsAMismatch:
    """`compare()` must agree with `compare_with()` on a mixed-kind pair.

    `compare_with` has no branch for a plain model against a bare dict: CASE 5
    needs BOTH sides to be a model, so the pair lands in the type-mismatch branch
    and reports `fd=1`. `compare()` reached the field's comparator instead and
    scored the same content 1.0 -- the #233 disagreement again, and again on the
    reader that fills the Hungarian cost matrix. On `dev` the same call raised
    `TypeError`, so this is a crash and a disagreement replaced by one answer.
    """

    class Item(BaseModel):
        a: Optional[str] = None

    def _holder(self):
        Item = self.Item

        class Doc(StructuredModel):
            kid: Optional[Item] = ComparableField(default=None)

        return Doc

    @pytest.mark.parametrize(
        "other", ({"a": "x"}, "a='x'", 7), ids=("dict", "str", "int")
    )
    def test_compare_agrees_with_compare_with(self, other):
        Doc = self._holder()
        raw = Doc(kid=self.Item(a="x")).compare_field_raw("kid", other)
        assert raw == pytest.approx(0.0)

    def test_the_same_kind_on_both_sides_still_scores(self):
        """The gate must not swallow the ordinary case."""
        Doc = self._holder()
        assert Doc(kid=self.Item(a="x")).compare_field_raw(
            "kid", self.Item(a="x")
        ) == pytest.approx(1.0)


class TestTheRuleIsNotYetEnforcedForStructuredModels:
    """The documented gap, pinned so the docs cannot quietly become wrong.

    The rule is that two objects of different classes are a false discovery,
    whatever their attributes say. It is stated in
    `docs/docs/Advanced/classification-logic.md` under "Objects of different
    classes", and this release enforces it for a plain `BaseModel` and for the
    elements of a list of them.

    It is NOT enforced for two `StructuredModel` classes: `CASE 3` requires only
    that BOTH sides be a `StructuredModel` and then recurses field by field, so
    the class gate is never consulted. That is the pre-existing behaviour on `dev`
    rather than a deliberate exception, and closing it touches `CASE 3` and the
    `List[StructuredModel]` Hungarian pairing, so it is its own change.

    This test asserts the CURRENT behaviour, not the desired behaviour. When the
    rule is extended it will fail, which is the point: the failure is the reminder
    to update the two doc sections that describe the gap in the same commit.
    """

    class PetSM(StructuredModel):
        name: Optional[str] = ComparableField(default=None)

    class CatSM(PetSM):
        """A SUBCLASS, which is the only way the gap is reachable on a declared field.

        Pydantic refuses a sibling class outright -- `Optional[PetSM]` will not
        accept a `DogSM` at construction -- so a declared annotation can only reach
        a heterogeneous pair through subtyping. That makes `Pet` against `Cat` the
        exact shape the rule is written about.
        """

        name: Optional[str] = ComparableField(default=None)

    def _holder(self):
        PetSM = self.PetSM

        class Doc(StructuredModel):
            pet: Optional[PetSM] = ComparableField(default=None)

        return Doc

    def test_the_plain_half_of_the_rule_is_enforced(self):
        """The half this release delivers, stated beside the half it does not."""

        class PetPlain(BaseModel):
            name: Optional[str] = None

        class CatPlain(PetPlain):
            name: Optional[str] = None

        class Doc(StructuredModel):
            pet: Optional[PetPlain] = ComparableField(default=None)

        result = Doc(pet=PetPlain(name="rex")).compare_with(
            Doc(pet=CatPlain(name="rex")), include_confusion_matrix=True
        )
        assert result["field_scores"]["pet"] == pytest.approx(0.0)
        assert result["confusion_matrix"]["overall"]["fd"] == 1

    def test_two_structured_model_classes_are_still_scored_field_by_field(self):
        """Currently a true positive. The docs say so; this holds them to it.

        The same `Pet` / `Cat` pair that is `0.0` with `fd=1` when both are plain
        models. Asserting `tp=1` here is deliberately asserting the WRONG answer,
        because an undisclosed gap is worse than a disclosed one.
        """
        Doc = self._holder()
        result = Doc(pet=self.PetSM(name="rex")).compare_with(
            Doc(pet=self.CatSM(name="rex")), include_confusion_matrix=True
        )
        assert result["field_scores"]["pet"] == pytest.approx(1.0)
        assert result["confusion_matrix"]["overall"]["tp"] == 1
        assert result["confusion_matrix"]["overall"]["fd"] == 0

    def test_a_sibling_class_cannot_even_be_constructed(self):
        """Why the fixture above uses a subclass, measured rather than asserted.

        This bounds the gap: on a field that names its model type, only subtyping
        can produce a heterogeneous pair, because pydantic rejects anything else
        before stickler sees it. The unbounded version needs `Any` or a union.
        """

        class DogSM(StructuredModel):
            name: Optional[str] = ComparableField(default=None)

        Doc = self._holder()
        with pytest.raises(Exception, match="PetSM"):
            Doc(pet=DogSM(name="rex"))

    def test_both_doc_sections_disclose_the_gap(self):
        """A gap the docs stop mentioning is a gap that reads as endorsed.

        Wrap-aware: the pages are read with whitespace collapsed, so the sentence
        is found whether or not a reflow moved it across a line break.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        disclosure = "older behaviour rather than a deliberate exception"
        pages = (
            root / "docs" / "docs" / "Advanced" / "classification-logic.md",
            root / "docs" / "docs" / "Guides" / "Comparators" / "README.md",
        )
        for page in pages:
            assert page.exists(), page
            text = " ".join(page.read_text().split())
            assert disclosure in text, (
                f"{page.name} no longer discloses that the different-class rule is "
                f"unenforced for two StructuredModel classes, which the test above "
                f"measures as still true. Remove both together or neither."
            )
