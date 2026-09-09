"""A threshold set on a comparator reaches the field that names it.

`Comparator(threshold=...)` was accepted everywhere and read almost nowhere. The
only functional reader outside ANLS* is `binary_compare()`, which has no callers
in `src/`, so a caller writing

    ComparableField(comparator=LevenshteinComparator(threshold=0.95))

got a field whose verdict threshold was `0.5`. The value was stored on the
comparator, visible in `repr`, and never consulted.

A threshold is only meaningful beside the metric that produced the score: 0.85
means one thing on edit distance and another on a semantic embedding. So a
threshold on the comparator is a statement about the field, and discarding it in
silence is the wrong default.

Only an EXPLICITLY set comparator threshold is adopted. A comparator's own
default is not, because those defaults were never audited as verdict thresholds
and several are wrong for the job: `DateComparator` defaults to `1.0` while
awarding `0.7` partial credit for a year-less match, so adopting it would clip
that feature to zero.

See https://github.com/awslabs/stickler/issues/246
"""

import warnings
from copy import deepcopy
from typing import Dict, List, Optional

import pytest

from stickler.comparators.base import BaseComparator
from stickler.comparators.date import DateComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    _LEGACY_DEFAULT_THRESHOLD,
    ComparableField,
    _named_comparator_threshold,
)
from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)


def _threshold_of(field) -> float:
    """The verdict threshold a model ends up with for a field built this way."""
    model = type(
        "Probe",
        (StructuredModel,),
        {"__annotations__": {"value": Optional[str]}, "value": field},
    )
    return model._get_comparison_info("value").threshold


class TestAnExplicitComparatorThresholdIsAdopted:
    def test_a_threshold_set_on_the_comparator_reaches_the_field(self):
        assert (
            _threshold_of(
                ComparableField(
                    comparator=LevenshteinComparator(threshold=0.95), default=None
                )
            )
            == 0.95
        )

    @pytest.mark.parametrize("declared", (0.6, 0.75, 0.99))
    def test_it_tracks_whatever_value_the_caller_chose(self, declared):
        assert (
            _threshold_of(
                ComparableField(
                    comparator=ExactComparator(threshold=declared), default=None
                )
            )
            == declared
        )

    def test_a_threshold_on_the_field_always_wins(self):
        """Two statements of intent, and the more specific one governs."""
        assert (
            _threshold_of(
                ComparableField(
                    comparator=LevenshteinComparator(threshold=0.95),
                    threshold=0.8,
                    default=None,
                )
            )
            == 0.8
        )


class TestAComparatorDefaultIsNotAdopted:
    """The distinction the fix rests on, and the reason it is not simpler.

    Adopting any non-None comparator threshold would silently change the
    behaviour of every field that names a comparator without a threshold, using
    values that were never audited for this purpose.
    """

    def test_the_levenshtein_default_is_not_taken(self):
        """Levenshtein defaults to 0.7, which must not become the field's."""
        assert LevenshteinComparator().threshold == 0.7
        assert (
            _threshold_of(
                ComparableField(comparator=LevenshteinComparator(), default=None)
            )
            == _LEGACY_DEFAULT_THRESHOLD
        )

    def test_the_date_default_is_not_taken(self):
        """The case that makes this rule load-bearing rather than cautious.

        `DateComparator` defaults to `1.0` and awards `0.7` for a match with no
        year. Adopting the default would put the verdict threshold above that
        partial credit, clipping the comparator's own feature to zero.
        """
        assert DateComparator().threshold == 1.0
        assert (
            _threshold_of(ComparableField(comparator=DateComparator(), default=None))
            == _LEGACY_DEFAULT_THRESHOLD
        )

    def test_no_comparator_means_the_legacy_default(self):
        assert _threshold_of(ComparableField(default=None)) == _LEGACY_DEFAULT_THRESHOLD

    def test_a_threshold_equal_to_the_class_default_is_still_honoured(self):
        """Naming a value must not depend on which value it happens to be.

        An earlier revision recovered explicitness by comparing the resolved
        threshold against the class default, so `DateComparator(threshold=1.0)`
        was indistinguishable from `DateComparator()` and fell back to the
        legacy 0.5. That made the effective threshold non-monotonic --
        `LevenshteinComparator(threshold=0.69)` gave 0.69 while a *stricter*
        0.70 gave 0.50 -- and it collapsed the strictest and most natural
        spelling of the 1.0-default comparators. `threshold_was_set` records
        the caller's intent at construction instead, so the two cases stay
        apart no matter what number is named.
        """
        assert (
            _threshold_of(
                ComparableField(comparator=DateComparator(threshold=1.0), default=None)
            )
            == 1.0
        )

    @pytest.mark.parametrize("value", [0.69, 0.70, 0.71])
    def test_the_effective_threshold_is_monotonic_across_the_class_default(self, value):
        """0.70 is Levenshtein's default and must not be a hole in the range."""
        assert (
            _threshold_of(
                ComparableField(
                    comparator=LevenshteinComparator(threshold=value), default=None
                )
            )
            == value
        )

    def test_the_strictest_spelling_does_not_become_the_loosest(self):
        """`threshold=1.0` on a 1.0-default comparator used to yield 0.5."""
        for comparator in (
            ExactComparator(threshold=1.0),
            NumericComparator(threshold=1.0),
            DateComparator(threshold=1.0),
        ):
            assert (
                _threshold_of(ComparableField(comparator=comparator, default=None))
                == 1.0
            ), type(comparator).__name__


class TestTheHelperInIsolation:
    def test_it_returns_none_for_a_default_construction(self):
        assert _named_comparator_threshold(LevenshteinComparator()) is None

    def test_it_returns_the_value_for_an_explicit_one(self):
        assert _named_comparator_threshold(LevenshteinComparator(threshold=0.9)) == 0.9

    def test_it_returns_none_for_a_comparator_with_no_threshold_parameter(self):
        """Must not assume every comparator takes a threshold."""

        class NoThreshold:
            pass

        assert _named_comparator_threshold(NoThreshold()) is None


class TestZeroIsAValueNotAnOmission:
    """`0.0` is falsy, so a truthiness test here would silently become `0.5`.

    It is also a documented capture-all sentinel elsewhere in the codebase, so
    the difference between "the caller wrote 0.0" and "the caller wrote nothing"
    is real and observable.
    """

    def test_an_explicit_zero_threshold_survives(self):
        assert _threshold_of(ComparableField(threshold=0.0, default=None)) == 0.0

    def test_an_explicit_zero_still_warns(self):
        """The zero-threshold trap warning must not be lost to the sentinel."""
        with pytest.warns(UserWarning, match="threshold=0.0"):
            type(
                "Probe",
                (StructuredModel,),
                {
                    "__annotations__": {"value": Optional[str]},
                    "value": ComparableField(threshold=0.0, default=None),
                },
            )


class TestExplicitnessIsRecorded:
    """The marker other code needs to tell configured from defaulted.

    `_comparator_explicit` and `_clip_explicit` already existed;
    `_threshold_explicit` completes the set, which is what lets `explain()`
    report provenance accurately (#210) rather than labelling every field
    "explicit".
    """

    @staticmethod
    def _marker(field) -> bool:
        model = type(
            "Probe",
            (StructuredModel,),
            {"__annotations__": {"value": Optional[str]}, "value": field},
        )
        return model.model_fields["value"].json_schema_extra._threshold_explicit

    def test_a_stated_threshold_is_marked_explicit(self):
        assert self._marker(ComparableField(threshold=0.8, default=None)) is True

    def test_an_omitted_threshold_is_not(self):
        assert self._marker(ComparableField(default=None)) is False

    def test_a_threshold_adopted_from_the_comparator_is_not_marked_explicit(self):
        """It was not stated on the field, and the distinction is the point.

        The value is honoured either way; the marker records where it came from.
        """
        assert (
            self._marker(
                ComparableField(comparator=ExactComparator(threshold=0.9), default=None)
            )
            is False
        )


class TestTheAdoptedThresholdSurvivesSerialization:
    """A value that vanishes on export/import is only half honoured."""

    def test_it_round_trips_through_json_schema(self):
        class Doc(StructuredModel):
            value: Optional[str] = ComparableField(
                comparator=LevenshteinComparator(threshold=0.95), default=None
            )

        assert Doc._get_comparison_info("value").threshold == 0.95

        rebuilt = StructuredModel.from_json_schema(Doc.to_json_schema())
        assert rebuilt._get_comparison_info("value").threshold == 0.95

    def test_it_appears_in_the_stickler_config(self):
        class Doc(StructuredModel):
            value: Optional[str] = ComparableField(
                comparator=LevenshteinComparator(threshold=0.95), default=None
            )

        config = Doc.to_stickler_config()
        assert config["fields"]["value"]["threshold"] == 0.95


class TestTheVerdictActuallyMoves:
    """Not just the reported number: the classification has to change.

    Asserting the stored threshold alone would pass even if nothing downstream
    read it, which is exactly the bug being fixed.
    """

    def test_a_score_between_the_two_thresholds_flips_the_verdict(self):
        """`0.9` clears the old `0.5` and misses an adopted `0.95`."""

        def field_is_a_true_positive(field) -> bool:
            model = type(
                "Probe",
                (StructuredModel,),
                {"__annotations__": {"value": Optional[str]}, "value": field},
            )
            matrix = model(value="abcdefghij").compare_with(
                model(value="abcdefghiX"), include_confusion_matrix=True
            )["confusion_matrix"]
            return matrix["overall"]["tp"] == 1

        # One character of ten differs, so similarity is 0.9.
        assert field_is_a_true_positive(
            ComparableField(comparator=LevenshteinComparator(), default=None)
        )
        assert not field_is_a_true_positive(
            ComparableField(
                comparator=LevenshteinComparator(threshold=0.95), default=None
            )
        )


class TestAListOfModelsStillAcceptsAComparator:
    """The guard on `List[StructuredModel]` must blame only what was written.

    `__init_subclass__` refuses a `threshold` on a list-of-model field, because
    Hungarian matching reads each element class's `match_threshold` instead. It
    used to detect that threshold by comparing the resolved value against the
    legacy `0.5`, which was a serviceable proxy only while nothing else could
    fill the slot. Now a comparator threshold does, so the proxy refused a class
    whose call site has no `threshold` in it at all.
    """

    def test_a_comparator_threshold_does_not_refuse_the_class(self):
        class Item(StructuredModel):
            name: str = ComparableField(comparator=LevenshteinComparator())

        class Doc(StructuredModel):
            items: list[Item] = ComparableField(
                comparator=LevenshteinComparator(threshold=0.9)
            )

        assert Doc.model_fields["items"] is not None

    def test_a_field_threshold_is_still_refused(self):
        """The guard's actual purpose has to survive the fix."""

        class Item(StructuredModel):
            name: str = ComparableField(comparator=LevenshteinComparator())

        with pytest.raises(ValueError, match="cannot have a 'threshold' parameter"):

            class Doc(StructuredModel):
                items: list[Item] = ComparableField(threshold=0.9)


class TestASubclassThatForwardsKwargs:
    """Signature inspection could not see a threshold passed through `**kwargs`.

    Custom comparators are a documented extension point, and forwarding
    `**kwargs` to `super().__init__` is an ordinary way to write one. Reading
    `threshold_was_set` off the instance sees it; inspecting the concrete
    class's declared parameters did not, so the bug stayed live out of tree.
    """

    def test_a_forwarded_threshold_is_adopted(self):
        class KwLev(LevenshteinComparator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        assert KwLev(threshold=0.9).threshold == 0.9
        assert (
            _threshold_of(
                ComparableField(comparator=KwLev(threshold=0.9), default=None)
            )
            == 0.9
        )

    def test_a_forwarding_subclass_with_no_threshold_still_defaults(self):
        class KwLev(LevenshteinComparator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        assert (
            _threshold_of(ComparableField(comparator=KwLev(), default=None))
            == _LEGACY_DEFAULT_THRESHOLD
        )


class TestTheComparatorDefaultsThemselves:
    """`DEFAULT_THRESHOLD` must agree with what each comparator resolves to.

    The defaults moved from signature defaults to a class attribute. A mismatch
    would silently change a comparator's own behaviour, which is not what this
    change is for.
    """

    @pytest.mark.parametrize(
        "factory,expected",
        [
            (LevenshteinComparator, 0.7),
            (ExactComparator, 1.0),
            (NumericComparator, 1.0),
            (DateComparator, 1.0),
        ],
    )
    def test_a_bare_construction_still_resolves_the_documented_default(
        self, factory, expected
    ):
        assert factory().threshold == expected
        assert factory.DEFAULT_THRESHOLD == expected
        assert factory().threshold_was_set is False


class TestExportDoesNotEmitAThresholdForAListOfModels:
    """A schema stickler writes must be one stickler can read back.

    `to_json_schema()` used to emit `x-aws-stickler-threshold` for a
    `List[StructuredModel]` field, carrying a number that is never read there:
    Hungarian matching uses the element class's `match_threshold`, which the
    exported `items` schema already carries. Harmless while import treated the
    value as a placeholder, and fatal once import reads it as a threshold the
    caller named -- which a list-of-model field is not allowed to have. The
    model could be exported and then not imported.
    """

    @staticmethod
    def _cart():
        class Product(StructuredModel):
            match_threshold = 0.8
            name: str = ComparableField(comparator=LevenshteinComparator())

        class Cart(StructuredModel):
            products: list[Product] = ComparableField(weight=2.0)

        return Cart

    def test_the_key_is_absent_from_the_array_property(self):
        schema = self._cart().to_json_schema()
        assert "x-aws-stickler-threshold" not in schema["properties"]["products"]

    def test_the_element_match_threshold_still_travels(self):
        """What was dropped must not be information anyone needed."""
        products = self._cart().to_json_schema()["properties"]["products"]
        assert products["items"]["x-aws-stickler-match-threshold"] == 0.8

    def test_the_weight_still_travels(self):
        """Only the threshold is dropped, not the whole extension block."""
        products = self._cart().to_json_schema()["properties"]["products"]
        assert products["x-aws-stickler-weight"] == 2.0

    def test_the_schema_imports_back(self):
        schema = self._cart().to_json_schema()
        rebuilt = StructuredModel.from_json_schema(schema)
        assert "products" in rebuilt.model_fields


class TestAPreviouslyExportedSchemaStillImports:
    """A schema a released version wrote must still be readable.

    `to_json_schema()` on every released version emitted
    `x-aws-stickler-threshold` on an array-of-model property, carrying the
    placeholder the field resolved to. Import forwarded it, and the old
    `threshold != 0.5` proxy let the placeholder slide. With the proxy replaced by
    an explicitness marker, forwarding it makes `__init_subclass__` refuse the
    class -- so every schema artifact already on disk containing a list of models
    stopped importing.
    """

    LEGACY_EXPORT = {
        "type": "object",
        "x-aws-stickler-model-name": "Cart",
        "properties": {
            "products": {
                "type": "array",
                "x-aws-stickler-threshold": 0.5,
                "x-aws-stickler-weight": 2.0,
                "items": {
                    "type": "object",
                    "x-aws-stickler-model-name": "Product",
                    "x-aws-stickler-match-threshold": 0.8,
                    "properties": {"name": {"type": "string"}},
                },
            }
        },
        "required": [],
    }

    def test_it_imports(self):
        model = StructuredModel.from_json_schema(self.LEGACY_EXPORT)
        assert "products" in model.model_fields

    def test_the_legacy_placeholder_is_ignored_silently(self):
        """Warning here would fire on every artifact this class exists to rescue.

        `0.5` is the only value a released `to_json_schema()` could write on this
        shape, since `__init_subclass__` refuses a named threshold on it. So the
        placeholder carries no authorial intent, and warning about it told the
        author to change a key the library itself wrote. This asserted the opposite
        first time -- "silently ignoring it would be the drop this work exists to
        remove" -- which confused a value a human chose with one we emitted.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StructuredModel.from_json_schema(self.LEGACY_EXPORT)
        assert [str(w.message) for w in caught] == []

    def test_a_value_a_human_chose_is_still_reported(self):
        """Suppressing the placeholder must not suppress the real case."""
        authored = deepcopy(self.LEGACY_EXPORT)
        authored["properties"]["products"]["x-aws-stickler-threshold"] = 0.88
        with pytest.warns(UserWarning, match="has no effect on array property"):
            StructuredModel.from_json_schema(authored)

    def test_that_warning_does_not_name_the_discarded_value_as_the_fix(self):
        """The element gate is a different number; echoing 0.88 would overwrite it."""
        authored = deepcopy(self.LEGACY_EXPORT)
        authored["properties"]["products"]["x-aws-stickler-threshold"] = 0.88
        with pytest.warns(UserWarning) as caught:
            StructuredModel.from_json_schema(authored)
        message = str(caught[0].message)
        assert "x-aws-stickler-match-threshold" in message
        assert "'x-aws-stickler-match-threshold': 0.88" not in message

    def test_the_element_match_threshold_still_governs(self):
        """What was dropped must not be information anyone needed."""
        model = StructuredModel.from_json_schema(self.LEGACY_EXPORT)
        element = model.model_fields["products"].annotation
        while getattr(element, "__args__", None):
            element = element.__args__[0]
        assert element.match_threshold == 0.8

    def test_a_non_placeholder_value_is_also_ignored_not_refused(self):
        """0.8 here is a misconfiguration, but not one worth refusing an import for."""
        schema = deepcopy(self.LEGACY_EXPORT)
        schema["properties"]["products"]["x-aws-stickler-threshold"] = 0.8
        with pytest.warns(UserWarning, match="0.8"):
            model = StructuredModel.from_json_schema(schema)
        assert "products" in model.model_fields


class TestAnOutOfTreeComparatorKeepsItsOldBehaviour:
    """`threshold is not None` is exact only for a subclass that defaults to None.

    A comparator written to the pattern the docs taught until now forwards a number
    on a bare construction, so the marker would read as set and the field would
    adopt it -- silently zeroing every imperfect score under
    `clip_under_threshold`. That is the outcome this change prevents, and it would
    have landed on exactly the population that cannot have migrated yet.
    """

    class Legacy(BaseComparator):
        """The documented pre-0.8 shape: a concrete default, forwarded."""

        def __init__(self, threshold: float = 1.0):
            super().__init__(threshold=threshold)

        def _compare(self, str1, str2):
            return 1.0 if str1 == str2 else 0.0

    class Migrated(BaseComparator):
        """The shape the docs now teach."""

        DEFAULT_THRESHOLD = 1.0

        def __init__(self, threshold: Optional[float] = None):
            super().__init__(threshold=threshold)

        def _compare(self, str1, str2):
            return 1.0 if str1 == str2 else 0.0

    def test_a_bare_legacy_comparator_does_not_impose_its_default(self):
        with pytest.warns(UserWarning, match="rather than None"):
            comparator = self.Legacy()
        assert (
            _threshold_of(ComparableField(comparator=comparator, default=None))
            == _LEGACY_DEFAULT_THRESHOLD
        )

    def test_a_legacy_comparator_is_still_honoured_away_from_its_default(self):
        with pytest.warns(UserWarning):
            comparator = self.Legacy(threshold=0.9)
        assert (
            _threshold_of(ComparableField(comparator=comparator, default=None)) == 0.9
        )

    def test_the_warning_names_the_migration(self):
        with pytest.warns(UserWarning, match="DEFAULT_THRESHOLD"):
            self.Legacy()

    def test_a_migrated_comparator_gets_exact_explicitness(self):
        """Including at its own default, which the legacy fallback cannot manage."""
        assert (
            _threshold_of(
                ComparableField(comparator=self.Migrated(threshold=1.0), default=None)
            )
            == 1.0
        )
        assert (
            _threshold_of(ComparableField(comparator=self.Migrated(), default=None))
            == _LEGACY_DEFAULT_THRESHOLD
        )

    def test_a_migrated_comparator_does_not_warn(self):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.Migrated()
        assert not [w for w in caught if "rather than None" in str(w.message)]


class TestAListOfModelsRefusesAFieldThresholdAtEveryValue:
    """The guard used to be legal at exactly one value, which is not a rule.

    It detected a field threshold by comparing against the literal `0.5`, so
    `ComparableField(threshold=0.5)` was accepted on a list-of-model field while
    `threshold=0.9` raised. Breaking, and in the breaking list.
    """

    @pytest.mark.parametrize("value", (0.0, 0.5, 0.9, 1.0))
    def test_every_value_is_refused(self, value):
        class Line(StructuredModel):
            sku: Optional[str] = ComparableField(default=None)

        with pytest.raises(ValueError, match="cannot have a 'threshold' parameter"):

            class Doc(StructuredModel):
                items: Optional[List[Line]] = ComparableField(
                    threshold=value, default=None
                )

    def test_the_advice_still_names_match_threshold(self):
        class Line(StructuredModel):
            sku: Optional[str] = ComparableField(default=None)

        with pytest.raises(ValueError, match="match_threshold"):

            class Doc(StructuredModel):
                items: Optional[List[Line]] = ComparableField(
                    threshold=0.5, default=None
                )


class _SwallowLine(StructuredModel):
    """A list element whose own gate is deliberately not 0.9."""

    match_threshold = 0.8

    sku: Optional[str] = ComparableField(default=None)


class TestAListOfModelsReportsASwallowedComparatorThreshold:
    """The same number, two spellings, and only one of them used to be answered.

    Adopting a comparator threshold makes it reachable on a shape that cannot use
    it: it resolves, is never read (Hungarian matching pairs items with the element
    class's `match_threshold`), and said nothing -- while the identical value
    written as `threshold=` raises with remediation. One spelling refused loudly
    and the other swallowed is the asymmetry this work exists to remove.

    Warned rather than raised because a comparator instance can be bound to several
    fields, so refusing the class would reject a construction that is legitimate
    wherever else it appears. A field-level `threshold=` cannot be shared that way,
    which is why that one stays an error.
    """

    @staticmethod
    def _build(comparator):
        return type(
            "Doc",
            (StructuredModel,),
            {
                "__annotations__": {"rows": List[_SwallowLine]},
                "rows": ComparableField(comparator=comparator, default=None),
            },
        )

    def test_it_warns(self):
        with pytest.warns(UserWarning, match="threshold set on its comparator"):
            self._build(LevenshteinComparator(threshold=0.9))

    def test_the_warning_names_the_knob_that_works(self):
        with pytest.warns(UserWarning) as caught:
            self._build(LevenshteinComparator(threshold=0.91))
        assert "match_threshold" in str(caught[0].message)

    def test_a_bare_comparator_stays_quiet(self):
        """The note keys on explicitness, not on the resolved value.

        A comparator's own default is never adopted, so nothing was swallowed here
        and there is nothing to report.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._build(LevenshteinComparator())
        assert [str(w.message) for w in caught] == []

    def test_a_scalar_field_stays_quiet_and_keeps_the_value(self):
        """The threshold IS consulted there, so reporting it would be wrong."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = type(
                "Doc",
                (StructuredModel,),
                {
                    "__annotations__": {"name": Optional[str]},
                    "name": ComparableField(
                        comparator=LevenshteinComparator(threshold=0.9), default=None
                    ),
                },
            )
        assert [str(w.message) for w in caught] == []
        assert model._get_comparison_info("name").threshold == 0.9

    def test_the_element_gate_still_decides_pairing(self):
        """What is ignored must not be information anyone needed."""
        with pytest.warns(UserWarning):
            model = self._build(LevenshteinComparator(threshold=0.9))
        element = model.model_fields["rows"].annotation
        while getattr(element, "__args__", None):
            element = element.__args__[0]
        assert element.match_threshold == 0.8

    def test_the_field_level_spelling_still_raises(self):
        """The two spellings must not swap places: one warns, one is an error."""
        with pytest.raises(ValueError, match="cannot have a 'threshold' parameter"):
            type(
                "Doc",
                (StructuredModel,),
                {
                    "__annotations__": {"rows": List[_SwallowLine]},
                    "rows": ComparableField(threshold=0.9, default=None),
                },
            )


class TestTheWarningKeyDoesNotCollideAcrossDynamicModels:
    """Two anonymous models sharing a field name must both be told.

    The swallowed-comparator-threshold warning keyed on `cls.__qualname__`. Every
    dynamically built model is named `DynamicModel`, so two unrelated ones sharing a
    field name -- `lines`, `amount`, `date`, the names that recur across document
    schemas -- collided in `warn_once`'s process-global memo and only the first ever
    warned. The sibling zero-threshold check thirty lines below already used
    `_model_identity` for exactly this reason.
    """

    @staticmethod
    def _dynamic(extra_field: str):
        annotations = {"rows": List[_SwallowLine], extra_field: Optional[str]}
        body = {
            "rows": ComparableField(
                comparator=LevenshteinComparator(threshold=0.9), default=None
            ),
            extra_field: ComparableField(default=None),
        }
        return type(
            "DynamicModel",
            (StructuredModel,),
            {"__annotations__": annotations, **body},
        )

    def test_both_models_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._dynamic("alpha")
            self._dynamic("beta")
        assert len(caught) == 2

    def test_the_same_model_still_warns_only_once(self):
        """Deduplication must survive the key change, or every field spams."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = self._dynamic("gamma")
            model.to_json_schema()
            model.to_json_schema()
        assert len(caught) == 1


class TestAZeroThresholdNamesWhereItWasWritten:
    """`0.0` adopted from a comparator must not be reported as a field argument.

    The zero-threshold warning said "M.f sets threshold=0.0", naming a
    `ComparableField` parameter absent from the call site -- the same
    misattribution this work fixes for the `List[StructuredModel]` error.
    """

    def test_a_comparator_zero_is_attributed_to_the_comparator(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            type(
                "Doc",
                (StructuredModel,),
                {
                    "__annotations__": {"f": Optional[str]},
                    "f": ComparableField(
                        comparator=LevenshteinComparator(threshold=0.0), default=None
                    ),
                },
            )
        assert len(caught) == 1
        assert "sets comparator threshold=0.0" in str(caught[0].message)

    def test_a_field_zero_is_still_attributed_to_the_field(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            type(
                "Doc",
                (StructuredModel,),
                {
                    "__annotations__": {"g": Optional[str]},
                    "g": ComparableField(threshold=0.0, default=None),
                },
            )
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "sets threshold=0.0" in message
        assert "comparator threshold" not in message


class TestAContainerKeepsItsPartialCreditInBothShapes:
    """`Dict[...]` and `List[Dict[...]]` must answer a named threshold the same way.

    Honouring a comparator's threshold exposed two gaps that together zeroed a
    real number. Same annotation shape, same comparator, same content:

        d:  Dict[str, str]       = ComparableField(comparator=ANLSStarComparator(threshold=0.9))
        ld: List[Dict[str, str]] = ComparableField(comparator=ANLSStarComparator(threshold=0.9))

                                    d         ld
        dev                         0.5625    0.5625
        before this fix             0.5625    0.0

    1. `_install_mapping_comparators` gated the clip-default amendment on
       `is_mapping_annotation` alone, so a `List[Dict[...]]` field never got
       `clip_under_threshold=False` and kept clipping on. `get_comparison_info`
       does cover the list shape, but only when the comparator is NOT explicit,
       which is why naming one is what broke it.

    2. Turning the flag off was not enough, because `clip_under_threshold` was a
       no-op on every list path. `PrimitiveListComparator` says "for lists we
       NEVER clip" and sets `threshold_applied_score = raw_similarity`, but the
       zeroing had already happened upstream in
       `ComparisonHelper.unordered_list_metrics`, which clipped each sub-threshold
       pair before averaging regardless of the field's setting. So the line
       claiming lists preserve partial credit was preserving a score that had
       already been thrown away.

    The flag now means one thing everywhere, applied per ELEMENT on a list.
    Classification is untouched: a sub-threshold pair is still one `fd`. Only the
    score moves, and only for a field that asked to keep partial credit.
    """

    GT = {"a": "Acme Corporation"}
    PRED = {"a": "Acme Corp"}

    def _model(self):
        from stickler.comparators.anls import ANLSStarComparator

        class M(StructuredModel):
            d: Optional[Dict[str, str]] = ComparableField(
                comparator=ANLSStarComparator(threshold=0.9), default=None
            )
            ld: Optional[List[Dict[str, str]]] = ComparableField(
                comparator=ANLSStarComparator(threshold=0.9), default=None
            )

        return M

    def test_the_two_shapes_score_identically(self):
        """The reviewer's blocker, stated as the invariant it violates."""
        M = self._model()
        scores = M(d=dict(self.GT), ld=[dict(self.GT)]).compare_with(
            M(d=dict(self.PRED), ld=[dict(self.PRED)])
        )["field_scores"]
        assert scores["d"] == pytest.approx(0.5625)
        assert scores["ld"] == pytest.approx(scores["d"])

    def test_both_shapes_resolve_the_named_threshold(self):
        """Neither shape may quietly keep 0.5; that is what this PR is for."""
        M = self._model()
        for field in ("d", "ld"):
            assert M._get_comparison_info(field).threshold == pytest.approx(0.9), field

    def test_both_shapes_turn_clipping_off_as_containers(self):
        """The container policy, which is why partial credit survives at all."""
        M = self._model()
        for field in ("d", "ld"):
            assert M._get_comparison_info(field).clip_under_threshold is False, field

    def test_the_element_is_still_classified_as_a_false_discovery(self):
        """Keeping the score must NOT launder the verdict.

        0.5625 is below the declared 0.9, so the element missed its bar and the
        confusion matrix has to say so. If this ever reads `tp=1`, the fix has
        turned a scoring change into a classification change.
        """
        M = self._model()
        cm = M(ld=[dict(self.GT)]).compare_with(
            M(ld=[dict(self.PRED)]), include_confusion_matrix=True
        )["confusion_matrix"]
        node = cm["fields"]["ld"]
        assert node["similarity_score"] == pytest.approx(0.5625)
        assert (node["overall"]["tp"], node["overall"]["fd"]) == (0, 1)

    def test_a_default_clip_list_still_zeroes_a_sub_threshold_element(self):
        """The blast radius, bounded: an ordinary list is unchanged.

        `clip_under_threshold` defaults to True, so a `List[str]` that never asked
        to keep partial credit still contributes 0.0 for a missed element, exactly
        as on `dev`. Without this, the fix would silently raise scores on every
        list field in every existing model.
        """

        class Plain(StructuredModel):
            tags: Optional[List[str]] = ComparableField(
                comparator=LevenshteinComparator(threshold=0.9), default=None
            )

        assert Plain._get_comparison_info("tags").clip_under_threshold is True
        score = Plain(tags=["Acme Corporation"]).compare_with(
            Plain(tags=["Acme Corp"])
        )["field_scores"]["tags"]
        assert score == pytest.approx(0.0)

    def test_an_explicit_clip_choice_is_honoured_on_a_list(self):
        """Both directions, since the flag was previously inert on lists."""

        class KeepIt(StructuredModel):
            tags: Optional[List[str]] = ComparableField(
                comparator=LevenshteinComparator(threshold=0.9),
                clip_under_threshold=False,
                default=None,
            )

        class ZeroIt(StructuredModel):
            tags: Optional[List[str]] = ComparableField(
                comparator=LevenshteinComparator(threshold=0.9),
                clip_under_threshold=True,
                default=None,
            )

        def score(model):
            return model(tags=["Acme Corporation"]).compare_with(
                model(tags=["Acme Corp"])
            )["field_scores"]["tags"]

        assert score(KeepIt) == pytest.approx(0.5625)
        assert score(ZeroIt) == pytest.approx(0.0)
