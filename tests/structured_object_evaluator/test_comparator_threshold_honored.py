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

from typing import Optional

import pytest

from stickler.comparators.date import DateComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.models.comparable_field import (
    _LEGACY_DEFAULT_THRESHOLD,
    ComparableField,
    _comparator_threshold_was_set,
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

    def test_a_threshold_equal_to_the_class_default_reads_as_unset(self):
        """A known and documented consequence, asserted so it is not a surprise.

        Explicitness is recovered by comparing against the declared default,
        because each subclass resolves its own before calling `super().__init__`.
        So passing the default value explicitly is indistinguishable from not
        passing it. The docstring tells callers to state it on the field instead.
        """
        assert (
            _threshold_of(
                ComparableField(comparator=DateComparator(threshold=1.0), default=None)
            )
            == _LEGACY_DEFAULT_THRESHOLD
        )


class TestTheHelperInIsolation:
    def test_it_returns_none_for_a_default_construction(self):
        assert _comparator_threshold_was_set(LevenshteinComparator()) is None

    def test_it_returns_the_value_for_an_explicit_one(self):
        assert (
            _comparator_threshold_was_set(LevenshteinComparator(threshold=0.9)) == 0.9
        )

    def test_it_returns_none_for_a_comparator_with_no_threshold_parameter(self):
        """Must not assume every comparator takes a threshold."""

        class NoThreshold:
            pass

        assert _comparator_threshold_was_set(NoThreshold()) is None


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
