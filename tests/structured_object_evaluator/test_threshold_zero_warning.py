"""A threshold of exactly 0.0 disables classification, so it warns.

The threshold test is ``>=``, so ``0.0`` is satisfied by every score including
``0.0`` itself: every assigned pair becomes a true positive and a wholly
incorrect prediction is reported as a true positive. Nothing errors and
the numbers look ideal, which makes it the hardest misconfiguration to notice.

``0.0`` is a cliff rather than a slope -- ``0.01`` already classifies
correctly -- so the check is for exactly ``0.0`` and low-but-positive
thresholds are left alone.

See https://github.com/awslabs/stickler/issues/234
"""

import gc
import warnings
from typing import List

import pytest

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.threshold_helper import (
    THRESHOLD_DOCS_URL,
)
from stickler.utils import deprecation


@pytest.fixture(autouse=True)
def _reset_warn_once():
    """`warn_once` is process-scoped; clear it so each test sees its warning."""
    deprecation._warned.clear()
    yield
    deprecation._warned.clear()


def _user_warnings(recorded):
    return [w for w in recorded if w.category is UserWarning]


class TestFieldThreshold:
    def test_zero_threshold_warns(self):
        with pytest.warns(UserWarning, match="threshold=0.0"):

            class Doc(StructuredModel):
                tags: List[str] = ComparableField(
                    comparator=ExactComparator(), threshold=0.0
                )

    def test_warning_names_the_field_and_links_the_docs(self):
        with pytest.warns(UserWarning) as record:

            class Doc(StructuredModel):
                vendor: str = ComparableField(
                    comparator=ExactComparator(), threshold=0.0
                )

        message = str(record[0].message)
        assert "Doc.vendor" in message, "the warning must say which field"
        assert "0.01" in message, "the warning must suggest a usable alternative"
        # Assert the literal target, not `THRESHOLD_DOCS_URL in message` --
        # that compares the constant to itself and passes for any value,
        # including a URL that explains nothing. The old target was a docs page
        # whose only mention of zero was "raw similarity score (0.0 -- 1.0)".
        assert "github.com/awslabs/stickler/issues/234" in message, (
            "the warning must link somewhere that explains the zero-threshold cliff"
        )
        assert THRESHOLD_DOCS_URL in message, "the constant must be what is emitted"

    def test_message_claims_no_metric_outcome(self):
        """The message names the invariant, not precision or recall.

        Both metrics are false in reachable cases, symmetrically: unmatched
        predictions are still FAs and unmatched ground truth is still FNs, so
        neither is guaranteed at ``0.0``. What *is* invariant is that no false
        discovery can be reported. Claiming a metric would make the warning
        false for unequal-length lists, and a user seeing imperfect precision
        would conclude it did not apply to them.
        """
        with pytest.warns(UserWarning) as record:

            class Doc(StructuredModel):
                vendor: str = ComparableField(
                    comparator=ExactComparator(), threshold=0.0
                )

        message = str(record[0].message)
        assert "false discovery" in message, "the invariant must be named"
        assert "perfect precision" not in message
        assert "perfect recall" not in message

    @pytest.mark.parametrize("threshold", [0.01, 0.1, 0.5, 0.7, 1.0])
    def test_positive_thresholds_do_not_warn(self, threshold):
        """0.01 already classifies correctly, so only exactly 0.0 is a problem."""
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")

            class Doc(StructuredModel):
                tags: List[str] = ComparableField(
                    comparator=ExactComparator(), threshold=threshold
                )

        assert _user_warnings(recorded) == []

    def test_default_threshold_does_not_warn(self):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")

            class Doc(StructuredModel):
                tags: List[str] = ComparableField(comparator=ExactComparator())

        assert _user_warnings(recorded) == []


class TestModelMatchThreshold:
    def test_zero_match_threshold_warns(self):
        with pytest.warns(UserWarning, match="match_threshold=0.0"):

            class Line(StructuredModel):
                match_threshold = 0.0

                sku: str = ComparableField(comparator=ExactComparator())

    def test_match_threshold_message_is_unconditional(self):
        """The consequence is not conditional on being a list element.

        `match_threshold` also serves as the default field threshold for any
        field with no explicit config, so a standalone model is affected too.
        An earlier version of this message hedged with "if this model is
        compared as a list element", which told exactly the affected users that
        it did not apply to them (#237).
        """
        with pytest.warns(UserWarning) as record:

            class Line(StructuredModel):
                match_threshold = 0.0

                sku: str = ComparableField(comparator=ExactComparator())

        message = str(record[0].message)
        assert "if this model is compared as a list element" not in message
        assert "inherits this value" in message, (
            "the message must name the field-threshold route, not only lists"
        )

    def test_the_value_is_not_inert_on_a_standalone_model(self):
        """The behaviour the message now describes, measured.

        A plainly annotated field (no ComparableField) inherits
        `match_threshold` as its own threshold, so a wholly wrong prediction
        scores as a true positive with no list involved. Probing with
        `ComparableField()` hides this -- that path takes an earlier branch and
        gets a hardcoded 0.5.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            class Zero(StructuredModel):
                match_threshold = 0.0

                name: str

            class Normal(StructuredModel):
                match_threshold = 0.7

                name: str

            def overall(model_cls):
                return model_cls(name="Acme Corporation").compare_with(
                    model_cls(name="zzzzzzzz"), include_confusion_matrix=True
                )["confusion_matrix"]["overall"]

            zero, normal = overall(Zero), overall(Normal)

        # At 0.0 a wholly wrong value is a true positive with perfect metrics.
        assert zero["tp"] == 1 and zero["fd"] == 0
        assert zero["derived"]["cm_precision"] == 1.0
        assert zero["derived"]["cm_recall"] == 1.0

        # At a normal threshold the same comparison is a false discovery.
        assert normal["tp"] == 0 and normal["fd"] == 1

    def test_a_comparable_field_does_not_inherit_match_threshold(self):
        """Why the inertness claim looked true: this path ignores the value.

        Documents the asymmetry that made the original message wrong, so a
        future reader probing with ComparableField is not misled the same way.
        """
        from stickler.structured_object_evaluator.models.configuration_helper import (
            ConfigurationHelper,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            class Explicit(StructuredModel):
                match_threshold = 0.0

                n: str = ComparableField(comparator=ExactComparator())

            class Plain(StructuredModel):
                match_threshold = 0.0

                n: str

        assert ConfigurationHelper.get_comparison_info(Explicit, "n").threshold == 0.5
        assert ConfigurationHelper.get_comparison_info(Plain, "n").threshold == 0.0

    def test_warning_names_the_model(self):
        with pytest.warns(UserWarning) as record:

            class Line(StructuredModel):
                match_threshold = 0.0

                sku: str = ComparableField(comparator=ExactComparator())

        assert "Line" in str(record[0].message)

    @pytest.mark.parametrize("threshold", [0.01, 0.7, 1.0])
    def test_positive_match_thresholds_do_not_warn(self, threshold):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")

            class Line(StructuredModel):
                match_threshold = threshold

                sku: str = ComparableField(comparator=ExactComparator())

        assert _user_warnings(recorded) == []

    @pytest.mark.parametrize("wrong_type", [False, True])
    def test_bool_is_not_reported_as_a_zero_threshold(self, wrong_type):
        """`bool` is an `int` and `False == 0.0`, so an unguarded check lies.

        Reporting "sets match_threshold=0.0" for `match_threshold = False`
        names a value the user never wrote. A wrong type is a different
        problem from a zero threshold.
        """
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")

            class Line(StructuredModel):
                match_threshold = wrong_type

                sku: str = ComparableField(comparator=ExactComparator())

        assert _user_warnings(recorded) == []

    def test_inherited_match_threshold_does_not_re_warn(self):
        """Only the class that sets it warns, not every subclass of it.

        A subclass inherits the attribute; re-warning would fire for code that
        never chose the value.
        """
        with pytest.warns(UserWarning):

            class Base(StructuredModel):
                match_threshold = 0.0

                sku: str = ComparableField(comparator=ExactComparator())

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")

            class Child(Base):
                pass

        assert _user_warnings(recorded) == []


class TestJsonConfigPath:
    """A config-driven model must warn too, not only a hand-written class.

    Both validation sites accept ``0.0`` (they check ``0.0 <= value <= 1.0``),
    so a config is a realistic way to arrive at this state.
    """

    def test_field_threshold_from_config_warns(self):
        config = {
            "model_name": "FromConfig",
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "ExactComparator",
                    "threshold": 0.0,
                }
            },
        }

        with pytest.warns(UserWarning, match="threshold=0.0"):
            StructuredModel.model_from_json(config)

    def test_match_threshold_from_config_warns(self):
        """`match_threshold` is assigned after class creation by the factory.

        `__init_subclass__` has already run by then, so without an explicit
        check in `ModelFactory` this case would warn nowhere.
        """
        config = {
            "model_name": "FromConfigMT",
            "match_threshold": 0.0,
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "ExactComparator",
                    "threshold": 0.5,
                }
            },
        }

        with pytest.warns(UserWarning, match="match_threshold=0.0"):
            StructuredModel.model_from_json(config)

    def test_distinct_anonymous_configs_each_warn(self):
        """`model_name` defaults to "DynamicModel", so name keying collides.

        A process building models from several configs -- a service per
        request, a batch job over a directory -- would report the first
        misconfiguration and swallow every later one, which is the gap this
        call site exists to close.
        """
        configs = [
            {
                "match_threshold": 0.0,
                "fields": {
                    name: {
                        "type": "str",
                        "comparator": "ExactComparator",
                        "threshold": 0.5,
                    }
                },
            }
            for name in ("alpha", "beta")
        ]

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            for config in configs:
                StructuredModel.model_from_json(config)

        assert len(_user_warnings(recorded)) == 2

    def test_same_config_loaded_repeatedly_warns_once(self):
        """warn-once must survive the dedup key, not be defeated by it.

        A new class object is created per call, so keying on ``id()`` made a
        loop over one config emit a warning every time -- 192 for 200 loads --
        and grew the process-global ``_warned`` set without bound. That is the
        stderr flood ``warn_once`` exists to prevent.
        """
        config = {
            "model_name": "Repeated",
            "match_threshold": 0.0,
            "fields": {
                "a": {
                    "type": "str",
                    "comparator": "ExactComparator",
                    "threshold": 0.5,
                }
            },
        }

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            for _ in range(50):
                StructuredModel.model_from_json(dict(config))

        assert len(_user_warnings(recorded)) == 1
        assert len(deprecation._warned) == 1, "the dedup set must not grow per load"

    def test_configs_still_warn_when_each_class_is_collected(self):
        """The key must survive garbage collection, not just co-existence.

        An earlier attempt keyed on ``id(DynamicClass)``. CPython recycles an
        address once the class is freed, so a batch loop that builds, uses and
        drops each model collided on the same key and dropped most of its
        warnings (measured: 19 of 50). The previous version of this test held
        every class alive at once, which is exactly the shape that hides the
        bug.
        """
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            for i in range(25):
                model = StructuredModel.model_from_json(
                    {
                        "match_threshold": 0.0,
                        "fields": {
                            f"f{i}": {
                                "type": "str",
                                "comparator": "ExactComparator",
                                "threshold": 0.5,
                            }
                        },
                    }
                )
                del model
                gc.collect()

        assert len(_user_warnings(recorded)) == 25, (
            "a recyclable key silently drops warnings for collected classes"
        )

    def test_anonymous_configs_sharing_a_field_name_each_warn(self):
        """Field names like `amount`, `date`, `id` recur across schemas.

        The field path keyed on `f"{cls.__name__}.{field_name}"`, which is
        `DynamicModel.amount` for every anonymous config, so the second config
        with a zero-threshold `amount` was silent.
        """
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            for other in ("alpha", "beta"):
                StructuredModel.model_from_json(
                    {
                        "fields": {
                            "amount": {
                                "type": "str",
                                "comparator": "ExactComparator",
                                "threshold": 0.0,
                            },
                            other: {
                                "type": "str",
                                "comparator": "ExactComparator",
                                "threshold": 0.5,
                            },
                        }
                    }
                )

        assert len(_user_warnings(recorded)) == 2

    def test_warning_text_carries_no_internal_identity(self):
        """No object address leaks into user-visible text.

        An earlier attempt keyed dedup on ``id(DynamicClass)`` and interpolated
        it, printing "DynamicModel#4355291632 sets ...".
        """
        config = {
            "match_threshold": 0.0,
            "fields": {
                "a": {
                    "type": "str",
                    "comparator": "ExactComparator",
                    "threshold": 0.5,
                }
            },
        }

        with pytest.warns(UserWarning) as record:
            StructuredModel.model_from_json(config)

        message = str(record[0].message)
        assert "#" not in message
        assert not any(ch.isdigit() and len(part) > 8 for part in message.split() for ch in part[:1]), (
            "no long numeric token that could be an address"
        )

    def test_anonymous_configs_are_distinguishable_in_the_message(self):
        """Identical text would be swallowed by Python's warning registry.

        ``__warningregistry__`` is keyed on the message, so two anonymous
        configs producing byte-identical text print only once under the default
        action -- the second misconfiguration is lost even though ``warn_once``
        approved it. The message names the fields to keep them distinct.
        """
        messages = []
        for field in ("alpha", "beta"):
            deprecation._warned.clear()
            with pytest.warns(UserWarning) as record:
                StructuredModel.model_from_json(
                    {
                        "match_threshold": 0.0,
                        "fields": {
                            field: {
                                "type": "str",
                                "comparator": "ExactComparator",
                                "threshold": 0.5,
                            }
                        },
                    }
                )
            messages.append(str(record[0].message))

        assert messages[0] != messages[1], (
            "identical text collapses in Python's warning registry"
        )
        assert "alpha" in messages[0] and "beta" in messages[1]

    def test_positive_config_thresholds_do_not_warn(self):
        config = {
            "model_name": "FineFromConfig",
            "match_threshold": 0.7,
            "fields": {
                "name": {
                    "type": "str",
                    "comparator": "ExactComparator",
                    "threshold": 0.5,
                }
            },
        }

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            StructuredModel.model_from_json(config)

        assert _user_warnings(recorded) == []


class TestCoverageOfEveryEntryPoint:
    """Every way a user can configure a threshold, and whether it warns."""

    def test_from_json_schema_field_threshold_warns(self):
        schema = {
            "type": "object",
            "properties": {"v": {"type": "string", "x-aws-stickler-threshold": 0.0}},
        }

        with pytest.warns(UserWarning, match="threshold=0.0"):
            StructuredModel.from_json_schema(schema)

    def test_from_json_schema_match_threshold_warns(self):
        schema = {
            "type": "object",
            "x-aws-stickler-match-threshold": 0.0,
            "properties": {"v": {"type": "string"}},
        }

        with pytest.warns(UserWarning, match="match_threshold=0.0"):
            StructuredModel.from_json_schema(schema)

    def test_create_model_from_fields_warns(self):
        """The second ModelFactory entry point, not just create_model_from_json."""
        from stickler.structured_object_evaluator.models.model_factory import (
            ModelFactory,
        )

        with pytest.warns(UserWarning, match="match_threshold=0.0"):
            ModelFactory.create_model_from_fields(
                "FromFields",
                {"v": (str, ComparableField(comparator=ExactComparator()))},
                match_threshold=0.0,
            )

    def test_subclass_setting_its_own_zero_warns(self):
        """Inheriting silently is right; choosing 0.0 yourself is not."""

        class Parent(StructuredModel):
            v: str = ComparableField(comparator=ExactComparator())

        with pytest.warns(UserWarning, match="match_threshold=0.0"):

            class Child(Parent):
                match_threshold = 0.0

    def test_post_hoc_setattr_is_a_known_gap(self):
        """Documents a limitation rather than asserting desired behaviour.

        The checks run at class creation and config conversion, so assigning
        the attribute afterwards is invisible to them. Catching it would need a
        metaclass ``__setattr__``. If this ever starts warning, that is an
        improvement -- update this test rather than treating it as a break.
        """

        class Line(StructuredModel):
            v: str = ComparableField(comparator=ExactComparator())

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            Line.match_threshold = 0.0

        assert _user_warnings(recorded) == [], (
            "if this now warns, the coverage gap is closed -- update the docstring"
        )


class TestInternalSentinelIsUnaffected:
    """Stickler uses ``match_threshold=0.0`` internally on purpose."""

    def test_normal_comparison_does_not_warn(self):
        """`ComparisonHelper.compare_unordered_lists` builds a 0.0 matcher.

        That is a deliberate capture-all: it reads only ``matched_pairs`` and
        reclassifies against its own threshold. It must not surface a warning
        to a user who configured nothing wrong.
        """

        class Doc(StructuredModel):
            tags: List[str] = ComparableField(
                comparator=ExactComparator(), threshold=0.5
            )

        gt, pred = Doc(tags=["a", "b"]), Doc(tags=["a", "c"])

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            gt.compare_with(pred, include_confusion_matrix=True)

        assert _user_warnings(recorded) == []


class TestWarnOncePerSite:
    def test_repeated_instantiation_warns_once(self):
        """A bulk run must not emit one warning per document."""
        with pytest.warns(UserWarning) as record:

            class Doc(StructuredModel):
                tags: List[str] = ComparableField(
                    comparator=ExactComparator(), threshold=0.0
                )

            for _ in range(50):
                Doc(tags=["x"]).compare_with(Doc(tags=["y"]))

        assert len(_user_warnings(record.list)) == 1


def test_no_false_discovery_is_possible_at_any_list_length():
    """The invariant the message names, swept across list lengths.

    FD means "compared and scored below threshold", and nothing scores below
    0.0, so FD must be 0 for every combination. Precision is deliberately not
    asserted: it drops below 1.0 whenever the prediction is longer, which is
    why the message does not claim it.
    """

    class Line(StructuredModel):
        match_threshold = 0.0

        sku: str = ComparableField(comparator=ExactComparator(), threshold=1.0)

    class Doc(StructuredModel):
        lines: List[Line] = ComparableField(weight=1.0)

    precision_was_imperfect = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n_gt in range(1, 5):
            for n_pred in range(1, 5):
                gt = Doc(lines=[Line(sku=f"A{i}") for i in range(n_gt)])
                pred = Doc(lines=[Line(sku=f"Z{i}") for i in range(n_pred)])
                overall = gt.compare_with(pred, include_confusion_matrix=True)[
                    "confusion_matrix"
                ]["fields"]["lines"]["overall"]

                assert overall["fd"] == 0, (
                    f"{n_gt} vs {n_pred}: nothing can score below a zero threshold"
                )
                if overall["derived"]["cm_precision"] != 1.0:
                    precision_was_imperfect = True

    assert precision_was_imperfect, (
        "if precision is now always perfect the message may claim it again"
    )


def test_match_threshold_zero_on_a_real_list_element():
    """The conditional in the message is true when the condition holds.

    Equal-length lists at ``match_threshold=0.0``: every object pairs, every
    pair clears the threshold, so wholly wrong objects are all true positives.
    """

    class Line(StructuredModel):
        match_threshold = 0.0

        sku: str = ComparableField(comparator=ExactComparator(), threshold=1.0)

    class Doc(StructuredModel):
        lines: List[Line] = ComparableField(weight=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Doc(lines=[Line(sku="A"), Line(sku="B")]).compare_with(
            Doc(lines=[Line(sku="Y"), Line(sku="Z")]), include_confusion_matrix=True
        )

    overall = result["confusion_matrix"]["fields"]["lines"]["overall"]
    assert overall["tp"] == 2, "every paired object clears a zero threshold"
    assert overall["fd"] == 0
    assert overall["derived"]["cm_precision"] == 1.0


def test_unmatched_items_keep_both_metrics_honest():
    """Why the message names no metric outcome.

    Unmatched items are not subject to any threshold, so an extra ground-truth
    object stays an FN and an extra prediction stays an FA. Recall and
    precision therefore both stay honest at ``match_threshold=0.0``, in
    opposite directions -- which is why the message claims neither.
    """

    class Line(StructuredModel):
        match_threshold = 0.0

        sku: str = ComparableField(comparator=ExactComparator(), threshold=1.0)

    class Doc(StructuredModel):
        lines: List[Line] = ComparableField(weight=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Doc(lines=[Line(sku="A"), Line(sku="B")]).compare_with(
            Doc(lines=[Line(sku="Z")]), include_confusion_matrix=True
        )

    overall = result["confusion_matrix"]["fields"]["lines"]["overall"]
    assert overall["fn"] == 1, "the unpaired ground-truth object is still a miss"
    assert overall["derived"]["cm_recall"] == 0.5

    # Mirror case: an extra *prediction* is an FA, so precision drops instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mirrored = Doc(lines=[Line(sku="A")]).compare_with(
            Doc(lines=[Line(sku="Y"), Line(sku="Z")]), include_confusion_matrix=True
        )

    mirror_overall = mirrored["confusion_matrix"]["fields"]["lines"]["overall"]
    assert mirror_overall["fa"] == 1, "the unpaired prediction is still a false alarm"
    assert mirror_overall["derived"]["cm_precision"] == 0.5

    # Neither metric is safe to claim; FD is zero in both directions.
    assert overall["fd"] == 0 and mirror_overall["fd"] == 0


def test_the_behavior_the_warning_describes():
    """Pin the misbehavior itself, so the warning cannot outlive its cause.

    A wholly wrong prediction on a ``threshold=0.0`` field is scored as a true
    positive with perfect recall. Uses a two-item list, which behaves this way
    on every list length; the one-item case only joins it once #224 lands (the
    single-item fast path currently drops the pair instead), so asserting on
    n=1 would pass or fail depending on merge order.

    If a later change makes ``threshold=0.0`` classify sensibly, this fails and
    the warning should be deleted rather than left to mislead.
    """

    class Doc(StructuredModel):
        tags: List[str] = ComparableField(comparator=ExactComparator(), threshold=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Doc(tags=["X", "Y"]).compare_with(
            Doc(tags=["A", "B"]), include_confusion_matrix=True
        )

    overall = result["confusion_matrix"]["aggregate"]
    assert overall["tp"] == 2, "every pair clears a zero threshold"
    assert overall["fd"] == 0
    assert overall["derived"]["cm_recall"] == 1.0, (
        "a wholly wrong prediction reports perfect recall -- this is the trap"
    )
    assert overall["derived"]["cm_precision"] == 1.0
