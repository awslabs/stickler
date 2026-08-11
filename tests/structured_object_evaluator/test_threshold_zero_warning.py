"""A threshold of exactly 0.0 disables classification, so it warns.

The threshold test is ``>=``, so ``0.0`` is satisfied by every score including
``0.0`` itself: every assigned pair becomes a true positive and a wholly
incorrect prediction reports perfect precision and recall. Nothing errors and
the numbers look ideal, which makes it the hardest misconfiguration to notice.

``0.0`` is a cliff rather than a slope -- ``0.01`` already classifies
correctly -- so the check is for exactly ``0.0`` and low-but-positive
thresholds are left alone.

See https://github.com/awslabs/stickler/issues/234
"""

import warnings
from typing import List

import pytest

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
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
        assert "https://" in message, "the warning must link to an explanation"

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
