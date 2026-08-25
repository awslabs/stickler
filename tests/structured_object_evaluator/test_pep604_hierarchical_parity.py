"""The spelling of an optional must not flatten a hierarchical metric breakdown.

``ConfigurationHelper.is_structured_field_type`` is the #149 gate: when it
returns False for a nested-object field, ``StructuredListComparator`` routes
that field down the non-hierarchical path and, as the comment at
``configuration_helper.py:165-168`` already warned, it "loses its nested metric
breakdown".

It returned True for ``Optional[Inner]`` and False for ``Inner | None``, so a
PEP 604-spelled optional nested object silently lost the very breakdown #149
restored -- while the top-level counts still agreed, which is why it hid.

This is the more serious half of the PEP 604 gap: schema export produces
visibly wrong output, but this produces a report that looks complete and is
missing rows.
"""

from typing import List, Optional

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.configuration_helper import (
    ConfigurationHelper,
)
from stickler.structured_object_evaluator.models.structured_list_comparator import (
    StructuredListComparator,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class _Inner(StructuredModel):
    city: str = ComparableField(comparator=ExactComparator())


class TypingOuter(StructuredModel):
    opt_obj: Optional[_Inner] = ComparableField(default=None, comparator=ExactComparator())
    opt_list: Optional[List[_Inner]] = ComparableField(default=None)
    opt_str: Optional[str] = ComparableField(default=None, comparator=ExactComparator())


class Pep604Outer(StructuredModel):
    opt_obj: _Inner | None = ComparableField(default=None, comparator=ExactComparator())
    opt_list: list[_Inner] | None = ComparableField(default=None)
    opt_str: str | None = ComparableField(default=None, comparator=ExactComparator())


class TestIsStructuredFieldTypeSpellingParity:
    """Mirrors test_configuration_helper_optional_structured.py in PEP 604."""

    def test_optional_structured_model_is_structured_in_both_spellings(self):
        assert (
            ConfigurationHelper.is_structured_field_type(TypingOuter.model_fields["opt_obj"])
            is True
        )
        assert (
            ConfigurationHelper.is_structured_field_type(Pep604Outer.model_fields["opt_obj"])
            is True
        )

    def test_optional_list_of_models_is_structured_in_both_spellings(self):
        assert (
            ConfigurationHelper.is_structured_field_type(TypingOuter.model_fields["opt_list"])
            is True
        )
        assert (
            ConfigurationHelper.is_structured_field_type(Pep604Outer.model_fields["opt_list"])
            is True
        )

    def test_optional_primitive_is_not_structured_in_either_spelling(self):
        assert (
            ConfigurationHelper.is_structured_field_type(TypingOuter.model_fields["opt_str"])
            is False
        )
        assert (
            ConfigurationHelper.is_structured_field_type(Pep604Outer.model_fields["opt_str"])
            is False
        )


# ---------------------------------------------------------------------------
# The list-element models. Each item carries three matching sibling fields so
# the pair scores 0.75 -- above the 0.7 match threshold -- because
# StructuredListComparator only generates field_details for a good match
# (structured_list_comparator.py:264-270).
# ---------------------------------------------------------------------------


class TypingItem(StructuredModel):
    label: str = ComparableField(comparator=ExactComparator())
    p1: str = ComparableField(comparator=ExactComparator())
    p2: str = ComparableField(comparator=ExactComparator())
    addr: Optional[_Inner] = ComparableField(default=None, comparator=ExactComparator())


class Pep604Item(StructuredModel):
    label: str = ComparableField(comparator=ExactComparator())
    p1: str = ComparableField(comparator=ExactComparator())
    p2: str = ComparableField(comparator=ExactComparator())
    addr: _Inner | None = ComparableField(default=None, comparator=ExactComparator())


class TypingDoc(StructuredModel):
    items: List[TypingItem] = ComparableField()


class Pep604Doc(StructuredModel):
    items: List[Pep604Item] = ComparableField()


def _addr_detail(doc_cls, item_cls):
    """field_details['addr'] for a list whose item has an optional nested model."""
    gt = [item_cls(label="a", p1="x", p2="y", addr=_Inner(city="Seattle"))]
    pred = [item_cls(label="a", p1="x", p2="y", addr=_Inner(city="Portland"))]
    comparator = StructuredListComparator(doc_cls(items=gt))
    result = comparator.compare_struct_list_with_scores(gt, pred, "items")
    return result["fields"]["addr"]


class TestHierarchicalBreakdownSpellingParity:
    def test_both_spellings_produce_equal_field_detail(self):
        assert _addr_detail(TypingDoc, TypingItem) == _addr_detail(Pep604Doc, Pep604Item)

    def test_pep604_keeps_the_nested_field_breakdown(self):
        """``addr.fields.city`` was absent entirely for ``_Inner | None``."""
        detail = _addr_detail(Pep604Doc, Pep604Item)
        assert "fields" in detail
        assert "city" in detail["fields"]

    def test_pep604_keeps_the_derived_metrics(self):
        detail = _addr_detail(Pep604Doc, Pep604Item)
        assert "derived" in detail["overall"]
        for key in ("cm_precision", "cm_recall", "cm_f1", "cm_accuracy"):
            assert key in detail["overall"]["derived"]

    def test_pep604_keeps_the_similarity_scores(self):
        detail = _addr_detail(Pep604Doc, Pep604Item)
        for key in (
            "raw_similarity_score",
            "similarity_score",
            "threshold_applied_score",
        ):
            assert key in detail

    def test_the_top_level_counts_agreed_all_along(self):
        """Why this hid: the confusion matrix looked right either way."""
        typing_counts = _addr_detail(TypingDoc, TypingItem)["overall"]
        pep604_counts = _addr_detail(Pep604Doc, Pep604Item)["overall"]
        for key in ("tp", "fd", "fp"):
            assert typing_counts[key] == pep604_counts[key]


class TestListDispatchSpellingParity:
    """``_is_list_field`` read ``__origin__`` as an attribute, which a PEP 604
    union does not have -- so the branch was unreachable rather than merely
    narrow.
    """

    def test_optional_list_is_recognised_as_a_list_field(self):
        typing_doc = TypingOuter(opt_list=[_Inner(city="Seattle")])
        pep604_doc = Pep604Outer(opt_list=[_Inner(city="Seattle")])
        assert typing_doc._is_list_field("opt_list") is True
        assert pep604_doc._is_list_field("opt_list") is True

    def test_optional_scalar_is_not_a_list_field(self):
        typing_doc = TypingOuter(opt_str="x")
        pep604_doc = Pep604Outer(opt_str="x")
        assert typing_doc._is_list_field("opt_str") is False
        assert pep604_doc._is_list_field("opt_str") is False

    def test_optional_list_of_models_type_check_agrees(self):
        assert (
            StructuredModel._is_list_of_structured_model_type(Optional[List[_Inner]])
            is True
        )
        assert (
            StructuredModel._is_list_of_structured_model_type(list[_Inner] | None) is True
        )
