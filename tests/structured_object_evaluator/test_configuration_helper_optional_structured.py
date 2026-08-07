"""Tests for ConfigurationHelper.is_structured_field_type on Optional[StructuredModel].

Regression for issue #149 review (@adiadd): the annotation-level fix widens
non-required nested object fields to ``Optional[NestedModel]``, but
``is_structured_field_type`` only recognized ``StructuredModel`` and
``List[StructuredModel]`` (bare or Optional-wrapped) — never a bare
``Optional[StructuredModel]``. That routed optional nested objects inside
list-of-object items down the non-hierarchical (flat-counts) path, silently
dropping the nested metric breakdown.
"""

from typing import List, Optional

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.configuration_helper import (
    ConfigurationHelper,
)
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class _Inner(StructuredModel):
    city: str = ComparableField(comparator=ExactComparator())


class _Outer(StructuredModel):
    req_obj: _Inner = ComparableField(comparator=ExactComparator())
    opt_obj: Optional[_Inner] = ComparableField(
        default=None, comparator=ExactComparator()
    )
    opt_list: Optional[List[_Inner]] = ComparableField(default=None)
    opt_str: Optional[str] = ComparableField(default=None, comparator=ExactComparator())


class TestIsStructuredFieldTypeOptional:
    """is_structured_field_type must recognize Optional[StructuredModel]."""

    def test_bare_structured_model_is_structured(self):
        fi = _Outer.model_fields["req_obj"]
        assert ConfigurationHelper.is_structured_field_type(fi) is True

    def test_optional_structured_model_is_structured(self):
        """Optional[StructuredModel] must be recognized as structured (#149)."""
        fi = _Outer.model_fields["opt_obj"]
        assert ConfigurationHelper.is_structured_field_type(fi) is True

    def test_optional_list_of_structured_model_is_structured(self):
        """Pre-existing support: Optional[List[StructuredModel]] stays structured."""
        fi = _Outer.model_fields["opt_list"]
        assert ConfigurationHelper.is_structured_field_type(fi) is True

    def test_optional_primitive_is_not_structured(self):
        """Optional[str] must NOT be treated as structured."""
        fi = _Outer.model_fields["opt_str"]
        assert ConfigurationHelper.is_structured_field_type(fi) is False
