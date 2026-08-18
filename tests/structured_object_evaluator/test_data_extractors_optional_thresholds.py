"""Regression test for issue #149 review (@adiadd): HTML report thresholds.

The Optional[T] widening means a non-required nested object on a schema-built
model is annotated ``Optional[NestedModel]`` (and a non-required list of objects
``Optional[List[NestedModel]]``). ``DataExtractor.extract_all_field_thresholds``
walked into nested models via ``hasattr(field_type, '__fields__')`` and into
lists via ``__origin__ is list`` but never unwrapped ``Optional[...]``, so nested
field thresholds silently vanished from HTML reports for those fields.

``TestPEP604Spelling`` below covers the half #149 left behind (issue #162). The
fix tested for ``get_origin(field_type) is Union``, which is False for a PEP 604
union -- ``X | None`` has origin ``types.UnionType`` -- so that spelling was
still skipped. 0.7.0 raises the floor to Python 3.10, where ``X | None`` is the
idiomatic way to write it, which makes the gap far more reachable than it was.

The failure was silent twice over: the keys were simply absent, and the
enclosing ``try/except Exception`` in the extractor only logs at warning level.
So the assertions here compare the two spellings' output for *equality* rather
than checking that a key exists.
"""

import sys
import types
from typing import List, Optional, Union, get_origin

from stickler.reporting.html.utils.data_extractors import DataExtractor
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class TestOptionalNestedThresholds:
    def test_optional_nested_object_thresholds_extracted(self):
        """Thresholds on an OPTIONAL nested object's inner fields survive."""
        schema = {
            "type": "object",
            "properties": {
                "addr": {  # OPTIONAL nested object -> Optional[NestedModel]
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "x-aws-stickler-threshold": 0.9},
                    },
                    "required": ["city"],
                },
            },
            "required": [],
        }

        M = StructuredModel.from_json_schema(schema)
        thresholds = DataExtractor.extract_all_field_thresholds(M)

        assert "addr.city" in thresholds, (
            "nested threshold on an optional nested object was dropped "
            "(Optional[NestedModel] not unwrapped)"
        )
        assert thresholds["addr.city"] == 0.9

    def test_optional_list_of_objects_thresholds_extracted(self):
        """Thresholds on an OPTIONAL list-of-objects' inner fields survive."""
        schema = {
            "type": "object",
            "properties": {
                "items": {  # OPTIONAL array of objects -> Optional[List[NestedModel]]
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string", "x-aws-stickler-threshold": 0.8},
                        },
                        "required": ["sku"],
                    },
                },
            },
            "required": [],
        }

        M = StructuredModel.from_json_schema(schema)
        thresholds = DataExtractor.extract_all_field_thresholds(M)

        assert "items.sku" in thresholds, (
            "nested threshold on an optional list-of-objects was dropped "
            "(Optional[List[NestedModel]] not unwrapped)"
        )
        assert thresholds["items.sku"] == 0.8


class Nested(StructuredModel):
    """Shared nested model. Its own threshold is what must survive unwrapping."""

    city: str = ComparableField(threshold=0.9)


class Other(StructuredModel):
    """A second nested model, for the multi-arm union case."""

    region: str = ComparableField(threshold=0.6)


class Item(StructuredModel):
    sku: str = ComparableField(threshold=0.8)


class TestPEP604Spelling:
    """`X | None` must behave exactly like `Optional[X]` (issue #162).

    These models are declared directly rather than built with
    ``from_json_schema``, because the schema builder chooses the annotation
    spelling itself and so cannot exercise the PEP 604 path.
    """

    def test_both_spellings_land_in_the_widened_check(self):
        """Guard the guard: both origins must be ones the extractor accepts.

        Asserted as tuple membership rather than identity because **Python 3.14
        unified typing.Union and types.UnionType**: there, ``X | None`` and
        ``Optional[X]`` have the same origin and the widened check is correct
        but redundant. Before 3.14 they are distinct, which is what made
        checking typing.Union alone silently skip ``X | None`` (#162).

        The version-specific half is asserted below so the reason for the tuple
        is still on the record rather than looking like defensive padding. If
        the equality tests above ever started passing for an unrelated reason,
        this is what says which regime they ran in.
        """
        accepted = (Union, types.UnionType)

        assert get_origin(Nested | None) in accepted
        assert get_origin(Optional[Nested]) in accepted

        if sys.version_info < (3, 14):
            # The defect's precondition: two distinct origins, so the
            # single-origin check this fix replaced could not see `X | None`.
            assert get_origin(Nested | None) is types.UnionType
            assert get_origin(Nested | None) is not Union
            assert get_origin(Optional[Nested]) is Union
        else:
            assert get_origin(Nested | None) is get_origin(Optional[Nested])

    def test_optional_and_pep604_nested_object_agree(self):
        class OptionalSpelling(StructuredModel):
            addr: Optional[Nested] = ComparableField(default=None, threshold=0.7)

        class PipeSpelling(StructuredModel):
            addr: Nested | None = ComparableField(default=None, threshold=0.7)

        from_optional = DataExtractor.extract_all_field_thresholds(OptionalSpelling)
        from_pipe = DataExtractor.extract_all_field_thresholds(PipeSpelling)

        assert from_pipe == from_optional, (
            "`Nested | None` extracted different thresholds from "
            "`Optional[Nested]`; the PEP 604 union was not unwrapped"
        )
        assert from_pipe["addr.city"] == 0.9

    def test_optional_and_pep604_list_of_objects_agree(self):
        # No `threshold` on a List[StructuredModel] field: Hungarian matching
        # uses the element class's `match_threshold`, and StructuredModel
        # rejects the combination outright.
        class OptionalSpelling(StructuredModel):
            items: Optional[List[Item]] = ComparableField(default=None)

        class PipeSpelling(StructuredModel):
            items: list[Item] | None = ComparableField(default=None)

        from_optional = DataExtractor.extract_all_field_thresholds(OptionalSpelling)
        from_pipe = DataExtractor.extract_all_field_thresholds(PipeSpelling)

        assert from_pipe == from_optional, (
            "`list[Item] | None` extracted different thresholds from "
            "`Optional[List[Item]]`"
        )
        assert from_pipe["items.sku"] == 0.8

    def test_a_multi_arm_union_is_left_alone(self):
        """Two non-None arms means no single nested model to descend into.

        The single-non-None-arg guard must keep this case unwrapped rather than
        picking an arbitrary arm -- and must not raise.
        """

        class PipeSpelling(StructuredModel):
            thing: Nested | Other | None = ComparableField(default=None, threshold=0.7)

        class TypingSpelling(StructuredModel):
            thing: Union[Nested, Other, None] = ComparableField(
                default=None, threshold=0.7
            )

        from_pipe = DataExtractor.extract_all_field_thresholds(PipeSpelling)
        from_typing = DataExtractor.extract_all_field_thresholds(TypingSpelling)

        assert from_pipe == from_typing
        assert not [key for key in from_pipe if key.startswith("thing.")], (
            "a multi-arm union was unwrapped to one of its arms"
        )
        # The field's own threshold is still reported; only the descent stops.
        assert from_pipe["thing"] == 0.7

    def test_a_required_nested_field_is_unaffected(self):
        """The path that always worked keeps working."""

        class Required(StructuredModel):
            addr: Nested = ComparableField(threshold=0.7)

        thresholds = DataExtractor.extract_all_field_thresholds(Required)

        assert thresholds["addr.city"] == 0.9
