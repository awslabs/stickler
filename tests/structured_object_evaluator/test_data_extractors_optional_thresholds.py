"""Regression test for issue #149 review (@adiadd): HTML report thresholds.

The Optional[T] widening means a non-required nested object on a schema-built
model is annotated ``Optional[NestedModel]`` (and a non-required list of objects
``Optional[List[NestedModel]]``). ``DataExtractor.extract_all_field_thresholds``
walked into nested models via ``hasattr(field_type, '__fields__')`` and into
lists via ``__origin__ is list`` but never unwrapped ``Optional[...]``, so nested
field thresholds silently vanished from HTML reports for those fields.
"""

from stickler.reporting.html.utils.data_extractors import DataExtractor
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
