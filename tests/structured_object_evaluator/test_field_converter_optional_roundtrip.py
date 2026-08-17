"""Regression tests for the stickler-config path of issue #149 (@adiadd review).

The JSON-Schema path (json_schema_field_converter.py) widens non-required fields
to ``Optional[T]``. The parallel stickler-config path (field_converter.py, used by
``StructuredModel.model_from_json``) did not, so a schema-built model that is
exported via ``to_stickler_config`` and rebuilt via ``model_from_json`` round-trips
a fixed model back into a broken one: ``from_json(process_rich_values=True)`` then
crashes on the materialized ``None`` default exactly as in #149.
"""

from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class TestFieldConverterOptionalRoundTrip:
    def test_model_from_json_optional_field_accepts_none_via_from_json(self):
        """A non-required field built via model_from_json accepts an omitted value
        through the rich-value path."""
        config = {
            "model_name": "Doc",
            "fields": {
                "title": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "required": True,
                },
                "note": {
                    "type": "str",
                    "comparator": "LevenshteinComparator",
                    "required": False,  # optional -> Optional[str]
                },
            },
        }

        M = StructuredModel.model_from_json(config)
        data = {"title": "x"}

        plain = M(**data)
        rich = M.from_json(data, process_rich_values=True)
        assert plain.note is None
        assert rich.note is None
        assert plain.model_dump() == rich.model_dump()

    def test_schema_to_stickler_config_to_model_round_trip(self):
        """schema -> to_stickler_config -> model_from_json yields a WORKING model.

        Pins the round-trip hole @adiadd flagged: the optional inner field must
        survive the round-trip and still accept an omitted value through
        from_json(process_rich_values=True).
        """
        schema = {
            "type": "object",
            "properties": {
                "C": {
                    "type": "object",
                    "properties": {
                        "A": {"type": "string"},
                        "B": {"type": "string"},  # OPTIONAL inner
                    },
                    "required": ["A"],
                },
            },
            "required": ["C"],
        }

        M = StructuredModel.from_json_schema(schema)
        rebuilt = StructuredModel.model_from_json(M.to_stickler_config())

        data = {"C": {"A": "x"}}
        plain = rebuilt(**data)
        rich = rebuilt.from_json(data, process_rich_values=True)
        assert plain.C.B is None
        assert rich.C.B is None
        assert plain.model_dump() == rich.model_dump()
