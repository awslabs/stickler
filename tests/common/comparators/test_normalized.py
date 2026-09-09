"""Behavior and integration tests for NormalizedComparator."""

import json

from stickler import ComparableField, NormalizedComparator, StructuredModel, eval_for
from stickler.structured_object_evaluator.models.comparator_registry import (
    create_comparator,
    get_global_registry,
)


class TestDefaultNormalization:
    """The default policy ignores formatting, not semantic symbols or accents."""

    def setup_method(self):
        self.comparator = NormalizedComparator()

    def test_ascii_formatting_differences_match(self):
        assert self.comparator.compare("U.S.A.", "USA") == 1.0
        assert self.comparator.compare("John  Smith", "john smith") == 1.0

    def test_unicode_punctuation_and_whitespace_are_removed(self):
        assert self.comparator.compare("it’s ready", "itsready") == 1.0
        assert self.comparator.compare("Ａ－Ｂ", "ＡＢ") == 1.0

    def test_symbols_are_not_punctuation(self):
        assert self.comparator.compare("$1,247.50", "$124750") == 1.0
        assert self.comparator.compare("$1,247.50", "124750") == 0.0
        assert self.comparator.compare("x±y", "xy") == 0.0
        assert self.comparator.compare("emoji🎉here", "emojihere") == 0.0

    def test_accents_are_preserved_and_canonicalized(self):
        assert self.comparator.compare("côte", "cote") == 0.0
        assert self.comparator.compare("côte", "co\u0302te") == 1.0


class TestConfiguration:
    """Every transform can be selected independently and serialized."""

    def test_case_can_be_preserved(self):
        comparator = NormalizedComparator(case_sensitive=True)
        assert comparator.compare("Hello!", "hello") == 0.0

    def test_whitespace_can_be_preserved(self):
        comparator = NormalizedComparator(ignore_whitespace=False)
        assert comparator.compare("John Smith", "JohnSmith") == 0.0
        assert comparator.compare("John-Smith", "JohnSmith") == 1.0

    def test_punctuation_can_be_preserved(self):
        comparator = NormalizedComparator(ignore_punctuation=False)
        assert comparator.compare("U.S.A.", "USA") == 0.0
        assert comparator.compare("U S A", "USA") == 1.0

    def test_config_omits_defaults_and_round_trips(self):
        assert NormalizedComparator().config is None
        original = NormalizedComparator(
            case_sensitive=True,
            ignore_whitespace=False,
            ignore_punctuation=False,
        )
        encoded = json.loads(json.dumps(original.config))
        rebuilt = create_comparator("NormalizedComparator", encoded)

        assert rebuilt.config == original.config
        assert rebuilt.compare("A-B", "a b") == 0.0

    def test_name_and_repr_describe_the_policy(self):
        comparator = NormalizedComparator(ignore_punctuation=False)
        assert comparator.name == "normalized"
        assert "ignore_punctuation=False" in repr(comparator)


class TestFrameworkIntegration:
    """The comparator participates in registry, schema, and explain flows."""

    def test_registry_exposes_the_canonical_class(self):
        registry = get_global_registry()
        assert registry.is_registered("NormalizedComparator")
        assert registry.get("NormalizedComparator") is NormalizedComparator

    def test_json_schema_round_trip_preserves_configuration(self):
        class Label(StructuredModel):
            value: str = ComparableField(
                comparator=NormalizedComparator(ignore_punctuation=False)
            )

        schema = Label.to_json_schema()
        property_schema = schema["properties"]["value"]
        assert property_schema["x-aws-stickler-comparator"] == "NormalizedComparator"
        assert property_schema["x-aws-stickler-comparator-config"] == {
            "ignore_punctuation": False
        }

        rebuilt = StructuredModel.from_json_schema(schema)
        assert rebuilt(value="U.S.A.").compare(rebuilt(value="USA")) == 0.0
        assert rebuilt(value="U S A").compare(rebuilt(value="USA")) == 1.0

    def test_explain_reports_explicit_comparator(self):
        class Label(StructuredModel):
            value: str = ComparableField(comparator=NormalizedComparator())

        explanation = eval_for(Label).explain()["value"]
        assert explanation["comparator"] == "NormalizedComparator"
        assert explanation["source"] == "explicit"
