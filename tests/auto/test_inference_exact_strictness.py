"""Inference rules state their case sensitivity, rather than inheriting it.

Several name-token rules route fields to ``ExactComparator``. That comparator's
default flipped from lenient to strict in #199/#220, which silently redefined
what those rules meant: fields the rules deliberately sent to Exact went from
matching formatting-only differences to never matching them.

These tests pin each rule's *intent* so a future change to a comparator default
cannot quietly change inference behaviour again. They assert on the score a
zero-config user actually gets, not on the inferred spec, because the spec being
right is only useful if it survives being built into a model.

See https://github.com/awslabs/stickler/issues/242
"""

import warnings

import pytest
from pydantic import BaseModel

import stickler
from stickler.auto.inference import infer_field_config
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


def _score(model_cls, field, gt_value, pred_value):
    """Score one field through the zero-config path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = model_cls(**{field: gt_value})
        pred = model_cls(**{field: pred_value})
        return stickler.evaluate(gt, pred).field_scores[field]


class TestCaseInsensitiveByRule:
    """Email and URL are case-insensitive by specification."""

    @pytest.mark.parametrize(
        "field, gt, pred",
        [
            ("email", "A.Buyer@Example.COM", "a.buyer@example.com"),
            ("email", "User@example.com", "user@example.com"),
            ("url", "https://Example.com/path", "https://example.com/path"),
            ("uri", "HTTPS://Example.COM", "https://example.com"),
        ],
    )
    def test_case_only_difference_matches(self, field, gt, pred):
        model = type(
            "M", (BaseModel,), {"__annotations__": {field: str}}
        )

        assert _score(model, field, gt, pred) == 1.0

    def test_genuinely_different_email_is_still_a_non_match(self):
        """Case-insensitive, not fuzzy.

        A similarity comparator would score this 0.857, which is a false
        near-match on a field where one character changes the recipient.
        """
        model = type("M", (BaseModel,), {"__annotations__": {"email": str}})

        assert _score(model, "email", "a@b.com", "a@c.com") == 0.0

    def test_the_rule_carries_the_flag_explicitly(self):
        """The intent is in the rule, not inherited from a comparator default."""
        model = type("M", (BaseModel,), {"__annotations__": {"email": str}})
        spec = infer_field_config("email", model.model_fields["email"])

        assert spec.comparator_name == "ExactComparator"
        assert spec.comparator_config.get("case_sensitive") is False


class TestCaseSensitiveByRule:
    """Identifiers stay strict: a different case may be a different ID."""

    @pytest.mark.parametrize(
        "field, gt, pred",
        [
            ("invoice_id", "INV-001", "inv-001"),
            ("sku", "AB-77", "ab-77"),
            ("customer_code", "XY9", "xy9"),
        ],
    )
    def test_case_difference_is_a_non_match(self, field, gt, pred):
        model = type("M", (BaseModel,), {"__annotations__": {field: str}})

        assert _score(model, field, gt, pred) == 0.0

    def test_the_rule_carries_the_flag_explicitly(self):
        model = type("M", (BaseModel,), {"__annotations__": {"invoice_id": str}})
        spec = infer_field_config("invoice_id", model.model_fields["invoice_id"])

        assert spec.comparator_config.get("case_sensitive") is True


class TestPhoneIsParsedNotCompared:
    """Phone numbers route to PhoneComparator, which parses both sides."""

    @pytest.mark.parametrize(
        "gt, pred",
        [
            ("206-555-0100", "(206) 555-0100"),
            ("+1-206-555-0100", "2065550100"),
            ("206.555.0100", "206-555-0100"),
            ("+1 (206) 555-0100 ext. 89", "+12065550100x89"),
        ],
    )
    def test_formatting_only_difference_matches(self, gt, pred):
        model = type("M", (BaseModel,), {"__annotations__": {"phone_num": str}})

        assert _score(model, "phone_num", gt, pred) == 1.0

    def test_a_different_number_is_still_a_non_match(self):
        model = type("M", (BaseModel,), {"__annotations__": {"phone_num": str}})

        assert _score(model, "phone_num", "206-555-0100", "206-555-0101") == 0.0

    def test_the_rule_selects_the_phone_comparator(self):
        model = type("M", (BaseModel,), {"__annotations__": {"phone_num": str}})
        spec = infer_field_config("phone_num", model.model_fields["phone_num"])

        assert spec.comparator_name == "PhoneComparator"


class TestPostalCodesAreDeliberatelyStrict:
    """Postal codes stay exact, and that is a decision rather than an oversight.

    Formats are country-specific in ways a generic normalizer gets wrong: a UK
    postcode's internal space is significant, Dutch codes mix letters and
    digits. Stripping punctuation would be right for the US and quietly wrong
    elsewhere, so the guidance is to write a comparator for your own country
    (docs/docs/Guides/Comparators/postal-codes.md).
    """

    @pytest.mark.parametrize(
        "gt, pred",
        [
            ("98101-1234", "98101 1234"),
            ("98101", "98101-1234"),
        ],
    )
    def test_formatting_only_difference_does_not_match(self, gt, pred):
        model = type("M", (BaseModel,), {"__annotations__": {"zip_code": str}})

        assert _score(model, "zip_code", gt, pred) == 0.0

    def test_identical_postal_codes_match(self):
        model = type("M", (BaseModel,), {"__annotations__": {"zip_code": str}})

        assert _score(model, "zip_code", "98101-1234", "98101-1234") == 1.0

    def test_a_similarity_comparator_would_not_fix_it(self):
        """Why postal codes wait for a domain-specific comparator.

        Levenshtein cannot separate the cases: a genuinely different postal code
        scores the same as the same code reformatted, so no threshold works.
        """
        from stickler.comparators.levenshtein import LevenshteinComparator

        same_code_reformatted = LevenshteinComparator().compare(
            "98101-1234", "98101 1234"
        )
        different_code = LevenshteinComparator().compare("98101-1234", "98102-1234")

        assert different_code >= same_code_reformatted


class TestInferredConfigSurvivesARoundTrip:
    """The inferred flag has to reach a rebuilt model, not just the spec."""

    def test_case_insensitive_email_survives_export_and_reimport(self):
        class Doc(BaseModel):
            email: str

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = StructuredModel.from_pydantic(Doc)
            config = model.to_stickler_config()

            assert config["fields"]["email"]["comparator_config"] == {
                "case_sensitive": False
            }

            rebuilt = StructuredModel.model_from_json(config)
            gt = rebuilt.from_json({"email": "A@B.com"})
            pred = rebuilt.from_json({"email": "a@b.com"})

        assert gt.compare_with(pred)["field_scores"]["email"] == 1.0
