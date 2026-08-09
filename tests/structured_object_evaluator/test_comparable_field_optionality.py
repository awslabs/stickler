"""Tests for ComparableField optionality following Pydantic semantics (#189).

This test verifies that:
1. Fields without default= are required
2. Fields with default=None are optional
3. The JSON schema correctly reflects required vs optional
4. Model validation enforces required fields
"""

from typing import Optional

import pytest
from pydantic import ValidationError

from stickler import ComparableField, StructuredModel


class TestComparableFieldOptionality:
    """Test that ComparableField follows Pydantic optionality semantics."""

    def test_field_without_default_is_required(self):
        """A field with no default should be required."""

        class Model(StructuredModel):
            name: str = ComparableField(threshold=0.8)

        # Should raise ValidationError when required field is missing
        with pytest.raises(ValidationError) as exc_info:
            Model()

        # Verify the error mentions the missing field
        assert "name" in str(exc_info.value)

    def test_field_with_default_none_is_optional(self):
        """A field with default=None should be optional."""

        class Model(StructuredModel):
            name: Optional[str] = ComparableField(threshold=0.8, default=None)

        # Should work without providing the field
        m = Model()
        assert m.name is None

    def test_field_with_explicit_default_value(self):
        """A field with a non-None default should use that default."""

        class Model(StructuredModel):
            count: int = ComparableField(threshold=1.0, default=0)

        m = Model()
        assert m.count == 0

    def test_json_schema_required_list(self):
        """JSON schema should correctly list required fields."""

        class Model(StructuredModel):
            required_name: str = ComparableField(threshold=0.8)
            optional_note: Optional[str] = ComparableField(threshold=0.6, default=None)
            required_id: str = ComparableField(threshold=1.0)

        schema = Model.model_json_schema()

        # Required fields should be in the required list
        assert "required_name" in schema["required"]
        assert "required_id" in schema["required"]

        # Optional fields should NOT be in the required list
        assert "optional_note" not in schema["required"]

    def test_json_schema_type_rendering(self):
        """JSON schema types should reflect nullability correctly."""

        class Model(StructuredModel):
            required_str: str = ComparableField()
            optional_str: Optional[str] = ComparableField(default=None)

        schema = Model.model_json_schema()

        # Required string should have type "string"
        assert schema["properties"]["required_str"]["type"] == "string"

        # Optional string should be nullable
        # (Pydantic renders Optional[str] as anyOf or type: ["string", "null"])
        opt_schema = schema["properties"]["optional_str"]
        is_nullable = (
            opt_schema.get("type") == ["string", "null"]
            or "anyOf" in opt_schema
            or opt_schema.get("type") is None  # anyOf without type
        )
        assert is_nullable, f"Optional field should be nullable, got: {opt_schema}"

    def test_mixed_required_optional_model(self):
        """A model with both required and optional fields should work correctly."""

        class Invoice(StructuredModel):
            invoice_id: str = ComparableField(threshold=1.0)
            amount: float = ComparableField(threshold=0.95)
            note: Optional[str] = ComparableField(threshold=0.7, default=None)

        # Valid with all fields
        inv1 = Invoice(invoice_id="INV-001", amount=100.0, note="Paid")
        assert inv1.invoice_id == "INV-001"
        assert inv1.amount == 100.0
        assert inv1.note == "Paid"

        # Valid with optional field omitted
        inv2 = Invoice(invoice_id="INV-002", amount=200.0)
        assert inv2.invoice_id == "INV-002"
        assert inv2.amount == 200.0
        assert inv2.note is None

        # Invalid: missing required field
        with pytest.raises(ValidationError):
            Invoice(invoice_id="INV-003")  # missing amount

    def test_comparison_with_required_fields(self):
        """Comparison should work correctly with required fields."""

        class Model(StructuredModel):
            name: str = ComparableField(threshold=0.8)

        gt = Model(name="Alice")
        pred = Model(name="Alice")

        result = gt.compare_with(pred)
        assert result["overall_score"] == pytest.approx(1.0)

    def test_comparison_with_optional_none_values(self):
        """Comparison should handle optional None values correctly."""

        class Model(StructuredModel):
            name: str = ComparableField(threshold=0.8)
            nickname: Optional[str] = ComparableField(threshold=0.7, default=None)

        gt = Model(name="Alice", nickname=None)
        pred = Model(name="Alice", nickname=None)

        result = gt.compare_with(pred, include_confusion_matrix=True)
        assert result["overall_score"] == pytest.approx(1.0)

        # Both None should be TN, not causing errors
        nickname_cm = result["confusion_matrix"]["fields"]["nickname"]["overall"]
        assert nickname_cm["tn"] == 1

    def test_ellipsis_default_is_required(self):
        """Using default=... (Ellipsis) should also create a required field."""

        class Model(StructuredModel):
            name: str = ComparableField(threshold=0.8, default=...)

        with pytest.raises(ValidationError):
            Model()

        # But providing it should work
        m = Model(name="test")
        assert m.name == "test"


class TestBackwardCompatibilityMigration:
    """Tests to help users migrate from the old behavior.

    Before the fix, `name: str = ComparableField()` was optional because
    ComparableField always passed default=None to Field(). After the fix,
    it's required. Users who want optional should add `default=None`.
    """

    def test_old_pattern_now_required(self):
        """The old pattern without default= is now required."""

        class OldStyleModel(StructuredModel):
            # This used to be optional (silently defaulted to None)
            # Now it's required as Pydantic semantics dictate
            name: str = ComparableField(threshold=0.8)

        # This would have worked before but should fail now
        with pytest.raises(ValidationError):
            OldStyleModel()

    def test_migration_pattern(self):
        """Show how to migrate to make a field optional."""

        class MigratedModel(StructuredModel):
            # To keep the old behavior, explicitly add default=None
            name: Optional[str] = ComparableField(threshold=0.8, default=None)

        # Now it works without providing the field
        m = MigratedModel()
        assert m.name is None
