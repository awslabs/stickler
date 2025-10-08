

"""Test validation for StructuredModel field configurations."""

import pytest
from typing import List

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


class TestStructuredModelValidation:
    """Test that StructuredModel validation prevents incorrect field configurations."""

    def test_threshold_validation_on_list_structured_model_field(self):
        """Test that threshold parameter on List[StructuredModel] fields raises ValueError."""

        # First, create a valid StructuredModel to use in the list
        class Product(StructuredModel):
            name: str = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
            )
            match_threshold = 0.7

        # Test that threshold parameter on List[StructuredModel] field raises ValueError
        with pytest.raises(ValueError) as exc_info:

            class OrderWithThreshold(StructuredModel):
                order_id: str = ComparableField(threshold=1.0, weight=1.0)
                products: List[Product] = ComparableField(
                    threshold=0.9,  # This should raise an error
                    weight=3.0,
                )

        error_message = str(exc_info.value)
        assert "cannot have a 'threshold' parameter" in error_message
        assert (
            "Hungarian matching uses each StructuredModel's 'match_threshold'"
            in error_message
        )
        assert "Set 'match_threshold = 0.9'" in error_message

    def test_comparator_validation_on_list_structured_model_field(self):
        """Test that comparator parameter on List[StructuredModel] fields raises ValueError."""

        # First, create a valid StructuredModel to use in the list
        class Product(StructuredModel):
            name: str = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
            )
            match_threshold = 0.7

        # Test that comparator parameter on List[StructuredModel] field raises ValueError
        with pytest.raises(ValueError) as exc_info:

            class OrderWithComparator(StructuredModel):
                order_id: str = ComparableField(threshold=1.0, weight=1.0)
                products: List[Product] = ComparableField(
                    comparator=ExactComparator(),  # This should raise an error
                    weight=3.0,
                )

        error_message = str(exc_info.value)
        assert "cannot have a 'comparator' parameter" in error_message
        assert (
            "Object comparison uses each StructuredModel's individual field comparators"
            in error_message
        )

    def test_both_threshold_and_comparator_validation(self):
        """Test that both threshold and comparator parameters raise ValueError (catches first error)."""

        # First, create a valid StructuredModel to use in the list
        class Product(StructuredModel):
            name: str = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
            )
            match_threshold = 0.7

        # Test that having both parameters raises ValueError (should catch threshold first)
        with pytest.raises(ValueError) as exc_info:

            class OrderWithBoth(StructuredModel):
                order_id: str = ComparableField(threshold=1.0, weight=1.0)
                products: List[Product] = ComparableField(
                    threshold=0.9,  # This should be caught first
                    comparator=ExactComparator(),  # This would also be an error
                    weight=3.0,
                )

        error_message = str(exc_info.value)
        # Should catch the threshold error first
        assert "cannot have a 'threshold' parameter" in error_message

    def test_valid_list_structured_model_field_creation(self):
        """Test that valid List[StructuredModel] fields can be created without errors."""

        # First, create a valid StructuredModel to use in the list
        class Product(StructuredModel):
            name: str = ComparableField(
                comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
            )
            price: float = ComparableField(threshold=1.0, weight=1.0)
            match_threshold = 0.7

        # This should NOT raise any errors
        try:

            class ValidOrder(StructuredModel):
                order_id: str = ComparableField(threshold=1.0, weight=1.0)
                products: List[Product] = ComparableField(
                    weight=3.0,  # Weight is allowed
                    # Aggregate is allowed
                )
        except ValueError:
            pytest.fail("Valid Order class should not raise ValueError")

        # Test that we can actually instantiate and use the model
        order = ValidOrder(
            order_id="ORD-001",
            products=[
                Product(name="Widget A", price=10.99),
                Product(name="Widget B", price=15.50),
            ],
        )

        assert order.order_id == "ORD-001"
        assert len(order.products) == 2
        assert order.products[0].name == "Widget A"

    def test_validation_ignores_non_list_structured_model_fields(self):
        """Test that validation only applies to List[StructuredModel] fields."""

        # These should all be allowed - validation only applies to List[StructuredModel]
        try:

            class ValidModel(StructuredModel):
                # Regular fields with threshold/comparator are allowed
                name: str = ComparableField(
                    threshold=0.8, comparator=LevenshteinComparator(), weight=1.0
                )

                # List of primitives with threshold/comparator are allowed
                tags: List[str] = ComparableField(
                    threshold=0.7, comparator=ExactComparator(), weight=2.0
                )

                # Single StructuredModel field with threshold/comparator are allowed
                # (Note: This would be unusual but not prohibited by this validation)

        except ValueError:
            pytest.fail(
                "Valid model with non-List[StructuredModel] fields should not raise ValueError"
            )

    def test_error_message_includes_correct_threshold_value(self):
        """Test that error messages include the actual threshold value that was used."""

        class Product(StructuredModel):
            name: str = ComparableField(threshold=0.8, weight=1.0)
            match_threshold = 0.7

        # Test with a specific threshold value
        with pytest.raises(ValueError) as exc_info:

            class OrderWithSpecificThreshold(StructuredModel):
                products: List[Product] = ComparableField(
                    threshold=0.95,  # Specific value to check in error message
                    weight=3.0,
                )

        error_message = str(exc_info.value)
        assert "Set 'match_threshold = 0.95'" in error_message

    def test_validation_works_with_different_threshold_values(self):
        """Test that validation catches different threshold values correctly."""

        class Product(StructuredModel):
            name: str = ComparableField(threshold=0.8, weight=1.0)
            match_threshold = 0.7

        # Test with default threshold (should not raise error since it equals default)
        try:

            class OrderWithDefaultThreshold(StructuredModel):
                products: List[Product] = ComparableField(
                    # Using default threshold (0.5) should be allowed
                    weight=3.0
                )
        except ValueError:
            pytest.fail("Order with default threshold should not raise error")

        # Test with non-default threshold (should raise error)
        with pytest.raises(ValueError):

            class OrderWithNonDefaultThreshold(StructuredModel):
                products: List[Product] = ComparableField(
                    threshold=0.6,  # Different from default 0.5, should raise error
                    weight=3.0,
                )
