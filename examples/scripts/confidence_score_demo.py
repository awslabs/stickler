#!/usr/bin/env python3
"""
Example script demonstrating the Rich Value Pattern in StructuredModel.

This script shows how to:
1. Define a StructuredModel with various field types
2. Create instances from JSON data with rich values (value + metadata)
3. Access field values directly (unwrapped from rich values)
4. Access confidence scores via the API
"""

from typing import List, Optional

from stickler import (
    ComparableField,
    LevenshteinComparator,
    NumericComparator,
    StructuredModel,
)


# Define a Product model
class Product(StructuredModel):
    product_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0
    )
    price: float = ComparableField(
        comparator=NumericComparator(),
    )
    sku: str = ComparableField(comparator=LevenshteinComparator(), threshold=0.95)
    description: Optional[str] = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7
    )


# Define a more complex model with nested structures
class Address(StructuredModel):
    street: str = ComparableField(comparator=LevenshteinComparator())
    city: str = ComparableField(comparator=LevenshteinComparator())
    zip_code: str = ComparableField(comparator=LevenshteinComparator())


class Customer(StructuredModel):
    name: str = ComparableField(comparator=LevenshteinComparator())
    email: str = ComparableField(comparator=LevenshteinComparator())
    address: Address = ComparableField(comparator=LevenshteinComparator())
    orders: List[Product] = ComparableField()


def main():
    print("Rich Value Pattern Demo\n")

    # Example 1: Simple Product with mixed confidence/plain values
    print("1. Simple Product Example:")

    json_output = {
        "product_name": {"_value": "Widget Pro", "_confidence": 0.95},
        "price": {"_value": 29.99, "_confidence": 0.72},
        "sku": "ABC123",  # plain value still works
        "description": {
            "_value": "High-quality widget for professional use",
            "_confidence": 0.88,
        },
    }

    # Create Product instance
    prediction = Product.from_json(json_output)

    # Access field values directly (unwrapped)
    print(f"Prediction: {prediction}")
    print(f"Product Name: {prediction.product_name}")
    print(f"Price: ${prediction.price}")
    print(f"SKU: {prediction.sku}")
    print(f"Description: {prediction.description}")

    print("\nConfidence Scores:")
    print(
        f"product_name confidence: {prediction.get_field_confidence('product_name')}"
    )  # 0.95
    print(f"price confidence: {prediction.get_field_confidence('price')}")  # 0.72
    print(f"sku confidence: {prediction.get_field_confidence('sku')}")  # None
    print(
        f"description confidence: {prediction.get_field_confidence('description')}"
    )  # 0.88

    print("\n2. Nested Structure Example:")

    complex_json = {
        "name": {"_value": "John Doe", "_confidence": 0.92},
        "email": "john.doe@example.com",  # plain value
        "address": {
            "street": {"_value": "123 Main Street", "_confidence": 0.85},
            "city": {"_value": "New York", "_confidence": 0.98},
            "zip_code": "10001",  # plain value
        },
        "orders": [
            {
                "product_name": {"_value": "Laptop Pro", "_confidence": 0.89},
                "price": {"_value": 1299.99, "_confidence": 0.76},
                "sku": "LAP001",
                "description": {
                    "_value": "Professional laptop computer",
                    "_confidence": 0.91,
                },
            },
            {
                "product_name": "Mouse Wireless",
                "price": {"_value": 49.99, "_confidence": 0.83},
                "sku": {"_value": "MOU001", "_confidence": 0.94},
                "description": "Wireless optical mouse",
            },
        ],
    }

    # Create Customer instance
    customer = Customer.from_json(complex_json)

    # Access nested values
    print(f"Customer Name: {customer.name}")
    print(f"Customer Email: {customer.email}")
    print(
        f"Address: {customer.address.street}, {customer.address.city} {customer.address.zip_code}"
    )
    print(f"Number of Orders: {len(customer.orders)}")

    print("\nNested Confidence Scores:")
    print(f"name confidence: {customer.get_field_confidence('name')}")  # 0.92
    print(f"email confidence: {customer.get_field_confidence('email')}")  # None
    print(
        f"address.street confidence: {customer.get_field_confidence('address.street')}"
    )  # 0.85
    print(
        f"address.city confidence: {customer.get_field_confidence('address.city')}"
    )  # 0.98
    print(
        f"address.zip_code confidence: {customer.get_field_confidence('address.zip_code')}"
    )  # None

    # List item confidences (using array notation)
    print(
        f"orders[0].product_name confidence: {customer.get_field_confidence('orders[0].product_name')}"
    )  # 0.89
    print(
        f"orders[0].price confidence: {customer.get_field_confidence('orders[0].price')}"
    )  # 0.76
    print(
        f"orders[1].sku confidence: {customer.get_field_confidence('orders[1].sku')}"
    )  # 0.94

    print("\nOrder Details:")
    for i, order in enumerate(customer.orders):
        print(f"  Order {i + 1}: {order.product_name} - ${order.price}")

    print("\n3. All Confidences Summary:")

    all_confidences = prediction.get_all_confidences()
    print("Product confidences:")
    for field, confidence in all_confidences.items():
        print(f"  {field}: {confidence}")

    print("\nCustomer confidences:")
    all_customer_confidences = customer.get_all_confidences()
    for field, confidence in all_customer_confidences.items():
        print(f"  {field}: {confidence}")

    # Add AUROC testing
    test_auroc_functionality()

    print("\nAUROC TESTING COMPLETE")


def test_auroc_functionality():
    """Test AUROC calculation with various confidence scenarios."""
    print("\nAUROC FUNCTIONALITY TEST")

    # Test Case 1: Well-calibrated confidence (high AUROC expected)
    print("\n1. Well-Calibrated Confidence Test:")

    gt_well_calibrated = Product(product_name="Widget Pro", price=29.99, sku="ABC123")

    # Prediction with well-calibrated confidence
    pred_well_calibrated = Product.from_json(
        {
            "product_name": {
                "_value": "Widget Pro",
                "_confidence": 0.95,
            },  # High conf, correct
            "price": {"_value": 29.99, "_confidence": 0.90},  # High conf, correct
            "sku": {"_value": "XYZ789", "_confidence": 0.30},  # Low conf, incorrect
        }
    )

    try:
        result_well_calibrated = gt_well_calibrated.compare_with(
            pred_well_calibrated,
            add_confidence_metrics=True,
            document_field_comparisons=True,
        )

        auroc = result_well_calibrated["confidence_metrics"]["overall"]["auroc"]["value"]
        print(f"AUROC (well-calibrated): {auroc}")

    except Exception as e:
        print(f"Error in well-calibrated test: {e}")

    # Test Case 2: Poorly-calibrated confidence (low AUROC expected)
    print("\n2. Poorly-Calibrated Confidence Test:")

    gt_poor_calibrated = Product(product_name="Widget Pro", price=29.99, sku="ABC123")

    # Prediction with poorly-calibrated confidence
    pred_poor_calibrated = Product.from_json(
        {
            "product_name": {
                "_value": "Wrong Name",
                "_confidence": 0.95,
            },  # High conf, incorrect
            "price": {"_value": 50.00, "_confidence": 0.90},  # High conf, incorrect
            "sku": {"_value": "ABC123", "_confidence": 0.20},  # Low conf, correct
        }
    )

    try:
        result_poor_calibrated = gt_poor_calibrated.compare_with(
            pred_poor_calibrated,
            add_confidence_metrics=True,
            document_field_comparisons=True,
        )

        auroc = result_poor_calibrated["confidence_metrics"]["overall"]["auroc"]["value"]
        print(f"AUROC (poorly-calibrated): {auroc}")

    except Exception as e:
        print(f"Error in poorly-calibrated test: {e}")

    # Test Case 3: Mixed scenario with nested objects
    print("\n3. Nested Objects Confidence Test:")

    gt_nested = Customer(
        name="John Doe",
        email="john@example.com",
        address=Address(street="123 Main St", city="New York", zip_code="10001"),
        orders=[],  # Empty orders list for simplicity
    )

    pred_nested = Customer.from_json(
        {
            "name": {"_value": "John Doe", "_confidence": 0.95},
            "email": {"_value": "john@example.com", "_confidence": 0.85},
            "address": {
                "street": {"_value": "123 Main St", "_confidence": 0.90},
                "city": {
                    "_value": "Boston",
                    "_confidence": 0.70,
                },  # Incorrect but medium confidence
                "zip_code": {"_value": "10001", "_confidence": 0.95},
            },
            "orders": [],
        }
    )

    try:
        result_nested = gt_nested.compare_with(
            pred_nested, add_confidence_metrics=True, document_field_comparisons=True
        )
        auroc = result_nested["confidence_metrics"]["overall"]["auroc"]["value"]
        print(f"AUROC (nested): {auroc}")

    except Exception as e:
        print(f"Error in nested test: {e}")

    # Test Case 4: Simple comparison to show field_comparisons structure
    print("\n4. Field Comparisons Structure Test:")

    gt_simple = Product(product_name="Test Product", price=100.0, sku="TEST123")

    pred_simple = Product.from_json(
        {
            "product_name": {"_value": "Test Product", "_confidence": 0.95},
            "price": {"_value": 99.0, "_confidence": 0.80},
            "sku": {"_value": "WRONG", "_confidence": 0.30},
        }
    )

    try:
        result_simple = gt_simple.compare_with(
            pred_simple, document_field_comparisons=True
        )

        print("Field comparisons available:", "field_comparisons" in result_simple)
        if "field_comparisons" in result_simple:
            print(
                f"Number of field comparisons: {len(result_simple['field_comparisons'])}"
            )
            for i, comp in enumerate(
                result_simple["field_comparisons"][:3]
            ):  # Show first 3
                print(
                    f"  Comparison {i + 1}: {comp.get('actual_key', 'N/A')} -> match: {comp.get('match', 'N/A')}"
                )

        # Test confidence access
        confidences = pred_simple.get_all_confidences()
        print(f"\nConfidence data available: {bool(confidences)}")
        if confidences:
            print(f"Confidence keys: {list(confidences.keys())}")

    except Exception as e:
        print(f"Error in simple test: {e}")


if __name__ == "__main__":
    main()
