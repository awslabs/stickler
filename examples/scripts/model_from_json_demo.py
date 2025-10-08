#!/usr/bin/env python3
"""
Demonstration of StructuredModel.model_from_json() functionality.

This script shows how to create StructuredModel classes dynamically from JSON
configuration, enabling configuration-driven model creation with full comparison
capabilities including nested models and custom comparators.
"""

import json
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


def demo_basic_model_creation():
    """Demonstrate basic model creation from JSON configuration."""
    print("=== Basic Model Creation Demo ===")

    # Define a simple person model configuration
    person_config = {
        "model_name": "Person",
        "fields": {
            "name": {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "threshold": 0.8,
                "weight": 1.0,
                "required": True,
            },
            "age": {
                "type": "int",
                "comparator": "NumericComparator",
                "threshold": 0.9,
                "weight": 0.5,
                "required": True,
            },
            "email": {
                "type": "str",
                "comparator": "ExactComparator",
                "threshold": 1.0,
                "weight": 1.5,
                "required": False,
                "default": None,
            },
        },
    }

    # Create the model class dynamically
    Person = StructuredModel.model_from_json(person_config)

    # Create instances
    person1 = Person(name="John Smith", age=30, email="john@example.com")
    person2 = Person(
        name="Jon Smith", age=31, email="john@example.com"
    )  # Slight differences

    # Compare them
    result = person1.compare_with(person2)

    print(f"Person 1: {person1}")
    print(f"Person 2: {person2}")
    print(f"Comparison Score: {result['overall_score']:.3f}")
    print(f"Field Scores: {result['field_scores']}")
    print()


def demo_nested_structured_models():
    """Demonstrate nested StructuredModel creation and comparison."""
    print("=== Nested StructuredModel Demo ===")

    # Define a company with nested employee models
    company_config = {
        "model_name": "Company",
        "fields": {
            "name": {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "threshold": 0.8,
                "weight": 2.0,
            },
            "ceo": {
                "type": "structured_model",
                "threshold": 0.7,
                "weight": 1.5,
                "fields": {
                    "name": {
                        "type": "str",
                        "comparator": "LevenshteinComparator",
                        "threshold": 0.8,
                        "weight": 1.0,
                    },
                    "salary": {
                        "type": "float",
                        "comparator": "NumericComparator",
                        "threshold": 0.9,
                        "weight": 0.8,
                    },
                },
            },
            "employees": {
                "type": "list_structured_model",
                "weight": 1.0,
                "match_threshold": 0.7,
                "fields": {
                    "name": {
                        "type": "str",
                        "comparator": "LevenshteinComparator",
                        "threshold": 0.8,
                        "weight": 1.0,
                    },
                    "department": {
                        "type": "str",
                        "comparator": "ExactComparator",
                        "threshold": 1.0,
                        "weight": 0.5,
                    },
                    "salary": {
                        "type": "float",
                        "comparator": "NumericComparator",
                        "threshold": 0.95,
                        "weight": 0.7,
                    },
                },
            },
        },
    }

    # Create the model class
    Company = StructuredModel.model_from_json(company_config)

    # Create test instances
    company1 = Company(
        name="TechCorp Inc",
        ceo={"name": "Alice Johnson", "salary": 250000.0},
        employees=[
            {"name": "Bob Smith", "department": "Engineering", "salary": 85000.0},
            {"name": "Carol Davis", "department": "Marketing", "salary": 70000.0},
            {"name": "David Wilson", "department": "Engineering", "salary": 90000.0},
        ],
    )

    company2 = Company(
        name="TechCorp LLC",  # Slight name difference
        ceo={"name": "Alice Johnson", "salary": 255000.0},  # Slight salary difference
        employees=[
            {
                "name": "David Wilson",
                "department": "Engineering",
                "salary": 92000.0,
            },  # Reordered
            {
                "name": "Bob Smith",
                "department": "Engineering",
                "salary": 87000.0,
            },  # Reordered
            {
                "name": "Carol Davis",
                "department": "Marketing",
                "salary": 72000.0,
            },  # Reordered
        ],
    )

    # Compare companies
    result = company1.compare_with(company2)

    print(f"Company 1: {company1.name}")
    print(f"  CEO: {company1.ceo.name} (${company1.ceo.salary:,.0f})")
    print(f"  Employees: {len(company1.employees)}")

    print(f"\nCompany 2: {company2.name}")
    print(f"  CEO: {company2.ceo.name} (${company2.ceo.salary:,.0f})")
    print(f"  Employees: {len(company2.employees)}")

    print(f"\nComparison Results:")
    print(f"  Overall Score: {result['overall_score']:.3f}")
    print(f"  Field Scores:")
    for field, score in result["field_scores"].items():
        print(f"    {field}: {score:.3f}")

    # Show detailed nested comparison
    if "nested_scores" in result:
        print(f"  Nested Scores:")
        for field, nested in result["nested_scores"].items():
            if isinstance(nested, dict) and "overall_score" in nested:
                print(f"    {field}: {nested['overall_score']:.3f}")
    print()


def demo_custom_comparators():
    """Demonstrate using different comparator types."""
    print("=== Custom Comparators Demo ===")

    # Product model with various comparator types
    product_config = {
        "model_name": "Product",
        "fields": {
            "name": {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "comparator_config": {"case_sensitive": False},
                "threshold": 0.8,
                "weight": 1.0,
            },
            "sku": {
                "type": "str",
                "comparator": "ExactComparator",
                "threshold": 1.0,
                "weight": 2.0,
            },
            "price": {
                "type": "float",
                "comparator": "NumericComparator",
                "comparator_config": {"tolerance": 0.05},  # 5% tolerance
                "threshold": 0.9,
                "weight": 1.5,
            },
            "description": {
                "type": "str",
                "comparator": "FuzzyComparator",
                "threshold": 0.7,
                "weight": 0.8,
            },
            "tags": {
                "type": "list",
                "comparator": "ExactComparator",
                "threshold": 0.8,
                "weight": 0.5,
            },
        },
    }

    # Create model
    Product = StructuredModel.model_from_json(product_config)

    # Create test products
    product1 = Product(
        name="Wireless Bluetooth Headphones",
        sku="WBH-001",
        price=99.99,
        description="High-quality wireless headphones with noise cancellation",
        tags=["electronics", "audio", "wireless", "bluetooth"],
    )

    product2 = Product(
        name="Wireless Bluetooth Headphone",  # Slight name difference
        sku="WBH-001",  # Same SKU
        price=104.99,  # 5% price difference
        description="Premium wireless headphones featuring noise cancellation technology",  # Different description
        tags=["electronics", "audio", "wireless"],  # Missing "bluetooth" tag
    )

    # Compare products
    result = product1.compare_with(product2)

    print(f"Product 1: {product1.name}")
    print(f"  SKU: {product1.sku}, Price: ${product1.price}")
    print(f"  Tags: {product1.tags}")

    print(f"\nProduct 2: {product2.name}")
    print(f"  SKU: {product2.sku}, Price: ${product2.price}")
    print(f"  Tags: {product2.tags}")

    print(f"\nComparison Results:")
    print(f"  Overall Score: {result['overall_score']:.3f}")
    print(f"  Field Scores:")
    for field, score in result["field_scores"].items():
        print(f"    {field}: {score:.3f}")
    print()


def demo_json_file_loading():
    """Demonstrate loading model configuration from JSON file."""
    print("=== JSON File Loading Demo ===")

    # Create a sample JSON configuration file
    config = {
        "model_name": "Invoice",
        "match_threshold": 0.8,
        "fields": {
            "invoice_id": {
                "type": "str",
                "comparator": "ExactComparator",
                "threshold": 1.0,
                "weight": 3.0,
                "required": True,
            },
            "customer_name": {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "threshold": 0.8,
                "weight": 2.0,
                "required": True,
            },
            "total_amount": {
                "type": "float",
                "comparator": "NumericComparator",
                "threshold": 0.95,
                "weight": 2.5,
                "required": True,
            },
            "line_items": {
                "type": "list_structured_model",
                "weight": 1.5,
                "match_threshold": 0.7,
                "fields": {
                    "product_name": {
                        "type": "str",
                        "comparator": "FuzzyComparator",
                        "threshold": 0.8,
                        "weight": 1.0,
                    },
                    "quantity": {
                        "type": "int",
                        "comparator": "NumericComparator",
                        "threshold": 1.0,
                        "weight": 0.8,
                    },
                    "unit_price": {
                        "type": "float",
                        "comparator": "NumericComparator",
                        "threshold": 0.95,
                        "weight": 1.2,
                    },
                },
            },
        },
    }

    # Save to temporary file
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        config_file = f.name

    try:
        # Load from file
        with open(config_file, "r") as f:
            loaded_config = json.load(f)

        # Create model from loaded config
        Invoice = StructuredModel.model_from_json(loaded_config)

        # Create test invoices
        invoice1 = Invoice(
            invoice_id="INV-2024-001",
            customer_name="Acme Corporation",
            total_amount=1250.00,
            line_items=[
                {"product_name": "Widget A", "quantity": 10, "unit_price": 50.0},
                {"product_name": "Widget B", "quantity": 5, "unit_price": 100.0},
                {"product_name": "Service Fee", "quantity": 1, "unit_price": 250.0},
            ],
        )

        invoice2 = Invoice(
            invoice_id="INV-2024-001",  # Same ID
            customer_name="ACME Corp",  # Slight name variation
            total_amount=1275.00,  # Slight amount difference
            line_items=[
                {
                    "product_name": "Service Fee",
                    "quantity": 1,
                    "unit_price": 275.0,
                },  # Reordered
                {
                    "product_name": "Widget A",
                    "quantity": 10,
                    "unit_price": 50.0,
                },  # Reordered
                {
                    "product_name": "Widget B",
                    "quantity": 5,
                    "unit_price": 100.0,
                },  # Reordered
            ],
        )

        # Compare invoices
        result = invoice1.compare_with(invoice2)

        print(f"Loaded model configuration from: {config_file}")
        print(f"Model: {Invoice.__name__}")
        print(f"\nInvoice 1: {invoice1.invoice_id}")
        print(f"  Customer: {invoice1.customer_name}")
        print(f"  Total: ${invoice1.total_amount}")
        print(f"  Items: {len(invoice1.line_items)}")

        print(f"\nInvoice 2: {invoice2.invoice_id}")
        print(f"  Customer: {invoice2.customer_name}")
        print(f"  Total: ${invoice2.total_amount}")
        print(f"  Items: {len(invoice2.line_items)}")

        print(f"\nComparison Results:")
        print(f"  Overall Score: {result['overall_score']:.3f}")
        print(f"  Field Scores:")
        for field, score in result["field_scores"].items():
            print(f"    {field}: {score:.3f}")

    finally:
        # Clean up temp file
        os.unlink(config_file)

    print()


def main():
    """Run all demonstrations."""
    print("StructuredModel.model_from_json() Demonstration")
    print("=" * 50)
    print()

    demo_basic_model_creation()
    demo_nested_structured_models()
    demo_custom_comparators()
    demo_json_file_loading()

    print("=" * 50)
    print("All demonstrations completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✓ Dynamic model creation from JSON configuration")
    print("✓ Nested StructuredModel support with recursive comparison")
    print("✓ List of StructuredModels with Hungarian matching")
    print("✓ Custom comparator configuration")
    print("✓ Flexible field weights and thresholds")
    print("✓ JSON file-based configuration loading")
    print("✓ Full comparison capabilities with detailed scoring")


if __name__ == "__main__":
    main()
