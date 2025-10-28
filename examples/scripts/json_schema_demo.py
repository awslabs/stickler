"""
JSON Schema Model Construction Demo

This example demonstrates how to create StructuredModel classes directly from
JSON Schema documents and use them for evaluation.

Key Features:
- Create models from JSON Schema (Draft 7 compatible)
- Support for nested objects and arrays
- Custom comparison behavior via x-stickler extensions
- Full evaluation with confusion matrices and metrics
"""

from stickler.structured_object_evaluator.models import StructuredModel


def basic_json_schema_example():
    """Basic example: Create a model from a simple JSON Schema."""
    print("=" * 80)
    print("BASIC JSON SCHEMA EXAMPLE")
    print("=" * 80)
    
    # Define a JSON Schema for a product
    product_schema = {
        "type": "object",
        "title": "Product",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "in_stock": {"type": "boolean"},
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "price"]
    }
    
    # Create the model class from the schema
    Product = StructuredModel.from_json_schema(product_schema)
    
    # Create instances from JSON dictionaries (typical usage)
    ground_truth_json = {
        "name": "Laptop",
        "price": 999.99,
        "in_stock": True,
        "tags": ["electronics", "computers"]
    }
    
    prediction_json = {
        "name": "Laptop Pro",
        "price": 999.99,
        "in_stock": True,
        "tags": ["electronics", "computers"]
    }
    
    # Construct models from JSON using **dict unpacking
    ground_truth = Product(**ground_truth_json)
    prediction = Product(**prediction_json)
    
    # Compare
    result = ground_truth.compare_with(prediction)
    
    print(f"\nOverall Score: {result['overall_score']:.3f}")
    print(f"\nField Scores:")
    for field, score in result['field_scores'].items():
        print(f"  {field}: {score:.3f}")
    
    print("\n")


def nested_json_schema_example():
    """Example with nested objects and arrays of objects."""
    print("=" * 80)
    print("NESTED JSON SCHEMA EXAMPLE")
    print("=" * 80)
    
    # Define a JSON Schema for an invoice with nested structure
    invoice_schema = {
        "type": "object",
        "title": "Invoice",
        "properties": {
            "invoice_number": {"type": "string"},
            "total": {"type": "number"},
            "customer": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                },
                "required": ["name"]
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "unit_price": {"type": "number"}
                    },
                    "required": ["description", "quantity", "unit_price"]
                }
            }
        },
        "required": ["invoice_number", "total", "customer", "items"]
    }
    
    # Create the model class
    Invoice = StructuredModel.from_json_schema(invoice_schema)
    
    # Create instances from JSON dictionaries (typical usage)
    ground_truth_json = {
        "invoice_number": "INV-001",
        "total": 150.00,
        "customer": {"name": "John Doe", "email": "john@example.com"},
        "items": [
            {"description": "Widget A", "quantity": 2, "unit_price": 50.00},
            {"description": "Widget B", "quantity": 1, "unit_price": 50.00}
        ]
    }
    
    prediction_json = {
        "invoice_number": "INV-001",
        "total": 150.00,
        "customer": {"name": "John Doe", "email": "john@example.com"},
        "items": [
            {"description": "Widget A", "quantity": 2, "unit_price": 50.00},
            {"description": "Widget C", "quantity": 1, "unit_price": 50.00}  # Different item
        ]
    }
    
    # Construct models from JSON using **dict unpacking
    ground_truth = Invoice(**ground_truth_json)
    prediction = Invoice(**prediction_json)
    
    # Compare with detailed metrics
    result = ground_truth.compare_with(
        prediction,
        include_confusion_matrix=True
    )
    
    print(f"\nOverall Score: {result['overall_score']:.3f}")
    print(f"\nCustomer Score: {result['field_scores']['customer']:.3f}")
    print(f"Items Score: {result['field_scores']['items']:.3f}")
    
    # Show confusion matrix summary
    if 'confusion_matrix' in result:
        cm = result['confusion_matrix']
        print(f"\nOverall Confusion Matrix:")
        print(f"  True Positives: {cm.get('true_positives', cm.get('tp', 0))}")
        print(f"  False Positives: {cm.get('false_positives', cm.get('fp', 0))}")
        print(f"  False Negatives: {cm.get('false_negatives', cm.get('fn', 0))}")
    
    print("\n")


def custom_extensions_example():
    """Example using x-stickler extensions for custom comparison behavior."""
    print("=" * 80)
    print("CUSTOM EXTENSIONS EXAMPLE")
    print("=" * 80)
    
    # Define a schema with custom comparison behavior
    document_schema = {
        "type": "object",
        "title": "Document",
        "properties": {
            "title": {
                "type": "string",
                "x-stickler-comparator": "fuzzy",  # Use fuzzy string matching
                "x-stickler-threshold": 0.8  # Require 80% similarity
            },
            "content": {
                "type": "string",
                "x-stickler-comparator": "fuzzy"
            },
            "priority": {
                "type": "integer",
                "x-stickler-weight": 2.0  # Double weight for priority
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["title", "content", "priority"]
    }
    
    # Create the model
    Document = StructuredModel.from_json_schema(document_schema)
    
    # Create instances from JSON dictionaries with slight differences
    ground_truth_json = {
        "title": "Important Report",
        "content": "This is the quarterly financial report.",
        "priority": 1,
        "tags": ["finance", "quarterly"]
    }
    
    prediction_json = {
        "title": "Important Reprot",  # Typo, but fuzzy match should catch it
        "content": "This is the quarterly financial report.",
        "priority": 1,
        "tags": ["finance", "quarterly"]
    }
    
    # Construct models from JSON using **dict unpacking
    ground_truth = Document(**ground_truth_json)
    prediction = Document(**prediction_json)
    
    # Compare
    result = ground_truth.compare_with(prediction)
    
    print(f"\nOverall Score: {result['overall_score']:.3f}")
    print(f"\nField Scores:")
    for field, score in result['field_scores'].items():
        print(f"  {field}: {score:.3f}")
    
    print("\nNote: The 'title' field uses fuzzy matching, so the typo")
    print("'Reprot' still scores highly against 'Report'")
    
    print("\n")


def real_world_api_schema_example():
    """Example using a realistic API response schema."""
    print("=" * 80)
    print("REAL-WORLD API SCHEMA EXAMPLE")
    print("=" * 80)
    
    # Define a schema for a typical API response
    api_response_schema = {
        "type": "object",
        "title": "APIResponse",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "username": {"type": "string"},
                    "email": {"type": "string"},
                    "profile": {
                        "type": "object",
                        "properties": {
                            "first_name": {"type": "string"},
                            "last_name": {"type": "string"},
                            "age": {"type": "integer"}
                        }
                    },
                    "permissions": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "request_id": {"type": "string"}
                }
            }
        },
        "required": ["status", "data"]
    }
    
    # Create the model
    APIResponse = StructuredModel.from_json_schema(api_response_schema)
    
    # Create instances from JSON dictionaries (typical API response usage)
    ground_truth_json = {
        "status": "success",
        "data": {
            "user_id": "12345",
            "username": "johndoe",
            "email": "john@example.com",
            "profile": {
                "first_name": "John",
                "last_name": "Doe",
                "age": 30
            },
            "permissions": ["read", "write", "admin"]
        },
        "metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "request_id": "req-abc-123"
        }
    }
    
    prediction_json = {
        "status": "success",
        "data": {
            "user_id": "12345",
            "username": "johndoe",
            "email": "john@example.com",
            "profile": {
                "first_name": "John",
                "last_name": "Doe",
                "age": 31  # Different age
            },
            "permissions": ["read", "write"]  # Missing admin permission
        },
        "metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "request_id": "req-abc-123"
        }
    }
    
    # Construct models from JSON using **dict unpacking
    ground_truth = APIResponse(**ground_truth_json)
    prediction = APIResponse(**prediction_json)
    
    # Compare with full metrics
    result = ground_truth.compare_with(
        prediction,
        include_confusion_matrix=True
    )
    
    print(f"\nOverall Score: {result['overall_score']:.3f}")
    print(f"\nTop-level Field Scores:")
    for field in ['status', 'data', 'metadata']:
        if field in result['field_scores']:
            print(f"  {field}: {result['field_scores'][field]:.3f}")
    
    print(f"\nNote: The 'data' field score reflects differences in nested")
    print(f"profile.age and missing permissions array item.")
    
    print("\n")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("JSON SCHEMA MODEL CONSTRUCTION EXAMPLES")
    print("*" * 80)
    print("\n")
    
    basic_json_schema_example()
    nested_json_schema_example()
    custom_extensions_example()
    real_world_api_schema_example()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key Takeaways:
1. Use StructuredModel.from_json_schema() to create models from JSON Schema
2. Supports nested objects, arrays, and all JSON Schema primitive types
3. Use x-stickler-* extensions for custom comparison behavior:
   - x-stickler-comparator: Choose comparison algorithm (fuzzy, exact, etc.)
   - x-stickler-threshold: Set matching threshold (0.0 to 1.0)
   - x-stickler-weight: Adjust field importance in scoring
4. Full compatibility with compare_with() for evaluation and metrics
5. Works seamlessly with existing StructuredModel features

For more information, see the README documentation.
    """)


if __name__ == "__main__":
    main()
