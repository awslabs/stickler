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

from stickler.structured_object_evaluator.models.structured_model import StructuredModel


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
    print("\nField Scores:")
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
        cm = result['confusion_matrix']['overall']
        print("\nOverall Confusion Matrix:")
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
    print("\nField Scores:")
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
    print("\nTop-level Field Scores:")
    for field in ['status', 'data', 'metadata']:
        if field in result['field_scores']:
            print(f"  {field}: {result['field_scores'][field]:.3f}")
    
    print("\nNote: The 'data' field score reflects differences in nested")
    print("profile.age and missing permissions array item.")
    
    print("\n")


def advanced_extensions_and_refs_example():
    """Example showcasing comprehensive x-stickler extensions with $ref usage."""
    print("=" * 80)
    print("ADVANCED EXTENSIONS AND $REF EXAMPLE")
    print("=" * 80)
    
    # Define a comprehensive schema with $ref and all x-aws-stickler extensions
    ecommerce_schema = {
        "type": "object",
        "x-aws-stickler-model-name": "ECommerceOrder",
        "x-aws-stickler-match-threshold": 0.8,
        "definitions": {
            "Address": {
                "type": "object",
                "properties": {
                    "street": {
                        "type": "string",
                        "x-aws-stickler-comparator": "LevenshteinComparator",
                        "x-aws-stickler-threshold": 0.85,
                        "x-aws-stickler-weight": 1.2
                    },
                    "city": {
                        "type": "string",
                        "x-aws-stickler-comparator": "LevenshteinComparator",
                        "x-aws-stickler-threshold": 0.9,
                        "x-aws-stickler-weight": 1.5
                    }
                },
                "required": ["street", "city"]
            },
            "Customer": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "x-aws-stickler-comparator": "LevenshteinComparator",
                        "x-aws-stickler-threshold": 0.9,
                        "x-aws-stickler-weight": 2.0,
                        "x-aws-stickler-aggregate": True
                    },
                    "address": {"$ref": "#/definitions/Address"}
                },
                "required": ["name", "address"]
            }
        },
        "properties": {
            "order_id": {
                "type": "string",
                "x-aws-stickler-comparator": "ExactComparator",
                "x-aws-stickler-weight": 3.0,
                "x-aws-stickler-aggregate": True,
                "x-aws-stickler-clip-under-threshold": True
            },
            "customer": {"$ref": "#/definitions/Customer"},
            "shipping_address": {"$ref": "#/definitions/Address"},
            "total_amount": {
                "type": "number",
                "x-aws-stickler-comparator": "NumericComparator",
                "x-aws-stickler-threshold": 0.95,
                "x-aws-stickler-weight": 2.5,
                "x-aws-stickler-aggregate": True
            },
            "items": {
                "type": "array",
                "x-aws-stickler-weight": 2.0,
                "items": {
                    "type": "object",
                    "properties": {
                        "product_name": {
                            "type": "string",
                            "x-aws-stickler-comparator": "LevenshteinComparator",
                            "x-aws-stickler-threshold": 0.7,
                            "x-aws-stickler-weight": 1.3
                        },
                        "price": {
                            "type": "number",
                            "x-aws-stickler-comparator": "NumericComparator",
                            "x-aws-stickler-threshold": 0.98,
                            "x-aws-stickler-weight": 1.5
                        }
                    },
                    "required": ["product_name", "price"]
                }
            }
        },
        "required": ["order_id", "customer", "shipping_address", "total_amount", "items"]
    }
    
    # Create the model class from the schema
    ECommerceOrder = StructuredModel.from_json_schema(ecommerce_schema)
    
    print(f"Created model class: {ECommerceOrder.__name__}")
    print(f"Model match threshold: {ECommerceOrder.match_threshold}")
    
    # Create test data demonstrating the impact of different x-stickler configurations
    ground_truth_json = {
        "order_id": "ORD-2024-001",
        "customer": {
            "name": "John Smith",
            "address": {
                "street": "123 Main Street",
                "city": "Springfield"
            }
        },
        "shipping_address": {
            "street": "456 Oak Avenue",
            "city": "Springfield"
        },
        "total_amount": 299.99,
        "items": [
            {"product_name": "Wireless Headphones", "price": 149.99},
            {"product_name": "Phone Case", "price": 24.99},
            {"product_name": "USB Cable", "price": 19.99}
        ]
    }
    
    # Prediction with various differences to showcase x-stickler behavior
    prediction_json = {
        "order_id": "ORD-2024-001",  # Exact match (ExactComparator)
        "customer": {
            "name": "Jon Smith",  # Typo (LevenshteinComparator with 0.9 threshold)
            "address": {
                "street": "123 Main St",  # Abbreviated (LevenshteinComparator 0.85 threshold)
                "city": "Springfield"  # Exact match
            }
        },
        "shipping_address": {
            "street": "456 Oak Ave",  # Abbreviated
            "city": "Springfield"
        },
        "total_amount": 299.95,  # Slight difference (NumericComparator 0.95 threshold)
        "items": [
            {"product_name": "Wireless Headphones", "price": 149.99},  # Perfect match
            {"product_name": "Phone Cover", "price": 24.99},  # Different name
            {"product_name": "USB Cable", "price": 19.95}  # Slight price difference
        ]
    }
    
    # Create model instances
    ground_truth = ECommerceOrder(**ground_truth_json)
    prediction = ECommerceOrder(**prediction_json)
    
    # Compare with detailed metrics
    result = ground_truth.compare_with(
        prediction,
        include_confusion_matrix=True
    )
    
    print(f"\nOverall Score: {result['overall_score']:.3f}")
    print(f"All Fields Matched: {result['all_fields_matched']}")
    
    print(f"\nTop-level Field Scores:")
    for field in ['order_id', 'customer', 'shipping_address', 'total_amount', 'items']:
        if field in result['field_scores']:
            print(f"  {field}: {result['field_scores'][field]:.3f}")
    
    # Show confusion matrix summary
    if 'confusion_matrix' in result:
        cm = result['confusion_matrix']['overall']
        print("\nOverall Confusion Matrix:")
        print(f"  True Positives: {cm.get('true_positives', cm.get('tp', 0))}")
        print(f"  False Positives: {cm.get('false_positives', cm.get('fp', 0))}")
        print(f"  False Negatives: {cm.get('false_negatives', cm.get('fn', 0))}")


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
    advanced_extensions_and_refs_example()
    
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
