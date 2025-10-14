#!/usr/bin/env python3
"""
Complete JSON-to-Evaluation Demonstration

This script demonstrates a fully JSON-driven workflow:
1. Load model configuration from JSON
2. Load test data from JSON files
3. Create dynamic StructuredModel classes
4. Perform comparisons and evaluations

No Python object construction - everything comes from JSON!
"""

import json
import tempfile
import os
from pathlib import Path
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


def create_sample_json_files():
    """Create sample JSON configuration and data files."""

    # Create temporary directory for JSON files
    temp_dir = Path(tempfile.mkdtemp())

    # 1. Person Model Configuration
    person_config = {
        "model_name": "Person",
        "match_threshold": 0.8,
        "fields": {
            "name": {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "threshold": 0.8,
                "weight": 2.0,
            },
            "age": {
                "type": "int",
                "comparator": "NumericComparator",
                "threshold": 0.9,
                "weight": 1.0,
            },
            "email": {
                "type": "str",
                "comparator": "ExactComparator",
                "threshold": 1.0,
                "weight": 1.5,
            },
        },
    }

    # 2. Person Test Data
    person_data = {
        "ground_truth": {
            "name": "John Smith",
            "age": 30,
            "email": "john.smith@company.com",
        },
        "prediction": {
            "name": "Jon Smith",  # Slight typo
            "age": 31,  # Off by 1
            "email": "john.smith@company.com",  # Exact match
        },
    }

    # 3. Invoice Model Configuration (Complex Nested)
    invoice_config = {
        "model_name": "Invoice",
        "match_threshold": 0.7,
        "fields": {
            "invoice_id": {
                "type": "str",
                "comparator": "ExactComparator",
                "threshold": 1.0,
                "weight": 3.0,
            },
            "customer": {
                "type": "structured_model",
                "threshold": 0.8,
                "weight": 2.0,
                "fields": {
                    "name": {
                        "type": "str",
                        "comparator": "LevenshteinComparator",
                        "threshold": 0.8,
                        "weight": 1.0,
                    },
                    "address": {
                        "type": "str",
                        "comparator": "FuzzyComparator",
                        "threshold": 0.7,
                        "weight": 0.8,
                    },
                },
            },
            "line_items": {
                "type": "list_structured_model",
                "weight": 2.0,
                "match_threshold": 0.7,
                "fields": {
                    "product": {
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
                    "price": {
                        "type": "float",
                        "comparator": "NumericComparator",
                        "threshold": 0.95,
                        "weight": 1.2,
                    },
                },
            },
            "total": {
                "type": "float",
                "comparator": "NumericComparator",
                "threshold": 0.95,
                "weight": 2.5,
            },
        },
    }

    # 4. Invoice Test Data
    invoice_data = {
        "ground_truth": {
            "invoice_id": "INV-2024-001",
            "customer": {
                "name": "Acme Corporation",
                "address": "123 Business St, Suite 100, New York, NY 10001",
            },
            "line_items": [
                {
                    "product": "Professional Software License",
                    "quantity": 5,
                    "price": 299.99,
                },
                {
                    "product": "Technical Support Package",
                    "quantity": 1,
                    "price": 500.00,
                },
                {"product": "Training Materials", "quantity": 10, "price": 49.99},
            ],
            "total": 2999.85,
        },
        "prediction": {
            "invoice_id": "INV-2024-001",  # Exact match
            "customer": {
                "name": "ACME Corp",  # Slight variation
                "address": "123 Business Street, Ste 100, NYC, NY 10001",  # Address variation
            },
            "line_items": [
                {
                    "product": "Training Materials",  # Reordered
                    "quantity": 10,
                    "price": 49.99,
                },
                {
                    "product": "Pro Software License",  # Slight name variation
                    "quantity": 5,
                    "price": 299.99,
                },
                {
                    "product": "Tech Support Package",  # Slight name variation
                    "quantity": 1,
                    "price": 500.00,
                },
            ],
            "total": 2999.85,  # Exact match
        },
    }

    # 5. Product Catalog Model (List Comparison)
    catalog_config = {
        "model_name": "ProductCatalog",
        "match_threshold": 0.8,
        "fields": {
            "catalog_name": {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "threshold": 0.8,
                "weight": 1.0,
            },
            "products": {
                "type": "list_structured_model",
                "weight": 3.0,
                "match_threshold": 0.75,
                "fields": {
                    "sku": {
                        "type": "str",
                        "comparator": "ExactComparator",
                        "threshold": 1.0,
                        "weight": 2.0,
                    },
                    "name": {
                        "type": "str",
                        "comparator": "FuzzyComparator",
                        "threshold": 0.8,
                        "weight": 1.5,
                    },
                    "category": {
                        "type": "str",
                        "comparator": "ExactComparator",
                        "threshold": 1.0,
                        "weight": 1.0,
                    },
                    "price": {
                        "type": "float",
                        "comparator": "NumericComparator",
                        "comparator_config": {"tolerance": 0.05},
                        "threshold": 0.9,
                        "weight": 1.2,
                    },
                },
            },
        },
    }

    # 6. Product Catalog Test Data
    catalog_data = {
        "ground_truth": {
            "catalog_name": "Electronics Catalog 2024",
            "products": [
                {
                    "sku": "LAPTOP-001",
                    "name": "Professional Laptop 15-inch",
                    "category": "Computers",
                    "price": 1299.99,
                },
                {
                    "sku": "MOUSE-002",
                    "name": "Wireless Optical Mouse",
                    "category": "Accessories",
                    "price": 29.99,
                },
                {
                    "sku": "MONITOR-003",
                    "name": "4K Ultra HD Monitor 27-inch",
                    "category": "Displays",
                    "price": 399.99,
                },
            ],
        },
        "prediction": {
            "catalog_name": "Electronics Catalog 2024",  # Exact match
            "products": [
                {
                    "sku": "MONITOR-003",  # Reordered
                    "name": '4K UHD Monitor 27"',  # Slight variation
                    "category": "Displays",
                    "price": 419.99,  # Price difference within tolerance
                },
                {
                    "sku": "LAPTOP-001",  # Reordered
                    "name": 'Professional Laptop 15"',  # Slight variation
                    "category": "Computers",
                    "price": 1299.99,  # Exact match
                },
                {
                    "sku": "MOUSE-002",  # Reordered
                    "name": "Wireless Mouse",  # Shorter name
                    "category": "Accessories",
                    "price": 31.99,  # Slight price difference
                },
            ],
        },
    }

    # Write all files
    files = {
        "person_config.json": person_config,
        "person_data.json": person_data,
        "invoice_config.json": invoice_config,
        "invoice_data.json": invoice_data,
        "catalog_config.json": catalog_config,
        "catalog_data.json": catalog_data,
    }

    file_paths = {}
    for filename, data in files.items():
        file_path = temp_dir / filename
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        file_paths[filename] = str(file_path)

    return file_paths


def demo_json_to_evaluation_workflow(config_file, data_file, demo_name):
    """Complete JSON-to-evaluation workflow."""

    print("=" * 60)
    print(f"🚀 {demo_name}")
    print("=" * 60)

    # Step 1: Load model configuration from JSON
    print("📋 Step 1: Loading model configuration from JSON...")
    with open(config_file, "r") as f:
        model_config = json.load(f)

    print(f"   ✓ Loaded config for: {model_config['model_name']}")
    print(f"   ✓ Fields: {list(model_config['fields'].keys())}")

    # Step 2: Create dynamic model class
    print("\n🏗️  Step 2: Creating dynamic StructuredModel class...")
    ModelClass = StructuredModel.model_from_json(model_config)
    print(f"   ✓ Created class: {ModelClass.__name__}")

    # Step 3: Load test data from JSON
    print("\n📊 Step 3: Loading test data from JSON...")
    with open(data_file, "r") as f:
        test_data = json.load(f)

    ground_truth_data = test_data["ground_truth"]
    prediction_data = test_data["prediction"]

    print("   ✓ Loaded ground truth data")
    print("   ✓ Loaded prediction data")

    # Step 4: Create model instances from JSON data
    print("\n🔧 Step 4: Creating model instances from JSON data...")
    ground_truth = ModelClass(**ground_truth_data)
    prediction = ModelClass(**prediction_data)

    print("   ✓ Created ground truth instance")
    print("   ✓ Created prediction instance")

    # Step 5: Perform comparison
    print("\n⚖️  Step 5: Performing comparison...")
    result = ground_truth.compare_with(prediction)

    print("   ✓ Comparison completed")
    print(f"   ✓ Overall Score: {result['overall_score']:.3f}")

    # Step 6: Display detailed results
    print("\n📈 Step 6: Detailed Results")
    print("-" * 40)
    print(f"Overall Similarity Score: {result['overall_score']:.3f}")
    print("\nField-by-Field Scores:")

    for field_name, score in result["field_scores"].items():
        print(f"  • {field_name:20} {score:.3f}")

    # Show nested scores if available
    if "nested_scores" in result and result["nested_scores"]:
        print("\nNested Structure Scores:")
        for field_name, nested_result in result["nested_scores"].items():
            if isinstance(nested_result, dict) and "overall_score" in nested_result:
                print(f"  • {field_name:20} {nested_result['overall_score']:.3f}")
                if "field_scores" in nested_result:
                    for sub_field, sub_score in nested_result["field_scores"].items():
                        print(f"    - {sub_field:18} {sub_score:.3f}")

    # Show list matching details if available
    if "list_scores" in result and result["list_scores"]:
        print("\nList Matching Details:")
        for field_name, list_result in result["list_scores"].items():
            if isinstance(list_result, dict):
                matched = list_result.get("matched_pairs", [])
                print(f"  • {field_name}: {len(matched)} pairs matched")

    return result


def demo_complete_json_workflow():
    """Demonstrate complete JSON-driven evaluation workflow."""

    print("🎯 Complete JSON-to-Evaluation Workflow Demo")
    print("=" * 60)
    print("This demo shows how to go from JSON configuration + JSON data")
    print("to complete structured object evaluation - no Python object construction!")
    print()

    # Create sample JSON files
    print("📁 Creating sample JSON files...")
    file_paths = create_sample_json_files()
    print("   ✓ Created configuration and data files")

    try:
        # Demo 1: Simple Person Comparison
        demo_json_to_evaluation_workflow(
            file_paths["person_config.json"],
            file_paths["person_data.json"],
            "Demo 1: Simple Person Comparison",
        )

        # Demo 2: Complex Invoice with Nested Structures
        demo_json_to_evaluation_workflow(
            file_paths["invoice_config.json"],
            file_paths["invoice_data.json"],
            "Demo 2: Complex Invoice with Nested Structures",
        )

        # Demo 3: Product Catalog with List Matching
        demo_json_to_evaluation_workflow(
            file_paths["catalog_config.json"],
            file_paths["catalog_data.json"],
            "Demo 3: Product Catalog with Hungarian List Matching",
        )

        print("=" * 60)
        print("🎉 All JSON-to-Evaluation Demos Completed Successfully!")
        print("=" * 60)
        print("\n✨ Key Benefits Demonstrated:")
        print("  ✓ Zero Python object construction required")
        print("  ✓ Complete configuration-driven workflow")
        print("  ✓ Complex nested structure support")
        print("  ✓ Hungarian algorithm for list matching")
        print("  ✓ Detailed scoring and analysis")
        print("  ✓ Production-ready JSON-based evaluation")

        print("\n📂 Sample JSON files created in:")
        for filename, path in file_paths.items():
            print(f"   • {filename}")
        print("\n💡 You can modify these JSON files to test different scenarios!")

    finally:
        # Clean up temporary files
        print("\n🧹 Cleaning up temporary files...")
        for file_path in file_paths.values():
            try:
                os.unlink(file_path)
            except Exception:
                pass
        # Remove temp directory
        try:
            os.rmdir(os.path.dirname(list(file_paths.values())[0]))
        except Exception:
            pass
        print("   ✓ Cleanup completed")


def demo_json_batch_evaluation():
    """Demonstrate batch evaluation from multiple JSON files."""

    print("=" * 60)
    print("🔄 Bonus Demo: Batch JSON Evaluation")
    print("=" * 60)

    # Create batch test data
    temp_dir = Path(tempfile.mkdtemp())

    # Simple product model
    product_config = {
        "model_name": "Product",
        "fields": {
            "name": {
                "type": "str",
                "comparator": "FuzzyComparator",
                "threshold": 0.8,
                "weight": 1.0,
            },
            "price": {
                "type": "float",
                "comparator": "NumericComparator",
                "threshold": 0.9,
                "weight": 1.0,
            },
        },
    }

    # Multiple test cases
    test_cases = [
        {
            "ground_truth": {"name": "Wireless Headphones", "price": 99.99},
            "prediction": {"name": "Wireless Headphone", "price": 99.99},
        },
        {
            "ground_truth": {"name": "Gaming Mouse", "price": 49.99},
            "prediction": {"name": "Gaming Mouse Pro", "price": 54.99},
        },
        {
            "ground_truth": {"name": "Mechanical Keyboard", "price": 129.99},
            "prediction": {"name": "Mech Keyboard", "price": 125.00},
        },
    ]

    config_file = None
    try:
        # Save config
        config_file = temp_dir / "product_config.json"
        with open(config_file, "w") as f:
            json.dump(product_config, f, indent=2)

        # Create model
        Product = StructuredModel.model_from_json(product_config)

        print("📊 Evaluating multiple JSON test cases...")
        print("-" * 40)

        total_score = 0
        for i, test_case in enumerate(test_cases, 1):
            # Create instances from JSON data
            gt = Product(**test_case["ground_truth"])
            pred = Product(**test_case["prediction"])

            # Compare
            result = gt.compare_with(pred)
            score = result["overall_score"]
            total_score += score

            print(f"Test Case {i}: {score:.3f}")
            print(f"  GT:   {test_case['ground_truth']}")
            print(f"  Pred: {test_case['prediction']}")
            print()

        avg_score = total_score / len(test_cases)
        print("📈 Batch Results:")
        print(f"   Average Score: {avg_score:.3f}")
        print(f"   Total Cases: {len(test_cases)}")

    finally:
        # Cleanup
        try:
            if config_file is not None:
                os.unlink(config_file)
            os.rmdir(temp_dir)
        except Exception:
            pass


def main():
    """Run the complete JSON-to-evaluation demonstration."""

    print("🎯 JSON-to-Evaluation Complete Workflow Demo")
    print("=" * 60)
    print("Demonstrating end-to-end JSON-driven structured object evaluation")
    print("No Python object construction - everything from JSON!")
    print()

    # Main workflow demo
    demo_complete_json_workflow()

    # Bonus batch demo
    demo_json_batch_evaluation()

    print("=" * 60)
    print("🏁 Complete JSON-to-Evaluation Demo Finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()
