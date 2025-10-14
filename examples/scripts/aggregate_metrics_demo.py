#!/usr/bin/env python3
"""
Aggregate Metrics Demo - Universal Aggregate Field Feature

This script demonstrates the new universal aggregate field feature that automatically
provides field-level granularity for confusion matrix analysis at every node in the
comparison result tree.

Key Features:
- Every node automatically includes an 'aggregate' field
- Aggregate fields sum all primitive field metrics below that node
- No configuration required - works out of the box
- Provides hierarchical confusion matrix analysis
"""

from typing import Optional, List
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.exact import ExactComparator
from stickler.comparators.numeric import NumericComparator
from stickler.comparators.levenshtein import LevenshteinComparator


# Define nested data models
class Contact(StructuredModel):
    phone: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    email: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class Address(StructuredModel):
    street: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    city: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    zip_code: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class Pet(StructuredModel):
    match_threshold = 0.8  # Threshold for object-level matching

    pet_id: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.85, weight=1.0
    )
    species: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.85, weight=1.0
    )
    age: Optional[int] = ComparableField(
        default=None, comparator=NumericComparator(), threshold=0.9, weight=1.0
    )


class Customer(StructuredModel):
    customer_id: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    contact: Contact = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    address: Address = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    pets: List[Pet] = ComparableField(weight=1.0)


def print_aggregate_analysis(result, title):
    """Print detailed aggregate analysis."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    cm = result["confusion_matrix"]

    # Top-level aggregate
    print("\n🔍 TOP-LEVEL AGGREGATE (All Primitive Fields):")
    top_agg = cm["aggregate"]
    print(
        f"   TP: {top_agg['tp']}, FD: {top_agg['fd']}, FP: {top_agg['fp']}, FA: {top_agg['fa']}, FN: {top_agg['fn']}"
    )
    derived = top_agg.get("derived", {}) or {}
    print(f"   Precision: {derived.get('cm_precision', 0):.3f}")
    print(f"   Recall: {derived.get('cm_recall', 0):.3f}")
    print(f"   F1: {derived.get('cm_f1', 0):.3f}")

    # Field-level aggregates
    print("\n📊 FIELD-LEVEL AGGREGATES:")

    for field_name, field_data in cm["fields"].items():
        if "aggregate" in field_data:
            agg = field_data["aggregate"]
            print(f"\n   {field_name.upper()}:")
            print(
                f"     TP: {agg['tp']}, FD: {agg['fd']}, FP: {agg['fp']}, FA: {agg['fa']}, FN: {agg['fn']}"
            )
            agg_derived = agg.get("derived", {}) or {}
            print(f"     F1: {agg_derived.get('cm_f1', 0):.3f}")

            # Show nested field aggregates if they exist
            if "fields" in field_data:
                for nested_name, nested_data in field_data["fields"].items():
                    if "aggregate" in nested_data:
                        nested_agg = nested_data["aggregate"]
                        print(
                            f"     └─ {nested_name}: TP={nested_agg['tp']}, FD={nested_agg['fd']}, FP={nested_agg['fp']}"
                        )


def demonstrate_aggregate_feature():
    """Demonstrate the universal aggregate field feature."""

    print("🚀 UNIVERSAL AGGREGATE FIELD FEATURE DEMO")
    print("=" * 60)
    print("This demo shows how aggregate fields automatically provide")
    print("field-level granularity for confusion matrix analysis.")

    # Ground truth data
    gt_data = {
        "customer_id": 12345,
        "name": "John Smith",
        "contact": {"phone": "555-123-4567", "email": "john@example.com"},
        "address": {"street": "123 Main St", "city": "Seattle", "zip_code": "98101"},
        "pets": [
            {"pet_id": 1001, "name": "Buddy", "species": "Dog", "age": 5},
            {"pet_id": 1002, "name": "Whiskers", "species": "Cat", "age": 3},
        ],
    }

    # Prediction with various types of errors
    pred_data = {
        "customer_id": 12345,  # ✅ TP
        "name": "Jon Smith",  # ✅ TP (close enough with Levenshtein)
        "contact": {
            "phone": "555-999-8888",  # ❌ FD (wrong phone)
            "email": "john@example.com",  # ✅ TP
        },
        "address": {
            "street": "456 Oak Ave",  # ❌ FD (wrong street)
            "city": "Seattle",  # ✅ TP
            "zip_code": "98102",  # ❌ FD (wrong zip)
        },
        "pets": [
            {
                "pet_id": 1001,  # ✅ TP
                "name": "Buddy",  # ✅ TP
                "species": "Dog",  # ✅ TP
                "age": 6,  # ❌ FD (wrong age)
            },
            {
                "pet_id": 1002,  # ✅ TP
                "name": "Whiskers",  # ✅ TP
                "species": "Cat",  # ✅ TP
                "age": 4,  # ❌ FD (wrong age)
            },
        ],
    }

    # Create model instances
    gt_customer = Customer(**gt_data)
    pred_customer = Customer(**pred_data)

    # Compare with confusion matrix and aggregate metrics
    result = gt_customer.compare_with(pred_customer, include_confusion_matrix=True)

    # Print detailed aggregate analysis
    print_aggregate_analysis(result, "AGGREGATE METRICS ANALYSIS")

    # Show the structure
    print("\n📋 STRUCTURE OVERVIEW:")
    cm = result["confusion_matrix"]
    print(f"   Top-level keys: {list(cm.keys())}")
    print("   Each field has: overall, fields, aggregate")
    print("   Aggregate = sum of all primitive fields below that node")

    # Demonstrate hierarchical access
    print("\n🔗 HIERARCHICAL ACCESS EXAMPLES:")
    print(f"   Total primitive TP across all fields: {cm['aggregate']['tp']}")
    print(f"   Contact-related TP: {cm['fields']['contact']['aggregate']['tp']}")
    print(f"   Address-related TP: {cm['fields']['address']['aggregate']['tp']}")
    print(f"   Pet-related TP: {cm['fields']['pets']['aggregate']['tp']}")

    # Show that no configuration was needed
    print("\n✨ KEY BENEFITS:")
    print("   ✅ No configuration required - works automatically")
    print("   ✅ Every node has aggregate field as sibling of 'overall'")
    print("   ✅ Hierarchical analysis at any level")
    print("   ✅ Backward compatible - existing code unchanged")
    print("   ✅ Includes derived metrics (precision, recall, F1)")


def demonstrate_deprecation_warning():
    """Show the deprecation warning for legacy aggregate parameter."""

    print("\n⚠️  DEPRECATION WARNING DEMO")
    print("=" * 40)
    print("The old aggregate=True parameter is now deprecated:")

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # This will trigger a deprecation warning
        field = ComparableField(aggregate=True, comparator=ExactComparator())

        if w:
            print(f"   Warning: {w[0].message}")
            print(f"   Category: {w[0].category.__name__}")

    print("   ✅ Use the new universal aggregate fields instead!")


if __name__ == "__main__":
    # Run the main demonstration
    demonstrate_aggregate_feature()

    # Show deprecation warning
    demonstrate_deprecation_warning()

    print("\n🎉 DEMO COMPLETE!")
    print("The universal aggregate field feature provides automatic")
    print("field-level granularity for confusion matrix analysis.")
