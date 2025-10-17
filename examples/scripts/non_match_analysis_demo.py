"""
Non-Match Analysis for Error Debugging

This example shows how to use the non-match reporting feature to debug
evaluation errors and improve model performance. Learn how to identify
specific issues in your structured data comparisons.

Usage:
    python non_match_analysis_demo.py
"""

from typing import List
from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator
import json


# Define our data structures
class Product(StructuredModel):
    """Product model for demonstration."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=2.0
    )

    price: float = ComparableField(
        threshold=0.9,  # Strict threshold for price matching
        weight=1.5,
    )

    category: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )


class Order(StructuredModel):
    """Order model containing a list of products."""

    order_id: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=2.0
    )

    customer: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )

    products: List[Product] = ComparableField(
        weight=3.0  # Most important field
    )


def create_sample_data():
    """Create sample ground truth and prediction data."""
    print("📋 Creating Sample Data")
    print("=" * 50)

    # Ground Truth Order
    gt_products = [
        Product(name="Wireless Mouse", price=25.99, category="Electronics"),
        Product(name="USB Cable", price=12.99, category="Electronics"),
        Product(name="Notebook", price=8.50, category="Office Supplies"),
        Product(name="Coffee Mug", price=15.99, category="Kitchenware"),
    ]

    gt_order = Order(
        order_id="ORD-2024-001", customer="John Smith", products=gt_products
    )

    print("Ground Truth Order:")
    print(f"  Order ID: {gt_order.order_id}")
    print(f"  Customer: {gt_order.customer}")
    print("  Products:")
    for i, product in enumerate(gt_order.products):
        print(f"    {i}: {product.name} - ${product.price} - {product.category}")

    # Prediction with various types of mismatches
    pred_products = [
        Product(
            name="Wireless Mouse", price=25.99, category="Electronics"
        ),  # Perfect match
        Product(
            name="USB Cord", price=12.99, category="Electronics"
        ),  # Name mismatch (FD)
        Product(
            name="Legal Pad", price=9.99, category="Office Supplies"
        ),  # Different product (FD)
        Product(
            name="Water Bottle", price=18.50, category="Drinkware"
        ),  # Extra product (FA)
    ]

    pred_order = Order(
        order_id="ORD-2024-001", customer="John Smith", products=pred_products
    )

    print("\nPredicted Order:")
    print(f"  Order ID: {pred_order.order_id}")
    print(f"  Customer: {pred_order.customer}")
    print("  Products:")
    for i, product in enumerate(pred_order.products):
        print(f"    {i}: {product.name} - ${product.price} - {product.category}")

    return gt_order, pred_order


def demonstrate_basic_evaluation(gt_order, pred_order):
    """Show basic evaluation without non-match documentation."""
    print("\n🔍 Basic Evaluation (No Non-Match Documentation)")
    print("=" * 60)

    evaluator = StructuredModelEvaluator(document_non_matches=False)
    result = evaluator.evaluate(gt_order, pred_order)

    print("Overall Scores:")
    print(f"  Precision: {result['overall']['precision']:.3f}")
    print(f"  Recall:    {result['overall']['recall']:.3f}")
    print(f"  F1 Score:  {result['overall']['f1']:.3f}")
    print(f"  ANLS:      {result['overall']['anls_score']:.3f}")

    print("\nField Scores:")
    for field, metrics in result["fields"].items():
        if field == "products":
            overall_score = metrics.get("overall", {}).get("anls_score", "N/A")
            print(f"  {field:12}: {overall_score:.3f} (overall)")
        else:
            anls_score = metrics.get("anls_score", "N/A")
            print(f"  {field:12}: {anls_score:.3f}")


def demonstrate_enhanced_non_matches(gt_order, pred_order):
    """Show enhanced non-match documentation with object-level granularity."""
    print("\n🔍 Enhanced Non-Match Analysis")
    print("=" * 50)

    evaluator = StructuredModelEvaluator(document_non_matches=True)
    result = evaluator.evaluate(gt_order, pred_order)

    # Show non-matches
    non_matches = result.get("non_matches", [])

    if non_matches:
        print(f"\n📊 Found {len(non_matches)} non-matches:")
        print("-" * 50)

        for i, nm in enumerate(non_matches, 1):
            print(f"\nNon-Match #{i}:")
            print(f"  Field Path: {nm['field_path']}")
            print(f"  Type: {nm['non_match_type']}")

            if "similarity_score" in nm:
                print(f"  Similarity: {nm['similarity_score']:.3f}")

            if "details" in nm and "reason" in nm["details"]:
                print(f"  Reason: {nm['details']['reason']}")

            # Show object details for actionable debugging
            if isinstance(nm.get("ground_truth_value"), dict):
                gt_obj = nm["ground_truth_value"]
                print(
                    f"  GT Object: {gt_obj.get('name', 'N/A')} - ${gt_obj.get('price', 'N/A')} - {gt_obj.get('category', 'N/A')}"
                )

            if isinstance(nm.get("prediction_value"), dict):
                pred_obj = nm["prediction_value"]
                print(
                    f"  Pred Object: {pred_obj.get('name', 'N/A')} - ${pred_obj.get('price', 'N/A')} - {pred_obj.get('category', 'N/A')}"
                )
    else:
        print("\n✅ No non-matches found")


def demonstrate_compare_with_method(gt_order, pred_order):
    """Show how to use the compare_with method directly for non-match analysis."""
    print("\n🔍 Using StructuredModel.compare_with() Method")
    print("=" * 55)

    # Use compare_with method directly
    result = gt_order.compare_with(pred_order, document_non_matches=True)

    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"All Fields Matched: {result['all_fields_matched']}")

    # Show enhanced non-matches in JSON format
    non_matches = result.get("non_matches", [])

    if non_matches:
        print("\n📋 Non-Matches in JSON Format:")
        print(
            json.dumps(
                {
                    "total_non_matches": len(non_matches),
                    "non_matches": non_matches[:2],  # Show first 2 for brevity
                },
                indent=2,
            )
        )


def analyze_non_matches_for_debugging(non_matches):
    """Show how to analyze non-matches for practical debugging."""
    print("\n🔧 Debugging Analysis")
    print("=" * 50)

    if not non_matches:
        print("✅ No issues found - all objects matched successfully!")
        return

    print("📋 Issue Summary:")
    print(f"   Total non-matches: {len(non_matches)}")

    # Group by type for analysis
    false_discoveries = [
        nm
        for nm in non_matches
        if "false_discovery" in str(nm["non_match_type"]).lower()
    ]
    false_alarms = [
        nm for nm in non_matches if "false_alarm" in str(nm["non_match_type"]).lower()
    ]
    false_negatives = [
        nm
        for nm in non_matches
        if "false_negative" in str(nm["non_match_type"]).lower()
    ]

    if false_discoveries:
        print(
            f"   False Discoveries: {len(false_discoveries)} (matched objects below threshold)"
        )
    if false_alarms:
        print(f"   False Alarms: {len(false_alarms)} (extra predicted objects)")
    if false_negatives:
        print(f"   False Negatives: {len(false_negatives)} (missing predicted objects)")

    print("\n🎯 Actionable Insights:")

    # Analyze False Discoveries (most common debugging case)
    if false_discoveries:
        print("\n   False Discoveries - Objects that matched but below threshold:")
        for nm in false_discoveries:
            field_path = nm["field_path"]
            similarity = nm.get("similarity_score", 0)
            print(
                f"   • {field_path}: similarity {similarity:.3f} - check field values for OCR/extraction errors"
            )

            # Show specific field differences if available
            gt_obj = nm.get("ground_truth_value", {})
            pred_obj = nm.get("prediction_value", {})
            if isinstance(gt_obj, dict) and isinstance(pred_obj, dict):
                for field in gt_obj:
                    if gt_obj.get(field) != pred_obj.get(field):
                        print(
                            f"     - {field}: '{gt_obj.get(field)}' vs '{pred_obj.get(field)}'"
                        )

    # Analyze False Alarms (extra predictions)
    if false_alarms:
        print("\n   False Alarms - Extra objects in prediction:")
        for nm in false_alarms:
            field_path = nm["field_path"]
            pred_obj = nm.get("prediction_value", {})
            if isinstance(pred_obj, dict):
                name = pred_obj.get("name", "Unknown")
                print(
                    f"   • {field_path}: '{name}' - check if this should be filtered out"
                )

    # Analyze False Negatives (missing predictions)
    if false_negatives:
        print("\n   False Negatives - Missing objects in prediction:")
        for nm in false_negatives:
            field_path = nm["field_path"]
            gt_obj = nm.get("ground_truth_value", {})
            if isinstance(gt_obj, dict):
                name = gt_obj.get("name", "Unknown")
                print(
                    f"   • {field_path}: '{name}' - check if extraction missed this object"
                )


def main():
    """Run the non-match analysis demo."""
    print("🚀 Non-Match Analysis for Error Debugging")
    print("=" * 60)
    print("Learn how to use non-match reporting to debug evaluation errors")
    print("=" * 60)

    # Create sample data with various mismatch types
    gt_order, pred_order = create_sample_data()

    # Show basic evaluation
    demonstrate_basic_evaluation(gt_order, pred_order)

    # Show enhanced non-match analysis
    demonstrate_enhanced_non_matches(gt_order, pred_order)

    # Show compare_with method usage
    demonstrate_compare_with_method(gt_order, pred_order)

    # Analyze non-matches for practical debugging
    evaluator = StructuredModelEvaluator(document_non_matches=True)
    result = evaluator.evaluate(gt_order, pred_order)
    non_matches = result.get("non_matches", [])
    analyze_non_matches_for_debugging(non_matches)

    print("\n🎯 Key Takeaways")
    print("=" * 50)
    print("✅ Use document_non_matches=True to get detailed error information")
    print("✅ Look for object-level field paths like products[0], products[1]")
    print("✅ Check similarity scores to prioritize which issues to fix first")
    print("✅ False discoveries often indicate OCR/extraction quality issues")
    print("✅ False alarms may indicate filtering or validation problems")
    print("✅ False negatives suggest missing extraction or detection")

    print("\n📚 Next Steps:")
    print("1. Apply this analysis to your own evaluation data")
    print("2. Focus on the highest-impact non-matches (lowest similarity scores)")
    print("3. Use field-level differences to identify systematic issues")
    print("4. Iterate on your extraction/detection pipeline based on insights")


if __name__ == "__main__":
    main()
