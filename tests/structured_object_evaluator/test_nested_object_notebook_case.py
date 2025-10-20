"""

Test case based on testing_nested_object_results.ipynb notebook.
This test is designed to have nice printing and allow for easy debugging
of nested object evaluation results.
"""

import json
import pytest
from typing import List, Optional

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.evaluator import StructuredModelEvaluator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.exact import ExactComparator


class Attribute(StructuredModel):
    """Attribute model for testing nested objects."""

    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.99, weight=1.0
    )


class Product(StructuredModel):
    """A product model with an ID, name, price, and optional attributes."""

    product_id: str = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=3.0,  # High weight - this is a key identifier
    )

    name: str = ComparableField(
        comparator=LevenshteinComparator(),
        threshold=0.7,  # Allow some variation in names
        weight=2.0,
    )

    price: float = ComparableField(
        threshold=0.9,  # Price should be very close
        weight=1.0,
    )

    # Override the default match threshold for Products
    match_threshold = 0.75
    attributes: Optional[List[Attribute]] = ComparableField()


class Order(StructuredModel):
    """Order containing customer information and a list of products."""

    order_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )

    customer_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )

    # This uses Hungarian matching to find best matches between product lists
    products: List[Product] = ComparableField(
        weight=3.0  # But it's very important to the overall match
    )


def create_ground_truth_order() -> Order:
    """Create a sample ground truth order."""
    return Order(
        order_id="ORD-12345",
        customer_name="Jane Smith",
        products=[
            Product(
                product_id="PROD-001",
                name="Laptop Computer",
                price=999.9,
                attributes=[Attribute(name="one two")],
            ),
            Product(
                product_id="PROD-002",
                name="Wireless Mouse",
                price=29.99,
                attributes=[Attribute(name="one two")],
            ),
            Product(
                product_id="PROD-003",
                name="HDMI Cable",
                price=14.99,
                attributes=[Attribute(name="one two")],
            ),
        ],
    )


def create_prediction_order(case: str) -> Order:
    """Create a prediction order with various differences based on the case."""
    if case == "good_match":
        # Good match but with some minor differences
        return Order(
            order_id="ORD-12345",
            customer_name="J. Smith",  # Abbreviated name
            products=[
                # Different order and minor differences
                Product(
                    product_id="PROD-002", name="Wireless Mouse", price=29.95
                ),  # Slight price difference
                Product(
                    product_id="PROD-003", name="HDMI Cable 6ft", price=14.99
                ),  # Name has extra detail
                Product(
                    product_id="PROD-001", name="Laptop", price=999.99
                ),  # Shortened name
            ],
        )
    elif case == "missing_item":
        # Good match but missing one product
        return Order(
            order_id="ORD-12345",
            customer_name="Jane Smith",
            products=[
                Product(
                    product_id="PROD-001",
                    name="Laptop",
                    price=999.99,
                    attributes=[Attribute(name="one two")],
                ),
                Product(
                    product_id="PROD-003",
                    name="HDMI Cable",
                    price=14.99,
                    attributes=[Attribute(name="one two")],
                ),
                # PROD-002 is missing
            ],
        )
    elif case == "extra_item":
        # Good match but with an extra product
        return Order(
            order_id="ORD-12345",
            customer_name="Jane Smith",
            products=[
                Product(product_id="PROD-001", name="Laptop Computer", price=999.99),
                Product(product_id="PROD-002", name="Wireless Mouse", price=29.99),
                Product(product_id="PROD-003", name="HDMI Cable", price=14.99),
                Product(
                    product_id="PROD-004", name="USB Drive", price=19.99
                ),  # Extra item
            ],
        )
    else:
        # Poor match with different order ID and mostly different products
        return Order(
            order_id="ORD-54321",  # Different order ID
            customer_name="Jane Smith",
            products=[
                Product(
                    product_id="PROD-001", name="Laptop Computer", price=999.99
                ),  # Same
                Product(
                    product_id="PROD-005", name="External Hard Drive", price=89.99
                ),  # Different
                Product(
                    product_id="PROD-006", name="Monitor", price=199.99
                ),  # Different
            ],
        )


def print_detailed_results(result: dict, case_name: str):
    """Print detailed results in a nice, readable format."""
    print("=" * 60)
    print(f"DETAILED RESULTS FOR: {case_name}")
    print("=" * 60)

    # Handle both output formats
    overall_score = result.get("overall_score", "N/A")
    if overall_score == "N/A" and "overall" in result:
        overall_score = result["overall"].get("anls_score", "N/A")

    match_status = result.get("match", "N/A")
    if match_status == "N/A" and "overall" in result:
        # Try to determine match status from overall score
        if isinstance(overall_score, (int, float)):
            match_status = "MATCH" if overall_score >= 0.8 else "NO MATCH"

    print(f"\nOVERALL SCORE: {overall_score}")
    print(f"MATCH STATUS: {match_status}")

    # Handle field scores for both formats
    if "field_scores" in result:
        print("\nFIELD SCORES (Method 2 format):")
        print("-" * 40)
        for field, score in result["field_scores"].items():
            print(f"  {field:20}: {score}")
    elif "fields" in result:
        print("\nFIELD SCORES (Method 1 format):")
        print("-" * 40)
        for field, data in result["fields"].items():
            if isinstance(data, dict) and "anls_score" in data:
                print(f"  {field:20}: {data['anls_score']}")

    # Handle confusion matrix
    if "confusion_matrix" in result:
        cm = result["confusion_matrix"]
        if "overall" in cm:
            cm_overall = cm["overall"]
            print("\nCONFUSION MATRIX (Overall):")
            print("-" * 40)
            print(f"  True Positives : {cm_overall.get('tp', 'N/A')}")
            print(f"  False Positives: {cm_overall.get('fp', 'N/A')}")
            print(f"  False Negatives: {cm_overall.get('fn', 'N/A')}")
            print(f"  True Negatives : {cm_overall.get('tn', 'N/A')}")

            if "derived" in cm_overall:
                derived = cm_overall["derived"]
                print(f"  Precision      : {derived.get('cm_precision', 'N/A'):.4f}")
                print(f"  Recall         : {derived.get('cm_recall', 'N/A'):.4f}")
                print(f"  F1 Score       : {derived.get('cm_f1', 'N/A'):.4f}")
        else:
            print("\nCONFUSION MATRIX:")
            print("-" * 40)
            print(f"  True Positives : {cm.get('tp', 'N/A')}")
            print(f"  False Positives: {cm.get('fp', 'N/A')}")
            print(f"  False Negatives: {cm.get('fn', 'N/A')}")
            print(f"  True Negatives : {cm.get('tn', 'N/A')}")

    print("\nFULL JSON RESULT:")
    print("-" * 40)
    print(json.dumps(result, indent=2))


class TestNestedObjectNotebookCase:
    """Test cases based on the notebook example with detailed output."""

    @pytest.fixture
    def ground_truth(self):
        """Ground truth order for testing."""
        return create_ground_truth_order()

    @pytest.fixture
    def evaluator(self):
        """Structured model evaluator instance."""
        return StructuredModelEvaluator()

    def test_missing_item_case_detailed(self, ground_truth, evaluator):
        """Test the missing item case with detailed output for debugging."""
        print("=" * 60)
        print("TESTING MISSING ITEM CASE")
        print("=" * 60)

        # Create prediction with missing item
        prediction = create_prediction_order("missing_item")

        print("\nGROUND TRUTH PRODUCTS:")
        for i, product in enumerate(ground_truth.products):
            print(
                f"  {i + 1}. ID: {product.product_id}, Name: {product.name}, Price: {product.price}"
            )
            if product.attributes:
                print(f"     Attributes: {[attr.name for attr in product.attributes]}")

        print("\nPREDICTION PRODUCTS:")
        for i, product in enumerate(prediction.products):
            print(
                f"  {i + 1}. ID: {product.product_id}, Name: {product.name}, Price: {product.price}"
            )
            if product.attributes:
                print(f"     Attributes: {[attr.name for attr in product.attributes]}")

        # Evaluate using both methods
        print("-" * 40)
        print("METHOD 1: StructuredModelEvaluator.evaluate()")
        print("-" * 40)
        result1 = evaluator.evaluate(ground_truth, prediction)
        print_detailed_results(result1, "Missing Item - Method 1")

        print("-" * 40)
        print("METHOD 2: model.compare_with()")
        print("-" * 40)
        result2 = ground_truth.compare_with(
            prediction, include_confusion_matrix=True, document_non_matches=True
        )
        print_detailed_results(result2, "Missing Item - Method 2")

        # Compare the two methods
        print("-" * 40)
        print("COMPARISON OF METHODS")
        print("-" * 40)

        # Extract overall scores correctly from both methods
        method1_score = result1.get("overall_score", "N/A")
        if method1_score == "N/A" and "overall" in result1:
            method1_score = result1["overall"].get("anls_score", "N/A")

        method2_score = result2.get("overall_score", "N/A")

        print(f"Method 1 Overall Score: {method1_score}")
        print(f"Method 2 Overall Score: {method2_score}")
        print(f"Scores Match: {method1_score == method2_score}")

        # Check if there's a discrepancy
        if method1_score != method2_score:
            print("\n⚠️  POTENTIAL ISSUE: Different overall scores!")
            print(f"   Method 1 (StructuredModelEvaluator): {method1_score}")
            print(f"   Method 2 (compare_with): {method2_score}")

        # This assertion might fail - that's what we want to investigate
        # assert method1_score == method2_score

    def test_all_cases_comparison(self, ground_truth, evaluator):
        """Test all cases and compare results for debugging."""
        cases = ["good_match", "missing_item", "extra_item", "poor_match"]

        print("=" * 80)
        print("COMPREHENSIVE COMPARISON OF ALL CASES")
        print("=" * 80)

        for case in cases:
            prediction = create_prediction_order(case)
            result = evaluator.evaluate(ground_truth, prediction)

            # Extract overall score correctly
            overall_score = result.get("overall_score", "N/A")
            if overall_score == "N/A" and "overall" in result:
                overall_score = result["overall"].get("anls_score", "N/A")

            # Determine match status
            match_status = result.get("match", "N/A")
            if match_status == "N/A" and isinstance(overall_score, (int, float)):
                match_status = "MATCH" if overall_score >= 0.8 else "NO MATCH"

            print(f"\n{case.upper()}:")
            print(f"  Overall Score: {overall_score}")
            print(f"  Match Status: {match_status}")

            # Show field scores if available
            if "field_scores" in result:
                print(f"  Field Scores: {result['field_scores']}")
            elif "fields" in result:
                field_scores = {}
                for field, data in result["fields"].items():
                    if isinstance(data, dict) and "anls_score" in data:
                        field_scores[field] = data["anls_score"]
                print(f"  Field Scores: {field_scores}")

    def test_products_list_evaluation(self, ground_truth):
        """Focus specifically on the products list evaluation."""
        print("=" * 60)
        print("DETAILED PRODUCTS LIST EVALUATION")
        print("=" * 60)

        prediction = create_prediction_order("missing_item")

        # Test just the products comparison
        gt_products = ground_truth.products
        pred_products = prediction.products

        print(f"Ground Truth Products: {len(gt_products)}")
        print(f"Prediction Products: {len(pred_products)}")

        # Use the compare_with method on the products list
        # Note: This might need to be adapted based on how the list comparison works
        result = ground_truth.compare_with(prediction, include_confusion_matrix=True)

        if "field_scores" in result and "products" in result["field_scores"]:
            products_score = result["field_scores"]["products"]
            print(f"Products Field Score: {products_score}")

        print_detailed_results(result, "Products List Focus")


if __name__ == "__main__":
    # Run the tests directly for interactive debugging
    test_case = TestNestedObjectNotebookCase()
    gt = create_ground_truth_order()
    evaluator = StructuredModelEvaluator()

    test_case.test_missing_item_case_detailed(gt, evaluator)
    test_case.test_all_cases_comparison(gt, evaluator)
    test_case.test_products_list_evaluation(gt)
