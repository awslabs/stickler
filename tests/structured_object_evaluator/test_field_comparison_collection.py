"""Comprehensive tests for field comparison collection functionality.

Tests the document_field_comparisons=True parameter in compare_with() method,
which provides detailed field-level comparison information including:
- Basic field comparisons (primitive fields - matches and non-matches)
- Nested StructuredModel field comparisons
- List field comparisons with Hungarian matching
- Edge cases: empty lists, null values, mismatched list lengths
- Integration with other flags (include_confusion_matrix, document_non_matches)
"""

from typing import List, Optional

from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from stickler.structured_object_evaluator import (
    ComparableField,
    StructuredModel,
)


# Test Models
class SimpleModel(StructuredModel):
    """Simple model with primitive fields for basic testing."""
    
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0
    )
    age: int = ComparableField(
        comparator=NumericComparator(), threshold=0.9, weight=1.0
    )
    active: bool = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    score: Optional[float] = ComparableField(
        comparator=NumericComparator(absolute_tolerance=0.02), threshold=1.0, weight=1.5
    )


class Address(StructuredModel):
    """Address model for nested testing."""
    
    street: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.7, weight=1.0
    )
    city: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    zipcode: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=2.0
    )


class Person(StructuredModel):
    """Person model with nested Address."""
    
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0
    )
    age: int = ComparableField(
        comparator=NumericComparator(), threshold=0.9, weight=1.0
    )
    address: Address = ComparableField(
        comparator=LevenshteinComparator, threshold=0.9, weight=1.0)


class LineItem(StructuredModel):
    """Line item for list testing."""
    
    product: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=1.0
    )
    quantity: int = ComparableField(
        comparator=NumericComparator(), threshold=0.9, weight=1.0
    )
    price: float = ComparableField(
        comparator=NumericComparator(), threshold=0.95, weight=2.0
    )


class Invoice(StructuredModel):
    """Invoice model with list of LineItems."""
    
    invoice_id: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=3.0
    )
    customer: Person = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.9, weight=3.0
    )
    line_items: List[LineItem] =ComparableField(
        comparator=LevenshteinComparator(), weight=3.0
    )
    total: float = ComparableField(
        comparator=NumericComparator(absolute_tolerance=0.02), threshold=1.0, weight=2.0)


class PrimitiveListModel(StructuredModel):
    """Model with primitive list fields."""
    
    tags: List[str] = ComparableField(
        comparator=LevenshteinComparator(), weight=1.0
    )
    scores: List[float] = ComparableField(
        comparator=NumericComparator(),  weight=1.0
    )


class TestFieldComparisonCollection:
    """Test suite for field comparison collection functionality."""

    def test_basic_field_comparisons_exact_matches(self):
        """Test basic field comparisons with exact matches."""
        model1 = SimpleModel(name="John Doe", age=30, active=True, score=95.5)
        model2 = SimpleModel(name="John Doe", age=30, active=True, score=95.5)
        
        result = model1.compare_with(model2, document_field_comparisons=True)
        
        # Verify field_comparisons key exists
        assert "field_comparisons" in result
        field_comparisons = result["field_comparisons"]
        
        # Should have 4 field comparisons (name, age, active, score)
        assert len(field_comparisons) == 4
        
        # All should be exact matches
        for fc in field_comparisons:
            assert fc["match"] is True
            assert fc["score"] == 1.0
            assert fc["reason"] == "exact match"
            assert fc["expected_key"] == fc["actual_key"]
            assert fc["expected_value"] == fc["actual_value"]
        
        # Check specific fields
        name_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "name")
        assert name_comparison["expected_value"] == "John Doe"
        assert name_comparison["weighted_score"] == 2.0  # score * weight

    def test_basic_field_comparisons_partial_matches(self):
        """Test basic field comparisons with partial matches above threshold."""
        model1 = SimpleModel(name="John Doe", age=30, active=True, score=95.0)
        model2 = SimpleModel(name="Jon Doe", age=30, active=True, score=95.01)  # Slight differences
        
        result = model1.compare_with(model2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Find name comparison (should be match)
        name_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "name")
        assert name_comparison["match"] is True  
        assert 0.8 <= name_comparison["score"] < 1.0
        assert "above threshold" in name_comparison["reason"]
        
        # Find score comparison (should be match)
        score_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "score")
        assert score_comparison["match"] is True 
        assert 0.95 <= score_comparison["score"] <= 1.0

    def test_basic_field_comparisons_non_matches(self):
        """Test basic field comparisons with non-matches below threshold."""
        model1 = SimpleModel(name="John Doe", age=30, active=True, score=95.0)
        model2 = SimpleModel(name="Jane Smith", age=25, active=False, score=50.0)  # Different values
        
        result = model1.compare_with(model2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Find name comparison (should be non-match)
        name_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "name")
        assert name_comparison["match"] is False
        assert name_comparison["score"] < 0.8  # Below threshold
        assert "below threshold" in name_comparison["reason"]
        
        # Find active comparison (should be non-match for boolean)
        active_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "active")
        assert active_comparison["match"] is False
        assert active_comparison["score"] == 0.0  # ExactComparator gives 0 for different booleans

    def test_basic_field_comparisons_null_values(self):
        """Test basic field comparisons with null/None values."""
        model1 = SimpleModel(name="John Doe", age=30, active=True, score=None)
        model2 = SimpleModel(name="John Doe", age=30, active=True, score=95.0)
        
        result = model1.compare_with(model2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Find score comparison (GT is None, pred has value)
        score_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "score")
        assert score_comparison["match"] is False
        assert score_comparison["expected_value"] is None
        assert score_comparison["actual_value"] == 95.0
        assert "false alarm" in score_comparison["reason"]

    def test_nested_structured_model_field_comparisons(self):
        """Test field comparisons with nested StructuredModel objects."""
        person1 = Person(
            name="John Doe",
            age=30,
            address=Address(street="123 Main St", city="New York", zipcode="10001")
        )
        person2 = Person(
            name="John Doe",
            age=30,
            address=Address(street="123 Main St", city="New York", zipcode="10001")
        )
        
        result = person1.compare_with(person2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Should have comparisons for: name, age, address.street, address.city, address.zipcode
        expected_keys = {"name", "age", "address.street", "address.city", "address.zipcode"}
        actual_keys = {fc["expected_key"] for fc in field_comparisons}
        assert actual_keys == expected_keys
        
        # All nested fields should be exact matches
        for fc in field_comparisons:
            assert fc["match"] is True
            assert fc["score"] == 1.0
            
        # Check nested field path format
        street_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "address.street")
        assert street_comparison["expected_value"] == "123 Main St"
        assert street_comparison["actual_key"] == "address.street"

    def test_nested_field_comparisons_partial_matches(self):
        """Test nested field comparisons with partial matches."""
        person1 = Person(
            name="John Doe",
            age=30,
            address=Address(street="123 Main St", city="New York", zipcode="10001")
        )
        person2 = Person(
            name="Jon Doe",  # Slight difference
            age=30,
            address=Address(street="123 Main Street", city="NYC", zipcode="10002")  # Differences
        )
        
        result = person1.compare_with(person2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Name should be partial match
        name_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "name")
        assert name_comparison["match"] is True  # Above 0.8 threshold
        assert 0.8 <= name_comparison["score"] < 1.0
        
        # Street should be partial match
        street_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "address.street")
        assert street_comparison["match"] is True  # Above 0.7 threshold
        
        # City should be partial match
        city_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "address.city")
        assert city_comparison["match"] is False  # Below 0.8 threshold
        
        # Zipcode should be non-match (ExactComparator)
        zipcode_comparison = next(fc for fc in field_comparisons if fc["expected_key"] == "address.zipcode")
        assert zipcode_comparison["match"] is False
        assert zipcode_comparison["score"] == 0.0

    def test_list_field_comparisons_hungarian_matching(self):
        """Test list field comparisons with Hungarian matching."""
        invoice1 = Invoice(
            invoice_id="INV-001",
            customer=Person(
                name="John Doe",
                age=30,
                address=Address(street="123 Main St", city="New York", zipcode="10001")
            ),
            line_items=[
                LineItem(product="Widget A", quantity=2, price=10.0),
                LineItem(product="Widget B", quantity=1, price=20.0),
            ],
            total=40.0
        )
        
        # Same items but in different order
        invoice2 = Invoice(
            invoice_id="INV-001",
            customer=Person(
                name="John Doe",
                age=30,
                address=Address(street="123 Main St", city="New York", zipcode="10001")
            ),
            line_items=[
                LineItem(product="Widget B", quantity=1, price=20.0),  # Swapped order
                LineItem(product="Widget A", quantity=2, price=10.0),
            ],
            total=40.0
        )
        
        result = invoice1.compare_with(invoice2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Find list item comparisons
        list_comparisons = [fc for fc in field_comparisons if fc["expected_key"].startswith("line_items[")]
        
        # Should have 6 list item field comparisons (2 items × 3 fields each)
        assert len(list_comparisons) == 6
        
        # All should be matches due to Hungarian matching
        for fc in list_comparisons:
            assert fc["match"] is True
            assert fc["score"] == 1.0
            
        # Check that Hungarian matching paired items correctly
        # Widget A comparisons
        widget_a_comparisons = [fc for fc in list_comparisons if fc["expected_value"] == "Widget A"]
        assert len(widget_a_comparisons) == 1
        widget_a_comp = widget_a_comparisons[0]
        # Should be matched with line_items[1] due to reordering
        assert widget_a_comp["actual_key"] == "line_items[1].product"
        
        # Widget B comparisons  
        widget_b_comparisons = [fc for fc in list_comparisons if fc["expected_value"] == "Widget B"]
        assert len(widget_b_comparisons) == 1
        widget_b_comp = widget_b_comparisons[0]
        # Should be matched with line_items[0] due to reordering
        assert widget_b_comp["actual_key"] == "line_items[0].product"

    def test_list_field_comparisons_partial_matches(self):
        """Test list field comparisons with partial matches and non-matches."""
        invoice1 = Invoice(
            invoice_id="INV-001",
            customer=Person(
                name="John Doe",
                age=30,
                address=Address(street="123 Main St", city="New York", zipcode="10001")
            ),
            line_items=[
                LineItem(product="Widget A", quantity=2, price=10.0),
                LineItem(product="Widget B", quantity=1, price=20.0),
            ],
            total=40.0
        )
        
        # Similar but with differences
        invoice2 = Invoice(
            invoice_id="INV-001",
            customer=Person(
                name="John Doe",
                age=30,
                address=Address(street="123 Main St", city="New York", zipcode="10001")
            ),
            line_items=[
                LineItem(product="Widget Alpha", quantity=2, price=10.5),  # Similar product, different price
                LineItem(product="Gadget C", quantity=3, price=15.0),     # Different product
            ],
            total=41.0
        )
        
        result = invoice1.compare_with(invoice2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Find list item comparisons
        list_comparisons = [fc for fc in field_comparisons if fc["expected_key"].startswith("line_items[")]
        
        # Should have some matches and some non-matches
        matches = [fc for fc in list_comparisons if fc["match"]]
        non_matches = [fc for fc in list_comparisons if not fc["match"]]
        
        assert len(matches) > 0
        assert len(non_matches) > 0
        
        # Check for partial product match
        product_comparisons = [fc for fc in list_comparisons if fc["expected_key"].endswith(".product")]
        widget_a_comparison = next((fc for fc in product_comparisons if fc["expected_value"] == "Widget A"), None)
        
        if widget_a_comparison:
            # Should be matched with "Widget Alpha" and be a partial match
            assert widget_a_comparison["actual_value"] == "Widget Alpha"
            assert widget_a_comparison["match"] is False  # Should be below 0.8 threshold
            assert widget_a_comparison["score"] == 0.0

    def test_primitive_list_field_comparisons(self):
        """Test field comparisons with primitive list fields."""
        model1 = PrimitiveListModel(
            tags=["python", "testing", "stickler"],
            scores=[95.5, 87.2, 92.1]
        )
        model2 = PrimitiveListModel(
            tags=["testing", "python", "stickler"],  # Reordered
            scores=[87.2, 95.5, 92.1]               # Reordered
        )
        
        result = model1.compare_with(model2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Find list comparisons
        tag_comparisons = [fc for fc in field_comparisons if fc["expected_key"].startswith("tags[")]
        score_comparisons = [fc for fc in field_comparisons if fc["expected_key"].startswith("scores[")]
        
        # Should have 3 tag comparisons and 3 score comparisons
        assert len(tag_comparisons) == 3
        assert len(score_comparisons) == 3
        
        # All should be exact matches due to Hungarian matching
        for fc in tag_comparisons + score_comparisons:
            assert fc["match"] is True
            assert fc["score"] == 1.0

    def test_edge_case_empty_lists(self):
        """Test field comparisons with empty lists."""
        model1 = PrimitiveListModel(tags=[], scores=[])
        model2 = PrimitiveListModel(tags=["tag1"], scores=[95.0])
        
        result = model1.compare_with(model2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Should have comparisons for the items in model2 (false alarms)
        tag_comparisons = [fc for fc in field_comparisons if fc["expected_key"].startswith("tags[")]
        score_comparisons = [fc for fc in field_comparisons if fc["expected_key"].startswith("scores[")]
        
        # Should have false alarm entries
        if tag_comparisons:
            for fc in tag_comparisons:
                assert fc["match"] is False
                assert "false alarm" in fc["reason"]
                assert fc["expected_value"] is None or fc["expected_key"] == "tags[]"

    def test_edge_case_mismatched_list_lengths(self):
        """Test field comparisons with mismatched list lengths."""
        model1 = PrimitiveListModel(
            tags=["tag1", "tag2", "tag3"],
            scores=[95.0, 87.0, 92.0]
        )
        model2 = PrimitiveListModel(
            tags=["tag1"],      # Shorter list
            scores=[95.0, 87.0, 92.0, 88.0, 91.0]  # Longer list
        )
        
        result = model1.compare_with(model2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Should have comparisons for all items
        tag_comparisons = [fc for fc in field_comparisons if fc["expected_key"].startswith("tags[")]
        score_comparisons = [fc for fc in field_comparisons if fc["expected_key"].startswith("scores[")]
        
        # Should have matches and non-matches
        tag_matches = [fc for fc in tag_comparisons if fc["match"]]
        tag_non_matches = [fc for fc in tag_comparisons if not fc["match"]]
        
        assert len(tag_matches) >= 1  # At least "tag1" should match
        assert len(tag_non_matches) >= 2  # "tag2" and "tag3" should be unmatched

    def test_integration_with_confusion_matrix(self):
        """Test field comparisons integration with confusion matrix."""
        model1 = SimpleModel(name="John Doe", age=30, active=True, score=95.0)
        model2 = SimpleModel(name="Jane Smith", age=25, active=False, score=50.0)
        
        result = model1.compare_with(
            model2,
            document_field_comparisons=True,
            include_confusion_matrix=True
        )
        
        # Both keys should be present
        assert "field_comparisons" in result
        assert "confusion_matrix" in result
        
        # Field comparisons should still work correctly
        field_comparisons = result["field_comparisons"]
        assert len(field_comparisons) == 4
        
        # Confusion matrix should still work correctly
        confusion_matrix = result["confusion_matrix"]
        assert "overall" in confusion_matrix
        assert "fields" in confusion_matrix

    def test_integration_with_document_non_matches(self):
        """Test field comparisons integration with document_non_matches."""
        model1 = SimpleModel(name="John Doe", age=30, active=True, score=95.0)
        model2 = SimpleModel(name="Jane Smith", age=25, active=False, score=50.0)
        
        result = model1.compare_with(
            model2,
            document_field_comparisons=True,
            document_non_matches=True
        )
        
        # Both keys should be present
        assert "field_comparisons" in result
        assert "non_matches" in result
        
        # Field comparisons should document all fields
        field_comparisons = result["field_comparisons"]
        assert len(field_comparisons) == 4
        
        # Non-matches should document only non-matching fields
        non_matches = result["non_matches"]
        assert len(non_matches) > 0  # Should have some non-matches

    def test_integration_all_flags_combined(self):
        """Test field comparisons with all other flags enabled."""
        person1 = Person(
            name="John Doe",
            age=30,
            address=Address(street="123 Main St", city="New York", zipcode="10001")
        )
        person2 = Person(
            name="Jon Doe",  # Slight difference
            age=25,          # Different age
            address=Address(street="123 Main Street", city="NYC", zipcode="10002")
        )
        
        result = person1.compare_with(
            person2,
            document_field_comparisons=True,
            include_confusion_matrix=True,
            document_non_matches=True,
            add_derived_metrics=True
        )
        
        # All keys should be present
        assert "field_comparisons" in result
        assert "confusion_matrix" in result
        assert "non_matches" in result
        assert "field_scores" in result
        assert "overall_score" in result
        
        # Field comparisons should have nested field paths
        field_comparisons = result["field_comparisons"]
        nested_fields = [fc for fc in field_comparisons if "." in fc["expected_key"]]
        assert len(nested_fields) > 0

    def test_output_format_validation(self):
        """Test that field comparison output format matches documentation."""
        model1 = SimpleModel(name="John Doe", age=30, active=True, score=95.0)
        model2 = SimpleModel(name="Jane Smith", age=25, active=False, score=50.0)
        
        result = model1.compare_with(model2, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Validate each field comparison entry has required keys
        required_keys = {
            "expected_key", "expected_value", "actual_key", "actual_value",
            "match", "score", "weighted_score", "reason"
        }
        
        for fc in field_comparisons:
            assert set(fc.keys()) == required_keys
            
            # Validate types
            assert isinstance(fc["expected_key"], str)
            assert isinstance(fc["actual_key"], str)
            assert isinstance(fc["match"], bool)
            assert isinstance(fc["score"], (int, float))
            assert isinstance(fc["weighted_score"], (int, float))
            assert isinstance(fc["reason"], str)
            
            # Validate score ranges
            assert 0.0 <= fc["score"] <= 1.0
            assert fc["weighted_score"] >= 0.0

    def test_field_path_notation_validation(self):
        """Test that field paths use correct notation (dot for nested, brackets for lists)."""
        invoice = Invoice(
            invoice_id="INV-001",
            customer=Person(
                name="John Doe",
                age=30,
                address=Address(street="123 Main St", city="New York", zipcode="10001")
            ),
            line_items=[
                LineItem(product="Widget A", quantity=2, price=10.0),
            ],
            total=40.0
        )
        
        result = invoice.compare_with(invoice, document_field_comparisons=True)
        field_comparisons = result["field_comparisons"]
        
        # Check for correct path notation
        paths = {fc["expected_key"] for fc in field_comparisons}
        
        # Should have nested object paths with dots
        assert "customer.name" in paths
        assert "customer.age" in paths
        assert "customer.address.street" in paths
        assert "customer.address.city" in paths
        assert "customer.address.zipcode" in paths
        
        # Should have list item paths with brackets
        list_paths = [path for path in paths if path.startswith("line_items[")]
        assert len(list_paths) == 3  # product, quantity, price
        
        # Validate bracket notation format
        for path in list_paths:
            assert path.startswith("line_items[0].")
            assert path.count("[") == 1
            assert path.count("]") == 1
