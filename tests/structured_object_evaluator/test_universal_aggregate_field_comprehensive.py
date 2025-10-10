

"""
Comprehensive tests for the Universal Aggregate Field feature.

This test suite validates that the new universal aggregate field feature works correctly:
1. Every node in the comparison result tree has an 'aggregate' field
2. Aggregate fields appear as siblings of 'overall' and 'fields' 
3. Aggregate calculations sum all primitive field metrics below each node
4. Derived metrics are included in aggregate fields
5. Structure is consistent across all levels
"""

import unittest
import warnings
from typing import Optional, List

from stickler.structured_object_evaluator.models.structured_model import (
    StructuredModel,
)
from stickler.structured_object_evaluator.models.comparable_field import (
    ComparableField,
)
from stickler.comparators.exact import ExactComparator
from stickler.comparators.numeric import NumericComparator
from stickler.comparators.levenshtein import LevenshteinComparator


# Define test models for comprehensive testing
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


class Owner(StructuredModel):
    id: int = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    name: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    contact: Contact = ComparableField( 
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    address: Address = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class Pet(StructuredModel):
    match_threshold = 1.0
    
    pet_id: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.85, weight=1.0
    )
    species: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.85, weight=1.0
    )
    breed: Optional[str] = ComparableField(
        default=None, comparator=LevenshteinComparator(), threshold=0.85, weight=1.0
    )
    age: Optional[int] = ComparableField(
        default=None, comparator=NumericComparator(), threshold=0.9, weight=1.0
    )


class VeterinaryRecord(StructuredModel):
    record_id: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    owner: Owner = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    pets: List[Pet] = ComparableField(weight=1.0)


class TestUniversalAggregateField(unittest.TestCase):
    """Test cases for universal aggregate field feature."""
    
    def setUp(self):
        """Set up test data for comprehensive aggregate field testing."""
        # Complex nested structure with multiple levels
        self.gt_record = {
            "record_id": 12345,
            "owner": {
                "id": 1001,
                "name": "John Smith",
                "contact": {"phone": "555-123-4567", "email": "john@example.com"},
                "address": {
                    "street": "123 Main St",
                    "city": "Seattle",
                    "zip_code": "98101",
                },
            },
            "pets": [
                {
                    "pet_id": 2001,
                    "name": "Buddy",
                    "species": "Dog",
                    "breed": "Golden Retriever",
                    "age": 5,
                },
                {
                    "pet_id": 2002,
                    "name": "Whiskers",
                    "species": "Cat",
                    "breed": "Siamese",
                },
            ],
        }
        
        # Prediction with various types of mismatches
        self.pred_record = {
            "record_id": 12345,  # TP
            "owner": {
                "id": 1001,  # TP
                "name": "John Smith",  # TP
                "contact": {
                    "phone": "555-999-8888",  # FD (wrong phone)
                    "email": "john@example.com",  # TP
                },
                "address": {
                    "street": "456 Oak Ave",  # FD (wrong street)
                    "city": "Seattle",  # TP
                    "zip_code": "98102",  # FD (wrong zip)
                },
            },
            "pets": [
                {
                    "pet_id": 2001,  # TP
                    "name": "Buddy",  # TP
                    "species": "Dog",  # TP
                    "breed": "Golden Retriever",  # TP
                    "age": 6,  # FD (wrong age)
                },
                {
                    "pet_id": 2002,  # TP
                    "name": "Whiskers",  # TP
                    "species": "Cat",  # TP
                    # Missing breed (FN)
                    "age": 3,  # FA (extra age)
                },
            ],
        }

    def test_deprecation_warning_for_legacy_aggregate_parameter(self):
        """Test that using aggregate=True triggers deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should trigger a deprecation warning
            field = ComparableField(aggregate=True)
            
            # Verify warning was triggered
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("aggregate", str(w[0].message))
            self.assertIn("deprecated", str(w[0].message))

    def test_universal_aggregate_field_presence(self):
        """Test that aggregate fields are present at every level."""
        gt = VeterinaryRecord(**self.gt_record)
        pred = VeterinaryRecord(**self.pred_record)
        
        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        
        # Top level should have aggregate
        self.assertIn("aggregate", cm, "Top level missing aggregate field")
        
        # Overall level should NOT have aggregate (it's a sibling, not nested)
        self.assertNotIn(
            "aggregate", cm["overall"], "Overall should not contain nested aggregate"
        )
        
        # Every field should have aggregate as sibling of overall/fields
        for field_name, field_data in cm["fields"].items():
            self.assertIn(
                "aggregate", field_data, f"Field '{field_name}' missing aggregate"
            )
            
            # If field has nested fields, check them too
            if "fields" in field_data and field_data["fields"]:
                for nested_name, nested_data in field_data["fields"].items():
                    self.assertIn(
                        "aggregate",
                        nested_data,
                        f"Nested field '{field_name}.{nested_name}' missing aggregate",
                    )

    def test_aggregate_field_structure_consistency(self):
        """Test that aggregate fields have consistent structure."""
        gt = VeterinaryRecord(**self.gt_record)
        pred = VeterinaryRecord(**self.pred_record)
        
        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        
        def validate_aggregate_structure(aggregate_data, path=""):
            """Recursively validate aggregate field structure."""
            # Must have confusion matrix metrics
            required_metrics = ["tp", "fa", "fd", "fp", "tn", "fn"]
            for metric in required_metrics:
                self.assertIn(
                    metric, aggregate_data, f"Aggregate at '{path}' missing {metric}"
                )
                self.assertIsInstance(
                    aggregate_data[metric],
                    int,
                    f"Aggregate {metric} at '{path}' not integer",
                )
            
            # Must have derived metrics
            self.assertIn(
                "derived",
                aggregate_data,
                f"Aggregate at '{path}' missing derived metrics",
            )
            derived = aggregate_data["derived"]
            
            derived_metrics = ["cm_precision", "cm_recall", "cm_f1", "cm_accuracy"]
            for metric in derived_metrics:
                self.assertIn(
                    metric, derived, f"Aggregate derived at '{path}' missing {metric}"
                )
                self.assertIsInstance(
                    derived[metric],
                    (int, float),
                    f"Derived {metric} at '{path}' not numeric",
                )
        
        # Validate top-level aggregate
        validate_aggregate_structure(cm["aggregate"], "top")
        
        # Validate field-level aggregates
        for field_name, field_data in cm["fields"].items():
            validate_aggregate_structure(
                field_data["aggregate"], f"fields.{field_name}"
            )
            
            # Validate nested field aggregates
            if "fields" in field_data:
                for nested_name, nested_data in field_data["fields"].items():
                    if "aggregate" in nested_data:
                        validate_aggregate_structure(
                            nested_data["aggregate"],
                            f"fields.{field_name}.{nested_name}",
                        )

    def test_aggregate_calculation_correctness(self):
        """Test that aggregate calculations sum primitive fields correctly."""
        gt = VeterinaryRecord(**self.gt_record)
        pred = VeterinaryRecord(**self.pred_record)
        
        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        
        # Based on actual behavior observed:
        # record_id: TP=1 (matches)
        # owner.id: TP=1 (matches)
        # owner.name: TP=1 (matches)  
        # owner.contact.phone: FD=1, FP=1 (mismatch)
        # owner.contact.email: TP=1 (matches)
        # owner.address.street: FD=1, FP=1 (mismatch)
        # owner.address.city: TP=1 (matches)
        # owner.address.zip_code: FD=1, FP=1 (mismatch)
        # pets.pet_id: TP=2 (both match)
        # pets.name: TP=2 (both match)
        # pets.species: TP=2 (both match)
        # pets.breed: TP=1, FN=1 (one matches, one missing)
        # pets.age: FA=1, FD=1, FP=2 (one wrong, one extra)
        
        # Total actual: TP=12, FA=1, FD=4, FP=5, FN=1
        expected_top_aggregate = {
            'tp': 12, 'fa': 1, 'fd': 4, 'fp': 5, 'tn': 0, 'fn': 1
        }
        
        top_aggregate = cm["aggregate"]
        for metric, expected in expected_top_aggregate.items():
            self.assertEqual(
                top_aggregate[metric],
                expected,
                f"Top aggregate {metric}: expected {expected}, got {top_aggregate[metric]}",
            )

    def test_nested_aggregate_calculations(self):
        """Test aggregate calculations for nested structures."""
        gt = VeterinaryRecord(**self.gt_record)
        pred = VeterinaryRecord(**self.pred_record)
        
        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        
        # Test owner aggregate (should sum all owner.* primitive fields)
        # owner.id: TP=1, owner.name: TP=1, contact.phone: FD=1,FP=1, contact.email: TP=1
        # address.street: FD=1,FP=1, address.city: TP=1, address.zip_code: FD=1,FP=1
        # Total: TP=4, FD=3, FP=3
        owner_aggregate = cm["fields"]["owner"]["aggregate"]
        self.assertEqual(owner_aggregate["tp"], 4, "Owner aggregate TP incorrect")
        self.assertEqual(owner_aggregate["fd"], 3, "Owner aggregate FD incorrect")
        self.assertEqual(owner_aggregate["fp"], 3, "Owner aggregate FP incorrect")
        
        # Test contact aggregate (should sum contact.* primitive fields)
        # contact.phone: FD=1,FP=1, contact.email: TP=1
        # Total: TP=1, FD=1, FP=1
        contact_aggregate = cm["fields"]["owner"]["fields"]["contact"]["aggregate"]
        self.assertEqual(contact_aggregate["tp"], 1, "Contact aggregate TP incorrect")
        self.assertEqual(contact_aggregate["fd"], 1, "Contact aggregate FD incorrect")
        self.assertEqual(contact_aggregate["fp"], 1, "Contact aggregate FP incorrect")
        
        # Test address aggregate (should sum address.* primitive fields)
        # address.street: FD=1,FP=1, address.city: TP=1, address.zip_code: FD=1,FP=1
        # Total: TP=1, FD=2, FP=2
        address_aggregate = cm["fields"]["owner"]["fields"]["address"]["aggregate"]
        self.assertEqual(address_aggregate["tp"], 1, "Address aggregate TP incorrect")
        self.assertEqual(address_aggregate["fd"], 2, "Address aggregate FD incorrect")
        self.assertEqual(address_aggregate["fp"], 2, "Address aggregate FP incorrect")

    def test_list_field_aggregate_calculations(self):
        """Test aggregate calculations for list fields."""
        gt = VeterinaryRecord(**self.gt_record)
        pred = VeterinaryRecord(**self.pred_record)
        
        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        
        # Test pets aggregate - based on actual behavior, the Hungarian matching
        # successfully matches the pets and aggregates all their field metrics:
        # pets.pet_id: TP=2, pets.name: TP=2, pets.species: TP=2
        # pets.breed: TP=1, FN=1, pets.age: FA=1, FD=1, FP=2
        # Total: TP=7, FA=1, FD=1, FP=2, FN=1
        pets_aggregate = cm['fields']['pets']['aggregate']
        self.assertEqual(pets_aggregate['tp'], 7, "Pets aggregate TP incorrect")
        self.assertEqual(pets_aggregate['fa'], 1, "Pets aggregate FA incorrect")
        self.assertEqual(pets_aggregate['fd'], 1, "Pets aggregate FD incorrect")
        self.assertEqual(pets_aggregate['fp'], 2, "Pets aggregate FP incorrect")
        self.assertEqual(pets_aggregate['tn'], 0, "Pets aggregate TN incorrect")
        self.assertEqual(pets_aggregate['fn'], 1, "Pets aggregate FN incorrect")

    def test_primitive_field_aggregate_equals_overall(self):
        """Test that for primitive fields, aggregate equals overall metrics."""
        gt = VeterinaryRecord(**self.gt_record)
        pred = VeterinaryRecord(**self.pred_record)
        
        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        
        # Check primitive field: record_id
        record_id_field = cm["fields"]["record_id"]
        overall_metrics = record_id_field["overall"]
        aggregate_metrics = record_id_field["aggregate"]
        
        # For primitive fields, aggregate should equal overall (excluding derived)
        confusion_metrics = ["tp", "fa", "fd", "fp", "tn", "fn"]
        for metric in confusion_metrics:
            self.assertEqual(
                overall_metrics[metric],
                aggregate_metrics[metric],
                f"Primitive field record_id: aggregate {metric} != overall {metric}",
            )

    def test_aggregate_derived_metrics_calculation(self):
        """Test that derived metrics in aggregate fields are calculated correctly."""
        gt = VeterinaryRecord(**self.gt_record)
        pred = VeterinaryRecord(**self.pred_record)
        
        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        
        # Test top-level aggregate derived metrics
        top_aggregate = cm["aggregate"]
        derived = top_aggregate["derived"]
        
        # Calculate expected derived metrics
        tp, fp, fn = top_aggregate["tp"], top_aggregate["fp"], top_aggregate["fn"]
        
        expected_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        expected_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        expected_f1 = (
            2
            * expected_precision
            * expected_recall
            / (expected_precision + expected_recall)
            if (expected_precision + expected_recall) > 0
            else 0
        )
        expected_accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        self.assertAlmostEqual(derived["cm_precision"], expected_precision, places=3)
        self.assertAlmostEqual(derived["cm_recall"], expected_recall, places=3)
        self.assertAlmostEqual(derived["cm_f1"], expected_f1, places=3)
        self.assertAlmostEqual(derived["cm_accuracy"], expected_accuracy, places=3)

    def test_aggregate_field_placement_as_sibling(self):
        """Test that aggregate fields are siblings of overall/fields, not nested within."""
        gt = VeterinaryRecord(**self.gt_record)
        pred = VeterinaryRecord(**self.pred_record)
        
        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        
        # Top level structure validation
        expected_top_keys = {"overall", "fields", "aggregate", "non_matches"}
        actual_top_keys = set(cm.keys())
        self.assertTrue(
            expected_top_keys.issubset(actual_top_keys),
            f"Top level missing keys. Expected subset: {expected_top_keys}, Got: {actual_top_keys}",
        )
        
        # Field level structure validation - updated for unified structure
        for field_name, field_data in cm["fields"].items():
            if (
                "overall" in field_data
            ):  # All fields now have 'overall' in unified structure
                # All fields must have 'overall' and 'aggregate'
                required_keys = {"overall", "aggregate"}
                actual_field_keys = set(field_data.keys())
                self.assertTrue(
                    required_keys.issubset(actual_field_keys),
                    f"Field '{field_name}' missing required keys. Expected subset: {required_keys}, Got: {actual_field_keys}",
                )
                
                # Parent container fields (List, StructuredModel with nested fields) also have 'fields'
                # Primitive fields (str, int, etc.) do not have 'fields' - this is the semantic meaning
                
                # Aggregate should NOT be nested in overall
                self.assertNotIn(
                    "aggregate",
                    field_data["overall"],
                    f"Field '{field_name}' has aggregate nested in overall (should be sibling)",
                )

    def test_no_configuration_required(self):
        """Test that aggregate fields work without any configuration."""

        # Simple model with no aggregate=True parameters
        class SimpleModel(StructuredModel):
            name: str = ComparableField(comparator=ExactComparator(), threshold=1.0)
            age: int = ComparableField(comparator=ExactComparator(), threshold=1.0)
        
        gt = SimpleModel(name="John", age=30)
        pred = SimpleModel(name="John", age=25)
        
        result = gt.compare_with(pred, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        
        # Should have aggregate fields automatically
        self.assertIn("aggregate", cm)
        self.assertIn("aggregate", cm["fields"]["name"])
        self.assertIn("aggregate", cm["fields"]["age"])
        
        # Aggregate should have correct values
        self.assertEqual(cm["aggregate"]["tp"], 1)  # name matches
        self.assertEqual(cm["aggregate"]["fd"], 1)  # age mismatch
        self.assertEqual(cm["aggregate"]["fp"], 1)  # age mismatch

    def test_backward_compatibility(self):
        """Test that existing code continues to work unchanged."""
        gt = VeterinaryRecord(**self.gt_record)
        pred = VeterinaryRecord(**self.pred_record)
        
        # Old way of calling compare_with should still work
        result = gt.compare_with(pred, include_confusion_matrix=True)
        
        # All existing fields should still be present
        self.assertIn("field_scores", result)
        self.assertIn("overall_score", result)
        self.assertIn("all_fields_matched", result)
        self.assertIn("confusion_matrix", result)
        
        cm = result["confusion_matrix"]
        self.assertIn("overall", cm)
        self.assertIn("fields", cm)
        
        # New aggregate fields should be added without breaking existing structure
        self.assertIn("aggregate", cm)


if __name__ == "__main__":
    unittest.main()
