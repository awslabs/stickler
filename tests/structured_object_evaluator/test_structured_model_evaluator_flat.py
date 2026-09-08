"""Tests for compare_with() metrics calculation for veterinary records models.

This test verifies that we can calculate precision, recall, F1, and accuracy metrics
at the field level and object level for simple objects in the toy veterinary records models.
"""

from typing import Optional

import pytest

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


# Define the models for the test
# Simple structure data
class PetOwner(StructuredModel):
    match_threshold = 0.7

    ownerId: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )  # Unique identifier for the pet owner
    firstName: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )  # First name of the pet owner
    lastName: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )  # Last name of the pet owner
    phoneNumber: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )  # Contact phone number
    memberSince: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )  # Date when owner first registered at clinic


class TestVetRecordsMetricsCalculation:
    """Test cases for veterinary records metrics calculation."""

    def setup_method(self):
        """Set up test data for simple PetOwner"""
        # Simple structure data
        self.gold_owner = {
            "ownerId": 1842,
            "firstName": "Margaret",
            "lastName": "Wilson",
            "phoneNumber": "555-789-1234",
        }

        self.pred_owner = {
            "ownerId": 1842,
            "firstName": "Margaret",
            "lastName": "Wilson",
            "phoneNumber": "666-789-1234",  # false discovery
            "memberSince": "2019-03-15",  # false alarm
        }

    def test_simple_pet_owner_metrics(self):
        """Test metrics for simple PetOwner structure."""
        # Create PetOwner objects
        gold_owner = PetOwner(**self.gold_owner)
        pred_owner = PetOwner(**self.pred_owner)

        # Test with traditional recall (default)
        results = gold_owner.compare_with(pred_owner, include_confusion_matrix=True, evaluator_format=True)

        # Expected metrics
        # 3 true positive: ownerId, firstName, lastName
        # 1 false discovery: phoneNumber (incorrect value)
        # 1 false alarm: memberSince (present in pred but not in gold)
        # Expected precision = TP/ (TP+FD+FA)= 3/(3+1+1) = 0.6
        # Expected traditional recall = TP/(TP+FN) = 3/(3+0) = 1
        # Expected alternative option = TP/(TP+FN+FD) = 3/(3+0+1) = 0.75
        # Expected F1 with traditional recall = 2*0.6*1/(0.6+1) = 0.75
        # Expected F1 with alternative option = 2*0.6*0.75/(0.6+0.75) = 0.67

        # Confusion matrix metrics
        cm = results["confusion_matrix"]
        # field-level confusion metrix values
        assert cm["fields"]["ownerId"]["tp"] == 1, "Expected 1 true positives"
        assert cm["fields"]["ownerId"]["fd"] == 0, "Expected 0 false discovery"
        assert cm["fields"]["ownerId"]["fa"] == 0, "Expected 0 false alarm"
        assert cm["fields"]["ownerId"]["fn"] == 0, "Expected 0 false negatives"
        assert cm["fields"]["ownerId"]["tn"] == 0, "Expected 0 true negatives"

        assert cm["fields"]["firstName"]["tp"] == 1, "Expected 1 true positives"
        assert cm["fields"]["firstName"]["fd"] == 0, "Expected 0 false discovery"
        assert cm["fields"]["firstName"]["fa"] == 0, "Expected 0 false alarm"
        assert cm["fields"]["firstName"]["fn"] == 0, "Expected 0 false negatives"
        assert cm["fields"]["firstName"]["tn"] == 0, "Expected 0 true negatives"

        assert cm["fields"]["lastName"]["tp"] == 1, "Expected 1 true positives"
        assert cm["fields"]["lastName"]["fd"] == 0, "Expected 0 false discovery"
        assert cm["fields"]["lastName"]["fa"] == 0, "Expected 0 false alarm"
        assert cm["fields"]["lastName"]["fn"] == 0, "Expected 0 false negatives"
        assert cm["fields"]["lastName"]["tn"] == 0, "Expected 0 true negatives"

        assert cm["fields"]["phoneNumber"]["tp"] == 0, "Expected 0 true positives"
        assert cm["fields"]["phoneNumber"]["fd"] == 1, "Expected 1 false discovery"
        assert cm["fields"]["phoneNumber"]["fa"] == 0, "Expected 0 false alarm"
        assert cm["fields"]["phoneNumber"]["fn"] == 0, "Expected 0 false negatives"
        assert cm["fields"]["phoneNumber"]["tn"] == 0, "Expected 0 true negatives"

        assert cm["fields"]["memberSince"]["tp"] == 0, "Expected 0 true positives"
        assert cm["fields"]["memberSince"]["fd"] == 0, "Expected 0 false discovery"
        assert cm["fields"]["memberSince"]["fa"] == 1, "Expected 1 false alarm"
        assert cm["fields"]["memberSince"]["fn"] == 0, "Expected 0 false negatives"
        assert cm["fields"]["memberSince"]["tn"] == 0, "Expected 0 true negatives"

        assert cm["overall"]["tp"] == 3, "Expected 3 true positives"
        assert cm["overall"]["fd"] == 1, "Expected 1 false discovery"
        assert cm["overall"]["fa"] == 1, "Expected 1 false alarm"
        assert cm["overall"]["fn"] == 0, "Expected 0 false negatives"
        assert cm["overall"]["tn"] == 0, "Expected 0 true negatives"

        # Check field-level metrics
        assert results["fields"]["ownerId"]["precision"] == 1.0
        assert results["fields"]["firstName"]["precision"] == 1.0
        assert results["fields"]["lastName"]["precision"] == 1.0
        assert results["fields"]["phoneNumber"]["precision"] == 0.0
        assert results["fields"]["memberSince"]["precision"] == 0.0

        # Expected metrics with traditional recall
        # Expected recall = TP/(TP+FN) = 3/(3+0) = 1
        # Expected F1 = 2*0.6*1/(0.6+1) = 0.75
        assert results["overall"]["precision"] == pytest.approx(0.6)
        assert results["overall"]["recall"] == pytest.approx(1.0)
        assert results["overall"]["f1"] == pytest.approx(0.75)

        # Test with alternative recall formula
        results_alt = gold_owner.compare_with(
            pred_owner, include_confusion_matrix=True, evaluator_format=True, recall_with_fd=True
        )

        # Expected metrics with alternative recall
        # Expected recall = TP/(TP+FN+FD) = 3/(3+0+1) = 0.75
        # Expected F1 = 2*0.6*0.75/(0.6+0.75) = 0.67
        assert results_alt["overall"]["precision"] == pytest.approx(0.6)
        assert results_alt["overall"]["recall"] == pytest.approx(0.75)
        assert results_alt["overall"]["f1"] == pytest.approx(0.67, abs=0.01)
