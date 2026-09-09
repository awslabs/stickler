"""Tests for StructuredModel.compare_with() metrics calculation for veterinary records models.

This test verifies that we can calculate precision, recall, F1, and accuracy metrics
at the field level and object level for nested structures in the toy veterinary records models
using the direct compare_with() method.

The expectations follow threshold-gated evaluation: object pairs below
``match_threshold`` are atomic false discoveries and produce no field-level
breakdown. True negatives are excluded from the object similarity used for
that gate.
"""

from typing import List, Optional

import pytest

from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

# Note: No longer using StructuredModelEvaluator - using direct compare_with() method


# Define the models for the test
# Nested structure data including a list of StructuredModel
class Contact(StructuredModel):
    phone: str = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    email: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class Owner(StructuredModel):
    id: int = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)
    name: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)

    contact: Contact = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=1.0,
    )


class Pet(StructuredModel):
    petId: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.85, weight=1.0
    )
    species: str = ComparableField(
        default=None, comparator=LevenshteinComparator(), threshold=0.85, weight=1.0
    )
    breed: Optional[str] = ComparableField(
        default=None, comparator=LevenshteinComparator(), threshold=0.85, weight=1.0
    )
    birthdate: Optional[str] = ComparableField(
        default=None, comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    weight: Optional[float] = ComparableField(
        default=None, comparator=NumericComparator(), threshold=0.9, weight=1.0
    )


class VeterinaryRecord(StructuredModel):
    recordId: int = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )

    owner: Owner = ComparableField(
        comparator=ExactComparator(),
        threshold=1.0,
        weight=1.0,
    )

    pets: List[Pet] = ComparableField(
        weight=1.0,
    )


class TestVetRecordsMetricsCalculation:
    """Test cases for veterinary records metrics calculation."""

    def setup_method(self):
        """Set up test data for nested VeterinaryRecord structures."""
        # Nested structure data
        self.gold_record = {
            "recordId": 4721,
            "owner": {
                "id": 1501,
                "name": "Sarah Johnson",
                "contact": {"phone": "555-689-1234"},
            },
            "pets": [
                {
                    "petId": 3501,
                    "name": "Max",
                    "species": "Dog",
                    "breed": "Golden Retriever",
                    "birthdate": "2018-05-12",
                },
                {"petId": 3512, "name": "Buttons", "species": "Cat"},
            ],
        }

        self.pred_record = {
            "recordId": 4721,
            "owner": {
                "id": 1501,
                "name": "Sarah Johnson",
                "contact": {
                    "phone": "666-689-1234",  # false discovery
                    "email": "sjohnson@example.com",  # false alarm
                },
            },
            "pets": [
                {
                    "petId": 3501,
                    "name": "Max",
                    "species": "Dog",
                    "breed": "Golden Retriever",
                    "birthdate": "2008-05-12",  # false discovery
                    "weight": 68.5,  # false alarm
                },
                {
                    "petId": 3512,
                    "name": "Buttons",
                    # species missing - false negative
                },
            ],
        }

        # No need for evaluator - use direct compare_with method
        pass

    def test_owner_nested_structured_model(self):
        """Test that structured model fields like 'owners' are correctly matched based on nested objects."""
        # Create VeterinaryRecord objects
        gold_record = VeterinaryRecord(**self.gold_record)
        pred_record = VeterinaryRecord(**self.pred_record)

        # Use direct compare_with method instead of evaluator
        results = gold_record.compare_with(pred_record, include_confusion_matrix=True)

        # Expected metrics
        # 9 true positive: recordId, owner.id, owner.name, pets[0].petId, pets[0].name, pets[0].species, pets[0].breed, pets[1].petId, pets[1].name
        # 3 true negative: pets[1].breed, pets[1].birthdate, pets[1].weight
        # 2 false discovery: owner.contact.phone, pets[0].birthdate
        # 2 false alarm: owner.contact.email, pets[0].weight
        # 1 false negative: pets[1].species

        # Confusion matrix metrics
        cm = results["confusion_matrix"]
        # field-level confusion metrix values - use "overall" key for primitive fields
        assert (
            cm["fields"]["recordId"]["overall"]["tp"] == 1
        ), "Expected 1 true positives"
        assert (
            cm["fields"]["recordId"]["overall"]["fd"] == 0
        ), "Expected 0 false discovery"
        assert (
            cm["fields"]["recordId"]["overall"]["fa"] == 0
        ), "Expected 0 false alarm"
        assert (
            cm["fields"]["recordId"]["overall"]["fn"] == 0
        ), "Expected 0 false negatives"
        assert (
            cm["fields"]["recordId"]["overall"]["tn"] == 0
        ), "Expected 0 true negatives"

        # The id field metrics are inside the "overall" key
        assert (
            cm["fields"]["owner"]["fields"]["id"]["overall"]["tp"] == 1
        ), "Expected 1 true positives"
        assert (
            cm["fields"]["owner"]["fields"]["id"]["overall"]["fd"] == 0
        ), "Expected 0 false discovery"
        assert (
            cm["fields"]["owner"]["fields"]["id"]["overall"]["fa"] == 0
        ), "Expected 0 false alarm"
        assert (
            cm["fields"]["owner"]["fields"]["id"]["overall"]["fn"] == 0
        ), "Expected 0 false negatives"
        assert (
            cm["fields"]["owner"]["fields"]["id"]["overall"]["tn"] == 0
        ), "Expected 0 true negatives"

        assert (
            cm["fields"]["owner"]["fields"]["name"]["overall"]["tp"] == 1
        ), "Expected 1 true positives"
        assert (
            cm["fields"]["owner"]["fields"]["name"]["overall"]["fd"] == 0
        ), "Expected 0 false discovery"
        assert (
            cm["fields"]["owner"]["fields"]["name"]["overall"]["fa"] == 0
        ), "Expected 0 false alarm"
        assert (
            cm["fields"]["owner"]["fields"]["name"]["overall"]["fn"] == 0
        ), "Expected 0 false negatives"
        assert (
            cm["fields"]["owner"]["fields"]["name"]["overall"]["tn"] == 0
        ), "Expected 0 true negatives"

        # Contact field metrics are in the nested fields
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["fields"]["phone"]["overall"][
                "tp"
            ]
            == 0
        ), "Expected 0 true positives"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["fields"]["phone"]["overall"][
                "fd"
            ]
            == 1
        ), "Expected 1 false discovery"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["fields"]["phone"]["overall"][
                "fa"
            ]
            == 0
        ), "Expected 0 false alarm"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["fields"]["phone"]["overall"][
                "fn"
            ]
            == 0
        ), "Expected 0 false negatives"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["fields"]["phone"]["overall"][
                "tn"
            ]
            == 0
        ), "Expected 0 true negatives"

        assert (
            cm["fields"]["owner"]["fields"]["contact"]["fields"]["email"]["overall"][
                "tp"
            ]
            == 0
        ), "Expected 0 true positives"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["fields"]["email"]["overall"][
                "fd"
            ]
            == 0
        ), "Expected 0 false discovery"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["fields"]["email"]["overall"][
                "fa"
            ]
            == 1
        ), "Expected 1 false alarm"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["fields"]["email"]["overall"][
                "fn"
            ]
            == 0
        ), "Expected 0 false negatives"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["fields"]["email"]["overall"][
                "tn"
            ]
            == 0
        ), "Expected 0 true negatives"

        # Contact overall metrics are also in an "overall" key
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["overall"]["tp"] == 0
        ), "Expected 0 true positives"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["overall"]["fd"] == 1
        ), "Expected 1 false discovery"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["overall"]["fa"] == 0
        ), "Expected 0 false alarm (object-level counting - both GT and Pred have contact objects)"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["overall"]["fn"] == 0
        ), "Expected 0 false negatives"
        assert (
            cm["fields"]["owner"]["fields"]["contact"]["overall"]["tn"] == 0
        ), "Expected 0 true negatives"

        # Owner overall metrics
        assert (
            cm["fields"]["owner"]["overall"]["tp"] == 0
        ), "Expected 0 true positives (object-level counting - owner object similarity below threshold due to contact differences)"
        assert (
            cm["fields"]["owner"]["overall"]["fd"] == 1
        ), "Expected 1 false discovery (object-level counting - owner objects present but don't match)"
        assert (
            cm["fields"]["owner"]["overall"]["fa"] == 0
        ), "Expected 0 false alarm (object-level counting - both GT and Pred have owner objects)"
        assert (
            cm["fields"]["owner"]["overall"]["fn"] == 0
        ), "Expected 0 false negatives"
        assert (
            cm["fields"]["owner"]["overall"]["tn"] == 0
        ), "Expected 0 true negatives"

    def test_pets_list_of_structured_model(self):
        """Test that list fields like 'pets' are correctly matched based on nested objects."""
        gold_record = VeterinaryRecord(**self.gold_record)
        pred_record = VeterinaryRecord(**self.pred_record)

        results = gold_record.compare_with(pred_record, include_confusion_matrix=True)
        cm = results["confusion_matrix"]

        assert gold_record.pets[0].compare(pred_record.pets[0]) == pytest.approx(2 / 3)
        assert gold_record.pets[1].compare(pred_record.pets[1]) == pytest.approx(2 / 3)
        assert cm["fields"]["pets"]["overall"]["tp"] == 0
        assert cm["fields"]["pets"]["overall"]["fd"] == 2
        assert cm["fields"]["pets"]["overall"]["fa"] == 0
        assert cm["fields"]["pets"]["overall"]["fn"] == 0
        assert cm["fields"]["pets"]["fields"] == {}

    def test_overall_metrics(self):
        """Test correct aggregation and calculation of overall metrics."""
        # Create VeterinaryRecord objects
        gold_record = VeterinaryRecord(**self.gold_record)
        pred_record = VeterinaryRecord(**self.pred_record)

        # Use direct compare_with method instead of evaluator
        results = gold_record.compare_with(pred_record, include_confusion_matrix=True)

        cm = results["confusion_matrix"]

        # recordId is TP; owner and both pet pairs are atomic FDs.
        assert cm["overall"]["tp"] == 1
        assert cm["overall"]["tn"] == 0
        assert cm["overall"]["fd"] == 3
        assert cm["overall"]["fa"] == 0
        assert cm["overall"]["fp"] == 3
        assert cm["overall"]["fn"] == 0

        # Aggregate: TP=3, FD=3, FA=1, FN=0, TN=0.
        derived_metrics = cm["aggregate"]["derived"]
        assert derived_metrics["cm_precision"] == pytest.approx(3 / 7)
        assert derived_metrics["cm_recall"] == 1.0
        assert derived_metrics["cm_f1"] == pytest.approx(0.6)

        results_alt = gold_record.compare_with(
            pred_record, include_confusion_matrix=True, recall_with_fd=True
        )
        derived_metrics_alt = results_alt["confusion_matrix"]["aggregate"]["derived"]

        cm_alt = results_alt["confusion_matrix"]
        assert cm_alt["aggregate"]["tp"] == 3
        assert cm_alt["aggregate"]["fn"] == 0
        assert cm_alt["aggregate"]["fd"] == 3
        assert derived_metrics_alt["cm_precision"] == pytest.approx(3 / 7)
        assert derived_metrics_alt["cm_recall"] == pytest.approx(0.5)
        assert derived_metrics_alt["cm_f1"] == pytest.approx(6 / 13)
