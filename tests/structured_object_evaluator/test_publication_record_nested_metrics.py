"""Tests for compare_with() metrics calculation for publication records models.

This test verifies that we can calculate precision, recall, F1, and accuracy metrics
at the field level and object level for nested structures in publication record models.
"""

from typing import List, Optional

import pytest

from stickler.comparators.exact import ExactComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


# Define the models for the test
class Person(StructuredModel):
    Type: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    Name: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )

    match_threshold = 0.8


class BookInfo(StructuredModel):
    pages: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    author: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    weight: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )

    match_threshold = 0.8


class Reference(StructuredModel):
    Volume: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    Page: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )
    Publication: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )

    match_threshold = 0.8


class PublicationRecord(StructuredModel):
    record_id: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )

    people: Optional[List[Person]] = ComparableField(weight=1.0)

    book_info: Optional[List[BookInfo]] = ComparableField(weight=1.0)

    references: Optional[List[Reference]] = ComparableField(weight=1.0)

    State: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )

    County: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )

    Edition: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )

    Title: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )

    Date: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )

    Price: Optional[str] = ComparableField(
        comparator=ExactComparator(), threshold=1.0, weight=1.0
    )


class TestPublicationRecordsMetricsCalculation:
    """Test cases for publication records metrics calculation."""

    def setup_method(self):
        """Set up test data for nested PublicationRecord structures."""
        # Gold record based on the provided dataset
        self.gold_record = {
            "record_id": "12345",
            "people": [
                {"Type": "donator", "Name": "LOUISE MCDANIEL"},
                {"Type": "brower", "Name": "MAJOR CANO"},
                {"Type": "brower", "Name": "EGYPT SHORT"},
            ],
            "book_info": [
                {"pages": "156", "author": "HEZEKIAH ADKIN", "weight": "8.3"}
            ],
            "references": [
                {"Volume": "1", "Page": "2", "Publication": "NEWSPAPER"},
                {"Volume": "45", "Page": "50", "Publication": "BOOK REVIEW"},
                {"Volume": "12", "Page": "45", "Publication": "JOURNAL"},
            ],
            "State": "WASHINGTON",
            "County": "KING",
            "Edition": "1",
            "Title": "WHEN MOUNTAINS WALK",
            "Date": "2020-09-10",
            "Price": "$20.00",
        }

        # Predicted record with controlled errors for testing
        self.pred_record = {
            "record_id": "12345",
            "people": [
                {"Type": "donator", "Name": "LOUISE MCDANIEL"},  # exact match
                {
                    "Type": "brower",
                    "Name": "MAJOR CANOE",
                },  # false discovery (name typo)
                {"Type": "brower", "Name": "EGYPT SHORT"},  # exact match
                {
                    "Type": "donor",
                    "Name": "EXTRA PERSON",
                },  # false alarm (additional person)
            ],
            "book_info": [
                {
                    "pages": "156",
                    "author": "HEZEKIAH ADKINS",  # false discovery (typo)
                    "weight": "8.3",
                }
            ],
            "references": [
                {"Volume": "1", "Page": "2", "Publication": "NEWSPAPER"},  # exact match
                {
                    "Volume": "45",
                    "Page": "50",
                    "Publication": "BOOK REVIEW",
                },  # exact match
                {
                    "Volume": "12",
                    "Page": "46",
                    "Publication": "JOURNAL",
                },  # false discovery (page value differs)
            ],
            "State": "WASHINGTON",
            "County": "KING",
            "Edition": "1",
            # missing title (false negative)
            "Date": "2020-09-10",
            "Price": "$20.90",  # false discovery
        }

    def test_people_list_structured_model(self):
        """Test that people list is correctly matched based on nested Person objects."""
        # Create PublicationRecord objects
        gold_record = PublicationRecord(**self.gold_record)
        pred_record = PublicationRecord(**self.pred_record)

        # Evaluate
        results = gold_record.compare_with(
            pred_record, include_confusion_matrix=True, evaluator_format=True
        )

        # Confusion matrix metrics
        cm = results["confusion_matrix"]

        # Expected metrics for people
        # 2 TP: people[0], people[2]
        # 1 FD: people[1]
        # 1 FA: people[3]
        # 0 FN
        # 0 TN

        assert cm["fields"]["people"]["tp"] == 2, "Expected 2 TP (object-level)"
        assert cm["fields"]["people"]["fd"] == 1, "Expected 1 FD (object-level)"
        assert cm["fields"]["people"]["fa"] == 1, "Expected 1 FA (object-level)"
        assert cm["fields"]["people"]["fn"] == 0, "Expected 0 FN (object-level)"
        assert cm["fields"]["people"]["tn"] == 0, "Expected 0 TN (object-level)"

        # # Check nested field metrics within people
        # # Type field: TP=3, FA=1
        # assert cm["fields"]['people']["fields"]["Type"]["tp"] == 3, "Expected 3 TP for Type field"
        # assert cm["fields"]['people']["fields"]["Type"]["fd"] == 0, "Expected 0 FD for Type field"
        # assert cm["fields"]['people']["fields"]["Type"]["fa"] == 1, "Expected 1 FA for Type field"
        # assert cm["fields"]['people']["fields"]["Type"]["fn"] == 0, "Expected 0 FN for Type field"
        # assert cm["fields"]['people']["fields"]["Type"]["tn"] == 0, "Expected 0 TN for Type field"

        # # Name field: TP=2, FD=1, FA=1
        # assert cm["fields"]['people']["fields"]["Name"]["tp"] == 2, "Expected 2 TP for Name field"
        # assert cm["fields"]['people']["fields"]["Name"]["fd"] == 1, "Expected 1 FD for Name field"
        # assert cm["fields"]['people']["fields"]["Name"]["fa"] == 1, "Expected 1 FA for Name field"
        # assert cm["fields"]['people']["fields"]["Name"]["fn"] == 0, "Expected 0 FN for Name field"
        # assert cm["fields"]['people']["fields"]["Name"]["tn"] == 0, "Expected 0 TN for Name field"

    def test_book_info_list_structured_model(self):
        """Test that book_info list is correctly matched based on nested BookInfo objects."""
        # Create PublicationRecord objects
        gold_record = PublicationRecord(**self.gold_record)
        pred_record = PublicationRecord(**self.pred_record)

        # Evaluate
        results = gold_record.compare_with(
            pred_record, include_confusion_matrix=True, evaluator_format=True
        )

        # Confusion matrix metrics
        cm = results["confusion_matrix"]

        # Expected metrics for book_info
        # 0 TP
        # 1 FD: book_info[0]
        # 0 FA
        # 0 FN
        # 0 TN

        # Check book_info field metrics: FD=1
        assert cm["fields"]["book_info"]["tp"] == 0, "Expected 0 TP"
        assert cm["fields"]["book_info"]["fd"] == 1, "Expected 1 FD"
        assert cm["fields"]["book_info"]["fa"] == 0, "Expected 0 FA"
        assert cm["fields"]["book_info"]["fn"] == 0, "Expected 0 FN"
        assert cm["fields"]["book_info"]["tn"] == 0, "Expected 0 TN"

        # Check nested field metrics within book_info
        # # page field: TP=1
        # assert cm["fields"]['book_info']["fields"]["pages"]['overall']["tp"] == 1, "Expected 1 TP for pages field"
        # assert cm["fields"]['book_info']["fields"]["pages"]['overall']["fd"] == 0, "Expected 0 FD for pages field"
        # assert cm["fields"]['book_info']["fields"]["pages"]['overall']["fa"] == 0, "Expected 0 FA for pages field"
        # assert cm["fields"]['book_info']["fields"]["pages"]['overall']["fn"] == 0, "Expected 0 FN for pages field"
        # assert cm["fields"]['book_info']["fields"]["pages"]['overall']["tn"] == 0, "Expected 0 TN for pages field"

        # # author field: FD=1
        # assert cm["fields"]['book_info']["fields"]["author"]['overall']["tp"] == 0, "Expected 0 TP for author field"
        # assert cm["fields"]['book_info']["fields"]["author"]['overall']["fd"] == 1, "Expected 1 FD for author field"
        # assert cm["fields"]['book_info']["fields"]["author"]['overall']["fa"] == 0, "Expected 0 FA for author field"
        # assert cm["fields"]['book_info']["fields"]["author"]['overall']["fn"] == 0, "Expected 0 FN for author field"
        # assert cm["fields"]['book_info']["fields"]["author"]['overall']["tn"] == 0, "Expected 0 TN for author field"

        # # weight field: TP=1
        # assert cm["fields"]['book_info']["fields"]["weight"]['overall']["tp"] == 1, "Expected 1 TP for weight field"
        # assert cm["fields"]['book_info']["fields"]["weight"]['overall']["fd"] == 0, "Expected 0 FD for weight field"
        # assert cm["fields"]['book_info']["fields"]["weight"]['overall']["fa"] == 0, "Expected 0 FA for author field"
        # assert cm["fields"]['book_info']["fields"]["weight"]['overall']["fn"] == 0, "Expected 0 FN for author field"
        # assert cm["fields"]['book_info']["fields"]["weight"]['overall']["tn"] == 0, "Expected 0 TN for author field"

    def test_references_list_structured_model(self):
        """Test that references list is correctly matched based on nested Reference objects."""
        # Create PublicationRecord objects
        gold_record = PublicationRecord(**self.gold_record)
        pred_record = PublicationRecord(**self.pred_record)

        # Evaluate
        results = gold_record.compare_with(
            pred_record, include_confusion_matrix=True, evaluator_format=True
        )

        # Confusion matrix metrics
        cm = results["confusion_matrix"]

        # Expected metrics for references
        # 2 TP: references[0], references[1]
        # 1 FD: references[2]
        # 0 FA
        # 0 FN
        # 0 TN

        # Check references field metrics: TP=2, FD=1,
        assert cm["fields"]["references"]["tp"] == 2, "Expected 2 TP"
        assert cm["fields"]["references"]["fd"] == 1, "Expected 1 FD"
        assert cm["fields"]["references"]["fa"] == 0, "Expected 0 FA"
        assert cm["fields"]["references"]["fn"] == 0, "Expected 0 FN"
        assert cm["fields"]["references"]["tn"] == 0, "Expected 0 TN"

        # # Check nested field metrics within references
        # # Volumn field: TP=3
        # assert cm["fields"]['references']["fields"]["Volume"]["tp"] == 3, "Expected 3 TP for Volume field"
        # assert cm["fields"]['references']["fields"]["Volume"]["fd"] == 0, "Expected 0 FD for Volume field"
        # assert cm["fields"]['references']["fields"]["Volume"]["fa"] == 0, "Expected 0 FA for Volume field"
        # assert cm["fields"]['references']["fields"]["Volume"]["fn"] == 0, "Expected 0 FN for Volume field"
        # assert cm["fields"]['references']["fields"]["Volume"]["tn"] == 0, "Expected 0 TN for Volume field"

        # # Page field: TP=2, FD=1
        # assert cm["fields"]['references']["fields"]["Page"]["tp"] == 2, "Expected 2 TP for Page field"
        # assert cm["fields"]['references']["fields"]["Page"]["fd"] == 1, "Expected 1 FD for Page field"
        # assert cm["fields"]['references']["fields"]["Page"]["fa"] == 0, "Expected 0 FA for Page field"
        # assert cm["fields"]['references']["fields"]["Page"]["fn"] == 0, "Expected 0 FN for Page field"
        # assert cm["fields"]['references']["fields"]["Page"]["tn"] == 0, "Expected 0 TN for Page field"

        # # Publication field: TP=3
        # assert cm["fields"]['references']["fields"]["Publication"]["tp"] == 3, "Expected 3 TP for Publication field"
        # assert cm["fields"]['references']["fields"]["Publication"]["fa"] == 0, "Expected 0 FA for Publication field"
        # assert cm["fields"]['references']["fields"]["Publication"]["fa"] == 0, "Expected 0 FA for Publication field"
        # assert cm["fields"]['references']["fields"]["Publication"]["fn"] == 0, "Expected 0 FN for Publication field"
        # assert cm["fields"]['references']["fields"]["Publication"]["tn"] == 0, "Expected 0 TN for Publication field"

    def test_primitive_fields(self):
        """Test that primitive (non-list) fields are correctly evaluated."""
        # Create PublicationRecord objects
        gold_record = PublicationRecord(**self.gold_record)
        pred_record = PublicationRecord(**self.pred_record)

        # Evaluate
        results = gold_record.compare_with(
            pred_record, include_confusion_matrix=True, evaluator_format=True
        )

        # Confusion matrix metrics
        cm = results["confusion_matrix"]

        # Check exact match fields
        # 5 TP: record_id, State, County, Edition, Date
        assert cm["fields"]["record_id"]["tp"] == 1, "Expected 1 TP for record_id"
        assert cm["fields"]["record_id"]["fd"] == 0, "Expected 0 TP for record_id"
        assert cm["fields"]["record_id"]["fa"] == 0, "Expected 0 TP for record_id"
        assert cm["fields"]["record_id"]["fn"] == 0, "Expected 0 TP for record_id"
        assert cm["fields"]["record_id"]["tn"] == 0, "Expected 0 TP for record_id"

        assert cm["fields"]["State"]["tp"] == 1, "Expected 1 TP for State"
        assert cm["fields"]["State"]["fd"] == 0, "Expected 0 TP for State"
        assert cm["fields"]["State"]["fa"] == 0, "Expected 0 TP for State"
        assert cm["fields"]["State"]["fn"] == 0, "Expected 0 TP for State"
        assert cm["fields"]["State"]["tn"] == 0, "Expected 0 TP for State"

        assert cm["fields"]["County"]["tp"] == 1, "Expected 1 TP for County"
        assert cm["fields"]["County"]["fd"] == 0, "Expected 0 TP for County"
        assert cm["fields"]["County"]["fa"] == 0, "Expected 0 TP for County"
        assert cm["fields"]["County"]["fn"] == 0, "Expected 0 TP for County"
        assert cm["fields"]["County"]["tn"] == 0, "Expected 0 TP for County"

        assert cm["fields"]["Edition"]["tp"] == 1, "Expected 1 TP for Edition"
        assert cm["fields"]["Edition"]["fd"] == 0, "Expected 0 TP for Edition"
        assert cm["fields"]["Edition"]["fa"] == 0, "Expected 0 TP for Edition"
        assert cm["fields"]["Edition"]["fn"] == 0, "Expected 0 TP for Edition"
        assert cm["fields"]["Edition"]["tn"] == 0, "Expected 0 TP for Edition"

        assert cm["fields"]["Date"]["tp"] == 1, "Expected 1 TP for Date"
        assert cm["fields"]["Date"]["fd"] == 0, "Expected 0 TP for Date"
        assert cm["fields"]["Date"]["fa"] == 0, "Expected 0 TP for Date"
        assert cm["fields"]["Date"]["fn"] == 0, "Expected 0 TP for Date"
        assert cm["fields"]["Date"]["tn"] == 0, "Expected 0 TP for Date"

        # Check Price field: FD=1
        assert cm["fields"]["Price"]["tp"] == 0, "Expected 0 TP for Price"
        assert cm["fields"]["Price"]["fd"] == 1, "Expected 1 TP for Price"
        assert cm["fields"]["Price"]["fa"] == 0, "Expected 0 TP for Price"
        assert cm["fields"]["Price"]["fn"] == 0, "Expected 0 TP for Price"
        assert cm["fields"]["Price"]["tn"] == 0, "Expected 0 TP for Price"

        # Check Title field: FN=1
        assert cm["fields"]["Title"]["tp"] == 0, "Expected 0 TP for Title"
        assert cm["fields"]["Title"]["fd"] == 0, "Expected 0 FD for Title"
        assert cm["fields"]["Title"]["fa"] == 0, "Expected 0 FA for Title"
        assert cm["fields"]["Title"]["fn"] == 1, "Expected 1 FN for Title"
        assert cm["fields"]["Title"]["tn"] == 0, "Expected 0 TN for Title"

    def test_overall_metrics(self):
        """Test correct aggregation and calculation of overall metrics."""
        # Create PublicationRecord objects
        gold_record = PublicationRecord(**self.gold_record)
        pred_record = PublicationRecord(**self.pred_record)

        # Evaluate
        results = gold_record.compare_with(
            pred_record, include_confusion_matrix=True, evaluator_format=True
        )

        # Confusion matrix metrics
        cm = results["confusion_matrix"]

        # Expected overall metrics
        # 9 TP: record_id, people[0], people[2], references[0], references[1], State, Country, Edition, Date
        # 4 FD: people[1], book_info[0], references[2], Price
        # 1 FA: people[3]
        # 1 FN: Title
        # 0 TN

        assert cm["overall"]["tp"] == 9, "Expected 9 TP"
        assert cm["overall"]["fd"] == 4, "Expected 4 FD"
        assert cm["overall"]["fa"] == 1, "Expected 1 FA"
        assert cm["overall"]["fn"] == 1, "Expected 1 FN"
        assert cm["overall"]["tn"] == 0, "Expected 0 TN"

        # Check overall metrics with object-level counting
        # Expected precision = TP/(TP+FD+FA) = 9/(9+4+1) = 0.64
        # Expected recall option 1 = TP/(TP+FN) = 9/(9+1) = 0.9
        # Expected F1 with recall option 1 = 2*0.64*0.9/(0.64+0.9) = 0.75
        assert results["overall"]["precision"] == pytest.approx(0.64, abs=0.01)
        assert results["overall"]["recall"] == pytest.approx(0.9, abs=0.01)
        assert results["overall"]["f1"] == pytest.approx(0.75, abs=0.01)

        # Test with alternative recall formula (including FD in denominator)
        results_alt = gold_record.compare_with(
            pred_record,
            include_confusion_matrix=True,
            evaluator_format=True,
            recall_with_fd=True,
        )
        # Test with alternative recall formula
        # Expected recall option 2 = TP/(TP+FN+FD) = 9/(9+1+4) = 0.64 with recall_with_fd=True
        # Expected F1 = 2*precision*recall/(precision+recall) = 2*0.64*0.64/(0.64+0.64) = 0.64
        assert results["overall"]["precision"] == pytest.approx(0.64, abs=0.01)
        assert results_alt["overall"]["recall"] == pytest.approx(0.64, abs=0.01)
        assert results_alt["overall"]["f1"] == pytest.approx(0.64, abs=0.01)
