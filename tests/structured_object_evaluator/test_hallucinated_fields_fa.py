

"""Test handling of hallucinated fields (extra_fields) as False Alarms."""

from typing import Any, ClassVar, List

from pydantic import BaseModel

from stickler.comparators.exact import ExactComparator
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.structured_object_evaluator.bulk_structured_model_evaluator import (
    BulkStructuredModelEvaluator,
)
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel


class SimpleContract(StructuredModel):
    """Simple contract model with only essential fields."""

    match_threshold: ClassVar[float] = 0.7

    date: str = ComparableField(comparator=ExactComparator(), threshold=1.0, weight=1.0)

    company_name: str = ComparableField(
        comparator=LevenshteinComparator(), threshold=0.8, weight=2.0
    )


class TestHallucinatedFieldsFA:
    """Test that hallucinated fields are properly counted as False Alarms."""

    def test_single_model_with_hallucinated_fields(self):
        """Test individual model comparison with hallucinated fields."""

        # Ground truth: clean data with only schema fields
        gt_data = {"date": "2024-01-15", "company_name": "Acme Corp"}
        gt_model = SimpleContract(**gt_data)

        # Prediction: schema fields + hallucinated fields
        pred_data = {
            "date": "2024-01-15",
            "company_name": "Acme Corp",
            # These fields don't exist in SimpleContract schema
            "tenant_phone": "555-1234",
            "property_address": "123 Main St",
            "rent_amount": 2500.0,
        }
        pred_model = SimpleContract(**pred_data)

        print(f"\nGT __pydantic_extra__: {getattr(gt_model, '__pydantic_extra__', {})}")
        print(
            f"Pred __pydantic_extra__: {getattr(pred_model, '__pydantic_extra__', {})}"
        )

        # The hallucinated fields should be stored in __pydantic_extra__
        assert len(getattr(gt_model, "__pydantic_extra__", {})) == 0
        assert len(getattr(pred_model, "__pydantic_extra__", {})) == 3
        assert pred_model.__pydantic_extra__["tenant_phone"] == "555-1234"
        assert (
            pred_model.__pydantic_extra__["property_address"] == "123 Main St"
        )
        assert pred_model.__pydantic_extra__["rent_amount"] == 2500.0

        # Compare with confusion matrix
        result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        print(f"\nComparison result overall: {result['confusion_matrix']['overall']}")

        # BEFORE FIX: FA = 0 (hallucinated fields ignored)
        # AFTER FIX: FA = 3 (one for each hallucinated field)
        expected_fa = 3
        actual_fa = result["confusion_matrix"]["overall"]["fa"]

        print(f"Expected FA: {expected_fa}, Actual FA: {actual_fa}")
        assert (
            actual_fa == expected_fa
        ), "Hallucinated fields should be counted as False Alarms"

    def test_bulk_evaluator_with_hallucinated_fields(self):
        """Test BulkStructuredModelEvaluator with hallucinated fields."""

        # Create test data with multiple documents
        test_documents = [
            # Doc 1: 2 hallucinated fields
            {
                "gt": {"date": "2024-01-15", "company_name": "Acme Corp"},
                "pred": {
                    "date": "2024-01-15",
                    "company_name": "Acme Corp",
                    "hallucination_1": "value1",
                    "hallucination_2": "value2",
                },
            },
            # Doc 2: 3 hallucinated fields
            {
                "gt": {"date": "2024-02-20", "company_name": "Beta Inc"},
                "pred": {
                    "date": "2024-02-20",
                    "company_name": "Beta Inc",
                    "extra_field_a": "A",
                    "extra_field_b": "B",
                    "extra_field_c": "C",
                },
            },
            # Doc 3: No hallucinated fields (control)
            {
                "gt": {"date": "2024-03-10", "company_name": "Gamma LLC"},
                "pred": {"date": "2024-03-10", "company_name": "Gamma LLC"},
            },
        ]

        # Create bulk evaluator
        bulk_evaluator = BulkStructuredModelEvaluator(
            target_schema=SimpleContract, verbose=True
        )
        bulk_evaluator.reset()

        # Process each document
        for i, doc in enumerate(test_documents):
            gt_model = SimpleContract(**doc["gt"])
            pred_model = SimpleContract(**doc["pred"])

            bulk_evaluator.update(gt_model, pred_model, f"doc_{i}")

        # Get final results
        results = bulk_evaluator.compute()

        print(f"\nBulk evaluator results: {results.metrics}")

        # Expected: 2 + 3 + 0 = 5 total False Alarms from hallucinated fields
        expected_fa = 5
        actual_fa = results.metrics.get("fa", 0)

        print(f"Expected total FA: {expected_fa}, Actual FA: {actual_fa}")
        assert (
            actual_fa == expected_fa
        ), "Bulk evaluator should correctly count hallucinated fields as False Alarms"

    def test_nested_model_with_hallucinated_fields(self):
        """Test nested models with hallucinated fields."""

        class Address(StructuredModel):
            street: str = ComparableField(
                comparator=ExactComparator(), threshold=1.0, weight=1.0
            )
            city: str = ComparableField(
                comparator=ExactComparator(), threshold=1.0, weight=1.0
            )

        class Company(StructuredModel):
            name: str = ComparableField(
                comparator=ExactComparator(), threshold=1.0, weight=1.0
            )
            address: Address = ComparableField(threshold=0.8, weight=1.0)

        # Ground truth
        gt_data = {
            "name": "Acme Corp",
            "address": {"street": "123 Main St", "city": "Springfield"},
        }
        gt_model = Company(**gt_data)

        # Prediction with hallucinated fields at multiple levels
        pred_data = {
            "name": "Acme Corp",
            "phone": "555-0000",  # Hallucinated at root level
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "zipcode": "12345",  # Hallucinated in nested object
            },
            "website": "www.acme.com",  # Another hallucination at root
        }
        pred_model = Company(**pred_data)

        # Check extra_fields are captured
        print(
            f"\nRoot __pydantic_extra__: {getattr(pred_model, '__pydantic_extra__', {})}"
        )
        print(
            f"Nested __pydantic_extra__: {getattr(pred_model.address, '__pydantic_extra__', {})}"
        )

        assert (
            len(getattr(pred_model, "__pydantic_extra__", {})) == 2
        )  # phone, website
        assert (
            len(getattr(pred_model.address, "__pydantic_extra__", {})) == 1
        )  # zipcode

        # Compare and check FA counting
        result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        print(f"Nested comparison result: {result['confusion_matrix']['overall']}")

        # Should count: phone, website, zipcode = 3 FAs
        expected_fa = 3
        actual_fa = result["confusion_matrix"]["overall"]["fa"]

        print(f"Expected FA: {expected_fa}, Actual FA: {actual_fa}")
        assert (
            actual_fa == expected_fa
        ), "Hallucinated fields at all nesting levels should count as False Alarms"

    def test_list_structured_model_with_hallucinated_fields(self):
        """Test list of StructuredModel elements with hallucinated fields (#439)."""

        class ContractBatch(StructuredModel):
            contracts: list[SimpleContract] = ComparableField(weight=1.0)

        gt_data = {
            "contracts": [
                {"date": "2024-01-15", "company_name": "Acme Corp"},
                {"date": "2024-02-20", "company_name": "Beta Inc"},
            ]
        }
        gt_model = ContractBatch(**gt_data)

        pred_data = {
            "contracts": [
                {
                    "date": "2024-01-15",
                    "company_name": "Acme Corp",
                    "tenant_phone": "555-1234",
                },
                {
                    "date": "2024-02-20",
                    "company_name": "Beta Inc",
                    "property_address": "123 Main St",
                    "rent_amount": 2500.0,
                },
            ]
        }
        pred_model = ContractBatch(**pred_data)

        result = gt_model.compare_with(pred_model, include_confusion_matrix=True)

        # 1 extra field on first element + 2 extra fields on second element = 3 False Alarms
        expected_fa = 3
        actual_fa = result["confusion_matrix"]["overall"]["fa"]

        assert actual_fa == expected_fa, (
            "Hallucinated fields on list elements should count as False Alarms"
        )

    def test_unmatched_predicted_element_with_extra_fields_not_double_counted(self):
        """Test that an unmatched predicted list element is counted as 1 FA, not 1 + extra fields."""

        class Two(StructuredModel):
            a: str = ComparableField()
            b: str = ComparableField()

        class Container(StructuredModel):
            items: list[Two] = ComparableField()

        gt = Container(items=[Two(a="x", b="y")])
        pred = Container(items=[Two(a="x", b="y"), Two(a="p", b="q", z="h")])

        result = gt.compare_with(pred, include_confusion_matrix=True)
        # 0 extra fields on matched item + 1 FA for the unmatched element = 1 FA total
        assert result["confusion_matrix"]["overall"]["fa"] == 1

    def test_mixed_list_with_non_structured_models(self):
        """Test that lists with non-StructuredModel items do not crash."""

        class Scored(BaseModel):
            a: str = "x"

            def compare_with(self, other, **kw):
                return {"overall_score": 1.0}

        class Loose(StructuredModel):
            items: List[Any] = ComparableField(weight=1.0)

        v = [SimpleContract(date="2024-01-15", company_name="Acme Corp"), Scored()]
        result = Loose(items=v).compare_with(
            Loose(items=list(v)), include_confusion_matrix=True
        )
        assert result["overall_score"] == 1.0
        assert result["confusion_matrix"]["overall"]["fa"] == 0
