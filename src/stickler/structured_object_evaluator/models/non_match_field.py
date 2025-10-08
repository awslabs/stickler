"""Models for documenting non-matches in structured object evaluation.

This module provides data models for documenting and tracking non-matches
(false positives, false negatives, etc.) during structured object evaluation.
It also includes utilities for filtering, exporting, and analyzing non-matches.
"""

from enum import Enum
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NonMatchType(str, Enum):
    """Enum defining the types of non-matches."""

    FALSE_ALARM = "false_alarm"  # GT null, prediction non-null
    FALSE_DISCOVERY = "false_discovery"  # Both non-null but don't match
    FALSE_NEGATIVE = "false_negative"  # GT non-null, prediction null


class NonMatchField(BaseModel):
    """Model for documenting non-matches in structured object evaluation.

    This class stores detailed information about each non-match detected
    during the evaluation process, enabling more thorough analysis and
    debugging of evaluation results.
    """

    field_path: str = Field(
        description="Dot-notation path to the field (e.g., 'address.city')"
    )
    non_match_type: NonMatchType = Field(description="Type of non-match")
    ground_truth_value: Any = Field(description="Original ground truth value")
    prediction_value: Any = Field(description="Predicted value")
    similarity_score: Optional[float] = Field(
        default=None, description="Similarity score if available"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context or details"
    )
    document_id: Optional[str] = Field(
        default=None, description="ID of the document this non-match belongs to"
    )

    def __str__(self) -> str:
        """Return a string representation of the non-match document."""
        similarity_str = (
            f", similarity: {self.similarity_score:.4f}"
            if self.similarity_score is not None
            else ""
        )
        doc_id_str = f" (doc: {self.document_id})" if self.document_id else ""
        return (
            f"{self.non_match_type.value.upper()} at '{self.field_path}'{similarity_str}{doc_id_str}\n"
            f"  GT: {self.ground_truth_value}\n"
            f"  Pred: {self.prediction_value}"
        )

    @staticmethod
    def filter_by_type(
        documents: List["NonMatchField"], match_type: NonMatchType
    ) -> List["NonMatchField"]:
        """
        Filter non-match documents by their type.

        Args:
            documents: List of NonMatchField instances to filter
            match_type: Type of non-match to filter for

        Returns:
            Filtered list of NonMatchField instances
        """
        return [doc for doc in documents if doc.non_match_type == match_type]

    @staticmethod
    def export_to_dict(
        documents: List["NonMatchField"],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Export a list of non-match documents to a dictionary for serialization.

        Args:
            documents: List of NonMatchField instances

        Returns:
            Dictionary with categorized non-matches
        """
        result = {"false_alarms": [], "false_discoveries": [], "false_negatives": []}

        for doc in documents:
            # Create a simplified entry
            entry = {
                "field_path": doc.field_path,
                "ground_truth": str(doc.ground_truth_value),
                "prediction": str(doc.prediction_value),
            }

            if doc.similarity_score is not None:
                entry["similarity_score"] = doc.similarity_score

            if doc.details:
                entry["details"] = doc.details

            if doc.non_match_type == NonMatchType.FALSE_ALARM:
                result["false_alarms"].append(entry)
            elif doc.non_match_type == NonMatchType.FALSE_DISCOVERY:
                result["false_discoveries"].append(entry)
            elif doc.non_match_type == NonMatchType.FALSE_NEGATIVE:
                result["false_negatives"].append(entry)

        return result

    @staticmethod
    def export_to_json(documents: List["NonMatchField"], output_path: str):
        """
        Export a list of non-match documents to a JSON file.

        Args:
            documents: List of NonMatchField instances
            output_path: Path to save the JSON file
        """
        # Create parent directories if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Export as dictionary
        data = NonMatchField.export_to_dict(documents)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def print_summary(documents: List["NonMatchField"], detailed: bool = False):
        """
        Print a summary of non-match documents.

        Args:
            documents: List of NonMatchField instances
            detailed: Whether to print detailed information for each document
        """
        # Count by type
        false_alarms = NonMatchField.filter_by_type(documents, NonMatchType.FALSE_ALARM)
        false_discoveries = NonMatchField.filter_by_type(
            documents, NonMatchType.FALSE_DISCOVERY
        )
        false_negatives = NonMatchField.filter_by_type(
            documents, NonMatchType.FALSE_NEGATIVE
        )

        # Print summary counts
        print(f"Non-matches summary:")
        print(f"- False Alarms: {len(false_alarms)}")
        print(f"- False Discoveries: {len(false_discoveries)}")
        print(f"- False Negatives: {len(false_negatives)}")

        # Print details if requested
        if detailed and documents:
            print("\nDetailed non-matches:")
            for i, doc in enumerate(documents):
                print(f"\nNon-match #{i + 1}:")
                print(f"- Type: {doc.non_match_type}")
                print(f"- Field: {doc.field_path}")
                print(f"- Ground truth: {doc.ground_truth_value}")
                print(f"- Prediction: {doc.prediction_value}")
                if doc.similarity_score is not None:
                    print(f"- Similarity: {doc.similarity_score:.4f}")
