"""
Rich Value Helper for StructuredModel.

Handles the "Rich Value Pattern": a JSON convention where fields can carry
metadata alongside their actual values. A plain value like "Widget" becomes
a rich value when wrapped as {"value": "Widget", "confidence": 0.95}.

The helper unwraps rich values during from_json(), extracting the value for
the model field and storing metadata (currently confidence scores) separately.
The pattern is extensible to other metadata types (bounding boxes, source
spans, etc.) without changing the unwrapping logic.

See the Rich Value Pattern proposal for design rationale.
"""

from typing import Any, Dict

from pydantic import BaseModel


class RichValueHelper(BaseModel):
    @staticmethod
    def _is_rich_value(data: Dict) -> bool:
        """Check if a dict is a rich value (has "value" key plus metadata).

        A rich value is any dict containing a "value" key alongside one or
        more metadata keys. The metadata keys are optional and can include
        "confidence", "bbox", "source_span", etc.

        Args:
            data: The dict to check.

        Returns:
            True if this is a rich value structure.
        """
        if not isinstance(data, dict) or "value" not in data:
            return False
        # Must have at least one key besides "value" to be a rich value.
        # A dict with only {"value": ...} is ambiguous, but we treat it
        # as a rich value to support the case where the user wraps a value
        # without any metadata yet.
        return True

    @classmethod
    def process_rich_values(cls, data: Any, field_path: str = "") -> tuple:
        """Recursively unwrap rich values, extracting values and metadata.

        Walks the JSON data tree. When a rich value is found, extracts the
        "value" for the model field and stores the "confidence" in a
        separate dict keyed by field path.

        Args:
            data: The JSON data to process.
            field_path: Dot/bracket-notation path for the current position.

        Returns:
            Tuple of (unwrapped_data, confidences_dict).
            unwrapped_data has rich values replaced with their plain values.
            confidences_dict maps field paths to confidence scores.
        """
        if isinstance(data, dict):
            if cls._is_rich_value(data):
                value = data["value"]
                confidences = {}
                if "confidence" in data:
                    confidences[field_path] = data["confidence"]
                return value, confidences
            else:
                processed = {}
                all_confidences = {}
                for key, value in data.items():
                    new_path = f"{field_path}.{key}" if field_path else key
                    processed_value, confidences = cls.process_rich_values(
                        value, new_path
                    )
                    processed[key] = processed_value
                    all_confidences.update(confidences)
                return processed, all_confidences
        elif isinstance(data, list):
            processed_list = []
            all_confidences = {}
            for i, item in enumerate(data):
                item_path = f"{field_path}[{i}]"
                processed_item, confidences = cls.process_rich_values(
                    item, item_path
                )
                processed_list.append(processed_item)
                all_confidences.update(confidences)
            return processed_list, all_confidences
        else:
            return data, {}
