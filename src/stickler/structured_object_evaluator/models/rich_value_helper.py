"""
Rich Value Helper for StructuredModel.

Handles the "Rich Value Pattern": a JSON convention where fields can carry
metadata alongside their actual values. A plain value like "Widget" becomes
a rich value when wrapped as {"_value": "Widget", "_confidence": 0.95}.

All rich value keys use an underscore prefix to distinguish them from user
data. The "_value" key is the marker that identifies a rich value wrapper.
Known metadata keys (_confidence, _bbox, etc.) are extracted into typed
accessors. User-provided extras (any non-underscore-prefixed keys) are
preserved in a separate extras dict.

Convention:
    _value:       The actual field value (required, marks the wrapper)
    _confidence:  Model confidence score (float, 0.0 to 1.0)
    _bbox:        Bounding box coordinates (future)
    _source_span: Source text span (future)
    _source_text: Source text content (future)
    other keys:   User extras, stored verbatim in get_field_extras()

See the Rich Value Pattern proposal for design rationale.
"""

import warnings
from typing import Any, Dict, Tuple


class RichValueHelper:
    """Unwraps rich values during from_json(), extracting values and metadata."""

    # The key that marks a dict as a rich value wrapper
    VALUE_KEY = "_value"

    # Known metadata keys (underscore-prefixed). As new metadata types are
    # supported, add them here so they get extracted into typed accessors.
    CONFIDENCE_KEY = "_confidence"

    @staticmethod
    def _is_rich_value(data: Any) -> bool:
        """Check if a dict is a rich value (has "_value" key).

        A rich value is any dict containing a "_value" key. The underscore
        prefix prevents collision with user data fields.

        Args:
            data: The value to check.

        Returns:
            True if this is a rich value structure.
        """
        return isinstance(data, dict) and "_value" in data

    @classmethod
    def process_rich_values(
        cls, data: Any, field_path: str = ""
    ) -> Tuple[Any, Dict[str, float], Dict[str, Dict[str, Any]]]:
        """Recursively unwrap rich values, extracting values, confidence, and extras.

        Walks the JSON data tree. When a rich value is found, extracts:
        - "_value" into the model field
        - "_confidence" into the confidences dict
        - All other keys into the extras dict

        Args:
            data: The JSON data to process.
            field_path: Dot/bracket-notation path for the current position.

        Returns:
            Tuple of (unwrapped_data, confidences_dict, extras_dict).
            unwrapped_data has rich values replaced with their plain values.
            confidences_dict maps field paths to confidence scores.
            extras_dict maps field paths to dicts of extra metadata.
        """
        if isinstance(data, dict):
            if cls._is_rich_value(data):
                value = data["_value"]
                confidences: Dict[str, float] = {}
                extras: Dict[str, Dict[str, Any]] = {}

                if cls.CONFIDENCE_KEY in data:
                    confidences[field_path] = data[cls.CONFIDENCE_KEY]

                # Collect all non-system keys as user extras.
                # Warn on non-underscore-prefixed keys since all rich value
                # metadata should use the underscore convention.
                field_extras = {}
                for k, v in data.items():
                    if k == "_value" or k == cls.CONFIDENCE_KEY:
                        continue
                    if not k.startswith("_"):
                        warnings.warn(
                            f"Non-prefixed key '{k}' found inside rich value "
                            f"for field '{field_path}'. All rich value metadata "
                            f"keys should use underscore prefix (e.g., '_{k}'). "
                            f"This key will be stored in extras but may not be "
                            f"processed correctly by future features.",
                            UserWarning,
                            stacklevel=2,
                        )
                    field_extras[k] = v
                if field_extras:
                    extras[field_path] = field_extras

                return value, confidences, extras
            else:
                processed = {}
                all_confidences: Dict[str, float] = {}
                all_extras: Dict[str, Dict[str, Any]] = {}
                for key, value in data.items():
                    new_path = f"{field_path}.{key}" if field_path else key
                    processed_value, confidences, extras = cls.process_rich_values(
                        value, new_path
                    )
                    processed[key] = processed_value
                    all_confidences.update(confidences)
                    all_extras.update(extras)
                return processed, all_confidences, all_extras
        elif isinstance(data, list):
            processed_list = []
            all_confidences: Dict[str, float] = {}
            all_extras: Dict[str, Dict[str, Any]] = {}
            for i, item in enumerate(data):
                item_path = f"{field_path}[{i}]"
                processed_item, confidences, extras = cls.process_rich_values(
                    item, item_path
                )
                processed_list.append(processed_item)
                all_confidences.update(confidences)
                all_extras.update(extras)
            return processed_list, all_confidences, all_extras
        else:
            return data, {}, {}
