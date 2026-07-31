"""
Rich Value handling for StructuredModel.

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

Deprecation window:
    The pre-rename ``{"value": ..., "confidence": ...}`` shape is still
    accepted for one release. Both ``"value"`` AND ``"confidence"`` keys
    must be present (and ``"_value"`` absent) for the shim to fire — a
    plain dict like ``{"currency": "USD", "value": 100}`` is user data,
    not a rich value, and is passed through verbatim. When the shim does
    fire, a DeprecationWarning is emitted naming the field path and the
    dict is unwrapped the same way as the new ``_value``/``_confidence``
    form. The legacy shape will be removed in 0.5.0.

See the Rich Value Pattern proposal for design rationale.
"""

import warnings
from typing import Any, Dict, Tuple

from stickler.utils.deprecation import warn_once

# The key that marks a dict as a rich value wrapper
VALUE_KEY = "_value"

# Known metadata keys (underscore-prefixed). As new metadata types are
# supported, add them here so they get extracted into typed accessors.
CONFIDENCE_KEY = "_confidence"


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


def _is_legacy_rich_value(data: Any) -> bool:
    """Check if a dict uses the pre-underscore {"value", "confidence"} shape.

    Supports a one-release deprecation window so existing JSONL corpora
    continue to have their confidence scores extracted. Callers should
    emit a DeprecationWarning before treating a legacy shape as rich.

    Both ``"value"`` AND ``"confidence"`` are required for a dict to be
    treated as a legacy rich value. A value-only dict like
    ``{"currency": "USD", "value": 100}`` is ordinary user data and is
    not unwrapped — auto-unwrapping such dicts would silently discard
    every sibling key while emitting a misleading deprecation warning
    accusing the user of using a deprecated shape.
    """
    return (
        isinstance(data, dict)
        and "_value" not in data
        and "value" in data
        and "confidence" in data
    )


def process_rich_values(
    data: Any,
    field_path: str = "",
    max_depth: int = 64,
    _depth: int = 0,
) -> Tuple[Any, Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Recursively unwrap rich values, extracting values, confidence, and extras.

    Walks the JSON data tree. When a rich value is found, extracts:
    - "_value" into the model field
    - "_confidence" into the confidences dict
    - All other keys into the extras dict

    Also recognizes the deprecated ``{"value", "confidence"}`` shape
    for one release. When that shape is encountered a
    ``DeprecationWarning`` is emitted naming the field path, and the
    dict is unwrapped the same way as the underscore-prefixed form.
    Only the ``value`` and ``confidence`` keys are honored in the
    legacy path; no extras are collected.

    Args:
        data: The JSON data to process.
        field_path: Dot/bracket-notation path for the current position.
        max_depth: Maximum nesting depth before raising ``ValueError``.
            Guards against pathological inputs blowing the recursion
            stack. The default of 64 is well past anything a real
            structured model would need (the deepest existing test is
            six levels). Increase only if you have a documented need.

    Returns:
        Tuple of (unwrapped_data, confidences_dict, extras_dict).
        unwrapped_data has rich values replaced with their plain values.
        confidences_dict maps field paths to confidence scores.
        extras_dict maps field paths to dicts of extra metadata.

    Raises:
        ValueError: When the data tree exceeds ``max_depth`` levels.
    """
    if _depth > max_depth:
        raise ValueError(
            f"Rich value tree exceeds max_depth={max_depth} at "
            f"'{field_path}'"
        )
    if isinstance(data, dict):
        if _is_rich_value(data):
            value = data["_value"]
            confidences: Dict[str, float] = {}
            extras: Dict[str, Dict[str, Any]] = {}

            if CONFIDENCE_KEY in data:
                confidences[field_path] = data[CONFIDENCE_KEY]

            # Collect all non-system keys as user extras.
            # Warn on non-underscore-prefixed keys since all rich value
            # metadata should use the underscore convention.
            field_extras = {}
            for k, v in data.items():
                if k == "_value" or k == CONFIDENCE_KEY:
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
        elif _is_legacy_rich_value(data):
            # Remove in 0.5.0.
            warn_once(
                "legacy_rich_value_shape",
                field_path,
                f"Field '{field_path}' uses the legacy "
                f"{{'value', 'confidence'}} rich value shape. Rename "
                f"these keys to '_value' and '_confidence'. Support "
                f"for the legacy shape will be removed in 0.5.0.",
            )
            return data["value"], {field_path: data["confidence"]}, {}
        else:
            processed = {}
            all_confidences: Dict[str, float] = {}
            all_extras: Dict[str, Dict[str, Any]] = {}
            for key, value in data.items():
                new_path = f"{field_path}.{key}" if field_path else key
                processed_value, confidences, extras = process_rich_values(
                    value, new_path, max_depth=max_depth, _depth=_depth + 1
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
            processed_item, confidences, extras = process_rich_values(
                item, item_path, max_depth=max_depth, _depth=_depth + 1
            )
            processed_list.append(processed_item)
            all_confidences.update(confidences)
            all_extras.update(extras)
        return processed_list, all_confidences, all_extras
    else:
        return data, {}, {}
