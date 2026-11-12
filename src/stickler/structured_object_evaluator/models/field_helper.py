"""Field operations helper for StructuredModel comparisons."""

from typing import Any, List, Type

from .null_helper import NullHelper


class FieldHelper:
    """Helper class for field iteration and classification patterns."""

    @staticmethod
    def get_comparable_fields(model_class: Type) -> List[str]:
        """Get list of field names that should be compared (excluding extra_fields).

        Args:
            model_class: The StructuredModel class to get fields from

        Returns:
            List of field names excluding 'extra_fields'
        """
        fields = []
        for field_name in model_class.model_fields:
            if field_name != "extra_fields":
                fields.append(field_name)
        return fields

    @staticmethod
    def is_null_value(value: Any) -> bool:
        """Determine if a value should be considered null or empty.

        Args:
            value: The value to check

        Returns:
            True if the value is null/empty, False otherwise
        """
        if value is None or NullHelper.is_nan(value):
            return True
        elif isinstance(value, list):
            # A list of only None elements is empty (#386).
            return NullHelper.is_effectively_null_for_lists(value)
        elif hasattr(value, "__len__") and not isinstance(
            value, (str, bytes, bytearray)
        ):
            # Consider empty lists/collections as null values
            return len(value) == 0
        elif isinstance(value, (str, bytes, bytearray)):
            return len(value.strip()) == 0
        return False

    @staticmethod
    def is_primitive_field(self_value: Any, other_value: Any) -> bool:
        """Check if a field should be treated as primitive (leaf node).

        Args:
            self_value: Ground truth value
            other_value: Prediction value

        Returns:
            True if this is a primitive field, False if it has structure
        """
        # Primitive types: strings, numbers, booleans, None
        primitive_types = (str, int, float, bool, type(None))

        # Check if both values are primitive types
        if isinstance(self_value, primitive_types) and isinstance(
            other_value, primitive_types
        ):
            return True

        # Empty lists are treated as primitive
        if isinstance(self_value, list) and len(self_value) == 0:
            return True
        if isinstance(other_value, list) and len(other_value) == 0:
            return True

        # Lists of primitives are treated as primitive fields
        if (
            isinstance(self_value, list)
            and self_value
            and isinstance(self_value[0], primitive_types)
        ):
            return True

        return False

    @staticmethod
    def get_field_value_safely(obj: Any, field_name: str, default: Any = None) -> Any:
        """Get field value from object safely using getattr.

        Args:
            obj: Object to get field from
            field_name: Name of the field
            default: Default value if field doesn't exist

        Returns:
            Field value or default if field doesn't exist
        """
        return getattr(obj, field_name, default)
