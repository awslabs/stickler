"""Field operations helper for StructuredModel comparisons."""

from typing import Any, List, Dict, Type, get_origin, get_args
import inspect


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
        if value is None:
            return True
        elif hasattr(value, "__len__") and not isinstance(
            value, (str, bytes, bytearray)
        ):
            # Consider empty lists/collections as null values
            return len(value) == 0
        elif isinstance(value, (str, bytes, bytearray)):
            return len(value.strip()) == 0
        return False

    @staticmethod
    def is_structured_field_type(field_info) -> bool:
        """Check if a field represents a structured type that needs special handling.

        Args:
            field_info: Pydantic field info object

        Returns:
            True if the field is a List[StructuredModel] or StructuredModel type
        """
        try:
            # Get the field annotation
            annotation = field_info.annotation

            # Handle List[SomeType] annotations
            if get_origin(annotation) is list:
                args = get_args(annotation)
                if args:
                    # Check if List element type is a StructuredModel subclass
                    element_type = args[0]
                    # Import here to avoid circular imports
                    from .structured_model import StructuredModel

                    if inspect.isclass(element_type) and issubclass(
                        element_type, StructuredModel
                    ):
                        return True

            # Handle direct StructuredModel annotations
            elif inspect.isclass(annotation):
                # Import here to avoid circular imports
                from .structured_model import StructuredModel

                if issubclass(annotation, StructuredModel):
                    return True

        except (TypeError, AttributeError):
            # If we can't determine the type, assume it's not structured
            pass

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
    def is_immediate_child(nested_path: str, field_name: str) -> bool:
        """Determines if nested_path is an immediate child of field_name.

        Args:
            nested_path: The nested path to check, e.g., 'owner.contact.phone'
            field_name: The potential parent path, e.g., 'owner.contact'

        Returns:
            True if nested_path is an immediate child of field_name, False otherwise
        """
        # Check if field_name is a prefix of nested_path
        if not nested_path.startswith(field_name):
            return False

        # If field_name is a prefix, it should be followed by a dot
        if len(field_name) >= len(nested_path):
            return False

        if nested_path[len(field_name)] != ".":
            return False

        # The remaining part after field_name and the dot should not contain any more dots
        remaining = nested_path[len(field_name) + 1 :]
        return "." not in remaining

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
