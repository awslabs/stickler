"""Type resolution system for dynamic model creation.

This module provides utilities for converting string type names to Python types,
enabling configuration-based type specification in model_from_json().
"""

import sys
from typing import Dict, Type, Any, Optional, Union, List, get_origin, get_args
import re


class TypeResolver:
    """Resolver for converting string type names to Python types."""

    def __init__(self):
        """Initialize the resolver with built-in types."""
        self._type_registry: Dict[str, Any] = {}
        self._register_builtin_types()

    def _register_builtin_types(self):
        """Register all built-in Python types."""
        # Basic types
        self._type_registry["str"] = str
        self._type_registry["int"] = int
        self._type_registry["float"] = float
        self._type_registry["bool"] = bool
        self._type_registry["bytes"] = bytes

        # Container types
        self._type_registry["list"] = list
        self._type_registry["dict"] = dict
        self._type_registry["tuple"] = tuple
        self._type_registry["set"] = set

        # Typing module types
        from typing import List, Dict, Tuple, Set, Optional, Union, Any

        self._type_registry["List"] = List
        self._type_registry["Dict"] = Dict
        self._type_registry["Tuple"] = Tuple
        self._type_registry["Set"] = Set
        self._type_registry["Optional"] = Optional
        self._type_registry["Union"] = Union
        self._type_registry["Any"] = Any

        # Structured model types (special handling)
        self._type_registry["structured_model"] = "structured_model"
        self._type_registry["list_structured_model"] = "list_structured_model"
        self._type_registry["optional_structured_model"] = "optional_structured_model"

    def register_type(self, name: str, type_class: Type) -> None:
        """Register a custom type.

        Args:
            name: String name for the type
            type_class: Type class to register

        Raises:
            ValueError: If name is already registered
        """
        if name in self._type_registry:
            raise ValueError(f"Type '{name}' is already registered")

        self._type_registry[name] = type_class

    def resolve_type_string(self, type_string: str) -> Type:
        """Resolve a string type specification to a Python type.

        Supports:
        - Basic types: "str", "int", "float", "bool"
        - Generic types: "List[str]", "Dict[str, int]", "Optional[str]"
        - Union types: "Union[str, int]"
        - Nested generics: "List[Dict[str, int]]"

        Args:
            type_string: String representation of the type

        Returns:
            Resolved Python type

        Raises:
            ValueError: If type string cannot be resolved
        """
        type_string = type_string.strip()

        # Handle simple types first
        if type_string in self._type_registry:
            return self._type_registry[type_string]

        # Handle generic types like List[str], Dict[str, int], etc.
        if "[" in type_string and "]" in type_string:
            return self._resolve_generic_type(type_string)

        # If not found, raise error with helpful message
        available_types = list(self._type_registry.keys())
        raise ValueError(
            f"Unknown type: '{type_string}'. Available types: {available_types}"
        )

    def _resolve_generic_type(self, type_string: str) -> Type:
        """Resolve generic type specifications like List[str], Dict[str, int].

        Args:
            type_string: Generic type string like "List[str]"

        Returns:
            Resolved generic type

        Raises:
            ValueError: If generic type cannot be resolved
        """
        # Parse the generic type structure
        match = re.match(r"^([^[]+)\[(.+)\]$", type_string)
        if not match:
            raise ValueError(f"Invalid generic type format: '{type_string}'")

        base_type_name = match.group(1).strip()
        args_string = match.group(2).strip()

        # Get the base type
        if base_type_name not in self._type_registry:
            raise ValueError(f"Unknown base type: '{base_type_name}'")

        base_type = self._type_registry[base_type_name]

        # Parse the type arguments
        type_args = self._parse_type_arguments(args_string)

        # Resolve each argument
        resolved_args = []
        for arg in type_args:
            resolved_args.append(self.resolve_type_string(arg))

        # Create the generic type
        try:
            if len(resolved_args) == 1:
                return base_type[resolved_args[0]]
            else:
                return base_type[tuple(resolved_args)]
        except (TypeError, AttributeError) as e:
            raise ValueError(
                f"Cannot create generic type {base_type_name}[{', '.join(type_args)}]: {e}"
            )

    def _parse_type_arguments(self, args_string: str) -> List[str]:
        """Parse type arguments from a string like 'str, int' or 'Dict[str, int], bool'.

        Handles nested brackets correctly.

        Args:
            args_string: String containing type arguments

        Returns:
            List of individual type argument strings
        """
        args = []
        current_arg = ""
        bracket_depth = 0

        for char in args_string:
            if char == "," and bracket_depth == 0:
                # End of current argument
                args.append(current_arg.strip())
                current_arg = ""
            else:
                if char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth -= 1
                current_arg += char

        # Add the last argument
        if current_arg.strip():
            args.append(current_arg.strip())

        return args

    def is_optional_type(self, type_obj: Type) -> bool:
        """Check if a type is Optional (Union with None).

        Args:
            type_obj: Type to check

        Returns:
            True if the type is Optional
        """
        origin = get_origin(type_obj)
        if origin is Union:
            args = get_args(type_obj)
            return type(None) in args
        return False

    def get_optional_inner_type(self, type_obj: Type) -> Type:
        """Get the inner type from an Optional type.

        Args:
            type_obj: Optional type

        Returns:
            Inner type (the non-None type)

        Raises:
            ValueError: If type is not Optional
        """
        if not self.is_optional_type(type_obj):
            raise ValueError(f"Type {type_obj} is not Optional")

        args = get_args(type_obj)
        for arg in args:
            if arg is not type(None):
                return arg

        raise ValueError(f"Could not find non-None type in {type_obj}")

    def is_list_type(self, type_obj: Type) -> bool:
        """Check if a type is a List type.

        Args:
            type_obj: Type to check

        Returns:
            True if the type is List
        """
        origin = get_origin(type_obj)
        return origin is list or origin is List

    def get_list_element_type(self, type_obj: Type) -> Type:
        """Get the element type from a List type.

        Args:
            type_obj: List type

        Returns:
            Element type

        Raises:
            ValueError: If type is not a List
        """
        if not self.is_list_type(type_obj):
            raise ValueError(f"Type {type_obj} is not a List")

        args = get_args(type_obj)
        if not args:
            return Any  # Untyped list

        return args[0]

    def list_available_types(self) -> List[str]:
        """List all available type names.

        Returns:
            List of registered type names
        """
        return list(self._type_registry.keys())


# Global resolver instance
_global_resolver = TypeResolver()


def get_global_resolver() -> TypeResolver:
    """Get the global type resolver instance.

    Returns:
        Global TypeResolver instance
    """
    return _global_resolver


def register_type(name: str, type_class: Type) -> None:
    """Register a type in the global resolver.

    Args:
        name: String name for the type
        type_class: Type class to register
    """
    _global_resolver.register_type(name, type_class)


def resolve_type_string(type_string: str) -> Type:
    """Resolve a type string using the global resolver.

    Args:
        type_string: String representation of the type

    Returns:
        Resolved Python type
    """
    return _global_resolver.resolve_type_string(type_string)


def is_optional_type(type_obj: Type) -> bool:
    """Check if a type is Optional using the global resolver.

    Args:
        type_obj: Type to check

    Returns:
        True if the type is Optional
    """
    return _global_resolver.is_optional_type(type_obj)


def is_list_type(type_obj: Type) -> bool:
    """Check if a type is a List using the global resolver.

    Args:
        type_obj: Type to check

    Returns:
        True if the type is List
    """
    return _global_resolver.is_list_type(type_obj)
