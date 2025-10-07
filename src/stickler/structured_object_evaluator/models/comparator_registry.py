"""Comparator registry for dynamic model creation.

This module provides a registry system for mapping string names to comparator classes,
enabling configuration-based comparator selection in model_from_json().
"""

from typing import Dict, Type, Any, Optional, List
from stickler.comparators.base import BaseComparator


class ComparatorRegistry:
    """Registry for mapping comparator names to classes."""

    def __init__(self):
        """Initialize the registry with built-in comparators."""
        self._registry: Dict[str, Type[BaseComparator]] = {}
        self._register_builtin_comparators()

    def _register_builtin_comparators(self):
        """Register all built-in comparators."""
        try:
            from stickler.comparators.levenshtein import LevenshteinComparator

            self._registry["LevenshteinComparator"] = LevenshteinComparator
        except ImportError:
            pass

        try:
            from stickler.comparators.exact import ExactComparator

            self._registry["ExactComparator"] = ExactComparator
        except ImportError:
            pass

        try:
            from stickler.comparators.numeric import NumericComparator

            self._registry["NumericComparator"] = NumericComparator
        except ImportError:
            pass

        try:
            from stickler.comparators.fuzzy import FuzzyComparator

            self._registry["FuzzyComparator"] = FuzzyComparator
        except ImportError:
            pass

        try:
            from stickler.comparators.semantic import SemanticComparator

            self._registry["SemanticComparator"] = SemanticComparator
        except ImportError:
            pass

        try:
            from stickler.comparators.structured import StructuredModelComparator

            self._registry["StructuredModelComparator"] = StructuredModelComparator
        except ImportError:
            pass

        try:
            from stickler.comparators.bert import BertComparator

            self._registry["BertComparator"] = BertComparator
        except ImportError:
            pass

        try:
            from stickler.comparators.llm import LLMComparator

            self._registry["LLMComparator"] = LLMComparator
        except ImportError:
            pass

    def register(self, name: str, comparator_class: Type[BaseComparator]) -> None:
        """Register a new comparator class.

        Args:
            name: String name for the comparator
            comparator_class: Comparator class to register

        Raises:
            ValueError: If name is already registered or class is invalid
        """
        if not issubclass(comparator_class, BaseComparator):
            raise ValueError(
                f"Comparator class must inherit from BaseComparator, got {comparator_class}"
            )

        if name in self._registry:
            raise ValueError(f"Comparator '{name}' is already registered")

        self._registry[name] = comparator_class

    def get(self, name: str) -> Type[BaseComparator]:
        """Get a comparator class by name.

        Args:
            name: String name of the comparator

        Returns:
            Comparator class

        Raises:
            KeyError: If comparator name is not registered
        """
        if name not in self._registry:
            raise KeyError(
                f"Unknown comparator: '{name}'. Available: {list(self._registry.keys())}"
            )

        return self._registry[name]

    def create_instance(
        self, name: str, config: Optional[Dict[str, Any]] = None
    ) -> BaseComparator:
        """Create a comparator instance with optional configuration.

        Args:
            name: String name of the comparator
            config: Optional configuration dictionary

        Returns:
            Configured comparator instance

        Raises:
            KeyError: If comparator name is not registered
            TypeError: If configuration is invalid for the comparator
        """
        comparator_class = self.get(name)
        config = config or {}

        try:
            # Try to instantiate with config
            return comparator_class(**config)
        except TypeError as e:
            # If config fails, try without config
            try:
                return comparator_class()
            except TypeError:
                # Re-raise original error with config
                raise TypeError(f"Failed to create {name} with config {config}: {e}")

    def list_available(self) -> List[str]:
        """List all available comparator names.

        Returns:
            List of registered comparator names
        """
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a comparator name is registered.

        Args:
            name: Comparator name to check

        Returns:
            True if registered, False otherwise
        """
        return name in self._registry


# Global registry instance
_global_registry = ComparatorRegistry()


def get_global_registry() -> ComparatorRegistry:
    """Get the global comparator registry instance.

    Returns:
        Global ComparatorRegistry instance
    """
    return _global_registry


def register_comparator(name: str, comparator_class: Type[BaseComparator]) -> None:
    """Register a comparator in the global registry.

    Args:
        name: String name for the comparator
        comparator_class: Comparator class to register
    """
    _global_registry.register(name, comparator_class)


def get_comparator_class(name: str) -> Type[BaseComparator]:
    """Get a comparator class from the global registry.

    Args:
        name: String name of the comparator

    Returns:
        Comparator class
    """
    return _global_registry.get(name)


def create_comparator(
    name: str, config: Optional[Dict[str, Any]] = None
) -> BaseComparator:
    """Create a comparator instance from the global registry.

    Args:
        name: String name of the comparator
        config: Optional configuration dictionary

    Returns:
        Configured comparator instance
    """
    return _global_registry.create_instance(name, config)
