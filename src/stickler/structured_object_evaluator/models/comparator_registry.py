"""Comparator registry for dynamic model creation.

This module provides a registry system for mapping string names to comparator classes,
enabling configuration-based comparator selection in model_from_json().
"""

import importlib
import importlib.util
import sys
from typing import Any, Dict, List, Optional, Type

from stickler.comparators.base import BaseComparator


class ComparatorRegistry:
    """Registry for mapping comparator names to classes."""

    # Built-in comparator name -> (module path, dependency to probe, owning
    # extra). Stored as paths rather than classes so constructing the registry
    # does not import every comparator: BERTComparator pulls torch,
    # transformers, and datasets, and LLMComparator pulls strands-agents and
    # boto3. The global registry is built at module import, so eager
    # registration put all of that on the `import stickler` path.
    #
    # `probe` is the distribution whose absence means the comparator is
    # unavailable, or None when the comparator needs nothing beyond the core.
    _BUILTINS = {
        "LevenshteinComparator": ("stickler.comparators.levenshtein", None, None),
        "ExactComparator": ("stickler.comparators.exact", None, None),
        "NumericComparator": ("stickler.comparators.numeric", None, None),
        "DateComparator": ("stickler.comparators.date", None, None),
        "FuzzyComparator": ("stickler.comparators.fuzzy", None, None),
        "StructuredModelComparator": ("stickler.comparators.structured", None, None),
        "BBoxIoUComparator": ("stickler.comparators.bbox", None, None),
        "SemanticComparator": ("stickler.comparators.semantic", None, None),
        "BERTComparator": ("stickler.comparators.bert", "evaluate", "bert"),
        "LLMComparator": ("stickler.comparators.llm", "strands", "llm"),
    }

    def __init__(self):
        """Initialize the registry with built-in comparators.

        Built-ins are recorded lazily; user registrations via :meth:`register`
        are stored as classes directly.
        """
        self._registry: Dict[str, Type[BaseComparator]] = {}
        # Built-in names whose module has not been imported yet.
        self._pending = {
            name: spec
            for name, spec in self._BUILTINS.items()
            if self._builtin_is_available(spec)
        }

    @staticmethod
    def _builtin_is_available(spec) -> bool:
        """Whether a built-in's dependency is installed, without importing it.

        Mirrors the package-level ``_dependency_available`` helpers: consult
        ``sys.modules`` before the filesystem, so a test-injected mock counts as
        available. Diverging would make two public entry points disagree in one
        process -- ``stickler.LLMComparator`` resolving while
        ``registry.get("LLMComparator")`` reports the comparator does not exist.

        The ``find_spec`` fallback is guarded because it raises rather than
        returning None for a ``sys.modules`` entry with no ``__spec__``
        (``ValueError: <name>.__spec__ is not set``).
        """
        _, probe, _ = spec
        if probe is None:
            return True

        module = sys.modules.get(probe, False)
        if module is not None and module is not False:
            # Present in sys.modules, including a test-injected mock.
            return True
        if module is None:
            # Explicitly blocked (sys.modules[probe] = None), which is how tests
            # simulate a missing dependency.
            return False

        try:
            return importlib.util.find_spec(probe) is not None
        except (ImportError, ValueError):
            return False

    def _resolve(self, name: str) -> Optional[Type[BaseComparator]]:
        """Import and cache a pending built-in. Returns None if unavailable.

        A dependency that is installed but broken (a version-skewed transitive
        dep raising plain ImportError) is treated as unavailable rather than
        propagating, matching the previous try/except behavior.
        """
        spec = self._pending.pop(name, None)
        if spec is None:
            return None
        module_path, _, _ = spec
        try:
            # `module_path` is a literal from `_BUILTINS`, not the caller's
            # `name`. An unregistered `name` misses `_pending` and returns None
            # above, so a caller-supplied string is only ever a dict key and
            # never an import path. Semgrep matches a non-literal first
            # argument and does not follow the dict lookup.
            # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
            module = importlib.import_module(module_path)
        except ImportError:
            return None
        comparator_class = getattr(module, name)
        self._registry[name] = comparator_class
        return comparator_class

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

        if name in self._registry or name in self._pending:
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
            resolved = self._resolve(name)
            if resolved is not None:
                return resolved
            raise KeyError(
                f"Unknown comparator: '{name}'. Available: {self.list_available()}"
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
        return list(self._registry.keys()) + list(self._pending.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a comparator name is registered.

        Args:
            name: Comparator name to check

        Returns:
            True if registered, False otherwise
        """
        return name in self._registry or name in self._pending


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
