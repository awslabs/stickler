"""Comparator registry for dynamic model creation.

This module provides a registry system for mapping string names to comparator classes,
enabling configuration-based comparator selection in model_from_json().
"""

import importlib
import importlib.util
import inspect
import sys
from collections.abc import Mapping as abc_Mapping
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from stickler.comparators.base import BaseComparator
from stickler.utils.deprecation import warn_once


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
        "NormalizedComparator": ("stickler.comparators.normalized", None, None),
        "PhoneComparator": ("stickler.comparators.phone", None, None),
        "NumericComparator": ("stickler.comparators.numeric", None, None),
        "DateComparator": ("stickler.comparators.date", None, None),
        "FuzzyComparator": ("stickler.comparators.fuzzy", None, None),
        "StructuredModelComparator": ("stickler.comparators.structured", None, None),
        "ANLSStarComparator": ("stickler.comparators.anls", None, None),
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

        A failed import leaves the name in ``_pending``, so a later call retries
        and nothing the registry reports changes as a side effect of the
        failure. The entry is consumed only once the class is in hand.
        """
        spec = self._pending.get(name)
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
            # Leave the entry in `_pending`. A broken extra is unavailable now,
            # but the name is still a built-in, so `is_registered()` and
            # `list_available()` must not change as a side effect of a failed
            # lookup, and `register()` -- which rejects a name only when it is
            # in `_registry` or `_pending` -- must keep rejecting it rather than
            # letting a caller silently shadow a built-in (#260).
            return None
        comparator_class = getattr(module, name)
        self._pending.pop(name, None)
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

        # Drop only the keys this comparator cannot take, and say which.
        #
        # The previous behaviour was to try `comparator_class(**config)` and, on
        # `TypeError`, retry `comparator_class()` -- discarding the WHOLE config in
        # silence. Since `comparator_class(**{})` is already `comparator_class()`,
        # that fallback could only ever fire when the caller DID supply keys, and
        # it turned one bad key into a lost configuration and a wrong number:
        #
        #     comparator_config {"relative_tolerence": 0.001}   (note the typo)
        #       built NumericComparator() with NO tolerance, discarding the
        #       inferred {"relative_tolerance": 0.001} merged in beside it, so
        #       1000.00 against 1000.001 scored 0.0 rather than 1.0
        #
        # Raising instead is not right either: `to_stickler_config()` exports the
        # comparator's own config, so editing the exported `comparator` by hand --
        # a documented workflow -- leaves the previous comparator's keys behind,
        # and a hard failure there would reject a config stickler itself produced.
        #
        # Filtering keeps both honest. The valid keys are applied, the invalid ones
        # are named, and nothing is dropped without being reported.
        accepted, unknown = self._partition_config(comparator_class, config)
        if unknown:
            warn_once(
                "comparator-config-unknown-keys",
                f"{name}:{','.join(sorted(unknown))}",
                f"{name} does not accept "
                f"{', '.join(repr(k) for k in sorted(unknown))} in its "
                f"comparator config; ignored. It accepts "
                f"{', '.join(sorted(self._accepted_parameters(comparator_class)))}. "
                f"A stale key is usually left over from editing an exported "
                f"config's 'comparator' without clearing 'comparator_config'.",
                category=UserWarning,
            )
        try:
            return comparator_class(**accepted)
        except TypeError as e:
            raise TypeError(f"Failed to create {name} with config {accepted}: {e}")

    @staticmethod
    def _accepted_parameters(comparator_class: Type[BaseComparator]) -> Set[str]:
        """Keyword parameter names ``comparator_class.__init__`` will take.

        ``**kwargs`` in the signature means "anything", reported as the sentinel
        ``{"**kwargs"}`` so :meth:`_partition_config` can wave everything through
        rather than guessing at an out-of-tree comparator's real parameter set.
        """
        try:
            parameters = inspect.signature(comparator_class.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            return {"**kwargs"}
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return {"**kwargs"}
        return {name for name in parameters if name != "self"}

    @classmethod
    def _partition_config(
        cls, comparator_class: Type[BaseComparator], config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Set[str]]:
        """Split ``config`` into what this comparator takes and what it does not."""
        accepted_names = cls._accepted_parameters(comparator_class)
        if "**kwargs" in accepted_names:
            return dict(config), set()
        accepted = {k: v for k, v in config.items() if k in accepted_names}
        return accepted, set(config) - accepted_names

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


def normalize_comparator_config(value: Any, where: str) -> Dict[str, Any]:
    """Validate an authored ``comparator_config`` and return it as a dict.

    ``None`` and an absent key both mean "nothing supplied" and give ``{}``.
    Anything that is not a mapping raises a ``ValueError`` naming ``where``.

    Exists because the value is merged with ``{**inferred, **author}``, and that
    unpack raises a bare ``TypeError: 'str' object is not a mapping`` with no field
    in it. Only ``ValueError`` is wrapped with the field's name by
    ``convert_fields_config``, so a mis-typed config in a hand-written JSON file --
    the population this feature is for -- produced an unactionable traceback that
    named neither the field nor the key.

    Args:
        value: Whatever the author put under ``comparator_config``.
        where: Human-readable location, e.g. ``"field 'total'"``, quoted back in
            the error so the author can find it.

    Returns:
        The config as a plain dict, empty when nothing was supplied.

    Raises:
        ValueError: If ``value`` is neither absent nor a mapping.
    """
    if value is None:
        return {}
    if isinstance(value, abc_Mapping):
        non_string = sorted(
            (repr(key) for key in value if not isinstance(key, str)),
        )
        if non_string:
            # Checked here rather than left to the caller. A parameter name is
            # passed as a keyword argument, so a non-string key cannot become one;
            # and the unknown-key report below joins the names with `str.join`,
            # which raised `TypeError: sequence item 0: expected str instance, int
            # found` from inside the reporting code -- an error about the error,
            # naming neither the field nor the offending key.
            raise ValueError(
                f"'comparator_config' on {where} has non-string parameter "
                f"{'names' if len(non_string) > 1 else 'name'} "
                f"{', '.join(non_string)}. Parameter names are passed as keyword "
                f"arguments, so they must be strings."
            )
        return dict(value)
    raise ValueError(
        f"'comparator_config' on {where} must be a mapping of parameter names to "
        f"values, got {type(value).__name__}: {value!r}"
    )
