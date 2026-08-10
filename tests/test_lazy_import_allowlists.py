"""Lazy import paths resolve from an allowlist, never from caller input.

Three sites import a module by name at runtime to keep heavy optional
dependencies off the ``import stickler`` path: the two package
``__getattr__`` hooks and ``ComparatorRegistry._resolve``. Each looks the
caller's string up in a module-level dict and imports the *value*, so the
caller's string is only ever a key.

ASH/semgrep flags all three as `non-literal-import` because its taint rule
cannot see through the dict lookup. These tests pin the property the
suppressions claim, so the claim is enforced rather than asserted in a
comment. A future change that passed a caller-supplied string to
``import_module`` would fail here.
"""

import importlib

import pytest

import stickler
import stickler.comparators as sc
from stickler.structured_object_evaluator.models.comparator_registry import (
    ComparatorRegistry,
)

# Names an attacker would reach for: importable stdlib modules, dotted
# attribute paths, and traversal attempts.
HOSTILE_NAMES = [
    "os",
    "subprocess",
    "sys",
    "os.path",
    "importlib",
    "builtins",
    "../../etc/passwd",
    "stickler.comparators.llm",
    "",
]


@pytest.mark.parametrize("module", [stickler, sc], ids=["stickler", "comparators"])
@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_package_getattr_rejects_names_outside_the_allowlist(module, name):
    """``__getattr__`` resolves allowlisted comparators only.

    Anything else raises AttributeError before reaching ``import_module``,
    so an arbitrary string cannot become an import.
    """
    with pytest.raises(AttributeError):
        getattr(module, name)


@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_registry_rejects_names_outside_the_allowlist(name):
    """``ComparatorRegistry.get`` resolves built-in names only."""
    with pytest.raises(KeyError):
        ComparatorRegistry().get(name)


# The rejection tests above check the exception. These check the side effect
# that actually matters: importing a module executes its top-level code, so a
# hostile name must never reach `import_module` at all, whatever is raised
# afterwards. Spying on the import is what makes these load-bearing.


@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_registry_never_imports_a_caller_supplied_name(name, monkeypatch):
    """No import is attempted for a name outside `_BUILTINS`."""
    calls = []
    real = importlib.import_module

    def spy(path, *args, **kwargs):
        calls.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(
        "stickler.structured_object_evaluator.models.comparator_registry"
        ".importlib.import_module",
        spy,
    )

    with pytest.raises(KeyError):
        ComparatorRegistry().get(name)

    assert calls == [], f"attempted import(s) for a rejected name: {calls}"


@pytest.mark.parametrize(
    "module, module_name",
    [(stickler, "stickler"), (sc, "stickler.comparators")],
    ids=["stickler", "comparators"],
)
@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_package_getattr_never_imports_a_caller_supplied_name(
    module, module_name, name, monkeypatch
):
    """Same guarantee for both package ``__getattr__`` hooks."""
    calls = []
    real = importlib.import_module

    def spy(path, *args, **kwargs):
        calls.append(path)
        return real(path, *args, **kwargs)

    # Patch the `_il` alias each package imported, so only imports made by the
    # hook under test are recorded.
    monkeypatch.setattr(f"{module_name}._il.import_module", spy)

    with pytest.raises(AttributeError):
        getattr(module, name)

    assert calls == [], f"attempted import(s) for a rejected name: {calls}"


@pytest.mark.parametrize(
    "module, attr",
    [(stickler, "_LAZY_COMPARATORS"), (sc, "_LAZY_COMPARATORS")],
    ids=["stickler", "comparators"],
)
def test_lazy_comparator_paths_are_literals_under_stickler(module, attr):
    """Every allowlisted import path is a hardcoded `stickler.*` module.

    The import target comes from this table, so pinning its contents pins
    what can be imported.
    """
    for name, entry in getattr(module, attr).items():
        module_path = entry[0]
        assert isinstance(module_path, str)
        # The top-level package uses relative paths (".comparators.llm"),
        # the subpackage uses absolute ones.
        assert module_path.startswith((".comparators.", "stickler.comparators.")), (
            f"{name} resolves to {module_path!r}, outside stickler.comparators"
        )


def test_registry_builtin_paths_are_literals_under_stickler():
    """Same guarantee for the registry's built-in table."""
    for name, (module_path, _, _) in ComparatorRegistry._BUILTINS.items():
        assert module_path.startswith("stickler.comparators."), (
            f"{name} resolves to {module_path!r}, outside stickler.comparators"
        )


def test_registered_comparators_cannot_redirect_the_import_path():
    """A user registering a comparator supplies a class, not a module path.

    ``register`` takes the class directly, so user input never reaches
    ``import_module`` on that path either.
    """
    from stickler.comparators.base import BaseComparator

    class Custom(BaseComparator):
        def _compare(self, a, b):
            return 1.0

    registry = ComparatorRegistry()
    registry.register("Custom", Custom)

    assert registry.get("Custom") is Custom
    # The registration did not add anything to the lazily-imported table.
    assert "Custom" not in registry._pending


def test_allowlisted_name_still_resolves():
    """The suppressions must not have broken the behavior they annotate."""
    for name in ("LevenshteinComparator", "ExactComparator"):
        assert ComparatorRegistry().get(name).__name__ == name

    # A lazily-imported name resolves when its extra is installed, and raises
    # AttributeError (not ImportError) when it is not, so `hasattr` stays False.
    if importlib.util.find_spec("strands") is not None:
        assert stickler.LLMComparator.__name__ == "LLMComparator"
    else:
        assert not hasattr(stickler, "LLMComparator")
