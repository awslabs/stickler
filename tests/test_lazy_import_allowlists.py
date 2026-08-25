"""Lazy import paths resolve from an allowlist, never from caller input.

Three sites import a module by name at runtime to keep heavy optional
dependencies off the ``import stickler`` path: the two package
``__getattr__`` hooks and ``ComparatorRegistry._resolve``. Each looks the
caller's string up in a module-level dict and imports the *value*, so the
caller's string is only ever a key.

ASH/semgrep flags all three as `non-literal-import` because its pattern match
sees a non-literal first argument and cannot follow the dict lookup. These
tests pin the property the suppressions claim, so the claim is enforced rather
than asserted in a comment. A change that passed a caller-supplied string to
the import machinery would fail here, whatever spelling it used.

``HOSTILE_NAMES`` sweeps the *rejection* tests, which assert the exception. The
side-effect tests use a single canary name instead, because a canary is the only
name whose import can be observed -- every other plausible hostile name is
already in ``sys.modules``, so importing it is a no-op that nothing can detect.
Path-shaped and empty-string inputs are therefore covered for rejection but not
for import side effects.
"""

import importlib
import importlib.util
import sys
import types
from unittest import mock

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
# hostile name must never reach the import machinery at all, whatever is raised
# afterwards. Recording attempted imports is what makes these load-bearing.


CANARY = "stickler_import_canary"


@pytest.fixture
def canary(tmp_path, monkeypatch):
    """A real importable module that records the fact of being imported.

    Spying on ``import_module`` is not sufficient here, for two reasons found by
    measurement:

    1. ``stickler._il``, ``stickler.comparators._il`` and the registry's
       ``importlib`` are all the *same module object*, so patching one is
       process-wide rather than scoped to a single hook -- there is no per-hook
       isolation to be had by that route.
    2. It only covers one spelling. Code doing
       ``from importlib import import_module as f`` binds the real function at
       its own import time, before any patch, so ``f(name)`` is invisible to a
       spy on the attribute. Patching ``builtins.__import__`` does not close
       this either, since it does not fire for a module already in
       ``sys.modules`` -- which every plausible hostile name (``os``, ``sys``,
       ``builtins``) already is.

    Importing a module *executes its top-level code*, and that side effect is
    the actual hazard. So this writes a module whose body appends to a shared
    list: if anything imports it by any spelling, the list is non-empty. It
    cannot be evaded by rebinding, and it needs no patching.
    """
    witness = []
    monkeypatch.setitem(
        sys.modules,
        "_canary_witness",
        types.ModuleType("_canary_witness"),
    )
    sys.modules["_canary_witness"].hits = witness

    (tmp_path / f"{CANARY}.py").write_text(
        "import _canary_witness\n_canary_witness.hits.append('imported')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, CANARY, raising=False)
    importlib.invalidate_caches()

    yield witness

    sys.modules.pop(CANARY, None)


def test_the_canary_is_wired_up(canary):
    """Guard the guard: an actual import must register, or the tests below lie."""
    importlib.import_module(CANARY)

    assert canary == ["imported"], "the canary fixture is not detecting imports"


def test_registry_never_imports_a_caller_supplied_name(canary):
    """A name outside `_BUILTINS` never reaches the import machinery."""
    with pytest.raises(KeyError):
        ComparatorRegistry().get(CANARY)

    assert canary == [], "the rejected name was imported"


@pytest.mark.parametrize("module", [stickler, sc], ids=["stickler", "comparators"])
def test_package_getattr_never_imports_a_caller_supplied_name(module, canary):
    """Same guarantee for both package ``__getattr__`` hooks."""
    with pytest.raises(AttributeError):
        getattr(module, CANARY)

    assert canary == [], "the rejected name was imported"


def test_no_spelling_of_import_reaches_a_caller_supplied_name(canary):
    """Every entry point, one canary: nothing imports it, however it is called.

    Because the canary observes the *effect* rather than one function, this
    holds for `importlib.import_module`, a rebound
    `from importlib import import_module as f`, `__import__`, or `exec`.
    """
    for target in (stickler, sc):
        with pytest.raises(AttributeError):
            getattr(target, CANARY)

    with pytest.raises(KeyError):
        ComparatorRegistry().get(CANARY)

    assert canary == [], "a rejected name was imported by some path"


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


def test_registry_constructs_when_a_dependency_is_mocked():
    """A ``sys.modules`` entry with no ``__spec__`` must not break construction.

    ``find_spec`` raises ``ValueError: <name>.__spec__ is not set`` rather than
    returning None for a mock injection, which is how the suite simulates an
    optional dependency being present. Both package-level
    ``_dependency_available`` helpers already guard this; the registry did not,
    so constructing one while a mock was installed raised.
    """
    with mock.patch.dict(sys.modules, {"strands": mock.MagicMock()}):
        registry = ComparatorRegistry()

        assert "LevenshteinComparator" in registry._pending

        # A mock counts as available, matching the package-level
        # `_dependency_available` helpers. Returning False here instead would
        # make two public entry points disagree in one process:
        # `stickler.LLMComparator` would resolve while `registry.get(...)`
        # reported the comparator does not exist.
        assert "LLMComparator" in registry._pending
        assert stickler._dependency_available("strands") is True


def test_a_boto3_shim_without_a_module_spec_does_not_break_the_import():
    """An optional dependency must never be probed at module scope.

    ``comparators/utils.py`` used to run
    ``importlib.util.find_spec("boto3")`` at import time, and that module is on
    the ``import stickler`` path via ``comparators/semantic.py``. ``find_spec``
    raises ``ValueError`` -- rather than returning None -- for an installed
    module whose ``__spec__`` is None, which is what a hand-rolled shim looks
    like, so the whole package became unimportable in such an environment
    (#257).

    The probe is gone, so this asserts the property rather than the line: with a
    spec-less boto3 shim installed, reimporting the package succeeds.
    """
    shim = types.ModuleType("boto3")
    shim.__spec__ = None

    with mock.patch.dict(sys.modules, {"boto3": shim}):
        # The precondition this regresses on: the call the old probe made raises.
        with pytest.raises(ValueError):
            importlib.util.find_spec("boto3")

        importlib.reload(stickler)
        importlib.reload(importlib.import_module("stickler.comparators.utils"))

    # Leave the module registry as the rest of the session found it.
    importlib.reload(stickler)


def test_no_module_scope_boto3_availability_flag_remains():
    """The removed probe left no successor.

    A module-level flag is the shape of the bug: computing it requires
    inspecting boto3 at import time. Availability is reported by the
    ``ImportError`` that ``generate_bedrock_embedding`` raises instead.
    """
    import stickler.comparators.utils as utils

    assert not [name for name in vars(utils) if "BOTO3" in name.upper()]


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
    if stickler._dependency_available("strands"):
        assert stickler.LLMComparator.__name__ == "LLMComparator"
    else:
        assert not hasattr(stickler, "LLMComparator")
