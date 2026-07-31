"""Guards that `import stickler` stays light.

The comparison engine needs only pydantic, rapidfuzz, munkres, numpy, and
python-dateutil. Everything heavier (pandas, scipy, scikit-learn, jinja2, and
the ML stack behind the optional comparators) belongs to a peripheral module
and lives behind an extra.

These tests fail if a module-level import puts one of those packages back on
the ``import stickler`` path. That regression is easy to introduce and silent:
the suite installs every extra, so the import still succeeds here while a
default install breaks for users.
"""

import subprocess
import sys

import pytest

# Packages that must not be imported as a side effect of `import stickler`,
# mapped to the extra that owns them.
FORBIDDEN_ON_CORE_PATH = {
    "pandas": "docsplit / reporting",
    "scipy": "semantic / docsplit",
    "sklearn": "confidence / docsplit",
    "jinja2": "llm",
    "torch": "bert",
    "transformers": "bert",
    "strands": "llm",
    "boto3": "semantic",
}


def _modules_after_importing_stickler() -> set:
    """Return sys.modules keys after a bare `import stickler`.

    Runs in a subprocess so the already-imported state of this test session
    (which has every extra installed) cannot mask a regression.
    """
    code = "import sys, json; import stickler; print(json.dumps(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    import json

    return set(json.loads(result.stdout))


@pytest.mark.parametrize("package,owning_extra", sorted(FORBIDDEN_ON_CORE_PATH.items()))
def test_heavy_package_not_on_core_import_path(package, owning_extra):
    """A bare `import stickler` must not pull an extra's dependency."""
    loaded = _modules_after_importing_stickler()
    top_level = {m.split(".")[0] for m in loaded}
    assert package not in top_level, (
        f"`import stickler` loaded {package!r}, which belongs to the "
        f"{owning_extra} extra. Move the import inside the function that needs "
        f"it, or guard it with TYPE_CHECKING if it is only an annotation."
    )


def test_core_import_stays_small():
    """Sanity bound on module count, to catch a broad new eager dependency.

    The threshold is deliberately loose: it is a tripwire for something large
    arriving on the core path, not a precise budget.
    """
    loaded = _modules_after_importing_stickler()
    assert len(loaded) < 900, (
        f"`import stickler` now loads {len(loaded)} modules. Something heavy "
        f"joined the core import path; check for a new module-level import."
    )
