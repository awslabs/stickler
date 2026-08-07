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

import json
import os
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
    arriving on the core path, not a precise budget. Measured baseline at the
    time of writing is 463 modules on a core-only install (Python 3.12), so 900
    leaves roughly 2x headroom.
    """
    loaded = _modules_after_importing_stickler()
    assert len(loaded) < 900, (
        f"`import stickler` now loads {len(loaded)} modules. Something heavy "
        f"joined the core import path; check for a new module-level import."
    )


def test_optional_comparator_resolves_on_access():
    """A lazily-exported comparator imports its dependency only when accessed.

    This is the other half of the guard above: deferring the import is only
    correct if the comparator still works. Skips when the extra is absent,
    since there is nothing to resolve.
    """
    code = (
        "import sys, json;"
        " import stickler;"
        " before = 'torch' in sys.modules;"
        " available = 'BERTComparator' in stickler.__all__;"
        " cls = getattr(stickler, 'BERTComparator', None) if available else None;"
        " print(json.dumps({"
        "  'available': available,"
        "  'before': before,"
        "  'after': 'torch' in sys.modules,"
        "  'name': getattr(cls, '__name__', None)}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f"could not probe optional comparator: {result.stderr[-200:]}")

    data = json.loads(result.stdout.strip().splitlines()[-1])
    if not data["available"]:
        pytest.skip("bert extra not installed")

    assert data["before"] is False, (
        "torch was already loaded at `import stickler`, so the lazy export is "
        "not actually deferring the import."
    )
    assert data["name"] == "BERTComparator", (
        "accessing stickler.BERTComparator did not resolve to the class; the "
        "lazy __getattr__ is broken."
    )
    assert data["after"] is True, (
        "accessing stickler.BERTComparator did not import torch, which means "
        "the probe did not exercise the real module."
    )


def test_broken_extra_does_not_break_import(tmp_path):
    """An installed-but-broken extra must not take down `import stickler`.

    A version-skewed transitive dependency raises plain ImportError rather than
    ModuleNotFoundError. `import stickler` has to survive that and surface the
    failure at first use, naming the extra.
    """
    code = "import sys, json; import stickler; print(json.dumps(sorted(stickler.__all__)))"
    baseline = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    if baseline.returncode != 0 or "BERTComparator" not in json.loads(
        baseline.stdout.strip().splitlines()[-1]
    ):
        pytest.skip("bert extra not installed")

    # Shadow a transitive dependency of the bert extra so importing it fails.
    shim = tmp_path / "datasets.py"
    shim.write_text('raise ImportError("simulated version-skewed dependency")\n')

    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")

    probe = (
        "import stickler;"
        " print('IMPORT_OK');"
        " import sys;"
        " sys.stdout.flush();"
        " err = None\n"
        "try:\n"
        "    stickler.BERTComparator\n"
        "except Exception as exc:\n"
        "    err = f'{type(exc).__name__}: {exc}'\n"
        "print(f'ACCESS={err}')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env
    )

    assert "IMPORT_OK" in result.stdout, (
        "`import stickler` failed with a broken optional extra. It must "
        f"survive and defer the failure. stderr: {result.stderr[-400:]}"
    )
    access_line = [
        line for line in result.stdout.splitlines() if line.startswith("ACCESS=")
    ]
    assert access_line, f"probe did not report access result: {result.stdout!r}"
    detail = access_line[0]
    assert "None" not in detail, (
        "accessing a broken extra's comparator unexpectedly succeeded"
    )
    assert "bert" in detail, (
        f"the error should name the owning extra so the user knows what to "
        f"fix, got: {detail}"
    )
