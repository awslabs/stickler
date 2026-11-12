"""Exercise the checkout guard through isolated pytest runs."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _run_guard(tmp_path, test_directory, import_location, xfail=False, parallel=False):
    checkout = tmp_path / "checkout"
    tests = checkout / "tests"
    tests.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
    )
    source_tests = Path(__file__).resolve().parent
    shutil.copyfile(source_tests / "conftest.py", tests / "conftest.py")
    (tests / "__init__.py").touch()

    import_root = {
        "checkout": checkout / "src",
        "other_checkout": tmp_path / "other_checkout" / "src",
        "prefix_sibling": checkout / "src-other",
        "symlink": checkout / "src",
        "namespace": tmp_path / "namespace" / "src",
    }[import_location]
    package = import_root / "stickler"
    package.mkdir(parents=True)
    if import_location == "namespace":
        # Load this real namespace package without falling back to an installed
        # regular stickler package in the interpreter used by the harness.
        (import_root / "sitecustomize.py").write_text(
            "import sys\n"
            "from importlib.machinery import PathFinder\n"
            "from importlib.util import module_from_spec\n"
            f"spec = PathFinder.find_spec('stickler', [{str(import_root)!r}])\n"
            "sys.modules['stickler'] = module_from_spec(spec)\n"
        )
    else:
        (package / "__init__.py").write_text(
            '"""Package for import-origin testing."""\n'
        )
    if import_location == "symlink":
        alias = tmp_path / "linked-src"
        try:
            alias.symlink_to(import_root, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Cannot create directory symlink: {exc}")
        import_root = alias

    target = tests / test_directory / "test_probe.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = checkout / "test-body-ran"
    probe = (
        "from pathlib import Path\n\n"
        "import pytest\n\n"
        + ("@pytest.mark.xfail(strict=True)\n" if xfail else "")
        + "def test_probe():\n"
        f"    Path({str(marker)!r}).write_text('ran')\n"
        + ("    assert False\n" if xfail else "")
    )
    target.write_text(probe)
    env = {
        **os.environ,
        "PYTHONPATH": str(import_root),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"]
        + (["-p", "xdist", "-n", "2"] if parallel else [])
        + [str(target)],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, marker, package / "__init__.py", checkout / "src"


@pytest.mark.parametrize("test_directory", ["", "integrations"])
@pytest.mark.parametrize("import_location", ["other_checkout", "prefix_sibling"])
@pytest.mark.parametrize("xfail", [False, True], ids=["plain", "xfail"])
def test_wrong_checkout_stops_before_test_body(
    tmp_path, test_directory, import_location, xfail
):
    result, marker, imported, expected = _run_guard(
        tmp_path, test_directory, import_location, xfail=xfail
    )

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert "not this checkout" in output
    assert "xfailed" not in output
    assert "passed" not in output
    assert str(imported.resolve()) in output
    assert str(expected.resolve()) in output
    assert not marker.exists()


@pytest.mark.parametrize("test_directory", ["", "integrations"])
def test_current_checkout_runs_test_body(tmp_path, test_directory):
    result, marker, _, _ = _run_guard(tmp_path, test_directory, "checkout")

    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert marker.read_text() == "ran"


def test_symlink_to_current_checkout_runs_test_body(tmp_path):
    result, marker, _, _ = _run_guard(tmp_path, "", "symlink")

    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    assert marker.read_text() == "ran"


@pytest.mark.parametrize("xfail", [False, True], ids=["plain", "xfail"])
@pytest.mark.parametrize("import_location", ["other_checkout", "namespace"])
@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "xdist"])
def test_wrong_import_exits_cleanly(tmp_path, xfail, import_location, parallel):
    result, marker, _, _ = _run_guard(
        tmp_path, "integrations", import_location, xfail=xfail, parallel=parallel
    )
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert "not this checkout" in output
    assert "INTERNALERROR" not in output
    assert "xfailed" not in output
    assert not marker.exists()


def test_current_checkout_runs_in_parallel(tmp_path):
    result, marker, _, _ = _run_guard(
        tmp_path, "integrations", "checkout", parallel=True
    )
    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert marker.read_text() == "ran"
