"""Exercise the checkout guard through isolated pytest runs."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _run_guard(tmp_path, test_directory, import_location):
    checkout = tmp_path / "checkout"
    tests = checkout / "tests"
    tests.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
    )
    source_tests = Path(__file__).resolve().parent
    for relative in ("conftest.py", "integrations/conftest.py"):
        source = source_tests / relative
        if source.exists():
            destination = tests / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    import_root = {
        "checkout": checkout / "src",
        "other_checkout": tmp_path / "other_checkout" / "src",
        "prefix_sibling": checkout / "src-other",
        "symlink": checkout / "src",
    }[import_location]
    package = import_root / "stickler"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Package for import-origin testing."""\n')
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
    target.write_text(
        "from pathlib import Path\n\n"
        "def test_probe():\n"
        f"    Path({str(marker)!r}).write_text('ran')\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(import_root),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(target)],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, marker, package / "__init__.py", checkout / "src"


@pytest.mark.parametrize(
    "test_directory", ["", "structured_object_evaluator", "integrations"]
)
@pytest.mark.parametrize("import_location", ["other_checkout", "prefix_sibling"])
def test_wrong_checkout_fails_before_test_body(
    tmp_path, test_directory, import_location
):
    result, marker, imported, expected = _run_guard(
        tmp_path, test_directory, import_location
    )

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, (
        result.stdout + result.stderr
    )
    assert "1 error" in result.stdout
    assert str(imported.resolve()) in result.stdout
    assert str(expected.resolve()) in result.stdout
    assert not marker.exists()


@pytest.mark.parametrize(
    "test_directory", ["", "structured_object_evaluator", "integrations"]
)
def test_current_checkout_runs_test_body(tmp_path, test_directory):
    result, marker, _, _ = _run_guard(tmp_path, test_directory, "checkout")

    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert marker.read_text() == "ran"


def test_symlink_to_current_checkout_runs_test_body(tmp_path):
    result, marker, _, _ = _run_guard(tmp_path, "", "symlink")

    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    assert marker.read_text() == "ran"
