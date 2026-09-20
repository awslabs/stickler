"""Exercise the documentation gate with real MkDocs builds (issue #355)."""

import logging
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILD_COMMAND = [sys.executable, str(ROOT / "docs" / "check_links.py")]


@pytest.fixture
def docs_project(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "other page.md").write_text("# Existing heading\n", encoding="utf-8")
    config = {
        "site_name": "Link gate fixture",
        "theme": {"name": "mkdocs"},
        "plugins": [],
    }

    def build(markdown="# Home\n", *, extra_config=None, hook=None):
        (docs / "index.md").write_text(markdown, encoding="utf-8")
        settings = {**config, **(extra_config or {})}
        if hook:
            (tmp_path / "fixture_hook.py").write_text(hook, encoding="utf-8")
            settings["hooks"] = ["fixture_hook.py"]
        config_file = tmp_path / "mkdocs.yml"
        config_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
        return subprocess.run(
            [*BUILD_COMMAND, "-f", str(config_file), "-d", str(tmp_path / "site")],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )

    return build


@pytest.mark.parametrize(
    "markdown",
    [
        "# Home\n",
        "# Home\n[Local](#home)",
        "# Home\n[Other](other%20page.md#existing-heading)",
        "# Home\n[External](https://example.invalid/not-fetched#heading)",
        "# Home\n`[Example](missing.md)`",
    ],
)
def test_valid_docs_build(docs_project, markdown):
    result = docs_project(markdown)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("link", "diagnostic"),
    [
        ("[Missing](missing.md)", "missing.md"),
        ("![Missing](missing.png)", "missing.png"),
        ("[Missing](#absent-heading)", "absent-heading"),
        ("[Missing](other%20page.md#absent-heading)", "absent-heading"),
    ],
)
def test_broken_links_fail(docs_project, link, diagnostic):
    result = docs_project(f"# Home\n{link}\n")
    output = result.stdout + result.stderr
    assert diagnostic in output, output
    assert result.returncode != 0, output


def test_missing_navigation_target_fails(docs_project):
    result = docs_project(extra_config={"nav": [{"Home": "missing.md"}]})
    output = result.stdout + result.stderr
    assert "missing.md" in output, output
    assert result.returncode != 0, output


def test_unrelated_warning_remains_visible_without_failing(docs_project):
    result = docs_project(
        hook="import logging\n"
        "def on_pre_build(**kwargs):\n"
        "    logging.getLogger('mkdocs.plugins.fixture').warning('existing API warning')\n"
    )
    assert "existing API warning" in result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


def test_invalid_configuration_fails(docs_project):
    result = docs_project(extra_config={"docs_dir": "not-a-directory"})
    assert "not-a-directory" in result.stdout + result.stderr
    assert result.returncode != 0


def test_build_exception_fails(docs_project):
    result = docs_project(
        hook="def on_pre_build(**kwargs):\n"
        "    raise RuntimeError('fixture build failure')\n"
    )
    assert "fixture build failure" in result.stdout + result.stderr
    assert result.returncode != 0


@pytest.mark.parametrize("fail", [False, True])
def test_plugin_lifecycle_is_preserved(docs_project, fail):
    result = docs_project(
        hook="def on_startup(**kwargs):\n"
        "    print('fixture startup')\n"
        "def on_shutdown():\n"
        "    print('fixture shutdown')\n"
        "def on_pre_build(**kwargs):\n"
        f"    if {fail}: raise RuntimeError('fixture build failure')\n"
    )
    assert "fixture startup" in result.stdout
    assert "fixture shutdown" in result.stdout
    assert (result.returncode != 0) == fail


def test_repeated_builds_do_not_leak_warning_state(docs_project, tmp_path, caplog):
    docs_project("# Home\n[Missing](missing.md)")
    checker = runpy.run_path(str(ROOT / "docs" / "check_links.py"))["check_links"]
    loggers = [
        logging.getLogger(f"mkdocs.structure.{name}") for name in ("pages", "nav")
    ]
    original_handlers = [list(logger.handlers) for logger in loggers]
    with caplog.at_level(logging.INFO):
        assert checker(str(tmp_path / "mkdocs.yml")) == 1
        (tmp_path / "docs" / "index.md").write_text("# Home\n", encoding="utf-8")
        assert checker(str(tmp_path / "mkdocs.yml")) == 0
        docs_project(
            hook="def on_pre_build(**kwargs):\n"
            "    raise RuntimeError('fixture build failure')\n"
        )
        with pytest.raises(RuntimeError, match="fixture build failure"):
            checker(str(tmp_path / "mkdocs.yml"))
    assert [logger.handlers for logger in loggers] == original_handlers


def test_pr_workflow_only_builds_with_read_permissions():
    workflow = yaml.load(
        (ROOT / ".github" / "workflows" / "docs-check.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert workflow["on"] == ["pull_request"]
    assert workflow["permissions"] == {"contents": "read"}
    steps = workflow["jobs"]["build"]["steps"]
    assert "permissions" not in workflow["jobs"]["build"]
    for step in steps:
        if "uses" in step:
            assert re.fullmatch(r"[\w/-]+@[0-9a-f]{40}", step["uses"])
    commands = [step["run"] for step in steps if "run" in step]
    assert commands == [
        "uv sync --group docs --frozen",
        "uv run --frozen pytest tests/test_docs_build.py",
        "uv run --frozen python docs/check_links.py",
    ]
    assert steps[0]["with"]["persist-credentials"] == "false"
