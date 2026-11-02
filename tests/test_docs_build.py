"""Exercise the documentation gate with real MkDocs builds (issue #355)."""

import logging
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("mkdocs")
pytest.importorskip("mkdocs_awesome_nav")

ROOT = Path(__file__).resolve().parents[1]
BUILD_COMMAND = [sys.executable, str(ROOT / "docs" / "check_links.py")]


@pytest.fixture
def docs_config(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "other page.md").write_text("# Existing heading\n", encoding="utf-8")
    config = {
        "site_name": "Link gate fixture",
        "theme": {"name": "mkdocs"},
        "plugins": [],
        "validation": yaml.load(
            (ROOT / "docs" / "mkdocs.yml").read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        ).get("validation", {}),
    }

    def write_config(markdown="# Home\n", *, extra_config=None, hook=None):
        (docs / "index.md").write_text(markdown, encoding="utf-8")
        settings = {**config, **(extra_config or {})}
        if hook:
            (tmp_path / "fixture_hook.py").write_text(hook, encoding="utf-8")
            settings["hooks"] = ["fixture_hook.py"]
        config_file = tmp_path / "mkdocs.yml"
        config_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
        return config_file

    return write_config


@pytest.fixture
def docs_project(docs_config, tmp_path):
    def build(markdown="# Home\n", **kwargs):
        config_file = docs_config(markdown, **kwargs)
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
        ("[Missing](Nope/)", "Nope/"),
        ("[Missing](/nope.md)", "/nope.md"),
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


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        (
            "bulk_structured_model_evaluator.py",
            "No type or annotation for parameter 'df'",
        ),
        (
            "models/structured_model.py",
            "No type or annotation for parameter '**kwargs'",
        ),
        (
            "models/structured_model.py",
            "No type or annotation for returned value 1",
        ),
        (
            "models/comparable_field.py",
            "No type or annotation for parameter '**field_kwargs'",
        ),
        (
            "models/comparable_field.py",
            "No type or annotation for returned value 1",
        ),
        (
            "models/field.py",
            "No type or annotation for parameter '**kwargs'",
        ),
    ],
)
@pytest.mark.parametrize("separator", ["/", "\\"])
@pytest.mark.parametrize("line", [1, 2500])
def test_known_warning_remains_visible_without_failing(
    docs_project, source, diagnostic, separator, line
):
    message = (
        "griffe: C:/checkout/src/stickler/structured_object_evaluator/"
        f"{source}:{line}: {diagnostic}"
    )
    message = message.replace("/", separator)
    result = docs_project(
        hook="import logging\n"
        "def on_pre_build(**kwargs):\n"
        f"    logging.getLogger('mkdocs.plugins.griffe').warning({message!r})\n"
    )
    assert message in result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("second_line", [51, 2500])
def test_duplicate_known_warning_fails(docs_project, second_line):
    first = (
        "griffe: src/stickler/structured_object_evaluator/models/field.py:51: "
        "No type or annotation for parameter '**kwargs'"
    )
    second = first.replace(":51:", f":{second_line}:")
    result = docs_project(
        hook="import logging\n"
        "def on_pre_build(**kwargs):\n"
        "    logger = logging.getLogger('mkdocs.plugins.griffe')\n"
        f"    logger.warning({first!r})\n"
        f"    logger.warning({second!r})\n"
    )
    assert result.stderr.count("No type or annotation for parameter '**kwargs'") == 2
    assert result.returncode != 0, result.stderr
    assert "Documentation check failed: 1 warning(s) or error(s)." in result.stderr


def test_known_warning_allowance_resets_per_build(docs_config, tmp_path, caplog):
    message = (
        "griffe: src/stickler/structured_object_evaluator/models/field.py:51: "
        "No type or annotation for parameter '**kwargs'"
    )
    config = docs_config(
        hook="import logging\n"
        "def on_pre_build(**kwargs):\n"
        f"    logging.getLogger('mkdocs.plugins.griffe').warning({message!r})\n"
    )
    checker = runpy.run_path(str(ROOT / "docs" / "check_links.py"))["check_links"]
    with caplog.at_level(logging.WARNING):
        assert checker(str(config), str(tmp_path / "site")) == 0
        assert checker(str(config), str(tmp_path / "site")) == 0
    assert caplog.messages.count(message) == 2


@pytest.mark.parametrize("nav", ["nav:\n  - Does-Not-Exist.md\n", "nav: [\n"])
def test_awesome_nav_failures_are_fatal(docs_project, tmp_path, nav):
    guides = tmp_path / "docs" / "Guides"
    guides.mkdir()
    (guides / "index.md").write_text("# Guides\n", encoding="utf-8")
    (guides / ".nav.yml").write_text(nav, encoding="utf-8")
    result = docs_project(extra_config={"plugins": ["awesome-nav"]})
    output = result.stdout + result.stderr
    assert "awesome-nav" in output, output
    assert result.returncode != 0, output


@pytest.mark.parametrize(
    ("logger_name", "level"),
    [
        ("mkdocs.plugins.future", "warning"),
        ("mkdocs.plugins.griffe", "warning"),
        ("mkdocs.plugins.mkdocs_autorefs._internal.plugin", "warning"),
        ("mkdocs.plugins.future", "error"),
        ("mkdocs.plugins.griffe_unrelated", "warning"),
        ("mkdocs.plugins.griffe", "error"),
        ("mkdocs.plugins.mkdocs_autorefs._internal.plugin", "error"),
    ],
)
def test_unexpected_diagnostics_fail(docs_project, logger_name, level):
    result = docs_project(
        hook="import logging\n"
        "def on_pre_build(**kwargs):\n"
        f"    logging.getLogger({logger_name!r}).{level}('new diagnostic')\n"
    )
    output = result.stdout + result.stderr
    assert "new diagnostic" in output, output
    assert result.returncode != 0, output


def test_configuration_warning_is_not_missed(docs_project):
    result = docs_project(extra_config={"misspelled_setting": True})
    output = result.stdout + result.stderr
    assert "misspelled_setting" in output, output
    assert result.returncode != 0, output


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


def test_repeated_builds_do_not_leak_warning_state(docs_config, tmp_path, caplog):
    docs_config("# Home\n[Missing](missing.md)")
    checker = runpy.run_path(str(ROOT / "docs" / "check_links.py"))["check_links"]
    loggers = [logging.getLogger("mkdocs")] + [
        logging.getLogger(f"mkdocs.structure.{name}") for name in ("pages", "nav")
    ]
    original_handlers = [list(logger.handlers) for logger in loggers]
    with caplog.at_level(logging.INFO):
        assert checker(str(tmp_path / "mkdocs.yml")) == 1
        (tmp_path / "docs" / "index.md").write_text("# Home\n", encoding="utf-8")
        assert checker(str(tmp_path / "mkdocs.yml")) == 0
        docs_config(
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
    assert workflow["permissions"] == {"contents": "read"}
    steps = workflow["jobs"]["build"]["steps"]
    assert "permissions" not in workflow["jobs"]["build"]
    for step in steps:
        if "uses" in step:
            assert re.fullmatch(r"[\w/-]+@[0-9a-f]{40}", step["uses"])
    assert steps[0]["with"]["persist-credentials"] == "false"


def test_omitted_page_fails(docs_project):
    result = docs_project(extra_config={"nav": [{"Home": "index.md"}]})
    output = result.stdout + result.stderr
    assert "other page.md" in output and 'not included in the "nav"' in output
    assert result.returncode != 0, output


@pytest.mark.parametrize("content", [None, "nav: ["])
def test_configuration_error_is_concise(tmp_path, content):
    config = tmp_path / "bad.yml"
    if content is not None:
        config.write_text(content, encoding="utf-8")
    result = subprocess.run(
        [*BUILD_COMMAND, "-f", str(config)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
    assert "bad.yml" in result.stderr
    assert "Traceback" not in result.stderr, result.stderr


@pytest.mark.parametrize(
    "logger_name,level,message",
    [
        (
            "mkdocs.plugins.griffe",
            "warning",
            "griffe: src/stickler/new.py:51: No type or annotation for parameter '**kwargs'",
        ),
        (
            "mkdocs.plugins.griffe",
            "error",
            "griffe: src/stickler/structured_object_evaluator/models/field.py:51: No type or annotation for parameter '**kwargs'",
        ),
        (
            "mkdocs.plugins.future",
            "warning",
            "griffe: src/stickler/structured_object_evaluator/models/field.py:51: No type or annotation for parameter '**kwargs'",
        ),
        (
            "mkdocs.plugins.griffe",
            "warning",
            "griffe: src/stickler/structured_object_evaluator/models/field.py:51: No type or annotation for parameter '**new_kwargs'",
        ),
        (
            "mkdocs.plugins.mkdocs_autorefs._internal.plugin",
            "warning",
            "mkdocs_autorefs: API-Reference/models.md: from /checkout/src/stickler/structured_object_evaluator/models/structured_model.py:263: (stickler.structured_object_evaluator.models.structured_model.StructuredModel) Could not find cross-reference target ''tp''",
        ),
        *[
            (
                "mkdocs.plugins.mkdocs_autorefs._internal.plugin",
                "warning",
                "mkdocs_autorefs: API-Reference/models.md: from /checkout/"
                "src/stickler/structured_object_evaluator/models/structured_model.py:262: "
                "(stickler.structured_object_evaluator.models.structured_model.StructuredModel) "
                f"Could not find cross-reference target ''{target}''",
            )
            for target in ("tp", "fd", "derived")
        ],
    ],
)
def test_warning_exemptions_do_not_spread(docs_project, logger_name, level, message):
    result = docs_project(
        hook="import logging\n"
        "def on_pre_build(**kwargs):\n"
        f"    logging.getLogger({logger_name!r}).{level}({message!r})\n"
    )
    assert message in result.stderr
    assert result.returncode != 0, result.stderr


def test_python_deprecation_warning_is_not_a_mkdocs_diagnostic(docs_project):
    result = docs_project(
        hook="import warnings\n"
        "def on_pre_build(**kwargs):\n"
        "    warnings.simplefilter('always', DeprecationWarning)\n"
        "    warnings.warn('fixture deprecation', DeprecationWarning)\n"
    )
    assert "fixture deprecation" in result.stderr
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "fault,from_docs",
    [
        (None, False),
        ("guide-reference", False),
        ("docstring-reference", False),
        ("known-target-elsewhere", False),
        ("orphan", False),
        (None, True),
        ("guide-reference", True),
        ("line-shift", False),
        ("line-shift", True),
        ("duplicate-warning", False),
    ],
)
def test_real_site_review_cases(tmp_path, fault, from_docs):
    pytest.importorskip("material")
    pytest.importorskip("mkdocstrings")
    pytest.importorskip("pymdownx")
    shutil.copytree(
        ROOT / "docs",
        tmp_path / "docs",
        ignore=shutil.ignore_patterns("site", "__pycache__"),
    )
    shutil.copytree(
        ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    pages = tmp_path / "docs" / "docs"
    expected = "stickler.DoesNotExist"
    if fault in ("guide-reference", "known-target-elsewhere"):
        page = pages / "Guides" / "Document_Packet_Splitting.md"
        expected = "tp" if fault == "known-target-elsewhere" else expected
        with page.open("a", encoding="utf-8") as stream:
            stream.write(f"\nSee [Nope][{expected}].\n")
    elif fault == "docstring-reference":
        source = (
            tmp_path
            / "src/stickler/structured_object_evaluator/models/structured_model.py"
        )
        source.write_text(
            source.read_text(encoding="utf-8").replace(
                "Base class for models with structured comparison capabilities.",
                "Base class for models with structured comparison capabilities. See [Nope][stickler.DoesNotExist].",
            ),
            encoding="utf-8",
        )
    elif fault == "orphan":
        expected = "review-orphan.md"
        (pages / "Guides" / expected).write_text("# Orphan\n", encoding="utf-8")
    elif fault == "line-shift":
        source = (
            tmp_path
            / "src/stickler/structured_object_evaluator/models/structured_model.py"
        )
        source.write_text(
            "# comment\n" + source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    elif fault == "duplicate-warning":
        source = tmp_path / "src/stickler/structured_object_evaluator/models/field.py"
        parameter = "            **kwargs: Additional field parameters\n"
        content = source.read_text(encoding="utf-8")
        assert content.count(parameter) == 1
        source.write_text(content.replace(parameter, parameter * 2), encoding="utf-8")
        expected = "No type or annotation for parameter '**kwargs'"
    result = subprocess.run(
        [
            *BUILD_COMMAND,
            "-f",
            str(tmp_path / "docs/mkdocs.yml"),
            "-d",
            str(tmp_path / "site"),
        ],
        cwd=tmp_path / "docs" if from_docs else tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )
    output = result.stdout + result.stderr
    if fault not in (None, "line-shift"):
        assert expected in output, output
        assert result.returncode != 0, output
        if fault == "duplicate-warning":
            assert output.count("WARNING - griffe:") == 7, output
            assert "Documentation check failed: 1 warning(s) or error(s)." in output
    else:
        assert "WARNING - mkdocs_autorefs:" not in output, output
        assert result.returncode == 0, output
        assert output.count("WARNING - griffe:") == 6, output
        assert 'not included in the "nav"' not in output, output
