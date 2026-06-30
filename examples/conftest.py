"""Pytest configuration for executing examples end-to-end.

This conftest is the single entry point that drives the example-execution CI
checks (notebooks via ``nbmake``, scripts via a custom collector) and the
local invocation ``uv run pytest examples/``. It owns:

* the two path-based registries (:data:`CREDENTIALED_EXAMPLES` and
  :data:`LONG_RUNNING_EXAMPLES`) used to gate execution per run mode,
* the :func:`pytest_collection_modifyitems` skip logic that consults those
  registries plus the ``EXAMPLES_RUN_MODE`` env var, and
* the :func:`pytest_collect_file` collector that runs each
  ``examples/scripts/*.py`` file as a standalone subprocess.

See ``examples/README.md`` for the contributor-facing documentation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from _pytest._code.code import ExceptionInfo, TerminalRepr

# ---------------------------------------------------------------------------
# Registries
#
# Each entry is a repo-relative POSIX path (forward slashes, no leading
# ``./``). Authors add a new example here when its execution requires AWS
# credentials or its typical wall-clock on ``ubuntu-latest`` exceeds the
# PR runtime budget (see ``examples/README.md``).
# ---------------------------------------------------------------------------

CREDENTIALED_EXAMPLES: frozenset[str] = frozenset(
    {
        "examples/scripts/llm_comparator_demo.py",
    }
)
"""Examples that require AWS credentials (e.g., Bedrock) to execute.

The collector skips these on PR runs and on environments without AWS
credentials. See ``examples/README.md`` for the procedure to add a new
credentialed example.
"""

LONG_RUNNING_EXAMPLES: frozenset[str] = frozenset(
    {
        "examples/scripts/bert_comparator_demo.py",
    }
)
"""Examples whose typical wall-clock on ``ubuntu-latest`` exceeds 5 minutes.

The collector skips these on every run mode except ``"scheduled"``. See
``examples/README.md`` for the procedure to add a new long-running example.
"""


# ---------------------------------------------------------------------------
# Run-mode helpers
# ---------------------------------------------------------------------------

RunMode = Literal["pr", "scheduled", "local"]


def _run_mode() -> RunMode:
    """Return the current example-execution run mode.

    Reads the ``EXAMPLES_RUN_MODE`` environment variable. Recognized values
    are ``"pr"`` (pull-request or push to ``dev``/``main``), ``"scheduled"``
    (daily cron or ``workflow_dispatch``), and ``"local"`` (a contributor
    running ``pytest examples/`` on their machine). Any unset or
    unrecognized value defaults to ``"local"``.
    """
    value = os.environ.get("EXAMPLES_RUN_MODE", "").strip().lower()
    if value in ("pr", "scheduled", "local"):
        return value  # type: ignore[return-value]
    return "local"


def _has_aws_credentials() -> bool:
    """Return ``True`` if any AWS credential indicator is present in env.

    Detects credentials by checking for any of:
    ``AWS_ACCESS_KEY_ID``, ``AWS_PROFILE``, ``AWS_WEB_IDENTITY_TOKEN_FILE``.
    A non-empty value for any of these is treated as "credentials may be
    available". This matches the documented Credential_Marker semantics:
    static, fast, no SDK calls during collection.
    """
    return any(
        os.environ.get(name)
        for name in ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_WEB_IDENTITY_TOKEN_FILE")
    )


def _item_repo_relative_path(item: pytest.Item) -> str:
    """Return a repo-relative POSIX path string for ``item``.

    Used to look an item up in :data:`CREDENTIALED_EXAMPLES` and
    :data:`LONG_RUNNING_EXAMPLES`. Falls back to the item's ``nodeid`` when
    the path cannot be made relative to the pytest rootpath (e.g., for
    synthetic items produced by a custom collector outside the repo root).
    """
    item_path = getattr(item, "path", None)
    rootpath = getattr(item.config, "rootpath", None)
    if item_path is not None and rootpath is not None:
        try:
            return item_path.relative_to(rootpath).as_posix()
        except ValueError:
            # ``item_path`` is not under ``rootpath``; fall through to nodeid.
            pass
    # ``nodeid`` is already POSIX-style and rootpath-relative; strip any
    # ``::`` test-id suffix (e.g., ``examples/notebooks/x.ipynb::``).
    return item.nodeid.split("::", 1)[0]


# ---------------------------------------------------------------------------
# Collection-time skip logic
# ---------------------------------------------------------------------------

_CREDENTIALED_SKIP_REASON = "credentialed example: requires AWS credentials"
_LONG_RUNNING_SKIP_REASON = "long-running example: scheduled runs only"


def _is_full_examples_run(config: pytest.Config) -> bool:
    """Return ``True`` when the current session is a full ``examples/`` run.

    The stale-registry warning is only meaningful when pytest had a chance
    to collect every example. On a scoped invocation (e.g.
    ``pytest examples/notebooks/`` or
    ``pytest examples/scripts/quick_start.py``) registry entries outside
    the requested scope will not match any item, so the warning would fire
    spuriously. We treat a run as "full" only when at least one positional
    argument resolves to (or contains) the ``examples/`` directory itself.
    """
    rootpath = getattr(config, "rootpath", None)
    if rootpath is None:
        return False
    examples_root = Path(rootpath) / "examples"
    if not examples_root.is_dir():
        return False

    args = list(getattr(config, "args", []))
    if not args:
        # ``pytest`` with no explicit args uses ``testpaths``; that's
        # ``["tests"]`` for this project, so registry-vs-collection
        # comparison is not meaningful.
        return False

    for arg in args:
        # Strip any pytest ``::`` test-id suffix before resolving.
        raw = str(arg).split("::", 1)[0]
        try:
            resolved = Path(raw).resolve()
        except OSError:
            continue
        try:
            resolved.relative_to(examples_root)
        except ValueError:
            # ``resolved`` is not within ``examples/``; ignore.
            continue
        if resolved == examples_root:
            return True
    return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply credentialed and long-running skip marks during collection.

    Behavior:

    * Items whose repo-relative path is in :data:`CREDENTIALED_EXAMPLES` are
      skipped on PR runs and on any environment without AWS credentials.
      On ``scheduled`` or ``local`` runs with credentials present, they are
      left untouched so they execute normally.
    * Items whose repo-relative path is in :data:`LONG_RUNNING_EXAMPLES` are
      skipped on every run mode except ``"scheduled"``.
    * Registry entries that do not match any collected item produce a
      :class:`pytest.PytestConfigWarning` so stale entries are visible. The
      warning is suppressed on scoped runs (e.g.,
      ``pytest examples/notebooks/``) where a non-match is expected.
    """
    run_mode = _run_mode()
    has_creds = _has_aws_credentials()
    skip_credentialed = run_mode == "pr" or not has_creds
    skip_long_running = run_mode != "scheduled"

    matched_credentialed: set[str] = set()
    matched_long_running: set[str] = set()

    for item in items:
        path = _item_repo_relative_path(item)
        if path in CREDENTIALED_EXAMPLES:
            matched_credentialed.add(path)
            if skip_credentialed:
                item.add_marker(pytest.mark.skip(reason=_CREDENTIALED_SKIP_REASON))
        if path in LONG_RUNNING_EXAMPLES:
            matched_long_running.add(path)
            if skip_long_running:
                item.add_marker(pytest.mark.skip(reason=_LONG_RUNNING_SKIP_REASON))

    if not _is_full_examples_run(config):
        return

    for stale in sorted(CREDENTIALED_EXAMPLES - matched_credentialed):
        warnings.warn(
            f"Registry entry '{stale}' does not match any example file; remove it",
            pytest.PytestConfigWarning,
            stacklevel=1,
        )
    for stale in sorted(LONG_RUNNING_EXAMPLES - matched_long_running):
        warnings.warn(
            f"Registry entry '{stale}' does not match any example file; remove it",
            pytest.PytestConfigWarning,
            stacklevel=1,
        )


# ---------------------------------------------------------------------------
# Script collector
#
# Collects every ``examples/scripts/*.py`` file (excluding files starting
# with ``_`` and ``__init__.py``) and runs each as a standalone subprocess
# using the configured Python interpreter. On non-zero exit the test fails
# with a readable message containing the path, exit code, and stderr; on
# subprocess timeout the test fails with a clear timeout message.
# ---------------------------------------------------------------------------

_SCRIPT_TIMEOUT_SECONDS = 600
_SCRIPTS_DIRNAME = "scripts"


def _is_collectible_script(file_path: Path) -> bool:
    """Return ``True`` if ``file_path`` is a script the collector should run.

    A file is collectible when it lives directly under an ``examples/scripts``
    directory, has a ``.py`` suffix, is not ``__init__.py``, and does not
    start with an underscore (so authors can keep ``_helpers.py`` modules
    next to their scripts without those modules being executed standalone).
    """
    if file_path.suffix != ".py":
        return False
    if file_path.parent.name != _SCRIPTS_DIRNAME:
        return False
    name = file_path.name
    if name == "__init__.py":
        return False
    if name.startswith("_"):
        return False
    return True


def _decode_captured(stream: str | bytes | None) -> str:
    """Decode a captured stdio stream to ``str`` for use in failure messages."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


class ScriptItem(pytest.Item):
    """A pytest item that runs a single example script as a subprocess."""

    def runtest(self) -> None:
        script_path = Path(self.path)
        rel_path = self._repo_relative_path(script_path)
        # Run scripts from their own directory so any incidental file
        # output (e.g., generated HTML reports, JSON dumps) lands next to
        # the script rather than dirtying the working tree of the local
        # ``pytest examples/`` invocation.
        cwd = script_path.parent
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_SCRIPT_TIMEOUT_SECONDS,
                cwd=str(cwd),
            )
        except subprocess.TimeoutExpired as exc:
            message = (
                f"{rel_path}: timed out after {_SCRIPT_TIMEOUT_SECONDS}s\n"
                f"--- stdout ---\n{_decode_captured(exc.stdout)}"
                f"--- stderr ---\n{_decode_captured(exc.stderr)}"
            )
            raise pytest.fail.Exception(message, pytrace=False) from None

        if result.returncode != 0:
            # Many demo scripts print error context to stdout (e.g., the
            # ``bert_comparator_demo`` "Missing dependency" message), so
            # we surface both streams to make CI failures self-diagnosing.
            message = (
                f"{rel_path} exited with code {result.returncode}\n"
                f"--- stdout ---\n{result.stdout}"
                f"--- stderr ---\n{result.stderr}"
            )
            raise AssertionError(message)

    def repr_failure(  # type: ignore[override]
        self,
        excinfo: ExceptionInfo[BaseException],
        style: str | None = None,
    ) -> str | TerminalRepr:
        """Render subprocess failures without a full Python traceback.

        For ``AssertionError`` (non-zero exit) and ``pytest.fail.Exception``
        (timeout), we surface only the message we constructed in
        :meth:`runtest`. Any other exception bubbles up through pytest's
        default representation so genuine collector bugs remain debuggable.
        """
        exc = excinfo.value
        if isinstance(exc, (AssertionError, pytest.fail.Exception)):
            return str(exc)
        return super().repr_failure(excinfo, style)

    def reportinfo(self) -> tuple[Path, int | None, str]:
        return self.path, 0, self._repo_relative_path(Path(self.path))

    def _repo_relative_path(self, script_path: Path) -> str:
        rootpath = getattr(self.config, "rootpath", None)
        if rootpath is not None:
            try:
                return script_path.relative_to(rootpath).as_posix()
            except ValueError:
                pass
        return script_path.as_posix()


class ScriptFile(pytest.File):
    """A pytest collector that yields a single :class:`ScriptItem`."""

    def collect(self):  # type: ignore[override]
        yield ScriptItem.from_parent(parent=self, name=self.path.name)


def pytest_collect_file(
    parent: pytest.Collector, file_path: Path
) -> pytest.Collector | None:
    """Collect ``examples/scripts/*.py`` files via :class:`ScriptFile`."""
    if not _is_collectible_script(file_path):
        return None
    return ScriptFile.from_parent(parent=parent, path=file_path)
