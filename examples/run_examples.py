#!/usr/bin/env python
"""Execute every example end to end, so library changes cannot silently break them.

The examples are onboarding material and they are not exercised by CI (#118), so
a removed dict key or a renamed argument stays invisible until a user hits it.
Removing `all_fields_matched` broke five scripts and three notebooks, including
`Quick_start`, and nothing caught it.

Usage::

    python examples/run_examples.py                 # everything that needs no credentials
    AWS_PROFILE=myprofile python examples/run_examples.py --aws   # add the AWS ones
    python examples/run_examples.py --only quick_start            # substring filter
    python examples/run_examples.py --list                        # show what would run

Exit code is 1 if any example fails, so this is usable as a gate.

Design note: examples are assumed to need nothing unless they are named in
``NEEDS_AWS`` or ``NEEDS_BERT`` below. That direction is deliberate. A new example
that quietly reaches for Bedrock will **fail** the default run rather than being
skipped, which is what surfaces it. The alternative, scanning source for `boto3`,
silently skips anything whose import is spelled unusually.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

# Examples that call a live AWS service. Keep this list short and justified;
# anything here is invisible to the default run.
NEEDS_AWS = {
    "bert_comparator_demo.py",       # SemanticComparator -> Bedrock embeddings
    "llm_comparator_demo.py",        # LLMComparator -> Bedrock
    "strands_agent_eval_demo.py",    # Strands agent -> Bedrock
}

# Examples that download a model from the Hugging Face Hub at import time.
# `stickler.comparators.bert` calls `evaluate.load()` at module scope, so these
# need network access and the `bert` extra even before they do anything.
NEEDS_BERT = {
    "bert_comparator_demo.py",
}


def discover() -> list[Path]:
    notebooks = sorted((EXAMPLES / "notebooks").glob("*.ipynb"))
    scripts = sorted(p for p in (EXAMPLES / "scripts").glob("*.py") if p.name != "__init__.py")
    return notebooks + scripts


def requirements(path: Path) -> set[str]:
    needs = set()
    if path.name in NEEDS_AWS:
        needs.add("aws")
    if path.name in NEEDS_BERT:
        needs.add("bert")
    if path.suffix == ".ipynb" and not notebook_deps_available():
        needs.add("nbclient")
    return needs


def run_script(path: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode == 0:
        return True, ""
    # The last few lines carry the exception; the rest is the example's own output.
    tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
    return False, tail


def notebook_deps_available() -> bool:
    """Whether the notebooks can be executed in this environment.

    Deliberately not a project dependency. Executing a notebook needs a kernel,
    and `ipykernel` drags in roughly thirty packages (ipython, jedi, pyzmq,
    tornado, debugpy). Paying that in every contributor's lockfile to run eleven
    example notebooks is the wrong trade, so notebooks are skipped with a hint
    unless someone opts in:

        uv pip install nbclient nbformat ipykernel

    The scripts, which are the majority, need nothing extra.
    """
    try:
        import ipykernel  # noqa: F401
        import nbclient  # noqa: F401
        import nbformat  # noqa: F401
    except ImportError:
        return False
    return True


def run_notebook(path: Path) -> tuple[bool, str]:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        allow_errors=False,
        # Execute with the repo root as cwd so relative data paths resolve the
        # same way they do for a reader running cells by hand.
        resources={"metadata": {"path": str(path.parent)}},
    )
    # The kernel is a subprocess and logs an unencrypted-TCP warning to *its*
    # stderr on every start, which is inherited and drowns the result lines.
    # Python-level logging config cannot reach it, so redirect the fd for the
    # duration of the run and restore it afterwards.
    saved_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        client.execute()
    except CellExecutionError as exc:
        return False, "\n".join(str(exc).strip().splitlines()[-6:])
    except Exception as exc:  # kernel start failures, timeouts
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull)
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aws", action="store_true", help="also run examples needing AWS credentials")
    parser.add_argument("--bert", action="store_true", help="also run examples needing the bert extra")
    parser.add_argument("--only", metavar="SUBSTRING", help="run only examples whose name contains this")
    parser.add_argument("--list", action="store_true", help="show what would run, then exit")
    args = parser.parse_args()

    enabled = set()
    if args.aws:
        enabled.add("aws")
    if args.bert:
        enabled.update({"bert", "aws"})  # the bert demo also needs Bedrock

    if "aws" in enabled and not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
        print("--aws given but neither AWS_PROFILE nor AWS_ACCESS_KEY_ID is set.", file=sys.stderr)
        return 2

    selected, skipped = [], []
    for path in discover():
        if args.only and args.only not in path.name:
            continue
        needs = requirements(path)
        if needs - enabled:
            skipped.append((path, sorted(needs - enabled)))
        else:
            selected.append(path)

    if args.list:
        for p in selected:
            print(f"  run   {p.relative_to(REPO_ROOT)}")
        for p, why in skipped:
            print(f"  skip  {p.relative_to(REPO_ROOT)}  (needs {', '.join(why)})")
        return 0

    failures = []
    for path in selected:
        label = str(path.relative_to(REPO_ROOT))
        print(f"  {label:58}", end="", flush=True)
        started = time.monotonic()
        ok, detail = run_notebook(path) if path.suffix == ".ipynb" else run_script(path)
        elapsed = time.monotonic() - started
        print(f"{'ok' if ok else 'FAIL':>6}  {elapsed:5.1f}s")
        if not ok:
            failures.append((label, detail))

    for path, why in skipped:
        print(f"  {str(path.relative_to(REPO_ROOT)):58}{'skip':>6}  needs {', '.join(why)}")

    print()
    print(f"  {len(selected) - len(failures)} passed, {len(failures)} failed, {len(skipped)} skipped")

    if any("nbclient" in why for _, why in skipped):
        print("\n  To run the notebooks too:  uv pip install nbclient nbformat ipykernel")

    for label, detail in failures:
        print(f"\n--- {label} ---\n{detail}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
