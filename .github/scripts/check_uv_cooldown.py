"""Fail when the uv dependency cooldown is weakened (stdlib only; run with python3 -I).

uv applies `[tool.uv] exclude-newer = "7 days"` only when it resolves; CI's
`uv sync --frozen` installs whatever uv.lock holds. So check the settings, and
that no registry entry added or changed since BASE_LOCK was uploaded inside the
window unless pyproject carries a dated, advisory-linked emergency exemption:

    # https://github.com/advisories/GHSA-xxxx-xxxx-xxxx
    exclude-newer-package = { somepkg = "2026-11-10T00:00:00Z" }

Usage: check_uv_cooldown.py BASE_LOCK [--stale-exemption-warns]
A missing BASE_LOCK file counts every entry as new. --stale-exemption-warns
downgrades an expired exemption to a warning (PRs that touch neither
pyproject.toml nor uv.lock, so forks are not blocked by someone else's cleanup).

Needs Python 3.11+ (tomllib). Not guarded: overrides set in workflows, such as
`--exclude-newer-package` in run_pytest.yaml or a `UV_EXCLUDE_NEWER` env var.
"""

import datetime as dt
import os
import re
import sys
import tomllib

WINDOW = dt.timedelta(days=7)
NOW = dt.datetime.now(dt.timezone.utc)
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
SPAN = re.compile(r"P\w+|\d[\d\s]*[a-zA-Z][\w\s]*")
ADVISORY = re.compile(r"https://\S*(GHSA-[\w-]+|CVE-\d{4}-\d+)", re.I)
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()  # noqa: E731 (PEP 503)


def when(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def stamp(v):
    """A strict YYYY-MM-DDTHH:MM:SSZ exemption date, else None."""
    try:
        return when(v) if STAMP.fullmatch(v) else None
    except ValueError:
        return None


def entries(path):
    """(name, version) -> (file signature, newest upload time, is registry-sourced)."""
    out = {}
    for p in tomllib.load(open(path, "rb")).get("package", []):
        files = [p.get("sdist", {}), *p.get("wheels", [])]
        times = [when(f.get("upload-time")) for f in files if f.get("upload-time")]
        sig = sorted((f.get("url", ""), f.get("hash", "")) for f in files)
        registry = "registry" in p.get("source", {})
        out[(norm(p["name"]), p.get("version"))] = (
            sig,
            max(times, default=None),
            registry,
        )
    return out


def main(base_lock, stale_warns=False):
    err, warn, exempt = [], [], {}
    text = open("pyproject.toml", encoding="utf-8").read()
    uv = tomllib.loads(text).get("tool", {}).get("uv", {})
    if uv.get("exclude-newer") != "7 days":
        err.append('pyproject.toml: [tool.uv] exclude-newer must be "7 days"')
    elif next(iter(uv)) != "exclude-newer":
        err.append(
            "pyproject.toml: exclude-newer must be the first key in [tool.uv] (semgrep uv-missing-dependency-cooldown)"
        )
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "[tool.uv]"), 0)
    for pkg, val in (uv.get("exclude-newer-package") or {}).items():
        where = f"pyproject.toml: exclude-newer-package {pkg} = {val!r}"
        # The exemption's line (inline comment included) and the line above it.
        key = re.compile(rf'\b"?{re.escape(pkg)}"?\s*=')
        i = next(
            (
                i
                for i, ln in enumerate(lines)
                if i > start and key.search(ln.split("#")[0])
            ),
            0,
        )
        notes = " ".join(lines[max(i - 1, 0) : i + 1])
        if not isinstance(val, str) or SPAN.fullmatch(val.strip()):
            err.append(
                f"{where} is a permanent weakening; use a YYYY-MM-DDTHH:MM:SSZ timestamp"
            )
        elif not stamp(val):
            err.append(f"{where} is malformed; use a YYYY-MM-DDTHH:MM:SSZ timestamp")
        elif stamp(val) > NOW:
            err.append(f"{where} is in the future; date it no later than now")
        elif stamp(val) < NOW - WINDOW:
            msg = f"{where} is over 7 days old; remove it and uv.lock's [options.exclude-newer-package] entry"
            (warn if stale_warns else err).append(msg)
        elif not ADVISORY.search(notes):
            err.append(
                f"{where} needs a comment above it linking its GHSA/CVE advisory"
            )
        else:
            exempt[norm(pkg)] = stamp(val)
    if os.path.exists("uv.toml"):
        err.append("uv.toml: it overrides [tool.uv]; remove it")
    if (
        tomllib.load(open("uv.lock", "rb")).get("options", {}).get("exclude-newer-span")
        != "P7D"
    ):
        err.append('uv.lock: [options] exclude-newer-span must be "P7D"')
    base = entries(base_lock) if os.path.exists(base_lock) else {}
    for (name, ver), (sig, newest, registry) in sorted(entries("uv.lock").items()):
        # Git, path, editable and virtual sources have no upload time and are outside uv's cooldown.
        if not registry or base.get((name, ver), (None,))[0] == sig:
            continue
        if newest is None:
            err.append(
                f"uv.lock: {name} {ver} was added or changed but has no upload-time to check"
            )
        elif newest <= NOW - WINDOW:
            continue
        elif name in exempt and newest <= exempt[name]:
            print(
                f"{name} {ver} ({newest:%Y-%m-%dT%H:%MZ}) admitted by its emergency exemption"
            )
        else:
            err.append(
                f"uv.lock: {name} {ver} was added or changed and uploaded {newest:%Y-%m-%dT%H:%MZ}, inside the 7-day window"
            )
    for w in warn:
        print(f"::warning::{w}")
    for e in err:
        print(f"::error::{e}")
    if err:
        sys.exit(
            "Dependency cooldown weakened. Restore the setting, or hand-edit uv.lock back to "
            "versions older than 7 days, and verify with `uv lock --check` (see AGENTS.md)."
        )
    print(
        f"uv cooldown intact: exclude-newer = 7 days, span P7D, no unexempted lock change newer than {(NOW - WINDOW):%Y-%m-%d}"
    )


if __name__ == "__main__":
    main(sys.argv[1], "--stale-exemption-warns" in sys.argv[2:])
