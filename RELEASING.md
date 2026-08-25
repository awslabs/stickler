# Releasing Stickler

How to ship a new version of `stickler-eval` to PyPI. The whole process takes
about 30 minutes plus CI time.

**Flow:** version bump on `dev` → release PR (`dev` → `main`) → squash merge →
GitHub Release with tag → automated PyPI publish.

## Prerequisites

- Maintainer access to `awslabs/stickler` (push to `dev`, merge to `main`,
  create Releases)
- CI green on `dev` — do not release on a red build
- [`gh` CLI](https://cli.github.com/) authenticated

## 1. Decide the version

Semantic versioning, judged by *content since the last release*
(`git log --oneline origin/main..origin/dev` or the
[compare view](https://github.com/awslabs/stickler/compare/main...dev)):

| Bump | When |
|------|------|
| **patch** (0.X.Y → 0.X.Y+1) | Only fixes, dependency bumps, docs — no new public API |
| **minor** (0.X → 0.X+1.0) | Additive features, no breaking changes (most common) |
| **major** | Breaking API change, or an explicit "this is stable" decision |

Check for unmerged dependabot security PRs first — consider merging them so
they ride the same release.

## 2. Bump the version on `dev`

The bump commit goes directly on `dev`, not a feature branch, so it lands
inside the release squash.

```bash
git checkout dev && git pull origin dev --ff-only
```

Edit **all three** files (they must always match):

- `pyproject.toml` → `version = "X.Y.Z"`
- `src/stickler/__init__.py` → `__version__ = "X.Y.Z"`
- `uv.lock` → the `version` line in the `name = "stickler-eval"` entry

The lockfile records this package's own version, not just its dependencies, so
leaving it behind fails CI: `uv lock --check` reports "the lockfile needs to be
updated", and every workflow runs `uv sync --frozen`.

Patch that one line **by hand**. Do not run `uv lock` to pick it up: in this
repo it also rewrites `exclude-newer` to `0001-01-01` and strips platform
markers from the CUDA and nvidia entries. See
[AGENTS.md](./AGENTS.md#dependencies-and-uvlock).

```bash
uv lock --check          # must pass before committing
git add pyproject.toml src/stickler/__init__.py uv.lock
git commit -m "chore: bump version to X.Y.Z for release"
git push origin dev
```

Also add a `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md` (move entries
out of `[Unreleased]`) in the same commit.

### What belongs in the changelog

Entries are for changes a user can observe. A PR needs one when it changes
behaviour, an exported format, a public signature, a default, or a dependency,
and when it fixes a bug someone could have hit.

Internal refactors with no runtime change do **not** get an entry: dead-code
removal, moving a private helper, renaming something not exported. `git log` is
the record for those. This is the existing convention, made explicit here
because it had been applied by default rather than by decision
([#213](https://github.com/awslabs/stickler/issues/213)).

When in doubt, ask whether a user reading the release notes could act on it. If
not, leave it out.

## 3. Open the release PR

```bash
gh pr create --base main --head dev --title "release(vX.Y.Z): <short summary>"
```

PR body structure (see past releases [#131](https://github.com/awslabs/stickler/pull/131),
[#102](https://github.com/awslabs/stickler/pull/102) for examples):

```markdown
## Summary
- **feat**: <one bullet per feature, with PR refs (#N)>
- **fix**: <one bullet per fix, with PR refs>
- **chore(deps)**: <dependency bumps; name CVEs explicitly>
- **docs**: <doc-only changes>

## Version
`<old>` → `<new>` (<bump-type> — <one-line justification>; breaking or not)

## Test plan
- [ ] CI passes (lint, tests, security)
- [ ] Verify PyPI publish succeeds after tagging vX.Y.Z
```

One bullet per merged PR, grouped by conventional-commit type. Get the PR list
with `gh pr list --state merged --base dev` filtered to PRs merged after the
previous release tag's date.

## 4. Merge

Wait for CI green, get a review if the branch protection requires one, then
**squash and merge** into `main`.

After merging, sync `dev` with `main` (the squash creates a new commit on
`main` that `dev` doesn't have):

```bash
git checkout main && git pull origin main
git checkout dev && git merge main && git push origin dev
```

## 5. Create the GitHub Release

At <https://github.com/awslabs/stickler/releases/new>:

- **Tag**: `vX.Y.Z` (exactly this format — created on publish), target `main`
- **Title**: `vX.Y.Z release`
- **Body**: Highlights paragraph, then `### New Features` / `### Bug Fixes` /
  `### Security & Dependencies` / `### Documentation` / `### What's Changed`
  sections with PR links. Match the style of the
  [v0.5.0 release](https://github.com/awslabs/stickler/releases/tag/v0.5.0).
- Leave "Set as latest release" checked.

> **Careful:** publishing the Release fires the PyPI publish workflow and is
> **not idempotent** — you cannot re-publish the same version. Get the tag
> name right the first time.

## 6. Verify the publish

The `workflow.yml` GitHub Action fires on `release: published`: builds with
`uv build`, publishes to TestPyPI and PyPI via OIDC trusted publishing (no
API tokens involved — publishing rights are tied to the `release` and
`testpypi` GitHub environments).

Watch it at <https://github.com/awslabs/stickler/actions>, then confirm:

```bash
curl -s https://pypi.org/pypi/stickler-eval/json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])"
uv venv /tmp/stickler-check && VIRTUAL_ENV=/tmp/stickler-check uv pip install stickler-eval==X.Y.Z
/tmp/stickler-check/bin/python -c "import stickler; print(stickler.__version__)"
```

## Troubleshooting

- **Publish workflow failed after the Release was created**: fix the cause,
  then re-run the failed workflow run from the Actions UI (the release event
  is still attached). If the version was partially published to PyPI, you must
  bump to a new patch version — PyPI never allows re-uploading a version.
- **Version mismatch between `pyproject.toml` and `__init__.py`**: fix both to
  the target version in a single commit before releasing.
- **Hotfix without releasing `dev`**: branch from `main`, bump the patch
  version on the hotfix branch, PR back to `main`, release, then merge `main`
  into `dev`.

## Historical tag quirks

Tags `v.0.1.3` and `v.0.1.4` have a stray dot (legacy inconsistency). All
tags from `v0.1.5` onward follow `vX.Y.Z` — keep it that way.
