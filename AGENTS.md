# General Guidance for Agentic Coding Assistants

## Documentation
- When writing documentation, always try to say more with less.
- Documentation should be minimal, and only say what needs to be said to communicate how to work with and extend the system.

## MKDocs Documentation Project
- An MKDocs documentation project exists at [docs/](./docs/). More information can be found in the [docs/README.md](./docs/README.md) file.

## Testing
- Test documentation and guidelines can be found in [tests/README.md](./tests/README.md).

## Dependencies and `uv.lock`
- `uv.lock` is committed deliberately. Do not gitignore or delete it: every CI workflow runs `uv sync --frozen`, which fails immediately if the lockfile is absent.
- It does not constrain consumers of the published package. Installers resolve the `dependencies` ranges in `pyproject.toml`, which is what the wheel metadata carries; the lockfile is uv-specific and governs only this repo's dev and CI environments.
- It also carries the dependency cooldown (`exclude-newer`, `exclude-newer-span`) that the semgrep supply-chain rules check for, so removing it regresses the security scan.
- **Prefer hand-editing the lines you need over running `uv lock`.** In this repo `uv lock` rewrites `exclude-newer` to `0001-01-01` and strips platform markers from the CUDA and nvidia entries. If you do run it, diff the result and restore both. Verify any edit with `uv lock --check`, which is the authoritative consistency check.
- The lockfile records this package's own `version`, so a release bump must update it alongside `pyproject.toml` and `src/stickler/__init__.py`. See [RELEASING.md](./RELEASING.md).

## README.md 
- In addition to MKDocs, The project strives to maintain developer documentation distributed in README.md files throughout the codebase. This documentation exists to help human and AI coding assistants when working with the codebase.
- When creating directories or working in a directory that does not have a README.md create a README.md file and document the final state of the code and logic that's in the directory. Do this in language that an AI coding assistant trying to understand the implementation and codebas would understand.
- When working in a directory, always look for a README.md and/or AGENTS.md file for important context about the directory/code contained within.