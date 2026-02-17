# General Guidance for Agentic Coding Assistants

## Documentation
- When writing documentation, always try to say more with less.
- Documentation should be minimal, and only say what needs to be said to communicate how to work with and extend the system.

## MKDocs Documentation Project
- An MKDocs documentation project exists at [docs/](./docs/). More information can be found in the [docs/README.md](./docs/README.md) and [docs/AGENTS.md](./docs/AGENTS.md) files.

## Testing
- Test documentation and guidelines can be found in [tests/README.md](./tests/README.md).

## README.md 
- In addition to MKDocs, The project strives to maintain developer documentation distributed in README.md files throughout the codebase. This documentation exists to help human and AI coding assistants when working with the codebase.
- When creating directories or working in a directory that does not have a README.md create a README.md file and document the final state of the code and logic that's in the directory. Do this in language that an AI coding assistant trying to understand the implementation and codebas would understand.
- When working in a directory, always look for a README.md and/or AGENTS.md file for important context about the directory/code contained within.