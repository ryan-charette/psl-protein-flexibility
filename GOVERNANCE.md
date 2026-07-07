# Governance

`psl-protein-flexibility` is currently maintained by Ryan Charette.

## Decision making

The maintainer makes final decisions about project scope, releases, and API
changes. Design decisions should prioritize:

- Reusable PSL feature generation for independent protein structures.
- Reproducible benchmark scripts and documented outputs.
- Minimal, well-understood dependencies.
- Clear data provenance and redistribution boundaries.
- Tests and examples that reviewers and users can run without private data.

## Project scope

In scope:

- PDB C-alpha parsing and residue-level metadata handling.
- Native PSL descriptor generation for local protein point clouds.
- Small examples, tests, and command-line workflows.
- Reproduction scripts for the accompanying benchmark analyses.

Out of scope:

- Redistributing third-party raw benchmark datasets.
- Maintaining a general-purpose molecular dynamics toolkit.
- Adding large model artifacts or generated benchmark matrices to Git history.
- Supporting every PDB edge case without a reproducible example.

## Releases

Releases should be made from tagged commits after tests pass. Each release
should have:

- An updated `CHANGELOG.md`.
- A version recorded in `pyproject.toml`.
- Installation and example commands checked against a clean environment.
- Any archival metadata needed for citation.

## Contributor roles

Contributors retain credit through Git history and pull request discussions.
Substantial sustained contributions can be recognized in project documentation
or future manuscript authorship, following normal scholarly authorship
expectations.
