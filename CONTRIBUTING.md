# Contributing

Contributions are welcome when they improve the package as reusable research
software for protein-structure feature generation and flexibility analysis.

## Ways to contribute

- Report bugs in PDB parsing, PSL feature generation, packaging, tests, or
  documentation.
- Propose examples that make the software easier to run on independent protein
  structures.
- Add focused tests for edge cases such as alternate PDB formatting,
  single-residue neighborhoods, or additional PSL degree options.
- Improve documentation for installation, feature interpretation, or manuscript
  reproduction.

Please open an issue before starting larger design changes so the scope can be
discussed first.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[test]"
pytest
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[test]"
pytest
```

## Pull request checklist

Before opening a pull request, please check that:

- The change is scoped to a clear user or reviewer need.
- New behavior has tests or a documented manual verification path.
- Generated feature matrices, raw benchmark data, caches, virtual
  environments, and full prediction tables are not committed.
- Documentation is updated when commands, outputs, or public APIs change.
- `git diff --check` passes.

## Coding style

The package uses typed, dependency-light Python. Prefer small functions with
clear inputs and outputs over broad script-only logic. Use NumPy arrays for
numeric data and keep data parsing, feature generation, and modeling concerns
separate.

The repository includes Ruff, mypy, and pre-commit configuration for maintainers
who want stricter local checks:

```bash
pip install -e ".[test]"
pip install pre-commit ruff mypy
pre-commit run --all-files
```

## Data policy

Do not commit third-party raw protein datasets, generated PSL matrices, or full
prediction outputs unless the maintainer explicitly decides to add them through
an appropriate archival or large-file workflow. Small synthetic examples used
for tests and documentation are appropriate.

## Review and release process

Small fixes can be merged after passing tests and maintainer review. Larger
changes should be discussed in an issue and should include a short design note
in the pull request.

Releases are expected to be tagged from a clean commit after tests pass. Update
`CHANGELOG.md`, confirm the package version in `pyproject.toml`, and archive the
release if a DOI is needed for citation.
