# PSL Protein Flexibility

`psl-protein-flexibility` generates persistent sheaf Laplacian (PSL)
descriptors from protein C-alpha geometry for residue-level flexibility
modeling. The package can be used on a directory of ordinary PDB files, writes
per-residue feature tables with residue identifiers and B-factors, and includes
a small synthetic demo that runs without downloading the MDG_bfactor benchmark
dataset.

The repository also contains the scripts and metric snapshots used for the
accompanying protein-flexibility study, but the maintained software path is now
self-contained: no upstream `PSL.py` file or external research-code checkout is
required.

## What the software does

- Parses C-alpha coordinates and B-factors from PDB files.
- Builds local center-labeled or constant-sheaf PSL descriptors around each
  residue at user-selected radii.
- Uses a native distance-threshold simplicial complex implementation with
  degree 0, 1, and 2 Laplacian accessors.
- Writes reviewer-friendly CSV outputs containing protein ID, residue metadata,
  raw B-factor, within-protein z-scored B-factor, and PSL feature columns.
- Provides a bundled toy demo with synthetic PDB files and a tiny
  leave-one-protein-out ridge-regression smoke test.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[test]"
pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run the toy demo

The demo uses three synthetic PDB files under `examples/toy_proteins/`.

```bash
python scripts/run_toy_demo.py
```

This writes:

```text
data/toy/processed/
|-- toy_psl_features.csv
|-- toy_demo_metrics.csv
|-- toy_demo_summary.json
|-- feature_config.json
`-- feature_names.json
```

The demo is a smoke test, not a scientific benchmark. It verifies that a fresh
checkout can parse PDB files, compute PSL features, and run a small grouped
modeling workflow without the full MDG_bfactor data.

## Generate features for your own proteins

Place PDB files in a directory and run:

```bash
python scripts/compute_psl_features.py \
  --pdb-dir path/to/pdb_files \
  --out-dir data/my_psl_features \
  --radii 6,9,12 \
  --sheaf center_labeled \
  --stats both
```

The same functionality is available after installation as:

```bash
psl-flexibility features \
  --pdb-dir path/to/pdb_files \
  --out-dir data/my_psl_features
```

Important options:

| Option | Meaning |
|---|---|
| `--radii 6,9,12` | Neighborhood radii in Angstrom. |
| `--sheaf center_labeled` | Label the target residue as 0 and neighbors as 1. Use `constant` for a constant-sheaf baseline. |
| `--stats both` | Write max, min, mean, median, standard deviation, and zero-eigenvalue count. |
| `--degrees 0` | PSL degrees to summarize. Degrees `1` and `2` are available for exploratory use. |
| `--p-widths 0.0` | Optional distance-weight damping values for sensitivity checks. |

The main CSV output is `psl_features.csv`. Each row is one C-alpha residue with
the columns:

```text
protein_id,residue_index,chain_id,residue_number,insertion_code,residue_name,
b_factor,z_b_factor,<PSL feature columns...>
```

## Python API

```python
from pathlib import Path

from psl_flexibility.features import FeatureConfig, feature_names, features_for_residues
from psl_flexibility.structure import parse_ca_pdb

records = parse_ca_pdb(Path("examples/toy_proteins/toy_alpha.pdb"))
config = FeatureConfig(radii=(6.0, 9.0, 12.0), sheaf="center_labeled", stats="both")
features = features_for_residues(records, config)
names = feature_names(config)
```

## Native PSL implementation

The original analysis depended on a separate research implementation of
persistent sheaf Laplacians. To keep this repository redistributable and useful
for JOSS review, the needed feature-generation path is now implemented natively
in `psl_flexibility.native_psl`.

The native implementation intentionally focuses on the repository's required
use case: local protein point clouds and spectral summaries at fixed radii. It
accepts the small compatibility surface used by the original scripts
(`build_filtration`, `build_simplicial_pair`, `build_matrices`, `psl_0`,
`psl_1`, and `psl_2`) and constructs a Euclidean distance-threshold
Vietoris-Rips style complex. `filtration_type="alpha"` is accepted by the
compatibility class as an alias, but no external GUDHI or upstream `PSL.py`
dependency is required.

## Repository layout

```text
.
|-- README.md
|-- pyproject.toml
|-- CONTRIBUTING.md
|-- SUPPORT.md
|-- GOVERNANCE.md
|-- CHANGELOG.md
|-- paper/
|   |-- paper.md
|   `-- paper.bib
|-- examples/
|   `-- toy_proteins/
|-- scripts/
|   |-- compute_psl_features.py
|   |-- run_toy_demo.py
|   |-- generate_psl_features.py
|   |-- psl_variant_experiment.py
|   |-- run_rf.py
|   |-- run_baselines.py
|   |-- hayes_protocol_runner.py
|   |-- run_protocol_exactness_checks.py
|   `-- run_biological_validation.py
|-- src/
|   `-- psl_flexibility/
|       |-- cli.py
|       |-- demo.py
|       |-- features.py
|       |-- metrics.py
|       |-- native_psl.py
|       |-- paths.py
|       `-- structure.py
|-- tests/
`-- results/
    `-- metrics/
```

Generated feature matrices, prediction tables, and toy-demo outputs should stay
out of Git history.

## Community and maintenance

This is currently a single-maintainer research software project with public
issue tracking, CI, tests, and documented contribution paths.

- Use `CONTRIBUTING.md` for development setup, pull request expectations, data
  policy, and release checklist.
- Use `SUPPORT.md` for help and bug-reporting expectations.
- Use `GOVERNANCE.md` for project scope, decision making, and release process.
- Use `CODE_OF_CONDUCT.md` for conduct expectations.
- Use `CHANGELOG.md` for release notes and notable source-level changes.

## Reproducing the manuscript analyses

The full study used the MDG_bfactor dataset, which is not redistributed here.
To reproduce those benchmark analyses, place the dataset at:

```text
MDG_bfactor-main/
  MDG_bfactor-main/
    datasets/
      list-small.txt
      list-medium.txt
      list-large.txt
      list-365.txt
      365/*.pdb
    features/
      features-blind-prediction/*.csv
```

Generate the main center-labeled PSL feature set:

```bash
python scripts/psl_variant_experiment.py \
  --root . \
  --dataset 365 \
  --sheaf center_labeled \
  --radii 6,9,12 \
  --stat4 both \
  --degrees 0 \
  --out-dir data/processed/features_psl_labeled_6_9_12_both
```

Run the main grouped protein-level Random Forest evaluation:

```bash
python scripts/run_rf.py \
  --feature-source psl \
  --psl-dir data/processed/features_psl_labeled_6_9_12_both \
  --run-name psl_labeled_6_9_12_both_rf2000_depth12_leaf2_sqrt \
  --folds 5 \
  --n-estimators 2000 \
  --max-depth 12 \
  --min-samples-leaf 2 \
  --max-features sqrt \
  --n-jobs 1 \
  --no-shap
```

Additional scripts reproduce the baseline comparisons, biological validation,
SHAP attribution workflow, and Hayes-style protocol checks described in
`paper/paper.md`.

## Curated metric snapshots

The tracked files under `results/metrics/` summarize the manuscript analyses:

| Analysis | Main result |
|---|---:|
| Main center-labeled PSL, grouped protein CV, within-protein normalized B | mean per-protein PCC 0.602; pooled PCC 0.571 |
| PSL+SHAP attribution model | absolute-attribution PCC 0.482; top-20% Jaccard 0.338 |
| Baseline, aligned 274-protein subset | PSL 0.608, classical structural 0.575, simple graph 0.629, classical+PSL 0.627 |
| Hayes Table 1-style PSL-only audit | pooled OLS 0.520-0.555; per-protein OLS 0.707 |
| Hayes blind-style raw-B protein split | GBDT 0.605; RF 0.600 mean per-protein PCC |
| Hayes random residue raw-B split | GBDT 0.853; RF 0.898 pooled PCC |
| Random-residue aggregation audit | mean per-protein PCC only 0.606 and 0.636 |
| Split-dependence controls | classical and graph features also reach pooled random-residue PCC near 0.85 |

Random residue-level pooled PCC should not be interpreted as
protein-independent generalization. The manuscript and analysis scripts report
protein-grouped metrics separately for that reason.

## Citation

Use `CITATION.cff` when citing this repository. The JOSS manuscript source is
in `paper/paper.md`, with references in `paper/paper.bib`.
