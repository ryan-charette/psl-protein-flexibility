# PSL Protein Flexibility

This repository accompanies the manuscript "Protocol-Dependent Evaluation of Persistent Sheaf Laplacian Descriptors for Explainable Protein Flexibility Prediction".

The project evaluates persistent sheaf Laplacian (PSL) descriptors for residue-level crystallographic B-factor prediction. It emphasizes protein-level validation, Hayes-style protocol comparability, SHAP attribution diagnostics, and biological validation against solvent exposure, packing, termini, and secondary-structure annotations.

## Main claims reproduced by this repository

The curated metric snapshots in `results/metrics/` summarize the final analyses:

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

The repository is intentionally structured to keep the main protein-level generalization result separate from raw-B-factor protocol comparability checks.

## Repository layout

```text
.
|-- README.md
|-- requirements.txt
|-- CITATION.cff
|-- LICENSE
|-- NOTICE.md
|-- data/
|   `-- README.md
|-- results/
|   `-- metrics/
|-- scripts/
|   |-- generate_psl_features.py
|   |-- psl_variant_experiment.py
|   |-- run_rf.py
|   |-- run_baselines.py
|   |-- hayes_protocol_runner.py
|   |-- run_protocol_exactness_checks.py
|   `-- run_biological_validation.py
|-- src/
|   `-- psl_flexibility/
|       |-- metrics.py
|       |-- paths.py
|       `-- vendor/PSL.py
`-- tests/
```

Large derived files are not tracked. Generated PSL matrices, full prediction tables, and raw third-party data should stay outside Git history.

## Data and third-party code requirements

The scripts expect the MDG_bfactor project to be placed at the repository root:

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

PSL feature generation also requires the upstream PSL research implementation. To avoid redistributing third-party code without explicit licensing terms, this repository ships a loader and expects the upstream `PSL.py` to be supplied locally:

```bash
git clone https://github.com/weixiaoqimath/persistent_sheaf_Laplacians.git \
  external/persistent_sheaf_Laplacians
```

Alternatively, set `PSL_UPSTREAM_DIR` to a directory containing `PSL.py`. Raw structures, annotation tables, and generated PSL matrices are not redistributed here. See `data/README.md` and `docs/attribution.md` for provenance.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Optional XGBoost experiments require:

```bash
pip install xgboost
```

Run the lightweight tests:

```bash
pytest
```

These tests do not require the external MDG_bfactor data or upstream PSL implementation; feature-generation scripts require both.

## Reproducing the main analyses

### 1. Generate center-labeled PSL features

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

The center-labeled construction assigns label 0 to the target C-alpha atom and label 1 to neighboring C-alpha atoms in the local neighborhood. The primary descriptor uses degree-0 PSL spectra at 6, 9, and 12 Angstrom and summarizes each spectrum by max, min, mean, median, standard deviation, and zero-eigenvalue count.

### 2. Main protein-level RF evaluation

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

### 3. SHAP attribution benchmark

```bash
python scripts/run_rf.py \
  --feature-source psl \
  --psl-dir data/processed/features_psl_labeled_6_9_12_both \
  --run-name psl_labeled_6_9_12_both_shap_25tree \
  --folds 5 \
  --n-estimators 25 \
  --max-depth 8 \
  --min-samples-leaf 2 \
  --max-features sqrt \
  --n-jobs 1
```

### 4. Baseline comparisons

```bash
python scripts/run_baselines.py \
  --psl-dir data/processed/features_psl_labeled_6_9_12_both \
  --folds 5 \
  --n-estimators 1000 \
  --max-depth 12 \
  --min-samples-leaf 2 \
  --max-features sqrt \
  --n-jobs 1
```

### 5. Biological validation

```bash
python scripts/run_biological_validation.py
```

### 6. Hayes-style protocol comparability

Generate the two feature variants used for comparability:

```bash
python scripts/psl_variant_experiment.py \
  --root . --dataset 365 --sheaf center_labeled \
  --radii 6,9,12 --stat4 median --degrees 0 \
  --out-dir data/processed/features_psl_labeled_6_9_12_median
```

```bash
python scripts/psl_variant_experiment.py \
  --root . --dataset 365 --sheaf center_labeled \
  --radii 7,10,13 --stat4 std --degrees 0 \
  --out-dir data/processed/features_psl_labeled_7_10_13_std
```

Run the protocol checks:

```bash
python scripts/hayes_protocol_runner.py \
  --root . \
  --psl-dir data/processed/features_psl_labeled_6_9_12_median \
  --mode table1-lr \
  --out-dir results/hayes_protocol
```

```bash
python scripts/hayes_protocol_runner.py \
  --root . \
  --psl-dir data/processed/features_psl_labeled_7_10_13_std \
  --mode blind-protein-kfold \
  --feature-source annotation+psl \
  --target raw \
  --model gbdt \
  --folds 10 \
  --align bfactor-subsequence \
  --out-dir results/hayes_protocol
```

```bash
python scripts/hayes_protocol_runner.py \
  --root . \
  --psl-dir data/processed/features_psl_labeled_7_10_13_std \
  --mode blind-atom-kfold \
  --feature-source annotation+psl \
  --target raw \
  --model gbdt \
  --folds 10 \
  --align bfactor-subsequence \
  --out-dir results/hayes_protocol
```

Final metric-audit scripts:

```bash
python scripts/run_protocol_exactness_checks.py --root . --only table1
python scripts/run_protocol_exactness_checks.py --root . --only random --n-estimators 500 --folds 10 --max-depth 0
python scripts/run_protocol_exactness_checks.py --root . --only controls --n-estimators 200 --folds 10
```

## Interpreting the protocol checks

Do not treat random residue-level pooled PCC as protein-independent generalization. In these data, random residue splits produce high pooled raw-B-factor PCCs for PSL+annotation models, but the same predictions have substantially lower mean per-protein PCC. Classical structural and simple graph features also show the same pooled-metric jump under random residue splits. This is why the manuscript reports random-residue scores as Hayes-style comparability checks rather than as the primary result.

## Citation

Use `CITATION.cff` when citing this repository (https://github.com/ryan-charette/psl-protein-flexibility). Please also cite the PSL and MDG_bfactor papers listed in the manuscript.
