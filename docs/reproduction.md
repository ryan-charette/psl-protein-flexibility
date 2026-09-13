# Reproducing the research

The study uses ordinary Python scripts from the repository root. The internal
`src/psl_flexibility` modules are implementation details of this analysis.
No editable package installation or external PSL implementation is required.

## Environment and inputs

The executed environment was Python 3.12.14 on Windows, on an Intel Core
i7-1165G7 CPU. Feature generation used four worker processes, each restricted
to one BLAS thread. Random Forest fitting used four jobs. The exact Python
package versions are in `results/study/environment.txt`; the two shorter
requirements files describe the numerical and rendering dependencies.

```sh
python -m venv .venv
# Activate .venv for your shell before the remaining commands.
python -m pip install -r results/study/environment.txt
python scripts/fetch_data.py
python scripts/prepare_dataset.py
```

The acquisition script pins the MDG benchmark source to commit
`c987feb2c45c785799685c03d270ea8053e5706d` of
[fenghon1/MDG_bfactor](https://github.com/fenghon1/MDG_bfactor).
The benchmark file named `list-365.txt` contains 364 entries. Its C-alpha
structures, and the full 1ULR/1X3O structures used for rendering, remain in
ignored `data/raw/` directories. The committed manifest records structure
hashes and preprocessing decisions. No identity is inferred from row counts
or B-factor equality.

The final sequence rule uses at least 30% identity, 80% coverage of each
observed chain, and 50 aligned nongap residue pairs. Exact complete-record
sequence duplicates are also grouped. This rule was amended before modeling:
the initially proposed shorter-chain rule connected every structure through
short auxiliary chains. The original graph, amendment, and limitations of
the resulting grouping remain recorded in `results/study/dataset/`.

## Features and fixed comparisons

```sh
# Optional fresh timing pilot; does not overwrite the original planning snapshot.
python scripts/profile_features.py
python scripts/generate_study_features.py --degree1 --workers 4 --timing-name d1_timing.csv
python scripts/augment_upper_features.py --workers 4
python scripts/run_evaluations.py --jobs 4
```

The stored pilot supported the full cohort rather than either nested fallback
cohort. All 364 structures and 78,400 retained residues enter every reported
comparison. Finalize **all** feature additions before starting evaluation:
each evaluation records hashes of its complete source caches. Regeneration
can use `--resume` on `run_evaluations.py` to reuse completed evaluations whose
recorded inputs and outputs still agree. An interrupted individual evaluation
is refitted; a partial metric table is never treated as a completed result.

The seven primary comparisons use sequence-controlled and protein-only folds.
Degree-one controls and six degree-zero ablations use the same primary folds.
Every forest has 200 trees, depth 12, minimum leaf size 2,
`max_features="sqrt"`, and seed `20260913`. There is no hyperparameter search.
The response is the within-protein standardized crystallographic C-alpha
B-factor. Normalization defines the response; no target-derived quantity is
included among input features. Correlations and RMSE are averaged across
proteins; confidence intervals resample entire sequence components and are
conditional on this fixed split and the fitted models.

The original degree-zero timing snapshot precedes the chemical-identity
amendment (78,223 residues). The production degree-one timing includes
recomputation of degree-zero features in the 24 affected structures. A fresh
single-pass execution combines those operations differently, so timing
equality is neither expected nor a correctness criterion.

## Explanations, figures, and paper

```sh
python scripts/explain_cases.py
python scripts/explain_cases.py --verify-only
python scripts/characterize_features.py
python scripts/summarize_study.py
python scripts/make_figures.py --all
python scripts/audit_study.py --require-complete --require-cases --out results/study/final_audit.json
python scripts/validate_operators.py
python -m pytest -q
tectonic -X compile paper/manuscript.tex
tectonic -X compile paper/supplement.tex
```

Tectonic 0.17.0 compiled the delivered PDFs. A conventional LaTeX/BibTeX
workflow also works. Molecular rendering uses PyVista/VTK off screen; it
requires a functional local OpenGL rendering environment. No image-generation
model is used. Vector text and plots are combined with 1800-pixel molecular
renders. The separate figure exports can be used without recompiling LaTeX.

The two explained models are the actual cluster-held-out graph-plus-center-zero
forests. Each uses 100 unique, seeded training-background residues. The CSV
files retain all 37 SHAP values per residue, expected values, grouped sums,
and residue identities. Provenance includes training/test identifiers,
background rows, model hashes, and a check against the full held-out
prediction table. SHAP here is model attribution, not a causal or dynamical
measurement.

## What is saved

| Location | Contents |
|---|---|
| `results/study/dataset/` | Source hashes, residue policy, exclusions, sequences, components, folds and amendments |
| `data/features/` (ignored) | Keyed per-protein spectral matrices and exact feature names |
| `results/study/core/`, `d1/`, `ablation_*/` | Compact per-protein metrics, paired differences, intervals, timings and configurations |
| Each run's `predictions.csv.gz` (ignored) | Complete held-out residue predictions |
| `results/study/core/models/` (ignored) | Exact forests used for the two case studies |
| `results/study/cases/` | Full case attributions, values, backgrounds and mapping/rendering audits |
| `results/study/report_numbers.json` | Generated numerical manuscript inputs and their source hashes |
| `results/study/configs/` | Immutable feature configuration snapshots and their original hashes |
| `results/study/final_audit.json` | Independent verification of all completed evaluations and both explanations |
| `results/study/compute_provenance.json` | Measured feature and fitting times, with accounting limitations |
| `paper/figures/` | PDF/SVG plots, PNG compositions, and molecular rendering inputs |
| `results/historical/` | Earlier snapshots, explicitly excluded from new evidence |

Historical drafts and archive inventories are discussed in
`docs/research_provenance.md`. Unkeyed historical structural-annotation arrays
are not used as verified covariates. Secondary-structure shapes in the two
case renderings use their deposited PDB HELIX/SHEET records and exact mapped
coordinates. The paper's endpoint and claims do not extend to measured
solution dynamics or to a new external benchmark.

## Executed verification

The completed study contains 160 forest fits (32,000 trees) and 2,508,800
held-out residue predictions across its 32 model/protocol combinations.
The independent audit recomputes per-protein metrics and checks the reported
means, joins, partitions, source hashes, and both SHAP reconstructions. The
scientific test suite contains 53 passing tests. The operator audit checks
168 persistent operators against an independent nullspace construction.

The two production degree-one stages used 1,909.2 seconds of measured wall
time in total, including the upper-endpoint controls and the documented
degree-zero repairs. Summed fit-and-predict elapsed times were 2,406.3 seconds.
Neither sum is an end-to-end runtime: bootstrapping, acquisition, explanation,
rendering, and document preparation are separate. The compute record retains
the individual measurements and clearly labels the historical degree-zero
stage. No paid computation was used.

Git attributes preserve the exact bytes of Python sources and hashed result,
manuscript, and figure artifacts, including their line endings. A new execution on a different
platform can produce different file bytes or floating-point results; its own
manifests and checks must be regenerated together rather than combined with
the saved run's hashes.
