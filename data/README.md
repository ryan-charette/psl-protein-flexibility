# Study data

The study uses the benchmark structures distributed with
[MDG_bfactor](https://github.com/fenghon1/MDG_bfactor/tree/c987feb2c45c785799685c03d270ea8053e5706d).
The committed protein manifest records the source files and their SHA-256
hashes. Full-atom RCSB structures for 1ULR and 1X3O provide the molecular
renderings; the rendering code checks their residue identities and C-alpha
coordinates against the benchmark before assigning colors.

Follow the [reproduction record](../docs/reproduction.md) for the environment,
acquisition, feature generation, evaluation, explanations, figures and checks.
That record includes the rendering dependencies and exact executed versions.
The full benchmark requires substantially more time than the toy example.
`fetch_data.py --archive path/to/MDG_bfactor.zip` can use a local copy of the
upstream archive. `fetch_data.py --verify-only` checks existing inputs without
downloading or writing them. Source mismatches cause an error.

The default layout is:

```text
data/
  raw/MDG_bfactor-main/datasets/   # Benchmark structures and annotations
  raw/figures/                    # Full-atom 1ULR and 1X3O structures
  features/                      # Per-protein feature matrices and provenance
  processed/mpl-cache/            # Local plotting cache
  toy/processed/                 # Optional synthetic smoke-demo outputs
results/study/
  dataset/                       # Audited manifest, residue table and frozen folds
  core/                          # Main evaluations, predictions and saved models
  d1/                            # Degree-one endpoint and persistence comparisons
  cases/                         # Keyed SHAP values, background samples and audits
paper/figures/                    # Reproducible static figures
```

Raw downloads, feature matrices, fitted models, large residue/prediction
tables and disposable caches are ignored by Git. The repository retains the
audited source manifest and folds, compact results, case-study explanations
and figure provenance needed to inspect the reported analyses. Reproduction
rebuilds the larger files locally. `run_evaluations.py --resume` reuses only
completed evaluations whose recorded input hashes still agree.

For a quick local smoke test, run `python scripts/run_toy_demo.py`. This uses
the three artificial chains in `examples/toy_proteins/` and the legacy
center-weighted graph descriptor. Its metrics do not measure performance on
the research benchmark; see the [example notes](../examples/README.md).
