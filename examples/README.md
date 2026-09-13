# Examples

`toy_proteins/` contains three synthetic PDB files, each with eight C-alpha
atoms and artificial B-factors. They provide a quick check of coordinate
parsing, feature export and regression; their scores have no scientific
interpretation.

After installing `requirements.txt`, run the demo from the repository root:

```bash
python scripts/run_toy_demo.py
```

The demo uses the historical center-weighted graph descriptor retained in
`src/psl_flexibility/native_psl.py`. It does not use the corrected alpha-complex
sheaf operators in `sheaf.py`. Historical output filenames such as
`toy_psl_features.csv` are retained for the smoke tests.

The default output directory is `data/toy/processed/` (ignored by Git). It
contains feature values and names, the feature configuration, per-chain
leave-one-out metrics and a JSON summary. Use `--out-dir path/to/output` to
choose another directory.

For the real structures, corrected operators and held-out evaluations used in
the manuscript, follow the study commands in the [main README](../README.md)
and the [data notes](../data/README.md).
