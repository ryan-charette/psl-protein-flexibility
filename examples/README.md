# Examples

`toy_proteins/` contains three synthetic PDB files with only C-alpha atoms.
They are designed for smoke tests and documentation examples, not for
scientific benchmarking.

Run the bundled demo from the repository root:

```bash
python scripts/run_toy_demo.py
```

Or generate only feature tables:

```bash
python scripts/compute_psl_features.py \
  --pdb-dir examples/toy_proteins \
  --out-dir data/toy/processed
```
