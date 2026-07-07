# Data Directory

This repository does not track raw benchmark structures, generated PSL feature
matrices, prediction tables, or toy-demo outputs.

For the bundled synthetic demo, run:

```bash
python scripts/run_toy_demo.py
```

The demo reads PDB files from `examples/toy_proteins/` and writes generated
outputs to `data/toy/processed/`, which is intentionally ignored by Git.

For the full manuscript reproduction, place the external MDG_bfactor dataset at:

```text
MDG_bfactor-main/
  MDG_bfactor-main/
    datasets/
    features/
```

To generate PSL features for your own PDB files, use:

```bash
python scripts/compute_psl_features.py \
  --pdb-dir path/to/pdb_files \
  --out-dir data/my_psl_features
```

The PSL implementation used by these commands is native to this repository;
no external `PSL.py` checkout is needed.
