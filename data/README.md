# Data Directory

This repository does not track raw protein structures or generated PSL feature matrices because they are third-party or derived artifacts.

Expected local layout after setup:

```text
MDG_bfactor-main/
  MDG_bfactor-main/
    datasets/
    features/

external/
  persistent_sheaf_Laplacians/
    PSL.py

data/
  processed/
    features_psl_labeled_6_9_12_both/
      <pdbid>_psl_features.npy
      <pdbid>_bfactors.npy
```

The upstream PSL implementation may also be supplied by setting `PSL_UPSTREAM_DIR` to the directory containing `PSL.py`.

To regenerate the primary processed PSL features:

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
