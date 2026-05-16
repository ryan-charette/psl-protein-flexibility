# Notice and Third-Party Components

This repository contains original analysis scripts, manuscript material, and curated metric snapshots for the PSL protein-flexibility study.

## Upstream PSL implementation

The feature-generation scripts require the research implementation from:

- Xiaoqi Wei, `weixiaoqimath/persistent_sheaf_Laplacians`
- https://github.com/weixiaoqimath/persistent_sheaf_Laplacians

The upstream README describes the project as a pedagogical Python implementation of persistent sheaf Laplacians and notes that `charges=None` with `constant=True` computes persistent Laplacians.

The upstream repository did not expose an explicit open-source license at the time this package was prepared. For publication-readiness and redistribution hygiene, this repository therefore does **not** redistribute the upstream `PSL.py` implementation. Instead, `src/psl_flexibility/vendor/PSL.py` is a small loader that imports a local copy supplied by the user. Clone the upstream repository into `external/persistent_sheaf_Laplacians/` or set `PSL_UPSTREAM_DIR` to the directory containing `PSL.py`.

## External data

Raw protein structures, B-factor labels, and annotation tables are expected from the MDG_bfactor project and are not included in this repository. Place those files locally under `MDG_bfactor-main/` as described in `data/README.md`.

## Generated artifacts

Generated PSL matrices (`*.npy`) and full prediction CSVs can be regenerated from the scripts and should not be committed unless intentionally stored with Git LFS.
