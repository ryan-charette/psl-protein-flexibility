# Notice and Third-Party Components

This repository contains original software, analysis scripts, manuscript
material, synthetic toy examples, and curated metric snapshots for PSL-based
protein-flexibility analysis.

## Native PSL implementation

The maintained feature-generation path is implemented in
`src/psl_flexibility/native_psl.py`. It reproduces the minimum PSL functionality
needed by this project for local protein point clouds: distance-threshold
simplicial complexes, center-labeled or constant sheaves, and degree 0, 1, and
2 Laplacian matrices at fixed radii.

No third-party `PSL.py` source file is copied, vendored, or required at runtime.
The implementation is informed by the persistent sheaf Laplacian literature
cited in `paper/paper.md`.

## External data

Raw protein structures, B-factor labels, and annotation tables used for the full
benchmark analyses are expected from the MDG_bfactor project and are not
included in this repository. Place those files locally under
`MDG_bfactor-main/` as described in `data/README.md`.

## Generated artifacts

Generated PSL matrices (`*.npy`), feature CSVs, toy-demo outputs, and full
prediction tables can be regenerated from the scripts and should not be
committed unless intentionally stored with Git LFS.
