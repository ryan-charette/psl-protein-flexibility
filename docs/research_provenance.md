# Research provenance

The research revision starts from main commit `0e59f81a38608b2e4bb82a1c1e4327d18a2a23a1`.
`source_inventory.json` records SHA-256 hashes and members of the supplied manuscript
archive, locally recovered research archives, and the MDG benchmark archive.
Archives and absolute local paths are not redistributed.

## Manuscript lineage

- `proposal.tex`: prospective aims; not evidence of completed experiments.
- `paper.tex`: mixes held-protein results in its abstract with in-sample per-protein
  fitting elsewhere. Its large correlations and early attribution claims are not
  carried into the new analysis.
- `manuscript.tex`: the most developed research account, used as background for
  a fresh draft. Its numerical results remain historical until independently reproduced.

## Implementation lineage

Earlier research archives used the external `weixiaoqimath/persistent_sheaf_Laplacians`
implementation or a loader for it. The current main branch instead supplies a
native distance-threshold weighted graph construction. That construction accepts
`alpha` as an alias for Rips and treats `p` as weight attenuation. These meanings
are different from alpha filtration and a two-scale persistent sheaf operator.
The higher-degree historical native matrices do not generally form a compatible
sheaf cochain complex. They are not evidence for the new topology experiments.

The new mathematical implementation is independently written from the cited
restriction and persistent-operator definitions. No upstream PSL source is vendored.
The existing graph construction is retained only as an explicitly named comparator.

## Results and data

The original `results/metrics` files are historical snapshots. Their generating
feature arrays and complete execution records are absent from the provided
archives, so matching current source code is not sufficient to verify them.
They are retained under `results/historical` for transparent comparison, and are
not the numerical source of the new manuscript.

The recovered `MDG_bfactor-main.zip` contains 364 entries in `list-365.txt`.
The numerical list name does not establish an additional protein. Preparation
records actual eligible structures, residue identities, exclusions, and data hashes.
Legacy annotation row identifiers are audited rather than matched using target
B-factor values or assumed row order.

The case studies 1ULR and 1X3O, geometric parameters, model settings, split rules,
and nested cohort selection are fixed before examining new predictive results.
The main endpoint is a within-protein standardized crystallographic B-factor
profile, not an independent measurement of molecular dynamics.
