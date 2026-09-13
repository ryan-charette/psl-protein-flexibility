# Integrated manuscript revision

This editorial revision expands the completed study into a self-contained graduate-level manuscript. The scientific scope, computational code, data, fitted models, numerical results, figures, and bibliography are unchanged.

The new reading guide and notation reference lead into structural-biology vocabulary, a survey map, graph incidence and energy, simplicial cochains, cohomology and harmonic representatives, alpha filtrations, sheaf restrictions, deletion/link decomposition, persistent operators through least squares, and a guide to evaluation and attribution.

The former supplement is integrated as follows:

| Former supplement topic | Integrated source |
|---|---|
| The universal-center spectrum in a broader form | `paper/sections/native_details.tex` |
| Weighted deletion and augmented link | `paper/sections/link_details.tex` |
| Persistent operator and independent matrix checks | `paper/sections/persistent_details.tex` |
| Structure selection and identity audit | `paper/sections/identity_details.tex` |
| Observed-sequence clustering and fold assignment | `paper/sections/fold_details.tex` |
| Ordered graph and spectral features | `paper/sections/feature_details.tex` |
| Degree-one cohort and computational limits | `paper/sections/compute_details.tex` |
| Attribution and figure integrity | `paper/sections/attribution_details.tex` |
| Evidence provenance | `paper/sections/evidence_details.tex` |

The former notation/scope material is included in the reading guide and persistent-operator derivation. The companion figure and generated detailed result tables are included through `paper/sections/companion_results.tex`. Repeated method descriptions were consolidated without dropping their substantive conditions.

Build from the repository root with `python paper/build.py` (Tectonic required). `--tectonic PATH` selects an executable outside PATH. The builder derives an editorial version of the frozen persistence results, updating document-location wording and line breaking while preserving every numerical token.

The original separate article and supplement remain in Git history at `bd1e404`. The current PDF was compiled and every page visually reviewed; the record is `results/study/document_validation.json`.
