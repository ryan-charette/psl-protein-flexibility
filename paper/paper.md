---
title: 'psl-protein-flexibility: Self-contained persistent sheaf Laplacian descriptors for protein flexibility analysis'
tags:
  - Python
  - protein structure
  - bioinformatics
  - topological data analysis
  - machine learning
authors:
  - name: Ryan Charette
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 07 July 2026
bibliography: paper.bib
---

# Summary

`psl-protein-flexibility` is a Python package and command-line toolkit for
generating persistent sheaf Laplacian (PSL) descriptors from protein structures.
Given ordinary PDB files, the software parses C-alpha coordinates and
B-factors, builds local sheaf-Laplacian spectra around each residue, and writes
per-residue feature tables suitable for downstream flexibility modeling. The
repository also includes scripts that reproduce the accompanying
protein-flexibility analyses on the external MDG_bfactor benchmark dataset
[@feng2025], along with curated metric snapshots for auditability.

The maintained feature-generation path is fully self-contained. Earlier
versions of the project depended on a separate research implementation of PSL
descriptors; this package now provides the minimum required native
implementation for local protein point clouds, including center-labeled and
constant-sheaf descriptors, degree 0, 1, and 2 Laplacian accessors, and spectral
summary features at user-selected radii.

# Statement of need

Protein flexibility modeling often starts from static structures because
crystallographic B-factors are widely available, but turning those structures
into reusable geometric features is still cumbersome. Many benchmark-oriented
repositories are organized around one dataset and one manuscript result, which
makes it difficult for another researcher to run the same descriptor pipeline on
their own proteins, inspect residue-level outputs, or test the software without
downloading a large benchmark collection.

`psl-protein-flexibility` addresses that software gap. It exposes PSL feature
generation as a normal package API and command-line workflow for arbitrary PDB
directories, not just for the original MDG_bfactor layout. The output CSV
retains residue identifiers, raw B-factors, within-protein normalized
B-factors, and named PSL feature columns, so researchers can join the features
with their own annotations or models. A bundled synthetic-data demo runs from a
fresh checkout and exercises parsing, feature generation, and a small grouped
regression workflow without external data.

The package is also designed for review and redistribution. The PSL routines
needed by the project are implemented natively rather than copied from an
upstream research repository, avoiding an external-code requirement while
keeping the software path transparent. The benchmark reproduction scripts remain
available for readers who want to inspect the accompanying study, but the core
software contribution is a reusable, self-contained descriptor generator for
protein structures.

# Functionality

The package provides:

- C-alpha PDB parsing with residue metadata and B-factor extraction.
- A native distance-threshold simplicial-complex implementation for local PSL
  descriptors.
- Center-labeled sheaves, where the target residue is assigned label 0 and its
  neighbors label 1, plus a constant-sheaf baseline.
- Spectral summaries of nonzero eigenvalues and zero-eigenvalue counts across
  one or more radii.
- A command-line feature generator for arbitrary PDB directories.
- A toy demo with synthetic PDB files for reviewer and user smoke tests.
- Benchmark scripts for grouped cross-validation, baseline comparison,
  SHAP-based attribution diagnostics, and protocol-comparability checks.

For example, users can generate features for their own PDB files with:

```bash
python scripts/compute_psl_features.py \
  --pdb-dir path/to/pdb_files \
  --out-dir data/my_psl_features \
  --radii 6,9,12 \
  --sheaf center_labeled \
  --stats both
```

The bundled demo can be run with:

```bash
python scripts/run_toy_demo.py
```

# Research context

The software is motivated by residue-level protein flexibility prediction from
structure. Elastic-network and rigidity-based methods show that local geometry
and packing are informative for B-factor prediction [@haliloglu1997;
@atilgan2001; @nguyen2016]. Topological and sheaf-based descriptors extend this
idea by summarizing multiscale geometric organization through spectral
quantities [@edelsbrunner2010; @xia2014; @hayes2025].

The accompanying analysis evaluates PSL descriptors under protein-grouped
cross-validation and compares them with classical structural and graph-based
features. Those scientific results are documented by the scripts and metric
snapshots in the repository, but they are separate from the package's primary
software purpose: making PSL descriptor generation available for new protein
sets.

# Implementation

The native PSL implementation constructs local Euclidean distance-threshold
complexes at requested radii and forms weighted incidence matrices for the
corresponding sheaf Laplacians. Degree-zero matrices are used for the main
descriptor workflow, while degree-one and degree-two Hodge-style Laplacians are
available for exploratory analyses. The implementation uses NumPy and SciPy
[@harris2020; @virtanen2020] and is tested on synthetic structures so that core
functionality can be verified without third-party benchmark data.

# Availability

The source code, documentation, tests, toy examples, and manuscript materials
are available at
<https://github.com/ryan-charette/psl-protein-flexibility>. The software is
distributed under the MIT license. External benchmark data are not
redistributed; the repository documents where those data should be placed for
full analysis reproduction.

# Acknowledgements

This project builds on the persistent sheaf Laplacian and protein-flexibility
literature cited below. The MDG_bfactor dataset and prior PSL flexibility study
provided the benchmark context for the accompanying analyses.

# References
