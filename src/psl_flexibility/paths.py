"""Path helpers for repository-local command-line scripts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "MDG_bfactor-main" / "MDG_bfactor-main" / "datasets"
ANNOTATION_DIR = ROOT / "MDG_bfactor-main" / "MDG_bfactor-main" / "features" / "features-blind-prediction"
PDB_DIR = DATASET_DIR / "365"
RESULTS_DIR = ROOT / "results"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
