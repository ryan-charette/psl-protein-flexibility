#!/usr/bin/env python3
"""Evaluate verified cached descriptors under frozen protein/cluster folds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psl_flexibility.dataset import SEED
from psl_flexibility.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("results/study/dataset"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/study/core"))
    parser.add_argument("--phase", choices=["core", "d1", "ablation", "custom"], default="core")
    parser.add_argument("--modes", help="Comma-separated block combinations, e.g. graph,graph+center0")
    parser.add_argument("--protocols", default="protein,cluster")
    parser.add_argument("--cohort", default="all", help="all, subset60, subset20, or a CSV/text protein-ID file")
    parser.add_argument("--feature-columns", help="Regex against qualified block:column names; retain graph with '^graph:|...' ")
    parser.add_argument("--exclude-feature-columns", help="Regex against qualified block:column names")
    parser.add_argument("--missing", choices=["error", "exclude"], default="error")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--save-model-modes", default="graph+center0")
    args = parser.parse_args()
    split = lambda text: [value.strip() for value in text.split(",") if value.strip()]
    run_evaluation(
        args.manifest_dir, args.feature_dir, args.out_dir,
        modes=split(args.modes) if args.modes else None,
        protocols=tuple(split(args.protocols)), cohort=args.cohort, phase=args.phase,
        feature_columns=args.feature_columns, exclude_feature_columns=args.exclude_feature_columns,
        missing=args.missing, n_estimators=args.n_estimators, n_jobs=args.n_jobs,
        seed=args.seed, repetitions=args.bootstrap_repetitions,
        save_model_modes=tuple(split(args.save_model_modes)),
    )


if __name__ == "__main__":
    main()
