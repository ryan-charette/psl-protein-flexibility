#!/usr/bin/env python3
"""Prepare audited coordinates, sequence components and frozen study folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psl_flexibility.dataset import SEED, finalize_partitions, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/MDG_bfactor-main"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/study/dataset"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume-clustering", action="store_true", help="Use existing audited coordinate manifests unchanged")
    args = parser.parse_args()
    last_report = [0.0]

    def progress(done: int, total: int) -> None:
        if time.monotonic() - last_report[0] > 20 or done == total:
            print(f"Sequence pairs: {done}/{total}", flush=True)
            last_report[0] = time.monotonic()

    if args.resume_clustering:
        metadata = finalize_partitions(args.out_dir, folds=args.folds, seed=args.seed, progress=progress)
    else:
        metadata = prepare_dataset(args.data_root, args.out_dir, folds=args.folds, seed=args.seed, progress=progress)
    print(json.dumps({key: metadata[key] for key in ["n_source_files", "n_proteins", "n_residues", "n_clusters", "n_edges"]}, indent=2))


if __name__ == "__main__":
    main()
