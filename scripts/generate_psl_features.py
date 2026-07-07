"""
Generate persistent sheaf Laplacian (PSL) feature matrices.

The default center-labeled construction labels the target C-alpha atom as 0
and all neighboring C-alpha atoms as 1. A constant-sheaf mode is also available
for ablation experiments.

Outputs:
  <pdbid>_psl_features.npy
  <pdbid>_bfactors.npy
  feature_names.json
  run_config.json

Run from the repository root, for example:
  python scripts/generate_psl_features.py \
      --dataset 365 \
      --sheaf center_labeled \
      --radii 6 9 12 \
      --stats median \
      --out-dir data/processed/features_psl_labeled_6_9_12_median \
      --force

For the 7/10/13 Angstrom standard-deviation descriptor variant:
  python scripts/generate_psl_features.py \
      --dataset 365 \
      --sheaf center_labeled \
      --radii 7 10 13 \
      --stats std \
      --out-dir data/processed/features_psl_labeled_7_10_13_std \
      --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psl_flexibility.native_psl import PSL  # noqa: E402

DATASET_DIR = ROOT / "MDG_bfactor-main" / "MDG_bfactor-main" / "datasets"
PDB_DIR = DATASET_DIR / "365"
ZERO_TOL = 1e-8


def load_pdbids(dataset: str) -> list[str]:
    path = DATASET_DIR / f"list-{dataset}.txt"
    return [line.strip().replace(".pdb", "") for line in path.read_text().splitlines() if line.strip()]


def parse_ca_pdb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords: list[list[float]] = []
    bfactors: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        bfactors.append(float(line[60:66]))
    if not coords:
        raise ValueError(f"No C-alpha atoms found in {path}")
    return np.asarray(coords, dtype=float), np.asarray(bfactors, dtype=float)


def stat_names(stats: str) -> list[str]:
    if stats in {"median", "current"}:
        return ["max", "min", "mean", "median", "num_zero"]
    if stats in {"std", "hayes_blind"}:
        return ["max", "min", "mean", "std", "num_zero"]
    if stats in {"both", "extended"}:
        return ["max", "min", "mean", "median", "std", "num_zero"]
    raise ValueError(f"Unknown stats mode: {stats}")


def eigen_stats(laplacian: np.ndarray, stats: str) -> list[float]:
    eigvals = np.linalg.eigvalsh(laplacian)
    nonzero = eigvals[np.abs(eigvals) > ZERO_TOL]
    zero_count = float(eigvals.size - nonzero.size)

    if nonzero.size == 0:
        values = {
            "max": 0.0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "num_zero": zero_count,
        }
    else:
        values = {
            "max": float(nonzero.max()),
            "min": float(nonzero.min()),
            "mean": float(nonzero.mean()),
            "median": float(np.median(nonzero)),
            "std": float(nonzero.std(ddof=0)),
            "num_zero": zero_count,
        }
    return [values[name] for name in stat_names(stats)]


def feature_names(radii: Iterable[float], stats: str) -> list[str]:
    out: list[str] = []
    for r in radii:
        r_label = f"{r:g}".replace(".", "p")
        out.extend([f"r{r_label}_{name}" for name in stat_names(stats)])
    return out


def psl_stats_for_points(
    local_points: np.ndarray,
    radius: float,
    center_local_index: int,
    p_width: float,
    sheaf: str,
    stats: str,
    scale_labels: bool,
) -> list[float]:
    if local_points.shape[0] < 2:
        values = {"max": 0.0, "min": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "num_zero": float(local_points.shape[0])}
        return [values[name] for name in stat_names(stats)]

    if sheaf == "constant":
        charges = None
        constant = True
    elif sheaf in {"center_labeled", "atom_centered"}:
        # The target atom is not guaranteed to be row 0 after boolean masking,
        # so center_local_index matters.
        charges = np.ones(local_points.shape[0], dtype=float)
        charges[center_local_index] = 0.0
        constant = False
    else:
        raise ValueError(f"Unknown sheaf mode: {sheaf}")

    psl = PSL(
        pts=local_points,
        charges=charges,
        filtration_type="alpha",
        radius_list=np.array([radius], dtype=float),
        p=float(p_width),
        constant=constant,
        scale=bool(scale_labels),
    )
    psl.build_filtration()
    psl.build_simplicial_pair()
    psl.build_matrices()
    return eigen_stats(psl.psl_0()[0], stats=stats)


def generate_one(
    pdbid: str,
    out_dir: Path,
    radii: np.ndarray,
    p_width: float,
    sheaf: str,
    stats: str,
    scale_labels: bool,
    force: bool = False,
) -> bool:
    x_path = out_dir / f"{pdbid}_psl_features.npy"
    y_path = out_dir / f"{pdbid}_bfactors.npy"
    if not force and x_path.exists() and y_path.exists():
        return False

    coords, bfactors = parse_ca_pdb(PDB_DIR / f"{pdbid}.pdb")
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    names = feature_names(radii, stats)
    features = np.zeros((coords.shape[0], len(names)), dtype=float)

    for i in range(coords.shape[0]):
        row: list[float] = []
        for radius in radii:
            local_indices = np.flatnonzero(dists[i] <= radius)
            local_points = coords[local_indices]
            center_matches = np.flatnonzero(local_indices == i)
            if center_matches.size != 1:
                raise RuntimeError(f"Could not locate center residue {i} in local neighborhood for {pdbid}")
            center_local_index = int(center_matches[0])
            row.extend(
                psl_stats_for_points(
                    local_points=local_points,
                    radius=float(radius),
                    center_local_index=center_local_index,
                    p_width=p_width,
                    sheaf=sheaf,
                    stats=stats,
                    scale_labels=scale_labels,
                )
            )
        features[i, :] = np.asarray(row, dtype=float)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_path, features)
    np.save(y_path, bfactors)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="365", choices=["small", "medium", "large", "365"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed" / "features_psl")
    parser.add_argument("--radii", nargs="+", type=float, default=[6.0, 9.0, 12.0])
    parser.add_argument("--p-width", type=float, default=0.0)
    parser.add_argument("--sheaf", choices=["constant", "center_labeled", "atom_centered"], default="center_labeled")
    parser.add_argument("--scale-labels", action="store_true", help="Pass scale=True to the PSL implementation")
    parser.add_argument(
        "--stats",
        choices=["median", "std", "both", "current", "hayes_blind", "extended"],
        default="median",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radii = np.asarray(args.radii, dtype=float)
    pdbids = load_pdbids(args.dataset)
    if args.limit > 0:
        pdbids = pdbids[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "feature_names.json").write_text(json.dumps(feature_names(radii, args.stats), indent=2), encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "radii": [float(x) for x in radii],
                "p_width": float(args.p_width),
                "sheaf": args.sheaf,
                "scale_labels": bool(args.scale_labels),
                "stats": args.stats,
                "zero_tol": ZERO_TOL,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    start = time.time()
    generated = 0
    for idx, pdbid in enumerate(pdbids, start=1):
        t0 = time.time()
        did_generate = generate_one(
            pdbid=pdbid,
            out_dir=args.out_dir,
            radii=radii,
            p_width=args.p_width,
            sheaf=args.sheaf,
            stats=args.stats,
            scale_labels=bool(args.scale_labels),
            force=args.force,
        )
        generated += int(did_generate)
        status = "generated" if did_generate else "cached"
        print(f"[{idx}/{len(pdbids)}] {pdbid}: {status} in {time.time() - t0:.2f}s")
    print(f"Generated {generated} new PSL matrices in {time.time() - start:.2f}s. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
