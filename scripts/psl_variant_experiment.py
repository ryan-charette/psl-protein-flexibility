#!/usr/bin/env python3
"""
Generate and evaluate PSL feature variants for residue-level B-factor prediction.

This script is intended to be run from the unpacked psls/ repository.  It is a
small experiment harness for the most publication-relevant PSL variants:

  1. constant sheaf, radii 6/9/12, median statistic
  2. center-labeled sheaf, radii 6/9/12, median statistic
  3. center-labeled sheaf, radii 7/10/13, std statistic
  4. optional higher-order PSL spectra and/or multiple p-widths

The center-labeled option follows the localized construction described in
Hayes et al.: for each residue neighborhood, the center atom receives label 0
and neighboring atoms receive label 1.  In the Wei PSL implementation this is
represented with charges=[0,1,1,...] and constant=False.

Example:
    python psl_variant_experiment.py \
        --root . --dataset 365 \
        --sheaf center_labeled --radii 6,9,12 --stat4 median --degrees 0 \
        --out-dir data/processed/features_psl_labeled_6_9_12_median \
        --evaluate

Notes:
  - Requires the same dependencies as your existing PSL feature generator,
    especially gudhi.
  - Evaluation uses 5-fold CV by protein and within-protein z-scored
    C-alpha B-factors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class Config:
    root: str
    dataset: str
    sheaf: str
    radii: list[float]
    stat4: str
    degrees: list[int]
    p_widths: list[float]
    scale_labels: bool
    out_dir: str
    force: bool
    limit: int | None


def import_psl(root: Path):
    """Import PSL from the existing repository layout."""
    candidate_dirs = [
        root / "src" / "psl_flexibility" / "vendor",
        root / "analysis",
        root / "scripts",
    ]
    for d in candidate_dirs:
        if (d / "PSL.py").exists():
            sys.path.insert(0, str(d))
            from PSL import PSL  # type: ignore
            return PSL
    raise FileNotFoundError(
        "Could not find PSL.py under analysis/, src/psl_flexibility/vendor/, or scripts/. "
        "Run this from the unpacked psls/ repository or pass --root correctly."
    )


def parse_ca_pdb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return C-alpha coordinates and B-factors from a PDB file."""
    coords: list[list[float]] = []
    bfactors: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b = float(line[60:66])
            except ValueError:
                continue
            coords.append([x, y, z])
            bfactors.append(b)
    if not coords:
        raise ValueError(f"No C-alpha atoms found in {path}")
    return np.asarray(coords, dtype=float), np.asarray(bfactors, dtype=float)


def nonzero_eig_stats(evals: np.ndarray, stat4: str, tol: float = 1e-8) -> list[float]:
    """Summarize eigenvalues as max/min/mean/{median|std|both}/zero_count."""
    if evals.size == 0:
        if stat4 == "both":
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    evals = np.real(evals)
    zeros = float(np.sum(np.abs(evals) <= tol))
    nz = evals[np.abs(evals) > tol]
    if nz.size == 0:
        if stat4 == "both":
            return [0.0, 0.0, 0.0, 0.0, 0.0, zeros]
        return [0.0, 0.0, 0.0, 0.0, zeros]
    max_v = float(np.max(nz))
    min_v = float(np.min(nz))
    mean_v = float(np.mean(nz))
    median_v = float(np.median(nz))
    std_v = float(np.std(nz))
    if stat4 == "median":
        return [max_v, min_v, mean_v, median_v, zeros]
    if stat4 == "std":
        return [max_v, min_v, mean_v, std_v, zeros]
    if stat4 == "both":
        return [max_v, min_v, mean_v, median_v, std_v, zeros]
    raise ValueError(f"Unknown stat4 value: {stat4}")


def evals_for_degree(psl_obj, degree: int) -> np.ndarray:
    """Compute eigenvalues for one PSL matrix at the single requested radius."""
    if degree == 0:
        mats = psl_obj.psl_0()
    elif degree == 1:
        mats = psl_obj.psl_1()
    elif degree == 2:
        mats = psl_obj.psl_2()
    else:
        raise ValueError("Only degrees 0, 1, and 2 are supported by this harness.")
    if not mats:
        return np.array([], dtype=float)
    mat = np.asarray(mats[0], dtype=float)
    if mat.size == 0:
        return np.array([], dtype=float)
    return np.linalg.eigvalsh(mat)


def build_local_charges(local_indices: np.ndarray, center_index: int, sheaf: str) -> np.ndarray | None:
    """Return charges for local sheaf; center-labeled means center=0, neighbors=1."""
    if sheaf == "constant":
        return None
    if sheaf != "center_labeled":
        raise ValueError(f"Unknown sheaf: {sheaf}")
    charges = np.ones(len(local_indices), dtype=float)
    center_positions = np.where(local_indices == center_index)[0]
    if center_positions.size != 1:
        raise RuntimeError("The center residue was not found exactly once in its local neighborhood.")
    charges[int(center_positions[0])] = 0.0
    return charges


def psl_features_for_protein(
    PSL,
    coords: np.ndarray,
    *,
    radii: Sequence[float],
    sheaf: str,
    stat4: str,
    degrees: Sequence[int],
    p_widths: Sequence[float],
    scale_labels: bool,
) -> np.ndarray:
    """Compute PSL feature matrix for one protein."""
    n = coords.shape[0]
    rows: list[list[float]] = []
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    for i in range(n):
        row: list[float] = []
        for radius in radii:
            local_idx = np.where(dist[i] <= radius)[0]
            local_points = coords[local_idx]
            charges = build_local_charges(local_idx, i, sheaf)
            constant = sheaf == "constant"
            if local_points.shape[0] < 2:
                # Match the manuscript definition for degenerate neighborhoods:
                # nonzero-eigenvalue summaries are zero, while the degree-0
                # zero-eigenvalue count records the observed component count.
                block_len = 6 if stat4 == "both" else 5
                for _p_width in p_widths:
                    for degree in degrees:
                        block = [0.0] * block_len
                        if int(degree) == 0:
                            block[-1] = float(local_points.shape[0])
                        row.extend(block)
                continue
            for p_width in p_widths:
                psl_obj = PSL(
                    local_points,
                    charges=charges,
                    filtration_type="alpha",
                    radius_list=np.array([float(radius)], dtype=float),
                    p=float(p_width),
                    constant=constant,
                    scale=bool(scale_labels),
                )
                psl_obj.build_filtration()
                psl_obj.build_simplicial_pair()
                psl_obj.build_matrices()
                for degree in degrees:
                    try:
                        evals = evals_for_degree(psl_obj, int(degree))
                    except Exception as exc:
                        # Higher-order matrices can be empty or undefined in very small neighborhoods.
                        # Treat those cases as empty spectra, but surface other repeated failures in logs.
                        if degree == 0:
                            raise
                        evals = np.array([], dtype=float)
                    row.extend(nonzero_eig_stats(evals, stat4=stat4))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def feature_names(radii: Sequence[float], stat4: str, degrees: Sequence[int], p_widths: Sequence[float]) -> list[str]:
    if stat4 == "median":
        stats = ["max", "min", "mean", "median", "zeros"]
    elif stat4 == "std":
        stats = ["max", "min", "mean", "std", "zeros"]
    elif stat4 == "both":
        stats = ["max", "min", "mean", "median", "std", "zeros"]
    else:
        raise ValueError(stat4)
    names: list[str] = []
    for r in radii:
        r_label = str(int(r)) if float(r).is_integer() else str(r).replace(".", "p")
        for p in p_widths:
            p_label = str(p).replace(".", "p")
            for deg in degrees:
                for s in stats:
                    names.append(f"r{r_label}_p{p_label}_d{deg}_{s}")
    return names


def protein_files(root: Path, dataset: str) -> list[Path]:
    candidate_dirs = [
        root / "MDG_bfactor-main" / "MDG_bfactor-main" / "datasets" / dataset,
        root / "MDG_bfactor-main" / "datasets" / dataset,
        root / "datasets" / dataset,
    ]
    data_dir = next((d for d in candidate_dirs if d.exists()), None)
    if data_dir is None:
        searched = ", ".join(str(d) for d in candidate_dirs)
        raise FileNotFoundError(f"Could not find dataset directory for {dataset!r}. Searched: {searched}")
    files = sorted(data_dir.glob("*.pdb"))
    if not files:
        raise FileNotFoundError(f"No PDB files found in {data_dir}")
    return files


def generate_features(config: Config) -> None:
    root = Path(config.root).resolve()
    out_dir = Path(config.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    PSL = import_psl(root)

    names = feature_names(config.radii, config.stat4, config.degrees, config.p_widths)
    (out_dir / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    files = protein_files(root, config.dataset)
    if config.limit is not None:
        files = files[: config.limit]

    manifest_rows: list[dict[str, str | int]] = []
    for k, pdb_path in enumerate(files, start=1):
        stem = pdb_path.stem
        x_path = out_dir / f"{stem}_psl_features.npy"
        y_path = out_dir / f"{stem}_bfactors.npy"
        if x_path.exists() and y_path.exists() and not config.force:
            coords, y = parse_ca_pdb(pdb_path)
            manifest_rows.append({"protein": stem, "n_residues": int(len(y)), "status": "cached"})
            print(f"[{k}/{len(files)}] cached {stem}", flush=True)
            continue
        coords, y = parse_ca_pdb(pdb_path)
        X = psl_features_for_protein(
            PSL,
            coords,
            radii=config.radii,
            sheaf=config.sheaf,
            stat4=config.stat4,
            degrees=config.degrees,
            p_widths=config.p_widths,
            scale_labels=config.scale_labels,
        )
        if X.shape[1] != len(names):
            raise RuntimeError(f"Feature-name mismatch for {stem}: {X.shape[1]} vs {len(names)}")
        np.save(x_path, X)
        np.save(y_path, y)
        manifest_rows.append({"protein": stem, "n_residues": int(len(y)), "status": "generated"})
        print(f"[{k}/{len(files)}] generated {stem}: {X.shape}", flush=True)

    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["protein", "n_residues", "status"])
        writer.writeheader()
        writer.writerows(manifest_rows)


def zscore_within_protein(y: np.ndarray) -> np.ndarray:
    sd = float(np.std(y))
    if sd == 0.0 or not np.isfinite(sd):
        return np.zeros_like(y, dtype=float)
    return (y - float(np.mean(y))) / sd


def safe_corr(fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]], y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(fn(y, pred)[0])


def load_generated(out_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    records: list[tuple[str, np.ndarray, np.ndarray]] = []
    for x_path in sorted(out_dir.glob("*_psl_features.npy")):
        stem = x_path.name[: -len("_psl_features.npy")]
        y_path = out_dir / f"{stem}_bfactors.npy"
        if not y_path.exists():
            continue
        X = np.load(x_path)
        y = zscore_within_protein(np.load(y_path))
        if len(y) == X.shape[0] and X.shape[0] >= 2:
            records.append((stem, X, y))
    if not records:
        raise FileNotFoundError(f"No generated feature matrices found in {out_dir}")
    return records


def make_model(kind: str, seed: int):
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=1000,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "hgbt":
        return HistGradientBoostingRegressor(
            max_iter=700,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=seed,
        )
    if kind == "xgboost_cpu" or kind == "xgboost_gpu":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install xgboost to use --model xgboost_cpu/gpu") from exc
        params = dict(
            n_estimators=1500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
        if kind == "xgboost_gpu":
            # XGBoost >=2 prefers device='cuda'; older versions use tree_method='gpu_hist'.
            params.update(dict(tree_method="hist", device="cuda"))
        else:
            params.update(dict(tree_method="hist"))
        return XGBRegressor(**params)
    raise ValueError(f"Unknown model: {kind}")


def evaluate(out_dir: Path, *, model_kind: str, seed: int) -> dict[str, float]:
    records = load_generated(out_dir)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    protein_metrics: list[dict[str, float | str | int]] = []
    pooled_y: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []

    indices = np.arange(len(records))
    for fold, (train_idx, test_idx) in enumerate(kf.split(indices), start=1):
        X_train = np.vstack([records[i][1] for i in train_idx])
        y_train = np.concatenate([records[i][2] for i in train_idx])
        model = make_model(model_kind, seed + fold)
        model.fit(X_train, y_train)
        for i in test_idx:
            protein, X_test, y_test = records[i]
            pred = np.asarray(model.predict(X_test), dtype=float)
            pooled_y.append(y_test)
            pooled_pred.append(pred)
            protein_metrics.append(
                {
                    "protein": protein,
                    "fold": int(fold),
                    "n_residues": int(len(y_test)),
                    "pearson": safe_corr(pearsonr, y_test, pred),
                    "spearman": safe_corr(spearmanr, y_test, pred),
                    "rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
                }
            )

    y_all = np.concatenate(pooled_y)
    pred_all = np.concatenate(pooled_pred)
    pearsons = np.array([m["pearson"] for m in protein_metrics], dtype=float)
    spearmans = np.array([m["spearman"] for m in protein_metrics], dtype=float)
    rmses = np.array([m["rmse"] for m in protein_metrics], dtype=float)
    summary = {
        "n_proteins": float(len(records)),
        "n_residues": float(len(y_all)),
        "mean_pearson": float(np.nanmean(pearsons)),
        "mean_spearman": float(np.nanmean(spearmans)),
        "mean_rmse": float(np.nanmean(rmses)),
        "pooled_pearson": safe_corr(pearsonr, y_all, pred_all),
        "pooled_rmse": float(math.sqrt(mean_squared_error(y_all, pred_all))),
    }

    with (out_dir / f"cv_metrics_{model_kind}.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["protein", "fold", "n_residues", "pearson", "spearman", "rmse"])
        writer.writeheader()
        writer.writerows(protein_metrics)
    (out_dir / f"cv_summary_{model_kind}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".", help="Unpacked psls/ repository root")
    p.add_argument("--dataset", default="365", help="Dataset subdirectory, e.g. 365, small, medium, large")
    p.add_argument("--sheaf", choices=["constant", "center_labeled"], required=True)
    p.add_argument("--radii", required=True, help="Comma-separated radii, e.g. 6,9,12 or 7,10,13")
    p.add_argument("--stat4", choices=["median", "std", "both"], default="median")
    p.add_argument("--degrees", default="0", help="Comma-separated PSL degrees, e.g. 0 or 0,1")
    p.add_argument("--p-widths", default="0.0", help="Comma-separated p widths, e.g. 0.0 or 0.0,0.5,1.0")
    p.add_argument("--scale-labels", action="store_true", help="Pass scale=True to the PSL implementation")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--force", action="store_true", help="Overwrite existing cached features")
    p.add_argument("--limit", type=int, default=None, help="Limit proteins for smoke tests")
    p.add_argument("--evaluate", action="store_true", help="Run grouped 5-fold CV after feature generation")
    p.add_argument("--model", choices=["rf", "hgbt", "xgboost_cpu", "xgboost_gpu"], default="rf")
    p.add_argument("--seed", type=int, default=20260504)
    args = p.parse_args()

    config = Config(
        root=args.root,
        dataset=args.dataset,
        sheaf=args.sheaf,
        radii=parse_float_list(args.radii),
        stat4=args.stat4,
        degrees=parse_int_list(args.degrees),
        p_widths=parse_float_list(args.p_widths),
        scale_labels=bool(args.scale_labels),
        out_dir=args.out_dir,
        force=bool(args.force),
        limit=args.limit,
    )
    generate_features(config)
    if args.evaluate:
        summary = evaluate(Path(args.out_dir), model_kind=args.model, seed=args.seed)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
