#!/usr/bin/env python3
"""
Hayes et al. PSL reproduction/evaluation harness.

Designed for the project layout used in psls.zip / MDG_bfactor-main:
  <root>/MDG_bfactor-main/MDG_bfactor-main/datasets
  <root>/MDG_bfactor-main/MDG_bfactor-main/features/features-blind-prediction
  <root>/analysis/features_psl or a user-supplied PSL feature directory

Main modes:
  1. table1-lr: per-protein in-sample linear regression, matching Hayes Table 1 more closely.
  2. blind-protein-kfold: blind ML-style protein-level evaluation using annotations plus PSL features.
  3. blind-atom-kfold: random residue-level comparability using annotations plus PSL features.

This script intentionally reports both pooled and mean-per-protein PCC, because Hayes' tables
and a publication-grade grouped-protein evaluation answer different questions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBRegressor = None


EXCLUDE_BLIND_PAPER = {
    "1OB4", "1OB7", "2OXL", "3MD5", "1AGN", "1NKO", "2OCT", "3FVA",
    "3DWV", "3MGN", "4DPZ", "2J32", "3MEA", "3A0M", "3IVV", "3W4Q", "3P6J", "2DKO",
}


@dataclass
class ProteinRecord:
    pdbid: str
    x_psl: np.ndarray
    y_raw: np.ndarray
    membership: str
    x_anno: np.ndarray | None = None


def pcc(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return float("nan")
    yy = y[mask]
    pp = pred[mask]
    if np.allclose(yy, yy[0]) or np.allclose(pp, pp[0]):
        return float("nan")
    return float(pearsonr(yy, pp).statistic)


def spcc(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return float("nan")
    yy = y[mask]
    pp = pred[mask]
    if np.allclose(yy, yy[0]) or np.allclose(pp, pp[0]):
        return float("nan")
    return float(spearmanr(yy, pp).statistic)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2)))


def zscore(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    sd = float(np.nanstd(y))
    if sd == 0 or not np.isfinite(sd):
        return np.full_like(y, np.nan, dtype=float)
    return (y - float(np.nanmean(y))) / sd


def clean_id(raw: str) -> str:
    return raw.strip().replace(".pdb", "").replace("_CA_A2", "")


def read_list(dataset_dir: Path, name: str) -> list[str]:
    path = dataset_dir / f"list-{name}.txt"
    return [clean_id(x) for x in path.read_text().splitlines() if x.strip()]


def membership_map(dataset_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ["small", "medium", "large"]:
        p = dataset_dir / f"list-{name}.txt"
        if not p.exists():
            continue
        for raw in p.read_text().splitlines():
            if raw.strip():
                out[clean_id(raw)] = name
    return out


def find_paths(root: Path, psl_dir_arg: str | None) -> tuple[Path, Path, Path]:
    candidates = [root]
    nested = root / "MDG_bfactor-main" / "MDG_bfactor-main"
    if nested.exists():
        candidates.append(nested)
    for base in candidates:
        dataset_dir = base / "datasets"
        annotation_dir = base / "features" / "features-blind-prediction"
        if dataset_dir.exists() and annotation_dir.exists():
            break
    else:
        raise FileNotFoundError("Could not locate MDG_bfactor-main datasets/features directories.")

    if psl_dir_arg:
        psl_dir = Path(psl_dir_arg)
    else:
        psl_candidates = [
            root / "analysis" / "features_psl",
            root / "data" / "processed" / "features_psl",
            root / "features_psl",
        ]
        psl_dir = next((p for p in psl_candidates if p.exists()), psl_candidates[0])
    if not psl_dir.exists():
        raise FileNotFoundError(f"PSL feature directory not found: {psl_dir}")
    return dataset_dir, annotation_dir, psl_dir


def psl_paths(psl_dir: Path, pdbid: str) -> tuple[Path | None, Path | None]:
    names = [pdbid, f"{pdbid}_CA_A2"]
    for name in names:
        x = psl_dir / f"{name}_psl_features.npy"
        y = psl_dir / f"{name}_bfactors.npy"
        if x.exists() and y.exists():
            return x, y
    return None, None


def psl_feature_mask(psl_dir: Path, pattern: str | None) -> tuple[np.ndarray | None, list[str] | None]:
    if not pattern:
        return None, None
    names_path = psl_dir / "feature_names.json"
    if not names_path.exists():
        raise FileNotFoundError(f"--psl-feature-regex requires {names_path}")
    names = json.loads(names_path.read_text(encoding="utf-8"))
    rx = re.compile(pattern)
    mask = np.asarray([bool(rx.search(str(name))) for name in names], dtype=bool)
    selected = [str(name) for name, keep in zip(names, mask) if keep]
    if not selected:
        raise ValueError(f"Pattern {pattern!r} selected no PSL features in {names_path}")
    return mask, selected


def greedy_bfactor_subsequence(short: np.ndarray, long: np.ndarray, tol: float) -> list[int] | None:
    idx: list[int] = []
    j = 0
    for val in short:
        found = False
        while j < len(long):
            if abs(float(long[j]) - float(val)) <= tol:
                idx.append(j)
                j += 1
                found = True
                break
            j += 1
        if not found:
            return None
    return idx


def trim_or_align_psl(
    pdbid: str,
    x_psl: np.ndarray,
    y_psl: np.ndarray,
    y_target: np.ndarray,
    mode: str,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    if x_psl.shape[0] == y_target.shape[0]:
        return x_psl, y_target, "exact"
    if mode == "skip":
        raise ValueError(f"row-count mismatch: PSL {x_psl.shape[0]}, target {y_target.shape[0]}")
    if mode == "prefix" and x_psl.shape[0] >= y_target.shape[0]:
        return x_psl[: y_target.shape[0]], y_target, "prefix"
    if mode == "bfactor-subsequence":
        idx = greedy_bfactor_subsequence(y_target, y_psl, tol)
        if idx is None:
            raise ValueError(f"could not align annotation B-factor subsequence for {pdbid}")
        return x_psl[np.asarray(idx, dtype=int)], y_target, "bfactor-subsequence"
    raise ValueError(f"cannot align {pdbid}: PSL {x_psl.shape[0]}, target {y_target.shape[0]}, mode={mode}")


def load_psl_records_for_table1(
    dataset_dir: Path,
    psl_dir: Path,
    dataset_name: str = "365",
    psl_mask: np.ndarray | None = None,
) -> list[ProteinRecord]:
    members = membership_map(dataset_dir)
    records: list[ProteinRecord] = []
    for pdbid in read_list(dataset_dir, dataset_name):
        x_path, y_path = psl_paths(psl_dir, pdbid)
        if x_path is None or y_path is None:
            continue
        x = np.load(x_path)
        if psl_mask is not None:
            x = x[:, psl_mask]
        y = np.load(y_path).astype(float)
        if x.shape[0] != y.shape[0] or y.size < 3 or np.nanstd(y) == 0:
            continue
        records.append(ProteinRecord(pdbid, x, y, members.get(pdbid, "365_only")))
    return records


def load_blind_records(
    dataset_dir: Path,
    annotation_dir: Path,
    psl_dir: Path,
    list_name: str,
    align: str,
    tol: float,
    use_paper_exclusions: bool,
    psl_mask: np.ndarray | None = None,
) -> tuple[list[ProteinRecord], dict[str, int]]:
    members = membership_map(dataset_dir)
    if list_name == "paper-exclusions-from-365":
        pdbids = [p for p in read_list(dataset_dir, "365") if p.upper() not in EXCLUDE_BLIND_PAPER]
    else:
        pdbids = read_list(dataset_dir, list_name)
    records: list[ProteinRecord] = []
    counts: dict[str, int] = {}
    for pdbid in pdbids:
        onehot = annotation_dir / f"{pdbid}-onehot.csv"
        labels = annotation_dir / f"{pdbid}.csv"
        x_path, y_path = psl_paths(psl_dir, pdbid)
        if not onehot.exists() or not labels.exists() or x_path is None or y_path is None:
            counts["missing"] = counts.get("missing", 0) + 1
            continue
        x_anno = pd.read_csv(onehot, header=None).to_numpy(dtype=float)
        y_target = pd.read_csv(labels)["B-factor"].to_numpy(dtype=float)
        x_psl = np.load(x_path)
        if psl_mask is not None:
            x_psl = x_psl[:, psl_mask]
        y_psl = np.load(y_path).astype(float)
        if x_anno.shape[0] != y_target.shape[0]:
            counts["annotation_target_mismatch"] = counts.get("annotation_target_mismatch", 0) + 1
            continue
        if np.nanstd(y_target) == 0 or y_target.size < 3:
            counts["bad_target"] = counts.get("bad_target", 0) + 1
            continue
        try:
            x_psl_aligned, y_aligned, align_status = trim_or_align_psl(pdbid, x_psl, y_psl, y_target, align, tol)
        except ValueError:
            counts["psl_target_mismatch"] = counts.get("psl_target_mismatch", 0) + 1
            continue
        counts[f"align_{align_status}"] = counts.get(f"align_{align_status}", 0) + 1
        if x_psl_aligned.shape[0] != x_anno.shape[0]:
            counts["postalign_annotation_psl_mismatch"] = counts.get("postalign_annotation_psl_mismatch", 0) + 1
            continue
        records.append(ProteinRecord(pdbid, x_psl_aligned, y_aligned, members.get(pdbid, "365_only"), x_anno))
    return records, counts


def model_factory(args: argparse.Namespace):
    if args.model == "rf":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            bootstrap=True,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    if args.model == "gbdt":
        return GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth if args.max_depth > 0 else 3,
            subsample=args.subsample,
            random_state=args.seed,
        )
    if args.model == "hgbt":
        return HistGradientBoostingRegressor(
            max_iter=args.n_estimators,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2,
            random_state=args.seed,
        )
    if args.model == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed")
        return XGBRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth if args.max_depth > 0 else 6,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            objective="reg:squarederror",
            tree_method=args.xgb_tree_method,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    raise ValueError(args.model)


def features_for(records: list[ProteinRecord], source: str) -> list[np.ndarray]:
    xs: list[np.ndarray] = []
    for r in records:
        if source == "psl":
            x = r.x_psl
        elif source == "annotation":
            if r.x_anno is None:
                raise ValueError("annotation features requested but absent")
            x = r.x_anno
        elif source == "annotation+psl":
            if r.x_anno is None:
                raise ValueError("annotation features requested but absent")
            x = np.column_stack([r.x_anno, r.x_psl])
        else:
            raise ValueError(source)
        xs.append(np.asarray(x, dtype=float))
    return xs


def maybe_scale_train_test(x_train: np.ndarray, x_test: np.ndarray, scale: str) -> tuple[np.ndarray, np.ndarray]:
    if scale == "none":
        return x_train, x_test
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


def table1_lr(records: list[ProteinRecord], ridge: bool = False, scale: str = "protein") -> pd.DataFrame:
    rows = []
    for r in records:
        x = np.asarray(r.x_psl, dtype=float)
        y = np.asarray(r.y_raw, dtype=float)
        if scale == "protein":
            x = StandardScaler().fit_transform(x)
        model = RidgeCV(alphas=np.logspace(-6, 6, 25)) if ridge else LinearRegression()
        pred = model.fit(x, y).predict(x)
        rows.append({
            "pdbid": r.pdbid,
            "membership": r.membership,
            "n_residues": len(y),
            "pcc": pcc(y, pred),
            "spearman": spcc(y, pred),
            "rmse_raw": rmse(y, pred),
        })
    df = pd.DataFrame(rows)
    return df


def summarize_per_protein(per: pd.DataFrame, label_col: str = "membership") -> pd.DataFrame:
    rows = []
    labels = ["small", "medium", "large", "365_only", "all"]
    for lab in labels:
        g = per if lab == "all" else per[per[label_col] == lab]
        if g.empty:
            continue
        rows.append({
            "dataset": lab,
            "n_proteins": int(g.shape[0]),
            "n_residues": int(g["n_residues"].sum()) if "n_residues" in g else int(g["n_test"].sum()),
            "mean_pcc": float(np.nanmean(g["pcc"])),
            "median_pcc": float(np.nanmedian(g["pcc"])),
            "mean_spearman": float(np.nanmean(g["spearman"])),
            "mean_rmse": float(np.nanmean(g[[c for c in g.columns if c.startswith("rmse")][0]])),
        })
    return pd.DataFrame(rows)


def evaluate_protein_kfold(records: list[ProteinRecord], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    xs = features_for(records, args.feature_source)
    ys = [zscore(r.y_raw) if args.target == "z" else r.y_raw for r in records]
    protein_idx = np.arange(len(records))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    pred_rows = []
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(protein_idx), start=1):
        x_train = np.vstack([xs[i] for i in train_idx])
        y_train = np.concatenate([ys[i] for i in train_idx])
        x_test = np.vstack([xs[i] for i in test_idx])
        y_test = np.concatenate([ys[i] for i in test_idx])
        x_train, x_test = maybe_scale_train_test(x_train, x_test, args.scale)
        model = model_factory(args)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        fold_rows.append({"fold": fold, "n_test": len(y_test), "pooled_pcc": pcc(y_test, pred), "pooled_rmse": rmse(y_test, pred)})
        offset = 0
        for i in test_idx:
            n = len(ys[i])
            pred_rows.append(pd.DataFrame({
                "fold": fold,
                "pdbid": records[i].pdbid,
                "membership": records[i].membership,
                "y": ys[i],
                "pred": pred[offset:offset+n],
            }))
            offset += n
    preds = pd.concat(pred_rows, ignore_index=True)
    per_rows = []
    for pdbid, g in preds.groupby("pdbid"):
        per_rows.append({
            "pdbid": pdbid,
            "membership": g["membership"].iloc[0],
            "n_residues": len(g),
            "pcc": pcc(g["y"].to_numpy(), g["pred"].to_numpy()),
            "spearman": spcc(g["y"].to_numpy(), g["pred"].to_numpy()),
            "rmse": rmse(g["y"].to_numpy(), g["pred"].to_numpy()),
        })
    per = pd.DataFrame(per_rows)
    folds = pd.DataFrame(fold_rows)
    overall = pd.DataFrame([{
        "protocol": "protein-kfold",
        "feature_source": args.feature_source,
        "model": args.model,
        "target": args.target,
        "n_proteins": len(records),
        "n_residues": int(sum(len(y) for y in ys)),
        "pooled_pcc_all_predictions": pcc(preds["y"].to_numpy(), preds["pred"].to_numpy()),
        "mean_fold_pooled_pcc": float(np.nanmean(folds["pooled_pcc"])),
        "mean_per_protein_pcc": float(np.nanmean(per["pcc"])),
        "mean_per_protein_spearman": float(np.nanmean(per["spearman"])),
        "mean_per_protein_rmse": float(np.nanmean(per["rmse"])),
    }])
    return overall, per


def evaluate_atom_kfold(records: list[ProteinRecord], args: argparse.Namespace) -> pd.DataFrame:
    xs = features_for(records, args.feature_source)
    ys = [zscore(r.y_raw) if args.target == "z" else r.y_raw for r in records]
    X = np.vstack(xs)
    y = np.concatenate(ys)
    groups = np.concatenate([[r.pdbid] * len(r.y_raw) for r in records])
    idx = np.arange(X.shape[0])
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    pred = np.empty_like(y, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(kf.split(idx), start=1):
        x_train, x_test = X[train_idx], X[test_idx]
        x_train, x_test = maybe_scale_train_test(x_train, x_test, args.scale)
        model = model_factory(args)
        model.fit(x_train, y[train_idx])
        pred[test_idx] = model.predict(x_test)
    # Protein overlap diagnostic: how many proteins appear in both train/test in atom CV is essentially all.
    return pd.DataFrame([{
        "protocol": "atom-kfold",
        "feature_source": args.feature_source,
        "model": args.model,
        "target": args.target,
        "n_proteins": len(records),
        "n_residues": int(X.shape[0]),
        "pooled_pcc_all_predictions": pcc(y, pred),
        "pooled_spearman_all_predictions": spcc(y, pred),
        "pooled_rmse_all_predictions": rmse(y, pred),
        "proteins_split_across_folds": int(pd.Series(groups).nunique()),
    }])


def write_outputs(out_dir: Path, name: str, frames: dict[str, pd.DataFrame], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    for key, df in frames.items():
        df.to_csv(out_dir / f"{name}_{key}.csv", index=False)
        print(f"\n[{key}]\n{df.to_string(index=False)}")
    print(f"\nWrote outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root containing MDG_bfactor-main or its parent")
    ap.add_argument("--psl-dir", default=None, help="Directory containing *_psl_features.npy and *_bfactors.npy")
    ap.add_argument("--psl-feature-regex", default=None, help="Optional regex selecting columns from feature_names.json")
    ap.add_argument("--mode", choices=["table1-lr", "blind-protein-kfold", "blind-atom-kfold"], required=True)
    ap.add_argument("--list-name", default="list-blind-prediction", help="For blind modes: list-blind-prediction, 365, or paper-exclusions-from-365")
    ap.add_argument("--align", choices=["skip", "prefix", "bfactor-subsequence"], default="bfactor-subsequence")
    ap.add_argument("--bfactor-align-tol", type=float, default=0.05)
    ap.add_argument("--feature-source", choices=["psl", "annotation", "annotation+psl"], default="annotation+psl")
    ap.add_argument("--target", choices=["raw", "z"], default="raw")
    ap.add_argument("--model", choices=["rf", "gbdt", "hgbt", "xgboost"], default="gbdt")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--scale", choices=["none", "global"], default="none")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--max-features", default="sqrt")
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--colsample-bytree", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--xgb-tree-method", default="hist", help="Use gpu_hist only if your XGBoost supports it")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260516)
    ap.add_argument("--out-dir", default="hayes_reproduction_results")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    dataset_dir, annotation_dir, psl_dir = find_paths(root, args.psl_dir)
    psl_mask, selected_psl_features = psl_feature_mask(psl_dir, args.psl_feature_regex)
    meta = {
        **vars(args),
        "dataset_dir": str(dataset_dir),
        "annotation_dir": str(annotation_dir),
        "psl_dir": str(psl_dir),
        "selected_psl_features": selected_psl_features,
    }
    if args.mode == "table1-lr":
        records = load_psl_records_for_table1(dataset_dir, psl_dir, "365", psl_mask)
        per = table1_lr(records, ridge=False, scale="protein")
        summary = summarize_per_protein(per)
        write_outputs(Path(args.out_dir), "table1_lr", {"summary": summary, "per_protein": per}, meta | {"n_loaded": len(records)})
        return

    # Normalize list-name argument to actual dataset filename stem.
    list_name = args.list_name
    if list_name.startswith("list-"):
        list_name = list_name.removeprefix("list-")
    records, counts = load_blind_records(
        dataset_dir,
        annotation_dir,
        psl_dir,
        list_name,
        args.align,
        args.bfactor_align_tol,
        False,
        psl_mask,
    )
    meta["load_counts"] = counts
    meta["n_loaded"] = len(records)
    if not records:
        raise RuntimeError(f"No records loaded. Counts: {counts}")
    print(f"Loaded {len(records)} proteins / {sum(len(r.y_raw) for r in records)} residues. Counts: {counts}")
    if args.mode == "blind-protein-kfold":
        overall, per = evaluate_protein_kfold(records, args)
        summary = summarize_per_protein(per)
        write_outputs(Path(args.out_dir), "blind_protein_kfold", {"overall": overall, "summary_by_set": summary, "per_protein": per}, meta)
    elif args.mode == "blind-atom-kfold":
        overall = evaluate_atom_kfold(records, args)
        write_outputs(Path(args.out_dir), "blind_atom_kfold", {"overall": overall}, meta)


if __name__ == "__main__":
    main()
