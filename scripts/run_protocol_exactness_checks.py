"""
Final protocol exactness checks for the PSL flexibility manuscript.

This script is intentionally narrow: it audits metric aggregation and split
effects around Hayes-style protocols without changing model capacity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from hayes_protocol_runner import (
    features_for,
    find_paths,
    load_blind_records,
    load_psl_records_for_table1,
    pcc,
    psl_feature_mask,
    rmse,
    spcc,
)
from run_baselines import (
    ANNOTATION_DIR,
    PDB_DIR,
    classical_features,
    graph_features,
    load_lists,
    membership_for,
    parse_ca_pdb,
)


DEFAULT_SEED = 20260516


@dataclass
class ControlProtein:
    pdbid: str
    membership: str
    y_raw: np.ndarray
    feature_sets: dict[str, np.ndarray]


def safe_pcc(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if y.size < 3 or np.allclose(y, y[0]) or np.allclose(pred, pred[0]):
        return float("nan")
    return float(pearsonr(y, pred).statistic)


def safe_spearman(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if y.size < 3 or np.allclose(y, y[0]) or np.allclose(pred, pred[0]):
        return float("nan")
    return float(spearmanr(y, pred).statistic)


def safe_rmse(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2)))


def scale_feature_blocks(xs: list[np.ndarray], mode: str) -> list[np.ndarray]:
    if mode == "none":
        return [np.asarray(x, dtype=float) for x in xs]
    if mode == "global_z":
        scaler = StandardScaler().fit(np.vstack(xs))
        return [scaler.transform(x) for x in xs]
    if mode == "per_protein_z":
        return [StandardScaler().fit_transform(x) for x in xs]
    raise ValueError(mode)


def summarize_predictions(
    label: dict[str, object],
    pdbids: list[str],
    memberships: list[str],
    ys: list[np.ndarray],
    preds: list[np.ndarray],
) -> dict[str, object]:
    per_rows = []
    for pdbid, membership, y, pred in zip(pdbids, memberships, ys, preds):
        per_rows.append(
            {
                "pdbid": pdbid,
                "membership": membership,
                "n_residues": int(len(y)),
                "pcc": safe_pcc(y, pred),
                "spearman": safe_spearman(y, pred),
                "rmse_raw": safe_rmse(y, pred),
            }
        )
    per = pd.DataFrame(per_rows)
    y_all = np.concatenate(ys)
    pred_all = np.concatenate(preds)
    return {
        **label,
        "n_proteins": int(len(ys)),
        "n_residues": int(sum(len(y) for y in ys)),
        "mean_per_protein_pcc": float(np.nanmean(per["pcc"])),
        "median_per_protein_pcc": float(np.nanmedian(per["pcc"])),
        "mean_per_protein_spearman": float(np.nanmean(per["spearman"])),
        "mean_per_protein_rmse_raw": float(np.nanmean(per["rmse_raw"])),
        "pooled_pcc": safe_pcc(y_all, pred_all),
        "pooled_spearman": safe_spearman(y_all, pred_all),
        "pooled_rmse_raw": safe_rmse(y_all, pred_all),
        "small_mean_pcc": float(np.nanmean(per.loc[per["membership"] == "small", "pcc"])),
        "medium_mean_pcc": float(np.nanmean(per.loc[per["membership"] == "medium", "pcc"])),
        "large_mean_pcc": float(np.nanmean(per.loc[per["membership"] == "large", "pcc"])),
    }


def table1_lr_audit(root: Path, psl_dir: Path) -> pd.DataFrame:
    dataset_dir, _, resolved_psl_dir = find_paths(root, str(psl_dir))
    records = load_psl_records_for_table1(dataset_dir, resolved_psl_dir, "365")
    pdbids = [r.pdbid for r in records]
    memberships = [r.membership for r in records]
    ys = [np.asarray(r.y_raw, dtype=float) for r in records]
    xs_raw = [np.asarray(r.x_psl, dtype=float) for r in records]
    rows: list[dict[str, object]] = []

    for scaling in ["none", "global_z", "per_protein_z"]:
        xs = scale_feature_blocks(xs_raw, scaling)
        model = LinearRegression()
        model.fit(np.vstack(xs), np.concatenate(ys))
        preds = [model.predict(x) for x in xs]
        rows.append(
            summarize_predictions(
                {
                    "audit": "table1_lr",
                    "fit_scope": "pooled_ols",
                    "feature_scaling": scaling,
                    "prediction_rescale": "none",
                    "target": "raw_B",
                },
                pdbids,
                memberships,
                ys,
                preds,
            )
        )

    for scaling in ["none", "global_z"]:
        xs = scale_feature_blocks(xs_raw, scaling)
        model = LinearRegression()
        model.fit(np.vstack(xs), np.concatenate(ys))
        base_preds = [model.predict(x) for x in xs]
        preds = []
        for y, pred in zip(ys, base_preds):
            if np.nanstd(pred) == 0 or len(y) < 3:
                preds.append(pred)
                continue
            affine = LinearRegression().fit(pred.reshape(-1, 1), y)
            preds.append(affine.predict(pred.reshape(-1, 1)))
        rows.append(
            summarize_predictions(
                {
                    "audit": "table1_lr",
                    "fit_scope": "pooled_ols",
                    "feature_scaling": scaling,
                    "prediction_rescale": "per_protein_affine",
                    "target": "raw_B",
                },
                pdbids,
                memberships,
                ys,
                preds,
            )
        )

    for scaling in ["none", "per_protein_z"]:
        preds = []
        for x_raw, y in zip(xs_raw, ys):
            x = StandardScaler().fit_transform(x_raw) if scaling == "per_protein_z" else x_raw
            preds.append(LinearRegression().fit(x, y).predict(x))
        rows.append(
            summarize_predictions(
                {
                    "audit": "table1_lr",
                    "fit_scope": "per_protein_ols",
                    "feature_scaling": scaling,
                    "prediction_rescale": "none",
                    "target": "raw_B",
                },
                pdbids,
                memberships,
                ys,
                preds,
            )
        )

    return pd.DataFrame(rows)


def blind_model(model_name: str, seed: int, n_estimators: int, max_depth: int, n_jobs: int):
    if model_name == "gbdt":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.03,
            max_depth=max_depth if max_depth > 0 else 3,
            subsample=1.0,
            random_state=seed,
        )
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
        )
    raise ValueError(model_name)


def blind_records(root: Path, psl_dir: Path):
    dataset_dir, annotation_dir, resolved_psl_dir = find_paths(root, str(psl_dir))
    psl_mask, _ = psl_feature_mask(resolved_psl_dir, None)
    records, counts = load_blind_records(
        dataset_dir=dataset_dir,
        annotation_dir=annotation_dir,
        psl_dir=resolved_psl_dir,
        list_name="blind-prediction",
        align="bfactor-subsequence",
        tol=0.05,
        use_paper_exclusions=False,
        psl_mask=psl_mask,
    )
    return records, counts


def random_residue_metric_audit(
    root: Path,
    psl_dir: Path,
    folds: int,
    seed: int,
    n_estimators: int,
    max_depth: int,
    n_jobs: int,
) -> pd.DataFrame:
    records, counts = blind_records(root, psl_dir)
    xs = features_for(records, "annotation+psl")
    ys = [np.asarray(r.y_raw, dtype=float) for r in records]
    X = np.vstack(xs)
    y = np.concatenate(ys)
    groups = np.concatenate([[r.pdbid] * len(r.y_raw) for r in records])
    memberships = {r.pdbid: r.membership for r in records}
    idx = np.arange(X.shape[0])
    rows = []
    for model_name in ["gbdt", "rf"]:
        pred = np.empty_like(y, dtype=float)
        fold_rows = []
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(kf.split(idx), start=1):
            model = blind_model(model_name, seed, n_estimators, max_depth, n_jobs)
            model.fit(X[train_idx], y[train_idx])
            pred[test_idx] = model.predict(X[test_idx])
            fold_rows.append(
                {
                    "fold": fold,
                    "n_test": int(len(test_idx)),
                    "fold_pcc": safe_pcc(y[test_idx], pred[test_idx]),
                    "fold_spearman": safe_spearman(y[test_idx], pred[test_idx]),
                    "fold_rmse": safe_rmse(y[test_idx], pred[test_idx]),
                }
            )
        per_rows = []
        for pdbid in pd.unique(groups):
            mask = groups == pdbid
            per_rows.append(
                {
                    "pdbid": pdbid,
                    "membership": memberships[str(pdbid)],
                    "n_residues": int(mask.sum()),
                    "pcc": safe_pcc(y[mask], pred[mask]),
                    "spearman": safe_spearman(y[mask], pred[mask]),
                    "rmse_raw": safe_rmse(y[mask], pred[mask]),
                }
            )
        folds_df = pd.DataFrame(fold_rows)
        per = pd.DataFrame(per_rows)
        rows.append(
            {
                "protocol": "random_residue_raw_B",
                "feature_set": "annotation+PSL 7/10/13 std",
                "model": model_name.upper(),
                "n_proteins": int(len(records)),
                "n_residues": int(len(y)),
                "pooled_pcc": safe_pcc(y, pred),
                "mean_fold_pcc": float(np.nanmean(folds_df["fold_pcc"])),
                "mean_per_protein_pcc": float(np.nanmean(per["pcc"])),
                "pooled_spearman": safe_spearman(y, pred),
                "mean_fold_spearman": float(np.nanmean(folds_df["fold_spearman"])),
                "mean_per_protein_spearman": float(np.nanmean(per["spearman"])),
                "pooled_rmse_raw": safe_rmse(y, pred),
                "mean_fold_rmse_raw": float(np.nanmean(folds_df["fold_rmse"])),
                "mean_per_protein_rmse_raw": float(np.nanmean(per["rmse_raw"])),
                "proteins_split_across_folds": int(pd.Series(groups).nunique()),
                "alignment_counts": json.dumps(counts, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def load_exact_control_proteins(psl_dir: Path) -> list[ControlProtein]:
    lists = load_lists()
    proteins: list[ControlProtein] = []
    for label_path in sorted(ANNOTATION_DIR.glob("*.csv")):
        if label_path.name.endswith("-onehot.csv"):
            continue
        pdbid = label_path.stem
        psl_name = f"{pdbid}_CA_A2"
        psl_path = psl_dir / f"{psl_name}_psl_features.npy"
        pdb_path = PDB_DIR / f"{psl_name}.pdb"
        if not psl_path.exists() or not pdb_path.exists():
            continue
        labels = pd.read_csv(label_path)
        psl = np.load(psl_path)
        coords, _ = parse_ca_pdb(pdb_path)
        y_raw = labels["B-factor"].to_numpy(dtype=float)
        if psl.shape[0] != y_raw.size or coords.shape[0] != y_raw.size:
            continue
        if y_raw.size < 3 or np.nanstd(y_raw) == 0:
            continue
        x_classical = classical_features(labels)
        x_graph = graph_features(coords)
        proteins.append(
            ControlProtein(
                pdbid=pdbid,
                membership=membership_for(pdbid, lists),
                y_raw=y_raw,
                feature_sets={
                    "PSL-only": psl,
                    "Classical structural": x_classical,
                    "Simple graph": x_graph,
                    "Classical+PSL": np.column_stack([x_classical, psl]),
                },
            )
        )
    if not proteins:
        raise RuntimeError("No exact-aligned control proteins loaded.")
    return proteins


def control_model(seed: int, n_estimators: int, n_jobs: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=n_jobs,
        random_state=seed,
    )


def evaluate_control_split(
    proteins: list[ControlProtein],
    feature_set: str,
    split: str,
    folds: int,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> dict[str, object]:
    if split == "protein":
        protein_idx = np.arange(len(proteins))
        pred_blocks: dict[str, np.ndarray] = {}
        fold_pccs = []
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(protein_idx):
            x_train = np.vstack([proteins[i].feature_sets[feature_set] for i in train_idx])
            y_train = np.concatenate([proteins[i].y_raw for i in train_idx])
            model = control_model(seed, n_estimators, n_jobs)
            model.fit(x_train, y_train)
            y_test_fold = []
            pred_test_fold = []
            for i in test_idx:
                pred = model.predict(proteins[i].feature_sets[feature_set])
                pred_blocks[proteins[i].pdbid] = pred
                y_test_fold.append(proteins[i].y_raw)
                pred_test_fold.append(pred)
            fold_pccs.append(safe_pcc(np.concatenate(y_test_fold), np.concatenate(pred_test_fold)))
    elif split == "random_residue":
        X = np.vstack([p.feature_sets[feature_set] for p in proteins])
        y = np.concatenate([p.y_raw for p in proteins])
        group = np.concatenate([[p.pdbid] * len(p.y_raw) for p in proteins])
        pred = np.empty_like(y, dtype=float)
        fold_pccs = []
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(np.arange(len(y))):
            model = control_model(seed, n_estimators, n_jobs)
            model.fit(X[train_idx], y[train_idx])
            pred[test_idx] = model.predict(X[test_idx])
            fold_pccs.append(safe_pcc(y[test_idx], pred[test_idx]))
        pred_blocks = {pdbid: pred[group == pdbid] for pdbid in pd.unique(group)}
    else:
        raise ValueError(split)

    ys = [p.y_raw for p in proteins]
    preds = [pred_blocks[p.pdbid] for p in proteins]
    row = summarize_predictions(
        {
            "protocol": "split_dependence_control",
            "feature_set": feature_set,
            "split": split,
            "target": "raw_B",
            "model": "RF-500 depth12 leaf2 sqrt" if n_estimators == 500 else f"RF-{n_estimators} depth12 leaf2 sqrt",
        },
        [p.pdbid for p in proteins],
        [p.membership for p in proteins],
        ys,
        preds,
    )
    row["mean_fold_pcc"] = float(np.nanmean(fold_pccs))
    return row


def split_dependence_controls(
    psl_dir: Path,
    folds: int,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> pd.DataFrame:
    proteins = load_exact_control_proteins(psl_dir)
    rows: list[dict[str, object]] = []
    for feature_set in ["PSL-only", "Classical structural", "Simple graph", "Classical+PSL"]:
        for split in ["protein", "random_residue"]:
            rows.append(evaluate_control_split(proteins, feature_set, split, folds, seed, n_estimators, n_jobs))
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--table1-psl-dir", default="data/processed/features_psl_labeled_6_9_12_median")
    parser.add_argument("--blind-psl-dir", default="data/processed/features_psl_labeled_7_10_13_std")
    parser.add_argument("--out-dir", default="results/metrics")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--only", choices=["all", "table1", "random", "controls"], default="all")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table1_psl_dir = resolve_path(root, args.table1_psl_dir)
    blind_psl_dir = resolve_path(root, args.blind_psl_dir)

    if args.only in {"all", "table1"}:
        print("Running Table 1 LR audit...")
        table1 = table1_lr_audit(root, table1_psl_dir)
        table1.to_csv(out_dir / "table1_lr_protocol_audit.csv", index=False)
        print(table1.to_string(index=False))

    if args.only in {"all", "random"}:
        print("\nRunning random residue metric aggregation audit...")
        random_metrics = random_residue_metric_audit(
            root,
            blind_psl_dir,
            args.folds,
            args.seed,
            args.n_estimators,
            args.max_depth,
            args.n_jobs,
        )
        random_metrics.to_csv(out_dir / "random_residue_metric_audit.csv", index=False)
        print(random_metrics.to_string(index=False))

    if args.only in {"all", "controls"}:
        print("\nRunning split-dependence controls...")
        controls = split_dependence_controls(
            blind_psl_dir,
            args.folds,
            args.seed,
            args.n_estimators,
            args.n_jobs,
        )
        controls.to_csv(out_dir / "split_dependence_controls.csv", index=False)
        print(controls.to_string(index=False))


if __name__ == "__main__":
    main()
