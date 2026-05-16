"""
Baseline feature-set comparisons for the protein flexibility manuscript.

All feature sets are evaluated with the same protein-level cross-validation,
within-protein B-factor normalization, Random Forest configuration, and
protein-level bootstrap uncertainty used by the primary PSL experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "MDG_bfactor-main" / "MDG_bfactor-main" / "datasets"
ANNOTATION_DIR = ROOT / "MDG_bfactor-main" / "MDG_bfactor-main" / "features" / "features-blind-prediction"
PDB_DIR = DATASET_DIR / "365"
PSL_DIR_CANDIDATES = [
    ROOT / "data" / "processed" / "features_psl",
    ROOT / "analysis" / "features_psl",
]
OUT_DIR = ROOT / "results" / "metrics"
RUN_DIR = ROOT / "results" / "runs"

RANDOM_STATE = 20260504
BOOTSTRAPS = 2000
CONTACT_RADII = (6.0, 8.0, 10.0, 12.0)
SEC_CODES = tuple(range(7))


@dataclass
class ProteinData:
    pdbid: str
    membership: str
    y: np.ndarray
    feature_sets: dict[str, np.ndarray]


def find_psl_dir(psl_dir_arg: str | None = None) -> Path:
    if psl_dir_arg:
        psl_dir = Path(psl_dir_arg)
        if not psl_dir.is_absolute():
            psl_dir = ROOT / psl_dir
        if psl_dir.exists() and any(psl_dir.glob("*_psl_features.npy")):
            return psl_dir
        raise FileNotFoundError(f"No PSL feature matrices found in {psl_dir}")
    for path in PSL_DIR_CANDIDATES:
        if path.exists() and any(path.glob("*_psl_features.npy")):
            return path
    searched = ", ".join(str(p) for p in PSL_DIR_CANDIDATES)
    raise FileNotFoundError(f"No PSL feature matrices found. Searched: {searched}")


def load_lists() -> dict[str, set[str]]:
    lists = {}
    for name in ["small", "medium", "large", "365"]:
        path = DATASET_DIR / f"list-{name}.txt"
        lists[name] = {line.strip().replace(".pdb", "") for line in path.read_text().splitlines() if line.strip()}
    return lists


def membership_for(pdbid: str, lists: dict[str, set[str]]) -> str:
    variants = {pdbid, f"{pdbid}_CA_A2"}
    memberships = [name for name in ["small", "medium", "large"] if variants.intersection(lists[name])]
    return memberships[0] if memberships else "365_only"


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(pearsonr(a, b).statistic)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(RANDOM_STATE)
    reps = np.empty(BOOTSTRAPS, dtype=float)
    for idx in range(BOOTSTRAPS):
        sample = rng.choice(values, size=values.size, replace=True)
        reps[idx] = np.mean(sample)
    return float(np.mean(values)), float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def parse_ca_pdb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    bfactors = []
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


def position_features(n_residues: int) -> np.ndarray:
    idx = np.arange(n_residues, dtype=float)
    denom = max(n_residues - 1, 1)
    norm_idx = idx / denom
    terminal_distance = np.minimum(idx, denom - idx) / denom
    terminal_decile = (terminal_distance <= 0.10).astype(float)
    length = np.full(n_residues, float(n_residues))
    inv_length = np.full(n_residues, 1.0 / max(n_residues, 1))
    return np.column_stack([norm_idx, terminal_distance, terminal_decile, length, inv_length])


def classical_features(labels: pd.DataFrame) -> np.ndarray:
    n = labels.shape[0]
    base_cols = ["area", "packing_density1", "packing_density2", "packing_density3"]
    base = labels[base_cols].to_numpy(dtype=float)
    sec = labels["sec_type"].to_numpy(dtype=int)
    sec_onehot = np.column_stack([(sec == code).astype(float) for code in SEC_CODES])
    return np.column_stack([base, position_features(n), sec_onehot])


def graph_features(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    cols = []

    for radius in CONTACT_RADII:
        adj = dists <= radius
        degree = adj.sum(axis=1).astype(float)
        cols.append(degree)
        cols.append(degree / max(n - 1, 1))

    for radius in (8.0, 12.0):
        adj = dists <= radius
        clustering = np.zeros(n, dtype=float)
        for i in range(n):
            neigh = np.flatnonzero(adj[i])
            k = neigh.size
            if k < 2:
                continue
            sub = adj[np.ix_(neigh, neigh)]
            edges = sub.sum() / 2.0
            clustering[i] = (2.0 * edges) / (k * (k - 1))
        cols.append(clustering)

    finite_dists = np.sort(dists, axis=1)
    for k in (4, 8, 12):
        kk = min(k, max(n - 1, 1))
        vals = finite_dists[:, :kk]
        vals[~np.isfinite(vals)] = np.nan
        cols.append(np.nanmean(vals, axis=1))

    return np.column_stack(cols)


def load_proteins(psl_dir_arg: str | None = None) -> list[ProteinData]:
    psl_dir = find_psl_dir(psl_dir_arg)
    lists = load_lists()
    proteins = []
    skipped_mismatch = 0
    bfactor_order_differences = 0

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
        coords, pdb_bfactors = parse_ca_pdb(pdb_path)
        y_raw = labels["B-factor"].to_numpy(dtype=float)
        if psl.shape[0] != y_raw.size or coords.shape[0] != y_raw.size:
            skipped_mismatch += 1
            continue
        if np.nanstd(y_raw) == 0:
            continue
        if not np.allclose(y_raw, pdb_bfactors, atol=0.05):
            bfactor_order_differences += 1

        y = (y_raw - np.nanmean(y_raw)) / np.nanstd(y_raw)
        x_classical = classical_features(labels)
        x_graph = graph_features(coords)
        feature_sets = {
            "residue_position": position_features(y_raw.size),
            "classical_structural": x_classical,
            "simple_graph": x_graph,
            "psl": psl,
            "classical_plus_psl": np.column_stack([x_classical, psl]),
        }
        proteins.append(
            ProteinData(
                pdbid=pdbid,
                membership=membership_for(pdbid, lists),
                y=y,
                feature_sets=feature_sets,
            )
        )

    if not proteins:
        raise FileNotFoundError("No common annotation, PDB, and PSL protein records were found.")
    if skipped_mismatch:
        print(f"Skipped {skipped_mismatch} proteins with mismatched annotation/PDB/PSL row counts.")
    if bfactor_order_differences:
        print(
            f"{bfactor_order_differences} retained proteins had annotation/PDB B-factor value "
            "differences beyond rounding tolerance; rows were still aligned by residue order."
        )
    return proteins


def model_factory(args: argparse.Namespace) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=args.n_jobs,
    )


def parse_max_features(value: str) -> str | int | float | None:
    value = value.strip()
    if value.lower() in {"none", "all", "1.0"}:
        return None
    if value.lower() in {"sqrt", "log2"}:
        return value.lower()
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid max_features value: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max_features must be positive")
    if parsed < 1:
        return parsed
    if float(parsed).is_integer():
        return int(parsed)
    raise argparse.ArgumentTypeError("max_features values greater than 1 must be integers")


def evaluate_feature_set(
    proteins: list[ProteinData],
    feature_set: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)
    protein_indices = np.arange(len(proteins))
    prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(protein_indices), start=1):
        train_proteins = [proteins[i] for i in train_idx]
        test_proteins = [proteins[i] for i in test_idx]
        x_train = np.vstack([p.feature_sets[feature_set] for p in train_proteins])
        y_train = np.concatenate([p.y for p in train_proteins])
        model = model_factory(args)
        model.fit(x_train, y_train)

        for p in test_proteins:
            pred = model.predict(p.feature_sets[feature_set])
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold_idx,
                        "feature_set": feature_set,
                        "pdbid": p.pdbid,
                        "membership": p.membership,
                        "bfactor_z": p.y,
                        "prediction_z": pred,
                    }
                )
            )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    per_protein = []
    for pdbid, group in predictions.groupby("pdbid", sort=True):
        y = group["bfactor_z"].to_numpy(dtype=float)
        pred = group["prediction_z"].to_numpy(dtype=float)
        per_protein.append(
            {
                "feature_set": feature_set,
                "pdbid": pdbid,
                "membership": group["membership"].iloc[0],
                "n_residues": int(group.shape[0]),
                "pcc": safe_pearson(y, pred),
                "spearman": safe_spearman(y, pred),
                "rmse_z": rmse(y, pred),
            }
        )
    return pd.DataFrame(per_protein), predictions


def summarize(feature_set: str, per_protein: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float | str | int]:
    pcc, pcc_lo, pcc_hi = bootstrap_ci(per_protein["pcc"].to_numpy(dtype=float))
    spear, spear_lo, spear_hi = bootstrap_ci(per_protein["spearman"].to_numpy(dtype=float))
    err, err_lo, err_hi = bootstrap_ci(per_protein["rmse_z"].to_numpy(dtype=float))
    return {
        "feature_set": feature_set,
        "n_proteins": int(per_protein.shape[0]),
        "n_residues": int(predictions.shape[0]),
        "mean_pcc": pcc,
        "pcc_ci95_low": pcc_lo,
        "pcc_ci95_high": pcc_hi,
        "mean_spearman": spear,
        "spearman_ci95_low": spear_lo,
        "spearman_ci95_high": spear_hi,
        "mean_rmse_z": err,
        "rmse_z_ci95_low": err_lo,
        "rmse_z_ci95_high": err_hi,
        "pooled_pcc": safe_pearson(predictions["bfactor_z"].to_numpy(), predictions["prediction_z"].to_numpy()),
        "pooled_spearman": safe_spearman(predictions["bfactor_z"].to_numpy(), predictions["prediction_z"].to_numpy()),
        "pooled_rmse_z": rmse(predictions["bfactor_z"].to_numpy(), predictions["prediction_z"].to_numpy()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=parse_max_features, default="sqrt")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--psl-dir", default=None, help="Directory containing PSL feature matrices")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    proteins = load_proteins(args.psl_dir)
    print(f"Loaded {len(proteins)} common proteins with {sum(p.y.size for p in proteins)} residues.")

    summaries = []
    all_per_protein = []
    all_predictions = []
    feature_sets = ["residue_position", "classical_structural", "simple_graph", "psl", "classical_plus_psl"]
    for feature_set in feature_sets:
        print(f"Evaluating {feature_set}...")
        per_protein, predictions = evaluate_feature_set(proteins, feature_set, args)
        summaries.append(summarize(feature_set, per_protein, predictions))
        all_per_protein.append(per_protein)
        all_predictions.append(predictions)

    summary = pd.DataFrame(summaries)
    per_protein = pd.concat(all_per_protein, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary.to_csv(OUT_DIR / "baseline_comparison_metrics.csv", index=False)
    per_protein.to_csv(OUT_DIR / "baseline_comparison_per_protein.csv", index=False)
    predictions.to_csv(RUN_DIR / "baseline_comparison_predictions.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
