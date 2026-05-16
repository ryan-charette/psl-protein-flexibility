"""
Random Forest evaluation for the protein flexibility project.

The primary protocol is cross-validation by protein: each prediction is made
for residues from proteins outside the corresponding training fold.

Feature sources:
  annotation
    Uses the residue-level one-hot/biophysical feature matrices already shipped in
    MDG_bfactor-main/features/features-blind-prediction. This is immediately
    runnable from the supplied project files.

  psl
    Uses cached PSL matrices under a PSL feature directory. Generate these first
    with generate_psl_features.py or psl_variant_experiment.py. If absent, this
    script fails with a clear error.

Outputs:
  results/runs/rf_<run_name>_summary.md
  results/runs/rf_<run_name>_dataset_metrics.csv
  results/runs/rf_<run_name>_per_protein_metrics.csv
  results/runs/rf_<run_name>_predictions.csv
  results/runs/rf_<run_name>_feature_importance.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "MDG_bfactor-main" / "MDG_bfactor-main" / "datasets"
ANNOTATION_DIR = ROOT / "MDG_bfactor-main" / "MDG_bfactor-main" / "features" / "features-blind-prediction"
PSL_DIR_CANDIDATES = [
    ROOT / "data" / "processed" / "features_psl",
    ROOT / "analysis" / "features_psl",
]
OUT_DIR = ROOT / "results" / "runs"

RANDOM_STATE = 20260504
BOOTSTRAPS = 2000


@dataclass
class ProteinData:
    pdbid: str
    dataset_membership: str
    x: np.ndarray
    y_raw: np.ndarray
    y: np.ndarray
    residue_index: np.ndarray


def load_lists() -> dict[str, set[str]]:
    lists = {}
    for name in ["small", "medium", "large", "365"]:
        path = DATASET_DIR / f"list-{name}.txt"
        lists[name] = {line.strip().replace(".pdb", "") for line in path.read_text().splitlines() if line.strip()}
    return lists


def membership_for(pdbid: str, lists: dict[str, set[str]]) -> str:
    variants = {pdbid, f"{pdbid}_CA_A2"}
    if pdbid.endswith("_CA_A2"):
        variants.add(pdbid.replace("_CA_A2", ""))
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
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def top_quantile_mask(values: np.ndarray, q: float = 0.80) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    return values >= np.quantile(values, q)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = np.sum(a | b)
    if denom == 0:
        return float("nan")
    return float(np.sum(a & b) / denom)


def load_annotation_data() -> tuple[list[ProteinData], list[str]]:
    lists = load_lists()
    proteins: list[ProteinData] = []
    feature_names: list[str] | None = None

    for onehot_path in sorted(ANNOTATION_DIR.glob("*-onehot.csv")):
        pdbid = onehot_path.stem.replace("-onehot", "")
        label_path = ANNOTATION_DIR / f"{pdbid}.csv"
        if not label_path.exists():
            continue
        labels = pd.read_csv(label_path)
        x = pd.read_csv(onehot_path, header=None).to_numpy(dtype=float)
        y_raw = labels["B-factor"].to_numpy(dtype=float)
        if x.shape[0] != y_raw.shape[0]:
            print(f"[skip] {pdbid}: X rows {x.shape[0]} != B-factor rows {y_raw.shape[0]}")
            continue
        if np.nanstd(y_raw) == 0:
            continue
        y = (y_raw - np.nanmean(y_raw)) / np.nanstd(y_raw)
        proteins.append(
            ProteinData(
                pdbid=pdbid,
                dataset_membership=membership_for(pdbid, lists),
                x=x,
                y_raw=y_raw,
                y=y,
                residue_index=np.arange(x.shape[0], dtype=int),
            )
        )
        if feature_names is None:
            feature_names = [f"annotation_feature_{i}" for i in range(x.shape[1])]

    if not proteins:
        raise FileNotFoundError(f"No annotation feature matrices found in {ANNOTATION_DIR}")
    return proteins, feature_names or []


def resolve_psl_dir(psl_dir_arg: str | None = None) -> Path:
    if psl_dir_arg:
        psl_dir = Path(psl_dir_arg)
        if not psl_dir.is_absolute():
            psl_dir = ROOT / psl_dir
        return psl_dir
    psl_dir = next((path for path in PSL_DIR_CANDIDATES if path.exists() and any(path.glob("*_psl_features.npy"))), None)
    if psl_dir is None:
        raise FileNotFoundError(
            "No PSL feature matrices found. Generate PSL features first or pass --psl-dir."
        )
    return psl_dir


def load_feature_names(psl_dir: Path, n_features: int) -> list[str]:
    feature_names_path = psl_dir / "feature_names.json"
    if feature_names_path.exists():
        names = json.loads(feature_names_path.read_text(encoding="utf-8"))
        if isinstance(names, list) and len(names) == n_features:
            return [str(name) for name in names]
    default_names = [f"r{r}_{stat}" for r in [6, 9, 12] for stat in ["max", "min", "mean", "median", "num_zero"]]
    if len(default_names) == n_features:
        return default_names
    return [f"psl_feature_{i}" for i in range(n_features)]


def load_psl_data(psl_dir_arg: str | None = None) -> tuple[list[ProteinData], list[str]]:
    lists = load_lists()
    proteins: list[ProteinData] = []
    psl_dir = resolve_psl_dir(psl_dir_arg)
    feature_names: list[str] | None = None
    for x_path in sorted(psl_dir.glob("*_psl_features.npy")):
        pdbid = x_path.name.replace("_psl_features.npy", "")
        y_path = psl_dir / f"{pdbid}_bfactors.npy"
        if not y_path.exists():
            continue
        x = np.load(x_path)
        if feature_names is None:
            feature_names = load_feature_names(psl_dir, x.shape[1])
        y_raw = np.load(y_path)
        if x.shape[0] != y_raw.shape[0]:
            print(f"[skip] {pdbid}: X rows {x.shape[0]} != B-factor rows {y_raw.shape[0]}")
            continue
        if np.nanstd(y_raw) == 0:
            continue
        y = (y_raw - np.nanmean(y_raw)) / np.nanstd(y_raw)
        proteins.append(
            ProteinData(
                pdbid=pdbid,
                dataset_membership=membership_for(pdbid, lists),
                x=x,
                y_raw=y_raw,
                y=y,
                residue_index=np.arange(x.shape[0], dtype=int),
            )
        )
    if not proteins:
        raise FileNotFoundError(f"No PSL feature matrices found in {psl_dir}")
    return proteins, feature_names or []


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


def model_factory(args: argparse.Namespace) -> RandomForestRegressor | ExtraTreesRegressor:
    common = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=RANDOM_STATE,
        n_jobs=args.n_jobs,
    )
    if args.model == "random_forest":
        return RandomForestRegressor(bootstrap=True, **common)
    if args.model == "extra_trees":
        return ExtraTreesRegressor(bootstrap=False, **common)
    raise ValueError(f"Unsupported model: {args.model}")


def bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(RANDOM_STATE)
    reps = []
    for _ in range(BOOTSTRAPS):
        sample = rng.choice(values, size=values.size, replace=True)
        reps.append(float(np.mean(sample)))
    return float(np.mean(values)), float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def evaluate_grouped(
    proteins: list[ProteinData],
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)
    protein_indices = np.arange(len(proteins))

    prediction_rows = []
    feature_importance_sum = np.zeros(len(feature_names), dtype=float)
    feature_importance_n = 0

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(protein_indices), start=1):
        train_proteins = [proteins[i] for i in train_idx]
        test_proteins = [proteins[i] for i in test_idx]
        x_train = np.vstack([p.x for p in train_proteins])
        y_train = np.concatenate([p.y for p in train_proteins])
        x_test = np.vstack([p.x for p in test_proteins])

        print(
            f"[fold {fold_idx}/{args.folds}] train proteins={len(train_proteins)} "
            f"test proteins={len(test_proteins)} train residues={x_train.shape[0]} test residues={x_test.shape[0]}"
        )
        model = model_factory(args)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        if args.shap:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_test, check_additivity=False)
            shap_values = np.asarray(shap_values, dtype=float)
            importance_abs = np.abs(shap_values).sum(axis=1)
            shap_signed_sum = shap_values.sum(axis=1)
            feature_importance_sum += np.abs(shap_values).sum(axis=0)
            feature_importance_n += shap_values.shape[0]
        else:
            importance_abs = np.full(x_test.shape[0], np.nan)
            shap_signed_sum = np.full(x_test.shape[0], np.nan)

        offset = 0
        for p in test_proteins:
            n = p.x.shape[0]
            rows = pd.DataFrame(
                {
                    "fold": fold_idx,
                    "pdbid": p.pdbid,
                    "dataset_membership": p.dataset_membership,
                    "residue_index": p.residue_index,
                    "bfactor_raw": p.y_raw,
                    "bfactor_z": p.y,
                    "prediction_z": y_pred[offset : offset + n],
                    "shap_importance_abs": importance_abs[offset : offset + n],
                    "shap_signed_sum": shap_signed_sum[offset : offset + n],
                }
            )
            prediction_rows.append(rows)
            offset += n

    predictions = pd.concat(prediction_rows, ignore_index=True)
    per_protein = make_per_protein_metrics(predictions)
    dataset_metrics = summarize_metrics(per_protein, predictions)

    if args.shap and feature_importance_n > 0:
        importance = feature_importance_sum / feature_importance_n
    else:
        importance = np.full(len(feature_names), np.nan)
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance,
        }
    ).sort_values("mean_abs_shap", ascending=False, na_position="last")

    return dataset_metrics, per_protein, predictions, feature_importance


def make_per_protein_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pdbid, g in predictions.groupby("pdbid", sort=True):
        y = g["bfactor_z"].to_numpy(dtype=float)
        pred = g["prediction_z"].to_numpy(dtype=float)
        imp = g["shap_importance_abs"].to_numpy(dtype=float)
        signed = g["shap_signed_sum"].to_numpy(dtype=float)
        has_shap = np.isfinite(imp).any()
        rows.append(
            {
                "pdbid": pdbid,
                "dataset_membership": g["dataset_membership"].iloc[0],
                "n_residues": int(g.shape[0]),
                "pcc_model": safe_pearson(y, pred),
                "spearman_model": safe_spearman(y, pred),
                "rmse_model_z": rmse(y, pred),
                "pcc_expl_abs": safe_pearson(y, imp) if has_shap else float("nan"),
                "spearman_expl_abs": safe_spearman(y, imp) if has_shap else float("nan"),
                "pcc_expl_signed": safe_pearson(y, signed) if has_shap else float("nan"),
                "spearman_expl_signed": safe_spearman(y, signed) if has_shap else float("nan"),
                "jaccard_top20_abs": jaccard(top_quantile_mask(y), top_quantile_mask(imp)) if has_shap else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_one(label: str, per_protein: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float | str | int]:
    if label == "all":
        pp = per_protein
        pred = predictions
    else:
        pp = per_protein[per_protein["dataset_membership"] == label]
        pred = predictions[predictions["dataset_membership"] == label]
    row: dict[str, float | str | int] = {
        "dataset": label,
        "n_proteins": int(pp.shape[0]),
        "n_residues": int(pred.shape[0]),
        "pooled_pcc_model": safe_pearson(pred["bfactor_z"].to_numpy(), pred["prediction_z"].to_numpy())
        if pred.shape[0]
        else float("nan"),
        "pooled_spearman_model": safe_spearman(pred["bfactor_z"].to_numpy(), pred["prediction_z"].to_numpy())
        if pred.shape[0]
        else float("nan"),
        "pooled_rmse_model_z": rmse(pred["bfactor_z"].to_numpy(), pred["prediction_z"].to_numpy())
        if pred.shape[0]
        else float("nan"),
    }
    for metric in [
        "pcc_model",
        "spearman_model",
        "rmse_model_z",
        "pcc_expl_abs",
        "spearman_expl_abs",
        "pcc_expl_signed",
        "spearman_expl_signed",
        "jaccard_top20_abs",
    ]:
        mean, lo, hi = bootstrap_ci(pp[metric].to_numpy(dtype=float))
        row[f"mean_{metric}"] = mean
        row[f"{metric}_ci95_low"] = lo
        row[f"{metric}_ci95_high"] = hi
    return row


def summarize_metrics(per_protein: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    labels = ["small", "medium", "large", "365_only", "all"]
    rows = [summarize_one(label, per_protein, predictions) for label in labels]
    return pd.DataFrame(rows)


def fmt(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.3f}"


def write_summary_markdown(
    feature_source: str,
    run_name: str,
    args: argparse.Namespace,
    dataset_metrics: pd.DataFrame,
    per_protein: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    path = OUT_DIR / f"rf_{run_name}_summary.md"
    lines = [
        f"# Tree-Ensemble Results ({feature_source})",
        "",
        "Protocol: cross-validation by protein.",
        f"Model: {args.model}(n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
        f"min_samples_leaf={args.min_samples_leaf}, max_features={args.max_features}).",
        "Target: within-protein z-scored C-alpha B-factor.",
        "Explanation metrics: SHAP values are computed on out-of-fold predictions.",
        "",
        "## Dataset Metrics",
        "",
        "| Dataset | Proteins | Residues | Mean PCC | 95% CI | Pooled PCC | Mean RMSE(z) | Mean SHAP PCC(abs) | Mean SHAP PCC(signed) | Jaccard top20(abs) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in dataset_metrics.iterrows():
        ci = f"[{fmt(row['pcc_model_ci95_low'])}, {fmt(row['pcc_model_ci95_high'])}]"
        lines.append(
            f"| {row['dataset']} | {int(row['n_proteins'])} | {int(row['n_residues'])} | "
            f"{fmt(row['mean_pcc_model'])} | {ci} | {fmt(row['pooled_pcc_model'])} | "
            f"{fmt(row['mean_rmse_model_z'])} | {fmt(row['mean_pcc_expl_abs'])} | "
            f"{fmt(row['mean_pcc_expl_signed'])} | {fmt(row['mean_jaccard_top20_abs'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Feature Importances",
            "",
            "| Rank | Feature | Mean absolute SHAP |",
            "|---:|---|---:|",
        ]
    )
    for rank, (_, row) in enumerate(feature_importance.head(20).iterrows(), start=1):
        lines.append(f"| {rank} | {row['feature']} | {fmt(row['mean_abs_shap'])} |")
    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "These are out-of-fold values. If the feature source is `annotation`, the numbers provide a biophysical-feature benchmark rather than a PSL-only result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["random_forest", "extra_trees"], default="random_forest")
    parser.add_argument("--feature-source", choices=["annotation", "psl"], default="annotation")
    parser.add_argument("--psl-dir", default=None, help="Directory containing *_psl_features.npy and *_bfactors.npy files")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=parse_max_features, default="sqrt")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.shap = not args.no_shap
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.feature_source == "annotation":
        proteins, feature_names = load_annotation_data()
    else:
        proteins, feature_names = load_psl_data(args.psl_dir)

    print(f"Loaded {len(proteins)} proteins with {sum(p.x.shape[0] for p in proteins)} residues.")
    print(f"Feature source: {args.feature_source}; feature count: {len(feature_names)}")

    dataset_metrics, per_protein, predictions, feature_importance = evaluate_grouped(proteins, feature_names, args)

    run_name = args.run_name or args.feature_source
    prefix = OUT_DIR / f"rf_{run_name}"
    dataset_metrics.to_csv(f"{prefix}_dataset_metrics.csv", index=False)
    per_protein.to_csv(f"{prefix}_per_protein_metrics.csv", index=False)
    predictions.to_csv(f"{prefix}_predictions.csv", index=False)
    feature_importance.to_csv(f"{prefix}_feature_importance.csv", index=False)
    (OUT_DIR / f"rf_{run_name}_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    write_summary_markdown(args.feature_source, run_name, args, dataset_metrics, per_protein, predictions, feature_importance)

    print(dataset_metrics[["dataset", "n_proteins", "n_residues", "mean_pcc_model", "pooled_pcc_model", "mean_rmse_model_z", "mean_pcc_expl_abs", "mean_jaccard_top20_abs"]].to_string(index=False))
    print(f"Wrote tree-ensemble outputs with prefix: {prefix}")


if __name__ == "__main__":
    main()
