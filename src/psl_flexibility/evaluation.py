"""Fixed-model, paired evaluation of cached geometric feature blocks."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import sklearn
import scipy
from sklearn.ensemble import RandomForestRegressor

from psl_flexibility.dataset import SEED, sha256_file


CORE_MODES = ["graph", "native0", "graph+native0", "center0", "graph+identity0",
              "graph+geometric0", "graph+center0"]
D1_MODES = [mode for label in ["identity", "geometric", "center"] for mode in
            [f"graph+{label}0", f"graph+{label}0+{label}1", f"graph+{label}0+{label}_p1"]]


@dataclass
class StudyProtein:
    protein_id: str
    cluster_id: str
    protein_fold: int
    cluster_fold: int
    residues: pd.DataFrame
    features: dict[str, np.ndarray]
    names: dict[str, list[str]]
    feature_path: str
    feature_sha256: str


def prediction_metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    y, predicted = np.asarray(y, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != predicted.shape or not np.isfinite(y).all() or not np.isfinite(predicted).all():
        raise ValueError("Targets and predictions must be finite arrays with matching shapes")
    variable = y.size >= 3 and np.std(y) > 1e-12 and np.std(predicted) > 1e-12
    return {
        "pcc": float(pearsonr(y, predicted).statistic) if variable else float("nan"),
        "spearman": float(spearmanr(y, predicted).statistic) if variable else float("nan"),
        "rmse_z": float(np.sqrt(np.mean((y - predicted) ** 2))),
    }


def cluster_bootstrap(
    values: np.ndarray, clusters: np.ndarray, *, repetitions: int = 2000, seed: int = SEED,
) -> tuple[float, float, float]:
    """Macro-protein mean and percentile CI from resampling whole clusters.

    Each sampled cluster contributes all its proteins; cluster-size variation is
    preserved. The interval is conditional on the already fitted models/folds.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    if values.shape != clusters.shape:
        raise ValueError("Values and cluster labels must have matching shapes")
    valid = np.isfinite(values)
    values, clusters = values[valid], clusters[valid]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    unique = np.unique(clusters)
    sums = np.array([values[clusters == cluster].sum() for cluster in unique])
    counts = np.array([(clusters == cluster).sum() for cluster in unique])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    means = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _cohort_ids(table: pd.DataFrame, cohort: str) -> set[str]:
    if cohort == "all":
        return set(table["protein_id"])
    if cohort in {"subset60", "subset20", "nested60", "nested20"}:
        mask = table[cohort].astype(str).str.lower().isin(["true", "1"])
        return set(table.loc[mask, "protein_id"])
    path = Path(cohort)
    if not path.exists():
        raise ValueError(f"Unknown cohort {cohort}; use all, subset60, subset20, or an ID file")
    if path.suffix.lower() == ".csv":
        selected = set(pd.read_csv(path, dtype={"protein_id": str})["protein_id"])
    else:
        selected = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    unknown = selected.difference(table["protein_id"])
    if unknown:
        raise ValueError(f"Cohort contains IDs absent from the dataset: {sorted(unknown)}")
    return selected


def load_study(
    manifest_dir: Path,
    feature_dir: Path,
    modes: list[str],
    *,
    cohort: str = "all",
    feature_columns: str | None = None,
    exclude_feature_columns: str | None = None,
    missing: str = "error",
) -> tuple[list[StudyProtein], pd.DataFrame]:
    """Load one common verified cohort for every requested feature combination.

    NPZ files contain residue_keys and numeric blocks with <block>_names. A
    missing block can only exclude the protein from *all* requested modes.
    Selection regexes match qualified column names, e.g. ``graph:degree6``.
    """
    manifest_dir, feature_dir = Path(manifest_dir), Path(feature_dir)
    proteins = pd.read_csv(manifest_dir / "proteins.csv", dtype={"protein_id": str})
    folds = pd.read_csv(manifest_dir / "folds.csv", dtype={"protein_id": str, "cluster_id": str})
    residues = pd.read_csv(manifest_dir / "residues.csv", keep_default_na=False,
                           dtype={"protein_id": str, "residue_key": str, "chain_id": str, "insertion_code": str})
    if proteins["protein_id"].duplicated().any() or folds["protein_id"].duplicated().any():
        raise ValueError("Duplicate protein IDs in manifests")
    selected = _cohort_ids(proteins, cohort)
    metadata = proteins.merge(folds, on="protein_id", validate="one_to_one")
    if set(proteins["protein_id"]) != set(folds["protein_id"]):
        raise ValueError("Fold manifest must contain exactly the dataset protein IDs")
    regex = re.compile(feature_columns) if feature_columns else None
    excluded_regex = re.compile(exclude_feature_columns) if exclude_feature_columns else None
    blocks = sorted({block for mode in modes for block in mode.split("+")})
    loaded, exclusions = [], []
    reference_names = {}
    for row in metadata.sort_values("protein_id").to_dict("records"):
        protein = row["protein_id"]
        if protein not in selected:
            continue
        path = feature_dir / f"{protein}.npz"
        frame = residues.loc[residues["protein_id"] == protein].sort_values("row_index").reset_index(drop=True)
        if len(frame) != row["n_residues"] or frame["residue_key"].duplicated().any():
            raise ValueError(f"{protein}: invalid residue manifest")
        try:
            if not path.exists():
                raise FileNotFoundError(f"Missing feature file {path}")
            with np.load(path, allow_pickle=False) as cache:
                if "residue_keys" not in cache:
                    raise ValueError(f"{protein}: feature file lacks residue identity keys")
                if not np.array_equal(np.asarray(cache["residue_keys"], dtype=str), frame["residue_key"].to_numpy(dtype=str)):
                    raise ValueError(f"{protein}: feature/residue keys do not match exactly in order")
                features, names = {}, {}
                for block in blocks:
                    if block not in cache or f"{block}_names" not in cache:
                        raise KeyError(f"{protein}: missing feature block or column names for {block}")
                    matrix = np.asarray(cache[block], dtype=float)
                    columns = [f"{block}:{name}" for name in np.asarray(cache[f"{block}_names"], dtype=str).tolist()]
                    if matrix.ndim != 2 or matrix.shape != (len(frame), len(columns)):
                        raise ValueError(f"{protein}: shape mismatch for {block}")
                    if len(set(columns)) != len(columns) or not np.isfinite(matrix).all():
                        raise ValueError(f"{protein}: duplicate names or nonfinite features for {block}")
                    features[block], names[block] = matrix, columns
                combined_features, combined_names = {}, {}
                for mode in modes:
                    cols = [name for block in mode.split("+") for name in names[block]]
                    matrix = np.column_stack([features[block] for block in mode.split("+")])
                    mask = np.array([(regex is None or regex.search(name) is not None) and
                                     (excluded_regex is None or excluded_regex.search(name) is None) for name in cols])
                    if not mask.any():
                        raise ValueError(f"{mode}: feature selection removed every column")
                    combined_names[mode] = [name for name, keep in zip(cols, mask, strict=True) if keep]
                    combined_features[mode] = matrix[:, mask]
                    if mode in reference_names and reference_names[mode] != combined_names[mode]:
                        raise ValueError(f"{protein}: columns differ across proteins for {mode}")
        except (FileNotFoundError, KeyError) as exc:
            if missing != "exclude":
                raise
            exclusions.append({"protein_id": protein, "reason": str(exc)})
            continue
        for mode in modes:
            reference_names[mode] = combined_names[mode]
        loaded.append(StudyProtein(
            protein, row["cluster_id"], int(row["protein_fold"]), int(row["cluster_fold"]),
            frame, combined_features, combined_names, str(path.resolve()), sha256_file(path),
        ))
    if not loaded:
        raise ValueError("No proteins have every requested feature block")
    return loaded, pd.DataFrame(exclusions, columns=["protein_id", "reason"])


def validate_partitions(proteins: list[StudyProtein], protocol: str, n_folds: int = 5) -> None:
    attr = {"protein": "protein_fold", "cluster": "cluster_fold"}.get(protocol)
    if attr is None:
        raise ValueError(f"Unknown protocol {protocol}")
    observed = {getattr(p, attr) for p in proteins}
    if observed != set(range(1, n_folds + 1)):
        raise ValueError(f"{protocol}: cohort does not populate all {n_folds} frozen folds: {sorted(observed)}")
    if protocol == "cluster":
        cluster_folds: dict[str, set[int]] = {}
        for p in proteins:
            cluster_folds.setdefault(p.cluster_id, set()).add(p.cluster_fold)
        if any(len(value) != 1 for value in cluster_folds.values()):
            raise ValueError("Sequence cluster crosses train/test folds")


def evaluate_mode(
    proteins: list[StudyProtein], mode: str, protocol: str, *,
    n_estimators: int = 200, n_jobs: int = 1, seed: int = SEED, n_folds: int = 5,
    model_dir: Path | None = None, save_case_ids: tuple[str, ...] = ("1ULR", "1X3O"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_partitions(proteins, protocol, n_folds)
    fold_attr = "protein_fold" if protocol == "protein" else "cluster_fold"
    prediction_frames, fold_rows = [], []
    for fold in range(1, n_folds + 1):
        train = [p for p in proteins if getattr(p, fold_attr) != fold]
        test = [p for p in proteins if getattr(p, fold_attr) == fold]
        started = time.monotonic()
        print(f"{protocol} {mode} fold {fold}/{n_folds}: train {len(train)}, test {len(test)} proteins", flush=True)
        x_train = np.vstack([p.features[mode] for p in train])
        y_train = np.concatenate([p.residues["z_b_factor"].to_numpy(dtype=float) for p in train])
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=12, min_samples_leaf=2,
            max_features="sqrt", bootstrap=True, random_state=seed, n_jobs=n_jobs,
        )
        model.fit(x_train, y_train)
        for p in test:
            predicted = model.predict(p.features[mode])
            rows = p.residues.copy()
            rows["protocol"], rows["mode"], rows["fold"] = protocol, mode, fold
            rows["cluster_id"] = p.cluster_id
            rows["prediction_z"] = predicted
            prediction_frames.append(rows)
        if model_dir is not None and any(p.protein_id in save_case_ids for p in test):
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": model, "mode": mode, "protocol": protocol, "fold": fold,
                "feature_names": proteins[0].names[mode], "train_ids": [p.protein_id for p in train],
                "test_ids": [p.protein_id for p in test], "seed": seed,
            }, model_dir / f"{protocol}_{mode.replace('+', '_plus_')}_fold{fold}.joblib", compress=3)
        fold_rows.append({"protocol": protocol, "mode": mode, "fold": fold,
                          "n_train_proteins": len(train), "n_test_proteins": len(test),
                          "n_train_residues": len(y_train), "n_test_residues": sum(len(p.residues) for p in test),
                          "fit_predict_seconds": time.monotonic() - started})
    predictions = pd.concat(prediction_frames, ignore_index=True)
    per_rows = []
    for protein, frame in predictions.groupby("protein_id", sort=True):
        per_rows.append({"protocol": protocol, "mode": mode, "protein_id": protein,
                         "cluster_id": frame["cluster_id"].iloc[0], "fold": int(frame["fold"].iloc[0]),
                         "n_residues": len(frame),
                         **prediction_metrics(frame["z_b_factor"].to_numpy(), frame["prediction_z"].to_numpy())})
    return predictions, pd.DataFrame(per_rows), pd.DataFrame(fold_rows)


def summarize_results(per_protein: pd.DataFrame, predictions: pd.DataFrame, *, repetitions: int = 2000,
                      seed: int = SEED) -> pd.DataFrame:
    summaries = []
    for (protocol, mode), frame in per_protein.groupby(["protocol", "mode"], sort=False):
        pooled = predictions[(predictions["protocol"] == protocol) & (predictions["mode"] == mode)]
        row = {"protocol": protocol, "mode": mode, "n_proteins": len(frame),
               "n_clusters": frame["cluster_id"].nunique(), "n_residues": int(frame["n_residues"].sum())}
        for metric in ["pcc", "spearman", "rmse_z"]:
            mean, low, high = cluster_bootstrap(frame[metric].to_numpy(), frame["cluster_id"].to_numpy(),
                                               repetitions=repetitions, seed=seed)
            row.update({f"mean_{metric}": mean, f"{metric}_ci95_low": low, f"{metric}_ci95_high": high,
                        f"n_valid_{metric}": int(np.isfinite(frame[metric]).sum())})
        row.update({f"pooled_{key}": value for key, value in prediction_metrics(
            pooled["z_b_factor"].to_numpy(), pooled["prediction_z"].to_numpy()).items()})
        summaries.append(row)
    return pd.DataFrame(summaries)


def paired_comparisons(per_protein: pd.DataFrame, *, repetitions: int = 2000, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for protocol, frame in per_protein.groupby("protocol", sort=False):
        for left, right in itertools.combinations(frame["mode"].unique(), 2):
            paired = frame[frame["mode"] == left].merge(
                frame[frame["mode"] == right], on="protein_id", suffixes=("_left", "_right"), validate="one_to_one")
            if len(paired) != (frame["mode"] == left).sum() or len(paired) != (frame["mode"] == right).sum():
                raise ValueError("Paired comparison requires identical protein cohorts")
            if not (paired["cluster_id_left"] == paired["cluster_id_right"]).all():
                raise ValueError("Paired comparison has inconsistent cluster identities")
            for metric in ["pcc", "spearman", "rmse_z"]:
                difference = paired[f"{metric}_right"].to_numpy() - paired[f"{metric}_left"].to_numpy()
                mean, low, high = cluster_bootstrap(difference, paired["cluster_id_left"].to_numpy(),
                                                   repetitions=repetitions, seed=seed)
                rows.append({"protocol": protocol, "left_mode": left, "right_mode": right,
                             "difference": "right_minus_left", "metric": metric,
                             "n_pairs": int(np.isfinite(difference).sum()),
                             "mean_difference": mean, "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


def run_evaluation(
    manifest_dir: Path, feature_dir: Path, out_dir: Path, *, modes: list[str] | None = None,
    protocols: tuple[str, ...] = ("protein", "cluster"), cohort: str = "all", phase: str = "core",
    feature_columns: str | None = None, exclude_feature_columns: str | None = None,
    missing: str = "error", n_estimators: int = 200, n_jobs: int = 1, seed: int = SEED,
    repetitions: int = 2000, save_model_modes: tuple[str, ...] = ("graph+center0",),
) -> dict:
    modes = modes or (D1_MODES if phase == "d1" else CORE_MODES)
    if len(set(modes)) != len(modes) or any(not mode or len(set(mode.split("+"))) != len(mode.split("+")) for mode in modes):
        raise ValueError("Modes and their constituent blocks must be nonempty and unique")
    manifest_dir, feature_dir, out_dir = Path(manifest_dir), Path(feature_dir), Path(out_dir)
    dataset_config = json.loads((manifest_dir / "dataset_config.json").read_text())
    proteins, exclusions = load_study(manifest_dir, feature_dir, modes, cohort=cohort,
                                      feature_columns=feature_columns, exclude_feature_columns=exclude_feature_columns,
                                      missing=missing)
    for protocol in protocols:
        validate_partitions(proteins, protocol, dataset_config["n_folds"])
    out_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "complete": False,
        "phase": phase, "cohort": cohort, "modes": modes, "protocols": protocols,
        "seed": seed, "n_folds": dataset_config["n_folds"], "n_estimators": n_estimators,
        "max_depth": 12, "min_samples_leaf": 2, "max_features": "sqrt", "bootstrap": True,
        "n_jobs": n_jobs, "sklearn_version": sklearn.__version__, "bootstrap_repetitions": repetitions,
        "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "code_sha256": {"evaluation.py": sha256_file(Path(__file__)),
                        "dataset.py": sha256_file(Path(__file__).with_name("dataset.py"))},
        "uncertainty": "percentile cluster bootstrap of whole proteins; conditional on fixed fitted models and folds",
        "feature_columns": feature_columns, "exclude_feature_columns": exclude_feature_columns,
        "missing_policy": missing, "target": dataset_config["target"],
        "manifest_sha256": {name: sha256_file(manifest_dir / name) for name in
                            ["proteins.csv", "residues.csv", "folds.csv", "dataset_config.json"]},
        "feature_files": {p.protein_id: {"path": p.feature_path, "sha256": p.feature_sha256} for p in proteins},
        "n_proteins": len(proteins), "n_residues": sum(len(p.residues) for p in proteins),
    }
    (out_dir / "evaluation_config.json").write_text(json.dumps(configuration, indent=2, sort_keys=True) + "\n")
    exclusions.to_csv(out_dir / "exclusions.csv", index=False)
    pd.DataFrame([{"protein_id": p.protein_id, "cluster_id": p.cluster_id,
                   "protein_fold": p.protein_fold, "cluster_fold": p.cluster_fold,
                   "n_residues": len(p.residues)} for p in proteins]).to_csv(out_dir / "evaluation_fold_manifest.csv", index=False)
    (out_dir / "feature_names.json").write_text(json.dumps(proteins[0].names, indent=2, sort_keys=True) + "\n")
    predictions, per_protein, timings = [], [], []
    for protocol in protocols:
        for mode in modes:
            saved_models = out_dir / "models" if mode in save_model_modes else None
            pred, per, timing = evaluate_mode(proteins, mode, protocol, n_estimators=n_estimators,
                                              n_jobs=n_jobs, seed=seed, n_folds=dataset_config["n_folds"],
                                              model_dir=saved_models)
            predictions.append(pred)
            per_protein.append(per)
            timings.append(timing)
            # Checkpoint each completed mode without marking an incomplete run complete.
            pd.concat(per_protein, ignore_index=True).to_csv(out_dir / "per_protein.csv", index=False)
            pd.concat(timings, ignore_index=True).to_csv(out_dir / "fold_timings.csv", index=False)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_per = pd.concat(per_protein, ignore_index=True)
    all_predictions.to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    summary = summarize_results(all_per, all_predictions, repetitions=repetitions, seed=seed)
    paired = paired_comparisons(all_per, repetitions=repetitions, seed=seed)
    summary.to_csv(out_dir / "summary.csv", index=False)
    paired.to_csv(out_dir / "paired_differences.csv", index=False)
    configuration["complete"] = True
    configuration["outputs_sha256"] = {name: sha256_file(out_dir / name) for name in
                                        ["summary.csv", "per_protein.csv", "predictions.csv.gz", "paired_differences.csv"]}
    (out_dir / "evaluation_config.json").write_text(json.dumps(configuration, indent=2, sort_keys=True) + "\n")
    print(summary[["protocol", "mode", "n_proteins", "mean_pcc", "pcc_ci95_low", "pcc_ci95_high"]].to_string(index=False))
    return configuration
