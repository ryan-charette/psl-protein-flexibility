"""Bundled synthetic-data demo."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from psl_flexibility.features import (
    FeatureConfig,
    feature_names,
    features_for_residues,
    write_config,
    write_feature_csv,
)
from psl_flexibility.metrics import rmse, safe_pearson, safe_spearman, within_group_zscore
from psl_flexibility.structure import ResidueRecord, bfactors, parse_ca_pdb, pdb_files


ROOT = Path(__file__).resolve().parents[2]
TOY_PDB_DIR = ROOT / "examples" / "toy_proteins"


def run_toy_demo(out_dir: Path) -> dict[str, float | int | str]:
    """Run a small end-to-end feature-generation and regression demo."""

    config = FeatureConfig(radii=(4.0, 6.0, 8.0), sheaf="center_labeled", stats="both")
    names = feature_names(config)
    records_and_features = []
    for pdb_path in pdb_files(TOY_PDB_DIR):
        records = parse_ca_pdb(pdb_path)
        features = features_for_residues(records, config)
        records_and_features.append((pdb_path.stem, records, features))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_count = write_feature_csv(out_dir / "toy_psl_features.csv", records_and_features, names)
    (out_dir / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    write_config(out_dir / "feature_config.json", config)

    metrics = _leave_one_protein_out(records_and_features)
    _write_metrics(out_dir / "toy_demo_metrics.csv", metrics)

    pearsons = np.asarray([row["pearson"] for row in metrics], dtype=float)
    spearmans = np.asarray([row["spearman"] for row in metrics], dtype=float)
    rmses = np.asarray([row["rmse"] for row in metrics], dtype=float)
    summary: dict[str, float | int | str] = {
        "pdb_dir": str(TOY_PDB_DIR),
        "out_dir": str(out_dir),
        "n_proteins": len(records_and_features),
        "n_residues": row_count,
        "n_features": len(names),
        "mean_leave_one_protein_pearson": float(np.nanmean(pearsons)),
        "mean_leave_one_protein_spearman": float(np.nanmean(spearmans)),
        "mean_leave_one_protein_rmse": float(np.nanmean(rmses)),
    }
    (out_dir / "toy_demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _leave_one_protein_out(
    records_and_features: list[tuple[str, list[ResidueRecord], np.ndarray]],
) -> list[dict[str, float | str | int]]:
    typed_records: list[tuple[str, np.ndarray, np.ndarray]] = []
    for protein_id, records, features in records_and_features:
        target = within_group_zscore(bfactors(records))
        typed_records.append((protein_id, features, target))

    rows: list[dict[str, float | str | int]] = []
    for held_out, (protein_id, x_test, y_test) in enumerate(typed_records):
        x_train = np.vstack(
            [features for index, (_pid, features, _target) in enumerate(typed_records) if index != held_out]
        )
        y_train = np.concatenate(
            [target for index, (_pid, _features, target) in enumerate(typed_records) if index != held_out]
        )
        beta = _fit_ridge(x_train, y_train, alpha=1e-3)
        pred = _predict_ridge(x_test, beta)
        rows.append(
            {
                "protein_id": protein_id,
                "n_residues": int(y_test.size),
                "pearson": safe_pearson(y_test, pred),
                "spearman": safe_spearman(y_test, pred),
                "rmse": rmse(y_test, pred),
            }
        )
    return rows


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    x = np.column_stack([np.ones(features.shape[0], dtype=float), features])
    penalty = np.eye(x.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ target)


def _predict_ridge(features: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(features.shape[0], dtype=float), features])
    return x @ beta


def _write_metrics(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "n_residues", "pearson", "spearman", "rmse"])
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["TOY_PDB_DIR", "run_toy_demo"]
