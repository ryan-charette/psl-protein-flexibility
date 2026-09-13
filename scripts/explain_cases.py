"""Explain the two prespecified proteins using their exact held-out forest models.

No model is trained here. Full interventional Tree SHAP uses exactly 100 seeded
training-residue background rows. Attribution groups describe the features of
the predicted residue, not contributions of its spatial neighbors or causes of
physical motion. Models, input files, background keys, feature order, baseline,
and numerical additivity checks are recorded with every case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "processed" / "mpl-cache"))
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from threadpoolctl import threadpool_limits

from psl_flexibility.dataset import sha256_file
from psl_flexibility.evaluation import StudyProtein, load_study
from render_protein import render_protein


CASES = ("1ULR", "1X3O")
MODE = "graph+center0"
PROTOCOL = "cluster"
BACKGROUND_SIZE = 100
ADDITIVITY_TOLERANCE = 1e-6


def _path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _matrix_hash(matrix: np.ndarray) -> str:
    array = np.ascontiguousarray(matrix, dtype="<f8")
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def feature_groups(names: list[str]) -> dict[str, np.ndarray]:
    """Partition all evaluated columns, retaining separate sheaf radii."""
    graph = np.array([name.startswith("graph:") for name in names])
    sheaf = np.array([name.startswith("center0:") for name in names])
    if not graph.any() or not sheaf.any() or not np.all(graph ^ sheaf):
        raise ValueError("Expected an exact graph + center0 feature partition.")
    groups = {"graph_shap": graph, "sheaf_shap": sheaf}
    assigned = np.zeros(len(names), dtype=int)
    for radius in (3, 4, 5, 6):
        mask = np.array([bool(re.search(rf"^center0:.*_a{radius}_d0_", name)) for name in names])
        if mask.sum() != 6:
            raise ValueError(f"Expected six center0 statistics at radius {radius}; found {mask.sum()}.")
        groups[f"sheaf_a{radius}_shap"] = mask
        assigned += mask
    if not np.array_equal(assigned, sheaf.astype(int)):
        raise ValueError("Radius groups do not partition the center0 features exactly.")
    return groups


def explain_model(model: Any, features: np.ndarray, background: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Compute and independently check full raw-output interventional Tree SHAP."""
    features, background = np.asarray(features, dtype=float), np.asarray(background, dtype=float)
    if features.ndim != 2 or background.shape != (BACKGROUND_SIZE, len(names)) or features.shape[1] != len(names):
        raise ValueError("Feature/background shapes disagree with names or the fixed 100-row background.")
    if not np.isfinite(features).all() or not np.isfinite(background).all():
        raise ValueError("SHAP features must be finite.")
    if int(model.n_features_in_) != len(names):
        raise ValueError("Model input width does not match the evaluated feature names.")
    groups = feature_groups(names)
    # Parallelism affects execution only, never the persisted trained trees.
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    # sklearn's forest predict/apply converts inputs to float32. Tree SHAP
    # converts explained rows but otherwise keeps its background's dtype.
    # Cast BOTH to the model's effective input precision so hybrid paths at
    # split thresholds represent the same prediction function. Source cache
    # values remain separately preserved in the explanation archive.
    effective_features = features.astype(np.float32)
    effective_background = background.astype(np.float32)
    with threadpool_limits(limits=1):
        explainer = shap.TreeExplainer(
            model, data=effective_background, feature_names=names,
            feature_perturbation="interventional", model_output="raw",
        )
        if not np.array_equal(explainer.data, effective_background):
            raise RuntimeError("TreeExplainer changed the requested background rows.")
        contributions = np.asarray(explainer.shap_values(effective_features, approximate=False, check_additivity=True), dtype=float)
        expected_array = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
        if contributions.shape != features.shape or expected_array.size != 1:
            raise ValueError("Only one-output regression explanations are supported.")
        expected = float(expected_array[0])
        predictions = np.asarray(model.predict(features), dtype=float)
        if not np.array_equal(predictions, model.predict(effective_features)):
            raise FloatingPointError("Effective float32 inputs changed the saved model's predictions.")
        residual = predictions - expected - contributions.sum(axis=1)
        maximum_error = float(np.max(np.abs(residual)))
        if maximum_error >= ADDITIVITY_TOLERANCE:
            raise FloatingPointError(f"SHAP additivity error {maximum_error:g} exceeds {ADDITIVITY_TOLERANCE:g}.")
        background_mean = float(np.mean(model.predict(background)))
        if not np.array_equal(model.predict(background), model.predict(effective_background)):
            raise FloatingPointError("Effective float32 background changed the saved model's predictions.")
        if abs(expected-background_mean) >= ADDITIVITY_TOLERANCE:
            raise FloatingPointError("Expected value does not equal the background's mean model prediction.")
    return {
        "shap_values": contributions, "expected_value": expected,
        "prediction_z": predictions, "additivity_residual": residual,
        "additivity_max_error": maximum_error, "background_prediction_mean": background_mean,
        "effective_feature_values": effective_features, "effective_background_values": effective_background,
        "groups": {name: contributions[:, mask].sum(axis=1) for name, mask in groups.items()},
    }


def prepare_background(proteins: dict[str, StudyProtein], bundle: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Uniformly sample distinct training residues, preserving recorded model order."""
    train_ids = list(bundle["train_ids"])
    if len(set(train_ids)) != len(train_ids) or set(train_ids) & set(bundle["test_ids"]):
        raise ValueError("Invalid model train/test identifiers.")
    training = [proteins[protein] for protein in train_ids]
    matrix = np.vstack([protein.features[MODE] for protein in training])
    keys = pd.concat([
        protein.residues[["protein_id", "residue_key", "row_index"]]
        for protein in training
    ], ignore_index=True)
    if len(keys) < BACKGROUND_SIZE or keys[["protein_id", "residue_key"]].duplicated().any():
        raise ValueError("Insufficient or duplicated training residue keys.")
    seed = int(bundle["seed"]) + int(bundle["fold"])
    indices = np.random.default_rng(seed).choice(len(keys), BACKGROUND_SIZE, replace=False)
    sampled = keys.iloc[indices].copy().reset_index(drop=True)
    sampled.insert(0, "background_index", np.arange(BACKGROUND_SIZE))
    sampled["training_row_index"] = indices
    return matrix[indices], sampled, seed


def prediction_check(frame: pd.DataFrame, prediction_path: Path, *, required: bool = False) -> dict[str, Any]:
    """Compare by exact residue keys with the main evaluation's saved predictions."""
    if not prediction_path.exists():
        if required:
            raise FileNotFoundError(prediction_path)
        return {"status": "pending", "reason": "Main prediction table is not yet present", "path": _path(prediction_path)}
    protein_id = str(frame["protein_id"].iloc[0])
    selected = []
    columns = ["protein_id", "residue_key", "protocol", "mode", "fold", "prediction_z"]
    for chunk in pd.read_csv(prediction_path, usecols=columns, chunksize=100000, keep_default_na=False):
        mask = (chunk["protein_id"] == protein_id) & (chunk["protocol"] == PROTOCOL) & (chunk["mode"] == MODE)
        if mask.any():
            selected.append(chunk.loc[mask])
    if not selected:
        raise ValueError(f"No main saved predictions for {protein_id}, {PROTOCOL}, {MODE}.")
    expected = pd.concat(selected, ignore_index=True)
    if expected["residue_key"].duplicated().any() or set(expected["residue_key"]) != set(frame["residue_key"]):
        raise ValueError("Case and main prediction residue keys disagree.")
    aligned = expected.set_index("residue_key").loc[frame["residue_key"]]
    if not np.array_equal(aligned["fold"].to_numpy(), frame["fold"].to_numpy()):
        raise ValueError("Case fold does not match saved predictions.")
    error = float(np.max(np.abs(aligned["prediction_z"].to_numpy()-frame["prediction_z"].to_numpy())))
    if error >= 1e-10:
        raise FloatingPointError(f"Saved/model prediction mismatch: {error:g}.")
    return {"status": "verified", "path": _path(prediction_path), "sha256": sha256_file(prediction_path), "maximum_error": error, "n_residues": len(frame)}


def explain_case(case: StudyProtein, proteins: dict[str, StudyProtein], model_path: Path, out_dir: Path,
                 prediction_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_sha = sha256_file(model_path)
    bundle = joblib.load(model_path)
    required = {"model", "mode", "protocol", "fold", "feature_names", "train_ids", "test_ids", "seed"}
    if not required.issubset(bundle) or bundle["mode"] != MODE or bundle["protocol"] != PROTOCOL:
        raise ValueError("Saved model bundle does not match the specified case protocol.")
    if int(bundle["fold"]) != case.cluster_fold or case.protein_id not in bundle["test_ids"] or case.protein_id in bundle["train_ids"]:
        raise ValueError("The case was not held out from this exact saved model.")
    if set(bundle["train_ids"]) | set(bundle["test_ids"]) != set(proteins):
        raise ValueError("Model and loaded feature cohort differ.")
    if list(bundle["feature_names"]) != case.names[MODE]:
        raise ValueError("Feature names/order differ from the saved model.")
    training_clusters = {proteins[pid].cluster_id for pid in bundle["train_ids"]}
    if case.cluster_id in training_clusters:
        raise ValueError("The case's sequence cluster appeared in training.")
    if any(proteins[pid].cluster_fold == case.cluster_fold for pid in bundle["train_ids"]):
        raise ValueError("Saved training IDs conflict with frozen cluster folds.")
    background, background_keys, seed = prepare_background(proteins, bundle)
    features, names = case.features[MODE], case.names[MODE]
    result = explain_model(bundle["model"], features, background, names)
    frame = case.residues.copy()
    frame["protocol"], frame["mode"], frame["fold"] = PROTOCOL, MODE, case.cluster_fold
    frame["cluster_id"] = case.cluster_id
    frame["prediction_z"] = result["prediction_z"]
    frame["expected_value"] = result["expected_value"]
    frame["additivity_residual"] = result["additivity_residual"]
    for name, values in result["groups"].items():
        frame[name] = values
    checked = prediction_check(frame, prediction_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{case.protein_id}_values.csv", index=False)
    shap_frame = pd.concat([
        frame[["protein_id", "residue_key", "row_index", "chain_id", "residue_number", "insertion_code"]].reset_index(drop=True),
        pd.DataFrame(result["shap_values"], columns=names),
    ], axis=1)
    shap_frame.to_csv(out_dir / f"{case.protein_id}_shap.csv", index=False)
    background_keys.to_csv(out_dir / f"{case.protein_id}_background.csv", index=False)
    np.savez_compressed(
        out_dir / f"{case.protein_id}_explanation.npz",
        feature_names=np.asarray(names), residue_keys=frame["residue_key"].to_numpy(dtype=str),
        feature_values=features, shap_values=result["shap_values"], expected_value=result["expected_value"],
        effective_feature_values=result["effective_feature_values"],
        effective_background_values=result["effective_background_values"],
        prediction_z=result["prediction_z"], background_values=background,
        background_protein_ids=background_keys["protein_id"].to_numpy(dtype=str),
        background_residue_keys=background_keys["residue_key"].to_numpy(dtype=str),
    )
    audit = {
        "protein_id": case.protein_id, "protocol": PROTOCOL, "mode": MODE, "fold": case.cluster_fold,
        "n_residues": len(frame), "n_features": len(names), "feature_names": names,
        "model_path": _path(model_path), "model_sha256": model_sha,
        "case_feature_path": _path(Path(case.feature_path)), "case_feature_file_sha256": case.feature_sha256,
        "case_input_matrix_sha256": _matrix_hash(features),
        "train_ids": list(bundle["train_ids"]), "test_ids": list(bundle["test_ids"]),
        "training_feature_file_sha256": {pid: proteins[pid].feature_sha256 for pid in bundle["train_ids"]},
        "training_seed": int(bundle["seed"]), "background_seed": seed,
        "background_size": BACKGROUND_SIZE, "background_sampling": "Uniform training residues without replacement",
        "background_matrix_sha256": _matrix_hash(background), "background_rows": background_keys.to_dict("records"),
        "effective_input_dtype": "float32",
        "effective_input_reason": "scikit-learn RandomForestRegressor converts inputs to float32; both SHAP foreground and background use that same prediction-function precision",
        "effective_feature_matrix_sha256": _matrix_hash(result["effective_feature_values"]),
        "effective_background_matrix_sha256": _matrix_hash(result["effective_background_values"]),
        "source_and_effective_predictions_identical": True,
        "expected_value": result["expected_value"], "background_prediction_mean": result["background_prediction_mean"],
        "additivity_max_error": result["additivity_max_error"], "additivity_required_below": ADDITIVITY_TOLERANCE,
        "main_prediction_check": checked,
        "shap_settings": {"feature_perturbation": "interventional", "model_output": "raw", "approximate": False, "tree_limit": None},
        "interpretation": "Each contribution describes the explained residue's input features. Group sums are model attributions, not spatial-neighbor attributions or causal effects.",
        "versions": {"shap": shap.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
        "source_sha256": sha256_file(Path(__file__)),
    }
    for suffix in ("values.csv", "shap.csv", "background.csv", "explanation.npz"):
        audit.setdefault("output_sha256", {})[suffix] = sha256_file(out_dir / f"{case.protein_id}_{suffix}")
    (out_dir / f"{case.protein_id}_provenance.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    return frame, audit


def render_cases(frames: dict[str, pd.DataFrame], audits: dict[str, dict[str, Any]], out_dir: Path,
                 render_dir: Path, pdb_dir: Path) -> None:
    # One observed/predicted scale across BOTH prespecified proteins, no clipping.
    prediction_limit = max(float(np.max(np.abs(frame[["z_b_factor", "prediction_z"]].to_numpy()))) for frame in frames.values())
    shap_limit = max(float(np.max(np.abs(frame["sheaf_shap"].to_numpy()))) for frame in frames.values())
    shap_limit = max(shap_limit, 1e-12)
    render_dir.mkdir(parents=True, exist_ok=True)
    for protein_id, frame in frames.items():
        camera = None
        rendering = {}
        for label, column, limit in (("observed", "z_b_factor", prediction_limit), ("predicted", "prediction_z", prediction_limit), ("shap", "sheaf_shap", shap_limit)):
            output = render_dir / f"{protein_id}_{label}.png"
            camera, map_audit = render_protein(pdb_dir / f"{protein_id}.pdb", frame, column, output, camera, limit)
            rendering[label] = {"path": _path(output), "sha256": sha256_file(output), "mapping": map_audit}
        (render_dir / f"{protein_id}_camera.json").write_text(json.dumps(camera, indent=2)+"\n", encoding="utf-8")
        audits[protein_id]["rendering"] = rendering
        audits[protein_id]["camera"] = camera
        (out_dir / f"{protein_id}_provenance.json").write_text(json.dumps(audits[protein_id], indent=2)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/study/dataset")
    parser.add_argument("--features", type=Path, default=ROOT / "data/features")
    parser.add_argument("--models", type=Path, default=ROOT / "results/study/core/models")
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/study/core/predictions.csv.gz")
    parser.add_argument("--out", type=Path, default=ROOT / "results/study/cases")
    parser.add_argument("--renders", type=Path, default=ROOT / "paper/figures/renders")
    parser.add_argument("--pdb-dir", type=Path, default=ROOT / "data/raw/figures")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Verify already explained cases against the completed main prediction table")
    args = parser.parse_args()
    if args.verify_only:
        for protein_id in CASES:
            frame = pd.read_csv(args.out / f"{protein_id}_values.csv", keep_default_na=False)
            audit_path = args.out / f"{protein_id}_provenance.json"
            audit = json.loads(audit_path.read_text())
            audit["main_prediction_check"] = prediction_check(frame, args.predictions, required=True)
            audit_path.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
            print(protein_id, audit["main_prediction_check"], flush=True)
        return
    loaded, exclusions = load_study(args.manifest, args.features, [MODE], missing="error")
    if len(exclusions):
        raise ValueError("Case attribution must use the exact complete evaluated cohort.")
    proteins = {protein.protein_id: protein for protein in loaded}
    frames, audits = {}, {}
    for protein_id in CASES:
        case = proteins[protein_id]
        model_path = args.models / f"cluster_graph_plus_center0_fold{case.cluster_fold}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"The exact held-out model is not ready: {model_path}")
        print(f"Explaining {protein_id}, held-out cluster fold {case.cluster_fold}, from {model_path.name}", flush=True)
        frame, audit = explain_case(case, proteins, model_path, args.out, args.predictions)
        frames[protein_id], audits[protein_id] = frame, audit
        print(f"{protein_id}: {len(frame)} residues, additivity error {audit['additivity_max_error']:.3g}", flush=True)
    if not args.skip_render:
        render_cases(frames, audits, args.out, args.renders, args.pdb_dir)


if __name__ == "__main__":
    main()
