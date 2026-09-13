#!/usr/bin/env python3
"""Read-only consistency audit of study inputs, completed runs and explanations.

This checks identities and recomputes prediction metrics; it does not refit a
model, recompute spectra, deserialize fitted models, or modify study artifacts.
JSON is written to stdout, or to an explicitly requested report path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
CORE_BLOCKS = ["graph", "native0", "identity0", "geometric0", "center0"]
STUDY_BLOCKS = CORE_BLOCKS + [kind + suffix for kind in ("identity", "geometric", "center")
                             for suffix in ("1", "_p1", "_upper1")]


def require_fixed_study_plan(study: Path) -> dict:
    """Reject a truncated study, including absent runs or omitted model modes."""
    core = {"graph", "native0", "graph+native0", "center0", "graph+identity0", "graph+geometric0", "graph+center0"}
    degree1 = {mode for kind in ("identity", "geometric", "center") for mode in
               (f"graph+{kind}0", f"graph+{kind}0+{kind}1", f"graph+{kind}0+{kind}_p1", f"graph+{kind}0+{kind}_upper1")}
    ablations = {"nullity": r"^graph:|_nullity$", "positive": r"^graph:|_(?:min_positive|max|mean|median|std)$",
                 **{f"a{a}": rf"^graph:|center0_a{a}_" for a in (3, 4, 5, 6)}}
    expected = {"core": (core, {"cluster", "protein"}, None, "core"),
                "d1": (degree1, {"cluster"}, None, "d1"),
                **{f"ablation_{name}": ({"graph+center0"}, {"cluster"}, regex, "ablation") for name, regex in ablations.items()}}
    for name, (modes, protocols, regex, phase) in expected.items():
        path = study / name / "evaluation_config.json"
        require(path.exists(), f"Missing required fixed-study evaluation: {name}")
        config = json.loads(path.read_text())
        require(config.get("complete") is True, f"Required fixed-study evaluation is incomplete: {name}")
        require(set(config["modes"]) == modes and len(config["modes"]) == len(modes), f"{name}: required study modes differ")
        require(set(config["protocols"]) == protocols and len(config["protocols"]) == len(protocols), f"{name}: required protocols differ")
        require(config["feature_columns"] == regex and config["exclude_feature_columns"] is None, f"{name}: feature selection differs from fixed plan")
        require(config["phase"] == phase and config["cohort"] == "all" and config["missing_policy"] == "error", f"{name}: fixed study phase/cohort changed")
        for key, value in {"seed": 20260913, "n_folds": 5, "n_estimators": 200, "max_depth": 12,
                           "min_samples_leaf": 2, "max_features": "sqrt", "bootstrap": True, "bootstrap_repetitions": 2000}.items():
            require(config[key] == value, f"{name}: fixed {key} differs")
    return {"status": "verified", "required_runs": list(expected), "expected_forest_fits": 160}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a, b, tolerance=1e-10) -> bool:
    return bool(np.allclose(a, b, atol=tolerance, rtol=tolerance, equal_nan=True))


def matrix_digest(matrix: np.ndarray) -> str:
    """Match the explanation archive's documented shape-plus-float64 digest."""
    array = np.ascontiguousarray(matrix, dtype="<f8")
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def metrics(y, predicted) -> dict:
    variable = len(y) >= 3 and np.std(y) > 1e-12 and np.std(predicted) > 1e-12
    return {"pcc": float(pearsonr(y, predicted).statistic) if variable else np.nan,
            "spearman": float(spearmanr(y, predicted).statistic) if variable else np.nan,
            "rmse_z": float(np.sqrt(np.mean((y - predicted) ** 2)))}


def audit_manifests(directory: Path, raw_data: Path) -> tuple[dict, dict, dict]:
    config = json.loads((directory / "dataset_config.json").read_text())
    require(config.get("complete") is True, "Dataset preparation is incomplete")
    for name, expected in config["manifest_sha256"].items():
        require(digest(directory / name) == expected, f"Dataset manifest hash changed: {name}")
    for name, expected in config.get("audit_files_sha256", {}).items():
        require(digest(directory / name) == expected, f"Historical audit evidence changed: {name}")
    original_hash = config.get("protocol_amendment", {}).get("original_edge_file_sha256")
    if original_hash:
        require(digest(directory / "sequence_edges_original.csv") == original_hash, "Original clustering evidence changed")
    proteins = pd.read_csv(directory / "proteins.csv", keep_default_na=False)
    residues = pd.read_csv(directory / "residues.csv", keep_default_na=False)
    folds = pd.read_csv(directory / "folds.csv", keep_default_na=False)
    require(not proteins.protein_id.duplicated().any(), "Duplicate protein manifest IDs")
    require(not folds.protein_id.duplicated().any(), "Duplicate fold manifest IDs")
    require(set(proteins.protein_id) == set(folds.protein_id) == set(residues.protein_id), "Protein/residue/fold cohorts differ")
    require(not residues.duplicated(["protein_id", "residue_key"]).any(), "Duplicate residue identities")
    table = proteins.merge(folds, on="protein_id", validate="one_to_one").set_index("protein_id")
    residue_groups = {}
    for protein, frame in residues.groupby("protein_id", sort=True):
        frame = frame.sort_values("row_index").reset_index(drop=True)
        require(np.array_equal(frame.row_index, np.arange(len(frame))), f"{protein}: noncontiguous row indices")
        require(len(frame) == table.loc[protein, "n_residues"], f"{protein}: manifest residue count mismatch")
        require(np.isfinite(frame[["x", "y", "z", "b_factor", "z_b_factor"]].to_numpy(float)).all(), f"{protein}: nonfinite coordinates/targets")
        require(not frame[["x", "y", "z"]].duplicated().any(), f"{protein}: repeated coordinates")
        y = frame.b_factor.to_numpy(float)
        require(np.std(y) > 0, f"{protein}: constant target")
        require(close((y-y.mean())/y.std(ddof=0), frame.z_b_factor.to_numpy(float)), f"{protein}: incorrect target normalization")
        keys = [f"{int(r.model_id)}|{r.chain_id}|{int(r.residue_number)}|{r.insertion_code}" for r in frame.itertuples()]
        require(keys == frame.residue_key.tolist(), f"{protein}: residue keys do not encode recorded identities")
        filename = PurePosixPath(str(table.loc[protein, "source_path"]).replace("\\", "/")).name
        require(digest(raw_data / "datasets/365" / filename) == table.loc[protein, "source_sha256"], f"{protein}: raw structure hash changed")
        residue_groups[protein] = frame
    for column in ["protein_fold", "cluster_fold"]:
        require(set(table[column]) == set(range(1, config["n_folds"]+1)), f"Missing frozen {column}")
    require(table.groupby("cluster_id").cluster_fold.nunique().max() == 1, "A sequence component spans cluster folds")
    edges = pd.read_csv(directory / "sequence_edges.csv", keep_default_na=False)
    for row in edges.itertuples():
        require(table.loc[row.protein_a, "cluster_id"] == table.loc[row.protein_b, "cluster_id"], "A similarity edge crosses clusters")
        if row.edge_type == "sequence_alignment":
            require(row.identity >= .30-1e-12 and row.coverage_a >= .80-1e-12 and row.coverage_b >= .80-1e-12
                    and row.aligned_pairs >= 50, "An alignment edge violates the amended criterion")
    report = {"proteins": len(table), "residues": len(residues), "sequence_components": table.cluster_id.nunique(),
              "sequence_edges": len(edges), "input_and_manifest_hashes": "verified"}
    return report, residue_groups, table.to_dict("index")


def audit_features(directory: Path, residue_groups: dict, required_blocks: list[str]) -> tuple[dict, dict]:
    h = hashlib.sha256()
    for source in [ROOT / "scripts/generate_study_features.py", ROOT / "src/psl_flexibility/sheaf.py"]:
        h.update(source.read_bytes())
    code_hash = h.hexdigest()
    upper_source = ROOT / "scripts/augment_upper_features.py"
    upper_hash = hashlib.sha256(upper_source.read_bytes() + (ROOT / "src/psl_flexibility/sheaf.py").read_bytes()).hexdigest() if upper_source.exists() else None
    coverage, hashes, names = Counter(), {}, {}
    for protein, frame in residue_groups.items():
        path = directory / f"{protein}.npz"
        require(path.exists(), f"Missing feature cache for {protein}")
        hashes[protein] = digest(path)
        signature = hashlib.sha256((code_hash + frame.to_csv(index=False)).encode()).hexdigest()
        with np.load(path, allow_pickle=False) as cache:
            require(np.array_equal(cache["residue_keys"], frame.residue_key.to_numpy(str)), f"{protein}: feature residue order mismatch")
            require(str(cache["source_signature"]) == signature, f"{protein}: stale feature source signature")
            for block in required_blocks:
                require(block in cache and block + "_names" in cache, f"{protein}: missing required block {block}")
            for key in cache.files:
                if not key.endswith("_names"):
                    continue
                block = key[:-6]
                if block not in cache:
                    continue
                matrix = np.asarray(cache[block])
                columns = np.asarray(cache[key], dtype=str).tolist()
                require(matrix.shape == (len(frame), len(columns)), f"{protein}: invalid shape for {block}")
                require(np.issubdtype(matrix.dtype, np.number) and np.isfinite(matrix).all(), f"{protein}: nonfinite/nonnumeric {block}")
                require(len(columns) == len(set(columns)), f"{protein}: duplicate columns in {block}")
                require(block not in names or names[block] == columns, f"{protein}: inconsistent columns for {block}")
                names[block] = columns
                coverage[block] += 1
            if "upper_source_sha256" in cache:
                require(str(cache["upper_source_sha256"]) == upper_hash, f"{protein}: stale upper-endpoint source hash")
    feature_set_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"verified_caches": len(hashes), "block_coverage": dict(coverage), "source_signatures": "verified",
            "feature_set_sha256": feature_set_hash}, hashes


def audit_run(directory: Path, manifest: Path, residue_groups: dict, metadata: dict, feature_hashes: dict) -> dict:
    config = json.loads((directory / "evaluation_config.json").read_text())
    if config.get("complete") is not True:
        return {"status": "pending", "directory": directory.name}
    for name, expected in config["manifest_sha256"].items():
        require(digest(manifest / name) == expected, f"{directory.name}: input manifest changed after evaluation: {name}")
    for protein, feature in config["feature_files"].items():
        require(feature_hashes[protein] == feature["sha256"], f"{directory.name}: feature cache changed after evaluation: {protein}")
    for name, expected in config.get("code_sha256", {}).items():
        require(digest(ROOT / "src/psl_flexibility" / name) == expected, f"{directory.name}: numerical evaluation source changed: {name}")
    for name, expected in config["outputs_sha256"].items():
        require(digest(directory / name) == expected, f"{directory.name}: completed output hash changed: {name}")
    cohort = pd.read_csv(directory / "evaluation_fold_manifest.csv", keep_default_na=False)
    require(set(cohort.protein_id) == set(config["feature_files"]), f"{directory.name}: evaluated feature/cohort mismatch")
    if config.get("cohort") == "all" and config.get("missing_policy") == "error":
        require(set(cohort.protein_id) == set(metadata), f"{directory.name}: full-cohort evaluation omitted proteins")
    require(config["n_proteins"] == len(cohort) and config["n_residues"] == sum(len(residue_groups[p]) for p in cohort.protein_id),
            f"{directory.name}: configured evaluation sample sizes differ")
    require(not cohort.protein_id.duplicated().any(), f"{directory.name}: repeated cohort IDs")
    for row in cohort.itertuples():
        expected = metadata[row.protein_id]
        require(row.cluster_id == expected["cluster_id"] and row.protein_fold == expected["protein_fold"]
                and row.cluster_fold == expected["cluster_fold"], f"{directory.name}: modified frozen partition")
    states = {}
    columns = ["protein_id", "row_index", "residue_key", "z_b_factor", "prediction_z", "protocol", "mode", "fold", "cluster_id"]
    for chunk in pd.read_csv(directory / "predictions.csv.gz", usecols=columns, chunksize=100000, keep_default_na=False):
        require(np.isfinite(chunk[["z_b_factor", "prediction_z"]].to_numpy(float)).all(), f"{directory.name}: nonfinite predictions")
        for (protocol, mode, protein), group in chunk.groupby(["protocol", "mode", "protein_id"], sort=False):
            require(protocol in config["protocols"] and mode in config["modes"] and protein in config["feature_files"], f"{directory.name}: unexpected prediction rows")
            frame = residue_groups[protein]
            indices = group.row_index.to_numpy(int)
            require(np.array_equal(indices, group.row_index) and (indices >= 0).all() and (indices < len(frame)).all(), f"{protein}: invalid prediction row indices")
            require(len(np.unique(indices)) == len(indices), f"{protein}: repeated prediction row")
            require(np.array_equal(group.residue_key.to_numpy(str), frame.residue_key.to_numpy(str)[indices]), f"{protein}: prediction identity mismatch")
            require(close(group.z_b_factor, frame.z_b_factor.to_numpy()[indices]), f"{protein}: prediction target mismatch")
            expected_fold = metadata[protein]["cluster_fold" if protocol == "cluster" else "protein_fold"]
            require((group.fold == expected_fold).all() and (group.cluster_id == metadata[protein]["cluster_id"]).all(), f"{protein}: out-of-fold assignment mismatch")
            key = (protocol, mode, protein)
            state = states.setdefault(key, {"seen": np.zeros(len(frame), bool), "pred": np.full(len(frame), np.nan)})
            require(not state["seen"][indices].any(), f"{protein}: duplicated prediction across chunks")
            state["seen"][indices] = True
            state["pred"][indices] = group.prediction_z.to_numpy(float)
    per = pd.read_csv(directory / "per_protein.csv", keep_default_na=False, na_values=[""])
    require(not per.duplicated(["protocol", "mode", "protein_id"]).any(), "Duplicate per-protein metrics")
    indexed = per.set_index(["protocol", "mode", "protein_id"])
    expected_count = len(config["protocols"]) * len(config["modes"]) * len(cohort)
    require(len(states) == len(per) == expected_count, f"{directory.name}: incomplete prediction/metric coverage")
    for key, state in states.items():
        require(state["seen"].all(), f"{directory.name}: missing residue predictions for {key}")
        actual = metrics(residue_groups[key[2]].z_b_factor.to_numpy(float), state["pred"])
        for name, value in actual.items():
            require(close(indexed.loc[key, name], value), f"{directory.name}: incorrect {name} for {key}")
    summary = pd.read_csv(directory / "summary.csv")
    for row in summary.itertuples():
        selected = per[(per.protocol == row.protocol) & (per["mode"] == row.mode)]
        require(row.n_proteins == len(selected), f"{directory.name}: summary sample size mismatch")
        for name in ["pcc", "spearman", "rmse_z"]:
            require(close(getattr(row, "mean_" + name), selected[name].mean()), f"{directory.name}: incorrect macro mean {name}")
    try:
        paired = pd.read_csv(directory / "paired_differences.csv")
    except pd.errors.EmptyDataError:
        require(len(config["modes"]) == 1, f"{directory.name}: missing paired comparisons")
        paired = pd.DataFrame()
    for row in paired.itertuples():
        left = per[(per.protocol == row.protocol) & (per["mode"] == row.left_mode)].set_index("protein_id")
        right = per[(per.protocol == row.protocol) & (per["mode"] == row.right_mode)].set_index("protein_id")
        difference = right[row.metric] - left[row.metric]
        require(row.difference == "right_minus_left" and close(row.mean_difference, difference.mean()), f"{directory.name}: paired difference mismatch")
    return {"status": "verified", "directory": directory.name, "proteins": len(cohort),
            "modes": len(config["modes"]), "protocols": config["protocols"], "residue_predictions": sum(len(s["pred"]) for s in states.values())}


def feature_matrix(directory: Path, protein: str, qualified_names: list[str]) -> np.ndarray:
    with np.load(directory / f"{protein}.npz", allow_pickle=False) as cache:
        columns = []
        for qualified in qualified_names:
            block, name = qualified.split(":", 1)
            index = np.flatnonzero(np.asarray(cache[block + "_names"], dtype=str) == name)
            require(len(index) == 1, f"{protein}: case feature name absent/ambiguous: {qualified}")
            columns.append(cache[block][:, int(index[0])])
    return np.column_stack(columns)


def audit_case(directory: Path, protein: str, features: Path, residue_groups: dict, metadata: dict, hashes: dict) -> dict:
    provenance = json.loads((directory / f"{protein}_provenance.json").read_text())
    require(digest(ROOT / "scripts/explain_cases.py") == provenance["source_sha256"], f"{protein}: explanation source changed")
    for suffix, expected in provenance["output_sha256"].items():
        require(digest(directory / f"{protein}_{suffix}") == expected, f"{protein}: case output changed: {suffix}")
    train, test = set(provenance["train_ids"]), set(provenance["test_ids"])
    require(len(train) == len(provenance["train_ids"]) and len(test) == len(provenance["test_ids"]), f"{protein}: repeated model partition IDs")
    require(not train & test and train | test == set(metadata), f"{protein}: model train/test partition invalid")
    require(provenance["protocol"] == "cluster" and provenance["mode"] == "graph+center0", f"{protein}: unexpected case model/protocol")
    require(protein in test and protein not in train, f"{protein}: case was not held out")
    fold = metadata[protein]["cluster_fold"]
    require(provenance["fold"] == fold and all(metadata[p]["cluster_fold"] != fold for p in train), f"{protein}: model training contains held-out fold")
    require(all(metadata[p]["cluster_fold"] == fold for p in test), f"{protein}: model test partition differs from frozen fold")
    require(all(metadata[p]["cluster_id"] != metadata[protein]["cluster_id"] for p in train), f"{protein}: case cluster leaked into training")
    require(hashes[protein] == provenance["case_feature_file_sha256"], f"{protein}: case features changed")
    require(set(provenance["training_feature_file_sha256"]) == train, f"{protein}: incomplete training feature provenance")
    for pid, expected in provenance["training_feature_file_sha256"].items():
        require(hashes[pid] == expected, f"{protein}: background model-training feature hash changed: {pid}")
    model_path = Path(provenance["model_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    require(digest(model_path) == provenance["model_sha256"], f"{protein}: saved model hash changed")
    with np.load(directory / f"{protein}_explanation.npz", allow_pickle=False) as case:
        values, shap = np.asarray(case["feature_values"]), np.asarray(case["shap_values"])
        names = case["feature_names"].astype(str).tolist()
        require(names == provenance["feature_names"], f"{protein}: feature names differ across explanation artifacts")
        require(values.shape == shap.shape and np.isfinite(values).all() and np.isfinite(shap).all(), f"{protein}: invalid SHAP arrays")
        require(matrix_digest(values) == provenance["case_input_matrix_sha256"], f"{protein}: original input matrix hash differs")
        require(np.array_equal(case["residue_keys"], residue_groups[protein].residue_key.to_numpy(str)), f"{protein}: SHAP residue mapping differs")
        require(close(values, feature_matrix(features, protein, names)), f"{protein}: explained inputs differ from evaluated cache")
        effective_values = case["effective_feature_values"]
        require(provenance["effective_input_dtype"] == "float32" and effective_values.dtype == np.dtype("float32"),
                f"{protein}: incorrect effective foreground precision")
        require(np.array_equal(effective_values, values.astype(np.float32)), f"{protein}: effective foreground is not the exact model-precision cast")
        require(matrix_digest(effective_values) == provenance["effective_feature_matrix_sha256"], f"{protein}: effective foreground hash differs")
        require(provenance["source_and_effective_predictions_identical"] is True, f"{protein}: model-precision prediction equivalence was not checked")
        require(close(float(case["expected_value"]), provenance["expected_value"]) and
                abs(float(case["expected_value"]) - provenance["background_prediction_mean"]) < 1e-6,
                f"{protein}: inconsistent explanation baseline")
        residual = case["prediction_z"] - float(case["expected_value"]) - shap.sum(axis=1)
        maximum = float(np.max(np.abs(residual)))
        require(maximum < 1e-6, f"{protein}: SHAP reconstruction fails ({maximum:g})")
        background_ids = case["background_protein_ids"].astype(str)
        background_keys = case["background_residue_keys"].astype(str)
        background = case["background_values"]
        require(matrix_digest(background) == provenance["background_matrix_sha256"], f"{protein}: original background hash differs")
        effective_background = case["effective_background_values"]
        require(effective_background.dtype == np.dtype("float32") and np.array_equal(effective_background, background.astype(np.float32)),
                f"{protein}: effective background is not the exact model-precision cast")
        require(matrix_digest(effective_background) == provenance["effective_background_matrix_sha256"], f"{protein}: effective background hash differs")
        require(len(background) == 100 and len(set(zip(background_ids, background_keys))) == 100, f"{protein}: background does not contain 100 unique rows")
        require(set(background_ids).issubset(train), f"{protein}: background includes held-out residues")
        training_keys = pd.concat([residue_groups[pid][["protein_id", "residue_key", "row_index"]]
                                   for pid in provenance["train_ids"]], ignore_index=True)
        expected_seed = int(provenance["training_seed"]) + int(provenance["fold"])
        require(provenance["background_seed"] == expected_seed, f"{protein}: inconsistent background seed")
        indices = np.random.default_rng(expected_seed).choice(len(training_keys), 100, replace=False)
        sampled = training_keys.iloc[indices]
        require(np.array_equal(background_ids, sampled.protein_id.to_numpy(str)) and
                np.array_equal(background_keys, sampled.residue_key.to_numpy(str)), f"{protein}: background does not match the seeded training sample")
        for pid in sorted(set(background_ids)):
            rows = residue_groups[pid].set_index("residue_key")
            indices = rows.loc[background_keys[background_ids == pid], "row_index"].to_numpy(int)
            require(close(background[background_ids == pid], feature_matrix(features, pid, names)[indices]), f"{protein}: saved background differs from training features")
        case_prediction = case["prediction_z"].copy()
    main_check = provenance["main_prediction_check"]
    require(main_check["status"] == "verified", f"{protein}: case/main prediction equivalence has not been verified")
    main_path = Path(main_check["path"])
    if not main_path.is_absolute():
        main_path = ROOT / main_path
    require(digest(main_path) == main_check["sha256"], f"{protein}: main prediction table changed since case verification")
    selected = []
    for chunk in pd.read_csv(main_path, chunksize=100000, keep_default_na=False,
                             usecols=["protein_id", "residue_key", "protocol", "mode", "fold", "prediction_z"]):
        mask = (chunk.protein_id == protein) & (chunk.protocol == "cluster") & (chunk["mode"] == "graph+center0")
        if mask.any():
            selected.append(chunk.loc[mask])
    require(bool(selected), f"{protein}: main evaluation contains no case predictions")
    selected = pd.concat(selected, ignore_index=True).set_index("residue_key")
    keys = residue_groups[protein].residue_key.tolist()
    require(not selected.index.duplicated().any() and set(selected.index) == set(keys), f"{protein}: main/case residue coverage differs")
    require((selected.loc[keys, "fold"] == fold).all() and close(selected.loc[keys, "prediction_z"], case_prediction),
            f"{protein}: explained predictions differ from the main held-out predictions")
    return {"protein_id": protein, "residues": len(values), "features": len(names), "background_rows": 100,
            "held_out_cluster": metadata[protein]["cluster_id"], "maximum_SHAP_reconstruction_error": maximum,
            "original_and_effective_matrix_hashes": "verified", "effective_input_dtype": "float32",
            "effective_arrays_equal_exact_float32_cast": True, "seeded_training_background": "verified",
            "status": "verified"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/study/dataset")
    parser.add_argument("--features", type=Path, default=ROOT / "data/features")
    parser.add_argument("--raw-data", type=Path, default=ROOT / "data/raw/MDG_bfactor-main")
    parser.add_argument("--study", type=Path, default=ROOT / "results/study")
    parser.add_argument("--cases", type=Path, default=ROOT / "results/study/cases")
    parser.add_argument("--required-blocks", help="Comma-separated override; complete/fixed study audits default to all 14 blocks, otherwise the five core blocks")
    parser.add_argument("--require-complete", action="store_true", help="Require complete evaluations; at the default study directory also require all eight fixed-plan runs and modes")
    parser.add_argument("--require-fixed-study", action="store_true", help="Require all eight fixed-plan runs, modes and settings even at a custom study directory")
    parser.add_argument("--require-cases", action="store_true", help="Require both 1ULR and 1X3O explanation artifacts")
    parser.add_argument("--out", type=Path, help="Write this JSON report; study inputs and results stay unchanged")
    args = parser.parse_args()
    fixed_study = args.require_fixed_study or (args.require_complete and args.study.resolve() == (ROOT / "results/study").resolve())
    required_blocks = ([b.strip() for b in args.required_blocks.split(",") if b.strip()] if args.required_blocks is not None
                       else list(STUDY_BLOCKS if args.require_complete or fixed_study else CORE_BLOCKS))
    report = {"status": "failed", "checks": {}, "failures": [],
              "provenance": {"audit_script_sha256": digest(Path(__file__)), "required_blocks": required_blocks,
                             "required_completed_evaluations": args.require_complete, "required_cases": args.require_cases,
                             "required_fixed_study": fixed_study,
                             "scope": "File hashes, residue identities, frozen folds, finite features, per-protein metrics and macro means, paired means, and saved held-out SHAP reconstruction; no refitting."}}
    try:
        if fixed_study:
            report["checks"]["fixed_study_plan"] = require_fixed_study_plan(args.study)
        manifest_report, residues, metadata = audit_manifests(args.manifest, args.raw_data)
        report["checks"]["dataset"] = manifest_report
        feature_report, hashes = audit_features(args.features, residues, required_blocks)
        report["checks"]["features"] = feature_report
        runs = [audit_run(path.parent, args.manifest, residues, metadata, hashes)
                for path in sorted(args.study.rglob("evaluation_config.json"))]
        report["checks"]["evaluations"] = runs
        complete = bool(runs) and all(run["status"] == "verified" for run in runs)
        if args.require_complete:
            require(complete, "No completed evaluations, or at least one evaluation remains incomplete")
        cases = []
        for protein in ["1ULR", "1X3O"]:
            exists = (args.cases / f"{protein}_explanation.npz").exists()
            if args.require_cases:
                require(exists, f"Missing prespecified case explanation: {protein}")
            if exists:
                cases.append(audit_case(args.cases, protein, args.features, residues, metadata, hashes))
        report["checks"]["cases"] = cases
        report["status"] = "verified" if complete else "inputs_verified_evaluations_pending"
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        report["failures"].append(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
