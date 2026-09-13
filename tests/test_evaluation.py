"""Paired evaluation must preserve residue identity and out-of-fold separation."""

import joblib
import numpy as np
import pandas as pd
import pytest

from psl_flexibility.evaluation import (
    StudyProtein, cluster_bootstrap, evaluate_mode, load_study, paired_comparisons,
    prediction_metrics, validate_partitions,
)


def make_proteins():
    result = []
    for i in range(10):
        protein = f"P{i}"
        frame = pd.DataFrame({"protein_id": [protein] * 6, "row_index": np.arange(6),
                              "residue_key": [f"1|A|{j}|." for j in range(6)],
                              "z_b_factor": np.linspace(-1, 1, 6), "b_factor": np.arange(6) + 10.0})
        result.append(StudyProtein(protein, f"C{i // 2}", i % 5 + 1, i // 2 + 1, frame,
                                   {"graph": np.column_stack([np.arange(6), np.full(6, i)])},
                                   {"graph": ["graph:position", "graph:protein"]}, "", ""))
    return result


def test_cluster_validation_rejects_cluster_crossing():
    proteins = make_proteins()
    validate_partitions(proteins, "cluster")
    proteins[0].cluster_fold = 2
    with pytest.raises(ValueError, match="crosses"):
        validate_partitions(proteins, "cluster")


def test_grouped_predictions_cover_every_held_out_residue_once(tmp_path):
    proteins = make_proteins()
    predictions, per, timings = evaluate_mode(proteins, "graph", "cluster", n_estimators=3,
                                             model_dir=tmp_path, save_case_ids=("P0",))
    assert len(predictions) == 60 and len(per) == 10 and len(timings) == 5
    assert not predictions.duplicated(["protein_id", "residue_key"]).any()
    assert predictions.groupby("protein_id")["fold"].nunique().max() == 1
    for p in proteins:
        assert set(predictions.loc[predictions["protein_id"] == p.protein_id, "fold"]) == {p.cluster_fold}
    checkpoint = joblib.load(tmp_path / "cluster_graph_fold1.joblib")
    assert set(checkpoint["test_ids"]) == {"P0", "P1"}
    assert set(checkpoint["train_ids"]).isdisjoint(checkpoint["test_ids"])
    assert checkpoint["feature_names"] == proteins[0].names["graph"]


def test_cluster_bootstrap_weights_proteins_and_is_deterministic():
    values = np.array([0.0, 0.0, 1.0])
    clusters = np.array(["large", "large", "small"])
    result = cluster_bootstrap(values, clusters, repetitions=500)
    assert result[0] == pytest.approx(1 / 3)
    assert result == cluster_bootstrap(values, clusters, repetitions=500)
    assert result[1] == 0 and result[2] == 1


def test_paired_improvement_and_cohort_mismatch():
    rows = []
    for mode, increase in [("baseline", 0), ("addition", 0.1)]:
        for i in range(5):
            rows.append({"protocol": "cluster", "mode": mode, "protein_id": f"P{i}", "cluster_id": f"C{i}",
                         "pcc": i / 10 + increase, "spearman": i / 10 + increase, "rmse_z": 1 - increase})
    per = pd.DataFrame(rows)
    paired = paired_comparisons(per, repetitions=100)
    pcc = paired.loc[paired["metric"] == "pcc"].iloc[0]
    assert pcc["mean_difference"] == pytest.approx(0.1)
    assert pcc["ci95_low"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="identical protein cohorts"):
        paired_comparisons(per.iloc[:-1], repetitions=100)


def make_manifest_cache(tmp_path):
    manifest = tmp_path / "manifest"
    features = tmp_path / "features"
    manifest.mkdir()
    features.mkdir()
    pd.DataFrame({"protein_id": ["P"], "n_residues": [3]}).to_csv(manifest / "proteins.csv", index=False)
    pd.DataFrame({"protein_id": ["P"], "cluster_id": ["C"], "protein_fold": [1], "cluster_fold": [1]}).to_csv(manifest / "folds.csv", index=False)
    pd.DataFrame({"protein_id": ["P"] * 3, "row_index": [0, 1, 2],
                  "residue_key": ["1|A|1|.", "1|A|2|.", "1|A|3|."], "z_b_factor": [-1, 0, 1]}).to_csv(manifest / "residues.csv", index=False)
    np.savez(features / "P.npz", residue_keys=np.array(["1|A|1|.", "1|A|2|.", "1|A|3|."]),
             graph=np.arange(6).reshape(3, 2), graph_names=np.array(["degree", "distance"]),
             center0=np.arange(6).reshape(3, 2), center0_names=np.array(["nullity", "mean"]))
    return manifest, features


def test_feature_selection_preserves_graph_and_selected_spectrum(tmp_path):
    manifest, features = make_manifest_cache(tmp_path)
    proteins, _ = load_study(manifest, features, ["graph+center0"], feature_columns=r"^graph:|nullity$")
    assert proteins[0].names["graph+center0"] == ["graph:degree", "graph:distance", "center0:nullity"]
    assert proteins[0].features["graph+center0"].shape == (3, 3)


def test_equal_length_wrong_residue_order_is_rejected(tmp_path):
    manifest, features = make_manifest_cache(tmp_path)
    np.savez(features / "P.npz", residue_keys=np.array(["1|A|3|.", "1|A|2|.", "1|A|1|."]),
             graph=np.arange(6).reshape(3, 2), graph_names=np.array(["degree", "distance"]))
    with pytest.raises(ValueError, match="keys do not match"):
        load_study(manifest, features, ["graph"], missing="exclude")


def test_constant_prediction_has_no_correlation_but_valid_error():
    values = prediction_metrics(np.array([-1, 0, 1]), np.array([0, 0, 0]))
    assert np.isnan(values["pcc"]) and np.isnan(values["spearman"])
    assert values["rmse_z"] == pytest.approx(np.sqrt(2 / 3))
