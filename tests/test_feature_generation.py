from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from psl_flexibility.demo import run_toy_demo
from psl_flexibility.features import FeatureConfig, feature_names, features_for_coordinates
from psl_flexibility.structure import parse_ca_pdb


def test_feature_names_match_generated_columns() -> None:
    config = FeatureConfig(radii=(2.0, 3.0), stats="both", degrees=(0,), p_widths=(0.0,))
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    features = features_for_coordinates(coords, config)

    assert features.shape == (3, len(feature_names(config)))


def test_parse_ca_pdb_reads_toy_residue_metadata() -> None:
    records = parse_ca_pdb(Path("examples") / "toy_proteins" / "toy_alpha.pdb")

    assert len(records) == 8
    assert records[0].protein_id == "toy_alpha"
    assert records[0].residue_name == "ALA"
    assert records[0].chain_id == "A"


def test_toy_demo_writes_runnable_outputs(tmp_path: Path) -> None:
    summary = run_toy_demo(tmp_path)

    feature_path = tmp_path / "toy_psl_features.csv"
    metrics_path = tmp_path / "toy_demo_metrics.csv"
    assert feature_path.exists()
    assert metrics_path.exists()
    assert summary["n_proteins"] == 3
    assert summary["n_residues"] == 24

    with feature_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert "r4_p0_d0_mean" in rows[0]
