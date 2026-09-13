#!/usr/bin/env python3
"""Describe frozen spectral features without targets, prediction scores or refits.

Fractions pool residue-centered neighborhoods and are descriptive, not estimates
from independent observations. This script never changes a feature cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTIONS = ("identity", "geometric", "center")
STATISTICS = ("nullity", "min_positive", "max", "mean", "median", "std")
FAMILIES = {
    "0": {"degree": 0, "kind": "ordinary", "pairs": ((3, 3), (4, 4), (5, 5), (6, 6))},
    "1": {"degree": 1, "kind": "ordinary_lower", "pairs": ((3, 3), (5, 5))},
    "_p1": {"degree": 1, "kind": "persistent", "pairs": ((3, 4), (5, 6))},
    "_upper1": {"degree": 1, "kind": "ordinary_upper", "pairs": ((4, 4), (6, 6))},
}
ATOL, RTOL = 1e-10, 1e-8


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask))


def nullity_statistics(values: np.ndarray) -> dict:
    require(np.isfinite(values).all() and (values >= 0).all(), "Invalid nullity values")
    require(np.allclose(values, np.rint(values), atol=ATOL, rtol=0), "Noninteger nullity")
    values = np.rint(values).astype(int)
    return {"n_neighborhoods": len(values), "n_zero": int((values == 0).sum()),
            "n_nonzero": int((values > 0).sum()), "fraction_zero": fraction(values == 0),
            "fraction_nonzero": fraction(values > 0), "mean_nullity": float(values.mean()),
            "median_nullity": float(np.median(values)), "max_nullity": int(values.max()),
            "min_nullity": int(values.min())}


def names_for(block: str, suffix: str) -> list[str]:
    if suffix == "0":
        return [f"{block}_a{a}_d0_{stat}" for a, _ in FAMILIES[suffix]["pairs"] for stat in STATISTICS]
    return [f"{block}_a{a}_b{b}_d1_{stat}" for a, b in FAMILIES[suffix]["pairs"] for stat in STATISTICS]


def characterize(manifest: Path, features: Path, out: Path) -> dict:
    config = json.loads((manifest / "dataset_config.json").read_text())
    require(config.get("complete") is True, "Dataset preparation is not complete")
    for name, expected in config["manifest_sha256"].items():
        require(digest(manifest / name) == expected, f"Changed frozen manifest: {name}")
    proteins = pd.read_csv(manifest / "proteins.csv", keep_default_na=False)
    # Deliberately do not load raw or standardized B-factor targets.
    coordinates = pd.read_csv(manifest / "residues.csv", keep_default_na=False,
                              usecols=["protein_id", "row_index", "residue_key", "x", "y", "z"])
    require(set(coordinates.protein_id) == set(proteins.protein_id), "Coordinate/protein cohorts differ")
    blocks = [kind + suffix for kind in CONSTRUCTIONS for suffix in FAMILIES]
    expected_columns = {kind + suffix: names_for(kind + suffix, suffix)
                        for kind in CONSTRUCTIONS for suffix in FAMILIES}
    expected_columns["native0"] = [f"native_r{r}_d0_{s}" for r in (6, 9, 12) for s in STATISTICS]
    collected = {block: [] for block in blocks + ["native0"]}
    neighborhood_sizes = {radius: [] for radius in (6, 9, 12)}
    source_hashes = {}
    count6_matches = []
    for protein, frame in coordinates.groupby("protein_id", sort=True):
        frame = frame.sort_values("row_index")
        require(np.array_equal(frame.row_index, np.arange(len(frame))), f"{protein}: incorrect row indices")
        path = features / f"{protein}.npz"
        before_hash = digest(path)
        with np.load(path, allow_pickle=False) as cache:
            require(np.array_equal(cache["residue_keys"], frame.residue_key.to_numpy(str)), f"{protein}: feature identity mismatch")
            for block in collected:
                expected = expected_columns[block]
                require(cache[block + "_names"].astype(str).tolist() == expected, f"{protein}: unexpected columns for {block}")
                values = np.asarray(cache[block], dtype=float)
                require(values.shape == (len(frame), len(expected)) and np.isfinite(values).all(), f"{protein}: invalid {block}")
                collected[block].append(values)
            distances = cdist(frame[["x", "y", "z"]].to_numpy(float), frame[["x", "y", "z"]].to_numpy(float))
            for radius in neighborhood_sizes:
                neighborhood_sizes[radius].append((distances <= radius + 1e-12).sum(axis=1))
            graph_names = cache["graph_names"].astype(str).tolist()
            graph_count6 = cache["graph"][:, graph_names.index("contact_count_r6")] + 1
            count6_matches.append(graph_count6 == neighborhood_sizes[6][-1])
        require(digest(path) == before_hash, f"{protein}: feature cache changed during characterization")
        source_hashes[protein] = before_hash
    collected = {block: np.vstack(parts) for block, parts in collected.items()}
    n_rows = len(coordinates)
    require(all(len(values) == n_rows for values in collected.values()), "Feature coverage differs")
    rows = []
    for construction in CONSTRUCTIONS:
        for suffix, family in FAMILIES.items():
            for index, (lower, upper) in enumerate(family["pairs"]):
                rows.append({"construction": construction, "block": construction + suffix,
                             "degree": family["degree"], "kind": family["kind"],
                             "lower_radius_angstrom": lower, "upper_radius_angstrom": upper,
                             **nullity_statistics(collected[construction + suffix][:, 6 * index])})
    for index, radius in enumerate((6, 9, 12)):
        rows.append({"construction": "native_graph", "block": "native0", "degree": 0, "kind": "ordinary",
                     "lower_radius_angstrom": radius, "upper_radius_angstrom": radius,
                     **nullity_statistics(collected["native0"][:, 6 * index])})

    comparisons = []
    for suffix, family in FAMILIES.items():
        for index, (lower, upper) in enumerate(family["pairs"]):
            left, right = collected["identity" + suffix][:, 6 * index], collected["geometric" + suffix][:, 6 * index]
            comparisons.append({"family": family["kind"], "degree": family["degree"], "lower": lower, "upper": upper,
                                "n_equal": int((left == right).sum()), "n_unequal": int((left != right).sum()),
                                "fraction_equal": fraction(left == right), "maximum_absolute_difference": float(np.abs(left - right).max())})

    degree1 = []
    for construction in CONSTRUCTIONS:
        for index, (lower, upper) in enumerate(((3, 4), (5, 6))):
            values = {name: collected[construction + suffix][:, 6 * index:6 * (index + 1)]
                      for name, suffix in (("lower", "1"), ("persistent", "_p1"), ("upper", "_upper1"))}
            row = {"construction": construction, "lower": lower, "upper": upper, "n_neighborhoods": n_rows}
            for endpoint in ("lower", "upper"):
                a, b = values["persistent"], values[endpoint]
                different_summary = ~np.isclose(a[:, 1:], b[:, 1:], atol=ATOL, rtol=RTOL).all(axis=1)
                row.update({f"fraction_persistent_nullity_differs_from_{endpoint}": fraction(a[:, 0] != b[:, 0]),
                            f"fraction_persistent_nullity_exceeds_{endpoint}": fraction(a[:, 0] > b[:, 0]),
                            f"fraction_positive_summaries_differ_from_{endpoint}": fraction(different_summary)})
            row["fraction_lower_nullity_differs_from_upper"] = fraction(values["lower"][:, 0] != values["upper"][:, 0])
            row["fraction_all_three_nullities_equal"] = fraction((values["lower"][:, 0] == values["persistent"][:, 0]) &
                                                                (values["persistent"][:, 0] == values["upper"][:, 0]))
            degree1.append(row)

    native = []
    for index, radius in enumerate((6, 9, 12)):
        sizes = np.concatenate(neighborhood_sizes[radius])
        values = collected["native0"][:, 6 * index:6 * (index + 1)]
        eligible = sizes >= 2
        error = np.abs(values[eligible, 2] - 2 * sizes[eligible])
        matches = np.isclose(values[eligible, 2], 2 * sizes[eligible], atol=ATOL, rtol=RTOL)
        native.append({"radius_angstrom": radius, "n_neighborhoods": n_rows,
                       "fraction_nullity_one": fraction(values[:, 0] == 1),
                       "n_at_least_two_vertices": int(eligible.sum()), "n_one_vertex": int((sizes == 1).sum()),
                       "fraction_max_equals_twice_vertex_count_for_n_ge_2": fraction(matches),
                       "maximum_absolute_error_from_twice_vertex_count": float(error.max()),
                       "minimum_vertices": int(sizes.min()), "median_vertices": float(np.median(sizes)),
                       "maximum_vertices": int(sizes.max())})

    report = {
        "status": "complete", "n_proteins": len(proteins), "n_residue_centered_neighborhoods": n_rows,
        "method": "Descriptive pooled-residue summaries of all frozen feature caches; no targets or prediction results loaded; no spectra or models recomputed.",
        "weighting": "Each residue-centered neighborhood receives equal weight; overlapping neighborhoods and residues within proteins are dependent.",
        "positive_summary_comparison": {
            "statistics": list(STATISTICS[1:]), "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
            "meaning": "At least one of five stored positive-spectrum summaries differs numerically. Equality of summaries does not prove equality of full spectra or matrices; upper and lower operators may have different chain-space dimensions. Zero summaries can represent empty positive spectra.",
        },
        "nullity_interpretation": "Identity-sheaf degree-one nullity describes ordinary or persistent first homology of the local alpha filtration. Geometric and center-label sheaf nullities describe their corresponding sheaf homology; center-label nullity must not be equated to the number of geometric loops.",
        "identity_geometric_nullity_comparisons": comparisons,
        "degree1_endpoint_comparisons": degree1,
        "native_graph_redundancy": native,
        "native_radius6_size_agreement_with_graph_contact_count_plus_one": fraction(np.concatenate(count6_matches)),
        "provenance": {
            "script_sha256": digest(Path(__file__)),
            "manifest_sha256": {name: digest(manifest / name) for name in ["proteins.csv", "residues.csv", "dataset_config.json"]},
            "feature_files": len(source_hashes),
            "feature_set_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "feature_set_digest_definition": "SHA256 of compact JSON mapping sorted protein IDs to complete NPZ SHA256 digests",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    report["provenance"]["nullity_csv_sha256"] = digest(csv_path)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/study/dataset")
    parser.add_argument("--features", type=Path, default=ROOT / "data/features")
    parser.add_argument("--out", type=Path, default=ROOT / "results/study/feature_characterization")
    args = parser.parse_args()
    report = characterize(args.manifest, args.features, args.out)
    print(json.dumps({key: report[key] for key in ["status", "n_proteins", "n_residue_centered_neighborhoods",
                     "identity_geometric_nullity_comparisons", "degree1_endpoint_comparisons", "native_graph_redundancy"]}, indent=2))


if __name__ == "__main__":
    main()
