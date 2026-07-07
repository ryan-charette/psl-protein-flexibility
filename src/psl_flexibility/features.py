"""Feature generation for PSL protein-flexibility descriptors."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from psl_flexibility.native_psl import NativePersistentSheafLaplacian
from psl_flexibility.structure import ResidueRecord, bfactors, coordinates


ZERO_TOL = 1e-8


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for local PSL feature generation."""

    radii: tuple[float, ...] = (6.0, 9.0, 12.0)
    sheaf: str = "center_labeled"
    stats: str = "both"
    degrees: tuple[int, ...] = (0,)
    p_widths: tuple[float, ...] = (0.0,)
    scale_labels: bool = False


def stat_names(stats: str) -> list[str]:
    """Return the spectral statistics used by a stats mode."""

    if stats in {"median", "current"}:
        return ["max", "min", "mean", "median", "zeros"]
    if stats in {"std", "hayes_blind"}:
        return ["max", "min", "mean", "std", "zeros"]
    if stats in {"both", "extended"}:
        return ["max", "min", "mean", "median", "std", "zeros"]
    raise ValueError(f"Unknown stats mode: {stats}")


def eigen_stats(eigenvalues: np.ndarray, stats: str, zero_tol: float = ZERO_TOL) -> list[float]:
    """Summarize a Laplacian spectrum over nonzero eigenvalues plus zero count."""

    evals = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if evals.size == 0:
        values = {
            "max": 0.0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "zeros": 0.0,
        }
        return [values[name] for name in stat_names(stats)]

    nonzero = evals[np.abs(evals) > zero_tol]
    zero_count = float(evals.size - nonzero.size)
    if nonzero.size == 0:
        values = {
            "max": 0.0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "zeros": zero_count,
        }
    else:
        values = {
            "max": float(np.max(nonzero)),
            "min": float(np.min(nonzero)),
            "mean": float(np.mean(nonzero)),
            "median": float(np.median(nonzero)),
            "std": float(np.std(nonzero)),
            "zeros": zero_count,
        }
    return [values[name] for name in stat_names(stats)]


def feature_names(config: FeatureConfig) -> list[str]:
    """Return feature names matching ``features_for_coordinates`` output."""

    names: list[str] = []
    for radius in config.radii:
        r_label = _float_label(radius)
        for p_width in config.p_widths:
            p_label = _float_label(p_width)
            for degree in config.degrees:
                for stat in stat_names(config.stats):
                    names.append(f"r{r_label}_p{p_label}_d{degree}_{stat}")
    return names


def features_for_coordinates(coords: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Compute a per-residue PSL feature matrix for one protein."""

    points = np.asarray(coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("coords must be an n x 3 coordinate array.")
    if points.shape[0] == 0:
        raise ValueError("coords must contain at least one point.")

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    rows: list[list[float]] = []
    for center_index in range(points.shape[0]):
        row: list[float] = []
        for radius in config.radii:
            local_indices = np.flatnonzero(distances[center_index] <= float(radius) + 1e-12)
            local_points = points[local_indices]
            center_matches = np.flatnonzero(local_indices == center_index)
            if center_matches.size != 1:
                raise RuntimeError(f"Could not locate center residue {center_index} in its local neighborhood.")
            center_local_index = int(center_matches[0])
            row.extend(_features_for_neighborhood(local_points, center_local_index, float(radius), config))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def features_for_residues(records: list[ResidueRecord], config: FeatureConfig) -> np.ndarray:
    """Compute a per-residue PSL feature matrix from parsed residue records."""

    return features_for_coordinates(coordinates(records), config)


def write_feature_csv(
    path: Path,
    records_by_protein: Iterable[tuple[str, list[ResidueRecord], np.ndarray]],
    names: Sequence[str],
) -> int:
    """Write residue metadata, targets, and features to a single CSV file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protein_id",
        "residue_index",
        "chain_id",
        "residue_number",
        "insertion_code",
        "residue_name",
        "b_factor",
        "z_b_factor",
        *names,
    ]
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for protein_id, records, features in records_by_protein:
            if features.shape != (len(records), len(names)):
                raise ValueError(
                    f"Feature shape mismatch for {protein_id}: {features.shape} does not match "
                    f"{len(records)} residues and {len(names)} names."
                )
            z_values = _zscore(bfactors(records))
            for index, (record, z_value) in enumerate(zip(records, z_values, strict=True), start=1):
                feature_values = {name: float(features[index - 1, col]) for col, name in enumerate(names)}
                writer.writerow(
                    {
                        "protein_id": protein_id,
                        "residue_index": index,
                        "chain_id": record.chain_id,
                        "residue_number": record.residue_number,
                        "insertion_code": record.insertion_code,
                        "residue_name": record.residue_name,
                        "b_factor": float(record.b_factor),
                        "z_b_factor": float(z_value),
                        **feature_values,
                    }
                )
                row_count += 1
    return row_count


def write_config(path: Path, config: FeatureConfig) -> None:
    """Write feature configuration JSON."""

    serializable = asdict(config)
    Path(path).write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse comma-separated floats."""

    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def parse_int_list(text: str) -> tuple[int, ...]:
    """Parse comma-separated integers."""

    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _features_for_neighborhood(
    local_points: np.ndarray,
    center_local_index: int,
    radius: float,
    config: FeatureConfig,
) -> list[float]:
    block_values: list[float] = []
    for p_width in config.p_widths:
        if local_points.shape[0] < 2:
            for degree in config.degrees:
                block_values.extend(_degenerate_stats(local_points.shape[0], degree, config.stats))
            continue

        psl = NativePersistentSheafLaplacian(
            pts=local_points,
            charges=_charges(local_points.shape[0], center_local_index, config.sheaf),
            filtration_type="rips",
            radius_list=np.array([radius], dtype=float),
            p=float(p_width),
            constant=config.sheaf == "constant",
            scale=config.scale_labels,
        )
        psl.build_filtration()
        psl.build_simplicial_pair()
        psl.build_matrices()
        for degree in config.degrees:
            matrix = _matrix_for_degree(psl, degree)
            eigenvalues = np.linalg.eigvalsh(matrix) if matrix.size else np.array([], dtype=float)
            block_values.extend(eigen_stats(eigenvalues, config.stats))
    return block_values


def _charges(n_points: int, center_local_index: int, sheaf: str) -> np.ndarray | None:
    if sheaf == "constant":
        return None
    if sheaf not in {"center_labeled", "atom_centered"}:
        raise ValueError(f"Unknown sheaf mode: {sheaf}")
    charges = np.ones(n_points, dtype=float)
    charges[center_local_index] = 0.0
    return charges


def _matrix_for_degree(psl: NativePersistentSheafLaplacian, degree: int) -> np.ndarray:
    if degree == 0:
        return psl.psl_0()[0]
    if degree == 1:
        return psl.psl_1()[0]
    if degree == 2:
        return psl.psl_2()[0]
    raise ValueError("Only PSL degrees 0, 1, and 2 are supported.")


def _degenerate_stats(n_points: int, degree: int, stats: str) -> list[float]:
    values = {
        "max": 0.0,
        "min": 0.0,
        "mean": 0.0,
        "median": 0.0,
        "std": 0.0,
        "zeros": float(n_points if degree == 0 else 0),
    }
    return [values[name] for name in stat_names(stats)]


def _zscore(values: np.ndarray) -> np.ndarray:
    scale = float(np.nanstd(values))
    if scale == 0.0 or not np.isfinite(scale):
        return np.zeros_like(values, dtype=float)
    return (values - float(np.nanmean(values))) / scale


def _float_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p").replace("-", "m")


__all__ = [
    "FeatureConfig",
    "ZERO_TOL",
    "eigen_stats",
    "feature_names",
    "features_for_coordinates",
    "features_for_residues",
    "parse_float_list",
    "parse_int_list",
    "stat_names",
    "write_config",
    "write_feature_csv",
]
