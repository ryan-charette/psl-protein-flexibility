"""Write a seeded, independent numerical validation report for the study operators.

This is a diagnostic experiment, not a substitute for the analytic identities
or the test suite. All point clouds are synthetic; no prediction outcomes enter
their selection. Run from any directory with the study requirements installed.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gudhi
import numpy as np
import scipy
from scipy import linalg
from threadpoolctl import threadpool_limits

from psl_flexibility.native_psl import NativePersistentSheafLaplacian
from psl_flexibility.sheaf import AlphaSheaf, DEFAULT_RTOL, persistent_laplacian, spectral_statistics


SEED = 20260913
KINDS = ("identity", "geometric", "center_zero")


def maxabs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array))) if array.size else 0.0


class Diagnostics:
    def __init__(self) -> None:
        self.metrics: dict[str, dict] = {}
        self.failures: list[dict] = []

    def compare(self, name: str, actual, expected, case: str, *, atol=1e-12, rtol=1e-8) -> None:
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            self.failures.append({"metric": name, "case": case, "reason": "shape mismatch",
                                  "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)})
            return
        error = maxabs(actual - expected)
        scale = max(maxabs(actual), maxabs(expected))
        threshold = atol + rtol * scale
        self.residual(name, error, scale, threshold, case, atol=atol, rtol=rtol)

    def residual(self, name: str, error: float, scale: float, threshold: float, case: str, **tolerances) -> None:
        record = self.metrics.setdefault(name, {"count": 0, "max_absolute_error": 0.0,
                    "max_relative_error": 0.0, "max_tolerance_fraction": 0.0,
                    "worst_absolute_case": None, **tolerances})
        record["count"] += 1
        if error >= record["max_absolute_error"]:
            record["max_absolute_error"], record["worst_absolute_case"] = float(error), case
        record["max_relative_error"] = max(record["max_relative_error"], error / scale if scale else 0.0)
        record["max_tolerance_fraction"] = max(record["max_tolerance_fraction"], error / threshold if threshold else 0.0)
        if not np.isfinite(error) or error > threshold:
            self.failures.append({"metric": name, "case": case, "error": float(error), "threshold": float(threshold)})


def restriction(points: np.ndarray, face: tuple, coface: tuple, kind: str) -> float:
    """Direct coordinate formula, independent of the operator's cached factors."""
    if kind == "identity":
        return 1.0
    missing = set(coface) - set(face)
    if kind == "center_zero" and 0 in missing:
        return 0.0
    f_face = np.prod([np.linalg.norm(points[i] - points[j]) for i, j in combinations(face, 2)])
    f_coface = np.prod([np.linalg.norm(points[i] - points[j]) for i, j in combinations(coface, 2)])
    return float(f_face / f_coface)


def check_composition(sheaf: AlphaSheaf, radius: float, checks: Diagnostics, label: str) -> None:
    d0, d1 = sheaf.coboundary(0, radius), sheaf.coboundary(1, radius)
    edges, triangles = sheaf.simplices(radius, 1), sheaf.simplices(radius, 2)
    edge_index = {edge: index for index, edge in enumerate(edges)}
    composition = d1 @ d0
    scale = maxabs(d1) * maxabs(d0) * max(1, d0.shape[0])
    checks.residual("D1_D0", maxabs(composition), scale, 1e-13 + 1e-12 * scale, label, atol=1e-13, rtol=1e-12)
    for row, triangle in enumerate(triangles):
        for vertex in triangle:
            paths = []
            for edge in combinations(triangle, 2):
                if vertex not in edge:
                    continue
                column = edge_index[edge]
                observed = abs(d1[row, column]) * abs(d0[column, vertex])
                direct = restriction(sheaf.points, (vertex,), triangle, sheaf.kind)
                checks.compare("restriction_composition", observed, direct, label, atol=1e-13, rtol=1e-12)
                paths.append(observed)
            checks.compare("restriction_path_independence", paths[0], paths[1], label, atol=1e-13, rtol=1e-12)


def restriction_rank(sheaf: AlphaSheaf, a: float, b: float) -> tuple[int, int, int]:
    """Rank on cohomology via harmonic representatives, without a persistent operator.

    Restrict harmonic Y cochains to X's edges, then project onto harmonic X.
    This computes the image of H^1(Y) -> H^1(X). The separate SVD cutoff here
    is intentionally stricter than the Laplacian's eigenvalue cutoff.
    """
    d0x, d1x = sheaf.coboundary(0, a), sheaf.coboundary(1, a)
    d0y, d1y = sheaf.coboundary(0, b), sheaf.coboundary(1, b)
    hx = linalg.null_space(np.vstack([d1x, d0x.T]), rcond=1e-9)
    hy = linalg.null_space(np.vstack([d1y, d0y.T]), rcond=1e-9)
    lookup = {edge: index for index, edge in enumerate(sheaf.simplices(b, 1))}
    old = np.asarray([lookup[edge] for edge in sheaf.simplices(a, 1)], dtype=int)
    singular = linalg.svdvals(hx.T @ hy[old])
    return int(np.count_nonzero(singular > 1e-8)), hx.shape[1], hy.shape[1]


def validate_cloud(name: str, points: np.ndarray, pairs: tuple, rng: np.random.Generator,
                   checks: Diagnostics, nullities: list, spectra: list) -> dict:
    base = AlphaSheaf(points, "identity")
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    moved = AlphaSheaf(points @ rotation + [20., -13., 7.], "identity")
    permutation = np.r_[0, rng.permutation(np.arange(1, len(points)))]
    reordered = AlphaSheaf(points[permutation], "identity")
    maximum_size, new_edge_cases = 0, 0
    gauge_nullities = {}
    for kind in KINDS:
        sheaf, motion, permuted = base.with_kind(kind), moved.with_kind(kind), reordered.with_kind(kind)
        for radius in sorted({endpoint for pair in pairs for endpoint in pair}):
            check_composition(sheaf, radius, checks, f"{name}/{kind}/a{radius}")
        for a, b in pairs:
            case = f"{name}/{kind}/{a}-{b}"
            full = sheaf.laplacian(1, a, b, use_blocks=False)
            block = sheaf.laplacian(1, a, b)
            reference = sheaf.laplacian_nullspace(1, a, b)
            checks.compare("block_vs_full", block, full, case)
            checks.compare("schur_vs_independent_nullspace", full, reference, case)
            checks.compare("symmetry", block, block.T, case, atol=1e-14, rtol=1e-13)
            eigenvalues = linalg.eigvalsh(block)
            minimum = float(eigenvalues.min()) if eigenvalues.size else 0.0
            scale = maxabs(eigenvalues)
            checks.residual("positive_semidefiniteness", max(0., -minimum), scale,
                            1e-12 + DEFAULT_RTOL * scale, case, atol=1e-12, rtol=DEFAULT_RTOL)
            spectra.append({"case": case, "dimension": len(block), "min_eigenvalue": minimum,
                            "max_abs_eigenvalue": scale})
            expected, hx, hy = restriction_rank(sheaf, a, b)
            measured = int(spectral_statistics(eigenvalues)[0])
            checks.compare("persistent_nullity_vs_restriction_rank", measured, expected, case, atol=0., rtol=0.)
            nullities.append({"case": case, "persistent_nullity": measured, "independent_restriction_rank": expected,
                              "H1_lower_dimension": hx, "H1_upper_dimension": hy})
            gauge_nullities[kind, a, b] = measured
            maximum_size = max(maximum_size, len(block))
            old_edges, all_edges = sheaf.simplices(a, 1), sheaf.simplices(b, 1)
            if len(all_edges) > len(old_edges):
                new_edge_cases += 1
            lower, upper = sheaf.coboundary(0, a), sheaf.coboundary(1, b)
            lookup = {edge: index for index, edge in enumerate(all_edges)}
            old = np.asarray([lookup[edge] for edge in old_edges], dtype=int)
            edge_sign = rng.choice([-1., 1.], size=len(all_edges))
            triangle_sign = rng.choice([-1., 1.], size=upper.shape[0])
            oriented = persistent_laplacian(lower * edge_sign[old, None],
                        upper * edge_sign * triangle_sign[:, None], old)
            checks.compare("orientation_conjugacy", oriented, full * edge_sign[old, None] * edge_sign[old], case)
            if a == b:
                checks.compare("zero_width_ordinary_operator", full, lower @ lower.T + upper.T @ upper, case)
            for degree in (0, 1):
                expected_spectrum = linalg.eigvalsh(sheaf.laplacian(degree, a, b))
                checks.compare("rigid_motion_spectrum", linalg.eigvalsh(motion.laplacian(degree, a, b)),
                               expected_spectrum, f"{case}/degree{degree}")
                # Co-spherical Delaunay ties can choose different triangulations.
                # Permutation checks use general-position random clouds only.
                if name.startswith("random"):
                    checks.compare("neighbor_permutation_spectrum", linalg.eigvalsh(permuted.laplacian(degree, a, b)),
                                   expected_spectrum, f"{case}/degree{degree}")
            checks.compare("degree_zero_upper_endpoint", sheaf.laplacian(0, a, b), sheaf.laplacian(0, b), case)
    for a, b in pairs:
        checks.compare("geometric_gauge_persistent_nullity", gauge_nullities["geometric", a, b],
                       gauge_nullities["identity", a, b], f"{name}/{a}-{b}", atol=0., rtol=0.)
    return {"name": name, "points": points.tolist(), "affine_dimension": base.affine_dimension,
            "pairs": pairs, "maximum_operator_dimension": maximum_size,
            "positive_new_edge_comparisons": new_edge_cases}


def validate_native(rng: np.random.Generator, checks: Diagnostics) -> list:
    reports = []
    for n in (4, 7, 12, 25):
        directions = rng.normal(size=(n - 1, 3))
        neighbors = directions / np.linalg.norm(directions, axis=1)[:, None] * rng.uniform(.2, .95, size=(n - 1, 1))
        points = np.vstack([np.zeros(3), neighbors])
        adjacency = np.linalg.norm(neighbors[:, None] - neighbors[None, :], axis=2) <= 1.
        np.fill_diagonal(adjacency, False)
        mu = linalg.eigvalsh(np.diag(adjacency.sum(axis=1)) - adjacency.astype(float))
        for labeled in (False, True):
            weight = 2 if labeled else 1
            expected = np.sort(np.r_[0., weight * n, weight + mu[1:]])
            native = NativePersistentSheafLaplacian(points, charges=np.r_[0., np.ones(n - 1)],
                        radius_list=[1.], p=0., constant=not labeled)
            actual = linalg.eigvalsh(native.psl_0()[0])
            case = f"native/n{n}/center_weight{weight}"
            checks.compare("native_universal_center_spectrum", actual, expected, case, rtol=1e-12)
            stats = spectral_statistics(actual)
            checks.compare("native_nullity", stats[0], 1., case, atol=0., rtol=0.)
            checks.compare("native_maximum_identity", stats[2], weight * n, case, rtol=1e-12)
            checks.compare("native_mean_identity", stats[3], 2 * weight + adjacency.sum() / (n - 1), case, rtol=1e-12)
            reports.append({"case": case, "n_points": n, "neighbor_edges": int(adjacency.sum() // 2),
                            "spectrum": actual.tolist(), "expected_spectrum": expected.tolist()})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/study/operator_validation.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    started = time.perf_counter()
    rng, checks = np.random.default_rng(args.seed), Diagnostics()
    nullities, spectra, clouds = [], [], []
    fixtures = [
        ("square", np.array([[0., 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]),
         ((.5, .5), (1.05, 1.05), (1.05, 1.5), (1.5, 1.5))),
        ("line", np.array([[0., 0, 0], [2, 0, 0], [5, 0, 0]]), ((.5, .5), (1.1, 2.))),
        ("singleton", np.zeros((1, 3)), ((0., 0.), (0., 1.))),
    ]
    for n in (6, 9, 14, 22):
        for repeat in range(3):
            points = np.vstack([np.zeros(3), rng.uniform(-6., 6., size=(n - 1, 3))])
            fixtures.append((f"random_n{n}_r{repeat}", points, ((3., 3.), (3., 4.), (5., 5.), (5., 6.))))
    with threadpool_limits(limits=1):
        for name, points, pairs in fixtures:
            clouds.append(validate_cloud(name, points, pairs, rng, checks, nullities, spectra))
        square = AlphaSheaf(fixtures[0][1], "identity")
        cycle = {"lower_radius": 1.05, "upper_radius": 1.5, "edge_stalk_dimension": 4,
                 "ordinary_nullity": int(spectral_statistics(square.laplacian(1, 1.05))[0]),
                 "persistent_nullity": int(spectral_statistics(square.laplacian(1, 1.05, 1.5))[0])}
        checks.compare("known_cycle_death", [cycle["ordinary_nullity"], cycle["persistent_nullity"]],
                       [1, 0], "unit_square_scaled_by_two", atol=0., rtol=0.)
        native = validate_native(rng, checks)
    source_paths = [Path(__file__), ROOT / "src/psl_flexibility/sheaf.py", ROOT / "src/psl_flexibility/native_psl.py"]
    report = {
        "status": "passed" if not checks.failures else "failed", "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started, "blas_threads": 1,
        "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "gudhi": gudhi.__version__},
        "numerical_conventions": {"operator_eigenvalue_rtol": DEFAULT_RTOL,
            "harmonic_basis_svd_rcond": 1e-9, "restriction_rank_singular_cutoff": 1e-8,
            "comparison_rule": "max absolute entry error <= atol + rtol * max absolute entry across both inputs",
            "rigid_motion": "proper orthogonal rotation plus translation; no coordinate scaling",
            "permutation": "center held at index zero; only general-position random clouds",
            "random_support": "center zero and independent uniform coordinates in [-6,6]^3, contained within R=13 A",
            "scope": "finite seeded diagnostic fixtures; not a proof or benchmark accuracy result"},
        "counts": {"clouds": len(clouds), "random_clouds": 12, "sheaf_kinds": len(KINDS),
            "persistent_operators": len(nullities), "checks": sum(record["count"] for record in checks.metrics.values()),
            "maximum_operator_dimension": max(record["maximum_operator_dimension"] for record in clouds),
            "positive_new_edge_comparisons": sum(record["positive_new_edge_comparisons"] for record in clouds)},
        "metrics": checks.metrics, "known_cycle_death": cycle, "clouds": clouds,
        "persistent_nullities": nullities, "spectral_extrema": spectra, "native_identities": native,
        "failures": checks.failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "elapsed_seconds": report["elapsed_seconds"],
                      "max_errors": {name: record["max_absolute_error"] for name, record in checks.metrics.items()},
                      "failures": checks.failures, "output": str(args.output)}, indent=2))
    if checks.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
