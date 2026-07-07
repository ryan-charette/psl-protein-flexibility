"""Native persistent sheaf Laplacian utilities.

The original project used an external research script named ``PSL.py``.  This
module provides the small subset of that interface needed by this repository:
construct a local distance-threshold simplicial complex and return degree 0, 1,
and 2 Laplacian matrices at one or more radii.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


_DIST_TOL = 1e-12


@dataclass(frozen=True)
class _ComplexAtRadius:
    radius: float
    edges: tuple[tuple[int, int], ...]
    edge_weights: np.ndarray
    incidence: np.ndarray


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2:
        raise ValueError("pts must be a two-dimensional coordinate array.")
    if points.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    finite = labels[np.isfinite(labels)]
    if finite.size == 0:
        return np.zeros_like(labels, dtype=float)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if np.isclose(lo, hi):
        return np.zeros_like(labels, dtype=float)
    return (labels - lo) / (hi - lo)


class NativePersistentSheafLaplacian:
    """Small native PSL implementation for local protein descriptors.

    Parameters mirror the upstream research class used by the original scripts.
    ``filtration_type="alpha"`` is accepted as a compatibility alias; this
    implementation uses a Euclidean distance-threshold, Vietoris-Rips style
    complex because the maintained software path must be self-contained.
    """

    def __init__(
        self,
        pts: Sequence[Sequence[float]] | np.ndarray,
        charges: Sequence[float] | np.ndarray | None = None,
        filtration_type: str = "alpha",
        radius_list: Sequence[float] | np.ndarray | None = None,
        p: float = 0.0,
        constant: bool = True,
        scale: bool = False,
    ) -> None:
        self.pts = np.asarray(pts, dtype=float)
        if self.pts.ndim != 2:
            raise ValueError("pts must be a two-dimensional coordinate array.")
        if not np.all(np.isfinite(self.pts)):
            raise ValueError("pts contains non-finite coordinates.")

        self.filtration_type = filtration_type.lower()
        if self.filtration_type not in {"alpha", "rips", "distance"}:
            raise ValueError("filtration_type must be 'alpha', 'rips', or 'distance'.")

        if radius_list is None:
            self.radius_list = np.array([np.inf], dtype=float)
        else:
            self.radius_list = np.asarray(radius_list, dtype=float).reshape(-1)
        if self.radius_list.size == 0:
            raise ValueError("radius_list must contain at least one radius.")
        if np.any(self.radius_list < 0):
            raise ValueError("radius_list cannot contain negative radii.")

        self.p = float(p)
        if self.p < 0:
            raise ValueError("p must be non-negative.")
        self.constant = bool(constant)
        self.scale = bool(scale)

        if charges is None:
            labels = np.zeros(self.pts.shape[0], dtype=float)
        else:
            labels = np.asarray(charges, dtype=float).reshape(-1)
            if labels.shape[0] != self.pts.shape[0]:
                raise ValueError("charges must have one entry per point.")
            if not np.all(np.isfinite(labels)):
                raise ValueError("charges contains non-finite values.")
        self.charges = _normalize_labels(labels) if self.scale else labels

        self._distances: np.ndarray | None = None
        self._complexes: list[_ComplexAtRadius] | None = None
        self._l0: list[np.ndarray] | None = None
        self._l1: list[np.ndarray] | None = None
        self._l2: list[np.ndarray] | None = None

    def build_filtration(self) -> None:
        """Precompute pairwise distances for the requested filtration radii."""

        self._distances = _pairwise_distances(self.pts)

    def build_simplicial_pair(self) -> None:
        """Build native distance-threshold complexes at each radius."""

        if self._distances is None:
            self.build_filtration()
        assert self._distances is not None
        self._complexes = [self._build_complex(float(radius), self._distances) for radius in self.radius_list]

    def build_matrices(self) -> None:
        """Build degree-zero Laplacian matrices."""

        if self._complexes is None:
            self.build_simplicial_pair()
        assert self._complexes is not None
        self._l0 = [
            complex_at_radius.incidence @ complex_at_radius.incidence.T
            for complex_at_radius in self._complexes
        ]

    def psl_0(self) -> list[np.ndarray]:
        """Return degree-zero sheaf Laplacian matrices, one per radius."""

        if self._l0 is None:
            self.build_matrices()
        assert self._l0 is not None
        return self._l0

    def psl_1(self) -> list[np.ndarray]:
        """Return degree-one Hodge-style Laplacian matrices, one per radius."""

        if self._l1 is None:
            if self._complexes is None:
                self.build_simplicial_pair()
            assert self._complexes is not None
            self._l1 = []
            for complex_at_radius in self._complexes:
                b1 = complex_at_radius.incidence
                b2 = self._triangle_boundary(complex_at_radius)
                self._l1.append(b1.T @ b1 + b2 @ b2.T)
        return self._l1

    def psl_2(self) -> list[np.ndarray]:
        """Return degree-two Hodge-style Laplacian matrices, one per radius."""

        if self._l2 is None:
            if self._complexes is None:
                self.build_simplicial_pair()
            assert self._complexes is not None
            self._l2 = []
            for complex_at_radius in self._complexes:
                b2 = self._triangle_boundary(complex_at_radius)
                self._l2.append(b2.T @ b2)
        return self._l2

    def _build_complex(self, radius: float, distances: np.ndarray) -> _ComplexAtRadius:
        n_points = self.pts.shape[0]
        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                distance = float(distances[i, j])
                if distance <= radius + _DIST_TOL:
                    edges.append((i, j))
                    weights.append(self._edge_weight(i, j, distance))

        incidence = np.zeros((n_points, len(edges)), dtype=float)
        for col, (edge, weight) in enumerate(zip(edges, weights, strict=True)):
            i, j = edge
            root_weight = float(np.sqrt(weight))
            incidence[i, col] = -root_weight
            incidence[j, col] = root_weight

        return _ComplexAtRadius(
            radius=radius,
            edges=tuple(edges),
            edge_weights=np.asarray(weights, dtype=float),
            incidence=incidence,
        )

    def _edge_weight(self, i: int, j: int, distance: float) -> float:
        weight = 1.0
        if not self.constant:
            label_gap = abs(float(self.charges[i]) - float(self.charges[j]))
            weight += label_gap
        if self.p > 0:
            weight /= 1.0 + self.p * distance
        return weight

    def _triangle_boundary(self, complex_at_radius: _ComplexAtRadius) -> np.ndarray:
        edges = complex_at_radius.edges
        if not edges:
            return np.zeros((0, 0), dtype=float)

        edge_lookup = {edge: idx for idx, edge in enumerate(edges)}
        triangles: list[tuple[int, int, int]] = []
        n_points = self.pts.shape[0]
        for i, j, k in combinations(range(n_points), 3):
            if (i, j) in edge_lookup and (i, k) in edge_lookup and (j, k) in edge_lookup:
                triangles.append((i, j, k))

        boundary = np.zeros((len(edges), len(triangles)), dtype=float)
        for col, (i, j, k) in enumerate(triangles):
            boundary[edge_lookup[(j, k)], col] = 1.0
            boundary[edge_lookup[(i, k)], col] = -1.0
            boundary[edge_lookup[(i, j)], col] = 1.0
        return boundary


PSL = NativePersistentSheafLaplacian

__all__ = ["NativePersistentSheafLaplacian", "PSL"]
