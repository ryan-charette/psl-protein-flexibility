"""Alpha-filtration sheaf spectra used by the research experiments.

The implementation follows the definitions in Wei and Wei, *Persistent sheaf
Laplacians*, arXiv:2112.10906, Sections 2.3 and 3.2.  It is an independent
implementation of the formulas, not a copy of the authors' software.

All radii are alpha-ball radii in Angstroms (GUDHI stores their squares).
Coordinates and the center, at row zero, remain fixed across the filtration.
The three constructions have real one-dimensional stalks with their usual
inner products. ``identity`` has identity restrictions; ``geometric`` has
the published geometric restrictions and all labels one; ``center_zero``
changes only the center label to zero. Distances in F are divided by 1 A.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import linalg


STATISTIC_NAMES = ("nullity", "min_positive", "max", "mean", "median", "std")
SPECTRAL_STATISTIC_NAMES = STATISTIC_NAMES
DEFAULT_RTOL = 1e-10
_KINDS = {"identity", "geometric", "center_zero"}


def _kind(value: str) -> str:
    value = {"all1": "geometric", "constant": "identity"}.get(value, value)
    if value not in _KINDS:
        raise ValueError(f"Unknown sheaf kind {value!r}; expected {sorted(_KINDS)}.")
    return value


def _radius(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Alpha radius must be finite and nonnegative.")
    return value


def _threshold(values: np.ndarray, rtol: float, atol: float = 0.0) -> float:
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("Numerical tolerances must be finite and nonnegative.")
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    # The relative criterion remains meaningful for geometrically small matrices.
    return atol + max(rtol, 64 * np.finfo(float).eps * max(1, values.size)) * scale


def spectral_statistics(values: np.ndarray, *, rtol: float = DEFAULT_RTOL, atol: float = 0.0) -> np.ndarray:
    """Return six statistics of eigenvalues, or of a supplied symmetric matrix.

    Order is ``STATISTIC_NAMES``. The five non-nullity statistics use positive
    eigenvalues only. Empty matrices have six zeros. Negative eigenvalues beyond
    a scale-relative roundoff threshold raise; no physical negative mode is
    silently clipped. The threshold and method must be recorded with a run.
    """
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("Spectrum or matrix contains non-finite values.")
    if array.ndim == 2:
        if array.shape[0] != array.shape[1]:
            raise ValueError("Laplacian must be square.")
        symmetry_tol = _threshold(array.reshape(-1), rtol, atol)
        if array.size and np.max(np.abs(array - array.T)) > symmetry_tol:
            raise ValueError("Laplacian must be symmetric.")
        array = linalg.eigvalsh((array + array.T) / 2, check_finite=False)
    elif array.ndim != 1:
        raise ValueError("Expected a one-dimensional spectrum or a square matrix.")
    tolerance = _threshold(array, rtol, atol)
    if np.any(array < -tolerance):
        raise FloatingPointError(f"Laplacian has a negative eigenvalue below {-tolerance:g}.")
    positive = array[array > tolerance]
    nullity = float(array.size - positive.size)
    if not positive.size:
        return np.array([nullity, 0, 0, 0, 0, 0], dtype=float)
    return np.array(
        [nullity, positive.min(), positive.max(), positive.mean(), np.median(positive), positive.std()],
        dtype=float,
    )


@dataclass(frozen=True)
class _Filtration:
    points: np.ndarray
    cells: tuple[tuple[tuple[int, ...], ...], ...]
    births: tuple[np.ndarray, ...]
    factors: tuple[np.ndarray, ...]
    affine_dimension: int


def _build_alpha(points: np.ndarray) -> _Filtration:
    """Build once; handle affine ranks 0 and 1 exactly without perturbations."""
    coordinates = np.array(points, dtype=float, copy=True)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0:
        raise ValueError("points must be a nonempty n x 3 coordinate array.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("points contains non-finite coordinates.")
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("Duplicate coordinates require explicit preprocessing; they are not merged or jittered.")
    centered = coordinates - coordinates[0]
    _, singular, directions = linalg.svd(centered, full_matrices=False, check_finite=False)
    rank_tol = (singular[0] if singular.size else 0.0) * max(centered.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > rank_tol))
    entries: list[tuple[tuple[int, ...], float]] = [((i,), 0.0) for i in range(len(coordinates))]
    if rank == 1:
        projected = centered @ directions[0]
        order = np.argsort(projected, kind="stable")
        for left, right in zip(order[:-1], order[1:], strict=True):
            cell = tuple(sorted((int(left), int(right))))
            distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
            entries.append((cell, (distance / 2) ** 2))
    elif rank >= 2:
        import gudhi

        # Projection removes only numerically zero affine directions, never jitter.
        alpha_points = centered @ directions[:rank].T
        alpha = gudhi.AlphaComplex(points=alpha_points.tolist(), precision="safe")
        tree = alpha.create_simplex_tree()
        entries = [(tuple(sorted(cell)), float(birth)) for cell, birth in tree.get_filtration() if len(cell) <= 3]
    cells: list[tuple[tuple[int, ...], ...]] = []
    births: list[np.ndarray] = []
    factors: list[np.ndarray] = []
    for degree in range(3):
        records = sorted((cell, birth) for cell, birth in entries if len(cell) == degree + 1)
        cells.append(tuple(cell for cell, _ in records))
        births.append(np.asarray([birth for _, birth in records], dtype=float))
        factors.append(np.asarray([
            np.prod([np.linalg.norm(coordinates[i] - coordinates[j]) for i, j in combinations(cell, 2)])
            for cell, _ in records
        ], dtype=float))
    coordinates.setflags(write=False)
    for array in (*births, *factors):
        array.setflags(write=False)
    return _Filtration(coordinates, tuple(cells), tuple(births), tuple(factors), rank)


def persistent_laplacian(
    lower: np.ndarray,
    upper: np.ndarray,
    old_indices: Sequence[int] | np.ndarray,
    *,
    method: str = "schur",
    rtol: float = DEFAULT_RTOL,
) -> np.ndarray:
    """Persistent cochain Laplacian from D_X^(q-1) and D_Y^q.

    ``old_indices`` locates X's q-cells, in their X ordering, among the columns
    of ``upper``. The lower term is formed at X. The upper Schur complement
    is taken in upper.T @ upper, NOT in the full ordinary Laplacian of Y.
    ``nullspace`` is an independent validation path using orthogonal projection
    onto ker(D_new.T), equivalent to the persistent admissible cochain space.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    old = np.asarray(old_indices, dtype=int)
    if lower.ndim != 2 or upper.ndim != 2 or old.ndim != 1 or lower.shape[0] != old.size:
        raise ValueError("Coboundary shapes and old-cell indices disagree.")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Coboundaries must be finite.")
    if len(np.unique(old)) != len(old) or np.any(old < 0) or np.any(old >= upper.shape[1]):
        raise ValueError("old_indices must be unique valid column indices.")
    _threshold(np.array([], dtype=float), rtol)
    new_mask = np.ones(upper.shape[1], dtype=bool)
    new_mask[old] = False
    old_part = upper[:, old]
    new_part = upper[:, new_mask]
    if method == "schur":
        result = old_part.T @ old_part
        if new_part.shape[1] and new_part.shape[0]:
            cross = old_part.T @ new_part
            gram = new_part.T @ new_part
            eigenvalues, eigenvectors = linalg.eigh(gram, check_finite=False)
            tolerance = _threshold(eigenvalues, rtol)
            if np.any(eigenvalues < -tolerance):
                raise FloatingPointError("The Gram matrix is not positive semidefinite within tolerance.")
            keep = eigenvalues > tolerance
            if np.any(keep):
                factor = (cross @ eigenvectors[:, keep]) / np.sqrt(eigenvalues[keep])
                result = result - factor @ factor.T
    elif method == "nullspace":
        # Gram-eigenvalue rtol corresponds to sqrt(rtol) for singular values.
        basis = linalg.null_space(new_part.T, rcond=np.sqrt(rtol))
        projected = basis.T @ old_part
        result = projected.T @ projected
    else:
        raise ValueError("method must be 'schur' or 'nullspace'.")
    result = result + lower @ lower.T
    return (result + result.T) / 2


class AlphaSheaf:
    """A fixed local point cloud and compatible sheaves on its alpha filtration.

    The target residue is point zero. ``with_kind`` reuses triangulation and
    geometric factors. Rank-one clouds use the exact adjacent-edge alpha
    complex; rank-two clouds are projected onto their affine plane. Duplicate
    points raise. No random or silent perturbation is performed. Co-spherical
    Delaunay ties follow GUDHI's deterministic choice for the supplied ordering.
    """

    def __init__(self, points: np.ndarray, kind: str = "center_zero", *, rtol: float = DEFAULT_RTOL) -> None:
        self.kind = _kind(kind)
        _threshold(np.array([], dtype=float), rtol)
        self.rtol = float(rtol)
        self._filtration = _build_alpha(points)
        self._operators: dict[int, np.ndarray] = {}
        self._restricted: dict[tuple[int, float], np.ndarray] = {}

    @property
    def points(self) -> np.ndarray:
        return self._filtration.points

    @property
    def affine_dimension(self) -> int:
        return self._filtration.affine_dimension

    def with_kind(self, kind: str) -> AlphaSheaf:
        result = object.__new__(AlphaSheaf)
        result.kind = _kind(kind)
        result.rtol = self.rtol
        result._filtration = self._filtration
        result._operators = {}
        result._restricted = {}
        return result

    def _indices(self, radius: float, dimension: int) -> np.ndarray:
        if dimension not in {0, 1, 2}:
            raise ValueError("Only simplex dimensions 0, 1 and 2 are exposed.")
        squared = _radius(radius) ** 2
        births = self._filtration.births[dimension]
        tolerance = 64 * np.finfo(float).eps * np.maximum(np.abs(births), squared)
        return np.flatnonzero(births <= squared + tolerance)

    def simplices(self, radius: float, dimension: int) -> tuple[tuple[int, ...], ...]:
        """Canonical ascending vertex tuples, in lexicographic order."""
        return tuple(self._filtration.cells[dimension][i] for i in self._indices(radius, dimension))

    def coboundary(self, degree: int, radius: float) -> np.ndarray:
        """D^degree, rows cofaces and columns faces, with inherited orientations."""
        if degree not in {0, 1}:
            raise ValueError("Only coboundaries of degrees 0 and 1 are needed.")
        radius = _radius(radius)
        key = (degree, radius)
        if key not in self._restricted:
            if degree not in self._operators:
                faces = self._filtration.cells[degree]
                cofaces = self._filtration.cells[degree + 1]
                lookup = {cell: i for i, cell in enumerate(faces)}
                matrix = np.zeros((len(cofaces), len(faces)), dtype=float)
                for row, coface in enumerate(cofaces):
                    for omitted, vertex in enumerate(coface):
                        face = coface[:omitted] + coface[omitted + 1:]
                        column = lookup[face]
                        restriction = 1.0
                        if self.kind != "identity":
                            restriction = self._filtration.factors[degree][column] / self._filtration.factors[degree + 1][row]
                            if self.kind == "center_zero" and vertex == 0:
                                restriction = 0.0
                        matrix[row, column] = (-1.0 if omitted % 2 else 1.0) * restriction
                matrix.setflags(write=False)
                self._operators[degree] = matrix
            matrix = self._operators[degree][np.ix_(self._indices(radius, degree + 1), self._indices(radius, degree))]
            matrix.setflags(write=False)
            self._restricted[key] = matrix
        return self._restricted[key]

    def laplacian(self, degree: int, a: float, b: float | None = None, *, use_blocks: bool = True) -> np.ndarray:
        """L_degree^(a,b); b defaults to a. Width is b-a, never a weight scale."""
        return self._laplacian(degree, a, b, method="schur", use_blocks=use_blocks)

    def laplacian_nullspace(self, degree: int, a: float, b: float | None = None) -> np.ndarray:
        """Independent persistent projection implementation for small validation cases."""
        return self._laplacian(degree, a, b, method="nullspace", use_blocks=False)

    def _laplacian(self, degree: int, a: float, b: float | None, *, method: str, use_blocks: bool) -> np.ndarray:
        a = _radius(a)
        b = a if b is None else _radius(b)
        if b < a:
            raise ValueError("The upper radius b must be at least a.")
        if degree == 0:
            d0 = self.coboundary(0, b)
            return d0.T @ d0
        if degree != 1:
            raise ValueError("Only Laplacian degrees 0 and 1 are supported.")
        lower = self.coboundary(0, a)
        upper = self.coboundary(1, b)
        edges_x, edges_y = self.simplices(a, 1), self.simplices(b, 1)
        edge_lookup = {edge: i for i, edge in enumerate(edges_y)}
        old = np.asarray([edge_lookup[edge] for edge in edges_x], dtype=int)
        if not use_blocks or self.kind != "center_zero":
            return persistent_laplacian(lower, upper, old, method=method, rtol=self.rtol)
        # Zero label removes all cross-block coboundary entries. This is an
        # orthogonal direct sum, so splitting preserves nonzero spectra too.
        result = np.zeros((len(edges_x), len(edges_x)), dtype=float)
        triangles_y = self.simplices(b, 2)
        vertices_x = self.simplices(a, 0)
        for contains_center in (False, True):
            ix = np.asarray([i for i, cell in enumerate(edges_x) if (0 in cell) == contains_center], dtype=int)
            if not ix.size:
                continue
            iy = np.asarray([i for i, cell in enumerate(edges_y) if (0 in cell) == contains_center], dtype=int)
            it = np.asarray([i for i, cell in enumerate(triangles_y) if (0 in cell) == contains_center], dtype=int)
            iv = np.asarray([i for i, cell in enumerate(vertices_x) if (0 in cell) == contains_center], dtype=int)
            local_lookup = {int(global_index): i for i, global_index in enumerate(iy)}
            local_old = [local_lookup[int(old[i])] for i in ix]
            block = persistent_laplacian(
                lower[np.ix_(ix, iv)], upper[np.ix_(it, iy)], local_old, method=method, rtol=self.rtol,
            )
            result[np.ix_(ix, ix)] = block
        return result


__all__ = [
    "AlphaSheaf", "DEFAULT_RTOL", "STATISTIC_NAMES", "SPECTRAL_STATISTIC_NAMES",
    "persistent_laplacian", "spectral_statistics",
]
