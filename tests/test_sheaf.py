"""Scientific validation fixtures; no benchmark result is assumed here."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from psl_flexibility.native_psl import NativePersistentSheafLaplacian
from psl_flexibility.sheaf import AlphaSheaf, STATISTIC_NAMES, persistent_laplacian, spectral_statistics


KINDS = ("identity", "geometric", "center_zero")
SQUARE = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=float)


@pytest.mark.parametrize("kind", KINDS)
def test_sheaf_coboundaries_compose_to_zero(kind: str) -> None:
    points = np.array([[0, 0, 0], [2, 0, 0], [.4, 1.5, 0], [.7, .3, 2], [2, 2, 2]], dtype=float)
    sheaf = AlphaSheaf(points, kind)
    for radius in (0, 1, 1.6, 3, 10):
        d0, d1 = sheaf.coboundary(0, radius), sheaf.coboundary(1, radius)
        np.testing.assert_allclose(d1 @ d0, 0, atol=1e-13)


def test_equilateral_triangle_matches_analytic_center_zero_spectrum() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [.5, np.sqrt(3) / 2, 0]])
    sheaf = AlphaSheaf(points)
    np.testing.assert_allclose(np.linalg.eigvalsh(sheaf.laplacian(0, 1)), [0, 2, 2], atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(sheaf.with_kind("identity").laplacian(0, 1)), [0, 3, 3], atol=1e-12)
    np.testing.assert_allclose(sheaf.laplacian(1, 1), np.eye(3) * 2, atol=1e-12)


def test_alpha_radius_is_ball_radius_and_degree_zero_uses_upper_endpoint() -> None:
    sheaf = AlphaSheaf(np.array([[0, 0, 0], [2, 0, 0]], dtype=float), "identity")
    assert sheaf.simplices(.99, 1) == ()
    assert sheaf.simplices(1, 1) == ((0, 1),)
    np.testing.assert_array_equal(sheaf.laplacian(0, .1, 1), sheaf.laplacian(0, 1))
    np.testing.assert_allclose(np.linalg.eigvalsh(sheaf.laplacian(0, 1)), [0, 2], atol=1e-12)


def test_square_cycle_dies_under_true_persistence() -> None:
    sheaf = AlphaSheaf(SQUARE, "identity")
    assert spectral_statistics(sheaf.laplacian(1, 1.05))[0] == 1
    assert spectral_statistics(sheaf.laplacian(1, 1.05, 1.5))[0] == 0
    # Changing the width changes the operator on X's SAME four edge stalks.
    assert sheaf.laplacian(1, 1.05, 1.5).shape == (4, 4)


@pytest.mark.parametrize("kind", KINDS)
def test_persistent_schur_matches_independent_projection_and_restriction_rank(kind: str) -> None:
    sheaf = AlphaSheaf(SQUARE, kind)
    a, b = 1.05, 1.5
    actual = sheaf.laplacian(1, a, b)
    np.testing.assert_allclose(actual, sheaf.laplacian_nullspace(1, a, b), atol=1e-12)
    np.testing.assert_allclose(actual, sheaf.laplacian(1, a, b, use_blocks=False), atol=1e-12)
    d0x, d1x = sheaf.coboundary(0, a), sheaf.coboundary(1, a)
    d0y, d1y = sheaf.coboundary(0, b), sheaf.coboundary(1, b)
    hx = linalg.null_space(np.vstack([d1x, d0x.T]))
    hy = linalg.null_space(np.vstack([d1y, d0y.T]))
    lookup = {edge: i for i, edge in enumerate(sheaf.simplices(b, 1))}
    old = [lookup[edge] for edge in sheaf.simplices(a, 1)]
    restriction = hx.T @ hy[old]
    rank = np.count_nonzero(linalg.svdvals(restriction) > 1e-8)
    assert spectral_statistics(actual)[0] == rank


def test_geometric_gauge_changes_spectrum_but_not_persistent_nullity() -> None:
    base = AlphaSheaf(SQUARE, "identity")
    weighted = base.with_kind("geometric")
    assert weighted._filtration is base._filtration
    assert not np.allclose(base.laplacian(1, 1.05), weighted.laplacian(1, 1.05))
    for a, b in ((.5, .5), (1.05, 1.05), (1.05, 1.5), (1.5, 1.5)):
        assert spectral_statistics(base.laplacian(1, a, b))[0] == spectral_statistics(weighted.laplacian(1, a, b))[0]


@pytest.mark.parametrize("kind", KINDS)
def test_rigid_motion_and_neighbor_permutation_preserve_spectra(kind: str) -> None:
    rng = np.random.default_rng(1842)
    points = rng.normal(size=(9, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    shifted = points @ rotation + [20, -13, 7]
    order = np.r_[0, rng.permutation(np.arange(1, len(points)))]
    original = AlphaSheaf(points, kind)
    moved, reordered = AlphaSheaf(shifted, kind), AlphaSheaf(points[order], kind)
    for degree, a, b in ((0, .8, 1.5), (1, .8, 1.5), (1, 1.5, 1.5)):
        expected = np.linalg.eigvalsh(original.laplacian(degree, a, b))
        np.testing.assert_allclose(np.linalg.eigvalsh(moved.laplacian(degree, a, b)), expected, atol=2e-10)
        np.testing.assert_allclose(np.linalg.eigvalsh(reordered.laplacian(degree, a, b)), expected, atol=2e-10)


def test_edge_and_triangle_orientation_changes_preserve_persistent_operator() -> None:
    sheaf = AlphaSheaf(SQUARE, "geometric")
    a, b = 1.05, 1.5
    lower, upper = sheaf.coboundary(0, a), sheaf.coboundary(1, b)
    lookup = {edge: i for i, edge in enumerate(sheaf.simplices(b, 1))}
    old = np.array([lookup[edge] for edge in sheaf.simplices(a, 1)])
    edge_signs = np.where(np.arange(upper.shape[1]) % 2, -1., 1.)
    face_signs = np.where(np.arange(upper.shape[0]) % 2, 1., -1.)
    reoriented = persistent_laplacian(lower * edge_signs[old, None], upper * edge_signs * face_signs[:, None], old)
    expected = sheaf.laplacian(1, a, b) * edge_signs[old, None] * edge_signs[old]
    np.testing.assert_allclose(reoriented, expected, atol=1e-12)


def test_singleton_collinear_and_duplicate_handling() -> None:
    singleton = AlphaSheaf(np.array([[0., 0, 0]]))
    assert singleton.affine_dimension == 0
    np.testing.assert_array_equal(spectral_statistics(singleton.laplacian(0, 1)), [1, 0, 0, 0, 0, 0])
    assert singleton.laplacian(1, 1).shape == (0, 0)
    line = AlphaSheaf(np.array([[0, 0, 0], [2, 0, 0], [5, 0, 0]], dtype=float), "identity")
    assert line.affine_dimension == 1
    assert line.simplices(1.1, 1) == ((0, 1),)
    assert line.simplices(2, 2) == ()
    assert spectral_statistics(line.laplacian(0, 1.1))[0] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        AlphaSheaf(np.zeros((2, 3)))


def test_statistics_are_scale_aware_and_reject_significant_negative_eigenvalues() -> None:
    assert STATISTIC_NAMES == ("nullity", "min_positive", "max", "mean", "median", "std")
    np.testing.assert_array_equal(spectral_statistics(np.empty((0, 0))), np.zeros(6))
    np.testing.assert_allclose(spectral_statistics(np.array([-1e-24, 2e-12, 4e-12])), [1, 2e-12, 4e-12, 3e-12, 3e-12, 1e-12])
    with pytest.raises(FloatingPointError):
        spectral_statistics(np.array([-1e-5, 1.]))
    with pytest.raises(ValueError, match="symmetric"):
        spectral_statistics(np.array([[1., 2], [0, 1]]))


@pytest.mark.parametrize("a,b", [(-1, 2), (2, 1), (np.nan, 2), (1, np.inf)])
def test_invalid_radius_pairs_raise(a: float, b: float) -> None:
    with pytest.raises(ValueError):
        AlphaSheaf(SQUARE).laplacian(1, a, b)


@pytest.mark.parametrize("labeled", [False, True])
def test_native_universal_center_graph_spectral_identities(labeled: bool) -> None:
    # This tests an analytical characterization of the historical comparator.
    points = np.array([[0, 0, 0], [.8, 0, 0], [-.8, 0, 0], [0, .8, 0], [.5, .5, 0]], dtype=float)
    n = len(points)
    adjacency = np.linalg.norm(points[1:, None] - points[None, 1:], axis=2) <= 1
    np.fill_diagonal(adjacency, False)
    neighbor_laplacian = np.diag(adjacency.sum(axis=1)) - adjacency.astype(float)
    mu = np.linalg.eigvalsh(neighbor_laplacian)
    multiplier = 2 if labeled else 1
    expected = np.sort(np.r_[0, multiplier * n, multiplier + mu[1:]])
    historical = NativePersistentSheafLaplacian(points, charges=np.r_[0, np.ones(n-1)], radius_list=[1], constant=not labeled)
    spectrum = np.linalg.eigvalsh(historical.psl_0()[0])
    np.testing.assert_allclose(spectrum, expected, atol=1e-12)
    stats = spectral_statistics(spectrum)
    assert stats[0] == 1
    assert stats[2] == pytest.approx(multiplier * n)
    assert stats[3] == pytest.approx(2 * multiplier + adjacency.sum() / (n - 1))
