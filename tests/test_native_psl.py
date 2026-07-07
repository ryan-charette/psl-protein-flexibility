from __future__ import annotations

import numpy as np

from psl_flexibility.native_psl import NativePersistentSheafLaplacian


def test_two_point_constant_laplacian_has_expected_spectrum() -> None:
    psl = NativePersistentSheafLaplacian(
        pts=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        radius_list=np.array([1.1]),
        constant=True,
    )
    psl.build_filtration()
    psl.build_simplicial_pair()
    psl.build_matrices()

    eigvals = np.linalg.eigvalsh(psl.psl_0()[0])

    np.testing.assert_allclose(eigvals, np.array([0.0, 2.0]), atol=1e-12)


def test_center_labels_change_edge_weights() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    constant = NativePersistentSheafLaplacian(points, radius_list=np.array([2.0]), constant=True)
    labeled = NativePersistentSheafLaplacian(
        points,
        charges=np.array([0.0, 1.0, 1.0]),
        radius_list=np.array([2.0]),
        constant=False,
    )

    assert not np.allclose(constant.psl_0()[0], labeled.psl_0()[0])


def test_degree_one_and_two_matrices_are_available_for_filled_triangle() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    psl = NativePersistentSheafLaplacian(points, radius_list=np.array([2.0]), constant=True)

    assert psl.psl_1()[0].shape == (3, 3)
    assert psl.psl_2()[0].shape == (1, 1)
