from __future__ import annotations

import math

import numpy as np

from psl_flexibility.metrics import rmse, safe_pearson, safe_spearman, within_group_zscore


def test_safe_pearson_matches_perfect_correlation() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert safe_pearson(x, 2 * x + 1) > 0.999999


def test_safe_pearson_returns_nan_for_constant_input() -> None:
    assert math.isnan(safe_pearson(np.ones(5), np.arange(5)))


def test_safe_spearman_handles_monotone_relationship() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert safe_spearman(x, y) > 0.999999


def test_rmse_ignores_nan_pairs() -> None:
    x = np.array([1.0, 2.0, np.nan])
    y = np.array([1.0, 4.0, 9.0])
    assert rmse(x, y) == math.sqrt(2.0)


def test_within_group_zscore() -> None:
    z = within_group_zscore(np.array([1.0, 2.0, 3.0]))
    assert abs(float(np.mean(z))) < 1e-12
    assert abs(float(np.std(z)) - 1.0) < 1e-12
