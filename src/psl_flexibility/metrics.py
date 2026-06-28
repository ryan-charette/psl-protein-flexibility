"""Metric helpers shared by tests and analysis scripts."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _finite_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation, or NaN when the inputs are too small or constant."""

    x, y = _finite_pairs(a, b)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(pearsonr(x, y).statistic)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Return Spearman correlation, or NaN when the inputs are too small or constant."""

    x, y = _finite_pairs(a, b)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Return root mean squared error over finite paired entries."""

    x, y = _finite_pairs(a, b)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((x - y) ** 2)))


def within_group_zscore(values: np.ndarray) -> np.ndarray:
    """Return z-scored values, using zeros for constant or non-finite groups."""

    x = np.asarray(values, dtype=float)
    scale = float(np.nanstd(x))
    if scale == 0.0 or not np.isfinite(scale):
        return np.zeros_like(x, dtype=float)
    return (x - float(np.nanmean(x))) / scale
