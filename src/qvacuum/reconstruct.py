"""Recovery of spatial structure from a correlation matrix.

This module is a diagnostic, not a result. For a free field in flat space the
correlator is by construction a monotone function of separation, so inverting
it and applying multidimensional scaling returns the coordinates that were
used to build the matrix in the first place. Success here validates the code;
it demonstrates nothing about the physics.

The informative use is the converse: establishing where the recovery fails as
the mass, the cutoff, the coarse-graining scale or the noise level are varied.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .correlators import correlator_free


def distance_from_correlator(g: np.ndarray, mass: float = 0.0,
                             r_bracket: tuple[float, float] = (1e-12, 1e-3),
                             ) -> np.ndarray:
    """Invert G(r) to obtain an inferred separation [m].

    Closed form for the massless case, G = 1 / (4 pi^2 r^2). For m > 0 the
    inversion is numerical and performed element by element, which is slow;
    vectorise via interpolation before using it on large matrices.
    """
    g = np.asarray(g, dtype=float)
    if mass == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(1.0 / (4.0 * np.pi**2 * g))

    flat = g.ravel()
    out = np.empty_like(flat)
    for i, value in enumerate(flat):
        if not np.isfinite(value) or value <= 0:
            out[i] = np.nan
            continue
        try:
            out[i] = brentq(
                lambda r, target=value: (
                    float(correlator_free(np.array([r]), mass)[0]) - target
                ),
                *r_bracket,
            )
        except ValueError:
            out[i] = np.nan
    return out.reshape(g.shape)


def inferred_distances(g: np.ndarray, floor: float | None = None) -> np.ndarray:
    """Distances inferred from a correlation matrix under the free-space law.

    Applies d = 1 / (2 pi sqrt(G)) elementwise, which is exact for an unbounded
    massless field and is a misspecified model for any correlator that is not a
    monotone function of separation alone. Non-positive entries have no image
    under the inversion and are assigned ``floor``, since a vanishing
    correlation corresponds to infinite inferred separation.

    Applying this to a boundary-aware correlator is the point: the failure of
    the inversion is the measurement.
    """
    g = np.asarray(g, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sqrt(1.0 / (4.0 * np.pi**2 * np.where(g > 0, g, np.nan)))
    if floor is None:
        floor = np.nanmax(d[np.isfinite(d)]) if np.any(np.isfinite(d)) else 0.0
    d = np.nan_to_num(d, nan=floor, posinf=floor)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def euclidean_defect(eigenvalues: np.ndarray) -> float:
    """Negative eigenvalue mass of the doubly centred matrix, as a fraction.

    Zero when the inferred distances embed exactly in a Euclidean space of some
    dimension. A growing value means the inferred distance matrix is not a
    metric of any Euclidean geometry, which is the sharpest single indicator
    that the inversion has broken down.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    positive = eigenvalues[eigenvalues > 0].sum()
    if positive == 0:
        return np.inf
    return float(abs(eigenvalues[eigenvalues < 0].sum()) / positive)


def classical_mds(distances: np.ndarray, dim: int = 3
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Classical multidimensional scaling.

    Returns
    -------
    coords
        Array of shape (n, dim), the embedding.
    eigenvalues
        Full descending eigenvalue spectrum of the doubly centred matrix. The
        gap after the third entry is the evidence for three-dimensionality;
        report it rather than assuming it.
    """
    d2 = np.asarray(distances, dtype=float) ** 2
    n = d2.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.clip(eigenvalues[:dim], 0.0, None)
    coords = eigenvectors[:, :dim] * np.sqrt(positive)
    return coords, eigenvalues


def procrustes_rmse(recovered: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square residual [m] after optimal rigid alignment.

    Translation and rotation are removed; reflection is permitted, since MDS
    fixes neither handedness nor orientation. Scale is not fitted, so a
    non-zero result reflects genuine distortion rather than an arbitrary
    normalisation.
    """
    a = recovered - recovered.mean(axis=0)
    b = truth - truth.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rotation - b) ** 2, axis=1))))
