"""Validation of the MDS reconstruction diagnostic.

These tests confirm that the pipeline recovers coordinates that were used to
build the correlation matrix. That is circular by construction and is a test of
the code, not a physics result.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvacuum.correlators import correlation_matrix, correlator_free
from qvacuum.geometry import Sphere, pair_distances
from qvacuum.reconstruct import (
    classical_mds,
    distance_from_correlator,
    procrustes_rmse,
)

RADIUS = 0.5e-6


def test_massless_inversion_is_exact():
    r = np.array([1e-9, 1e-8, 1e-7])
    g = correlator_free(r)
    assert np.allclose(distance_from_correlator(g), r, rtol=1e-12)


def test_mds_recovers_known_coordinates():
    sphere = Sphere(RADIUS)
    points = sphere.lattice_points(3)
    d = pair_distances(points)
    coords, eigenvalues = classical_mds(d, dim=3)
    assert procrustes_rmse(coords, points) < 1e-9 * RADIUS
    assert eigenvalues[2] > 1e6 * abs(eigenvalues[3])


def test_full_pipeline_massless():
    """Correlator -> distances -> coordinates, for a free massless field."""
    sphere = Sphere(RADIUS)
    points = sphere.lattice_points(3)
    a = sphere.lattice_spacing(3)
    g = correlation_matrix(points, regulator=a)
    d = distance_from_correlator(g)
    np.fill_diagonal(d, 0.0)
    coords, _ = classical_mds(d, dim=3)
    assert procrustes_rmse(coords, points) < 1e-6 * RADIUS


def test_mds_eigenvalue_gap_identifies_dimension():
    points = Sphere(RADIUS).lattice_points(4)
    _, eigenvalues = classical_mds(pair_distances(points), dim=3)
    assert np.sum(eigenvalues > 1e-6 * eigenvalues[0]) == 3


def test_procrustes_invariant_under_rotation():
    rng = np.random.default_rng(0)
    points = Sphere(RADIUS).lattice_points(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert procrustes_rmse(points @ q, points) == pytest.approx(0.0, abs=1e-18)
