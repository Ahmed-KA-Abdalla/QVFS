"""Validation of sphere geometry and lattice construction."""

from __future__ import annotations

import numpy as np
import pytest

from qvacuum.geometry import Sphere, pair_distances

RADIUS = 0.5e-6


def test_volume_and_area():
    sphere = Sphere(RADIUS)
    assert sphere.volume == pytest.approx(5.235_987_756e-19, rel=1e-9)
    assert sphere.surface_area == pytest.approx(3.141_592_653e-12, rel=1e-9)
    assert sphere.surface_area / sphere.volume == pytest.approx(3 / RADIUS)


def test_default_radius_is_half_micron():
    assert Sphere().radius == 0.5e-6


def test_all_lattice_points_inside():
    sphere = Sphere(RADIUS)
    points = sphere.lattice_points(6)
    assert np.all(np.sum(points**2, axis=1) <= RADIUS**2 * (1 + 1e-9))


def test_lattice_count_approaches_volume_ratio():
    """Site count converges to V / a^3 as the lattice is refined."""
    sphere = Sphere(RADIUS)
    for n, tol in ((6, 0.10), (12, 0.05)):
        a = sphere.lattice_spacing(n)
        n_sites = len(sphere.lattice_points(n))
        assert n_sites == pytest.approx(sphere.volume / a**3, rel=tol)


def test_lattice_rejects_zero_resolution():
    with pytest.raises(ValueError):
        Sphere(RADIUS).lattice_points(0)


def test_pair_distances_properties():
    points = Sphere(RADIUS).lattice_points(3)
    d = pair_distances(points)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert d.max() <= 2 * RADIUS * (1 + 1e-9)
