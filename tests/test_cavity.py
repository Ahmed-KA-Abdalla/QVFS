"""Validation of the boundary-aware cavity correlator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spherical_jn

from qvacuum.cavity import CavityField, free_correlator_regulated
from qvacuum.geometry import Sphere
from qvacuum.modes import enumerate_modes

RADIUS = 0.5e-6


@pytest.fixture(scope="module")
def cavity() -> CavityField:
    return CavityField(radius=RADIUS, regulator_scale=40.0 / RADIUS)


def test_mode_functions_are_normalised(cavity):
    """Integral of u^2 r^2 over the ball is unity for each mode."""
    for idx in (0, 7, 300):
        ell = int(cavity.spectrum.ell[idx])
        k = float(cavity.spectrum.k[idx])
        norm = cavity._norm[idx]
        value, _ = quad(
            lambda r, o=ell, kk=k, n=norm: (n * spherical_jn(o, kk * r)) ** 2 * r * r,
            0.0,
            RADIUS,
            limit=400,
        )
        assert value == pytest.approx(1.0, rel=1e-6)


def test_mode_functions_vanish_at_wall(cavity):
    at_wall = cavity.mode_functions(RADIUS)
    inside = cavity.mode_functions(0.5 * RADIUS)
    assert np.max(np.abs(at_wall)) < 1e-6 * np.max(np.abs(inside))


def test_phi_squared_vanishes_at_wall(cavity):
    """The Dirichlet condition forces <phi^2(R)> to zero exactly."""
    at_wall = float(cavity.phi_squared(RADIUS)[0])
    inside = float(cavity.phi_squared(0.5 * RADIUS)[0])
    assert at_wall < 1e-10 * inside


def test_phi_squared_approaches_free_value_in_interior(cavity):
    """Away from the wall the cavity result matches unbounded space."""
    interior = float(cavity.phi_squared(0.3 * RADIUS)[0])
    assert interior == pytest.approx(cavity.free_reference, rel=1e-2)


def test_phi_squared_is_suppressed_near_wall(cavity):
    profile = cavity.phi_squared(np.array([0.3, 0.9, 0.98, 1.0]) * RADIUS)
    assert np.all(np.diff(profile) < 0)


def test_insufficient_enumeration_is_rejected():
    """A spectrum enumerated only to Lambda truncates the Gaussian filter."""
    scale = 40.0 / RADIUS
    too_short = enumerate_modes(RADIUS, scale)
    with pytest.raises(ValueError, match="enumerated only to"):
        CavityField(radius=RADIUS, regulator_scale=scale, spectrum=too_short)


def test_interior_correlator_matches_free_space(cavity):
    """Two points straddling the centre reproduce the regulated free-space form."""
    for s_frac in (0.05, 0.10):
        s = s_frac * RADIUS
        cavity_value = float(cavity.correlator(s / 2, s / 2, -1.0)[0])
        free_value = float(free_correlator_regulated(s, cavity.regulator_scale)[0])
        assert cavity_value == pytest.approx(free_value, rel=2e-2)


def test_correlator_suppressed_near_wall(cavity):
    """A pair near the wall is less correlated than one at the centre.

    This is the property that removes the circularity from the reconstruction:
    G is no longer a function of separation alone.
    """
    s = 0.2 * RADIUS
    centre_pair = float(cavity.correlator(s / 2, s / 2, -1.0)[0])
    r0 = 0.9 * RADIUS
    cos_gamma = 1 - s**2 / (2 * r0**2)
    wall_pair = float(cavity.correlator(r0, r0, cos_gamma)[0])
    assert wall_pair < 0.7 * centre_pair


def test_correlation_matrix_symmetric(cavity):
    points = Sphere(RADIUS).lattice_points(2)
    g = cavity.correlation_matrix(points)
    assert np.allclose(g, g.T, rtol=1e-8)


def test_correlation_matrix_agrees_with_scalar_correlator(cavity):
    points = Sphere(RADIUS).lattice_points(2)
    g = cavity.correlation_matrix(points)
    rng = np.random.default_rng(1)
    for _ in range(5):
        i, j = rng.integers(0, len(points), 2)
        r1 = float(np.linalg.norm(points[i]))
        r2 = float(np.linalg.norm(points[j]))
        if r1 == 0 or r2 == 0:
            continue
        cos_gamma = float(points[i] @ points[j] / (r1 * r2))
        expected = float(cavity.correlator(r1, r2, cos_gamma)[0])
        assert g[i, j] == pytest.approx(expected, rel=1e-8)


def test_correlation_matrix_is_not_a_function_of_separation_alone(cavity):
    """The central result of this module.

    Two pairs of equal separation but different position give different
    correlators, so no inversion G -> d can be well defined.
    """
    s = 0.3 * RADIUS
    near_centre = float(cavity.correlator(s / 2, s / 2, -1.0)[0])
    r0 = 0.85 * RADIUS
    near_wall = float(
        cavity.correlator(r0, r0, 1 - s**2 / (2 * r0**2))[0]
    )
    assert abs(near_wall - near_centre) > 0.2 * abs(near_centre)


def test_points_outside_sphere_rejected(cavity):
    outside = np.array([[2 * RADIUS, 0.0, 0.0]])
    with pytest.raises(ValueError, match="inside the sphere"):
        cavity.correlation_matrix(outside)
