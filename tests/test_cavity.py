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


def test_cell_nodes_partition_correctly():
    """Weights average to one and nodes group by cell."""
    from qvacuum.cavity import cell_nodes_weights

    points = np.array([[0.0, 0.0, 0.0], [1e-7, 0.0, 0.0]])
    spacing = 5e-8
    nodes, weights = cell_nodes_weights(points, spacing, 3)
    assert weights.sum() == pytest.approx(1.0)
    assert len(nodes) == len(points) * 27
    grouped = nodes.reshape(len(points), 27, 3)
    for i, centre in enumerate(points):
        assert np.allclose(
            np.einsum("q,qd->d", weights, grouped[i]), centre, atol=1e-18
        )
        assert np.all(np.abs(grouped[i] - centre) <= spacing / 2 + 1e-18)


def test_cell_averaging_falls_below_uncertainty_bound(cavity):
    """The structural failure recorded in script 10.

    Refining the cell quadrature drives the single-site symplectic eigenvalue
    monotonically downwards, through the uncertainty bound, because a
    regulated continuum field restricted to cells is not a canonical system.
    """
    from qvacuum.cavity import cell_nodes_weights, momentum_correlation_matrix

    sphere = Sphere(RADIUS)
    spacing = sphere.lattice_spacing(3)
    points = sphere.lattice_points(3)
    margin = RADIUS - np.sqrt(3) / 2 * spacing
    points = points[np.linalg.norm(points, axis=1) <= margin][:4]
    field = CavityField(radius=RADIUS, regulator_scale=6.0 / spacing)

    minima = []
    for order in (1, 2, 3, 4):
        nodes, weights = cell_nodes_weights(points, spacing, order)
        n_p, n_q = len(points), len(weights)
        x = np.einsum(
            "q,iqjr,r->ij", weights,
            field.correlation_matrix(nodes).reshape(n_p, n_q, n_p, n_q),
            weights,
        )
        p = np.einsum(
            "q,iqjr,r->ij", weights,
            momentum_correlation_matrix(field, nodes).reshape(
                n_p, n_q, n_p, n_q),
            weights,
        ) * spacing**6
        minima.append(float(np.sqrt(np.diag(x) * np.diag(p)).min()))

    assert np.all(np.diff(minima) < 0)
    assert minima[0] > 0.5
    assert minima[-1] < 0.5


def test_cell_averaged_covariance_rejects_unphysical(cavity):
    from qvacuum.cavity import cell_averaged_covariance

    sphere = Sphere(RADIUS)
    spacing = sphere.lattice_spacing(3)
    points = sphere.lattice_points(3)
    margin = RADIUS - np.sqrt(3) / 2 * spacing
    points = points[np.linalg.norm(points, axis=1) <= margin][:4]
    field = CavityField(radius=RADIUS, regulator_scale=6.0 / spacing)

    with pytest.raises(ValueError, match="uncertainty bound"):
        cell_averaged_covariance(field, points, spacing, order=4)


def test_cell_averaged_covariance_rejects_cells_leaving_sphere(cavity):
    from qvacuum.cavity import cell_averaged_covariance

    sphere = Sphere(RADIUS)
    spacing = sphere.lattice_spacing(3)
    points = sphere.lattice_points(3)
    outermost = points[np.argmax(np.linalg.norm(points, axis=1))][None, :]
    field = CavityField(radius=RADIUS, regulator_scale=6.0 / spacing)

    with pytest.raises(ValueError, match="leave the sphere"):
        cell_averaged_covariance(field, outermost, spacing, order=2)
