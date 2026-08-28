"""Validation of the equal-time two-point function."""

from __future__ import annotations

import numpy as np
import pytest

from qvacuum.correlators import (
    correlation_matrix,
    correlator_cutoff,
    correlator_free,
    correlator_smooth,
)
from qvacuum.geometry import Sphere


def test_massless_closed_form():
    r = np.array([1e-9, 1e-8, 1e-7])
    assert np.allclose(correlator_free(r), 1.0 / (4 * np.pi**2 * r**2))


def test_massive_limits_to_massless():
    """m K_1(m r) / (4 pi^2 r) -> 1 / (4 pi^2 r^2) as m r -> 0."""
    r = 1e-9
    small_mass = 1e-3 / r
    assert correlator_free(np.array([r]), small_mass)[0] == pytest.approx(
        correlator_free(np.array([r]))[0], rel=1e-5
    )


def test_massive_decays_exponentially():
    """G(r) ~ exp(-m r) for m r >> 1."""
    mass = 1e7
    r = np.array([3.0, 4.0]) / mass
    g = correlator_free(r, mass)
    ratio = g[1] / g[0]
    assert ratio < np.exp(-0.9)


def test_correlator_decreasing_with_separation():
    r = np.logspace(-9, -6, 40)
    for mass in (0.0, 1e7):
        g = correlator_free(r, mass)
        assert np.all(np.diff(g) < 0)


def test_sharp_cutoff_matches_closed_form():
    """Quadrature reproduces (1 - cos(Lambda r)) / (4 pi^2 r^2) exactly."""
    r = 1e-7
    for r_lambda in (5.0, 50.0, 500.0):
        cutoff = r_lambda / r
        expected = (1 - np.cos(r_lambda)) / (4 * np.pi**2 * r**2)
        assert correlator_cutoff(r, cutoff) == pytest.approx(expected, rel=1e-6)


def test_sharp_cutoff_rings_rather_than_converging():
    """A sharp cutoff does not converge pointwise; it oscillates about the continuum.

    Documented behaviour, not a defect. The ratio to the continuum value is
    1 - cos(Lambda r), which visits both 0 and 2 however large Lambda becomes.
    """
    r = 1e-7
    continuum = float(correlator_free(np.array([r]))[0])
    ratios = [
        correlator_cutoff(r, r_lambda / r) / continuum
        for r_lambda in np.linspace(100.0, 130.0, 60)
    ]
    assert min(ratios) < 0.2
    assert max(ratios) > 1.8


def test_cycle_average_of_sharp_cutoff_matches_continuum():
    r = 1e-7
    continuum = float(correlator_free(np.array([r]))[0])
    r_lambda = np.linspace(200.0, 200.0 + 2 * np.pi, 200)
    mean = np.mean([correlator_cutoff(r, x / r) for x in r_lambda])
    assert mean == pytest.approx(continuum, rel=2e-2)


def test_smooth_regulator_converges():
    """A Gaussian regulator does converge to the continuum for Lambda r >> 1."""
    r = 1e-7
    continuum = float(correlator_free(np.array([r]))[0])
    assert correlator_smooth(r, 50.0 / r) == pytest.approx(continuum, rel=1e-3)


def test_correlator_diverges_at_zero():
    with pytest.raises(ValueError):
        correlator_free(np.array([0.0]))


def test_correlation_matrix_symmetric_and_regulated():
    sphere = Sphere(0.5e-6)
    points = sphere.lattice_points(3)
    g = correlation_matrix(points, regulator=sphere.lattice_spacing(3))
    assert np.allclose(g, g.T)
    assert np.all(np.isfinite(g))
    assert np.all(np.diag(g) >= g.max(axis=1) - 1e-30)
