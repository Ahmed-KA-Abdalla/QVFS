"""Validation of vacuum fluctuation amplitudes."""

from __future__ import annotations

import numpy as np
import pytest

from qvacuum.vacuum import mode_amplitude, omega, phi_squared_analytic


def test_massless_phi_squared_closed_form():
    """<phi^2> = Lambda^2 / (8 pi^2) for m = 0."""
    for cutoff in (1e6, 1e9, 1e12):
        assert phi_squared_analytic(cutoff) == pytest.approx(
            cutoff**2 / (8 * np.pi**2), rel=1e-14
        )


def test_massive_reduces_to_massless():
    """For Lambda >> m the massive result approaches the massless one."""
    cutoff = 1e12
    massive = phi_squared_analytic(cutoff, mass=1e6)
    massless = phi_squared_analytic(cutoff, mass=0.0)
    assert massive == pytest.approx(massless, rel=1e-10)


def test_massive_amplitude_is_suppressed():
    """A mass lowers the fluctuation amplitude at fixed cutoff."""
    cutoff = 1e9
    assert phi_squared_analytic(cutoff, mass=1e9) < phi_squared_analytic(cutoff)


def test_phi_squared_quadratic_divergence():
    """Doubling the cutoff quadruples <phi^2> in the massless case."""
    a = phi_squared_analytic(1e9)
    b = phi_squared_analytic(2e9)
    assert b / a == pytest.approx(4.0, rel=1e-12)


def test_numerical_integral_matches_closed_form():
    from scipy.integrate import quad

    cutoff, mass = 1e9, 3e8
    value, _ = quad(lambda k: k**2 / np.sqrt(k**2 + mass**2), 0, cutoff)
    assert value / (4 * np.pi**2) == pytest.approx(
        phi_squared_analytic(cutoff, mass), rel=1e-8
    )


def test_omega_and_amplitude_consistency():
    k = np.array([1e6, 1e8, 1e10])
    assert np.allclose(mode_amplitude(k, 0.0), 0.5 / omega(k, 0.0))
    assert np.allclose(omega(k, 0.0), k)
