"""Validation of the radial-chain entanglement entropy calculation."""

from __future__ import annotations

import numpy as np
import pytest

from qvacuum.entropy import (
    chain_entropy,
    covariance_matrices,
    entanglement_entropy,
    entropy_convergence,
    radial_coupling_matrix,
    region_radius,
    symplectic_spectrum,
)

N_SITES = 40
ELL_MAX = 300


def test_coupling_matrix_symmetric_and_positive_definite():
    for ell in (0, 1, 7, 40):
        k = radial_coupling_matrix(N_SITES, ell)
        assert np.allclose(k, k.T)
        assert np.linalg.eigvalsh(k).min() > 0


def test_coupling_matrix_is_tridiagonal():
    k = radial_coupling_matrix(N_SITES, 3)
    off_band = ~(np.abs(np.subtract.outer(np.arange(N_SITES), np.arange(N_SITES))) <= 1)
    assert np.allclose(k[off_band], 0.0)


def test_angular_term_raises_the_diagonal():
    """The l(l+1)/j^2 centrifugal term must increase every diagonal entry."""
    low = np.diag(radial_coupling_matrix(N_SITES, 0))
    high = np.diag(radial_coupling_matrix(N_SITES, 5))
    assert np.all(high > low)


def test_covariance_product_is_pure_state_identity():
    """X P = I / 4 exactly, since the global ground state is pure."""
    for ell in (0, 4):
        x, p = covariance_matrices(N_SITES, ell)
        assert np.allclose(x @ p, np.eye(N_SITES) / 4.0, atol=1e-10)


def test_covariance_matrices_symmetric_positive():
    x, p = covariance_matrices(N_SITES, 2)
    assert np.allclose(x, x.T)
    assert np.allclose(p, p.T)
    assert np.linalg.eigvalsh(x).min() > 0
    assert np.linalg.eigvalsh(p).min() > 0


def test_symplectic_eigenvalues_at_least_one_half():
    """Below one half would violate the uncertainty relation."""
    nu = symplectic_spectrum(N_SITES, 3, 10)
    assert np.all(nu >= 0.5 - 1e-12)


def test_whole_system_has_zero_entropy():
    """Tracing out nothing leaves a pure state."""
    assert chain_entropy(N_SITES, 2, N_SITES) == pytest.approx(0.0, abs=1e-8)


def test_entropy_positive_and_increasing_with_region_size():
    values = [chain_entropy(N_SITES, 0, n) for n in (3, 6, 10, 15)]
    assert all(v > 0 for v in values)
    assert np.all(np.diff(values) > 0)


def test_chain_entropy_falls_with_angular_momentum():
    """High-l chains are stiffer and contribute less per mode."""
    values = [chain_entropy(N_SITES, ell, 8) for ell in (0, 20, 60, 150)]
    assert np.all(np.diff(values) < 0)


def test_area_law_exponent():
    """S scales as R^2, not R^3.

    The central result of this module. A volume law would give a log-log slope
    near three; the area law gives two.
    """
    n_values = (4, 8, 14, 20)
    radii = np.array([region_radius(n) for n in n_values])
    entropies = np.array(
        [entanglement_entropy(N_SITES, n, ELL_MAX) for n in n_values]
    )
    slope = np.polyfit(np.log(radii), np.log(entropies), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.05)


def test_area_law_coefficient_matches_srednicki():
    """S / R^2 agrees with the published coefficient of about 0.30.

    The truncation at ELL_MAX underestimates the sum, so the recovered value
    sits slightly low; the tolerance reflects that rather than disguising it.
    """
    for n in (8, 14):
        s = entanglement_entropy(N_SITES, n, ELL_MAX)
        assert s / region_radius(n) ** 2 == pytest.approx(0.30, abs=0.03)


def test_entropy_convergence_is_monotonic():
    """Every chain contributes positively, so partial sums only rise."""
    partial = entropy_convergence(N_SITES, 8, (20, 60, 150, 300))
    values = [value for _, value in partial]
    assert np.all(np.diff(values) > 0)


def test_truncation_error_is_material():
    """Guards against quoting a converged figure that is not converged.

    Raising the ceiling from 100 to 300 still changes the answer by more than
    one per cent, so ell_max must be reported alongside any entropy.
    """
    coarse = entanglement_entropy(N_SITES, 10, 100)
    fine = entanglement_entropy(N_SITES, 10, 300)
    assert (fine - coarse) / fine > 0.01


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        radial_coupling_matrix(1, 0)
    with pytest.raises(ValueError):
        radial_coupling_matrix(10, -1)
    with pytest.raises(ValueError):
        symplectic_spectrum(10, 0, 11)
