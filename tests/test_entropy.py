"""Validation of the radial-chain entanglement entropy calculation."""

from __future__ import annotations

import numpy as np
import pytest

from qvacuum.entropy import (
    bose_entropy,
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


def test_mutual_information_non_negative():
    """Subadditivity: I(A:B) >= 0 for any disjoint regions."""
    a = np.arange(0, 6)
    for gap in (2, 5, 10):
        b = np.arange(6 + gap, 12 + gap)
        for ell in (0, 5, 30):
            from qvacuum.entropy import mutual_information_chain

            assert mutual_information_chain(N_SITES, ell, a, b) >= -1e-12


def test_mutual_information_falls_with_separation():
    from qvacuum.entropy import mutual_information

    a = np.arange(0, 6)
    values = [
        mutual_information(N_SITES, a, np.arange(6 + g, 12 + g), 100)
        for g in (2, 6, 12)
    ]
    assert np.all(np.diff(values) < 0)


def test_overlapping_regions_rejected():
    from qvacuum.entropy import mutual_information_chain

    with pytest.raises(ValueError, match="disjoint"):
        mutual_information_chain(N_SITES, 0, np.arange(0, 6), np.arange(4, 10))


def test_mutual_information_is_uv_finite():
    """The central result of script 09.

    Holding the geometry fixed and refining the lattice, the entropy grows as
    the area while the mutual information settles. Divergences cancel in the
    combination S_A + S_B - S_AB.
    """
    from qvacuum.entropy import entanglement_entropy, mutual_information

    infos, entropies = [], []
    for scale in (2, 4, 6):
        n_sites = 20 * scale
        infos.append(
            mutual_information(
                n_sites, np.arange(0, 2 * scale),
                np.arange(4 * scale, 6 * scale), 30 * scale,
            )
        )
        entropies.append(entanglement_entropy(n_sites, 2 * scale, 30 * scale))

    assert entropies[2] / entropies[0] > 5
    assert abs(infos[2] - infos[0]) / infos[0] < 0.2


def test_pairwise_mutual_information_properties():
    from qvacuum.cavity import CavityField, smeared_covariance
    from qvacuum.entropy import pairwise_mutual_information
    from qvacuum.geometry import Sphere

    sphere = Sphere(0.5e-6)
    spacing = sphere.lattice_spacing(2)
    points = sphere.lattice_points(2)
    radii = np.linalg.norm(points, axis=1)
    points = points[radii < 0.95 * sphere.radius]
    field = CavityField(radius=sphere.radius, regulator_scale=40.0 / sphere.radius)

    x, p = smeared_covariance(field, points, spacing)
    assert np.min(np.sqrt(np.diag(x) * np.diag(p))) > 0.5

    info = pairwise_mutual_information(x, p)
    assert np.allclose(info, info.T)
    off = ~np.eye(len(points), dtype=bool)
    assert np.all(info[off] >= -1e-12)


def test_unsmeared_correlators_are_not_canonical():
    """Guards the smearing step.

    Continuum correlators used directly give single-site symplectic
    eigenvalues vastly above one half, because they are delta-normalised
    densities rather than canonical variables.
    """
    from qvacuum.cavity import (
        CavityField,
        momentum_correlation_matrix,
        smeared_covariance,
    )
    from qvacuum.geometry import Sphere

    sphere = Sphere(0.5e-6)
    points = sphere.lattice_points(2)
    radii = np.linalg.norm(points, axis=1)
    points = points[radii < 0.95 * sphere.radius]
    field = CavityField(radius=sphere.radius, regulator_scale=40.0 / sphere.radius)

    raw = np.sqrt(
        np.diag(field.correlation_matrix(points))
        * np.diag(momentum_correlation_matrix(field, points))
    )
    smeared = np.sqrt(
        np.diag(smeared_covariance(field, points, sphere.lattice_spacing(2))[0])
        * np.diag(smeared_covariance(field, points, sphere.lattice_spacing(2))[1])
    )
    assert raw.min() > 1e10
    assert smeared.max() < 1e4


def test_unphysical_smearing_is_rejected():
    """Guards the failure found at a lattice-matched cutoff.

    Reducing Lambda * a towards pi drives the single-site symplectic
    eigenvalue below one half. No physical Gaussian state permits this, and
    the clipping inside symplectic_spectrum would otherwise hide it behind a
    plausible-looking entropy.
    """
    from qvacuum.cavity import CavityField, smeared_covariance
    from qvacuum.geometry import Sphere

    sphere = Sphere(0.5e-6)
    spacing = sphere.lattice_spacing(3)
    points = sphere.lattice_points(3)
    points = points[np.linalg.norm(points, axis=1) < 0.95 * sphere.radius]
    field = CavityField(radius=sphere.radius, regulator_scale=np.pi / spacing)

    with pytest.raises(ValueError, match="below the uncertainty bound"):
        smeared_covariance(field, points, spacing)


def test_thermal_whole_system_matches_bose_formula():
    """Tracing over nothing must reproduce the Bose thermal entropy exactly.

    Independent of the covariance machinery, so this validates that the
    thermal factor coth(omega / 2T) has been applied correctly.
    """
    for temperature in (0.1, 0.5, 2.0):
        numeric = chain_entropy(30, 3, 30, temperature)
        analytic = bose_entropy(30, 3, temperature)
        assert numeric == pytest.approx(analytic, rel=1e-8)


def test_zero_temperature_recovers_ground_state():
    assert chain_entropy(N_SITES, 2, 10, 0.0) == pytest.approx(
        chain_entropy(N_SITES, 2, 10)
    )
    assert bose_entropy(N_SITES, 2, 0.0) == 0.0


def test_entropy_increases_with_temperature():
    values = [chain_entropy(N_SITES, 1, 8, t) for t in (0.0, 0.1, 0.3, 1.0)]
    assert np.all(np.diff(values) > 0)


def test_negative_temperature_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        covariance_matrices(N_SITES, 0, -1.0)


def test_area_law_becomes_volume_law():
    """The central result of script 11.

    The area law belongs to the vacuum. Heating the state drives the scaling
    exponent from two towards three.
    """
    n_values = (4, 8, 12, 16)
    radii = np.array([region_radius(n) for n in n_values])

    def slope(temperature: float) -> float:
        entropies = np.array([
            entanglement_entropy(N_SITES, n, ELL_MAX, temperature)
            for n in n_values
        ])
        return float(np.polyfit(np.log(radii), np.log(entropies), 1)[0])

    assert slope(0.0) == pytest.approx(2.0, abs=0.06)
    assert slope(1.0) > 2.8


def test_thermal_state_is_globally_mixed():
    """X P = I / 4 holds at zero temperature and fails above it."""
    x, p = covariance_matrices(30, 2, 0.0)
    assert np.allclose(x @ p, np.eye(30) / 4.0, atol=1e-10)
    x, p = covariance_matrices(30, 2, 0.5)
    assert not np.allclose(x @ p, np.eye(30) / 4.0, atol=1e-3)
