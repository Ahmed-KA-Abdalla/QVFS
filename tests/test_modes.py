"""Validation of the mode enumerator against known analytic results."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import spherical_jn

from qvacuum.geometry import Sphere
from qvacuum.modes import (
    enumerate_modes,
    lattice_count,
    spherical_jn_zeros,
    weyl_count,
)

RADIUS = 0.5e-6


def test_l0_zeros_are_multiples_of_pi():
    """j_0(x) = sin(x)/x, so its zeros are exactly n*pi."""
    zeros = spherical_jn_zeros(0, 20.0)
    expected = np.arange(1, 7) * np.pi
    assert np.allclose(zeros, expected, rtol=1e-9)


def test_zeros_are_actually_zeros():
    for ell in (0, 1, 5, 12):
        for z in spherical_jn_zeros(ell, 40.0):
            assert abs(spherical_jn(ell, z)) < 1e-9


def test_zeros_are_ordered_and_spaced():
    """Consecutive zeros of j_l are separated by more than pi."""
    for ell in (0, 3, 9):
        zeros = spherical_jn_zeros(ell, 60.0)
        gaps = np.diff(zeros)
        assert np.all(gaps > np.pi - 1e-6)


def test_no_zeros_below_l():
    """j_l has no zero below x = l, so the enumerator must find none."""
    assert spherical_jn_zeros(20, 15.0).size == 0


def test_counting_function_matches_direct_count():
    spectrum = enumerate_modes(RADIUS, 60.0 / RADIUS)
    cutoffs = np.array([10.0, 30.0, 60.0]) / RADIUS
    vectorised = spectrum.counting_function(cutoffs)
    direct = [spectrum.count_below(c) for c in cutoffs]
    assert list(vectorised) == direct


def test_counting_function_is_monotonic():
    spectrum = enumerate_modes(RADIUS, 50.0 / RADIUS)
    cutoffs = np.linspace(1.0, 50.0, 200) / RADIUS
    counts = spectrum.counting_function(cutoffs)
    assert np.all(np.diff(counts) >= 0)


@pytest.mark.parametrize("r_lambda", [40.0, 60.0, 80.0])
def test_two_term_weyl_beats_bulk_term(r_lambda):
    """The boundary correction must reduce the residual against the exact count.

    This is the project's first physics result: the sphere's surface is visible
    in the mode count, with the sign and magnitude the Dirichlet Weyl law
    predicts.
    """
    sphere = Sphere(RADIUS)
    cutoff = r_lambda / RADIUS
    exact = enumerate_modes(RADIUS, cutoff).total
    bulk = weyl_count(sphere.volume, cutoff, order=1)
    two_term = weyl_count(sphere.volume, cutoff, sphere.surface_area, order=2)
    assert abs(two_term - exact) < abs(bulk - exact)


def test_weyl_relative_boundary_size():
    """Boundary/bulk ratio equals 9 pi / (8 R Lambda) for a sphere."""
    sphere = Sphere(RADIUS)
    cutoff = 100.0 / RADIUS
    bulk = weyl_count(sphere.volume, cutoff, order=1)
    two_term = weyl_count(sphere.volume, cutoff, sphere.surface_area, order=2)
    ratio = (bulk - two_term) / bulk
    assert ratio == pytest.approx(9.0 * np.pi / (8.0 * 100.0), rel=1e-12)


def test_lattice_and_cavity_counts_disagree():
    """The two definitions of 'number of degrees of freedom' are inequivalent.

    Recording the disagreement is the point; the assertion guards against a
    later change that would silently make them coincide.
    """
    sphere = Sphere(RADIUS)
    cutoff = 40.0 / RADIUS
    cavity = enumerate_modes(RADIUS, cutoff).total
    lattice = lattice_count(sphere.volume, cutoff)
    assert not np.isclose(cavity, lattice, rtol=0.1)
