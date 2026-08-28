"""Ultraviolet cutoff conventions and named physical scales.

The mode content of a bounded region is not a number but a function of the
ultraviolet cutoff. This module fixes one convention so that every result in
the project is mutually comparable, and collects the physically motivated
scales used in the cutoff scan.
"""

from __future__ import annotations

import numpy as np

NAMED_SCALES: dict[str, float] = {
    "optical (1 um)": 1e-6,
    "100 nm": 1e-7,
    "10 nm": 1e-8,
    "molecular (1 nm)": 1e-9,
    "atomic (1 A)": 1e-10,
    "nuclear (1 fm)": 1e-15,
    "electroweak (1e-18 m)": 1e-18,
    "Planck": 1.616_255e-35,
}
"""Shortest resolved wavelength [m] for each named scale."""


def cutoff_from_wavelength(wavelength: float) -> float:
    """Wavenumber cutoff [m^-1] for a given shortest wavelength [m].

    The convention adopted throughout is Lambda = pi / lambda_min, matching the
    Nyquist wavenumber of a lattice of spacing lambda_min. The alternative
    convention 2 pi / lambda_min differs by a factor of two and hence changes
    every mode count by a factor of eight. The choice is arbitrary; what
    matters is that it is stated and held fixed.
    """
    return np.pi / wavelength


def wavelength_from_cutoff(cutoff: float) -> float:
    """Inverse of :func:`cutoff_from_wavelength`."""
    return np.pi / cutoff


def cutoff_grid(
    r_lambda_min: float,
    r_lambda_max: float,
    radius: float,
    n_points: int = 64,
) -> np.ndarray:
    """Logarithmic grid of cutoffs, specified via the dimensionless R*Lambda."""
    return (
        np.logspace(np.log10(r_lambda_min), np.log10(r_lambda_max), n_points)
        / radius
    )
