"""Ground-state fluctuation amplitudes of a free scalar field.

Each field mode is a harmonic oscillator whose ground state has
<q_k^2> = 1 / (2 omega_k) in natural units. Summing over modes gives <phi^2>,
which is quadratically divergent and therefore cutoff dependent.
"""

from __future__ import annotations

import numpy as np


def omega(k: np.ndarray, mass: float = 0.0) -> np.ndarray:
    """Mode frequency [m^-1] for wavenumber k and mass parameter m."""
    return np.sqrt(np.asarray(k, dtype=float) ** 2 + mass**2)


def mode_amplitude(k: np.ndarray, mass: float = 0.0) -> np.ndarray:
    """Ground-state amplitude <q_k^2> = 1 / (2 omega_k) of a single mode."""
    return 0.5 / omega(k, mass)


def phi_squared_analytic(cutoff: float, mass: float = 0.0) -> float:
    """Continuum <phi^2> below a wavenumber cutoff, in natural units.

    <phi^2> = (1 / 4 pi^2) * integral_0^Lambda dk k^2 / sqrt(k^2 + m^2)

    For m = 0 this is Lambda^2 / (8 pi^2) exactly. For m > 0 the closed form is

        (1 / 8 pi^2) [ Lambda sqrt(Lambda^2 + m^2)
                       - m^2 arcsinh(Lambda / m) ]
    """
    if mass == 0.0:
        return cutoff**2 / (8.0 * np.pi**2)
    root = np.sqrt(cutoff**2 + mass**2)
    return float(
        (cutoff * root - mass**2 * np.arcsinh(cutoff / mass)) / (8.0 * np.pi**2)
    )


def phi_squared_from_spectrum(k: np.ndarray, degeneracy: np.ndarray,
                              volume: float, mass: float = 0.0) -> float:
    """<phi^2> obtained by summing an explicitly enumerated mode spectrum.

    Provided as the numerical counterpart to :func:`phi_squared_analytic`. The
    two should agree to within the discreteness error of the cavity spectrum,
    which falls as the cutoff rises.
    """
    return float(np.sum(degeneracy * mode_amplitude(k, mass)) / volume)
