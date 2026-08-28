"""Physical constants and unit conventions.

Internal convention
-------------------
All calculations use natural units, hbar = c = 1, with lengths measured in
metres. Consequently:

    wavenumber k   [m^-1]
    frequency  w   [m^-1]   (since w = c|k| and c = 1)
    mass       m   [m^-1]   (the inverse Compton wavelength, m c / hbar)

Conversion to SI is applied at the reporting boundary only, by the helper
functions below, and never inside the physics modules.
"""

from __future__ import annotations

HBAR = 1.054_571_817e-34
"""Reduced Planck constant [J s]."""

C_LIGHT = 2.997_924_58e8
"""Speed of light in vacuum [m s^-1]."""

EV = 1.602_176_634e-19
"""Electronvolt [J]."""

PLANCK_LENGTH = 1.616_255e-35
"""Planck length [m]."""

DEFAULT_RADIUS = 0.5e-6
"""Default sphere radius [m]. Corresponds to a diameter of 1 um."""


def mass_from_energy(energy_ev: float) -> float:
    """Convert a rest energy in eV to an inverse Compton wavelength in m^-1."""
    return energy_ev * EV / (HBAR * C_LIGHT)


def energy_from_wavenumber(k: float) -> float:
    """Convert a wavenumber in m^-1 to an energy in eV."""
    return k * HBAR * C_LIGHT / EV


def angular_frequency(k: float) -> float:
    """Convert a wavenumber in m^-1 to an angular frequency in rad s^-1."""
    return k * C_LIGHT
