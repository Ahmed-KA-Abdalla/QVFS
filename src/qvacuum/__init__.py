"""qvacuum: numerical study of free scalar field vacuum fluctuations in a sphere.

Scope is a single free real scalar field in its ground state. Every quantity
computed here has a closed form; the numerics exist to be validated against it
and to reach regimes where the closed form is awkward to evaluate by hand. No
claim about emergent spacetime is made or implied.
"""

from __future__ import annotations

from .cavity import CavityField, free_correlator_regulated
from .constants import DEFAULT_RADIUS
from .geometry import Sphere, pair_distances
from .modes import ModeSpectrum, enumerate_modes, lattice_count, weyl_count

__all__ = [
    "DEFAULT_RADIUS",
    "CavityField",
    "ModeSpectrum",
    "Sphere",
    "enumerate_modes",
    "free_correlator_regulated",
    "lattice_count",
    "pair_distances",
    "weyl_count",
]

__version__ = "0.1.0"
