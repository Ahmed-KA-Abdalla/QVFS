"""Mode enumeration for a scalar field in a spherical cavity.

Two inequivalent notions of "the number of degrees of freedom inside the
sphere" are implemented here, and they do not agree:

1. Cavity modes. Impose Dirichlet conditions on the sphere wall. The
   eigenmodes are j_l(k r) Y_lm with k R a zero of j_l, each l carrying a
   (2l + 1)-fold degeneracy. This is a well posed eigenvalue problem and the
   count is exact.

2. Subregion lattice. Discretise the interior at spacing a = pi / Lambda and
   count sites. This is a regularisation choice, not a property of the vacuum:
   a subregion of an infinite space does not possess its own modes.

The discrepancy between the two is itself a result and is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn


@dataclass(frozen=True)
class ModeSpectrum:
    """Enumerated Dirichlet eigenmodes of a sphere.

    Attributes
    ----------
    k
        Eigen-wavenumbers [m^-1], ascending, one entry per (l, n) pair.
    ell
        Angular momentum quantum number of each entry.
    degeneracy
        Multiplicity 2l + 1 of each entry.
    radius
        Sphere radius [m].
    """

    k: np.ndarray
    ell: np.ndarray
    degeneracy: np.ndarray
    radius: float

    @property
    def total(self) -> int:
        """Total number of modes, counting degeneracy."""
        return int(np.sum(self.degeneracy))

    def count_below(self, cutoff: float) -> int:
        """Number of modes, counting degeneracy, with k <= cutoff."""
        return int(np.sum(self.degeneracy[self.k <= cutoff]))

    def counting_function(self, cutoffs: np.ndarray) -> np.ndarray:
        """Vectorised N(Lambda) over an array of cutoffs."""
        order = np.argsort(self.k)
        k_sorted = self.k[order]
        cumulative = np.cumsum(self.degeneracy[order])
        idx = np.searchsorted(k_sorted, cutoffs, side="right")
        return np.where(idx > 0, cumulative[np.clip(idx - 1, 0, None)], 0)


def spherical_jn_zeros(ell: int, x_max: float, grid_step: float = 0.25) -> np.ndarray:
    """Zeros of the spherical Bessel function j_l in the open interval (0, x_max].

    Located by bracketing sign changes on a uniform grid and refining with
    Brent's method. The grid step must be smaller than the minimum spacing of
    consecutive zeros, which tends to pi from above, so the default is safe.
    """
    if x_max <= 0:
        return np.array([])
    start = max(grid_step, float(ell))
    if start >= x_max:
        return np.array([])
    x = np.arange(start, x_max + grid_step, grid_step)
    f = spherical_jn(ell, x)
    sign_change = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    roots = [
        brentq(lambda t: spherical_jn(ell, t), x[i], x[i + 1], xtol=1e-12)
        for i in sign_change
    ]
    roots = [r for r in roots if r <= x_max]
    return np.array(roots)


def enumerate_modes(radius: float, cutoff: float) -> ModeSpectrum:
    """Enumerate all Dirichlet eigenmodes of a sphere with k <= cutoff.

    Parameters
    ----------
    radius
        Sphere radius [m].
    cutoff
        Wavenumber cutoff [m^-1].

    Notes
    -----
    Cost grows as (R * cutoff)^2 in the number of (l, n) pairs, so exact
    enumeration is practical up to R * cutoff of order 10^3. Beyond that use
    :func:`weyl_count`.
    """
    x_max = radius * cutoff
    ks: list[float] = []
    ells: list[int] = []
    degs: list[int] = []
    ell = 0
    while ell <= x_max:
        zeros = spherical_jn_zeros(ell, x_max)
        if zeros.size == 0 and ell > 0:
            break
        for z in zeros:
            ks.append(z / radius)
            ells.append(ell)
            degs.append(2 * ell + 1)
        ell += 1
    order = np.argsort(ks) if ks else np.array([], dtype=int)
    return ModeSpectrum(
        k=np.asarray(ks)[order],
        ell=np.asarray(ells, dtype=int)[order],
        degeneracy=np.asarray(degs, dtype=int)[order],
        radius=radius,
    )


def weyl_count(volume: float, cutoff: float, surface_area: float | None = None,
               order: int = 2) -> np.ndarray:
    """Weyl asymptotic counting function for Dirichlet boundary conditions.

    N(Lambda) = V Lambda^3 / (6 pi^2) - S Lambda^2 / (16 pi) + O(Lambda)

    Parameters
    ----------
    volume
        Region volume [m^3].
    cutoff
        Wavenumber cutoff [m^-1]; may be an array.
    surface_area
        Bounding area [m^2]. Required when ``order`` is 2.
    order
        1 for the bulk term alone, 2 to include the boundary correction.

    Notes
    -----
    The boundary term is negative for Dirichlet conditions and would be
    positive for Neumann. Its relative size is 9 pi / (8 R Lambda) for a
    sphere, so it falls below one per cent once R Lambda exceeds about 350.
    The third term is O(Lambda) and is not implemented; it is left to be
    examined as a residual against the exact count.
    """
    cutoff = np.asarray(cutoff, dtype=float)
    bulk = volume * cutoff**3 / (6.0 * np.pi**2)
    if order == 1:
        return bulk
    if order != 2:
        raise ValueError("order must be 1 or 2")
    if surface_area is None:
        raise ValueError("surface_area is required for order=2")
    return bulk - surface_area * cutoff**2 / (16.0 * np.pi)


def lattice_count(volume: float, cutoff: float) -> np.ndarray:
    """Site count of a lattice of spacing pi / cutoff filling the volume.

    This is the second, inequivalent definition of the degree-of-freedom count
    referred to in the module docstring.
    """
    cutoff = np.asarray(cutoff, dtype=float)
    spacing = np.pi / cutoff
    return volume / spacing**3
