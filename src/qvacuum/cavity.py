"""Vacuum correlators built from the Dirichlet mode functions of the sphere.

The correlators in :mod:`qvacuum.correlators` are those of an unbounded space,
sampled at points that happen to lie inside a ball. They carry no information
about the boundary. This module constructs the correlator of the field that
actually satisfies phi = 0 on the sphere wall:

    G(x, y) = sum_{l n} (2l + 1) / (4 pi) * P_l(cos gamma)
                        * u_{l n}(r_x) u_{l n}(r_y) * f(k_{l n}) / (2 omega)

where gamma is the angle subtended by the two points at the centre, the sum
runs over the Dirichlet eigenvalues k_{l n} of :mod:`qvacuum.modes`, and

    u_{l n}(r) = sqrt(2 / R^3) * j_l(k_{l n} r) / |j_{l + 1}(k_{l n} R)|

is normalised so that the integral of u^2 r^2 over the ball is unity. The
angular sum over m has been collapsed with the spherical harmonic addition
theorem, which is why only P_l appears.

The consequence that matters for the reconstruction study is that G depends on
where the pair sits, not only on how far apart the points are. Deep inside the
sphere it agrees with the free-space form; approaching the wall it is
suppressed, since the field must vanish there.

Regularisation
--------------
A Gaussian filter exp(-k^2 / 2 Lambda^2) is applied by default rather than a
sharp truncation, for the ringing reason set out in
:mod:`qvacuum.correlators`. The enumeration cutoff must sit well above the
regulator scale or the filter is itself truncated: at k = Lambda the weight is
still exp(-1/2) = 0.61, and enumerating only to Lambda recovers a fraction
1 - exp(-1/2) = 0.3935 of the intended sum. A factor of four is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre, spherical_jn

from .modes import ModeSpectrum, enumerate_modes

ENUMERATION_MARGIN = 4.0
"""Required ratio of the enumeration cutoff to the regulator scale."""


@dataclass
class CavityField:
    """A free massless scalar field in its ground state inside a Dirichlet sphere.

    Parameters
    ----------
    radius
        Sphere radius [m].
    regulator_scale
        Gaussian filter scale Lambda [m^-1].
    spectrum
        Pre-enumerated modes. Built automatically if omitted, out to
        ``ENUMERATION_MARGIN`` times the regulator scale.
    """

    radius: float
    regulator_scale: float
    spectrum: ModeSpectrum | None = None
    _weight: np.ndarray = field(init=False, repr=False)
    _norm: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        required = ENUMERATION_MARGIN * self.regulator_scale
        if self.spectrum is None:
            self.spectrum = enumerate_modes(self.radius, required)
        elif self.spectrum.k.max() < required * (1 - 1e-9):
            raise ValueError(
                "spectrum enumerated only to k = "
                f"{self.spectrum.k.max():.3e}; the Gaussian regulator at "
                f"Lambda = {self.regulator_scale:.3e} requires enumeration to "
                f"{required:.3e}, otherwise the filter is itself truncated"
            )
        k = self.spectrum.k
        ell = self.spectrum.ell
        self._norm = np.sqrt(2.0 / self.radius**3) / np.abs(
            spherical_jn(ell + 1, k * self.radius)
        )
        damping = np.exp(-0.5 * (k / self.regulator_scale) ** 2)
        self._weight = (2 * ell + 1) / (4.0 * np.pi) * damping / (2.0 * k)

    @property
    def free_reference(self) -> float:
        """Unbounded-space <phi^2> under the same regulator.

        Equal to Lambda^2 / (4 pi^2), from
        (1 / 4 pi^2) * integral_0^inf dk k exp(-k^2 / 2 Lambda^2).
        """
        return self.regulator_scale**2 / (4.0 * np.pi**2)

    def mode_functions(self, r: float) -> np.ndarray:
        """Normalised radial mode functions u_{l n}(r) at radius r [m]."""
        return self._norm * spherical_jn(self.spectrum.ell, self.spectrum.k * r)

    def phi_squared(self, r: np.ndarray) -> np.ndarray:
        """Regulated <phi^2(r)> at radius r [m].

        Vanishes at r = R, since the Dirichlet condition forces the field to
        zero there, and approaches :attr:`free_reference` in the interior.
        """
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.empty_like(r)
        for i, radius in enumerate(r):
            u = self.mode_functions(radius)
            out[i] = np.sum(self._weight * u * u)
        return out

    def phi_squared_subtracted(self, r: np.ndarray) -> np.ndarray:
        """<phi^2(r)> with the unbounded-space value subtracted.

        Near the wall this approaches the flat Dirichlet plane result
        -1 / (16 pi^2 d^2), with d = R - r, in the window 1/Lambda << d << R.

        This is not a proper renormalisation. Subtracting the free-space mode
        sum term by term leaves a residual that grows with the regulator scale,
        because the cavity and free-space labels k are not the same quantity.
        The cutoff-independent construction is point splitting: take the
        difference of the two correlators at finite separation and let the
        separation go to zero. Values far from the wall should therefore be
        read as regulator dependent and are not reported as physical.
        """
        return self.phi_squared(r) - self.free_reference

    def correlator(self, r1: np.ndarray, r2: np.ndarray,
                   cos_gamma: np.ndarray) -> np.ndarray:
        """G(x, y) for points at radii r1, r2 subtending an angle gamma."""
        r1, r2, cos_gamma = (
            np.atleast_1d(np.asarray(a, dtype=float))
            for a in (r1, r2, cos_gamma)
        )
        r1, r2, cos_gamma = np.broadcast_arrays(r1, r2, cos_gamma)
        out = np.empty(r1.shape)
        for i in np.ndindex(r1.shape):
            u1 = self.mode_functions(float(r1[i]))
            u2 = self.mode_functions(float(r2[i]))
            legendre = eval_legendre(self.spectrum.ell, float(cos_gamma[i]))
            out[i] = np.sum(self._weight * legendre * u1 * u2)
        return out

    def correlation_matrix(self, points: np.ndarray) -> np.ndarray:
        """Full G_ij on a set of Cartesian points inside the sphere.

        Assembled by grouping modes by l so that each l contributes an outer
        product weighted by P_l(cos gamma_ij). Cost is O(n_l_max * n_sites^2),
        which is tolerable to a few hundred sites and not beyond; see
        docs/computational-limits.md.
        """
        points = np.asarray(points, dtype=float)
        radii = np.linalg.norm(points, axis=1)
        if np.any(radii > self.radius * (1 + 1e-9)):
            raise ValueError("all points must lie inside the sphere")

        with np.errstate(invalid="ignore", divide="ignore"):
            unit = np.where(radii[:, None] > 0, points / radii[:, None], 0.0)
        cos_gamma = np.clip(unit @ unit.T, -1.0, 1.0)

        ell = self.spectrum.ell
        k = self.spectrum.k
        out = np.zeros((len(points), len(points)))
        for l_value in np.unique(ell):
            sel = ell == l_value
            u = self._norm[sel] * spherical_jn(
                l_value, np.outer(radii, k[sel])
            )
            weighted = u * self._weight[sel]
            out += eval_legendre(l_value, cos_gamma) * (weighted @ u.T)
        return out


def free_correlator_regulated(separation: np.ndarray, cutoff: float) -> np.ndarray:
    """Unbounded-space massless correlator under the same Gaussian regulator.

    G(s) = (1 / 4 pi^2 s) * integral_0^inf dk sin(k s) exp(-k^2 / 2 Lambda^2)

    Provided as the comparison against which the cavity result is validated in
    the interior and shown to depart near the wall.
    """
    separation = np.atleast_1d(np.asarray(separation, dtype=float))
    if np.any(separation <= 0):
        raise ValueError("separation must be strictly positive")
    out = np.empty_like(separation)
    for i, s in enumerate(separation):
        value, _ = quad(
            lambda q, sep=s: np.sin(q * sep) * np.exp(-0.5 * (q / cutoff) ** 2),
            0.0,
            8.0 * cutoff,
            limit=800,
        )
        out[i] = value / (4.0 * np.pi**2 * s)
    return out
