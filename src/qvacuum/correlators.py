"""Equal-time vacuum two-point functions of a free scalar field.

For an unbounded 3+1 dimensional space the equal-time correlator is

    G(r) = <0| phi(x) phi(y) |0>
         = integral d^3k / (2 pi)^3 * exp(i k.r) / (2 omega_k)
         = m K_1(m r) / (4 pi^2 r)

which reduces to 1 / (4 pi^2 r^2) as m -> 0, since K_1(z) -> 1/z. The massive
form decays as exp(-m r) for m r >> 1, giving a finite correlation length 1/m.

Both the closed form and cutoff-regulated forms are provided; the latter are
what a truncated calculation actually sees.

A sharp momentum cutoff does not converge pointwise to the continuum result.
For m = 0 the truncated integral has the exact closed form

    G_Lambda(r) = (1 - cos(Lambda r)) / (4 pi^2 r^2)

which oscillates between zero and twice the continuum value for Lambda r >> 1
and never settles. Only its average over a cycle converges. This is Gibbs
ringing from the discontinuous filter, not a numerical defect, and it is the
reason a smooth regulator is preferable whenever the correlator itself is the
object of study rather than an integral over it.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import kv


def correlator_free(r: np.ndarray, mass: float = 0.0) -> np.ndarray:
    """Continuum equal-time correlator at separation r [m], natural units.

    Divergent at r = 0; the caller must supply a short-distance regulator, for
    which the lattice spacing is the natural choice.
    """
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0):
        raise ValueError("r must be strictly positive; G is divergent at r = 0")
    if mass == 0.0:
        return 1.0 / (4.0 * np.pi**2 * r**2)
    return mass * kv(1, mass * r) / (4.0 * np.pi**2 * r)


def correlator_cutoff(r: float, cutoff: float, mass: float = 0.0) -> float:
    """Correlator with modes above ``cutoff`` removed.

    G(r) = (1 / 4 pi^2 r) * integral_0^Lambda dk k sin(k r) / sqrt(k^2 + m^2)

    Evaluated by adaptive quadrature. Oscillatory for cutoff * r >> 1, so the
    integration limit is raised accordingly.
    """
    if r <= 0:
        raise ValueError("r must be strictly positive")

    def integrand(k: float) -> float:
        return k * np.sin(k * r) / np.sqrt(k * k + mass * mass)

    limit = max(50, int(10 * cutoff * r))
    value, _ = quad(integrand, 0.0, cutoff, limit=limit)
    return value / (4.0 * np.pi**2 * r)


def correlator_cutoff_analytic(r: np.ndarray) -> np.ndarray:
    """Exact sharp-cutoff massless correlator, (1 - cos(Lambda r)) / (4 pi^2 r^2).

    Takes the product Lambda * r as ``r`` is dimensionful; call as
    ``correlator_cutoff_analytic(cutoff * r) / (4 pi^2 r^2)`` if preferred.
    Here the argument is the dimensionless Lambda * r and the return value is
    the dimensionless factor 1 - cos(Lambda r).
    """
    return 1.0 - np.cos(np.asarray(r, dtype=float))


def correlator_smooth(r: float, cutoff: float, mass: float = 0.0) -> float:
    """Correlator with a Gaussian regulator exp(-k^2 / 2 Lambda^2).

    Free of the ringing described in the module docstring, at the cost of no
    longer corresponding to a strict mode truncation.
    """
    if r <= 0:
        raise ValueError("r must be strictly positive")

    def integrand(k: float) -> float:
        damping = np.exp(-0.5 * (k / cutoff) ** 2)
        return k * np.sin(k * r) * damping / np.sqrt(k * k + mass * mass)

    upper = 8.0 * cutoff
    limit = max(50, int(10 * upper * r))
    value, _ = quad(integrand, 0.0, upper, limit=limit)
    return value / (4.0 * np.pi**2 * r)


def correlation_matrix(points: np.ndarray, mass: float = 0.0,
                       regulator: float | None = None) -> np.ndarray:
    """Correlation matrix G_ij on a set of lattice points.

    Parameters
    ----------
    points
        Array of shape (n_sites, 3), coordinates in metres.
    mass
        Mass parameter [m^-1].
    regulator
        Short-distance cutoff [m] substituted for the vanishing diagonal
        separation. The lattice spacing is the usual choice. If omitted the
        diagonal is set to NaN.

    Notes
    -----
    This is the free-space correlator evaluated at lattice separations. It
    carries no information about the sphere boundary; a boundary-aware
    correlator must be built from the Dirichlet mode functions instead, which
    is deferred to a later stage of the project.
    """
    diff = points[:, None, :] - points[None, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    out = np.empty_like(r)
    off = r > 0
    out[off] = correlator_free(r[off], mass)
    if regulator is None:
        out[~off] = np.nan
    else:
        out[~off] = float(correlator_free(np.array([regulator]), mass)[0])
    return out
