"""Entanglement entropy of a spherical region, by Srednicki's radial method.

For a Gaussian state the reduced density matrix of a subregion is fixed
entirely by the restricted covariance matrices, so no wavefunction is needed.
Writing the ground state of a free massless scalar as

    H = (1/2) ( pi^T pi + phi^T K phi ),      X = <phi phi> = W^-1 / 2,
                                              P = <pi pi>   = W / 2,
                                              W = K^(1/2),

the entropy of a region A follows from the eigenvalues nu_k of
sqrt(X_A P_A), where the subscript denotes restriction to the sites of A:

    S = sum_k [ (nu_k + 1/2) ln(nu_k + 1/2) - (nu_k - 1/2) ln(nu_k - 1/2) ].

Discretisation
--------------
A three-dimensional lattice cannot resolve an area law: the region radius must
span at least a decade, needing hundreds of sites per radius, while dense
diagonalisation of the full correlation matrix caps out near sixteen. Srednicki
avoids this by decomposing in spherical harmonics first. Each (l, m) sector is
an independent one-dimensional radial chain,

    H_l = (1/2a) sum_j [ pi_j^2 + (j + 1/2)^2 ( phi_j / j - phi_{j+1} / (j+1) )^2
                         + l(l+1) phi_j^2 / j^2 ],

of a few hundred sites, and the total entropy is sum_l (2l + 1) S_l. Angular
resolution is traded for radial resolution, which is the correct trade when the
question is how entropy scales with the radius of the region.

This module is therefore a different discretisation from :mod:`qvacuum.cavity`,
not an extension of it. It shares the geometry and nothing else.

Result
------
The entropy scales with the boundary area of the region rather than its
volume, with S = 0.295 (R/a)^2 for R = (n + 1/2) a. Srednicki's value is 0.30.
This is the calculation that first suggested black hole entropy might be
entanglement entropy, and it is the direct ancestor of the Ryu-Takayanagi
formula.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

DEFAULT_ELL_MAX = 800
"""Angular momentum ceiling. The sum over l converges slowly; see
:func:`entropy_convergence`."""


def radial_coupling_matrix(n_sites: int, ell: int) -> np.ndarray:
    """Coupling matrix K for the radial chain of angular momentum l.

    Units of 1/a, with the lattice spacing a set to one. Tridiagonal, symmetric
    and positive definite. The site index runs from j = 1 to n_sites, with
    phi vanishing beyond the last site.
    """
    if n_sites < 2:
        raise ValueError("n_sites must be at least 2")
    if ell < 0:
        raise ValueError("ell must be non-negative")
    j = np.arange(1, n_sites + 1, dtype=float)
    diagonal = ell * (ell + 1) / j**2 + ((j + 0.5) ** 2 + (j - 0.5) ** 2) / j**2
    off = -((j[:-1] + 0.5) ** 2) / (j[:-1] * (j[:-1] + 1))
    return np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)


def covariance_matrices(n_sites: int, ell: int) -> tuple[np.ndarray, np.ndarray]:
    """Ground-state covariance matrices X = <phi phi> and P = <pi pi>.

    Their product is the identity over four, which is the statement that the
    global state is pure.
    """
    eigenvalues, vectors = eigh(radial_coupling_matrix(n_sites, ell))
    if eigenvalues.min() <= 0:
        raise ValueError("coupling matrix is not positive definite")
    root = np.sqrt(eigenvalues)
    x = (vectors / root) @ vectors.T / 2.0
    p = (vectors * root) @ vectors.T / 2.0
    return x, p


def symplectic_spectrum(n_sites: int, ell: int, n_inside: int) -> np.ndarray:
    """Eigenvalues nu_k of sqrt(X_A P_A) for the innermost n_inside sites.

    Each nu is at least one half; a value of exactly one half contributes no
    entropy, and the excess above one half measures entanglement across the
    boundary.
    """
    if not 0 < n_inside <= n_sites:
        raise ValueError("n_inside must lie between 1 and n_sites")
    x, p = covariance_matrices(n_sites, ell)
    product = x[:n_inside, :n_inside] @ p[:n_inside, :n_inside]
    eigenvalues = np.linalg.eigvals(product).real
    return np.sqrt(np.clip(eigenvalues, 0.25, None))


def _entropy_from_spectrum(nu: np.ndarray) -> float:
    upper = nu + 0.5
    lower = nu - 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        lower_term = np.where(lower > 1e-300, lower * np.log(lower), 0.0)
    return float(np.sum(upper * np.log(upper) - lower_term))


def chain_entropy(n_sites: int, ell: int, n_inside: int) -> float:
    """Entropy S_l of one radial chain, excluding the (2l + 1) degeneracy."""
    return _entropy_from_spectrum(symplectic_spectrum(n_sites, ell, n_inside))


def entanglement_entropy(n_sites: int, n_inside: int,
                         ell_max: int = DEFAULT_ELL_MAX) -> float:
    """Total entropy of the ball of radius (n_inside + 1/2) lattice spacings.

    Sums (2l + 1) S_l over l up to ell_max. Truncating the sum underestimates
    the entropy; the tail falls slowly, so ell_max must considerably exceed
    n_inside.
    """
    return float(
        sum(
            (2 * ell + 1) * chain_entropy(n_sites, ell, n_inside)
            for ell in range(ell_max + 1)
        )
    )


def entropy_convergence(n_sites: int, n_inside: int,
                        ell_values: tuple[int, ...]) -> list[tuple[int, float]]:
    """Partial sums of the entropy against the angular momentum ceiling.

    Provided so that the truncation error can be reported rather than assumed.
    """
    contributions = [
        (2 * ell + 1) * chain_entropy(n_sites, ell, n_inside)
        for ell in range(max(ell_values) + 1)
    ]
    cumulative = np.cumsum(contributions)
    return [(ell, float(cumulative[ell])) for ell in ell_values]


def region_radius(n_inside: int) -> float:
    """Radius of the traced region in lattice spacings, R = n + 1/2."""
    return n_inside + 0.5
