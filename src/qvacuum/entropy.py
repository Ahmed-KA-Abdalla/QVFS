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


def covariance_matrices(n_sites: int, ell: int, temperature: float = 0.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Covariance matrices X = <phi phi> and P = <pi pi> at temperature T.

    Each normal mode of frequency omega carries a thermal factor
    coth(omega / 2T), which is one at T = 0 and grows as 2T / omega in the
    classical limit. Temperature is measured in inverse lattice spacings,
    consistent with the rest of the module.

    At T = 0 the product X P is the identity over four, the statement that the
    global state is pure. At T > 0 it is not: the global state is mixed, and
    the entropy of a region then mixes thermal and entanglement contributions
    rather than measuring entanglement alone.
    """
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    eigenvalues, vectors = eigh(radial_coupling_matrix(n_sites, ell))
    if eigenvalues.min() <= 0:
        raise ValueError("coupling matrix is not positive definite")
    omega = np.sqrt(eigenvalues)
    if temperature == 0.0:
        thermal = np.ones_like(omega)
    else:
        thermal = 1.0 / np.tanh(omega / (2.0 * temperature))
    x = (vectors * (thermal / (2.0 * omega))) @ vectors.T
    p = (vectors * (omega * thermal / 2.0)) @ vectors.T
    return x, p


def bose_entropy(n_sites: int, ell: int, temperature: float) -> float:
    """Thermal entropy of a whole chain, from the Bose occupation numbers.

    S = sum_k [ (n_k + 1) ln(n_k + 1) - n_k ln n_k ],  n_k = 1 / (e^(w/T) - 1)

    Independent of the covariance machinery, so tracing over nothing must
    reproduce it exactly. This is the validation that the thermal factor has
    been applied correctly.
    """
    if temperature <= 0:
        return 0.0
    omega = np.sqrt(np.linalg.eigvalsh(radial_coupling_matrix(n_sites, ell)))
    occupation = 1.0 / np.expm1(omega / temperature)
    return float(
        np.sum(
            (occupation + 1) * np.log1p(occupation)
            - occupation * np.log(occupation)
        )
    )


def symplectic_spectrum(n_sites: int, ell: int, n_inside: int,
                        temperature: float = 0.0) -> np.ndarray:
    """Eigenvalues nu_k of sqrt(X_A P_A) for the innermost n_inside sites.

    Each nu is at least one half; a value of exactly one half contributes no
    entropy, and the excess above one half measures entanglement across the
    boundary.
    """
    if not 0 < n_inside <= n_sites:
        raise ValueError("n_inside must lie between 1 and n_sites")
    x, p = covariance_matrices(n_sites, ell, temperature)
    product = x[:n_inside, :n_inside] @ p[:n_inside, :n_inside]
    eigenvalues = np.linalg.eigvals(product).real
    return np.sqrt(np.clip(eigenvalues, 0.25, None))


def _entropy_from_spectrum(nu: np.ndarray) -> float:
    upper = nu + 0.5
    lower = nu - 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        lower_term = np.where(lower > 1e-300, lower * np.log(lower), 0.0)
    return float(np.sum(upper * np.log(upper) - lower_term))


def chain_entropy(n_sites: int, ell: int, n_inside: int,
                  temperature: float = 0.0) -> float:
    """Entropy S_l of one radial chain, excluding the (2l + 1) degeneracy."""
    return _entropy_from_spectrum(
        symplectic_spectrum(n_sites, ell, n_inside, temperature)
    )


def entanglement_entropy(n_sites: int, n_inside: int,
                         ell_max: int = DEFAULT_ELL_MAX,
                         temperature: float = 0.0) -> float:
    """Total entropy of the ball of radius (n_inside + 1/2) lattice spacings.

    Sums (2l + 1) S_l over l up to ell_max. Truncating the sum underestimates
    the entropy; the tail falls slowly, so ell_max must considerably exceed
    n_inside.
    """
    return float(
        sum(
            (2 * ell + 1) * chain_entropy(n_sites, ell, n_inside, temperature)
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


def region_entropy(x: np.ndarray, p: np.ndarray,
                   sites: np.ndarray) -> float:
    """Entropy of an arbitrary subset of sites, given covariance matrices.

    Unlike :func:`chain_entropy` the region need not be contiguous, which is
    what makes mutual information between separated regions accessible.
    """
    sites = np.asarray(sites, dtype=int)
    index = np.ix_(sites, sites)
    eigenvalues = np.linalg.eigvals(x[index] @ p[index]).real
    return _entropy_from_spectrum(np.sqrt(np.clip(eigenvalues, 0.25, None)))


def mutual_information_chain(n_sites: int, ell: int, region_a: np.ndarray,
                             region_b: np.ndarray) -> float:
    """I(A:B) = S_A + S_B - S_AB for one radial chain.

    Non-negative by subadditivity, and finite in the continuum limit: the
    area-law divergences of the three terms cancel, since every boundary
    appearing in S_A or S_B also appears in S_AB.
    """
    x, p = covariance_matrices(n_sites, ell)
    union = np.concatenate([np.asarray(region_a), np.asarray(region_b)])
    if len(np.unique(union)) != len(union):
        raise ValueError("regions must be disjoint")
    return (
        region_entropy(x, p, region_a)
        + region_entropy(x, p, region_b)
        - region_entropy(x, p, union)
    )


def mutual_information(n_sites: int, region_a: np.ndarray, region_b: np.ndarray,
                       ell_max: int = 200) -> float:
    """Total I(A:B) summed over angular momentum sectors.

    Converges in l far faster than the entropy does, because the divergent
    boundary contributions have already cancelled. A ceiling of 200 is usually
    ample where the entropy needs 800.
    """
    return float(
        sum(
            (2 * ell + 1)
            * mutual_information_chain(n_sites, ell, region_a, region_b)
            for ell in range(ell_max + 1)
        )
    )


def pairwise_mutual_information(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """I(i:j) between every pair of single sites, from covariance matrices.

    The matrices must describe canonical variables, satisfying
    [phi_i, pi_j] = i delta_ij. Continuum correlators evaluated at points do
    not: they are delta-normalised densities, and using them directly gives
    single-site symplectic eigenvalues of order 10^22. See
    :func:`qvacuum.cavity.CavityField.smeared_covariance` for the cell
    averaging that produces canonical variables from them.
    """
    n = len(x)
    single = np.array([
        _entropy_from_spectrum(
            np.sqrt(np.clip(np.array([x[i, i] * p[i, i]]), 0.25, None))
        )
        for i in range(n)
    ])
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            index = np.ix_([i, j], [i, j])
            eigenvalues = np.linalg.eigvals(x[index] @ p[index]).real
            joint = _entropy_from_spectrum(
                np.sqrt(np.clip(eigenvalues, 0.25, None))
            )
            out[i, j] = out[j, i] = single[i] + single[j] - joint
    return out


def region_radius(n_inside: int) -> float:
    """Radius of the traced region in lattice spacings, R = n + 1/2."""
    return n_inside + 0.5
