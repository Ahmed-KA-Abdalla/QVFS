"""Why cell averaging cannot produce canonical variables from a regulated field.

Script 09 built pairwise mutual information from covariance matrices obtained
by evaluating the continuum correlators at lattice points and scaling by a^6.
That is a midpoint approximation to a cell average, and it appeared to work:
the single-site symplectic eigenvalues came out near 80, comfortably above the
uncertainty bound of one half.

Performing the cell integral properly shows the appearance was a quadrature
artefact. As the order rises the eigenvalues fall monotonically and pass below
one half, where no physical Gaussian state can lie.

The reason is structural. Writing m_alpha for the cell average of mode alpha
and f_alpha for the regulator,

    X_ii P_ii  >=  a^6 ( sum_alpha f_alpha m_alpha^2 / 2 )^2

by Cauchy-Schwarz. Completeness gives sum_alpha m_alpha^2 = a^-3 at f = 1,
making the bound exactly one quarter. Any regulator lowers it. A regulated
continuum field restricted to cells is therefore not a canonical system, and
no amount of numerical care will make it one.

Run:  python scripts/10_cell_integration.py
"""

from __future__ import annotations

import numpy as np

from qvacuum.cavity import (
    CavityField,
    cell_nodes_weights,
    momentum_correlation_matrix,
)
from qvacuum.entropy import covariance_matrices
from qvacuum.geometry import Sphere

N_PER_RADIUS = 3
ORDERS = (1, 2, 3, 4, 6)
SITE_SAMPLE = (0, 5, 12, 20, 30)


def symplectic_minimum(field: CavityField, points: np.ndarray,
                       spacing: float, order: int) -> float:
    nodes, weights = cell_nodes_weights(points, spacing, order)
    n_points, n_nodes = len(points), len(weights)
    x = np.einsum(
        "q,iqjr,r->ij", weights,
        field.correlation_matrix(nodes).reshape(
            n_points, n_nodes, n_points, n_nodes),
        weights,
    )
    p = np.einsum(
        "q,iqjr,r->ij", weights,
        momentum_correlation_matrix(field, nodes).reshape(
            n_points, n_nodes, n_points, n_nodes),
        weights,
    ) * spacing**6
    return float(np.sqrt(np.diag(x) * np.diag(p)).min())


def main() -> None:
    sphere = Sphere()
    spacing = sphere.lattice_spacing(N_PER_RADIUS)
    points = sphere.lattice_points(N_PER_RADIUS)
    margin = sphere.radius - np.sqrt(3) / 2 * spacing
    contained = np.linalg.norm(points, axis=1) <= margin
    points = points[contained][list(SITE_SAMPLE)]

    print(f"{len(points)} sample sites, spacing {spacing * 1e9:.1f} nm")
    print("cells restricted to lie entirely inside the sphere\n")
    print("smallest single-site symplectic eigenvalue, which must exceed 0.5")
    print(f"{'Lambda*a':>10}{'order 1':>10}{'order 2':>10}{'order 3':>10}"
          f"{'order 4':>10}{'order 6':>10}")

    for lambda_a in (6.0, 12.0):
        field = CavityField(
            radius=sphere.radius, regulator_scale=lambda_a / spacing
        )
        row = [
            symplectic_minimum(field, points, spacing, order)
            for order in ORDERS
        ]
        print(f"{lambda_a:10.1f}" + "".join(f"{v:10.4f}" for v in row))

    print("\nOrder 1 is the midpoint evaluation used in script 09. Refining it")
    print("does not converge towards the midpoint value; it falls away from it,")
    print("monotonically, through the uncertainty bound. The midpoint figure was")
    print("a quadrature error that happened to point in the flattering direction.")

    x, p = covariance_matrices(40, 3)
    residual = np.abs(x @ p - np.eye(40) / 4).max()
    print("\nBy contrast the radial chains of qvacuum.entropy, where the theory")
    print("is discretised before the correlators are computed, satisfy")
    print(f"X P = I / 4 to {residual:.2e}. Discretise first, then correlate.")
    print("\nConsequence: the pairwise mutual information in script 09 is")
    print("withdrawn as a quantitative result. The UV finiteness result in")
    print("part 1 of that script is unaffected, since it uses the radial")
    print("chains and involves no smearing.")


if __name__ == "__main__":
    main()
