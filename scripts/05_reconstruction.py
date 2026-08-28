"""Recovery of lattice coordinates from the correlation matrix.

For a free massless field in flat space this succeeds by construction, since
G is a monotone function of separation. The script is a pipeline check, and
the printed residual is a statement about numerical accuracy, not physics.

The intended use is the failure study: rerun with a mass, a lower cutoff, or
added noise, and find where the recovery breaks down.

Run:  python scripts/05_reconstruction.py
"""

from __future__ import annotations

import numpy as np

from qvacuum.correlators import correlation_matrix
from qvacuum.geometry import Sphere
from qvacuum.reconstruct import (
    classical_mds,
    distance_from_correlator,
    procrustes_rmse,
)

N_PER_RADIUS = 6


def main() -> None:
    sphere = Sphere()
    points = sphere.lattice_points(N_PER_RADIUS)
    spacing = sphere.lattice_spacing(N_PER_RADIUS)
    print(f"lattice spacing {spacing:.3e} m, {len(points)} sites\n")

    g = correlation_matrix(points, mass=0.0, regulator=spacing)
    d = distance_from_correlator(g, mass=0.0)
    np.fill_diagonal(d, 0.0)

    coords, eigenvalues = classical_mds(d, dim=3)
    rmse = procrustes_rmse(coords, points)

    print("MDS eigenvalue spectrum, leading six:")
    for i, value in enumerate(eigenvalues[:6]):
        print(f"  lambda_{i + 1} = {value:12.4e}")
    gap = eigenvalues[2] / abs(eigenvalues[3]) if eigenvalues[3] != 0 else np.inf
    print(f"\ndimensional gap lambda_3 / |lambda_4| = {gap:.3e}")
    print(f"reconstruction RMSE = {rmse:.3e} m "
          f"({rmse / sphere.radius:.3e} of the radius)")

    print()
    print("This is circular: the correlator was built from these coordinates,")
    print("so recovering them tests the code and nothing else. The result of")
    print("interest is where the recovery stops working.")


if __name__ == "__main__":
    main()
