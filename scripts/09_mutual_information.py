"""Mutual information: UV finiteness, and a negative reconstruction result.

Every entropy in script 08 is cutoff dependent, scaling as the boundary area
in lattice units and diverging as the spacing goes to zero. The mutual
information between two disjoint regions,

    I(A:B) = S_A + S_B - S_AB,

is finite in the continuum limit, because each divergent boundary contributes
to S_AB as well as to S_A or S_B and cancels. Part one demonstrates that.

Part two asks whether this better-behaved quantity repairs the reconstruction
that failed in script 07. The question is left open: the pairwise construction
rests on a cell smearing with no valid window, so its numbers are not
quantitatively trustworthy.

Run:  python scripts/09_mutual_information.py
"""

from __future__ import annotations

import numpy as np

from qvacuum.cavity import CavityField, smeared_covariance
from qvacuum.entropy import (
    entanglement_entropy,
    mutual_information,
    pairwise_mutual_information,
)
from qvacuum.geometry import Sphere, pair_distances
from qvacuum.reconstruct import (
    classical_mds,
    euclidean_defect,
    inferred_distances,
)

N_PER_RADIUS = 3
R_LAMBDA = 40.0


def uv_finiteness() -> None:
    print("Part 1: mutual information is UV finite\n")
    print("Geometry held fixed (A is the inner fifth, B the shell from two")
    print("fifths to three fifths) while the lattice is refined.\n")
    print(f"{'scale':>6}{'sites':>7}{'l_max':>7}{'I':>12}{'S_A':>12}")
    for scale in (2, 3, 4, 5, 6, 8):
        n_sites = 20 * scale
        region_a = np.arange(0, 2 * scale)
        region_b = np.arange(4 * scale, 6 * scale)
        ell_max = 30 * scale
        info = mutual_information(n_sites, region_a, region_b, ell_max)
        entropy = entanglement_entropy(n_sites, 2 * scale, ell_max)
        print(f"{scale:6d}{n_sites:7d}{ell_max:7d}{info:12.5f}{entropy:12.4f}")
    print("\nI settles towards a constant while S_A grows as the area.")
    print("Only I survives the continuum limit.")

    print("\ndecay with the gap between the regions, 90 sites, l_max = 200")
    print(f"{'gap':>6}{'I':>12}")
    region_a = np.arange(0, 10)
    for gap in (2, 5, 10, 20, 30):
        region_b = np.arange(10 + gap, 20 + gap)
        print(f"{gap:6d}{mutual_information(90, region_a, region_b, 200):12.5f}")


def reconstruction() -> None:
    print("\n\nPart 2: pairwise reconstruction, inconclusive\n")
    sphere = Sphere()
    spacing = sphere.lattice_spacing(N_PER_RADIUS)
    points = sphere.lattice_points(N_PER_RADIUS)
    radii = np.linalg.norm(points, axis=1)
    points = points[radii < 0.95 * sphere.radius]
    field = CavityField(
        radius=sphere.radius, regulator_scale=R_LAMBDA / sphere.radius
    )

    x, p = smeared_covariance(field, points, spacing)
    single = np.sqrt(np.diag(x) * np.diag(p))
    print(f"{len(points)} sites, Lambda * a = {R_LAMBDA * spacing / sphere.radius:.1f}")
    print(f"single-site symplectic eigenvalues {single.min():.1f} to "
          f"{single.max():.1f} (must exceed 0.5)\n")

    info = pairwise_mutual_information(x, p)
    distances = pair_distances(points)
    off = ~np.eye(len(points), dtype=bool)
    order = np.argsort(distances[off])
    print("I is not a monotone function of separation either: "
          f"{np.mean(np.diff(info[off][order]) > 0):.4f} of adjacent pairs rise.")

    print("\nEuclidean defect of the inferred distances, by inversion law:")
    print(f"{'law':>34}{'defect':>10}")
    for exponent, label in (
        (-4.0, "analytic free-field, I ~ d^-4"),
        (-6.0, "arbitrary, I ~ d^-6"),
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(info > 0, info, np.nan) ** (1.0 / exponent)
        largest = np.nanmax(d[np.isfinite(d)])
        d = np.nan_to_num(d, nan=largest, posinf=largest)
        np.fill_diagonal(d, 0.0)
        d = 0.5 * (d + d.T)
        _, eigenvalues = classical_mds(d, dim=3)
        print(f"{label:>34}{euclidean_defect(eigenvalues):10.4f}")

    _, two_point = classical_mds(inferred_distances(x), dim=3)
    print(f"{'two-point function, exact free law':>34}"
          f"{euclidean_defect(two_point):10.4f}")

    print("\nTwo reasons not to read a conclusion into these numbers.")
    print("First, fitting the exponent against the true distances gives 0.38,")
    print("apparently beating the two-point function, but that is circular: the")
    print("two-point inversion uses the exact free-space law with no fitted")
    print("parameter. Second, and more seriously, the cell smearing has no valid")
    print("window. At this Lambda * a the single-site eigenvalues scale as")
    print("(Lambda a)^3, showing the midpoint evaluation is dominated by the")
    print("regulated delta function; lowering Lambda * a towards pi drives them")
    print("below one half, which is unphysical and now raises. Whether mutual")
    print("information reconstructs better than the two-point function is")
    print("therefore not settled here. Proper cell integration would settle it.")


def main() -> None:
    uv_finiteness()
    reconstruction()


if __name__ == "__main__":
    main()
