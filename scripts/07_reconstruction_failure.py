"""Where geometric reconstruction from vacuum correlations breaks down.

Script 05 showed that inverting the free-space correlator and applying
multidimensional scaling returns the coordinates used to build the matrix.
That is circular: G was a monotone function of separation by construction.

The cavity correlator of :mod:`qvacuum.cavity` is not. The Dirichlet boundary
makes G depend on where a pair sits, not only on how far apart its points are,
so the inversion d = 1 / (2 pi sqrt(G)) is misspecified. This script measures
how badly, as a function of proximity to the wall.

Run:  python scripts/07_reconstruction_failure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qvacuum.cavity import CavityField
from qvacuum.correlators import correlation_matrix as free_correlation_matrix
from qvacuum.geometry import Sphere, pair_distances
from qvacuum.reconstruct import (
    classical_mds,
    euclidean_defect,
    inferred_distances,
    procrustes_rmse,
)

N_PER_RADIUS = 4
R_LAMBDA = 40.0
CAPS = (0.95, 0.85, 0.75, 0.65, 0.55, 0.45)
FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"


def reconstruct(g: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """Return RMSE in metres, Euclidean defect, and the dimensional gap."""
    d = inferred_distances(g)
    coords, eigenvalues = classical_mds(d, dim=3)
    gap = (
        eigenvalues[2] / abs(eigenvalues[3])
        if eigenvalues[3] != 0
        else float("inf")
    )
    return procrustes_rmse(coords, truth), euclidean_defect(eigenvalues), gap


def main() -> None:
    sphere = Sphere()
    spacing = sphere.lattice_spacing(N_PER_RADIUS)
    points = sphere.lattice_points(N_PER_RADIUS)
    radii = np.linalg.norm(points, axis=1)
    field = CavityField(
        radius=sphere.radius, regulator_scale=R_LAMBDA / sphere.radius
    )

    on_wall = int(np.isclose(radii, sphere.radius).sum())
    print(f"lattice spacing {spacing * 1e9:.1f} nm, {len(points)} sites, "
          f"R*Lambda = {R_LAMBDA:g}")
    print(f"{on_wall} sites lie exactly on the sphere, where the Dirichlet")
    print("condition makes G vanish identically. They are infinitely distant")
    print("under any inversion and are excluded from every cap below.\n")

    print("reconstruction quality against distance from the wall")
    print(f"{'r < cap*R':>10}{'sites':>7}{'RMSE/spacing':>14}"
          f"{'defect':>9}{'dim gap':>12}")
    rows = []
    for cap in CAPS:
        keep = radii < cap * sphere.radius
        subset = points[keep]
        if len(subset) < 20:
            continue
        g = field.correlation_matrix(subset)
        rmse, defect, gap = reconstruct(g, subset)
        rows.append((cap, len(subset), rmse / spacing, defect))
        print(f"{cap:10.2f}{len(subset):7d}{rmse / spacing:14.3f}"
              f"{defect:9.4f}{gap:12.2f}")

    print("\nbaseline: the free-space correlator on the same point sets")
    for cap in (0.95, 0.65):
        keep = radii < cap * sphere.radius
        subset = points[keep]
        g = free_correlation_matrix(subset, regulator=spacing)
        rmse, defect, _ = reconstruct(g, subset)
        print(f"{cap:10.2f}{len(subset):7d}{rmse / spacing:14.3e}"
              f"{defect:9.4f}")
    print("Exact recovery, as script 05 established. The degradation above is")
    print("caused by the boundary, not by the pipeline.")

    print("\ndistance error by position of the pair, for r < 0.85R")
    keep = radii < 0.85 * sphere.radius
    subset = points[keep]
    g = field.correlation_matrix(subset)
    inferred = inferred_distances(g)
    truth = pair_distances(subset)
    subset_radii = np.linalg.norm(subset, axis=1)
    mean_radius = 0.5 * (subset_radii[:, None] + subset_radii[None, :])
    off_diagonal = ~np.eye(len(subset), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.abs(inferred - truth) / np.where(truth > 0, truth, 1.0)
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85)):
        band = (
            off_diagonal
            & (mean_radius >= lo * sphere.radius)
            & (mean_radius < hi * sphere.radius)
        )
        if band.sum() < 5:
            continue
        print(f"  mean r/R in [{lo:.2f}, {hi:.2f}): median error "
              f"{np.median(error[band]):.3f}  (n = {band.sum()})")

    caps, _, rmses, defects = zip(*rows, strict=True)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_a.plot(caps, rmses, "o-", lw=1.2, ms=4)
    ax_a.set_xlabel("points restricted to $r <$ cap $\\times R$")
    ax_a.set_ylabel("reconstruction RMSE / lattice spacing")
    ax_a.set_yscale("log")
    ax_a.set_title("Recovery degrades towards the wall")
    ax_b.plot(caps, defects, "o-", lw=1.2, ms=4)
    ax_b.set_xlabel("points restricted to $r <$ cap $\\times R$")
    ax_b.set_ylabel("negative eigenvalue mass / positive")
    ax_b.set_title("Inferred distances cease to be Euclidean")
    fig.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    out = FIGURE_DIR / "07_reconstruction_failure.png"
    fig.savefig(out, dpi=150)
    print(f"\nfigure written to {out}")


if __name__ == "__main__":
    main()
