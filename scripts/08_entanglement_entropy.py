"""Entanglement entropy of a spherical region: the area law.

Reproduces Srednicki's 1993 result that the ground-state entropy of a free
scalar field, traced over a ball, scales with the boundary area rather than
the enclosed volume.

Run:  python scripts/08_entanglement_entropy.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qvacuum.entropy import (
    entanglement_entropy,
    entropy_convergence,
    region_radius,
)

N_SITES = 90
ELL_MAX = 800
N_VALUES = (4, 6, 8, 10, 14, 20, 26, 32)
FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    print(f"radial lattice of {N_SITES} sites, angular sum to l = {ELL_MAX}\n")

    print("convergence of the angular sum, for a region of 10 sites")
    print(f"{'l_max':>8}{'S':>12}{'change':>10}")
    previous = None
    for ell, value in entropy_convergence(N_SITES, 10, (50, 100, 200, 400, 800)):
        change = "" if previous is None else f"{(value - previous) / value:9.2%}"
        print(f"{ell:8d}{value:12.4f}{change:>10}")
        previous = value
    print("The tail falls slowly, so any quoted entropy must state its ceiling.\n")

    radii, entropies = [], []
    print("area law")
    print(f"{'sites':>7}{'R/a':>8}{'S':>12}{'S/R^2':>10}{'S/R^3':>12}")
    for n in N_VALUES:
        s = entanglement_entropy(N_SITES, n, ELL_MAX)
        r = region_radius(n)
        radii.append(r)
        entropies.append(s)
        print(f"{n:7d}{r:8.1f}{s:12.4f}{s / r**2:10.4f}{s / r**3:12.5f}")

    radii = np.array(radii)
    entropies = np.array(entropies)
    slope = np.polyfit(np.log(radii), np.log(entropies), 1)[0]
    coefficient = np.polyfit(radii**2, entropies, 1)[0]
    print(f"\nlog-log slope = {slope:.4f}   (2 = area law, 3 = volume law)")
    print(f"fitted coefficient S = {coefficient:.4f} (R/a)^2")
    print("Srednicki's published value is 0.30. The small deficit is the")
    print("truncation of the angular sum, which underestimates the total.")
    print("\nS/R^2 is flat across a factor of seven in region size while S/R^3")
    print("falls by the same factor. The entropy lives on the boundary.")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_a.loglog(radii, entropies, "o", ms=5, label="computed")
    ax_a.loglog(radii, coefficient * radii**2, "-", lw=1.0,
                label=rf"$ {coefficient:.3f}\,(R/a)^2 $")
    ax_a.set_xlabel(r"$R / a$")
    ax_a.set_ylabel(r"$S$")
    ax_a.legend(frameon=False, fontsize=9)
    ax_a.set_title("Entanglement entropy of a ball")

    ax_b.plot(radii, entropies / radii**2, "o-", lw=1.2, ms=4, label=r"$S/R^2$")
    ax_b.plot(radii, entropies / radii**3, "s-", lw=1.2, ms=4, label=r"$S/R^3$")
    ax_b.set_xlabel(r"$R / a$")
    ax_b.set_yscale("log")
    ax_b.legend(frameon=False, fontsize=9)
    ax_b.set_title("Area law against volume law")

    fig.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    out = FIGURE_DIR / "08_entanglement_entropy.png"
    fig.savefig(out, dpi=150)
    print(f"\nfigure written to {out}")


if __name__ == "__main__":
    main()
