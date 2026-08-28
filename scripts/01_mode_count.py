"""Exact Dirichlet mode count of the sphere against the Weyl asymptotic law.

Enumerates the eigenmodes j_l(k r) Y_lm of a sphere of radius 0.5 um with
k R below a maximum, and compares the counting function N(Lambda) with the
bulk Weyl term and with the two-term expansion including the Dirichlet
boundary correction.

Run:  python scripts/01_mode_count.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qvacuum.geometry import Sphere
from qvacuum.modes import enumerate_modes, lattice_count, weyl_count

R_LAMBDA_MAX = 120.0
FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    sphere = Sphere()
    cutoff_max = R_LAMBDA_MAX / sphere.radius

    print(f"radius        {sphere.radius:.3e} m")
    print(f"volume        {sphere.volume:.6e} m^3")
    print(f"surface area  {sphere.surface_area:.6e} m^2")
    print(f"enumerating modes to R*Lambda = {R_LAMBDA_MAX:g} ...")

    spectrum = enumerate_modes(sphere.radius, cutoff_max)
    print(f"  {len(spectrum.k)} (l, n) pairs, {spectrum.total} modes with degeneracy")
    print(f"  highest l reached: {spectrum.ell.max()}")

    r_lambda = np.linspace(5.0, R_LAMBDA_MAX, 400)
    cutoffs = r_lambda / sphere.radius
    exact = spectrum.counting_function(cutoffs).astype(float)
    bulk = weyl_count(sphere.volume, cutoffs, order=1)
    two_term = weyl_count(sphere.volume, cutoffs, sphere.surface_area, order=2)

    print()
    print(f"{'R*Lambda':>10} {'exact':>10} {'bulk':>12} {'two-term':>12}"
          f" {'bulk err':>10} {'2-term err':>11}")
    for target in (10, 20, 40, 60, 80, 100, 120):
        i = int(np.argmin(np.abs(r_lambda - target)))
        if exact[i] == 0:
            continue
        print(f"{r_lambda[i]:10.1f} {exact[i]:10.0f} {bulk[i]:12.1f}"
              f" {two_term[i]:12.1f} {(bulk[i] - exact[i]) / exact[i]:9.2%}"
              f" {(two_term[i] - exact[i]) / exact[i]:10.2%}")

    print()
    print("Lattice-site count for the same cutoffs, as the alternative and")
    print("inequivalent definition of the degree-of-freedom count:")
    for target in (40, 80, 120):
        i = int(np.argmin(np.abs(r_lambda - target)))
        sites = lattice_count(sphere.volume, cutoffs[i])
        print(f"  R*Lambda = {r_lambda[i]:6.1f}   cavity {exact[i]:9.0f}"
              f"   lattice {sites:9.1f}   ratio {exact[i] / sites:6.2f}")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_top.plot(r_lambda, exact, lw=1.2, label="exact Dirichlet count")
    ax_top.plot(r_lambda, bulk, "--", lw=1.0, label=r"Weyl bulk $V\Lambda^3/6\pi^2$")
    ax_top.plot(r_lambda, two_term, ":", lw=1.2,
                label=r"two-term, $-S\Lambda^2/16\pi$")
    ax_top.set_ylabel(r"$N(\Lambda)$")
    ax_top.set_yscale("log")
    ax_top.legend(frameon=False, fontsize=9)
    ax_top.set_title(r"Mode count of a sphere, $R = 0.5\ \mu$m")

    with np.errstate(divide="ignore", invalid="ignore"):
        ax_bot.plot(r_lambda, (bulk - exact) / exact, "--", lw=1.0, label="bulk")
        ax_bot.plot(r_lambda, (two_term - exact) / exact, ":", lw=1.2,
                    label="two-term")
    ax_bot.axhline(0.0, color="k", lw=0.5)
    ax_bot.set_xlabel(r"$R\Lambda$")
    ax_bot.set_ylabel("fractional residual")
    ax_bot.set_ylim(-0.5, 0.5)
    ax_bot.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    out = FIGURE_DIR / "01_mode_count.png"
    fig.savefig(out, dpi=150)
    print(f"\nfigure written to {out}")


if __name__ == "__main__":
    main()
