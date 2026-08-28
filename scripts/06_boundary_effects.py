"""Boundary effects on the vacuum correlator inside a Dirichlet sphere.

Three results:

1. The radial profile of <phi^2(r)>, which vanishes at the wall and recovers
   the unbounded-space value in the interior.
2. The subtracted profile near the wall, compared with the flat Dirichlet
   plane result -1 / (16 pi^2 d^2).
3. The correlator at fixed separation but varying position, which is the
   property that breaks the reconstruction circularity.

Run:  python scripts/06_boundary_effects.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qvacuum.cavity import CavityField, free_correlator_regulated
from qvacuum.geometry import Sphere

R_LAMBDA = 60.0
FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    sphere = Sphere()
    field = CavityField(
        radius=sphere.radius, regulator_scale=R_LAMBDA / sphere.radius
    )
    print(f"radius {sphere.radius:.3e} m, R*Lambda = {R_LAMBDA:g}")
    print(f"{len(field.spectrum.k)} (l, n) pairs enumerated, "
          f"l up to {field.spectrum.ell.max()}")
    print(f"unbounded-space reference <phi^2> = {field.free_reference:.5e}\n")

    fractions = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    profile = field.phi_squared(fractions * sphere.radius)
    print("radial profile of <phi^2>")
    print(f"{'r/R':>7}{'<phi^2>':>15}{'/free':>10}")
    for frac, value in zip(fractions, profile, strict=True):
        print(f"{frac:7.2f}{value:15.4e}{value / field.free_reference:10.4f}")

    print("\nnear-wall behaviour of the subtracted profile")
    print(f"{'d/R':>8}{'subtracted':>15}{'-1/16 pi^2 d^2':>18}{'ratio':>9}")
    for d_frac in (0.20, 0.15, 0.10, 0.07, 0.05, 0.03):
        d = d_frac * sphere.radius
        value = float(field.phi_squared_subtracted(sphere.radius - d)[0])
        plane = -1.0 / (16.0 * np.pi**2 * d**2)
        print(f"{d_frac:8.3f}{value:15.4e}{plane:18.4e}{value / plane:9.3f}")
    print("Agreement to roughly ten per cent in the window 1/Lambda << d << R.")
    print("The subtraction is not a renormalisation; see the docstring of")
    print("CavityField.phi_squared_subtracted for why values far from the wall")
    print("are regulator dependent.")

    print("\ncorrelator at fixed separation s = 0.2 R, varying position")
    s = 0.2 * sphere.radius
    free_value = float(free_correlator_regulated(s, field.regulator_scale)[0])
    print(f"{'r/R':>7}{'G':>15}{'/free space':>14}")
    print(f"{'centre':>7}{float(field.correlator(s / 2, s / 2, -1.0)[0]):15.4e}"
          f"{float(field.correlator(s / 2, s / 2, -1.0)[0]) / free_value:14.4f}")
    for frac in (0.5, 0.7, 0.8, 0.85, 0.9):
        r0 = frac * sphere.radius
        cos_gamma = 1 - s**2 / (2 * r0**2)
        value = float(field.correlator(r0, r0, cos_gamma)[0])
        print(f"{frac:7.2f}{value:15.4e}{value / free_value:14.4f}")
    print("Equal separations, unequal correlators. No inversion G -> d exists.")

    fine = np.linspace(0.0, 1.0, 200)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_a.plot(fine, field.phi_squared(fine * sphere.radius) / field.free_reference,
              lw=1.2)
    ax_a.axhline(1.0, color="k", ls="--", lw=0.6)
    ax_a.set_xlabel(r"$r / R$")
    ax_a.set_ylabel(r"$\langle\phi^2(r)\rangle\ /\ \langle\phi^2\rangle_{\rm free}$")
    ax_a.set_title("Fluctuation amplitude, Dirichlet sphere")

    d_frac = np.linspace(0.02, 0.30, 60)
    d = d_frac * sphere.radius
    subtracted = field.phi_squared_subtracted(sphere.radius - d)
    ax_b.plot(d_frac, -subtracted, lw=1.2, label="cavity, subtracted")
    ax_b.plot(d_frac, 1.0 / (16 * np.pi**2 * d**2), "--", lw=1.0,
              label=r"plane, $1/16\pi^2 d^2$")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"$d / R$")
    ax_b.set_ylabel(
        r"$-\left[\langle\phi^2\rangle"
        r" - \langle\phi^2\rangle_{\rm free}\right]$"
    )
    ax_b.legend(frameon=False, fontsize=9)
    ax_b.set_title("Near-wall behaviour")

    fig.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    out = FIGURE_DIR / "06_boundary_effects.png"
    fig.savefig(out, dpi=150)
    print(f"\nfigure written to {out}")


if __name__ == "__main__":
    main()
