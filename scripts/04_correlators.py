"""Equal-time correlator and the mass-induced correlation length.

Plots G(r) for a massless field and for several masses, showing the crossover
from the 1/r^2 power law to Yukawa decay at r ~ 1/m.

Run:  python scripts/04_correlators.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qvacuum.correlators import correlator_free
from qvacuum.geometry import Sphere

FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    sphere = Sphere()
    r = np.logspace(np.log10(1e-9), np.log10(2 * sphere.radius), 400)

    masses = [0.0, 1.0 / sphere.radius, 10.0 / sphere.radius,
              100.0 / sphere.radius]
    labels = ["massless", r"$mR = 1$", r"$mR = 10$", r"$mR = 100$"]

    print(f"{'mass [1/m]':>14}{'1/m [m]':>14}{'G at R/10':>14}{'G at R':>14}")
    for mass in masses:
        g_small = float(correlator_free(np.array([sphere.radius / 10]), mass)[0])
        g_big = float(correlator_free(np.array([sphere.radius]), mass)[0])
        length = np.inf if mass == 0 else 1 / mass
        print(f"{mass:14.3e}{length:14.3e}{g_small:14.4e}{g_big:14.4e}")

    print()
    print("A massless field has no correlation length: G falls as a power law")
    print("across the whole sphere. A mass introduces the scale 1/m, beyond")
    print("which correlations are exponentially suppressed. This is the")
    print("control parameter for the reconstruction study in script 05.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mass, label in zip(masses, labels, strict=True):
        ax.plot(r / sphere.radius, correlator_free(r, mass), lw=1.2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r / R$")
    ax.set_ylabel(r"$\langle\phi(x)\phi(y)\rangle$ (natural units)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(r"Equal-time vacuum correlator, $R = 0.5\ \mu$m")
    fig.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    out = FIGURE_DIR / "04_correlators.png"
    fig.savefig(out, dpi=150)
    print(f"\nfigure written to {out}")


if __name__ == "__main__":
    main()
