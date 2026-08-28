"""Vacuum fluctuation amplitude <phi^2> and its cutoff dependence.

Compares the numerical mode sum over the enumerated cavity spectrum against
the continuum closed form, and shows the quadratic divergence.

Run:  python scripts/03_vacuum_fluctuations.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qvacuum.geometry import Sphere
from qvacuum.modes import enumerate_modes
from qvacuum.vacuum import phi_squared_analytic, phi_squared_from_spectrum

FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    sphere = Sphere()
    r_lambda_max = 120.0
    spectrum = enumerate_modes(sphere.radius, r_lambda_max / sphere.radius)

    r_lambda = np.linspace(20.0, r_lambda_max, 60)
    cutoffs = r_lambda / sphere.radius
    numerical = []
    for cutoff in cutoffs:
        keep = spectrum.k <= cutoff
        numerical.append(
            phi_squared_from_spectrum(
                spectrum.k[keep], spectrum.degeneracy[keep], sphere.volume
            )
        )
    numerical = np.array(numerical)
    analytic = np.array([phi_squared_analytic(c) for c in cutoffs])

    print(f"{'R*Lambda':>10}{'mode sum':>16}{'continuum':>16}{'ratio':>10}")
    for target in (20, 40, 60, 80, 100, 120):
        i = int(np.argmin(np.abs(r_lambda - target)))
        print(f"{r_lambda[i]:10.1f}{numerical[i]:16.4e}{analytic[i]:16.4e}"
              f"{numerical[i] / analytic[i]:10.4f}")

    print()
    print("The mode sum sits below the continuum result because the Dirichlet")
    print("spectrum is depleted near the boundary, the same effect that gives")
    print("the negative Weyl surface term. The ratio approaches unity as the")
    print("cutoff rises.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r_lambda, numerical, lw=1.2, label="cavity mode sum")
    ax.plot(r_lambda, analytic, "--", lw=1.0,
            label=r"continuum $\Lambda^2/8\pi^2$")
    ax.set_xlabel(r"$R\Lambda$")
    ax.set_ylabel(r"$\langle\phi^2\rangle$ (natural units)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(r"Vacuum fluctuation amplitude, $R = 0.5\ \mu$m")
    fig.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    out = FIGURE_DIR / "03_vacuum_fluctuations.png"
    fig.savefig(out, dpi=150)
    print(f"\nfigure written to {out}")


if __name__ == "__main__":
    main()
