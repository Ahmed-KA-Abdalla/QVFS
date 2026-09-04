"""Thermal states: the area law is a property of the vacuum, not of entropy.

Script 08 found that the ground-state entropy of a ball scales with its
boundary area. Raising the temperature destroys that. The thermal factor
coth(omega / 2T) adds an extensive contribution, and the scaling crosses over
from the area law to a volume law.

The distinction matters wherever Srednicki's result is invoked in a black hole
context: the area scaling belongs specifically to the vacuum, and any state
with appreciable thermal occupation does not show it.

At T > 0 the global state is mixed, so the entropy of a region is no longer a
measure of entanglement alone. It mixes entanglement with ordinary thermal
entropy, and at high temperature the latter dominates.

Run:  python scripts/11_thermal_states.py
"""

from __future__ import annotations

import numpy as np

from qvacuum.entropy import (
    bose_entropy,
    chain_entropy,
    entanglement_entropy,
    region_radius,
)

N_SITES = 60
ELL_MAX = 300
N_VALUES = (4, 6, 8, 12, 16, 20)
TEMPERATURES = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)


def main() -> None:
    print(f"radial lattice of {N_SITES} sites, angular sum to l = {ELL_MAX}")
    print("temperature in inverse lattice spacings\n")

    print("validation: tracing over nothing must give the Bose thermal entropy")
    print(f"{'T':>6}{'covariance':>14}{'occupation':>14}{'difference':>14}")
    for temperature in (0.1, 0.3, 1.0, 3.0):
        numeric = chain_entropy(30, 3, 30, temperature)
        analytic = bose_entropy(30, 3, temperature)
        print(f"{temperature:6.1f}{numeric:14.6f}{analytic:14.6f}"
              f"{abs(numeric - analytic):14.2e}")

    radii = np.array([region_radius(n) for n in N_VALUES])
    print("\ncrossover from area law to volume law")
    print(f"{'T':>6}{'slope':>9}{'S/R^2':>12}{'S/R^3':>12}{'S(R=20.5)':>13}")
    for temperature in TEMPERATURES:
        entropies = np.array([
            entanglement_entropy(N_SITES, n, ELL_MAX, temperature)
            for n in N_VALUES
        ])
        slope = np.polyfit(np.log(radii), np.log(entropies), 1)[0]
        print(f"{temperature:6.2f}{slope:9.3f}{entropies[-1] / radii[-1]**2:12.4f}"
              f"{entropies[-1] / radii[-1]**3:12.5f}{entropies[-1]:13.2f}")

    print("\nAt T = 0 the slope is 2, the area law. By T = 1 it is 3, a volume")
    print("law, and the entropy is two orders of magnitude larger. The")
    print("intermediate values are not a third scaling law but a sum of the")
    print("two contributions, with the crossover radius falling as T rises.")

    print("\nwhere the two contributions balance, for T = 0.2")
    print(f"{'R/a':>7}{'S':>12}{'area part':>12}{'excess':>12}")
    cold = {
        n: entanglement_entropy(N_SITES, n, ELL_MAX, 0.0) for n in N_VALUES
    }
    for n in N_VALUES:
        hot = entanglement_entropy(N_SITES, n, ELL_MAX, 0.2)
        print(f"{region_radius(n):7.1f}{hot:12.3f}{cold[n]:12.3f}"
              f"{hot - cold[n]:12.3f}")
    print("The excess over the vacuum result grows faster than the area term,")
    print("which is what drives the exponent upward.")


if __name__ == "__main__":
    main()
