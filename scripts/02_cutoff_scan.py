"""N(Lambda) for the 1 um sphere across physically motivated cutoffs.

Reports the mode count from the bulk Weyl term over roughly thirty decades of
cutoff, together with the relative size of the Dirichlet boundary correction.
The headline observation is that no single number answers the question; the
count spans some eighty decades across the range of defensible cutoffs.

Run:  python scripts/02_cutoff_scan.py
"""

from __future__ import annotations

import numpy as np

from qvacuum.constants import C_LIGHT, energy_from_wavenumber
from qvacuum.cutoffs import NAMED_SCALES, cutoff_from_wavelength
from qvacuum.geometry import Sphere
from qvacuum.modes import weyl_count


def main() -> None:
    sphere = Sphere()
    print(f"sphere radius {sphere.radius:.3e} m, volume {sphere.volume:.4e} m^3\n")
    header = (f"{'scale':<24}{'Lambda [1/m]':>14}{'R*Lambda':>12}"
              f"{'N':>12}{'bdy/bulk':>12}{'E [eV]':>12}")
    print(header)
    print("-" * len(header))

    for name, wavelength in NAMED_SCALES.items():
        cutoff = cutoff_from_wavelength(wavelength)
        bulk = float(weyl_count(sphere.volume, cutoff, order=1))
        # Evaluated in closed form as 9 pi / (8 R Lambda); differencing the two
        # Weyl terms directly underflows once the cutoff is large.
        ratio = 9 * np.pi / (8 * sphere.radius * cutoff)
        print(f"{name:<24}{cutoff:14.3e}{sphere.radius * cutoff:12.3e}"
              f"{bulk:12.3e}{ratio:12.3e}{energy_from_wavenumber(cutoff):12.3e}")

    print()
    print("Boundary correction reaches 10 per cent of the bulk term at "
          f"R*Lambda = {9 * np.pi / 8 / 0.10:.1f}")
    print("and 1 per cent at "
          f"R*Lambda = {9 * np.pi / 8 / 0.01:.1f}.")
    print("The sphere's geometry is therefore only visible in the count for")
    print("cutoff wavelengths between roughly 30 and 300 nm.")

    print()
    print("Recovering the 'per second' part of the question:")
    print("a spatial mode is not consumed by the passage of time, so the count")
    print("does not have a rate. The nearest well posed quantity is the number")
    print("of independent spacetime samples at the Nyquist rate of the highest")
    print("mode, N_spacetime = N * omega_max * T / pi.")
    for name in ("molecular (1 nm)", "nuclear (1 fm)"):
        cutoff = cutoff_from_wavelength(NAMED_SCALES[name])
        n_spatial = float(weyl_count(sphere.volume, cutoff, order=1))
        omega_max = cutoff * C_LIGHT
        print(f"  {name:<22} N = {n_spatial:.3e}   "
              f"N_spacetime(T = 1 s) = {n_spatial * omega_max / np.pi:.3e}")


if __name__ == "__main__":
    main()
