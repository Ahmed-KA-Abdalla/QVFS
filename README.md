[![ci](https://github.com/Ahmed-KA-Abdalla/QVFS/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmed-KA-Abdalla/QVFS/actions/workflows/ci.yml)

# Vacuum fluctuations in a 1 μm sphere

A numerical study of the ground state of a free real scalar field inside a
sphere of radius 0.5 μm (diameter 1 μm), in 3+1 dimensions.

The field is free and Gaussian, so every quantity computed here has a closed
form. The numerics exist to be checked against it and to reach regimes that
are awkward to evaluate by hand. Nothing in this repository constitutes novel
research, and no claim is made about spacetime emerging from vacuum
correlations.

## Questions

**How many independent degrees of freedom are inside the sphere?**

The question has no answer until an ultraviolet cutoff and a definition of
"degree of freedom" are fixed. The repository makes both explicit and reports
how sensitive the count is to each choice. Across cutoffs from optical to
Planckian the count runs from order unity to about 10⁸⁵.

**Can spatial structure be recovered from the correlation matrix?**

For a free field in flat space, trivially yes, because the correlator is a
monotone function of separation by construction. Recovering the coordinates
therefore tests the code and nothing else. The informative question is where
the recovery fails: as a mass is switched on, as modes are truncated, as the
lattice is coarse-grained, and as noise is added.

## Conventions

Natural units, ħ = c = 1, with lengths in metres. Wavenumbers, frequencies and
masses are all in m⁻¹. Conversion to SI happens only in `constants.py`.

The cutoff convention is Λ = π/λ_min, matching the Nyquist wavenumber of a
lattice of spacing λ_min. The alternative 2π/λ_min changes every count by a
factor of eight.

## Results so far

Exact Dirichlet mode count against the Weyl asymptotic law,
N(Λ) = VΛ³/6π² − SΛ²/16π + O(Λ):

| RΛ | exact | bulk term | two-term | bulk error | two-term error |
|---:|---:|---:|---:|---:|---:|
| 20 | 456 | 564.8 | 464.9 | +23.9% | +2.0% |
| 40 | 4 133 | 4 484.7 | 4 087.2 | +8.5% | −1.1% |
| 80 | 34 472 | 36 131.6 | 34 534.1 | +4.8% | +0.2% |
| 120 | 118 698 | 122 231.0 | 118 631.0 | +3.0% | −0.1% |

The boundary term carries the sign and magnitude the Dirichlet Weyl law
predicts, and reduces the residual by one to two orders of magnitude. Its
relative size is 9π/(8RΛ), so it falls below 10% at RΛ ≈ 35 and below 1% at
RΛ ≈ 353. The sphere's geometry is visible in the mode count only for cutoff
wavelengths between roughly 30 and 300 nm; above that the region is
indistinguishable from any other of equal volume.

The cavity count and the lattice-site count differ by a factor near 0.5. Most
of this is the cutoff convention — V/a³ with a = π/Λ exceeds the bulk Weyl term
by 6/π — rather than anything physical. The point stands that the two
definitions are not interchangeable, but the factor should not be
over-interpreted.

⟨φ²⟩ from the cavity mode sum approaches the continuum value Λ²/8π² from below,
reaching 96% at RΛ = 120. The deficit is the same boundary depletion that
produces the negative Weyl surface term.

A sharp momentum cutoff does not converge pointwise. For m = 0 the truncated
correlator is exactly (1 − cos Λr)/4π²r², which oscillates between zero and
twice the continuum value however large Λ becomes. Only the cycle average
converges. A Gaussian regulator is provided for cases where the correlator
itself, rather than an integral over it, is the object of interest.

## Layout

```
src/qvacuum/
    constants.py      units, physical constants, SI conversion
    geometry.py       sphere, cubic lattice clipped to it
    cutoffs.py        cutoff conventions and named physical scales
    modes.py          Dirichlet eigenmodes, Weyl law, lattice count
    vacuum.py         per-mode amplitudes, <phi^2>
    correlators.py    equal-time two-point functions and regulators
    reconstruct.py    distance inversion, classical MDS, Procrustes

scripts/
    01_mode_count.py           exact count against Weyl law
    02_cutoff_scan.py          N(Lambda) across physical scales
    03_vacuum_fluctuations.py  <phi^2> and its divergence
    04_correlators.py          G(r) and the mass correlation length
    05_reconstruction.py       MDS pipeline check

tests/                validation against analytic results
docs/                 derivations and computational limits
figures/              generated output, not tracked
```

## Installation

```
uv venv
uv pip install -e ".[dev]"
```

Or with pip:

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running

```
python scripts/01_mode_count.py
pytest
ruff check .
```

## Validation

Every module is checked against a closed form. Zeros of j₀ against nπ; the
counting function against the two-term Weyl law; ⟨φ²⟩ against Λ²/8π²; the
massive correlator against its massless limit and against exponential decay;
the sharp-cutoff correlator against (1 − cos Λr)/4π²r²; MDS against the
coordinates used to build the matrix. A module without such a check is not
merged.

## Not implemented

Standard Model field content. Interacting λφ⁴ theory, which would require
Euclidean lattice Monte Carlo and a compiled backend. Entanglement entropy from
the Gaussian covariance matrix, which is the intended second phase. A
boundary-aware correlator built from the Dirichlet mode functions; the
correlation matrix currently uses the free-space form and so knows nothing
about the sphere wall.

## Licence

MIT.
