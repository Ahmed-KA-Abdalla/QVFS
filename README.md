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

Building the correlator from the Dirichlet mode functions rather than the
free-space form changes the picture. The regulated fluctuation amplitude
recovers the unbounded-space value to within 0.1% for r < 0.6R and falls to
zero at the wall, as the boundary condition requires. Subtracting the
unbounded-space value reproduces the flat Dirichlet plane result
−1/16π²d² to about 8% in the window 1/Λ ≪ d ≪ R.

That subtraction is not a renormalisation. Removing the free-space mode sum
term by term leaves a residual that grows with the regulator scale, because
the cavity and free-space mode labels are not the same quantity. The
cutoff-independent construction is point splitting. Values far from the wall
are therefore regulator dependent and are not reported as physical.

The consequence for the reconstruction study is that the cavity correlator is
not a function of separation alone. At a fixed separation of 0.2R the
correlator falls from 97% of the free-space value for a pair straddling the
centre to 49% for a pair at r = 0.9R. No inversion G → d is well defined, so
the MDS pipeline is no longer circular when applied to this correlator.

Feeding the cavity correlator into the reconstruction pipeline breaks it, in a
way that can be measured. Inverting under the free-space law d = 1/2π√G and
applying classical MDS recovers the lattice exactly from the free-space
correlator (RMSE ~10⁻¹⁵ lattice spacings) and progressively fails from the
cavity correlator as points are allowed closer to the wall:

| points restricted to | sites | RMSE / spacing | Euclidean defect |
|---:|---:|---:|---:|
| r < 0.45R | 27 | 0.42 | 0.16 |
| r < 0.55R | 33 | 0.60 | 0.20 |
| r < 0.65R | 81 | 1.61 | 0.29 |
| r < 0.75R | 93 | 2.17 | 0.35 |
| r < 0.85R | 171 | 6.02 | 0.51 |
| r < 0.95R | 251 | 20.09 | 0.73 |

The Euclidean defect is the negative eigenvalue mass of the doubly centred
matrix as a fraction of the positive. It is zero for genuine distances and
rises to 0.73, meaning the inferred distances do not embed in a Euclidean
space of any dimension.

Broken down by where a pair sits, the median distance error runs from 0.089
for pairs in the core to 1.455 for pairs at mean radius above 0.7R — a factor
of sixteen across the sphere.

Points lying exactly on the wall are excluded throughout. The Dirichlet
condition makes G vanish there identically, so such points are infinitely
distant under any inversion.

## Entanglement entropy

Tracing the ground state over a ball gives an entropy proportional to the
boundary area, not the enclosed volume. Computed on a radial lattice of 90
sites with the angular sum taken to l = 800:

| R/a | S | S/R² | S/R³ |
|---:|---:|---:|---:|
| 4.5 | 5.92 | 0.2924 | 0.0650 |
| 8.5 | 21.27 | 0.2944 | 0.0346 |
| 14.5 | 61.95 | 0.2946 | 0.0203 |
| 20.5 | 123.72 | 0.2944 | 0.0144 |
| 32.5 | 309.91 | 0.2934 | 0.0090 |

The log-log slope is 2.0012, against 2 for an area law and 3 for a volume law.
The fitted coefficient is 0.2935, against Srednicki's published 0.30; the
deficit is the truncation of the angular sum, whose tail falls slowly enough
that raising the ceiling from 400 to 800 still adds 0.2%. Any quoted entropy
states its ceiling.

This uses a different discretisation from the rest of the repository. A three
dimensional lattice cannot resolve an area law, since the region radius must
span a decade while dense diagonalisation caps out near sixteen sites per
radius. Decomposing in spherical harmonics first turns each (l, m) sector into
an independent radial chain of a few hundred sites, trading angular resolution
for radial resolution.

## Thermal states

The area law belongs to the vacuum, not to entropy in general. Adding the
thermal factor coth(ω/2T) to each normal mode drives the scaling from an area
law to a volume law:

| T | log-log slope | S/R² | S/R³ | S at R = 20.5 |
|---:|---:|---:|---:|---:|
| 0.00 | 1.996 | 0.2904 | 0.0142 | 122.05 |
| 0.05 | 2.002 | 0.2949 | 0.0144 | 123.93 |
| 0.10 | 2.050 | 0.3268 | 0.0159 | 137.34 |
| 0.20 | 2.330 | 0.5940 | 0.0290 | 249.64 |
| 0.50 | 2.906 | 5.5439 | 0.2704 | 2329.84 |
| 1.00 | 2.993 | 34.9923 | 1.7069 | 14705.52 |

Temperature is in inverse lattice spacings. The intermediate exponents are not
a third scaling law but the sum of an area term and a volume term, with the
crossover radius falling as T rises.

At T > 0 the global state is mixed, so a region's entropy no longer measures
entanglement alone; it mixes entanglement with ordinary thermal entropy. The
validation is that tracing over nothing reproduces the Bose occupation entropy
S = Σ[(n+1)ln(n+1) − n ln n] to eight significant figures, which the
covariance construction has no way of knowing about in advance.

This is the caveat that matters wherever Srednicki's result is invoked in a
black hole context: area scaling is a vacuum property, and any state with
appreciable thermal occupation does not show it.

## Mutual information

Every entropy above is cutoff dependent: it scales as the boundary area in
lattice units and diverges as the spacing goes to zero. The mutual information
between disjoint regions is not, because each divergent boundary contributes
to S_AB as well as to S_A or S_B and cancels. Holding the geometry fixed and
refining the lattice:

| scale | sites | I | S_A |
|---:|---:|---:|---:|
| 2 | 40 | 0.1200 | 5.81 |
| 4 | 80 | 0.1093 | 20.92 |
| 6 | 120 | 0.1059 | 45.33 |
| 8 | 160 | 0.1042 | 79.06 |

I settles while S_A grows as the area. It is the only quantity in the
repository that survives the continuum limit.

Whether it repairs the reconstruction is not settled here. Judged by the
Euclidean defect of the inferred distances, with an inversion law not fitted
to the answer:

| inversion law | defect |
|---|---:|
| mutual information, analytic I ~ d⁻⁴ | 0.703 |
| mutual information, arbitrary I ~ d⁻⁶ | 0.505 |
| two-point function, exact free-space law | 0.668 |

Fitting the exponent to the true distances gives 0.379, which appears to beat
the two-point function, but that comparison is circular: the two-point
inversion uses the exact free-space law with no fitted parameter.

More seriously, the pairwise construction is withdrawn. It rests on a cell
smearing that cannot work. Continuum correlators are delta-normalised densities, not
canonical variables, and give single-site symplectic eigenvalues of order
10²². Midpoint cell averaging brings them to order 80, but they then scale as
(Λa)³. Performing the cell integral properly, by Gauss-Legendre quadrature on
an expanded node set, shows the midpoint value was a quadrature artefact:

| Λa | order 1 | order 2 | order 3 | order 4 | order 6 |
|---:|---:|---:|---:|---:|---:|
| 6 | 7.690 | 1.098 | 0.507 | 0.393 | 0.362 |
| 12 | 61.73 | 8.034 | 3.169 | 1.751 | 0.853 |

Refining does not converge towards the midpoint value; it falls monotonically
away from it, through the uncertainty bound of one half, below which no
physical Gaussian state lies.

The reason is structural rather than numerical. With m_α the cell average of
mode α and f_α the regulator, Cauchy-Schwarz gives
X_ii P_ii ≥ a⁶(Σ f_α m_α²/2)². Completeness gives Σ m_α² = a⁻³ at f = 1,
making the bound exactly one quarter; any regulator lowers it. A regulated
continuum field restricted to cells is not a canonical system, and no
quadrature will make it one.

The lesson is to discretise the theory first and compute correlators from the
discrete system, which is what the radial chains do — they satisfy X P = I/4
to 2 × 10⁻¹⁵ by construction. The UV finiteness result above is unaffected,
since it uses those chains and involves no smearing.

## Layout

```
src/qvacuum/
    constants.py      units, physical constants, SI conversion
    geometry.py       sphere, cubic lattice clipped to it
    cutoffs.py        cutoff conventions and named physical scales
    modes.py          Dirichlet eigenmodes, Weyl law, lattice count
    vacuum.py         per-mode amplitudes, <phi^2>
    correlators.py    equal-time two-point functions and regulators
    cavity.py         Dirichlet mode functions, boundary-aware correlator
    entropy.py        radial-chain entropy, mutual information, thermal states
    reconstruct.py    distance inversion, classical MDS, Procrustes

scripts/
    01_mode_count.py           exact count against Weyl law
    02_cutoff_scan.py          N(Lambda) across physical scales
    03_vacuum_fluctuations.py  <phi^2> and its divergence
    04_correlators.py          G(r) and the mass correlation length
    05_reconstruction.py       MDS pipeline check
    06_boundary_effects.py     wall suppression and position dependence
    07_reconstruction_failure.py  where recovery from correlations breaks down
    08_entanglement_entropy.py    the area law and its coefficient
    09_mutual_information.py      UV finiteness; reconstruction still fails
    10_cell_integration.py        why cell averaging cannot fix the smearing
    11_thermal_states.py          area law to volume law with temperature

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
the sharp-cutoff correlator against (1 − cos Λr)/4π²r²; the cavity mode
functions against unit normalisation on the ball and against vanishing at the
wall; the cavity correlator against the regulated free-space form in the
interior; the covariance matrices against X P = I/4 for a pure global state;
the entropy against Srednicki's area-law coefficient; MDS against the
coordinates used to build the matrix. A module without such a check is not
merged.

## Not implemented

Standard Model field content. Interacting λφ⁴ theory, which would require
Euclidean lattice Monte Carlo and a compiled backend. Entanglement entropy from
the Gaussian covariance matrix, which is the intended second phase. A
proper point-split renormalisation of <phi^2>, which the present term-by-term
subtraction only approximates. Mutual information as a reconstruction probe,
which is UV finite where the two-point function is not. Entanglement entropy of a
non-concentric or non-spherical region. The relation between the area-law coefficient and the cavity
correlator, which the two discretisations do not currently share.

## Licence

MIT.
