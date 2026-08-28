# Computational limits

Measured on a container with NumPy 2.4 and SciPy 1.17. Absolute timings will
differ between machines; the scaling will not.

## Mode enumeration

Cost grows as the number of (l, n) pairs, roughly (R Lambda)^2 / (2 pi), with a
root-find per pair. R Lambda = 120 gives 1770 pairs and 118 698 modes counting
degeneracy, in a few seconds. R Lambda of order 10^3 is the practical ceiling.
Above that use the Weyl expansion, which is exact enough that the residual is
below the level of anything else in the calculation.

## Lattice size

Cubic lattice of spacing a = R/n clipped to the sphere. Exact site counts:

| n | a | sites | dense matrix | eigvalsh |
|--:|--:|--:|--:|--:|
| 8 | 62.5 nm | 2 109 | 36 MB | ~1 s |
| 12 | 41.7 nm | 7 153 | 0.4 GB | ~30 s |
| 16 | 31.3 nm | 17 077 | 2.3 GB | minutes |
| 20 | 25.0 nm | 33 401 | 8.9 GB | not tractable |

Dense symmetric diagonalisation costs O(N^3): 4000 x 4000 takes about 4.5 s,
8000 x 8000 about 38 s.

The practical ceiling for full three-dimensional work is n around 12 to 16,
that is a spacing of 30 to 60 nm and an effective R Lambda of 25 to 50. This
sits inside the window where the Weyl boundary term is 5 to 10 per cent of the
bulk, which is convenient, but it means only about 25 points across the
diameter and no access to short-distance structure.

## Symmetry reduction

Decomposing in spherical harmonics separates the problem into independent
radial chains labelled by l, each of a few hundred sites and diagonalised in
milliseconds. This gives excellent radial resolution and no transverse
information. It is the right tool for boundary effects and for entanglement
entropy; it is the wrong tool for the MDS reconstruction, which needs the full
three-dimensional point set.

## What would need a compiled backend

Interacting lambda phi^4 theory, requiring Euclidean lattice Monte Carlo on a
16^4 or 32^4 lattice. Pure NumPy is too slow; Numba or JAX would be required.
Out of scope for the present phase.
