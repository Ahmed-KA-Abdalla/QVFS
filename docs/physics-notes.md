# Derivations

Natural units throughout, hbar = c = 1, lengths in metres.

## Mode counting

Free scalar field in a box of volume V with periodic conditions. Allowed
wavevectors are k = (2 pi / L)(n_x, n_y, n_z), one per cell of volume
(2 pi)^3 / V in k-space. The number with |k| <= Lambda is

    N(Lambda) = (V / (2 pi)^3) * (4/3) pi Lambda^3 = V Lambda^3 / (6 pi^2).

For a real field the modes k and -k are not independent; the count above is of
oscillator degrees of freedom, which is what is wanted, but the factor is easy
to lose. The tests guard it.

For a bounded region with Dirichlet conditions, Weyl's law gives

    N(Lambda) = V Lambda^3 / (6 pi^2) - S Lambda^2 / (16 pi) + O(Lambda).

The surface term is negative for Dirichlet and positive for Neumann. For a
sphere, S / V = 3 / R, so the ratio of the two leading terms is

    |boundary| / bulk = 9 pi / (8 R Lambda),

which is 10 per cent at R Lambda = 35.3 and 1 per cent at R Lambda = 353.4.

## Cavity eigenmodes

Separating the Helmholtz equation in spherical coordinates gives modes

    phi_{l m n}(r, theta, phi) = j_l(k_{l n} r) Y_{l m}(theta, phi)

with j_l(k R) = 0 at the wall. Each l carries degeneracy 2 l + 1. For l = 0,
j_0(x) = sin(x) / x, so the eigenvalues are exactly k R = n pi.

## Vacuum amplitude

Each mode is a harmonic oscillator with ground state variance

    <q_k^2> = 1 / (2 omega_k),     omega_k = sqrt(k^2 + m^2).

Summing over modes in the continuum,

    <phi^2> = (1 / 4 pi^2) int_0^Lambda dk k^2 / sqrt(k^2 + m^2),

which for m = 0 is Lambda^2 / (8 pi^2), quadratically divergent. For m > 0,

    <phi^2> = [ Lambda sqrt(Lambda^2 + m^2) - m^2 arcsinh(Lambda / m) ]
              / (8 pi^2).

## Equal-time correlator

    G(r) = int d^3k / (2 pi)^3 * exp(i k.r) / (2 omega_k)
         = (1 / 4 pi^2 r) int_0^inf dk k sin(k r) / sqrt(k^2 + m^2)
         = m K_1(m r) / (4 pi^2 r).

Since K_1(z) -> 1/z as z -> 0, the massless limit is 1 / (4 pi^2 r^2). For
m r >> 1, K_1(z) ~ sqrt(pi / 2z) exp(-z), so correlations are exponentially
suppressed beyond the Compton wavelength 1/m.

## Sharp cutoff and ringing

Truncating the k integral at Lambda for m = 0 gives, exactly,

    G_Lambda(r) = (1 - cos(Lambda r)) / (4 pi^2 r^2).

This does not converge pointwise as Lambda -> infinity; it oscillates between
0 and twice the continuum value. The behaviour is Gibbs ringing from the
discontinuous filter. Quantities that integrate over r are unaffected in the
mean, but the correlator itself is not usable at large Lambda r without a
smooth regulator. The Gaussian filter exp(-k^2 / 2 Lambda^2) is provided for
that purpose.

## Reconstruction

Given G_ij on a lattice, invert G(r) elementwise to obtain d_ij, then apply
classical multidimensional scaling: doubly centre the squared-distance matrix,

    B = -(1/2) J D^(2) J,     J = I - (1/n) 1 1^T,

and take the leading eigenvectors scaled by the square roots of their
eigenvalues. For an exactly Euclidean D of dimension three, B has rank three
and the spectrum shows a clean gap after the third eigenvalue. The gap is the
evidence for dimensionality and should be reported rather than assumed.

This procedure recovers what was put in. Its value is as a null model against
which failures under mass, truncation, coarse-graining and noise can be
measured.
