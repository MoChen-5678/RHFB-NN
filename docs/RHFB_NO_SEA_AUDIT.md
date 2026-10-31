# RHFB no-sea and Nambu-branch audit

## What “no-sea” means in the cited RHFB formulation

The no-sea approximation is a prescription for the Dirac-field vacuum and the
particle-hole source densities. By itself it does not specify a unique
projector acting on a numerical primitive basis, nor does a basis vector's
energy sign alone determine whether it is an occupied sea contribution.

Geng and Long derive their deformed RHFB energy from the expectation of the
full Hamiltonian in a Bogoliubov vacuum. The Dirac field's particle operators
are defined in the positive-energy single-particle sector for the no-sea
mean-field model. Separately, both positive- and negative-energy Dirac
Woods–Saxon basis states are retained in expansions of the quasiparticle
`U` and `V` components to maintain completeness. Their paper explicitly says
not to confuse these negative-energy expansion states with a Dirac-sea
contribution to densities/currents. For odd systems, the blocked orbit's
`V` contribution is replaced with its `U` contribution in kinetic and
potential energy terms. See [Geng & Long, arXiv:2202.00482, §§II.B–C and
space-cutoff discussion](https://arxiv.org/abs/2202.00482).

Serra and Ring make the pairing caveat explicit: in their convention the
no-sea approximation is applied to the particle-hole density; dropping the
corresponding antiparticle terms from the anomalous density as well can break
its antisymmetry because the quasiparticle solutions are no longer complete.
They studied the effect of retaining or omitting those antiparticle
contributions in the pairing tensor. See [Serra & Ring, Phys. Rev. C 65,
064324 (2002)](https://doi.org/10.1103/PhysRevC.65.064324).

The distinction can be stated in the quasiparticle sums used in Serra and
Ring (Eqs. (9)–(10), PDF p. 3):

\[
  \rho=V^*V^T,\qquad \kappa=V^*U^T.
\]

Their full relativistic expressions split each tensor into particle-like and
antiparticle-like components. In the stated no-sea particle-hole convention,
the antiparticle-like `V V†` terms are omitted from `ρ`; the corresponding
`V Uᵀ` terms are retained in the complete `κ` sum. Serra and Ring warn that
dropping those terms too can destroy skew symmetry when the quasiparticle
basis is incomplete. This is one explicit source prescription, not a universal
identity between quasiparticle sums and a projector in every computational
basis.

The distinction is visible under a basis change. If a quasiparticle component
is expanded as `V_k = Σ_a C_ak φ_a` in a complete DWS basis containing both
signs of Dirac energy, then its outer product contains all `C_ak C*_bk`
blocks, including negative-DWS-basis coefficients and positive/negative
cross-blocks. Omitting a selected antiparticle-like quasiparticle contribution
from the source sum and projecting each retained `V_k` with `P_DWS` are
different operations in general. Whether either projector representation is
equivalent to a cited no-sea prescription depends on the reference vacuum,
the HF particle space, the definition of the quasiparticle sums, and how the
full Nambu density is embedded. The cited papers use distinct constructions:
Serra and Ring examine a fully relativistic pairing tensor with
antiparticle-like terms, while Geng and Long define the field operators in a
positive-energy HF particle space and separately retain ±DWS basis functions
in the computational spinor expansion. They should not be collapsed into
one projector rule.

Geng and Long make a related but distinct split in their D-RHFB construction:
their field operators use positive-energy HF states in the no-sea mean-field
model (Eq. (10), PDF p. 4), while computational `U,V` spinors are expanded in
both positive- and negative-energy spherical DWS states for completeness
(Eq. (23), PDF p. 9). They explicitly distinguish negative-energy *basis
functions* from a Dirac-sea contribution to densities and currents. Thus a
negative Dirac eigenstate used as an expansion basis function, an occupied
negative-energy particle state, an antiparticle contribution in the
quasiparticle density sum, and a negative-energy BdG particle-hole partner
are four different labels. The latter is the `±E` Nambu symmetry; it is not
the Dirac-sea prescription.

This exposes the remaining limitation of the current tiny branch toy. Its
nonlinear normal sources use a fixed `h0` label mask (`P ρ P`), its bare
one-body energy still contracts the full `h0 ρ`, and its D1S functional uses
the full antisymmetric `κ`. It resembles source-only omission more closely
than simultaneous source-and-pairing projection, but the toy has not derived
the mapping from its fixed basis mask and pure generalized density to either
paper's normal-ordered RHFB quasiparticle sums. In particular,
`Tr(P_sea ρ)` reports support in basis vectors labelled negative by `h0`; by
itself it is not a field-theoretic Dirac-sea occupation. The toy therefore
validates neither a full PKA1 RHFB source functional nor the physical
interpretation of that diagnostic.

Three constructions need to be distinguished before assigning physical
meaning to a projector test:

1. **Construct in a nucleon-only HF particle space, then embed.** One can
   first build a canonical HFB generalized density in the positive-energy HF
   particle space and then embed it in a larger Nambu space with the chosen
   empty-complement/Dirac-hole branches. This can preserve purity when the
   embedding is defined as a direct sum of compatible projectors. It is a
   candidate realization of a nucleon-only no-sea prescription, but its
   equivalence to either cited paper's source sums still needs an explicit
   reference-vacuum derivation.
2. **Project an already formed full generalized density.** Applying
   `ρ → PρP` and `κ → PκPᵀ` after constructing full `R` does not generally
   preserve the HFB purity relations
   `ρ−ρ²=κκ†` and `ρκ=κρ*`; it therefore need not leave `R²=R`. Purity and
   particle-number/parity constraints must be checked rather than assumed.
3. **Move the projector with the self-consistent field.** If `P=P+(h[ρ])`,
   the projected variational problem has derivatives through `P[ρ]` and its
   stationarity conditions include the constraint/embedding response. It is
   not justified to identify its stationary points with those of the
   unconstrained full-space RHFB functional without deriving those terms.

The earlier moving-support experiment used the third construction. The
available literature and current tests have neither proved it equivalent to
the desired no-sea RHFB problem nor proved that every such projector
construction is wrong. The simultaneous projection of `ρ` and `κ` is a
distinct constrained model until that equivalence test is completed; a root
from it cannot by itself settle the physical question.

## Candidate nucleon-space construction and stationarity

A finite-dimensional implementation of the Geng–Long Eq. (10) particle-space
idea can begin with a fixed orthonormal frame `B` for the chosen positive-energy
HF nucleon space, `B†B=I_m`. Build an ordinary canonical HFB density there,

\[
R_a=\begin{pmatrix}\rho_a&\chi_a\\-\chi_a^*&I_m-\rho_a^*\end{pmatrix},
\qquad R_a^2=R_a,
\]

and embed it in the full primitive/Nambu space with

\[
W=\operatorname{diag}(B,B^*),\quad
R=W R_a W^\dagger+\begin{pmatrix}0&0\\0&I-B^*B^T\end{pmatrix}.
\]

Equivalently, `ρ=Bρ_aB†` and `χ=Bχ_aBᵀ`. This direct-sum embedding leaves
inactive particles empty and their hole partners occupied, and preserves
`R²=R` if `R_a` is pure. It provides a clean, testable *candidate* for a
nucleon-only active-space HFB state; it does not by itself identify which
full-RHFB quasiparticle sums or odd-blocking prescription a nuclear
functional must use.

For fixed `B`, let `H_BdG(μ)` be the full Nambu Hamiltonian including the
number multiplier. The active Hamiltonian is `H_a=W† H_BdG W`. The constrained
HFB Euler equation is

\[
[H_a,R_a]=0,\qquad \operatorname{Tr}\rho_a=N,
\]

with fermion parity and a tracked blocked branch enforced independently. A
full-space commutator may remain nonzero because those residual components
rotate active orbitals into the excluded particle complement; report it
separately from the active residual. The helper
[`no_sea_particle_space.py`](../dpl_rhf/rhfb/no_sea_particle_space.py)
implements this fixed-space embedding and active commutator without training
spectral decomposition.

A moving-projector candidate adds the coupled spectral condition

\[
Q^2=Q=Q^\dagger,\quad [h[\rho],Q]=0,\quad
Q=\mathbf 1_{(s,\infty)}(h[\rho]),\quad \rho=Q\rho Q,\quad
\chi=Q\chi Q^T,
\]

where a nonzero gap around cutoff `s` and a fixed rank are required. The
positive/negative spectral inequalities select the desired branch; commutation
alone is insufficient. If the projector is differentiated as part of an
energy objective, its response is the Fréchet derivative

\[
(DQ_h[\delta h])_{ij}
=\frac{q_i-q_j}{\epsilon_i-\epsilon_j}\,
  (U^\dagger\delta h U)_{ij},\qquad
q_i=\mathbf 1_{(s,\infty)}(\epsilon_i),
\]

in the eigenbasis of `h`. Only cross-cutoff blocks contribute. Here
`δh=D h[δρ,δχ]`; the induced `δB` must then be propagated through both
`ρ=Bρ_aB†` and `χ=Bχ_aBᵀ`. Thus the first variation
`δE=DE[ρ,χ](δρ,δχ)` contains a projector-response term whenever `B` moves.
For the present D1S code convention, `Δ` is returned with
`E_pair=½ Re Tr(χ†Δ)`; the derivative coefficient must follow that convention
rather than be guessed from another HFB normalization. A practical no-eigensolve candidate may evaluate `Q`
through a differentiable matrix-sign/rational action and a fixed-rank QR chart;
its derivative and fixed-point convergence must be checked against the
finite-dimensional divided-difference reference. If instead one alternates a
frozen-`Q` inner HFB solve with `Q←1_{(s,∞)}(h[ρ])`, that is a nested SCF root
procedure, not automatically the same variational problem as differentiating
through `Q[ρ]`. The precise outer stationarity/min-max condition has to be
specified before calling either procedure the physical solution.

The relativistic variational literature gives reasons to treat that outer
condition as a separate open gate. Talman's [min-max principle](https://doi.org/10.1103/PhysRevLett.57.1091)
uses a minimum over large components and maximum over small components for a
single Dirac orbital; it does not establish a paired-density theorem.
Barbaroux, Farkas, Helffer, and Siedentop prove that, for their electron–positron
Hartree–Fock model and a *fixed* admissible electron space `Ran(Q)`, minimizers
are projections that satisfy projected no-pair Dirac–Fock equations
([Theorem 4.4](https://arxiv.org/abs/math-ph/0404009), PDF p. 13). The same
paper says it cannot show existence of a maximizer over the allowed
one-particle spaces (PDF p. 2). Barbaroux, Esteban, and Séré formulate that
outer self-consistent-projector max-min problem for atomic Dirac–Fock and
prove it in a closed-shell perturbative regime, but their
[Theorem 11](https://arxiv.org/abs/math-ph/0402058) gives an open-shell regime
in which the max-min level is not attained by a Dirac–Fock solution with its
self-consistent projector (PDF p. 8). These are atomic electron–positron/DF
results, not nuclear RHFB results. They show why the fixed-projector inner
Euler equation, the outer projector optimization, and a projector
self-consistency condition must be stated separately; they do not prove or
disprove the corresponding odd paired nuclear construction. The odd paired
nuclear branch therefore needs an explicit finite-space reference comparison,
not an imported theorem.

For the small-space gate, compare (i) an independent dense reference that
forms the positive projector with `eigh`, constructs the active HFB solution,
and embeds it by the formula above; (ii) a candidate matrix-sign/QR path with
no training eigensolve; and (iii) the projector-response reference derivative
against autograd/finite differences. Report `R²−R`, `Trρ−N`, parity, active
and full commutator RMS, projector Hermiticity/idempotency/commutator, and
branch overlap. Only the dense reference may use eigendecomposition.

## Generalized density and branch selection

In this code's convention,

\[
R=\begin{pmatrix}\rho&\kappa\\-\kappa^*&I-\rho^*\end{pmatrix},
\qquad R^2=R,
\]

is a pure Bogoliubov vacuum (or a parity-odd blocked vacuum). For a Hermitian
BdG Hamiltonian \(\mathcal H\), stationarity is

\[
[\mathcal H,R]=0.
\]

This equation alone does not identify the physical root. If \(\Delta=0\),
\(\mathcal H=\operatorname{diag}(h-\mu,-h^*+\mu)\). In the ordinary
particle sector, occupied orbitals select the particle component when
\(\epsilon-\mu<0\), and empty orbitals select the hole component when
\(\epsilon-\mu>0\). For a negative-Dirac-energy orbital in the no-sea
vacuum, the nucleon orbital is also empty, so its hole component is selected;
that state lies on the *positive* BdG-energy branch. Consequently:

* the physical no-sea generalized density is not generally the projector
  onto all negative BdG eigenvalues;
* minimizing the BdG spectrum indiscriminately can occupy the Dirac sea;
* the HFB solution is a branch-selected stationary saddle, not the global
  minimum over every particle/antiparticle occupation sector.

The new [`branch.py`](../dpl_rhf/rhfb/branch.py) encodes this exactly in the
unpaired limit. `no_sea_branch_reference` produces a pure even or odd
generalized density whose negative-Dirac particle occupations are zero and
whose corresponding Nambu hole branches are occupied. Its training loss uses
full-space stationarity, number conservation, and a continuation overlap; it
does not diagonalize `h` or `H_BdG`. The dense
`select_ph_branches_reference_only` helper is an independent toy/reference
oracle only. It pairs exact particle-hole-related `±E` vectors and selects
one branch per pair by overlap with the no-sea reference, refusing ambiguous
zero crossings and invalid particle-hole complements.

That overlap construction is a branch-continuation diagnostic, not a
published odd-RHFB min-max theorem. The HFB operator has a spectrum unbounded
above and below, so ordinary imaginary-time descent can collapse into the
negative quasiparticle continuum; the inverse-Hamiltonian paper addresses
stable extraction of positive quasiparticle modes (Tanimura, Hagino & Ring,
arXiv:1304.4697, pp. 1–2), not the definition of an odd no-sea occupation
sector. The odd D-RHFB prescription in Geng and Long instead treats blocking
explicitly, replacing the blocked orbit's `V` component by `U` in kinetic and
potential-energy terms (PDF p. 7). Consequently, a small full-space
commutator alone is insufficient evidence for the correct blocked saddle:
parity, occupation branch continuation, and blocking-specific energy/source
contractions must also agree with the reference formulation.

For odd nuclei, parity is encoded by an exact particle/hole toggle in the
reference generalized density. This operation switches one Nambu branch and
changes `Tr(rho)` by one in the unpaired reference. The full canonical
generator preserves this parity while allowing self-consistent pairing
correlations.

## Validation boundary

The existing toy experiment proves only that the neural/canonical optimizer
can solve the *moving constrained no-pair toy functional* without training
eigensolves and that its field-dependent projector derivative is included.
It is not a PKA1 RHFB calculation. The new branch module provides exact
zero-pairing branch regressions and a reference-only spectral selector; it
does not yet establish an odd PKA1 solution or a complete field-theoretic
mapping between a pure `R` and the normal-ordered no-sea `rho` plus complete
`kappa` sources.

The next scientific gate must use a small self-consistent RHFB model with a
full Dirac basis and the same branch prescription in the neural/canonical
solver and the independent reference. It must report separately:

1. particle-hole stationarity and particle-number error;
2. exact parity and branch-overlap/continuation status;
3. negative-Dirac particle occupation in the no-pair limit;
4. the amount by which a `P rho P`, `P kappa P^T` truncation changes the D1S
   pairing energy and gap compared with full-basis `kappa`;
5. basis/cutoff convergence, since [Geng & Long](https://arxiv.org/abs/2202.00482)
   report that PKA1 results can be sensitive to how many negative-energy DWS
   expansion states are retained.

No result in this document should be read as validating standard no-sea
RHFB, PKA1 odd nuclei, or production readiness.

As a code-path regression (not a nuclear prediction), the D1S test uses four
unit-normalized sampled spinors and a deliberately cross-branch-only
antisymmetric `κ`. In that constructed case the full two-Gaussian D1S
pairing energy is `0.07191052393 MeV` and the corresponding gap Frobenius
norm is `0.9708574481 MeV`; projecting `κ` onto only the first two states
sets both to zero. The 100% removal is expected because this diagnostic
chooses no same-branch pairing components. It demonstrates that the
simultaneous-support rule can discard a finite D1S contribution; it does not
estimate how large the effect is for a converged physical RHFB solution.
