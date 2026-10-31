# RHFB no-sea prescription: variational review (2026-10-06)

## Finding

The primary RHFB formulations reviewed here do **not** state the moving-positive-projector energy problem

\[
\min_{\mathcal R_a,B}\ E[B\rho_aB^\dagger,\ B\chi_aB^T],\qquad
B^\dagger B=I,\quad BB^\dagger=P_+(h[\rho])
\]

as their variational definition. They derive a generalized-density HFB equation and prescribe which quasiparticle branches define the vacuum/densities. They do not add a KKT multiplier for differentiating a self-consistent positive-energy projector. That is evidence against calling full moving-\(P\) KKT the *standard* no-sea RHFB equation; it does not prove that every active-space embedding is inequivalent to the cited formulations.

The actionable standard route is therefore: specify the no-sea reference and quasiparticle occupation branches; form \(\rho,\kappa\) from those branches; calculate the self-consistent normal field \(h\) by varying the declared functional and the pairing field with the project’s pairing convention; and solve the resulting RHFB equation with number and blocking conditions. For the project’s full antisymmetric matrix \(\chi\), with \(E_{\rm pair}=\tfrac14\bar V_{ab,cd}\chi^*_{ab}\chi_{cd}\), the convention is

\[
\delta E_{\rm pair}=\operatorname{ReTr}(\Delta^\dagger\delta\chi),\qquad
\Delta_{ab}=\tfrac12\bar V_{ab,cd}\chi_{cd}
=2\,\frac{\partial E_{\rm pair}}{\partial\chi^*_{ab}}
\quad\text{(unrestricted full-matrix Wirtinger derivative).}
\]

The factor of two is essential; an independent \(a<b\) coordinate convention must account for the antisymmetric-coordinate multiplicity instead. A neural network may minimize the residual of these equations. Differentiating through a smooth projector used to *evaluate that residual* is an optimizer operation; it does not add a projector-response term to the physical Euler equation. A projected active-space equation is equivalent only after its density reconstruction and allowed variations have been shown to reproduce the chosen branch prescription.

## What the cited RHFB papers prescribe

**Geng–Long, D-RHFB (PKA1-capable framework, Gogny D1S pairing).** In Eq. (10), PDF p. 4, nucleon creation/annihilation operators are defined from positive-energy HF solutions and negative-energy Dirac solutions are omitted in that field quantization. Their Eq. (12) then makes the Bogoliubov transformation between HF-particle and quasiparticle operators. Their Eqs. (21)–(22), PDF p. 5, define

\[
\mathcal R=\begin{pmatrix}\rho&\kappa\\-\kappa^*&1-\rho^*\end{pmatrix},\qquad
\mathcal H=\frac{\partial E}{\partial\mathcal R}
=\begin{pmatrix}h&\Delta\\-\Delta^*&-h^*\end{pmatrix}.
\]

The paper says the full RHFB equation is obtained by varying the energy with respect to \(\mathcal R\), with \(h\) and \(\Delta\) its normal and pairing fields. This supports the generalized-density variational equation and a nucleon-only reference vacuum. Its derivative notation follows its own generalized-density convention; it should not be copied as the project’s unrestricted full-matrix Wirtinger derivative, which has the explicit factor of two above. The paper does not spell out a moving spectral constraint \(Q_+(h)\) or its KKT response. Separately, its Eq. (23) (PDF p. 6) expands the quasiparticle spinors in a DWS basis; the p. 9 cutoff discussion explicitly says not to confuse the positive/negative basis functions needed for completeness with sea contributions to densities/currents. Thus “negative DWS basis coefficient” is not itself a physical sea occupation. [Geng & Long, arXiv:2202.00482, Eqs. (10)–(23)](https://arxiv.org/abs/2202.00482).

**DIRHB relativistic HFB branch prescription.** Nikšić et al. derive the RHB eigenproblem and fields from the variational principle in Eqs. (37)–(42), PDF pp. 10–11. Their no-sea choice is more explicit than a particle-space double projection: for quasiparticle states associated with positive-energy nucleons choose \(E_p>0\), while for states associated with the Dirac sea choose \(E_a<0\) (Eq. (46), PDF p. 12). Their Eqs. (47)–(48) then include both the \(p\) and \(a\) branch sums in the normal density and pairing tensor. They note that the sea-branch \(V_{na}\) coefficients are small and those terms are often omitted; after doing so, \(\kappa\) is only approximately antisymmetric. This is a branch-defined quasiparticle vacuum, not the same definition as \(\rho=Q_+\rho_aQ_+\), \(\kappa=Q_+\kappa_aQ_+^T\). [Nikšić et al., arXiv:1403.4039, Eqs. (41)–(48)](https://arxiv.org/abs/1403.4039).

These sources give distinct model conventions, not one universal projector identity. In particular, the Geng–Long positive-energy HF particle-operator construction cannot be merged with the DIRHB full Dirac-space quasiparticle branch sums without deriving the reference-vacuum mapping. Nor do the negative-energy DWS basis functions, negative-energy eigenvalues of \(h\), negative-energy BdG partners, and occupied sea-labelled quasiparticle branches denote the same set of states.

**Full relativistic pairing sums.** Serra–Ring discuss the related prescription in which the no-sea omission is made in the particle-hole source, while anomalous-density terms are retained to preserve completeness/antisymmetry. That is another concrete choice of quasiparticle sums. It is not the same as projecting both \(\rho\) and \(\kappa\) with the same \(Q_+\), and it should not be silently substituted for Geng–Long’s nucleon-only field-operator construction. [Serra & Ring, Phys. Rev. C 65, 064324 (2002)](https://doi.org/10.1103/PhysRevC.65.064324).

## Stationarity equations suitable for implementation

For a declared RHFB functional \(E[\rho,\kappa,\kappa^*]\), define the number-constrained Nambu Hamiltonian in the code’s convention, for example

\[
\mathcal H_\mu=
\begin{pmatrix}h-\mu I&\Delta\\-\Delta^*&-h^*+\mu I\end{pmatrix},\quad
h=\frac{\delta E}{\delta\rho},\quad
\Delta_{ab}=\tfrac12\bar V_{ab,cd}\kappa_{cd}
=2\,\frac{\partial E_{\rm pair}}{\partial\kappa^*_{ab}}
\quad\text{for the unrestricted full antisymmetric matrix.}
\]

With the project’s full-matrix pairing convention above, the Nambu field uses that \(\Delta\). The standard variational stationarity condition, written here in the usual projector-density form equivalent to the RHFB eigenproblem under canonical Bogoliubov variations, is

\[
[\mathcal H_\mu,\mathcal R]=0,\quad \mathcal R^2=\mathcal R=\mathcal R^\dagger,
\quad \kappa^T=-\kappa,\quad \operatorname{Tr}\rho=N_0,
\]

with the RHFB no-sea prescription additionally fixing the quasiparticle branch occupation used to construct \(\mathcal R\). The commutator alone is insufficient: it admits multiple projectors/blocked roots, so branch labels and odd-quasiparticle occupation must be tracked. For odd nuclei, impose the selected blocked quasiparticle occupation (and the corresponding self-consistent time-odd fields when required) as part of the state definition, then compare admissible blocked solutions. Neither the cited branch prescription nor the Dirac min-max papers prove which blocked deformed nucleus is the global minimum.

For an explicitly **fixed** nucleon frame \(B\), \(B^\dagger B=I\), an active canonical density can be embedded as \(\rho=B\rho_aB^\dagger\), \(\kappa=B\kappa_aB^T\). Its restricted Euler equation is

\[
[\mathcal H_a,\mathcal R_a]=0,\quad \operatorname{Tr}\rho_a=N_0,
\qquad \mathcal H_a=W^\dagger\mathcal H_\mu W,\quad W=\operatorname{diag}(B,B^*),
\]

plus the explicitly chosen complement/vacuum branches. This is a well-defined fixed-space calculation. If \(B\) is redefined as \(P_+(h[\rho])\) at every iterate, this equation alone is not automatically the full-space branch equation; the equivalence needs to be established for the chosen embedding and source sums.

The **different moving-space energy model** is obtained only if one declares \(Q=P_+(h[\rho])\) to be an energy-dependent constraint and extremizes the composed functional \(E(Q(h[\rho]),\mathcal R_a)\). Its derivative contains \(D Q_h[D h[\delta\rho]]\); a KKT/implicit-response term is then required. That is a legitimate candidate model, but the reviewed standard RHFB equations do not establish that it is their variational definition. Do not add this term to the RHFB physical stationarity equations solely because an optimizer differentiates through \(Q\).

## No-eigensolve network residual and branch certification

A practical network can output a canonical generalized density (or Bogoliubov frame) and minimize a scaled residual assembled from

\[
r_{\rm stat}=\operatorname{vec}(\mathcal H_\mu[\rho,\kappa]\mathcal R-\mathcal R\mathcal H_\mu[\rho,\kappa]),\quad
r_N=\operatorname{Tr}\rho-N_0,\quad
r_{\rm field}=\mathcal H_\mu-\mathcal H[\rho,\kappa],
\]

along with purity, Hermiticity, antisymmetry, and blocked-branch constraints. The field residual is redundant if fields are constructed exactly from the functional, but is useful when the network separately represents fields. Train/evaluate the absolute residuals and number error; do not replace branch choice by a small commutator or by a hand-set \(\|\kappa\|\) threshold. A smooth rational sign/projector approximation or continuation can supply differentiable branch operators without diagonalizing during training, provided a genuine spectral gap and the desired branch split are certified. This keeps automatic differentiation of the *residual map* distinct from changing its target Euler equation.

For completeness, in a finite spectral window with a gap separating sea and particle quasiparticle sectors, our Nambu convention permits the unblocked branch projector to be represented by

\[
\mathcal R=\tfrac12\left[I+\operatorname{sgn}(\mathcal H_\mu)
\operatorname{sgn}(\mathcal H_\mu^2-E_c^2I)\right].
\]

This is an algebraic spectral-function derivation, **not** an equation quoted from the cited papers. It requires the spectrum to avoid 0 and \(\pm E_c\), and the selected particle continuum must be restricted to a declared finite window \(|E_p|<E_c\); it cannot be applied as one universal threshold across an unbounded particle continuum. Odd blocking changes the selected occupation and needs an explicit blocked projector/branch update.

## What the Dirac min-max results do and do not establish

Talman’s min-max principle concerns avoiding variational collapse in a one-particle Dirac eigenvalue calculation by minimizing over the large component and maximizing over the small component. It is not a theorem about the Nambu HFB density, D1S pairing, or odd blocking. [Talman, Phys. Rev. Lett. 57, 1091 (1986)](https://doi.org/10.1103/PhysRevLett.57.1091).

The rigorous electron–positron/Dirac–Fock analysis by Barbaroux, Esteban, and Séré is a further warning against transferring a min-max slogan to odd RHFB. For closed shells and weak interaction, they prove a regime where self-consistent Dirac–Fock and max-min agree. For open-shell particle counts, their Theorem 11 gives a counterexample to the general Mittleman max-min conjecture. This is an electronic Dirac–Fock result, not a counterexample theorem for nuclear odd RHFB; its relevance is that self-consistent positive-projector selection and a global max-min principle are not interchangeable by general logic. [Barbaroux, Esteban & Séré, arXiv:math-ph/0402058, Theorems 9 and 11](https://arxiv.org/abs/math-ph/0402058).

## Decision for this project

Use the literature-defined quasiparticle-branch RHFB equations as the physical target for the next NN solver, with the chosen no-sea source convention written explicitly. Permit AD through smooth branch/projector approximations when calculating residuals, but do not add a moving-projector energy KKT multiplier to physical stationarity unless the project explicitly chooses and labels that alternative constrained energy model. Keep `P_+ρP_+`/`P_+κP_+^T` as an unvalidated candidate until it reproduces the selected reference-vacuum branch densities, purity, fields, particle number, and blocked branch. The standard branch equations resolve the variational-target question; they do not by themselves validate a production-size PKA1/D1S odd deformed implementation or identify its lowest physical blocked root.
