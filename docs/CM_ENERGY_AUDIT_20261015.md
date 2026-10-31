# Center-of-mass energy audit for the blocked O17 RHFB result

## Finding

The active PKA1+D1S total-energy path contains no center-of-mass (CoM) term. If the reported `+281.772 MeV` is the runner's `total_energy` (the scalar formed from particle-hole plus D1S pairing energy), it is the uncorrected EDF value. It is not a value with the microscopic CoM correction applied, and it should not be presented as a CoM-corrected experimental binding energy.

This omission does not by itself make the current variational target inconsistent: Geng and Long explicitly state that their microscopic correction is evaluated after the RHFB variation and is not included in that variation. A separately labeled post-variation correction is the closest match to that published prescription. The 2008 Long et al. paper establishes RHFB with PKA1 and Gogny pairing, but its searchable text contains no CoM correction formula; it should not be cited as evidence for either including or omitting one.

## Source check

Geng and Long, *Relativistic Hartree-Fock-Bogoliubov model for axially deformed nuclei*, arXiv:2202.00482, state in Sec. II.C, Eq. (41), that

\[
 E_{\rm cm}=-\frac{1}{2 A M}
 \langle\mathrm{HFB}|\hat{\mathbf P}_{\rm cm}^{2}|\mathrm{HFB}\rangle,
 \qquad \hat{\mathbf P}_{\rm cm}=\sum_i\mathbf p_i,
\]

and say that this correction is not included in variation. Their Appendix C, Eq. (C1), gives the expectation in canonical Bogoliubov amplitudes with both the normal exchange and anomalous contractions; Eq. (C2) gives the one-body \(p^2\) matrix element using radial second derivatives and the upper/lower orbital angular momenta. These are direct source statements, not an inference. See [the paper's Eq. (41), PDF p. 8](https://arxiv.org/pdf/2202.00482#page=8) and [Appendix C, Eqs. (C1)–(C10), PDF pp. 27–28](https://arxiv.org/pdf/2202.00482#page=27). The paper is published as [Phys. Rev. C 105, 034329 (2022)](https://doi.org/10.1103/PhysRevC.105.034329).

Long et al., [arXiv:0812.1103](https://arxiv.org/pdf/0812.1103), derives and applies the spherical density-dependent RHFB model with PKA1 and Gogny D1S. Its accessible full text does not contain `center-of-mass`, `P_cm`, or `E_cm`; it supplies no explicit CoM prescription to transplant into the current code. That distinction matters because the 2022 paper is affirmative evidence for a post-variation correction, whereas absence of a term in the 2008 article is not proof that its tabulated energies share a particular CoM convention.

In the repository, [`selfconsistent_stationarity.py`](../dpl_rhf/rhfb/selfconsistent_stationarity.py#L119) obtains the particle-hole energy and D1S pairing energy and defines the total as their sum at lines 186–190. [`streamed_normal_functional.py`](../dpl_rhf/rhfb/streamed_normal_functional.py#L446) enumerates kinetic, meson, Coulomb, contact, and rearrangement channels; its total at lines 538–549 sums those channels. [`functional.py`](../dpl_rhf/rhfb/functional.py#L534) likewise sums the listed particle-hole and pairing terms. None constructs \(\mathbf P_{\rm cm}\), evaluates its expectation, or adds \(E_{\rm cm}\). The RHF trainer also records `center_of_mass_correction: False` ([`trainer.py`](../dpl_rhf/rhf/trainer.py#L190)). A repository-wide source search found no legacy Python CoM evaluator to generalize; the existing CoM documentation records the convention and formula, not an implementation.

## Odd-HFB Wick expression in the repository's density convention

The following is a direct Wick derivation for any pure quasiparticle vacuum, including a blocked odd HFB state. It does not assume time reversal. Use

\[
 \rho_{ab}=\langle c_b^\dagger c_a\rangle,\qquad
 \chi_{ab}=\langle c_b c_a\rangle=-\chi_{ba},\qquad
 (p_\mu)_{ab}=\langle a|p_\mu|b\rangle,
\]

where \(p_\mu\) is Hermitian and \(\mu\in\{x,y,z\}\). This \((\rho,\chi)\) convention is the one used by the local generalized density \(R=\bigl(\begin{smallmatrix}\rho&\chi\\-\chi^*&1-\rho^*\end{smallmatrix}\bigr)\) in [`manifold.py`](../dpl_rhf/rhfb/manifold.py#L1-L8).

For one Cartesian component,

\[
 \hat P_\mu=\sum_{ab}(p_\mu)_{ab}c_a^\dagger c_b,
\]

and the canonical anticommutator gives

\[
 \hat P_\mu^2
 =\sum_{ad}(p_\mu^2)_{ad}c_a^\dagger c_d
 -\sum_{abcd}(p_\mu)_{ab}(p_\mu)_{cd}
   c_a^\dagger c_c^\dagger c_b c_d.
\]

Wick's theorem in this ordering is

\[
 \langle c_a^\dagger c_c^\dagger c_b c_d\rangle
 =\chi^*_{ac}\chi_{db}
  -\rho_{ba}\rho_{dc}+\rho_{da}\rho_{bc}.
\]

Consequently, the complete expression is

\[
\boxed{\displaystyle
 \langle\hat{\mathbf P}_{\rm cm}^{2}\rangle
 =\sum_{\mu=x,y,z}\Bigl[
   \operatorname{Tr}(\rho p_\mu^2)
   +\left|\operatorname{Tr}(\rho p_\mu)\right|^2
   -\operatorname{Tr}(\rho p_\mu\rho p_\mu)
   +\operatorname{Tr}(\chi^\dagger p_\mu\chi p_\mu^T)
 \Bigr]. }
\]

There is no factor of one half in this expectation: the cross terms in \(\hat P_\mu^2\) are already counted by the ordered index sums. The anomalous term has a **plus** before the trace for the stated \(\chi=\langle cc\rangle\) convention; its value can have either sign. The transpose on the final momentum matrix is essential. The disconnected term \(|\operatorname{Tr}(\rho p_\mu)|^2\) is explicit. It vanishes for a state whose total momentum expectation has been established to be zero, but local time-odd currents alone do not establish that. This expression reduces to the Slater-determinant result when \(\chi=0\).

With the currently selected like-particle pairing and no proton-neutron coherence, evaluate the same spatial operator separately in neutron and proton spaces:

\[
 \langle\mathbf P_{\rm cm}^{2}\rangle
 =\sum_\mu\left[
 \sum_{s=n,p}\operatorname{Tr}(\rho_s p_\mu^2)
 +\left|\sum_{s=n,p}\operatorname{Tr}(\rho_s p_\mu)\right|^2
 -\sum_{s=n,p}\operatorname{Tr}(\rho_s p_\mu\rho_s p_\mu)
 +\sum_{s=n,p}\operatorname{Tr}(\chi_s^\dagger p_\mu\chi_s p_\mu^T)
 \right].
\]

The mean-momentum square is a square of the **total** neutron-plus-proton mean. The exchange and anomalous contractions are species diagonal under the stated model. If proton-neutron anomalous density is introduced later, the compact formula over the combined isospin space must be used instead.

For momentum matrices in \(\mathrm{fm}^{-1}\), Eq. (41)'s energy factor is \((\hbar c)^2/(2AM)\), with \(M\) in MeV and the resulting \(E_{\rm cm}\) in MeV. In units with momentum already in MeV, it is \(1/(2AM)\). Do not combine fm and MeV units without this conversion.

## What transfers from Appendix C, and what does not

Appendix C's Eq. (C1) is written in spherical canonical-orbit notation and uses time-reversed partners and orbit degeneracies. It is a compact specialization for the paper's spherical, time-reversal-invariant applications. It supports the one-body, normal, and anomalous structure above, but its orbit-degeneracy bookkeeping should not be copied unchanged into a general blocked, time-odd \(m\)-scheme state. The full matrix Wick formula above is the appropriate starting point for that extension. The paper's Eq. (41) explicitly authorizes the operator expectation as a correction after variation; extending the contractions to a blocked odd quasiparticle vacuum is a mathematically direct generalization, not a claim that Eq. (C1)'s spherical implementation already covers it.

## Momentum operator and the present ADF basis

The current O17 single-particle space has \(D=432\), 9 interior radial points at \(r=0.8,\ldots,7.2\,\mathrm{fm}\), and complete \(|\kappa|\leq3\) spinor multiplets. The radial mesh also has endpoint values at 0.001 and 8 fm; see [`basis.py`](../dpl_rhf/rhfb/basis.py#L100-L160). Its derivative order 5 selects a five-point forward stencil and pairs it with \(D_{\rm back}=-D_{\rm forward}^{T}\) in the radial Dirac operator ([`dirac.py`](../dpl_rhf/rhf/dirac.py#L16-L35), [lines 61–93](../dpl_rhf/rhf/dirac.py#L61-L93)). This pairing is part of the free Dirac \(\boldsymbol\alpha\cdot\mathbf p+\beta M\) discretization. It is **not** already a matrix representation of the three components of the translational momentum, and `apply_kinetic` is not a CoM operator.

A consistent extension needs matrix elements of each spinor-space \(p_\mu=-i\hbar\partial_\mu\) and of \(\mathbf p^2=-\hbar^2\nabla^2\), using the same radial metric, boundary treatment, and angular basis as the stored \(Q,\rho,\chi\). In spherical spinor notation the upper and lower components carry different orbital angular momenta \(l_u,l_d\); the scalar one-body \(p^2\) uses the corresponding \(-d^2/dr^2+l(l+1)/r^2\) radial operator, as reflected in Geng–Long Eq. (C2). A vector \(p_\mu\) is rank one and connects opposite-parity angular components with the associated \(j,m\) selection rules. At \(|\kappa|=3\), such a vector action can reach angular channels beyond the retained single-particle cutoff. Those channels must be handled consistently when forming \(\langle i|p^2|j\rangle\); do not silently replace the direct \(p^2\) matrix element by \(\sum_{k\in B}p_{ik}p_{kj}\), which inserts a truncated intermediate-state projector.

For the current ADF scheme, a future implementation should specify a discrete Hermitian \(p_\mu\) (including its adjoint/boundary rule), and a compatible direct \(p^2\) matrix or a demonstrated convergent \(p_\mu^\dagger p_\mu\) construction. It should document the outer-boundary condition and check convergence against radial spacing/order, box radius, and angular cutoff. Do not form a nominally Hermitian momentum from the forward stencil alone. The present grid's \(n_\theta=7,n_\phi=14\) resolves the basis harmonics through the configured \(K=3\) interaction bandwidth; that fact alone does not validate derivatives of the vector-transformed spinors or the missing \(K+1\) intermediate channels.

## Smallest next validation

1. Keep the current self-consistent objective and the saved \(+281.772\) result in the **uncorrected** convention; the cited 2022 recipe is post-variation.
2. Implement a read-only Python CoM evaluator on a small spherical closed-shell case first. Compare its normal-density limit with the Slater formula and with Geng–Long Appendix C in the same spherical DWS convention.
3. Check the full Wick expression on an exactly constructed paired two-level HFB vacuum (including its anomalous term and the total-momentum disconnected term), then on a blocked odd state. These are algebraic checks, not fits to an empirical \(-3/4\hbar\omega\) estimate.
4. Establish the ADF momentum matrix elements and convergence/boundary behavior before using the correction for O17. Only then report \(E_{\rm raw}\), \(E_{\rm cm}\), and \(E_{\rm raw}+E_{\rm cm}\) as separate quantities; leave the training loss unchanged unless the target/reference parameterization is explicitly changed to the same corrected convention.

No numerical CoM value was evaluated for this audit. No live runtime source was changed.
