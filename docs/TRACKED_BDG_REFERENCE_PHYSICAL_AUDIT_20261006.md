# Audit of the tracked projected-BdG reference branch

## Finding

The retry did **not** produce an accepted odd RHFB reference. It produced a pure, fixed-field BdG projector on a branch whose number decreased with increasing chemical potential, then lost its tracked frame on the next self-consistent field update. A direct parity invariant computed from the saved generalized density shows a more fundamental defect: the block-2 canonical seed is odd, while every saved chemical-potential continuation projector is even. The `fermion_parity=-1` value carried by the returned Python object was metadata; the selector did not enforce it. The separate parity-audit JSON re-evaluates the saved densities with `number_parity_from_density`; it diagnoses the old retry and is not a repair or new RHFB result.

The decreasing number is not evidence of a chemical-potential sign error by itself. Along the saved μ continuation, the selected member of every PH pair stayed unchanged, with a substantial branch margin. A decreasing response is consistent with following an excited stationary projector rather than a minimum. But because the saved projector has even parity, it cannot be interpreted as the intended one-quasiparticle odd branch. Neither “excited odd state” nor “odd ground state” is established by this run.

## What the primary papers prescribe

| Source and equation | Paper-supported statement | Relation to this reference | What is not supported |
|---|---|---|---|
| Li et al., [arXiv:1203.1363](https://arxiv.org/pdf/1203.1363), Eqs. (4)–(5), pp. 1–2 | The odd state is formed by creating one quasiparticle on the even vacuum; blocking exchanges the creation and annihilation operators for that selected quasiparticle. The other quasiparticle occupations retain the vacuum prescription. The paper uses equal filling to restore time reversal in the density. | Supports one designated blocked mode and a global odd-state construction. The initial block-2 canonical seed follows that kind of odd-parity construction. | It does not prescribe independently choosing the occupied side of every PH pair by overlap with the previous generalized-density subspace. It also does not validate this toy’s pure, time-odd full-blocking implementation. |
| Pan et al., [arXiv:2205.01329](https://arxiv.org/pdf/2205.01329), Sec. III A, pp. 11–14 | Their odd DRHBc study compares orbital-fixed blocking with automatic blocking of the lowest quasiparticle at each iteration. It reports difficulty near degenerate low quasiparticle levels and retains orbital-fixed blocking as the fallback. | Supports tracking/identifying a designated blocked orbital or deliberately selecting the lowest quasiparticle. It warns that branch near-degeneracies need explicit handling. | It does not support freezing the physical PH polarization of every unblocked pair by overlap. It uses equal filling and a different functional/pairing setup from this pure odd D1S toy. |
| Nikšić et al., [arXiv:1403.4039](https://arxiv.org/pdf/1403.4039), Eqs. (41)–(48), pp. 11–13 | Eq. (41) is the RHB eigenproblem and Eq. (42) defines Δ from κ. Eqs. (43)–(45) give the nonrelativistic quasiparticle-vacuum branch. Relativistic no-sea Eq. (46) selects positive-energy particle quasiparticles and negative-energy Dirac-sea quasiparticles; Eqs. (47)–(48) sum both sectors into ρ and κ. The authors note that omitting the sea terms makes κ only approximately antisymmetric. | Establishes that relativistic no-sea branch selection is a labeled quasiparticle prescription, with distinct particle and Dirac-sea branches. | The retry solves BdG only in a moving positive-nucleon frame. It omits the negative-Dirac quasiparticle sector and does not demonstrate equivalence to Eqs. (46)–(48), odd blocking, or the full RHFB prescription. The cited DIRHB paper describes an even-even implementation; its equations are useful for the branch convention, not an odd-blocking validation. |
| Kasuya & Yoshida, [arXiv:2005.03276](https://arxiv.org/pdf/2005.03276), Eqs. (16)–(23), pp. 6–8 | The real canonical Bogoliubov transform has determinant equal to number parity; Eq. (22) relates it to the Pfaffian of the HFB Hamiltonian and signed qp energies. The abstract explicitly restricts “lowest-energy is even” to a time-reversal-invariant HFB Hamiltonian with no zero modes. | Supports calculating parity as an invariant, independent of the expectation of N, and warns that an all-negative-energy vacuum is not automatically even once time reversal is broken. | This is nonrelativistic HFB. It does not establish projected relativistic no-sea parity conventions or odd PKA1 energy ordering. |

The executable selector now follows a parity-aware fixed-field construction. It first forms the all-negative BdG projector (the vacuum annihilated by the positive-energy qp operators in this matrix convention) and computes its Pfaffian parity. If that projector is odd, it toggles the lowest positive-energy qp to form an even reference at the same frozen fields; it then toggles only the designated blocked qp. It computes the even-reference parity and final odd-state parity from their actual `(rho, chi)` matrices and refuses a state unless the values are `+1` and `−1`, respectively. This compensation is the lowest even-sector construction for the **fixed quadratic BdG matrix**; it is not a theorem that a self-consistent EDF even reference or global odd minimum is obtained by one such toggle. The designated blocked qp overlap selects only that qp through continuation. The complete occupied frame may be Procrustes-aligned only after the physical projector is fixed, as a gauge operation.

An important distinction is between **basis gauge transport** and **occupation-branch transport**. A unitary rotation, phase alignment, or Procrustes alignment among vectors already assigned to the same annihilator subspace changes only its frame representation. By contrast, deciding for each PH pair whether the selected projector contains its positive- or negative-energy eigenvector changes the physical generalized density. Li et al.’s Eqs. (4)–(5) change that occupation only for the designated blocked quasiparticle; they do not authorize overlap-based physical occupation choices for all unblocked pairs. The retry used the latter rule.

## Saved retry diagnostics

The machine-readable source is [tracked_bdg_reference_retry_20261006.json](tracked_bdg_reference_retry_20261006.json). It contains a full source snapshot, 28 saved μ-continuation states/fields, and one outer-iteration summary. The old first-attempt artifact remains separate.

- At fixed fields, the 28 branch-resolved BdG states had minimum gap `0.11173`, minimum PH-choice margin `0.92417`, minimum complete-frame label overlap `0.71363`, and maximum idempotency residual `4.12e-15`. The six-pair positive/negative selection pattern read from the saved overlap scores was constant (`+-+-++`, with `+` denoting selection of that pair’s positive-energy member). These facts rule out an observed PH partner exchange along the μ path; they do not establish that the selected vacuum is the correct one.
- The sampled number was monotone decreasing: `N(-1.55)=3.853404`, `N(2.45)=2.083415`. The 12-step bisection stopped at `N=2.999790963`, absolute error `2.09e-4`. This is a sampled branch response, not a converged number root.
- The one accepted outer iterate remained pure (`||R²−R||F=2.72e-15`) but its normal/pairing field residual was `3.07735`. The next field update triggered a label-overlap value `0.042`, below the `0.20` continuation threshold, so the solver stopped. This is loss of trackability under that field update; it does not tell whether the cause was a physical crossing, a large field step, or chart transport error.
- The `fermion_parity` check used the active generalized density and
  \[
  A=(I-2R)\tau_x,\qquad \Pi(R)=\frac{\operatorname{Pf}A}{\operatorname{Pf}A_{\rm empty}}.
  \]
  In the package's Nambu convention, for a pure generalized density `A` is skew and this ratio is the parity relative to the empty reference. The block-2 canonical seed gives `Π=−1`; all 28 saved μ-continuation projectors give `Π=+1` (skew residual at most `1.7e-14`). Thus the per-pair selector changed the global parity before the number continuation, despite carrying odd-parity metadata. This direct algebraic check is stronger than inferring parity from the number expectation or from the number of positive BdG eigenvectors.
- The retry's saved continuation projectors are therefore **not odd-state reference densities** and cannot be used for odd-state energy, stationarity, or stability comparison. Do not reinterpret them as a blocked excitation based on `blocked_index=2` metadata.
- The runtime spectral counter recorded `33 eigh`, `0 eigvalsh`, and `0 eig` calls. Its one-call instrumentation probe passed. The run status remains nonconverged and unaccepted.

## Bounded stability and chemical-potential checks

For fixed fields, write the Hermitian BdG matrix as
\[
  K(\mu)=K_0-\mu\mathcal N,
\]
and let `R` be a rank-D generalized-density projector. The ambient rank-D Grassmannian is larger than the fermionic Gaussian manifold: an arbitrary occupied-to-empty rotation, especially a single Nambu eigenvector mixed with its own particle-hole partner, need not preserve the generalized-density particle-hole constraint, pairing antisymmetry, or parity sector. Stability must be evaluated only on canonical, particle-hole-correlated Bogoliubov generators. For the convention `Ω_quad(R)=½ Tr_Nambu(KR)+const`, and a canonical tangent with full Nambu amplitudes `z_ai`, the quadratic energy shift is
\[
  \Delta\Omega^{(2)}=\frac12\sum_{i\in R,\ a\notin R}
       (E_a-E_i)|z_{ai}|^2.
\]
The factor one-half is from Nambu double counting; the particle-hole-correlated pair contributes twice in the full-index sum. This coefficient is the energy shift at quadratic order; the second derivative along a real scalar path is twice that coefficient. For the reduced paired two-state angle `θ`, the stable/unstable stationary branches have `Ω(θ)-Ω(0)=±2E θ²+O(θ⁴)`, consistent with the full-index formula. This restricted paired model illustrates how the complementary Gaussian can have negative curvature and decreasing N(μ). It does not make an arbitrary Nambu Grassmann rotation admissible, nor does it prove that a one-quasiparticle odd state is unstable: such a state is compared and varied within its fixed parity/blocking sector, where allowed pair excitations are different. A self-consistent EDF Hessian also contains residual-interaction and rearrangement terms, and the retry did not close its fields.

A direct two-level paired Gaussian example makes the number-response test explicit. For one time-reversed pair with `ξ=ε−μ` and a fixed real pairing field `Δ`, set `E=sqrt(ξ²+Δ²)` and
\[
 K=\begin{pmatrix}\xi&\Delta\\\Delta&-\xi\end{pmatrix},\qquad
 R_\mp=\tfrac12(I\mp K/E).
\]
The lower-branch pure pair has physical pair number `N_-=1−ξ/E`; therefore `dN_-/dμ=Δ²/E³>0`. The complementary stationary pure Gaussian has `N_+=1+ξ/E`, `dN_+/dμ=−Δ²/E³`. In the reduced paired angle, their energy shifts have opposite signs, `±2E θ²`. The complementary branch is an excited maximum of this frozen-field paired model. It demonstrates that a negative number response can accompany a stationary, unstable even Gaussian; it does **not** identify the retry’s state as that eigenbranch, establish fixed-parity canonical stability of an odd state, or certify the retry as odd.

There is also a model-independent variational test. If, for every μ in a comparison, the solution is a global minimum of `E(R)−μN(R)` over the **same** admissible set `S`, then for `μ1<μ2` the two minimizing inequalities add to
\[
 (\mu_2-\mu_1)(N_2-N_1)\ge 0.
\]
Thus a decreasing `N(μ)` is incompatible with global minima over one fixed state set. It remains compatible with excited/saddle stationary states, a changing blocked-configuration set, or a parity/occupation branch change. Here both exclusions matter: the retry never minimized energy, and its saved projectors changed parity from the odd seed to even. The variational inequality therefore diagnoses lack of ground-state evidence; it does not diagnose a sign bug by itself.

## Evidence-based next change

Do not relax the overlap threshold or force the continuation through the `0.042` transition. Separate the two branch choices:

1. Build the unblocked quasiparticle vacuum from the declared no-sea annihilator branch. If matching the full relativistic prescription, include and label the Eq. (46) particle and Dirac-sea qp sectors; if using the projected nucleon-space approximation, say so explicitly.
2. Apply the odd blocking operation to exactly one designated qp mode by the creator/annihilator exchange of Li et al. Eqs. (4)–(5). Track that one blocked mode diabatically by overlap/quantum labels; align the remaining annihilator **basis** only by unitary gauge transport inside its selected subspace.
3. Enforce parity as a computed invariant (`Pf[(I−2R)τx]/Pf[(I−2R_empty)τx]`), not a metadata field. At every iteration require the declared odd parity and test the designated blocked-mode overlap separately from the full occupied-frame overlap.
4. Once a root is found, evaluate the constrained Hessian of the actual self-consistent energy on parity-preserving canonical tangents. Compare candidate blocked orbitals by converged energy before calling any branch lowest. The current retry supplies neither a converged root nor this stability/energy comparison.

These steps are a corrective research plan implied by the cited branch equations and the saved parity/overlap failures. They are not evidence that a corrected solver has already been implemented or validated.
