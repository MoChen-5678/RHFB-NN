# Finite-window BdG no-sea branch filters (2026-10-07)

## Result

The proposed scalar filter is an exact algebraic encoding of the DIRHB mixed quasiparticle sign choice **in a finite Hermitian BdG space only if** a divider `Λ` separates the full particle qp band from the sea qp band in magnitude, and the spectrum avoids `0` and `±Λ`. It retains particle/sea cross-sector components because it is applied to the full BdG Hamiltonian including `Δ`; it is not a normal-Dirac double projection. It is a promising eigen-free residual/reference diagnostic under that certified finite-window condition, not a production prescription for an expanding continuum.

Use the project's Nambu convention

`Hμ = [[h−μ, Δ], [−Δ*, −h*+μ]]`,

`R = [[ρ, κ], [−κ*, I−ρ*]]`,

and let `C` denote the particle-hole conjugation in the chosen Bogoliubov basis, so `C Hμ C⁻¹ = −Hμ`. In DIRHB's Eq. (41) convention its partner map is `(U,V) -> (V*,U*)`; a basis change or a different placement of pairing-field phases can add a fixed unitary/phase to this map. The implemented convention must verify the anticommutation identity before using the filter. Define

`SΛ(Hμ) = sgn(Hμ) sgn(Λ² I − Hμ²)`,

`RΛ = (I − SΛ(Hμ))/2`.

For an eigenvalue `E`, `SΛ` is `sgn(E)` for `|E|<Λ`, and `−sgn(E)` for `|E|>Λ`. Thus `RΛ` selects the negative-H eigenvectors inside the window and positive-H eigenvectors outside it. This sign is important: in the usual generalized-density convention, a positive-energy particle qp annihilator contributes its particle-hole partner (negative-H vector) to `R`; DIRHB's negative-energy sea qp annihilator contributes its partner (positive-H vector). In the zero-pairing limit, the low-energy particle sector has `ρ=0`, and the high-magnitude negative-Dirac sea sector also has `ρ=0`, as required by its no-sea reference. This is the mixed sign pattern of DIRHB Eq. (46), translated from annihilator eigenvectors to the occupied Nambu frame defining `R`.

Nikšić et al. state that each `(U,V)` eigenvector at `E>0` has a particle-hole partner `(V*,U*)` at `−E` (Eq. 41 discussion, PDF p. 11), then select `Eₚ>0` above the Dirac sea and `Eₐ<0` in it (Eq. 46, PDF p. 12). The density and pairing sums include both selected sectors (Eqs. 47–48, pp. 12–13). The paper says the large gap (about 1200 MeV in their shifted convention) makes those labels distinguishable; it does not propose this scalar filter. So the formula below is our finite-matrix derivation from their branch rule, not a quoted RHFB equation. [Nikšić et al., arXiv:1403.4039](https://arxiv.org/pdf/1403.4039).

Because the functional calculus acts on the complete `Hμ`, nonzero `Δ` can mix normal positive- and negative-energy sectors, and `RΛ` can retain the corresponding cross-sector anomalous density. The projector does not set `κ=P₊κP₊ᵀ`. Exact projector and purity claims require an exact sign function and an exact spectral gap at all three boundaries `0, ±Λ`; a rational sign approximation is only approximate, and its error must be bounded against the gap and checked against an independent small-space diagonalization/reference. Also check particle-hole symmetry, `R²=R`, Hermiticity, `κᵀ=−κ`, and the number convention. A sign error in the mapping from annihilator columns to `R` swaps the physical vacuum.

## Why one scalar threshold fails as the continuum grows

At fixed `Λ`, every positive qp state with `Eₚ>Λ` is assigned the outside-window sign and therefore treated like a sea-band partner. The positive qp continuum is unbounded as the momentum/basis cutoff is raised, so any fixed finite `Λ` eventually cuts through the particle continuum. Raising `Λ` does not cure the issue if the sea qp magnitudes overlap that particle continuum in the enlarged spectrum: no scalar magnitude divider can then classify both complete bands. DIRHB's own discussion distinguishes the original Dirac continua (after its `h_D−m` shift, the particle continuum starts at zero and the negative continuum at `−2m`) and uses the large gap to label RHB states; this is not a theorem that all positive qp continuum energies remain below all sea qp magnitudes at arbitrary cutoff (PDF p. 12). The exact failure test is direct: if no `Λ` satisfies `max |Eₚ| < Λ < min |Eₐ|` for the chosen finite bands, this formula cannot represent Eq. (46) for those bands. In an unbounded/overlapping limit, `sgn(Λ²−H²)` has no band-separating gap and the prescription ceases to define the desired branch.

## What Riesz/Kato continuation can and cannot rescue

If a reference at zero pairing has a physically labeled annihilator subspace and that **whole selected cluster** stays isolated from its complement along a specified pairing homotopy, a Riesz projector can transport the subspace without extracting eigenvectors:

`P(t) = (1/(2πi)) ∮Γ (zI−H(t))⁻¹ dz`.

The contour must remain in the resolvent set and enclose the selected cluster only. Differentiating the resolvent gives

`dP = (1/(2πi)) ∮Γ (zI−H)⁻¹ (dH) (zI−H)⁻¹ dz`.

Kato parallel transport of this projector is likewise a subspace construction. The contour-resolvent formula and smoothness under a maintained gap are standard spectral perturbation results; see the gap and contour discussion in [Fermanian-Kammerer et al., *Propagation of Wave Packets for Systems Presenting Codimension One Crossings*, Sec. 3](https://link.springer.com/article/10.1007/s00220-021-04147-2). For finite matrices, evaluating contour quadrature uses shifted linear solves only. Automatic differentiation can differentiate those solves while the contour remains fixed and nonsingular; changing or adaptively selecting the contour, closing the gap, or changing cluster rank is not smooth in general.

This gives a conditional answer to overlapping *energy ranges*: ranges may overlap away from the path while a particular cluster remains separated in the complex spectral plane at every homotopy point; then continuation is possible and path-labeled. But if a selected and unselected level cross, touch, or enter the same continuum so the separating gap closes, no Riesz contour isolates that chosen subset at the crossing. At an exact degeneracy, only the full degenerate eigenspace has a canonical spectral projector; a preferred vector or partial occupation inside it needs extra symmetry/constraints or a diabatic rule and is not determined by the contour. Symmetry labels can protect sectors only when the full `H(t)` preserves that symmetry. In a continuum, a resonance is not generally an isolated normalizable eigenspace, so the finite-matrix orthogonal-projector argument does not automatically apply.

Even when the projector exists, it transports the **initially declared subspace**, not the physical meaning of that initial label. The zero-pairing reference must therefore state which normal states are nucleons and which are sea, with the correct bare-vacuum convention. Homotopy path dependence, cluster rank, gap lower bound, particle-hole pairing, and blocked-state identity all need certification. A Riesz projector cannot discover the branch from the interacting Hamiltonian alone once energy/symmetry information no longer separates it.

For odd parity, start from that branch-defined even vacuum and explicitly exchange one quasiparticle column with its particle-hole partner (equivalently, toggle that mode's occupation in the generalized-density projector). Algebraically this is a rank-one swap `R_odd = R_even + |w⟩⟨w| − |Cw⟩⟨Cw|` when `w` is the previously unoccupied partner and the pair is normalized. Keep the selected mode's identity by overlap/parallel transport; recomputing a scalar `RΛ` alone always returns the even branch vacuum and cannot encode the odd swap. Degenerate or crossing blocked modes require an extra symmetry label or explicit subspace/occupation tracking. This agrees with the standard HFB blocking construction, but does not validate a relativistic blocked branch by itself. [Robledo, Bernard & Bertsch, arXiv:1210.2768, Sec. II](https://arxiv.org/pdf/1210.2768).

## Recommended use

Use the scalar filter only as a bounded low-momentum algebra/reference test after certifying the interval inequality for the declared particle and sea qp bands and checking the exact full-BdG density against Eq. (46) branch sums. For a larger but still gapped finite space, test Riesz/Kato transport from the explicitly labeled zero-pairing reference by contour solves, with no eigenvectors and no replacement of the NN's full primitive basis. Record the smallest contour-to-spectrum distance and fail closed when it shrinks to zero or the projector rank changes. Compare the transported projector and cross-sector `κ` against an independent spectral reference outside training. If the gap closes or continuum bands overlap, report that the branch cannot be certified by this method; do not smooth the threshold or claim continuity through a true crossing. Neither candidate changes the physical target or establishes a converged PKA1+D1S odd solution.
