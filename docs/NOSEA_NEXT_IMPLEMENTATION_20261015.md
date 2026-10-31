# A physics-led, eigen-free no-sea reference for RHB

## Recommendation

Do not add a guessed projection to the live PKA1+D1S stationarity run. Keep that result labelled “no-sea branch unverified.” For a later, separately certified reference, implement the **finite-space RHB quasiparticle-vacuum prescription of DIRHB as a matrix-free occupied-subspace map**, and first validate its odd-blocked extension. This is a concrete route to avoid a dense eigendecomposition while retaining the full Nambu space, all (G/F) blocks, and the full pairing action. It is a proposed numerical realization of that paper’s prescription; the paper itself uses basis-space RHB equations, not this projector algorithm.

## Physical prescription to reproduce

Nikšić et al. write the full RHB operator as

\[
\mathcal H_{\rm RHB}=\begin{pmatrix}
h_D-m-\lambda & \Delta\\
-\Delta^* & -h_D^*+m+\lambda
\end{pmatrix}
\tag{14}
\]

and give the no-sea vacuum in terms of *labelled quasiparticle branches*: occupy particle/Fermi-sector annihilators with (E_p>0), and Dirac-sector annihilators with (E_a<0) (their Eq. (46)). Their normal density and pairing tensor contain both corresponding (V^*V^T) and (V^*U^T) sums (Eqs. (47)–(48), arXiv PDF pp. 11–13). They say the sea-labelled (V_a) amplitudes are small because of the large gap and are often dropped; they explicitly note this leaves κ antisymmetric only approximately. Thus the exact branch prescription retains the full anomalous sector. It is not the strict normal-space double projection (P_+\rho P_+,P_+\kappa P_+^T), nor does it say to delete (G/F) couplings from Δ. [Nikšić et al., *DIRHB*, Eqs. (14), (46)–(48)](https://arxiv.org/pdf/1403.4039).

This is distinct from the RHFB field-operator convention in Long et al. and Geng–Long. Those works define no-sea at the reference field/occupied-orbital level while retaining positive- and negative-energy Dirac basis functions for (U,V) completeness. The two formulations need a term-by-term map before treating them as the same model. [Long et al., arXiv:0812.1103, Eqs. (7), (19)–(21), (31)–(34)](https://arxiv.org/pdf/0812.1103); [Geng & Long, arXiv:2202.00482, Eqs. (10)–(23)](https://arxiv.org/pdf/2202.00482).

## Implementable projector iteration

1. **Declare the target and finite basis.** For this first reference, use the DIRHB mixed-sign vacuum above, with the complete declared particle and Dirac basis sectors and the same full PKA1 (h_D[\rho]) and D1S (\Delta[\kappa]) in every iteration. Do not silently substitute the different Long/Geng operator convention. Fix an odd blocking prescription separately; DIRHB’s stated implementation is even-even, so its paper does not validate O17 blocking.
2. **Label the branches once at a reference field.** At Δ=0, separate the particle and Dirac spectra of (h_D-m) with a contour point (W) inside the *demonstrated gap of that field*. (W) is only the contour placement: moving it within the open gap must leave the projector unchanged, and no universal (W=-M) is assumed. Then turn on the full Δ continuously. In the finite matrix, track the *whole spectral projectors* along this homotopy, not individual eigenvectors. This gauge-independent continuation is valid only while the selected and unselected clusters remain separated by a gap. If a gap closes or rank changes, report the physical branch as unresolved; do not repair it with an overlap cutoff or an energy-density threshold.
3. **Apply the mixed-sign projector without diagonalizing.** If a complete finite-space enclosure establishes every Fermi-sector eigenvalue has `|E_p| < E_c` and every Dirac-sector eigenvalue has `|E_a| > E_c`, with no eigenvalues at `0` or `±E_c`, evaluate
   `R_even = 1/2 [ I - sgn(H) + sgn(H + E_c I) + sgn(H - E_c I) ]`,
   where `H` is the full RHB matrix. Here `sgn` is the Hermitian matrix sign, evaluable by rational approximations and matrix-free shifted solves. On an eigenvalue `E`, the formula gives occupancy 1 for `E<0` inside the Fermi window and for `E>0` in the Dirac window; it gives occupancy 0 on the complementary branches. This is the complement of the projector onto DIRHB-selected quasiparticle annihilators (`E_p>0`, `E_a<0`), matching the usual generalized-density convention.
   This requires a **complete-band** condition: a mere gap at `±E_c` is insufficient if any high-energy Fermi-sector state lies outside the window. If the finite space lacks this separation, the scalar formula does not reproduce Eq. (46); use explicitly branch-labelled Riesz projectors or report the prescription unresolved. No full `G/F` element is deleted, no normal `P_+ kappa P_+^T` replacement is made, and no basis is refreshed.
4. **Close the self-consistency map.** Recompute `h_D[rho]`, rearrangement terms, and full `Delta[kappa]` from the resulting densities; iterate `R -> R_occ(H[R])` with mixing until the density, number, field, and projector residuals converge. For O17, validate the odd quasiparticle occupation swap and time-odd fields against an independently labelled small-space reference before using this map at production size. An odd-parity flag alone does not identify the desired blocked orbit.
5. **Certify numerics and physics separately.** Check projector idempotency, particle-hole covariance, rank, complete basis accounting, gap margins, (\rho,\kappa), energy, and every full-H commutator block against an independently diagonalized small fixed-field reference. Then increase the declared basis and contour accuracy and report changes. The independent diagonalization is a validation oracle for small spaces, not part of the production iteration. Do not claim that stationarity alone proves the selected projector is the global minimum.

This proposal is an inference from the published finite-space branch rule and matrix-sign spectral calculus. It has not been shown equivalent to the PKA1 RHFB field-operator convention, and the odd blocked extension is unproved. The project’s saved O17 fixed-field comparison already shows why strict normal (P_+\rho P_+,P_+\kappa P_+^T) cannot be substituted as though it were the same full-field state.

## What the inverse-Hamiltonian result does and does not justify

Tanimura et al. use an inverse-Hamiltonian evolution to select positive-energy eigenstates of a **fixed Dirac Hamiltonian**. Their variational bound (Eq. (15)) and evolution (Eqs. (16)–(17)) rely on (W) lying between the top Dirac-sea and bottom Fermi-sea levels; their self-consistent single-particle iteration is Eqs. (22)–(23). This establishes a practical fixed-field/SCF eigenstate solver for that Dirac problem. It does not derive the mixed particle/sea RHB occupation projector with finite-range pairing, much less its odd blocked generalization. (W) is a spectral shift for that solver, not a universal physical axiom (W=-M). [Tanimura et al., *3D mesh calculations for CDFT*, Sec. 3.2, Eqs. (15)–(17), (22)–(23)](https://academic.oup.com/ptep/article/2015/7/073D01/1561088).

The EFT account of the no-sea approximation is also about which vacuum contributions are represented in the static functional: vacuum dynamics is absorbed into fitted local couplings, while valence orbitals enter explicit ground-state sums; negative-energy intermediate states still enter consistent response calculations. It does not define a universal projector on every component of a paired generalized density. [Furnstahl, Piekarewicz & Serot, arXiv:nucl-th/0205048](https://arxiv.org/abs/nucl-th/0205048).
