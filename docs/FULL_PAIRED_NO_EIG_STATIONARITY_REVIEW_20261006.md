# Full paired no-eigensolver stationarity review

This note records the primary equations behind an eigen-free full-space RHFB stationarity path and a small convention-level algebra check. It does not modify the production functional, the saved O17 state, the frozen-field solver, or any branch/reference result. A stationary point from this path would still require separate physical branch, number, parity, and self-consistency validation.

## Variational equations and an eigen-free residual

Geng and Long define the generalized density

\[
\mathcal R=\begin{pmatrix}\rho&\kappa\\-\kappa^*&1-\rho^*\end{pmatrix},
\qquad
\mathcal H=\frac{\partial E}{\partial\mathcal R}
=\begin{pmatrix}h&\Delta\\-\Delta^*&-h^*\end{pmatrix}
\]

in Eqs. (21)–(22) and derive the RHFB equation by varying the full generalized density. For Gogny D1S, their pairing prescription replaces the two-body pairing kernel by \((\gamma^0)_x(\gamma^0)_{x'}V^{pp}_{\rm Gogny}\) in Eq. (20d). Thus the full pairing field is part of the same variational problem; the reference to eigen-free stationarity does not authorize deleting its cross-sector matrix elements. [Geng & Long, arXiv:2202.00482, PDF pp. 5–6, Eqs. (20d), (21)–(22)](https://arxiv.org/pdf/2202.00482).

For fixed fields and a number multiplier \(\mu\), set

\[
\mathcal H_\mu=\begin{pmatrix}h-\mu I&\Delta\\-\Delta^*&-h^*+\mu I\end{pmatrix}.
\]

An infinitesimal canonical Bogoliubov variation is \(\mathcal R(t)=e^{tK}\mathcal R e^{-tK}\), with \(K^\dagger=-K\) and particle-hole-compatible block structure. Then \(\delta\mathcal R=[K,\mathcal R]\), and for the Nambu trace convention \(\mathcal L=\tfrac12\operatorname{Tr}(\mathcal H_\mu\mathcal R)\),

\[
\delta\mathcal L=\tfrac12\operatorname{Tr}(\mathcal H_\mu[K,\mathcal R])
=\tfrac12\operatorname{Tr}([\mathcal R,\mathcal H_\mu]K).
\]

Stationarity against all allowed canonical tangents is therefore equivalent to the occupied/unoccupied tangent block \(H^{20}=0\); in a full primitive representation this is measured by \([\mathcal H_\mu,\mathcal R]=0\). If \(\mathcal R=FF^\dagger\) is a pure rank-D projector with orthonormal occupied frame \(F\), then \(H^{20}=(I-FF^\dagger)\mathcal H_\mu F\) and the exact identity

\[
\|[\mathcal H_\mu,\mathcal R]\|_F^2=2\|(I-FF^\dagger)\mathcal H_\mu F\|_F^2
\]

holds without diagonalizing either matrix. This is a residual identity, not a proof that a particular branch or minimum has been selected.

The full-frame fixed-field equation can be evaluated from \(h,\Delta,\rho,\kappa\) by matrix products and automatic differentiation. The number equation \(\operatorname{Tr}\rho-N=0\) determines the chemical-potential multiplier jointly with the canonical coordinates. For a self-consistent EDF solve, \(h\) and \(\Delta\) must be recomputed from the same full primitive density at every residual evaluation, with all derivatives retained. A compressed \(P_+\)-only frame instead gives only the projected Euler equation; it need not make the full-space commutator vanish.

## Odd parity and branch identity

Robledo and Bertsch write the generalized density as \(\mathcal R=W\mathcal R_qW^\dagger\), give the odd blocked reference by swapping one quasiparticle occupation/column, and derive the first variation in terms of \([[\mathcal R,\mathcal H],Z]\) (printed pp. 7–10, Eqs. (9)–(21)). Their gradient chart preserves number parity. The same algebra is implemented by a pure odd Slater reference plus a connected canonical transform: the Pfaffian parity remains odd, while full \(\rho\) and \(\kappa\) can mix across the entire primitive basis. [Robledo & Bertsch, arXiv:1205.4443, printed pp. 7–10, Eqs. (9)–(21)](https://arxiv.org/pdf/1205.4443).

Parity alone does not select the physical no-sea odd branch. DIRHB's no-sea prescription chooses positive quasiparticle energies for particle-sector modes and negative quasiparticle energies for Dirac-sea modes in Eq. (46); Eqs. (47)–(48) contain separate particle and sea sums in both normal and anomalous densities. The paper says the sea amplitudes are often negligible because of the large gap and often drops those terms, which makes its pairing tensor only approximately antisymmetric (printed pp. 12–13). [Niksic et al., arXiv:1403.4039, printed pp. 12–13, Eqs. (46)–(48)](https://arxiv.org/pdf/1403.4039).

Accordingly, an eigen-free implementation still needs an explicit, physically justified initial no-sea occupation pattern in a declared fixed basis, a specified particle-addition or hole-blocking operation for the odd state, and a continuation/identity guard for the blocked quasiparticle. A Pfaffian parity check, particle number, stationarity, and adjacent blocked-vector overlap are all necessary diagnostics but do not certify the physical Dirac-sector labels or global lowest branch. A G/F-certified gap or rational-sign projector may identify a mathematical normal subspace, but its physical label remains a separate assumption until validated. No spectral eigenvectors should enter the neural target/loss; independent spectra may remain post-training reference diagnostics only.

## Practical solver route and limits

The directly testable sequence is: (1) encode a pure even reference and one explicit blocked column swap, (2) generate a full-space canonical frame with particle-hole-compatible Thouless coordinates, (3) recompute the complete normal and pairing fields from its primitive density, (4) evaluate \([\mathcal H_\mu,\mathcal R]\), \(\operatorname{Tr}\rho-N\), parity, and blocked identity, then (5) update coordinates with damped Gauss–Newton or a Newton–Krylov step. A matrix-free Jacobian-vector product differentiates the full residual and includes induced field response; a preconditioner may use a Sylvester or two-quasiparticle-gap approximation only after its branch assumptions are stated. A Kato projector derivative \(\dot P=(2\pi i)^{-1}\oint(z-h)^{-1}\dot h(z-h)^{-1}dz\) is an eigenvector-free way to transport a *gapped normal subspace*, but does not define which separated subspace is the physical nuclear sea and should not project the final pairing density to \(P_+\).

The small algebra prototype in `dpl_rhf/rhfb/full_branch_stationarity.py` implements only \(\mathcal H_\mu\), canonical tangents, full commutator, and the frame \(H^{20}\) residual. Its tests compare the exact Thouless first derivative to centered finite differences, verify the commutator/H20 norm identity, and independently preserve odd Pfaffian parity while eig/eigval APIs are guarded. This validates matrix conventions only. It does not solve a frozen-field branch, update any PKA1 or D1S field, or establish a self-consistent O17 result.
