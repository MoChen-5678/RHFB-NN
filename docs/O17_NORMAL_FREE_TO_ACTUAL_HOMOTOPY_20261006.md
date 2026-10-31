# O17 normal free-to-interacting Dirac-sector homotopy

## Result

For the exact saved O17 normal matrices and the same certified normal-sector shifts \(W_n=-245.6643363\) MeV and \(W_p=-249.2458240\) MeV used by the conditional branch reference, there is an eigen-free finite-matrix proof that the mathematical upper/lower normal spectral sectors connect to the free positive-particle / negative-sea sectors along

\[
h(t)=(1-t)K+t h_{\rm saved}^{\rm Herm},\qquad 0\le t\le1.
\]

Here \(K\) is the actual mass-subtracted finite-basis Dirac kinetic operator from `AngularRadialBasis.apply_kinetic(I, species)` with the unchanged PKA1 mass \(M=938.9\) MeV. This strengthens the interpretation of this *normal* G/F split beyond an arbitrary block label: the same \(W\) remains inside a gap all along the straight-line normal interpolation, and the upper normal subspace at \(t=0\) is exactly the free positive-energy subspace. The audit called no eigenvalue/eigenvector routine; it used two SVDs, one per species, only to verify that the free G/F coupling has no zero singular value.

The saved density SHA-256 is `703b4c089bdfe2d993b1c851feabdad7730832782943559b81b9ccb3e8d328ac`, matching the frozen field worker's exact source. The symmetrized actual \(h\) hashes match the saved conditional-reference run: neutron `6b430af662829bf91a2195a980ee9c4bf30c061ba6ed03904fb201a5b114a79b`, proton `29440639e88abb8a7a08f1520e85e9c77ee10d415e1c39559ed4db59ed3a1124`. The raw-field anti-Hermitian max corrections are only \(2.76\times10^{-13}\) and \(2.71\times10^{-13}\) MeV.

## Why the gap stays open

In the actual basis component order \((G,F)\), the free kinetic matrix is exactly

\[
K=\begin{pmatrix}0&T^\dagger\\T&-2M I\end{pmatrix}.
\]

The implementation confirms \(K_{GG}=0\), \(K_{FF}=-1877.8 I\) MeV, and \(K_{FG}=K_{GF}^\dagger\), all exactly at the stored matrix precision. For a fixed species, set

\[
A(t)=h_{GG}(t)-W I,\qquad C(t)=W I-h_{FF}(t).
\]

Both are affine combinations of their free and interacting endpoints. The actual endpoint has outward-padded Gershgorin bounds \(\lambda_{\min}(h_{GG})\ge g_{\rm lo}\) and \(\lambda_{\max}(h_{FF})\le f_{\rm hi}\), so for every \(t\):

\[
A(t)\succeq\min(-W,g_{\rm lo}-W)I,\qquad
C(t)\succeq\min(W+2M,W-f_{\rm hi})I.
\]

Both right-hand sides are strictly positive. Veselić and Veselić's quasi-definite block theorem applies to \(h(t)-WI=[[A(t),B(t)],[B(t)^\dagger,-C(t)]]\) for arbitrary off-diagonal \(B(t)\); it gives a resolvent interval around zero and, by continuously scaling \(B\) to zero without closing that gap, inertia \((216,216)\). Thus the number of normal states below/above this same \(W\) is fixed over the entire path.

| Species | Free \(A(0)\) bound | Free \(C(0)\) bound | Actual \(A(1)\) certified bound | Actual \(C(1)\) certified bound | Uniform gap around \(W\) for all \(t\) |
|---|---:|---:|---:|---:|---:|
| n | 245.6643 MeV | 1632.1357 MeV | 129.8730 MeV | 129.8730 MeV | 129.8730 MeV |
| p | 249.2458 MeV | 1628.5542 MeV | 121.2826 MeV | 121.2826 MeV | 121.2826 MeV |

There is also a direct free-operator particle/sea identification. If \(\sigma_j(T)>0\) are the singular values of the actual free coupling block, the two free eigenvalues associated with each singular pair obey

\[
\epsilon_j^\pm=-M\pm\sqrt{M^2+\sigma_j(T)^2}.
\]

The observed smallest singular value is 0.1786364 MeV (both species; largest 2664.8732 MeV), giving a derived free particle edge \(1.6994\times10^{-5}\) MeV above zero and sea edge \(-1877.8000170\) MeV. The free spectrum therefore lies strictly above zero on its particle side and below \(-2M\) on its sea side. Since both \(W\)'s lie between \(-2M\) and zero, the upper sector at \(K\) is the positive-energy free Dirac sector. The open fixed-\(W\) homotopy then identifies the actual saved \(h\) upper normal sector by continuous transport from that free branch.

This proof is specific to the saved finite-dimensional normal matrices and the declared straight-line interpolation. Gershgorin bounds use float64/complex128 arithmetic and conservative padding rather than formal interval arithmetic; their margins are over 120 MeV, far above the recorded Hermitian symmetrization corrections.

## Physical boundary

This result certifies normal-sector branch identity for this frozen \(h\), not the full no-sea paired vacuum. DIRHB writes the relativistic no-sea prescription in terms of quasiparticle annihilator choices: positive quasiparticle energies for particle-sector modes and negative quasiparticle energies for Dirac-sea modes (Eq. 46), with separate particle/sea sums in both \(\rho\) and \(\kappa\) (Eqs. 47–48, printed pp. 12–13). It also distinguishes the unpaired Dirac eigenvalue sign from the paired RHB branch labels. Therefore the normal homotopy does not prove a full BdG branch gap, a blocked odd quasiparticle identity, or that \(\chi=P_+\chi P_+^T\). It cannot erase the saved whole-versus-strict projected fixed-field discrepancy or identify the candidate block solely from this normal subspace.

Tanimura, Hagino, and Ring place \(W\) between positive and negative spectra for their inverse-Hamiltonian method (PDF pp. 1–2, Eqs. 3–5). The theorem here independently establishes a same-\(W\) gap for the **normal** finite matrices along the stated path. It does not claim a gap for the full paired Nambu matrix, nor a new self-consistent solver. [Veselić & Veselić, arXiv:1404.4991, PDF p. 1, Eqs. (1)–(2)](https://arxiv.org/pdf/1404.4991), [Tanimura, Hagino & Ring, arXiv:1304.4697, PDF pp. 1–2, Eqs. (3)–(5)](https://arxiv.org/pdf/1304.4697), [DIRHB, arXiv:1403.4039, PDF pp. 11–13, Eqs. (41)–(48)](https://arxiv.org/pdf/1403.4039).

The machine-readable proof inputs, endpoint certificates, exact source hashes, free-operator singular check, and observed zero eigensolver counts are in [normal_free_to_interacting_homotopy.json](../outputs/validation/O17_full_fields_physics_20261006/normal_free_to_interacting_homotopy.json). The reproducible audit is [normal_free_to_interacting_homotopy.py](../outputs/validation/O17_full_fields_physics_20261006/normal_free_to_interacting_homotopy.py). It leaves all saved fields, densities, W choices, and branch results unchanged.
