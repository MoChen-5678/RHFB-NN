# RHFB microscopic center-of-mass correction: formula and comparison convention

## Conclusion for the current binding-energy comparison

The historical even-even experiment design fixes **no-CoM on both sides**: `EXPERIMENT_DESIGN.md` fixes the CoM convention and explicitly compares uncorrected `E/A`; the strict implementation summary likewise labels its energy as no-CoM. The functional's omission is consistent with that energy definition. Historical Core comparisons apply only to their validated even-even scope. The user's current instruction prohibits Fortran; new odd-RHFB validation must use an independent Python reference, with its CoM convention matched explicitly.

That does not make the reported energy a CoM-corrected physical binding energy. A microscopic correction is absent from the current scalar functional. If results are compared with an experimental mass, or with a published reference whose convention includes a microscopic CoM correction, the convention must be reconciled and the correction reported separately. The literature prescription most directly relevant here is evaluated **after** variation, so it does not require changing the current training objective for a no-CoM benchmark. This review does not attribute any current O-17 energy discrepancy to CoM.

## Source and scope

Geng and Long, *Relativistic Hartree-Fock-Bogoliubov model for axially deformed nuclei*, arXiv:2202.00482, published as *Phys. Rev. C* **105**, 034329 (2022), gives the microscopic correction in the main text Eq. (41) and its Bogoliubov contractions in Appendix C, Eq. (C1). The paper states in Sec. II.C that this correction is **not included in the variation**. Its Appendix C works in the Bogoliubov quasiparticle basis and retains both normal and anomalous contractions.

The local `RHF-e.pdf` formalism manuscript supplies PKA1 particle-hole vertices; it is not the source of the Bogoliubov CoM prescription used here. It therefore supplies no basis for adding a CoM term to the no-CoM training target. The project experiment design and energy-output documentation provide evidence of the existing convention.

Primary references: [arXiv:2202.00482](https://arxiv.org/abs/2202.00482), [published DOI 10.1103/PhysRevC.105.034329](https://doi.org/10.1103/PhysRevC.105.034329).

## Bogoliubov expectation of total momentum squared

The paper's Eq. (41) uses the nonrelativistic collective kinetic-energy estimate

\[
 E_{\rm cm}=-\frac{1}{2 A M}\,\langle\hat{\mathbf P}^{2}\rangle,
 \qquad \hat{\mathbf P}=\sum_{a=1}^{A}\hat{\mathbf p}_a,
\]

where `A` is the mass number and `M` is the nucleon mass. Appendix C, Eq. (C1), writes the expectation in canonical Bogoliubov notation as a one-body term, a normal-density exchange contraction, and an anomalous contraction:

\[
\langle\hat{\mathbf P}^{2}\rangle
=\sum_i \mathbf p^{2}_{ii}
-\sum_{ii'}\mathbf p^{VV}_{ii'}\!\cdot\mathbf p^{VV}_{i'i}
+\sum_{ii'}\mathbf p^{VV}_{ii'}\!\cdot\mathbf p^{UU}_{\bar i\bar i'}.
\tag{C1}
\]

Here the `V` and `U` matrix elements are Bogoliubov amplitudes, and bars denote time-reversed partners as defined in Appendix C. In a coordinate basis-independent form that matches this project's normal and anomalous density convention, let

\[
 \rho_{ab}=\langle c_b^\dagger c_a\rangle,\qquad
 \chi_{ab}=\langle c_b c_a\rangle=-\chi_{ba},\qquad
 (p_k)_{ab}=\langle a|\hat p_k|b\rangle,
\]

with Hermitian `p_k`. Wick contraction gives

\[
\begin{aligned}
 \langle\hat P_k\rangle &= \operatorname{Tr}(\rho p_k),\\
 \langle\hat{\mathbf P}^{2}\rangle
 &=\sum_{k=x,y,z}\left[
   \left|\operatorname{Tr}(\rho p_k)\right|^2
   +\operatorname{Tr}(\rho p_k^2)
   -\operatorname{Tr}(\rho p_k\rho p_k)
   +\operatorname{Tr}(\chi^\dagger p_k\chi p_k^T)
 \right].
\end{aligned}
\tag{1}
\]

Equation (1) makes the disconnected \(\langle\hat{\mathbf P}\rangle^2\) term explicit. Eq. (C1)'s displayed canonical contractions do not show that term separately; their usual even-even, time-reversal-invariant applications have zero total mean momentum. For a blocked/time-odd state it must either be retained as in (1), or its vanishing must be established for that particular stationary state. Local time-odd currents alone do not prove that the integrated total momentum is nonzero, but they also do not justify silently dropping the mean term.

The last trace in (1) is the anomalous Wick contraction. Its algebraic sign is `+` for the stated `chi=<cc>` convention, but its value need not be positive. Setting `chi=0` removes it and recovers the Slater/Hartree-Fock expression; that is not valid for a paired RHFB state. The normal one-body and exchange contractions must also both be retained. For separate neutron and proton pairing without proton-neutron pairing, use block-diagonal species `rho` and `chi` and the same spatial `p_k`; the disconnected mean is the square of the **total** neutron-plus-proton mean momentum.

In the project's factorization, the primitive-space matrices represented by the active factors are

\[
 \rho_s=Q_s\rho_{{\rm active},s}Q_s^\dagger,
 \qquad
 \chi_s=Q_s\chi_{{\rm active},s}Q_s^T.
\]

The transpose in the anomalous density is essential. The contractions in (1) are evaluated with the same one-body momentum operator used by the chosen basis and truncation. In natural units the CoM prefactor is `1/(2 A M)`; if momentum matrix elements are represented in `fm^-1` and energies in MeV, restore the matching powers of `ħc` consistently (the prefactor multiplying `〈P²〉` is `(ħc)^2/(2 A M)` in MeV·fm² units).

## What this prescription does and does not establish

The cited RHFB prescription is a nonrelativistic collective center-of-mass kinetic-energy estimate applied to relativistic single-particle states. The 2022 paper uses it after variation and omits meson degrees of freedom from `P_cm`; it is not an exact translational projection and it is not a Lorentz-covariant correction. Applying it to a blocked odd system is a defensible extension of the operator expectation, provided the normal, exchange, anomalous, and possible nonzero-mean terms are all evaluated for that state. A spherical even-even formula with `chi=0` is not that extension.

**Recommendation:** keep training and independent Python comparisons on the documented no-CoM convention. If a corrected result is needed, compute Eq. (1) from the converged state as a separately labeled post-variation result and compare it only with a reference using the same correction prescription. Putting `E_cm` into the loss or the self-consistent equations changes the variational objective; do so only if the intended benchmark itself is CoM-corrected and the reference calculation/parameterization is treated consistently. No new run or numerical validation was performed for this literature review.
