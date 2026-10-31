# ADF momentum operators for post-variation center-of-mass reporting

This note defines an isolated operator-construction and reporting contract. It does not add a term to the RHFB variational energy, normal field, or pairing field.

Geng and Long define the correction as

\[
E_{\rm cm}=-\frac{\langle \mathbf P_{\rm cm}^2\rangle}{2AM},
\qquad \mathbf P_{\rm cm}=\sum_i\mathbf p_i,
\]

and explicitly state that CoM corrections are not included in the variation (arXiv:2202.00482, Eq. (41), PDF p. 8). Appendix C, Eq. (C1), gives the one-body \(p^2\), normal exchange, and anomalous Wick pieces; Eq. (C2) evaluates the one-body term with the direct radial \(-d^2/dr^2+l(l+1)/r^2\) action on upper and lower Dirac components. The attached RHFB `Expect` routines call the center-of-mass routine for reporting with the variation flag false: `reference_fortran/user-rhfb-20261015/RHFB-BI/Expect.f90:93-95` and `reference_fortran/user-rhfb-20261015/S-RHFB-C/Expect.f90:76-78`.

For the ADF coordinate basis, `cm_momentum_adf.py` samples reduced Dirac amplitudes (G,F), applies the parity-correct ADF radial derivative, then uses the exact continuum (1/r) product rule and analytic spin-angular gradients. Its vector momentum uses the split weak form

\[
 (p_\mu)_{ab}=-\frac{i}{2}\left(\int d^3r\,\psi_a^\dagger\partial_\mu\psi_b-\int d^3r\,[\partial_\mu\psi_a]^\dagger\psi_b\right),
 \qquad
 (p^2)_{ab}=\sum_\mu\int d^3r\,[\partial_\mu\psi_a]^\dagger[\partial_\mu\psi_b].
\]

The momentum split follows continuum integration by parts for vanishing surface flux, but it is not the same finite-grid operator as the one-sided ADF derivative. The raw right-acting matrix and the discrete integration-by-parts defect are exported and must be checked for convergence; the split weak form must not conceal a large defect. The direct (p^2) matrix is independent and is not computed as \(\sum_\mu p_\mu p_\mu\), which would insert the finite primitive-space projector between momenta and omit derivative components outside that space. Identifying its gradient norm with the direct \(-\nabla^2\) matrix element assumes vanishing surface flux at the origin and box boundary.

The builder returns a **candidate**, not a validated O17 operator. A hash-matched certificate is only recorded as an external assertion; this helper does not machine-verify the evidence and always returns `physical_result_accepted=false`. Independent checks must cover momentum Hermiticity, weak/raw and integration-by-parts defect convergence, direct-(p^2) Hermiticity, boundary flux, radial derivative-order convergence, box-radius convergence, angular-cutoff convergence, spherical reference, and direct-(p^2) reference.

`evaluate_basis_center_of_mass_post_variation` accepts accepted-state objects with `q`, `rho_active`, and `chi_active` for neutrons and protons, lifts each species into the common primitive basis, and calls the existing Wick evaluator. Its momentum matrices are in fm\(^{-1}\), its independently constructed scalar matrix in fm\(^{-2}\), and the energy conversion is \((\hbar c)^2/(2AM)\). The caller must provide the isotope mass number \(A\) and nucleon mass \(M\) in MeV explicitly. The default \(\hbar c\) value only supplies unit conversion; it does not select an isotope mass convention. The result is marked as excluded from variation and contains no fields.

The small-basis tests include a Cartesian (s_{1/2}\to p_{1/2}) oscillator (p_z) matrix element and an independent Gaussian (langle p^2\rangle=3/(2a^2)) reference. They do not establish O17 convergence. A finite Hermiticity residual alone does not establish the continuum surface condition, truncation convergence, or equivalence to the source's spherical radial formula. Therefore this diagnostic must not be reported as an accepted CoM-corrected O17 energy.

Sources:

- Geng and Long, *Relativistic Hartree-Fock-Bogoliubov theory for finite nuclei*, [arXiv:2202.00482](https://arxiv.org/abs/2202.00482), Eq. (41), PDF p. 8; Appendix C Eqs. (C1)-(C2), PDF p. 27.
- Local algebra and post-variation Wick convention: `dpl_rhf/rhfb/center_of_mass.py`.
- Local ADF sampler and discrete derivative: `dpl_rhf/rhfb/basis.py`, `dpl_rhf/rhfb/spatial.py`.

The split weak momentum is algebraically the Hermitian part of the raw one-sided ADF matrix. Its continuum motivation is integration by parts with vanishing surface flux, but the split operator is not identical to the one-sided finite-grid derivative. The exported raw matrix and integration-by-parts defect must be examined and shown to converge; the split form alone is not evidence that the discrete boundary defect is small.
